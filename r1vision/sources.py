"""Pluggable frame sources.

Every backend exposes the same three calls -- open(), read(), close() -- so the
viewer does not care whether frames arrive over DDS, RTSP, HTTP or a local
webcam. read() returns a BGR numpy array or None if no frame was available.

Why several backends: Unitree does not document one blessed video path for the
humanoids. Go2 exposes a DDS `videohub` RPC, the EDU boards usually have a
GStreamer/RTSP pipeline on the onboard computer, and people commonly bolt an
MJPEG server onto the Jetson. Discovery figures out which one this R1 answers
on, and the viewer just plays it.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import cv2
import numpy as np


class SourceError(RuntimeError):
    pass


class FrameSource(ABC):
    """Base class for anything that can hand us BGR frames."""

    name = "source"

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray | None: ...

    def close(self) -> None:
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class WebcamSource(FrameSource):
    """Local camera. Lets you exercise the whole viewer before the robot ships."""

    def __init__(self, index: int = 0):
        self.index = index
        self.name = f"webcam:{index}"
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise SourceError(f"could not open webcam index {self.index}")
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class OpenCVUrlSource(FrameSource):
    """RTSP/HTTP/RTMP via OpenCV's FFmpeg backend.

    Forces RTSP over TCP -- UDP loses packets on a busy robot link and produces
    smeared frames that look like a camera fault but are just dropped slices.
    """

    def __init__(self, url: str, latency_ms: int = 100):
        self.url = url
        self.latency_ms = latency_ms
        self.name = f"opencv:{url}"
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        import os

        if self.url.startswith("rtsp://"):
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;tcp|max_delay;{self.latency_ms * 1000}",
            )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise SourceError(f"could not open stream: {self.url}")
        # Keep the internal queue short so we display "now", not backlog.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class MjpegSource(FrameSource):
    """multipart/x-mixed-replace MJPEG over HTTP, parsed by hand.

    OpenCV can sometimes read these, but it buffers aggressively and stalls on
    servers with sloppy boundaries. Scanning for JPEG SOI/EOI markers ourselves
    is both lower latency and more tolerant.
    """

    SOI = b"\xff\xd8"  # start of image
    EOI = b"\xff\xd9"  # end of image

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout
        self.name = f"mjpeg:{url}"
        self._stream = None
        self._buf = bytearray()

    def open(self) -> None:
        from urllib.request import Request, urlopen

        req = Request(self.url, headers={"User-Agent": "r1vision/0.1"})
        try:
            self._stream = urlopen(req, timeout=self.timeout)
        except OSError as exc:
            raise SourceError(f"could not open MJPEG stream {self.url}: {exc}") from exc

    def read(self) -> np.ndarray | None:
        if self._stream is None:
            return None
        # Pull until a complete JPEG is in the buffer.
        for _ in range(512):
            start = self._buf.find(self.SOI)
            end = self._buf.find(self.EOI, start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpeg = bytes(self._buf[start : end + 2])
                del self._buf[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                continue  # corrupt frame, keep scanning
            chunk = self._stream.read(8192)
            if not chunk:
                return None
            self._buf.extend(chunk)
            # Guard against a non-MJPEG endpoint feeding us junk forever.
            if len(self._buf) > 8 * 1024 * 1024:
                raise SourceError(f"no JPEG markers in first 8MB of {self.url}")
        return None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class FfmpegSource(FrameSource):
    """Decode with the ffmpeg binary and pipe raw BGR frames in.

    This is the fallback that handles what pip's OpenCV cannot: bare `udp://`
    and `rtp://` pushes and .sdp files. The pip wheels are built without
    GStreamer, so a gst pipeline string would silently fail -- ffmpeg does not.
    """

    def __init__(self, url: str, width: int = 0, height: int = 0, extra: list[str] | None = None):
        self.url = url
        self.width = width
        self.height = height
        self.extra = extra or []
        self.name = f"ffmpeg:{url}"
        self._proc: subprocess.Popen | None = None
        self._frame_bytes = 0

    def _probe(self) -> tuple[int, int]:
        """Ask ffprobe for the frame size; needed to slice the raw byte stream."""
        if self.width and self.height:
            return self.width, self.height
        if not shutil.which("ffprobe"):
            raise SourceError("ffprobe not found; pass --size WxH explicitly")
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            "-timeout", "5000000",
            self.url,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
            w, h = out.strip().splitlines()[0].split("x")[:2]
            return int(w), int(h)
        except (subprocess.SubprocessError, ValueError, IndexError) as exc:
            raise SourceError(
                f"could not determine frame size for {self.url}; pass --size WxH"
            ) from exc

    def open(self) -> None:
        if not shutil.which("ffmpeg"):
            raise SourceError("ffmpeg not found on PATH (brew install ffmpeg)")
        self.width, self.height = self._probe()
        self._frame_bytes = self.width * self.height * 3
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "nobuffer", "-flags", "low_delay",
            *self.extra,
            "-i", self.url,
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )

    def read(self) -> np.ndarray | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        # A pipe read returns whatever is available -- typically ~64KB, far
        # short of a full frame -- so keep pulling until the frame is complete.
        buf = bytearray()
        while len(buf) < self._frame_bytes:
            chunk = self._proc.stdout.read(self._frame_bytes - len(buf))
            if not chunk:
                return None  # ffmpeg exited or the stream ended
            buf.extend(chunk)
        return np.frombuffer(bytes(buf), np.uint8).reshape((self.height, self.width, 3))

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


class UnitreeDdsSource(FrameSource):
    """Unitree SDK2 `videohub` RPC -- polls JPEG stills as fast as it can.

    Confirmed on Go2. On the humanoids the service may not be registered, in
    which case open() raises with a clear message and discovery moves on to the
    network transports.
    """

    def __init__(self, interface: str, timeout: float = 3.0):
        self.interface = interface
        self.timeout = timeout
        self.name = f"dds:{interface}"
        self._client = None
        self._consecutive_errors = 0

    def open(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.video.video_client import VideoClient
        except ImportError as exc:
            raise SourceError(
                "unitree_sdk2py not installed -- see README 'DDS backend'"
            ) from exc

        ChannelFactoryInitialize(0, self.interface)
        client = VideoClient()
        client.SetTimeout(self.timeout)
        client.Init()
        code, _ = client.GetImageSample()
        if code != 0:
            raise SourceError(
                f"videohub service did not answer (code {code}); "
                "the R1 may not expose this RPC -- try an RTSP/MJPEG source"
            )
        self._client = client

    def read(self) -> np.ndarray | None:
        if self._client is None:
            return None
        code, data = self._client.GetImageSample()
        if code != 0 or not data:
            # Error 3104 shows up intermittently under sustained polling; it is
            # transient, so ride out a few before giving up on the stream.
            self._consecutive_errors += 1
            if self._consecutive_errors > 30:
                return None
            time.sleep(0.01)
            return np.empty((0, 0, 3), np.uint8)  # "no frame yet", not "dead"
        self._consecutive_errors = 0
        return cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)


def build_source(spec: str, interface: str = "", size: str = "") -> FrameSource:
    """Turn a CLI --source value into a live backend.

    Accepts: 'webcam', 'webcam:1', 'dds', a full URL, or 'host:port' shorthand.
    """
    width = height = 0
    if size:
        try:
            width, height = (int(v) for v in size.lower().split("x"))
        except ValueError as exc:
            raise SourceError(f"--size must look like 1280x720, got {size!r}") from exc

    if spec == "dds":
        if not interface:
            raise SourceError("--source dds needs --interface (e.g. en5)")
        return UnitreeDdsSource(interface)

    if spec.startswith("webcam"):
        _, _, idx = spec.partition(":")
        return WebcamSource(int(idx) if idx else 0)

    scheme = urlparse(spec).scheme
    if scheme in ("udp", "rtp", "srt") or spec.endswith(".sdp"):
        return FfmpegSource(spec, width, height)
    if scheme in ("rtsp", "rtmp"):
        return OpenCVUrlSource(spec)
    if scheme in ("http", "https"):
        # A still-image endpoint is not a stream; MJPEG parsing handles both
        # since a multipart body and a bare JPEG both start with SOI.
        return MjpegSource(spec)
    if scheme == "file" or spec.startswith("/"):
        return OpenCVUrlSource(spec)

    raise SourceError(f"do not know how to open source {spec!r}")
