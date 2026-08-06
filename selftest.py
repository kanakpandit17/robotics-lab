#!/usr/bin/env python3
"""Prove the viewer works before the robot is on the bench.

Stands up a local MJPEG server and a local UDP video push, pulls real frames
through the same backends the R1 will use, and renders the HUD to a PNG.

    python selftest.py

Everything here is loopback-only; no robot required.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from r1vision import viewer
from r1vision.discover import probe_http
from r1vision.sources import FfmpegSource, MjpegSource, SourceError, build_source

PASS, FAIL = "  ok  ", " FAIL "
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))


def make_frame(i: int, w: int = 640, h: int = 480) -> np.ndarray:
    """A moving target, so we can tell live frames from a frozen one."""
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = (40, 30, 30)
    x = int((w - 80) * (0.5 + 0.5 * np.sin(i * 0.15)))
    cv2.rectangle(frame, (x, h // 2 - 40), (x + 80, h // 2 + 40), (60, 200, 90), -1)
    cv2.putText(frame, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (240, 240, 240), 2, cv2.LINE_AA)
    return frame


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------
# 1. MJPEG over HTTP -- the transport a Jetson-side streamer most often uses
# --------------------------------------------------------------------------
class MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the test output clean
        pass

    def do_GET(self):
        if self.path not in ("/stream", "/"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            for i in range(200):
                ok, buf = cv2.imencode(".jpg", make_frame(i))
                if not ok:
                    continue
                data = buf.tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass


def test_mjpeg() -> None:
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), MjpegHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)
    url = f"http://127.0.0.1:{port}/stream"

    try:
        hit = probe_http("127.0.0.1", port, "/stream")
        check("discover detects MJPEG endpoint", hit is not None and hit.kind == "mjpeg",
              hit.detail if hit else "no hit")

        src = build_source(url)
        check("build_source picks MjpegSource", isinstance(src, MjpegSource),
              type(src).__name__)

        src.open()
        frames = [src.read() for _ in range(10)]
        src.close()
        good = [f for f in frames if f is not None and f.size]
        check("MJPEG: read 10 live frames", len(good) == 10, f"{len(good)}/10")
        # Hash the whole frame: the top rows are static background, so a
        # prefix comparison would call a frozen stream "live".
        moving = len({hash(f.tobytes()) for f in good}) > 1
        check("MJPEG: frames are actually changing", moving)
        check("MJPEG: correct shape", good[0].shape == (480, 640, 3), str(good[0].shape))
    finally:
        server.shutdown()


# --------------------------------------------------------------------------
# 2. UDP push -- what you'll use if you stream off the robot with ffmpeg
# --------------------------------------------------------------------------
def test_udp() -> None:
    port = free_port()
    # Reader first so the port is bound before any packets are sent; UDP has no
    # retransmit, so anything pushed before the bind is simply gone.
    src = FfmpegSource(f"udp://127.0.0.1:{port}?fifo_size=1000000&overrun_nonfatal=1",
                       width=640, height=480)
    push = None
    try:
        src.open()
        time.sleep(0.5)
        push = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
             "-f", "lavfi", "-i", "testsrc=size=640x480:rate=15",
             "-c:v", "mpeg2video", "-q:v", "5", "-f", "mpegts",
             f"udp://127.0.0.1:{port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        frames = [src.read() for _ in range(8)]
        good = [f for f in frames if f is not None and f.size]
        check("UDP/ffmpeg: read live frames", len(good) >= 5, f"{len(good)}/8")
        if good:
            check("UDP/ffmpeg: correct shape", good[0].shape == (480, 640, 3),
                  str(good[0].shape))
    except SourceError as exc:
        check("UDP/ffmpeg: read live frames", False, str(exc))
    finally:
        src.close()
        if push is not None:
            push.terminate()
            push.wait(timeout=5)


# --------------------------------------------------------------------------
# 3. Overlay rendering -- the HUD must not crash on odd frame sizes
# --------------------------------------------------------------------------
def test_overlays() -> None:
    out = Path("captures")
    out.mkdir(exist_ok=True)
    for w, h in [(640, 480), (1920, 1080), (320, 240)]:
        frame = make_frame(7, w, h)
        viewer.draw_grid(frame)
        viewer.draw_reticle(frame)
        viewer.draw_hud(frame, source="selftest", fps=29.97, frames=123, dropped=2,
                        elapsed=75, recording=True, paused=False)
        if (w, h) == (640, 480):
            cv2.imwrite(str(out / "selftest-hud.png"), frame)
    check("HUD renders at 320p / 480p / 1080p", True, "wrote captures/selftest-hud.png")

    waiting = viewer.draw_waiting(960, 540, "waiting for frames")
    check("waiting screen renders", waiting.shape == (540, 960, 3))

    meter = viewer.FpsMeter()
    for _ in range(5):
        meter.tick()
        time.sleep(0.02)
    check("FPS meter produces a sane rate", 20 < meter.fps < 120, f"{meter.fps:.1f} fps")


# --------------------------------------------------------------------------
# 4. Source routing -- a URL must land on the right backend
# --------------------------------------------------------------------------
def test_routing() -> None:
    cases = {
        "rtsp://192.168.123.161:8554/live": "OpenCVUrlSource",
        "http://192.168.123.161:8080/stream": "MjpegSource",
        "udp://@:5000": "FfmpegSource",
        "webcam": "WebcamSource",
        "webcam:1": "WebcamSource",
    }
    ok = True
    for spec, expected in cases.items():
        got = type(build_source(spec)).__name__
        if got != expected:
            ok = False
            print(f"        {spec} -> {got}, expected {expected}")
    check("URL routing picks the right backend", ok, f"{len(cases)} cases")

    try:
        build_source("gopher://nope")
        check("unknown scheme is rejected", False, "no error raised")
    except SourceError:
        check("unknown scheme is rejected", True)

    try:
        build_source("dds")
        check("dds without interface is rejected", False, "no error raised")
    except SourceError:
        check("dds without interface is rejected", True)


# --------------------------------------------------------------------------
# 5. Recording -- mp4v can fail silently on macOS and leave a 0-byte file
# --------------------------------------------------------------------------
def test_recording() -> None:
    out = Path("captures") / "selftest-clip.mp4"
    out.unlink(missing_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480))
    check("VideoWriter opens", writer.isOpened())
    for i in range(40):
        writer.write(make_frame(i))
    writer.release()

    size = out.stat().st_size if out.exists() else 0
    check("recording produces a non-empty mp4", size > 1000, f"{size} bytes")

    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    check("recorded mp4 is readable back", frames >= 35, f"{frames} frames")
    out.unlink(missing_ok=True)


def main() -> int:
    print("r1vision selftest -- no robot required\n")
    print("-- overlays " + "-" * 44)
    test_overlays()
    print("-- routing " + "-" * 45)
    test_routing()
    print("-- MJPEG over HTTP " + "-" * 37)
    test_mjpeg()
    print("-- UDP via ffmpeg " + "-" * 38)
    test_udp()
    print("-- recording " + "-" * 43)
    test_recording()

    failed = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 56)
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        return 1
    print("Viewer pipeline is working. Join the R1's Wi-Fi and run:")
    print("  ./watch.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
