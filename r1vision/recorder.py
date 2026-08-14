"""Record a teleop episode: video + per-hand action + per-arm state.

Started and stopped explicitly (from the headset's thumbstick-click, or the
CLI) -- never always-on, the same "opt in per session" model `viewer.py`
already uses for its own `r`-to-record video snapshots. This generalises that
to a full state/action/video episode, laid out simply enough to turn into a
LeRobot-style dataset later without committing to that format now:

    episodes/<timestamp>/
      video.mp4        the head camera, one frame per sample
      episode.jsonl     one line per sample: t, per-hand controller pose
                        (the action) and per-arm commanded/measured angles
                        (the state)
      episode.json      manifest: duration, sample count, link lengths used

Deliberately decoupled from `ArmTeleopDriver`: this only ever calls its
public `status()`, and a recorder failure (disk full, bad frame) must never
be able to affect robot control. `frame_provider` and `status_provider` are
plain callables so this module has no import-time dependency on
`quest_stream.py` or `arm_teleop.py` either.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self, frame_provider, status_provider,
                rate_hz: float = 30.0, out_root: str = "episodes"):
        """`frame_provider() -> (jpeg_bytes | None, seq: int)`,
        `status_provider() -> dict` (e.g. `ArmTeleopDriver.status()`)."""
        self.frame_provider = frame_provider
        self.status_provider = status_provider
        self.rate = rate_hz
        self.out_root = Path(out_root)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._writer = None
        self._jsonl = None

        self.recording = False
        self.out_dir: Path | None = None
        self.samples = 0
        self.error: str | None = None
        self._t0 = 0.0

    def start(self) -> tuple[bool, str]:
        if self.recording:
            return False, "already recording"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_dir = self.out_root / ts
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._jsonl = open(self.out_dir / "episode.jsonl", "w")
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False, self.error

        self._writer = None   # created on the first frame, once size is known
        self.samples = 0
        self.error = None
        self._t0 = time.monotonic()
        self.recording = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, f"recording -> {self.out_dir}"

    def stop(self) -> tuple[bool, str]:
        if not self.recording:
            return False, "not recording"
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.recording = False

        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None

        from . import arm_kinematics   # lazy: keeps this importable without
                                       # the rest of teleop
        manifest = {
            "recorded": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.monotonic() - self._t0, 1),
            "samples": self.samples,
            "sample_rate_hz": self.rate,
            "link_lengths_m": {
                "upper_arm": arm_kinematics.UPPER_ARM_M,
                "forearm": arm_kinematics.FOREARM_M,
            },
        }
        (self.out_dir / "episode.json").write_text(json.dumps(manifest, indent=2))
        out_dir = self.out_dir
        return True, f"saved {self.samples} samples -> {out_dir}"

    def status(self) -> dict:
        return {
            "recording": self.recording,
            "out_dir": str(self.out_dir) if self.out_dir else None,
            "samples": self.samples,
            "error": self.error,
        }

    def close(self) -> None:
        if self.recording:
            self.stop()

    # ------------------------------------------------------------- send loop
    def _loop(self) -> None:
        import cv2   # lazy: only needed while actually recording

        period = 1.0 / self.rate
        last_seq = -1
        while not self._stop.is_set():
            cycle = time.monotonic()
            try:
                jpeg, seq = self.frame_provider()
                if jpeg is not None and seq != last_seq:
                    last_seq = seq
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        if self._writer is None:
                            h, w = frame.shape[:2]
                            path = str(self.out_dir / "video.mp4")
                            self._writer = cv2.VideoWriter(
                                path, cv2.VideoWriter_fourcc(*"mp4v"),
                                self.rate, (w, h))
                        self._writer.write(frame)

                record = {
                    "t": round(time.monotonic() - self._t0, 3),
                    "status": self.status_provider(),
                }
                self._jsonl.write(json.dumps(record) + "\n")
                self.samples += 1
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

            remaining = period - (time.monotonic() - cycle)
            if remaining > 0:
                self._stop.wait(remaining)
