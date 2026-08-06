"""Live view of what the R1 is looking at.

    python -m r1vision.view --source rtsp://192.168.123.161:8554/live

Keys:
    q / ESC   quit              s   save a snapshot
    space     pause             r   start/stop recording
    g         grid overlay      h   toggle the HUD
    [ / ]     shrink / grow the window
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import net
from .sources import FrameSource, SourceError, build_source

CAPTURE_DIR = Path("captures")

# BGR. Deliberately high-contrast so the overlay stays readable against
# whatever the robot happens to be pointed at.
GREEN = (120, 255, 120)
AMBER = (60, 200, 255)
RED = (80, 80, 255)
WHITE = (245, 245, 245)
SHADOW = (0, 0, 0)


class FpsMeter:
    """Rolling FPS over the last `window` frames."""

    def __init__(self, window: int = 60):
        self.stamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self.stamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else 0.0


def draw_text(img, text, org, color=WHITE, scale=0.5, thick=1):
    """Text with a dark outline so it survives a bright or busy background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_reticle(img) -> None:
    """Centre crosshair -- the optical axis, i.e. where the head is pointed."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    arm, gap = max(14, w // 40), max(5, w // 160)
    cv2.line(img, (cx - arm, cy), (cx - gap, cy), GREEN, 1, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + arm, cy), GREEN, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - arm), (cx, cy - gap), GREEN, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + arm), GREEN, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(20, w // 26), GREEN, 1, cv2.LINE_AA)


def draw_grid(img) -> None:
    """Rule-of-thirds guides, handy for judging how the head is aimed."""
    h, w = img.shape[:2]
    overlay = img.copy()
    for i in (1, 2):
        cv2.line(overlay, (w * i // 3, 0), (w * i // 3, h), WHITE, 1, cv2.LINE_AA)
        cv2.line(overlay, (0, h * i // 3), (w, h * i // 3), WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def draw_hud(img, *, source, fps, frames, dropped, elapsed, recording, paused) -> None:
    h, w = img.shape[:2]
    bar = img[0:30, 0:w]
    cv2.addWeighted(bar, 0.35, np.zeros_like(bar), 0.65, 0, bar)

    draw_text(img, f"R1 VISION  {source}", (10, 20), GREEN, 0.5)
    right = f"{w}x{h}  {fps:5.1f} fps  {frames} frames"
    if dropped:
        right += f"  {dropped} dropped"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    draw_text(img, right, (w - tw - 10, 20), WHITE, 0.5)

    mins, secs = divmod(int(elapsed), 60)
    draw_text(img, f"{mins:02d}:{secs:02d}", (10, h - 12), WHITE, 0.45)
    draw_text(img, datetime.now().strftime("%H:%M:%S"), (w - 80, h - 12), WHITE, 0.45)

    if recording:
        cv2.circle(img, (w - 22, 46), 7, RED, -1, cv2.LINE_AA)
        draw_text(img, "REC", (w - 68, 51), RED, 0.5)
    if paused:
        draw_text(img, "PAUSED", (w // 2 - 34, 52), AMBER, 0.6)


def draw_waiting(width, height, message) -> np.ndarray:
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:] = (28, 24, 22)
    draw_text(canvas, message, (24, height // 2), AMBER, 0.6)
    draw_text(canvas, "q to quit", (24, height // 2 + 28), WHITE, 0.45)
    return canvas


def open_with_retry(spec, interface, size, attempts, delay) -> FrameSource:
    last: Exception | None = None
    for n in range(1, attempts + 1):
        src = build_source(spec, interface, size)
        try:
            src.open()
            return src
        except Exception as exc:  # DDS and ffmpeg backends raise assorted types
            last = exc
            src.close()
            if n < attempts:
                print(f"  attempt {n}/{attempts} failed: {exc}; retrying in {delay:.0f}s")
                time.sleep(delay)
    raise SourceError(f"could not open {spec}: {last}")


def run(args) -> int:
    print(f"opening {args.source} ...")
    try:
        source = open_with_retry(
            args.source, args.interface, args.size, args.retries, args.retry_delay
        )
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("\nRun `python -m r1vision.discover` to find a working stream.", file=sys.stderr)
        return 1
    print(f"connected: {source.name}")

    window = "R1 Vision"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.width:
        cv2.resizeWindow(window, args.width, int(args.width * 9 / 16))

    meter = FpsMeter()
    started = time.monotonic()
    frames = dropped = 0
    misses = 0
    show_hud, show_grid, show_reticle = True, args.grid, not args.no_reticle
    paused = False
    writer: cv2.VideoWriter | None = None
    last_good: np.ndarray | None = None
    scale = args.scale

    CAPTURE_DIR.mkdir(exist_ok=True)

    try:
        while True:
            if not paused:
                try:
                    frame = source.read()
                except Exception as exc:
                    print(f"read error: {exc}")
                    frame = None

                if frame is None:
                    misses += 1
                    if misses >= args.max_misses:
                        print("stream dropped; reconnecting ...")
                        source.close()
                        try:
                            source = open_with_retry(
                                args.source, args.interface, args.size,
                                args.retries, args.retry_delay,
                            )
                            print("reconnected")
                            misses = 0
                        except SourceError as exc:
                            print(f"error: {exc}", file=sys.stderr)
                            return 1
                elif frame.size == 0:
                    # Backend is alive but had nothing ready this tick.
                    dropped += 1
                else:
                    misses = 0
                    frames += 1
                    meter.tick()
                    last_good = frame

            if last_good is None:
                canvas = draw_waiting(960, 540, f"waiting for frames from {args.source}")
            else:
                canvas = last_good.copy()
                if args.mirror:
                    canvas = cv2.flip(canvas, 1)
                if args.rotate:
                    rotations = {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }
                    canvas = cv2.rotate(canvas, rotations[args.rotate])
                if scale != 1.0:
                    canvas = cv2.resize(
                        canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )

                if writer is not None:
                    writer.write(canvas)
                if show_grid:
                    draw_grid(canvas)
                if show_reticle:
                    draw_reticle(canvas)
                if show_hud:
                    draw_hud(
                        canvas,
                        source=source.name,
                        fps=meter.fps,
                        frames=frames,
                        dropped=dropped,
                        elapsed=time.monotonic() - started,
                        recording=writer is not None,
                        paused=paused,
                    )

            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord(" "):
                paused = not paused
            elif key == ord("["):
                scale = max(0.25, scale - 0.1)
            elif key == ord("]"):
                scale = min(3.0, scale + 0.1)
            elif key == ord("s") and last_good is not None:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = CAPTURE_DIR / f"r1-{stamp}.png"
                cv2.imwrite(str(path), last_good)
                print(f"saved {path}")
            elif key == ord("r") and last_good is not None:
                if writer is None:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = CAPTURE_DIR / f"r1-{stamp}.mp4"
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(
                        str(path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        meter.fps if meter.fps > 1 else 20.0,
                        (w, h),
                    )
                    print(f"recording -> {path}")
                else:
                    writer.release()
                    writer = None
                    print("recording stopped")

            # Window closed with the red button rather than a keypress.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if writer is not None:
            writer.release()
        source.close()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    print(f"{frames} frames in {elapsed:.1f}s ({frames / elapsed if elapsed else 0:.1f} fps avg)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Live view of the Unitree R1's camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "-s", "--source", default="webcam",
        help="'webcam', 'webcam:1', 'dds', or a URL "
             "(rtsp://..., http://.../stream, udp://@:5000). Default: webcam",
    )
    ap.add_argument("-i", "--interface", default="",
                    help="ethernet interface for --source dds (auto-detected if omitted)")
    ap.add_argument("--size", default="", help="force frame size for raw udp/rtp, e.g. 1280x720")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale factor")
    ap.add_argument("--width", type=int, default=0, help="initial window width in pixels")
    ap.add_argument("--mirror", action="store_true", help="flip horizontally")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    ap.add_argument("--grid", action="store_true", help="start with the thirds grid on")
    ap.add_argument("--no-reticle", action="store_true", help="hide the centre crosshair")
    ap.add_argument("--retries", type=int, default=5, help="connection attempts before giving up")
    ap.add_argument("--retry-delay", type=float, default=2.0, help="seconds between attempts")
    ap.add_argument("--max-misses", type=int, default=60,
                    help="empty reads tolerated before reconnecting")
    args = ap.parse_args(argv)

    if args.source == "dds" and not args.interface:
        itf = net.pick_interface()
        if itf is None:
            print("error: no ethernet interface found for DDS", file=sys.stderr)
            return 1
        args.interface = itf.name
        print(f"using interface {itf.name}")

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
