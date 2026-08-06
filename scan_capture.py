#!/usr/bin/env python3
"""Capture a frame sequence from the R1 for SLAM3R reconstruction.

    python scan_capture.py --out scans/cell01

Drive or carry the robot around the cell while this runs. Press q to stop.

Why not just record an mp4
--------------------------
SLAM3R takes an image directory, and reconstruction quality depends far more on
which frames you feed it than on how many. This drops blurry frames (motion
blur wrecks feature matching), paces capture at a steady rate rather than the
stream's jittery one, and reports overlap so you can tell during the scan
whether you are moving too fast.

Frame rate
----------
~6 fps is the sweet spot for walking-pace scanning. Higher just gives near
duplicate frames that slow reconstruction without adding information; much
lower and consecutive frames stop overlapping enough to register.

Output layout (what SLAM3R expects)
-----------------------------------
    scans/cell01/
        000000.jpg  000001.jpg  ...
        manifest.json      timing, settings, per-frame sharpness
        preview.mp4        for a quick look before transferring
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from r1vision.sources import build_source  # noqa: E402

DEFAULT_SOURCE = "rtsp://192.168.123.161:8551/video"


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian -- low means motion blur or out of focus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def overlap(prev: np.ndarray | None, cur: np.ndarray) -> float:
    """Rough fraction of the frame that is unchanged.

    Not real feature matching, just a cheap proxy: if this drops below ~0.5 the
    robot is moving too fast for consecutive frames to register.
    """
    if prev is None:
        return 1.0
    diff = cv2.absdiff(prev, cur)
    return float((diff < 25).mean())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=None,
                    help="output directory (default scans/<timestamp>)")
    ap.add_argument("-s", "--source", default=DEFAULT_SOURCE)
    ap.add_argument("--fps", type=float, default=6.0, help="frames saved per second")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="raw stream is already upright")
    ap.add_argument("--max-frames", type=int, default=3000)
    ap.add_argument("--min-sharpness", default="auto",
                    help="blur threshold: 'auto' (default), a number, or 0 to disable")
    ap.add_argument("--width", type=int, default=0,
                    help="resize longest side to this (SLAM3R uses 224/512)")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    out = Path(args.out or f"scans/{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"opening {args.source} ...")
    src = build_source(args.source)
    try:
        src.open()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"connected: {src.name}")
    print(f"saving to {out.resolve()}")
    print("\n  Move the robot slowly and steadily. Press q to finish.\n")

    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    # A fixed threshold cannot suit every scene: real frames from this camera
    # score ~310 while a blurred one scores ~2, but that spread shifts with
    # lighting and content. Calibrating from the stream itself avoids both
    # rejecting everything and rejecting nothing.
    import statistics
    CALIB_N = 30
    calib: list[float] = []
    if args.min_sharpness == "auto":
        threshold = None            # set after CALIB_N frames
    else:
        threshold = float(args.min_sharpness)
        if threshold == 0:
            threshold = -1.0        # accept everything

    interval = 1.0 / args.fps
    next_at = time.monotonic()
    saved, seen, rejected_blur = 0, 0, 0
    recent_rejects = 0
    prev_small = None
    records = []
    started = time.monotonic()

    try:
        while saved < args.max_frames:
            frame = src.read()
            if frame is None:
                print("stream ended")
                break
            if frame.size == 0:
                continue
            seen += 1

            if args.rotate:
                frame = cv2.rotate(frame, rotations[args.rotate])
            if args.width:
                h, w = frame.shape[:2]
                scale = args.width / max(h, w)
                if scale < 1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)

            now = time.monotonic()
            due = now >= next_at

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90) if gray.shape[1] > gray.shape[0]
                               else (90, 160))
            sharp = sharpness(gray)
            ov = overlap(prev_small, small)

            if threshold is None:
                calib.append(sharp)
                if len(calib) >= CALIB_N:
                    med = statistics.median(calib)
                    threshold = 0.5 * med
                    print(f"  blur threshold auto-set to {threshold:.0f} "
                          f"(scene median {med:.0f})")
                # Nothing is saved while calibrating: with no threshold yet
                # there is no way to tell a blurred frame from a sharp one,
                # and blurry frames poison the reconstruction. This costs
                # about a second at the start of the scan.
                continue

            if due:
                if threshold is not None and threshold > 0 and sharp < threshold:
                    rejected_blur += 1
                    recent_rejects += 1
                    # Rejecting everything means the threshold is wrong for this
                    # scene, not that every frame is bad. Back off rather than
                    # silently recording nothing.
                    if recent_rejects >= 40:
                        threshold *= 0.5
                        recent_rejects = 0
                        print(f"  too many rejects; relaxing threshold to {threshold:.0f}")
                    # Do not advance next_at: retry on the next frame rather
                    # than skipping this slot entirely.
                else:
                    recent_rejects = 0
                    path = out / f"{saved:06d}.jpg"
                    cv2.imwrite(str(path), frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    records.append({
                        "file": path.name,
                        "t": round(now - started, 3),
                        "sharpness": round(sharp, 1),
                        "overlap_prev": round(ov, 3),
                    })
                    saved += 1
                    prev_small = small
                    next_at = now + interval

            if not args.no_preview:
                view = frame.copy()
                h, w = view.shape[:2]
                bar = view[0:64, 0:w]
                cv2.addWeighted(bar, 0.35, np.zeros_like(bar), 0.65, 0, bar)
                cv2.putText(view, f"saved {saved}   blur-rejected {rejected_blur}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (120, 255, 120), 1, cv2.LINE_AA)
                ov_col = (120, 255, 120) if ov > 0.5 else (60, 200, 255)
                cv2.putText(view,
                            f"sharpness {sharp:5.0f}   overlap {ov:.2f}"
                            + ("" if ov > 0.5 else "  SLOW DOWN"),
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ov_col,
                            1, cv2.LINE_AA)
                cv2.imshow("R1 scan capture", view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        src.close()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    manifest = {
        "captured": datetime.now().isoformat(timespec="seconds"),
        "source": args.source,
        "rotate": args.rotate,
        "target_fps": args.fps,
        "frames": saved,
        "frames_seen": seen,
        "rejected_blur": rejected_blur,
        "duration_s": round(elapsed, 1),
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if saved:
        first = cv2.imread(str(out / "000000.jpg"))
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(str(out / "preview.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for r in records:
            img = cv2.imread(str(out / r["file"]))
            if img is not None:
                vw.write(img)
        vw.release()

    sharps = [r["sharpness"] for r in records]
    overs = [r["overlap_prev"] for r in records]
    print(f"\n  {saved} frames in {elapsed:.0f}s -> {out.resolve()}")
    print(f"  rejected for blur : {rejected_blur}")
    if sharps:
        print(f"  sharpness  min/med: {min(sharps):.0f} / {sorted(sharps)[len(sharps)//2]:.0f}")
        low = sum(1 for o in overs if o < 0.5)
        print(f"  frames with weak overlap (<0.5): {low}"
              + ("  <- moved too fast in places" if low > len(overs) * 0.2 else ""))
    if saved < 50:
        print("\n  Fewer than 50 frames is unlikely to reconstruct well.")
    print(f"\n  Next: copy {out} to a CUDA machine and run SLAM3R (see SCANNING.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
