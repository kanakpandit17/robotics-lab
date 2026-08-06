#!/usr/bin/env python3
"""Re-rotate an already-captured scan in place.

    python fix_rotation.py scans/cell01 --rotate 90

Saves re-walking the robot when the capture rotation turns out to be wrong.
`--rotate 90` undoes a capture made with `--rotate 270`, and vice versa.

Originals are copied to <dir>.orig the first time, so this is reversible.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

import cv2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--rotate", type=int, choices=[90, 180, 270], required=True)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    d = Path(args.directory)
    jpgs = sorted(d.glob("[0-9]*.jpg"))
    if not jpgs:
        print(f"no frames in {d}", file=sys.stderr)
        return 1

    if not args.no_backup:
        backup = Path(str(d) + ".orig")
        if not backup.exists():
            print(f"backing up originals -> {backup}")
            shutil.copytree(d, backup)

    codes = {90: cv2.ROTATE_90_CLOCKWISE,
             180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    print(f"rotating {len(jpgs)} frames by {args.rotate} deg ...")
    for i, f in enumerate(jpgs):
        im = cv2.imread(str(f))
        if im is None:
            continue
        cv2.imwrite(str(f), cv2.rotate(im, codes[args.rotate]),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(jpgs)}")

    first = cv2.imread(str(jpgs[0]))
    h, w = first.shape[:2]
    print(f"done. frames are now {w}x{h}")

    # The preview is rebuilt from the rotated frames so it matches.
    mp4 = d / "preview.mp4"
    if mp4.exists():
        vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (w, h))
        for f in jpgs:
            im = cv2.imread(str(f))
            if im is not None:
                vw.write(im)
        vw.release()
        print(f"rebuilt {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
