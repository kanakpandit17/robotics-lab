#!/usr/bin/env python3
"""Undistort a fisheye scan into pinhole frames SLAM3R can use.

    python undistort.py scans/cell01 --tune          # find the right FOV by eye
    python undistort.py scans/cell01 --in-fov 150    # apply to every frame

SLAM3R inherits DUSt3R's assumption of roughly pinhole imagery, and its own
recon_tips.md lists wide-angle lenses and unfamiliar distortion as known failure
modes. The R1's camera is a hard fisheye, so this runs first.

Why square output
-----------------
The I2P/L2W models run at 224x224. DUSt3R's loader resizes the short side to 224
and centre-crops a square, so a 1280x720 frame loses 44% of its width before the
network sees it. Undistorting straight to a square means the crop is a no-op and
nothing is wasted -- and the pixels it would have discarded are the badly warped
edges anyway.

Calibrating without a chessboard
--------------------------------
Proper calibration needs the robot in front of a printed target. Failing that,
--tune overlays a straight-line grid on one frame and gives you a slider: turn
it until the cell's aluminium beams and door frames actually look straight. The
model is equidistant (r = f*theta), which is what most fisheyes approximate, so
one parameter gets you most of the way there.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

VIGNETTE_LUMA = 25      # below this in the median frame is lens barrel, not scene
VALID_COL_FRAC = 0.98   # a row/column this valid survives the auto-crop


def build_maps(src_w: int, src_h: int, in_fov: float,
               out_size: int, out_fov: float) -> tuple[np.ndarray, np.ndarray]:
    """Remap tables from an equidistant fisheye to a pinhole camera.

    in_fov is the full horizontal angle the source frame spans; out_fov the
    full angle the square output should cover. Both in degrees.
    """
    f_in = (src_w / 2.0) / np.radians(in_fov / 2.0)
    f_out = (out_size / 2.0) / np.tan(np.radians(out_fov / 2.0))
    cx_in, cy_in = src_w / 2.0, src_h / 2.0
    c_out = out_size / 2.0

    u, v = np.meshgrid(np.arange(out_size, dtype=np.float32),
                       np.arange(out_size, dtype=np.float32))
    x = (u - c_out) / f_out
    y = (v - c_out) / f_out

    rho = np.sqrt(x * x + y * y)
    theta = np.arctan(rho)
    # At the principal point rho is 0 and the direction is undefined; the scale
    # limit there is f_in / f_out, which keeps the centre pixel put.
    scale = np.where(rho > 1e-9, f_in * theta / np.maximum(rho, 1e-9), f_in / f_out)

    return (cx_in + x * scale).astype(np.float32), (cy_in + y * scale).astype(np.float32)


def vignette_mask(frames: list[Path], samples: int = 12) -> np.ndarray:
    """True where the lens barrel occludes the sensor across the whole scan."""
    idx = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    stack = []
    for i in idx:
        im = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            stack.append(im)
    if not stack:
        raise SystemExit("could not read any frames")
    return np.median(np.stack(stack), axis=0) < VIGNETTE_LUMA


def valid_crop(map_x: np.ndarray, map_y: np.ndarray, src_w: int, src_h: int,
               dark: np.ndarray) -> tuple[int, int, int, int]:
    """Largest centred axis-aligned rect containing only real, in-frame pixels."""
    # Beyond the lens's own image circle there is nothing to sample, only
    # extrapolation, so bound by that as well as by the sensor rectangle.
    ok = ((map_x >= 0) & (map_x < src_w - 1) &
          (map_y >= 0) & (map_y < src_h - 1) &
          (np.hypot(map_x - src_w / 2.0, map_y - src_h / 2.0) <= src_w / 2.0))

    sx = np.clip(map_x, 0, src_w - 1).astype(int)
    sy = np.clip(map_y, 0, src_h - 1).astype(int)
    ok &= ~dark[sy, sx]

    n = ok.shape[0]
    rows = ok.mean(axis=1) >= VALID_COL_FRAC
    cols = ok.mean(axis=0) >= VALID_COL_FRAC
    if not rows.any() or not cols.any():
        return 0, 0, n, n

    def span(good: np.ndarray) -> tuple[int, int]:
        idx = np.flatnonzero(good)
        return int(idx[0]), int(idx[-1] + 1)

    y0, y1 = span(rows)
    x0, x1 = span(cols)
    return x0, y0, x1, y1


def auto_fov(frames: list[Path], args: argparse.Namespace,
             lo: int = 110, hi: int = 210, step: int = 5) -> float:
    """Pick the input FOV that makes the scene's straight edges straightest.

    A built environment is full of genuinely straight lines -- extrusion, door
    frames, ceiling beams. Under- or over-correct the fisheye and those break
    into short segments; get it right and the line detector returns long ones.
    Maximising total detected length is a usable stand-in for a chessboard.
    """
    idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    imgs = [cv2.imread(str(frames[i])) for i in idx]
    imgs = [im for im in imgs if im is not None]
    h, w = imgs[0].shape[:2]
    lsd = cv2.createLineSegmentDetector()

    best_fov, best_score = float(lo), -1.0
    for fov in range(lo, hi + 1, step):
        mx, my = build_maps(w, h, float(fov), args.out_size, args.out_fov)
        total = 0.0
        for im in imgs:
            gray = cv2.cvtColor(
                cv2.remap(im, mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT),
                cv2.COLOR_BGR2GRAY)
            seg = lsd.detect(gray)[0]
            if seg is None:
                continue
            s = seg.reshape(-1, 4)
            length = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
            # Segments along the black border are an artefact of the remap,
            # not scene structure, so keep only those on real pixels.
            mx_, my_ = (s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2
            on_image = gray[np.clip(my_, 0, args.out_size - 1).astype(int),
                            np.clip(mx_, 0, args.out_size - 1).astype(int)] > 12
            total += length[(length >= 60) & on_image].sum()
        print(f"  {fov:>4} deg  {total:>9.0f}")
        if total > best_score:
            best_fov, best_score = float(fov), total

    print(f"auto: in-fov {best_fov:g} deg")
    return best_fov


def tune(frames: list[Path], args: argparse.Namespace) -> int:
    """Slider over in-FOV with a straight-line grid to judge against."""
    mid = cv2.imread(str(frames[len(frames) // 2]))
    h, w = mid.shape[:2]
    win = "undistort - drag until the beams are straight, q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("in FOV", win, int(args.in_fov), 220, lambda _v: None)
    cv2.createTrackbar("out FOV", win, int(args.out_fov), 170, lambda _v: None)

    while True:
        in_fov = max(40, cv2.getTrackbarPos("in FOV", win))
        out_fov = max(30, cv2.getTrackbarPos("out FOV", win))
        mx, my = build_maps(w, h, in_fov, args.out_size, out_fov)
        view = cv2.remap(mid, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

        step = args.out_size // 12
        for k in range(step, args.out_size, step):
            cv2.line(view, (k, 0), (k, args.out_size), (0, 200, 255), 1)
            cv2.line(view, (0, k), (args.out_size, k), (0, 200, 255), 1)
        cv2.putText(view, f"in {in_fov}  out {out_fov}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print(f"\nchosen:  --in-fov {in_fov} --out-fov {out_fov}")
            break
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--out", help="default: <directory>_undist")
    ap.add_argument("--in-fov", type=float, default=160.0,
                    help="full horizontal angle the fisheye frame spans (deg); "
                         "160 was measured for the R1 by --auto")
    ap.add_argument("--auto", action="store_true",
                    help="search for --in-fov by maximising scene line straightness")
    ap.add_argument("--out-fov", type=float, default=100.0,
                    help="full angle the pinhole output covers (deg)")
    ap.add_argument("--out-size", type=int, default=720,
                    help="square output edge; 224 is all SLAM3R keeps")
    ap.add_argument("--no-crop", action="store_true",
                    help="keep black borders instead of trimming to valid pixels")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    d = Path(args.directory)
    frames = sorted(d.glob("[0-9]*.jpg"))
    if not frames:
        print(f"no frames in {d}", file=sys.stderr)
        return 1

    if args.tune:
        return tune(frames, args)

    if args.auto:
        args.in_fov = auto_fov(frames, args)

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out = Path(args.out) if args.out else Path(str(d) + "_undist")
    out.mkdir(parents=True, exist_ok=True)

    mx, my = build_maps(w, h, args.in_fov, args.out_size, args.out_fov)

    if args.no_crop:
        x0, y0, x1, y1 = 0, 0, args.out_size, args.out_size
    else:
        print("measuring lens vignette ...")
        x0, y0, x1, y1 = valid_crop(mx, my, w, h, vignette_mask(frames))
    cw, ch = x1 - x0, y1 - y0
    print(f"{len(frames)} frames  {w}x{h} fisheye -> {cw}x{ch} pinhole "
          f"(in {args.in_fov:g} deg, out {args.out_fov:g} deg)")

    for i, f in enumerate(frames):
        im = cv2.imread(str(f))
        if im is None:
            continue
        warped = cv2.remap(im, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)[y0:y1, x0:x1]
        cv2.imwrite(str(out / f.name), warped,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames)}")

    manifest = d / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        m["undistort"] = {"in_fov": args.in_fov, "out_fov": args.out_fov,
                          "model": "equidistant", "size": [cw, ch],
                          "source": str(d)}
        (out / "manifest.json").write_text(json.dumps(m, indent=2))

    vw = cv2.VideoWriter(str(out / "preview.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (cw, ch))
    for f in sorted(out.glob("[0-9]*.jpg")):
        im = cv2.imread(str(f))
        if im is not None:
            vw.write(im)
    vw.release()

    print(f"\nwrote {out}\n  check {out/'preview.mp4'} before uploading")
    return 0


if __name__ == "__main__":
    sys.exit(main())
