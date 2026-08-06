"""Render readable top-down views of an .e57 scan to locate equipment (KUKA cell / MAiRA / LARA).
Produces (a) top-down in real color (brightened) and (b) top-down colored by HEIGHT so tall
machines/robots pop out of the floor. Optional --crop to zoom a region.

  e57venv\\Scripts\\python.exe e57_render.py --file scan.e57 --out overview.png
  ... --crop "xmin,xmax,ymin,ymax"     # zoom a region (top-down X-Y)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pye57

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scan", type=int, default=0)
ap.add_argument("--render-points", type=int, default=2_000_000)
ap.add_argument("--trim", type=float, default=0.5)
ap.add_argument("--brighten", type=float, default=2.2)
ap.add_argument("--crop", default="", help="xmin,xmax,ymin,ymax (top-down zoom)")
ap.add_argument("--point-size", type=float, default=0.3)
args = ap.parse_args()

e = pye57.E57(args.file)
d = e.read_scan(args.scan, ignore_missing_fields=True, colors=True, intensity=True, transform=True)
pts = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], axis=1).astype(np.float32)
cols = None
if "colorRed" in d and d["colorRed"] is not None:
    cols = np.stack([d["colorRed"], d["colorGreen"], d["colorBlue"]], axis=1).astype(np.float32)
    if cols.max() > 1.5:
        cols /= 255.0
    print(f"[RENDER] color mean={cols.mean():.3f} max={cols.max():.3f}", flush=True)

if args.trim > 0:
    lo = np.percentile(pts, args.trim, axis=0); hi = np.percentile(pts, 100 - args.trim, axis=0)
    m = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[m]; cols = cols[m] if cols is not None else None

if args.crop:
    xmin, xmax, ymin, ymax = (float(v) for v in args.crop.split(","))
    m = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    pts = pts[m]; cols = cols[m] if cols is not None else None
    print(f"[RENDER] cropped -> {len(pts):,} pts", flush=True)

if len(pts) > args.render_points:
    idx = np.random.default_rng(0).choice(len(pts), args.render_points, replace=False)
    pts = pts[idx]; cols = cols[idx] if cols is not None else None

rgb = None
if cols is not None:
    rgb = np.clip(cols * args.brighten, 0, 1)

height = pts[:, 2]
print(f"[RENDER] {len(pts):,} pts  X {pts[:,0].min():.1f}..{pts[:,0].max():.1f}  "
      f"Y {pts[:,1].min():.1f}..{pts[:,1].max():.1f}  Z {height.min():.1f}..{height.max():.1f}", flush=True)

fig, (axc, axh) = plt.subplots(1, 2, figsize=(24, 11), facecolor="white")
axc.scatter(pts[:, 0], pts[:, 1], c=(rgb if rgb is not None else "gray"), s=args.point_size, marker=".", linewidths=0)
axc.set_title("Top-down — real color (brightened)")
sc = axh.scatter(pts[:, 0], pts[:, 1], c=height, cmap="turbo", s=args.point_size, marker=".", linewidths=0)
axh.set_title("Top-down — colored by HEIGHT (tall machines/robots stand out)")
fig.colorbar(sc, ax=axh, shrink=0.7, label="height (Z)")
from matplotlib.ticker import MultipleLocator
for ax in (axc, axh):
    ax.set_aspect("equal"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.xaxis.set_major_locator(MultipleLocator(5)); ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5)); ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.5, color="k", linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.2, color="k", linewidth=0.3)
fig.suptitle(args.file.replace("\\", "/").split("/")[-1], fontsize=15)
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"[RENDER] wrote {args.out}", flush=True)
