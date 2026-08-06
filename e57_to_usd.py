"""Convert an .e57 laser scan to a USD point cloud (UsdGeom.Points).

RUN WITH THE ISOLATED ENV (not Isaac's python -- pye57 crashes inside Kit):
    C:\\Users\\aimat\\Desktop\\humanoid\\e57venv\\Scripts\\python.exe e57_to_usd.py --file scan.e57 --out scan.usd

Reads points (+color/intensity) with pye57, downsamples to a safe budget, writes a standard
UsdGeom.Points .usd you can then open directly in Isaac Sim / Isaac Lab (no plugin needed).
"""
import argparse
import time

import numpy as np
import pye57
from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True, help="input .e57")
ap.add_argument("--out", required=True, help="output .usd / .usda")
ap.add_argument("--scan", type=int, default=-1, help="scan index, or -1 for ALL scans")
ap.add_argument("--max-points", type=int, default=3_000_000)
ap.add_argument("--voxel", type=float, default=0.0, help="voxel size (m); 0 = random subsample")
ap.add_argument("--point-size", type=float, default=0.0, help="render width; 0 = auto from scene size")
ap.add_argument("--trim", type=float, default=0.5, help="drop points outside this..100-this percentile per axis (removes scanner outliers); 0 = keep all")
args = ap.parse_args()


def vec3f(a: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a.astype(np.float32)))


e = pye57.E57(args.file)
n_scans = e.scan_count
scans = range(n_scans) if args.scan < 0 else [args.scan]
print(f"[E57] {args.file}: {n_scans} scan(s); reading {list(scans)}", flush=True)

all_pts, all_cols = [], []
have_color = True
for si in scans:
    t0 = time.time()
    d = e.read_scan(si, ignore_missing_fields=True, colors=True, intensity=True, transform=True)
    p = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], axis=1).astype(np.float32)
    if "colorRed" in d and d["colorRed"] is not None:
        c = np.stack([d["colorRed"], d["colorGreen"], d["colorBlue"]], axis=1).astype(np.float32)
        if c.max() > 1.5:
            c /= 255.0
    elif "intensity" in d and d["intensity"] is not None:
        it = np.asarray(d["intensity"], np.float32)
        g = (it - it.min()) / (np.ptp(it) or 1.0)
        c = np.stack([g, g, g], axis=1)
    else:
        c = None
        have_color = False
    all_pts.append(p)
    if c is not None:
        all_cols.append(c)
    print(f"[E57]   scan {si}: {len(p):,} points ({time.time()-t0:.1f}s)", flush=True)

pts = np.concatenate(all_pts, axis=0)
cols = np.concatenate(all_cols, axis=0) if (have_color and all_cols) else None
print(f"[E57] total raw points: {len(pts):,}", flush=True)

# trim scanner outliers (stray far points that stretch the scene)
if args.trim > 0:
    lo = np.percentile(pts, args.trim, axis=0)
    hi = np.percentile(pts, 100.0 - args.trim, axis=0)
    m = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[m]
    cols = cols[m] if cols is not None else None
    print(f"[E57] trimmed to {args.trim}-{100-args.trim} pctile -> {len(pts):,}", flush=True)

# downsample
if args.voxel > 0.0:
    keys = np.floor(pts / args.voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    pts = pts[idx]
    cols = cols[idx] if cols is not None else None
    print(f"[E57] voxel {args.voxel} m -> {len(pts):,}", flush=True)
if len(pts) > args.max_points:
    idx = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
    pts = pts[idx]
    cols = cols[idx] if cols is not None else None
    print(f"[E57] subsampled -> {len(pts):,} (budget {args.max_points:,})", flush=True)

bbmin, bbmax = pts.min(0), pts.max(0)
diag = float(np.linalg.norm(bbmax - bbmin)) or 1.0
psize = args.point_size if args.point_size > 0 else max(diag / 1500.0, 1e-4)
print(f"[E57] bbox {bbmin} .. {bbmax}  (diag={diag:.2f}, point-size={psize:.4f})", flush=True)

# build USD
stage = Usd.Stage.CreateNew(args.out.replace("\\", "/"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)

points = UsdGeom.Points.Define(stage, "/World/Scan")
points.CreatePointsAttr(vec3f(pts))
points.CreateWidthsAttr(Vt.FloatArray([float(psize)]))
points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
points.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*bbmin.tolist()), Gf.Vec3f(*bbmax.tolist())]))
if cols is not None:
    points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(vec3f(cols))
    print("[E57] applied per-point color", flush=True)

stage.GetRootLayer().Save()
print(f"[E57] wrote {args.out}  ({len(pts):,} points)", flush=True)
