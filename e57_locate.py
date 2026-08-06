"""Extract real positions of features/robots from an .e57 scan and (if given CAD correspondences)
compute the scan->CAD 2D rigid transform and output placement poses in the CAD frame.

Run with e57venv:
  e57venv\\Scripts\\python.exe e57_locate.py --file scan.e57 --spec spec.json

spec.json:
{
  "features": {
      "colA":  {"box": [x0,x1,y0,y1]},          # a reference column/corner (scan XY box)
      "colB":  {"box": [x0,x1,y0,y1]},
      "kuka":  {"box": [x0,x1,y0,y1], "robot": true},
      "maira": {"box": [x0,x1,y0,y1], "robot": true},
      "lara":  {"box": [x0,x1,y0,y1], "robot": true},
      "r1":    {"box": [x0,x1,y0,y1], "robot": true}
  },
  "correspondences": {                            # OPTIONAL: CAD (x,y) of the reference features
      "colA": [cadx, cady],
      "colB": [cadx, cady]
  }
}

- For each box: prints scan centroid (median x,y), z-range, and yaw (PCA principal axis of the footprint).
- With >=2 correspondences: solves the best-fit 2D rigid transform (Umeyama/Kabsch, no scale),
  reports RMS fit error, and prints each robot's pose transformed into CAD coordinates.
"""
import argparse
import json

import numpy as np
import pye57

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--scan", type=int, default=0)
args = ap.parse_args()

spec = json.load(open(args.spec))
feats = spec["features"]

e = pye57.E57(args.file)
d = e.read_scan(args.scan, ignore_missing_fields=True, colors=False, intensity=False, transform=True)
P = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], axis=1).astype(np.float64)


def extract(box):
    x0, x1, y0, y1 = box
    m = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    q = P[m]
    if len(q) == 0:
        return None
    cx, cy = np.median(q[:, 0]), np.median(q[:, 1])
    zmin, zmax = q[:, 2].min(), q[:, 2].max()
    # yaw from PCA of the XY footprint (principal axis)
    xy = q[:, :2] - np.array([cx, cy])
    cov = xy.T @ xy / len(xy)
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]
    yaw = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return dict(cx=float(cx), cy=float(cy), zmin=float(zmin), zmax=float(zmax), n=int(len(q)), yaw=yaw)


print("=== scan-frame extraction ===")
res = {}
for name, f in feats.items():
    r = extract(f["box"])
    res[name] = r
    if r is None:
        print(f"  {name:8s}: NO POINTS in box {f['box']}")
    else:
        print(f"  {name:8s}: center=({r['cx']:.3f}, {r['cy']:.3f})  z={r['zmin']:.2f}..{r['zmax']:.2f}  "
              f"yaw={r['yaw']:+.1f}deg  pts={r['n']:,}")

corr = spec.get("correspondences", {})
if len(corr) >= 2:
    names = [n for n in corr if res.get(n)]
    scan_pts = np.array([[res[n]["cx"], res[n]["cy"]] for n in names])
    cad_pts = np.array([corr[n] for n in names], dtype=float)

    # Umeyama rigid 2D (rotation + translation, no scale)
    mu_s, mu_c = scan_pts.mean(0), cad_pts.mean(0)
    S = (scan_pts - mu_s); C = (cad_pts - mu_c)
    H = S.T @ C
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_c - R @ mu_s
    yaw_off = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    # fit error
    pred = (scan_pts @ R.T) + t
    rms = float(np.sqrt(((pred - cad_pts) ** 2).sum(1).mean()))

    print("\n=== scan -> CAD registration (2D rigid) ===")
    print(f"  used {len(names)} correspondences: {names}")
    print(f"  yaw offset = {yaw_off:+.3f} deg,  translation = ({t[0]:.3f}, {t[1]:.3f})")
    print(f"  RMS fit error = {rms*1000:.1f} mm  (per-point: " +
          ", ".join(f"{n}:{np.hypot(*(pred[i]-cad_pts[i]))*1000:.0f}mm" for i, n in enumerate(names)) + ")")

    print("\n=== robot placement poses in CAD frame ===")
    for name, f in feats.items():
        if f.get("robot") and res.get(name):
            r = res[name]
            p = R @ np.array([r["cx"], r["cy"]]) + t
            cad_yaw = r["yaw"] + yaw_off
            print(f"  {name:8s}: CAD (x={p[0]:.3f}, y={p[1]:.3f})  yaw={cad_yaw:+.1f}deg  "
                  f"floor_z(scan)={r['zmin']:.2f}")
else:
    print("\n(no/insufficient correspondences given -> extraction only. "
          "Add >=2 CAD coords in spec['correspondences'] to get CAD placement poses.)")
