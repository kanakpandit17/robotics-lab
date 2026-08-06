"""View a converted point-cloud USD (from e57_to_usd.py) in Isaac Sim / Isaac Lab.

  ...python.exe e57_view.py --usd C:\\Users\\aimat\\Desktop\\humanoid\\faro_scan_1.usd
  ...python.exe e57_view.py --usd faro_scan_1.usd --probe     # headless check, no window

This opens a STANDARD UsdGeom.Points USD -- no pye57, no fileformat plugin -> no crash.
(Convert first:  e57venv\\Scripts\\python.exe e57_to_usd.py --file scan.e57 --out scan.usd)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=r"C:\Users\aimat\Desktop\humanoid\faro_scan_1.usd")
parser.add_argument("--probe", action="store_true")
parser.add_argument("--headless", action="store_true")
args_cli, _ = parser.parse_known_args()

import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402
_h5 = _types.ModuleType("h5py"); _h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(headless=(args_cli.headless or args_cli.probe))
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

usd_path = args_cli.usd.replace("\\", "/")
print(f"[VIEW] opening {usd_path}", flush=True)
ctx = omni.usd.get_context()
ctx.open_stage(usd_path)
stage = ctx.get_stage()

for _ in range(60):
    simulation_app.update()

total = 0
bbmin = bbmax = None
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Points):
        pts = UsdGeom.Points(prim).GetPointsAttr().Get()
        n = len(pts) if pts else 0
        total += n
        print(f"[VIEW]   {prim.GetPath()} -> {n:,} points", flush=True)
        ext = UsdGeom.Points(prim).GetExtentAttr().Get()
        if ext:
            bbmin, bbmax = np.array(ext[0]), np.array(ext[1])
print(f"[VIEW] total {total:,} points", flush=True)

if args_cli.probe:
    print("[VIEW] probe done.", flush=True)
    simulation_app.close()
    raise SystemExit(0 if total > 0 else 3)

# frame the cloud
try:
    import omni.kit.viewport.utility as vpu
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    vp = vpu.get_active_viewport()
    if bbmin is not None:
        center = (bbmin + bbmax) / 2.0
        d = float(np.linalg.norm(bbmax - bbmin)) or 5.0
        st = ViewportCameraState(vp.camera_path, vp)
        st.set_target_world(Gf.Vec3d(*center.tolist()), True)
        st.set_position_world(Gf.Vec3d(*(center + np.array([0.4, 0.4, 0.3]) * d).tolist()), True)
except Exception as ex:
    print("[VIEW] (auto-frame skipped:", ex, ") -- press F in the viewport to frame.", flush=True)

print("[VIEW] viewing -- close the window to exit.", flush=True)
while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()
