"""Compose shopfloor + KUKA + R1, pre-frame a camera on the KUKA cell, and open the GUI.

Plain Isaac Sim app (no Isaac Lab exts -> no h5py crash). All assets in meters.
"""
import argparse
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
args_cli, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args_cli.headless})

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux

BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
SHOP = os.path.join(BASE, "arena_no_walls.usd")
KUKA = os.path.join(BASE, "kuka_assembly.usd")
R1 = os.path.join(BASE, "r1.usd")
OUT = os.path.join(BASE, "scene.usd")

Z_LIFT = 12.1
KUKA_X, KUKA_Y = -9.15, -14.85
# R1 placed INSIDE the KUKA cell footprint, beside the robot
R1_X, R1_Y, R1_Z = KUKA_X + 2.5, KUKA_Y + 1.0, 0.0

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")

UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1500.0)
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 0.0))


def add_ref(prim_path, asset, translate):
    parent = UsdGeom.Xform.Define(stage, prim_path)
    parent.AddTranslateOp().Set(Gf.Vec3d(*translate))
    child = stage.DefinePrim(prim_path + "/asset")
    child.GetReferences().AddReference(asset)
    return parent


add_ref("/World/Shopfloor", SHOP, (0.0, 0.0, Z_LIFT))
add_ref("/World/Kuka", KUKA, (0.0, 0.0, Z_LIFT))
add_ref("/World/R1", R1, (R1_X, R1_Y, R1_Z))

# --- pre-framed camera looking at the KUKA cell ---
cam = UsdGeom.Camera.Define(stage, "/World/CellCamera")
eye = Gf.Vec3d(KUKA_X + 9.0, KUKA_Y + 9.0, 6.0)
target = Gf.Vec3d(KUKA_X, KUKA_Y, 1.2)
cam_xform = Gf.Matrix4d(1).SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
UsdGeom.Xformable(cam).MakeMatrixXform().Set(cam_xform)
cam.CreateFocalLengthAttr(18.0)  # wide-ish so the whole cell fits

ctx.save_as_stage(OUT)
print(f"[SCENE] saved {OUT}", flush=True)

# let assets load, then report where each ended up (confirms R1 loaded + positions)
for _ in range(180):
    simulation_app.update()

try:
    stage = ctx.get_stage()  # save_as_stage swapped in a new stage object
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for path in ("/World/Shopfloor", "/World/Kuka", "/World/R1"):
        prim = stage.GetPrimAtPath(path)
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        print(f"[BBOX] {path}: min=({mn[0]:.2f},{mn[1]:.2f},{mn[2]:.2f}) max=({mx[0]:.2f},{mx[1]:.2f},{mx[2]:.2f})", flush=True)
except Exception as e:
    print(f"[SCENE] bbox diag skipped: {e}", flush=True)

# make the framed camera the active viewport camera
try:
    import omni.kit.viewport.utility as vpu

    vp = vpu.get_active_viewport()
    vp.camera_path = "/World/CellCamera"
    print("[SCENE] viewport set to CellCamera", flush=True)
except Exception as e:
    print(f"[SCENE] could not set viewport camera: {e}", flush=True)

print("[SCENE] window is live", flush=True)
while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()
