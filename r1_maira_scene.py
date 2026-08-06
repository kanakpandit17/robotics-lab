"""R1 humanoid + Neura MAiRA 7-DOF cobot in one scene (kinematic/posed — just to view them
together and verify both imported with geometry)."""
import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
args_cli, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args_cli.headless})

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux

BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
R1 = BASE + r"\unitree_r1.usd"
MAIRA = BASE + r"\maira7M.usd"

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")

UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1200.0)
key = UsdLux.DistantLight.Define(stage, "/World/Key")
key.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 0))

# ground
g = UsdGeom.Cube.Define(stage, "/World/Ground")
g.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(g)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
g.CreateDisplayColorAttr([(0.22, 0.22, 0.25)])


def add_floor_aligned(path, asset, xy):
    p = UsdGeom.Xform.Define(stage, path)
    t = p.AddTranslateOp()
    t.Set(Gf.Vec3d(xy[0], xy[1], 0.0))
    stage.DefinePrim(path + "/asset").GetReferences().AddReference(asset)
    for _ in range(30):
        simulation_app.update()
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    t.Set(Gf.Vec3d(xy[0], xy[1], -mn[2]))
    size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    print(f"[SCENE] {path}: size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})  {'OK geometry' if max(size) > 0.05 else 'EMPTY!'}", flush=True)
    return size


add_floor_aligned("/World/R1", R1, (0.0, 0.0))
add_floor_aligned("/World/MAiRA", MAIRA, (1.4, 0.0))

cam = UsdGeom.Camera.Define(stage, "/World/Cam")
eye = Gf.Vec3d(0.7, -4.0, 2.0)
tgt = Gf.Vec3d(0.7, 0.0, 0.7)
UsdGeom.Xformable(cam).MakeMatrixXform().Set(Gf.Matrix4d(1).SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
try:
    import omni.kit.viewport.utility as vpu
    vpu.get_active_viewport().camera_path = "/World/Cam"
except Exception as e:
    print(f"[SCENE] cam set skipped: {e}", flush=True)

print("[SCENE] R1 + MAiRA loaded — window live", flush=True)
while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()
