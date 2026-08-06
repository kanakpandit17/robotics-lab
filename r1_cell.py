"""One Unitree R1 humanoid in the KUKA cell: walks toward the ball obstacle and stops a few cm
away — kinematic + posed so it can NEVER fall or flail (a legged robot needs a balance policy
to stand under physics; that is a separate follow-on).
"""
import argparse
import math

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
args_cli, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args_cli.headless})

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux

BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
SHOP = BASE + r"\arena_no_walls.usd"
KUKA = BASE + r"\kuka_assembly.usd"
R1 = BASE + r"\unitree_r1.usd"

Z_LIFT = 12.1
CELL = {"xmin": -14.0, "xmax": -4.0, "ymin": -19.0, "ymax": -10.0}
R1_START = (-11.0, -14.5)
BALL_POS = (-6.5, -14.5)
STOP_GAP = 0.08          # stop this far (m) short of the ball surface (a few cm)
BALL_R = 0.4
WALK_SPEED = 0.012       # m per frame

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")

UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1500.0)
key = UsdLux.DistantLight.Define(stage, "/World/Key")
key.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 0))


def add_ref(path, asset, translate):
    p = UsdGeom.Xform.Define(stage, path)
    t = p.AddTranslateOp()
    t.Set(Gf.Vec3d(*translate))
    stage.DefinePrim(path + "/asset").GetReferences().AddReference(asset)
    return p, t


add_ref("/World/Shopfloor", SHOP, (0, 0, Z_LIFT))
add_ref("/World/Kuka", KUKA, (0, 0, Z_LIFT))

# yellow cell fence (visual)
WALL_H = 3.0
for name, cx, cy, sx, sy in [
    ("xp", CELL["xmax"], (CELL["ymin"] + CELL["ymax"]) / 2, 0.2, CELL["ymax"] - CELL["ymin"]),
    ("xm", CELL["xmin"], (CELL["ymin"] + CELL["ymax"]) / 2, 0.2, CELL["ymax"] - CELL["ymin"]),
    ("yp", (CELL["xmin"] + CELL["xmax"]) / 2, CELL["ymax"], CELL["xmax"] - CELL["xmin"], 0.2),
    ("ym", (CELL["xmin"] + CELL["xmax"]) / 2, CELL["ymin"], CELL["xmax"] - CELL["xmin"], 0.2),
]:
    w = UsdGeom.Cube.Define(stage, f"/World/Fence/{name}")
    w.CreateSizeAttr(1.0)
    wx = UsdGeom.Xformable(w)
    wx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, WALL_H / 2))
    wx.AddScaleOp().Set(Gf.Vec3f(sx, sy, WALL_H))
    w.CreateDisplayColorAttr([(0.95, 0.75, 0.05)])

# ball obstacle (red sphere)
ball = UsdGeom.Sphere.Define(stage, "/World/Ball")
ball.CreateRadiusAttr(BALL_R)
UsdGeom.Xformable(ball).AddTranslateOp().Set(Gf.Vec3d(BALL_POS[0], BALL_POS[1], BALL_R))
ball.CreateDisplayColorAttr([(0.9, 0.15, 0.1)])

# Unitree R1 (posed, kinematic) — placed then auto-aligned so its feet sit on the floor
r1_parent, r1_t = add_ref("/World/R1", R1, (R1_START[0], R1_START[1], 0.0))
for _ in range(60):
    simulation_app.update()

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/R1")).ComputeAlignedRange()
mn, mx = rng.GetMin(), rng.GetMax()
foot_z = mn[2]
r1_z = -foot_z  # lift so feet touch z=0
r1_t.Set(Gf.Vec3d(R1_START[0], R1_START[1], r1_z))
print(f"[R1CELL] R1 size=({mx[0]-mn[0]:.2f},{mx[1]-mn[1]:.2f},{mx[2]-mn[2]:.2f}) floor-align z={r1_z:.2f}", flush=True)

# camera looking at the R1 / ball corridor
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
eye = Gf.Vec3d(-8.5, -10.0, 3.5)
tgt = Gf.Vec3d(-8.5, -14.5, 0.8)
UsdGeom.Xformable(cam).MakeMatrixXform().Set(Gf.Matrix4d(1).SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
try:
    import omni.kit.viewport.utility as vpu

    vpu.get_active_viewport().camera_path = "/World/Cam"
except Exception as e:
    print(f"[R1CELL] cam set skipped: {e}", flush=True)

print("[R1CELL] window live — R1 walks to the ball and stops a few cm short", flush=True)
x, y = R1_START
r1_half = max(mx[0] - mn[0], mx[1] - mn[1]) / 2.0  # R1 body half-width
stop_dist = BALL_R + r1_half + STOP_GAP            # center-to-center stop distance
stopped = False
while simulation_app.is_running():
    if not stopped:
        dx, dy = BALL_POS[0] - x, BALL_POS[1] - y
        dist = math.hypot(dx, dy)
        if dist <= stop_dist:
            stopped = True
            gap = dist - BALL_R - r1_half
            print(f"[R1CELL] stopped {gap * 100:.0f} cm from ball surface", flush=True)
        else:
            x += WALK_SPEED * dx / dist
            y += WALK_SPEED * dy / dist
            # face the ball: yaw so +x looks toward it
            r1_t.Set(Gf.Vec3d(x, y, r1_z))
    simulation_app.update()

simulation_app.close()
