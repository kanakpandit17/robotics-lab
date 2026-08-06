"""Physically simulate R1 driving across the shopfloor: add a ground collider, set the 3 wheel
joints to velocity drive, play physics, and log R1's position so we can confirm it moved.

R1 is a wheeled robot (3-wheel chassis) — it drives, it doesn't walk.
"""
import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--wheel_vel", type=float, default=400.0, help="wheel target velocity (deg/s)")
args_cli, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args_cli.headless})

import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

SCENE = r"C:\Users\aimat\Desktop\humanoid\imported_cad\scene.usd"
ctx = omni.usd.get_context()
ctx.open_stage(SCENE)
stage = ctx.get_stage()

# 1) physics scene
if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

# 2) static ground collider (thin box, top surface at z=0) so R1 doesn't fall through
gp = UsdGeom.Cube.Define(stage, "/World/GroundCollider")
gp.CreateSizeAttr(1.0)
xf = UsdGeom.Xformable(gp)
xf.AddTranslateOp().Set(Gf.Vec3d(-9.15, -14.85, -0.05))  # under the cell
xf.AddScaleOp().Set(Gf.Vec3f(120.0, 120.0, 0.1))
UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
gp.CreateDisplayColorAttr([(0.25, 0.25, 0.28)])

# 3) set the 3 wheel joints to velocity drive (spin), steer joints hold at 0
WHEELS = {"wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3"}
STEERS = {"steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3"}
n_wheel = n_steer = 0
for prim in stage.Traverse():
    name = prim.GetName()
    if name in WHEELS and (prim.HasAPI(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.RevoluteJoint) or "Joint" in prim.GetTypeName()):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(1.0e4)
        drive.CreateTargetVelocityAttr().Set(args_cli.wheel_vel)
        n_wheel += 1
    elif name in STEERS and "Joint" in prim.GetTypeName():
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr().Set(1.0e5)
        drive.CreateDampingAttr().Set(1.0e4)
        drive.CreateTargetPositionAttr().Set(0.0)
        n_steer += 1
print(f"[DRIVE] configured wheels={n_wheel} steers={n_steer}, wheel_vel={args_cli.wheel_vel} deg/s", flush=True)


def r1_center():
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/R1")).ComputeAlignedRange()
    c = (rng.GetMin() + rng.GetMax()) * 0.5
    return c


start = r1_center()
print(f"[DRIVE] R1 start center = ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})", flush=True)

timeline = omni.timeline.get_timeline_interface()
timeline.play()

step = 0
while simulation_app.is_running():
    simulation_app.update()
    step += 1
    if step % 120 == 0:
        c = r1_center()
        d = ((c[0] - start[0]) ** 2 + (c[1] - start[1]) ** 2) ** 0.5
        print(f"[DRIVE] step {step}: R1 center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  moved {d:.2f} m from start", flush=True)

simulation_app.close()
