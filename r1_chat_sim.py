"""Isaac Sim R1 controller driven by a JSON command file (written by the chat UI).

Loads the shopfloor+KUKA+R1 scene, adds a ground collider, wraps R1 as an articulation, and
every frame reads r1_cmd.json = {vx, turn, arm, cam_dist, cam_height, cam_yaw} to drive wheels,
pose arms, AND move a follow-camera that always keeps R1 centered and zoomed in.
"""
import argparse
import json
import math
import os
import random

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
args_cli, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args_cli.headless})

import numpy as np
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

BASE = r"C:\Users\aimat\Desktop\humanoid"
SCENE = os.path.join(BASE, "imported_cad", "scene.usd")
CMD_FILE = os.path.join(BASE, "r1_cmd.json")

omni.usd.get_context().open_stage(SCENE)
stage = omni.usd.get_context().get_stage()

if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
    sc = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr(9.81)
gp = UsdGeom.Cube.Define(stage, "/World/GroundCollider")
gp.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(gp)
gx.AddTranslateOp().Set(Gf.Vec3d(-9.15, -14.85, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(140.0, 140.0, 0.1))
UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
gp.CreateDisplayColorAttr([(0.25, 0.25, 0.28)])

# ---- KUKA cell boundary: collider "safety fence" so R1 stays inside the cell ----
CELL = {"xmin": -14.0, "xmax": -4.0, "ymin": -19.0, "ymax": -10.0}
WALL_H, WALL_T = 3.0, 0.3


def add_wall(name, cx, cy, sx, sy):
    w = UsdGeom.Cube.Define(stage, f"/World/CellWalls/{name}")
    w.CreateSizeAttr(1.0)
    wx = UsdGeom.Xformable(w)
    wx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, WALL_H / 2.0))
    wx.AddScaleOp().Set(Gf.Vec3f(sx, sy, WALL_H))
    UsdPhysics.CollisionAPI.Apply(w.GetPrim())
    w.CreateDisplayColorAttr([(0.95, 0.75, 0.05)])  # yellow safety fence


_cxm = (CELL["xmin"] + CELL["xmax"]) / 2.0
_cym = (CELL["ymin"] + CELL["ymax"]) / 2.0
_lenx = CELL["xmax"] - CELL["xmin"]
_leny = CELL["ymax"] - CELL["ymin"]
add_wall("xplus", CELL["xmax"], _cym, WALL_T, _leny)
add_wall("xminus", CELL["xmin"], _cym, WALL_T, _leny)
add_wall("yplus", _cxm, CELL["ymax"], _lenx, WALL_T)
add_wall("yminus", _cxm, CELL["ymin"], _lenx, WALL_T)
print("[R1SIM] cell fence added", flush=True)

# ---- goal marker (green sphere) for autonomous navigation ----
goal_marker = UsdGeom.Sphere.Define(stage, "/World/GoalMarker")
goal_marker.CreateRadiusAttr(0.35)
goal_marker_t = UsdGeom.Xformable(goal_marker).AddTranslateOp()
goal_marker_t.Set(Gf.Vec3d(-9.0, -14.0, 0.4))
goal_marker.CreateDisplayColorAttr([(0.1, 0.9, 0.2)])

root_path = None
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and str(prim.GetPath()).startswith("/World/R1"):
        root_path = str(prim.GetPath())
        break
print(f"[R1SIM] articulation root: {root_path}", flush=True)

world = World(stage_units_in_meters=1.0)
r1 = SingleArticulation(prim_path=root_path, name="r1")
world.scene.add(r1)
world.reset()
r1.initialize()

dof = list(r1.dof_names)
print(f"[R1SIM] {len(dof)} dofs", flush=True)


def ids(names):
    return [dof.index(n) for n in names if n in dof]


wheel_idx = ids([f"wheel_motor_joint{i}" for i in (1, 2, 3)])
steer_idx = ids([f"steer_motor_joint{i}" for i in (1, 2, 3)])
larm_idx = ids([f"left_arm_joint{i}" for i in range(1, 7)])
rarm_idx = ids([f"right_arm_joint{i}" for i in range(1, 7)])

num = r1.num_dof
kps = np.full(num, 8.0e4, dtype=np.float32)
kds = np.full(num, 2.0e3, dtype=np.float32)
for wi in wheel_idx:
    kps[wi] = 0.0
    kds[wi] = 3.0e3
r1.get_articulation_controller().set_gains(kps=kps, kds=kds)

# capture the clean upright pose at startup (before any tipping) for the "stand up" reset
_p0, _q0 = r1.get_world_pose()
UPRIGHT_QUAT = np.array(_q0, dtype=np.float32)
SPAWN_POS = np.array([float(_p0[0]), float(_p0[1]), float(_p0[2]) + 0.15], dtype=np.float32)
print(f"[R1SIM] upright pose captured: pos={SPAWN_POS.tolist()} quat={UPRIGHT_QUAT.tolist()}", flush=True)


def stand_up():
    """Reset R1 to its upright spawn pose (center of cell), zero motion + arms."""
    r1.set_world_pose(position=SPAWN_POS, orientation=UPRIGHT_QUAT)
    r1.set_joint_velocities(np.zeros(num, dtype=np.float32))
    r1.set_joint_positions(np.zeros(num, dtype=np.float32))
    print("[R1SIM] stand up: reset to upright", flush=True)

ARM_POSES = {
    "home": {"l": [0, 0, 0, 0, 0, 0], "r": [0, 0, 0, 0, 0, 0]},
    "raise": {"l": [0.0, -1.4, 0.0, 1.2, 0.0, 0.0], "r": [0.0, 1.4, 0.0, -1.2, 0.0, 0.0]},
    "wave": {"l": [0, 0, 0, 0, 0, 0], "r": [0.0, 1.4, 0.0, -1.0, 0.0, 0.0]},
    "reach": {"l": [0.6, -0.8, 0.0, 1.0, 0.0, 0.0], "r": [-0.6, 0.8, 0.0, -1.0, 0.0, 0.0]},
}
WHEEL_SPEED = 4.0  # gentler so R1 tips less
STEER_MAX = 0.5
NAV_SPEED = 3.0    # wheel speed while auto-navigating
GOAL_TOL = 0.8     # distance to consider the goal reached


def base_yaw():
    try:
        _, q = r1.get_world_pose()  # (w, x, y, z)
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    except Exception:
        return 0.0


def random_goal():
    return (
        random.uniform(CELL["xmin"] + 1.5, CELL["xmax"] - 1.5),
        random.uniform(CELL["ymin"] + 1.5, CELL["ymax"] - 1.5),
    )

# ---- follow camera ----
cam_prim = stage.GetPrimAtPath("/World/CellCamera")
if not cam_prim.IsValid():
    cam_prim = UsdGeom.Camera.Define(stage, "/World/CellCamera").GetPrim()
cam_xf = UsdGeom.Xformable(cam_prim)
cam_xf.ClearXformOpOrder()
cam_matrix_op = cam_xf.AddTransformOp()
try:
    import omni.kit.viewport.utility as vpu

    vpu.get_active_viewport().camera_path = "/World/CellCamera"
except Exception as e:
    print(f"[R1SIM] viewport cam set skipped: {e}", flush=True)


def update_camera(dist, height, yaw):
    try:
        pos, _ = r1.get_world_pose()
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    except Exception:
        cx, cy, cz = -6.7, -13.9, 0.3
    target = Gf.Vec3d(cx, cy, cz + 0.8)
    eye = Gf.Vec3d(cx - dist * math.cos(yaw), cy - dist * math.sin(yaw), cz + height)
    m = Gf.Matrix4d(1).SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    cam_matrix_op.Set(m)


def read_cmd():
    try:
        with open(CMD_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


if not os.path.isfile(CMD_FILE):
    json.dump({"seq": 0, "vx": 0.0, "turn": 0.0, "arm": "home"}, open(CMD_FILE, "w"))

omni.timeline.get_timeline_interface().play()
print("[R1SIM] ready — follow-camera on, reading r1_cmd.json", flush=True)

wave_t = 0.0
step = 0
last_stand = 0
last_nav = 0
nav_active = False
nav_target = (-9.0, -14.0)
rarm2_i = rarm_idx[1] if len(rarm_idx) > 1 else None
while simulation_app.is_running():
    cmd = read_cmd()

    # one-shot "stand up" when the counter increments
    stand_req = int(cmd.get("stand", 0))
    if stand_req != last_stand:
        stand_up()
        last_stand = stand_req

    vx = float(cmd.get("vx", 0.0))
    turn = float(cmd.get("turn", 0.0))
    arm = cmd.get("arm", "home") or "home"
    cam_dist = float(cmd.get("cam_dist", 6.0))
    cam_height = float(cmd.get("cam_height", 3.5))
    cam_yaw = float(cmd.get("cam_yaw", 0.7))

    positions = np.zeros(num, dtype=np.float32)
    velocities = np.zeros(num, dtype=np.float32)
    for wi in wheel_idx:
        velocities[wi] = vx * WHEEL_SPEED
    for si in steer_idx:
        positions[si] = turn * STEER_MAX

    pose = ARM_POSES.get(arm, ARM_POSES["home"])
    lp, rp = list(pose["l"]), list(pose["r"])
    if arm == "wave":
        wave_t += 0.05
        rp[5] = 0.6 * math.sin(wave_t * 4.0)
    for k, ji in enumerate(larm_idx):
        positions[ji] = lp[k] if k < len(lp) else 0.0
    for k, ji in enumerate(rarm_idx):
        positions[ji] = rp[k] if k < len(rp) else 0.0

    # ---- autonomous navigation to the goal marker ----
    nav_req = int(cmd.get("nav", 0))
    if nav_req != last_nav:
        nav_target = random_goal()
        goal_marker_t.Set(Gf.Vec3d(nav_target[0], nav_target[1], 0.4))
        nav_active = True
        last_nav = nav_req
        print(f"[NAV] new goal ({nav_target[0]:.1f},{nav_target[1]:.1f})", flush=True)
    if vx != 0.0 or turn != 0.0:  # any manual drive cancels autonomous nav
        nav_active = False
    if nav_active:
        try:
            p, _ = r1.get_world_pose()
            dx, dy = nav_target[0] - float(p[0]), nav_target[1] - float(p[1])
            dist = math.hypot(dx, dy)
            if dist < GOAL_TOL:
                nav_active = False
                for wi in wheel_idx:
                    velocities[wi] = 0.0
                print(f"[NAV] reached goal (dist={dist:.2f})", flush=True)
            else:
                yaw = base_yaw()
                desired = math.atan2(dy, dx)
                need = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))  # steer needed, [-pi,pi]
                if abs(need) <= math.pi / 2.0:
                    steer_ang, wheel_v = need, NAV_SPEED  # goal ahead: steer + drive forward
                else:
                    # goal behind: mirror the steer into the +/-90 cone and drive in reverse
                    mirror = need - math.copysign(math.pi, need)
                    steer_ang = math.atan2(math.sin(mirror), math.cos(mirror))
                    wheel_v = -NAV_SPEED
                steer_ang = max(-1.55, min(1.55, steer_ang))  # respect +/-90 deg joint limit
                for si in steer_idx:
                    positions[si] = steer_ang
                for wi in wheel_idx:
                    velocities[wi] = wheel_v
                if step % 60 == 0:
                    print(
                        f"[NAV] pos=({float(p[0]):.2f},{float(p[1]):.2f}) desired={desired:+.2f} "
                        f"steer={steer_ang:+.2f} wv={wheel_v:+.1f} dist={dist:.2f}",
                        flush=True,
                    )
        except Exception as e:
            print(f"[NAV] err {e}", flush=True)

    # safety: if R1 ever gets past the cell fence, stop the wheels (no running off)
    try:
        _p, _ = r1.get_world_pose()
        _px, _py = float(_p[0]), float(_p[1])
        if _px < CELL["xmin"] or _px > CELL["xmax"] or _py < CELL["ymin"] or _py > CELL["ymax"]:
            for wi in wheel_idx:
                velocities[wi] = 0.0
    except Exception:
        pass

    r1.apply_action(ArticulationAction(joint_positions=positions, joint_velocities=velocities))
    update_camera(cam_dist, cam_height, cam_yaw)
    world.step(render=True)

    step += 1
    if step % 120 == 0:
        try:
            pos, _ = r1.get_world_pose()
            jp = r1.get_joint_positions()
            rarm2 = float(jp[rarm2_i]) if rarm2_i is not None else 0.0
            print(
                f"[DIAG] cmd(vx={vx:+.1f},turn={turn:+.1f},arm={arm},cam_yaw={cam_yaw:.2f}) "
                f"| base=({float(pos[0]):.2f},{float(pos[1]):.2f}) r_arm_j2={rarm2:+.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"[DIAG] err: {e}", flush=True)

simulation_app.close()
