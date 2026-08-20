"""Scripted (non-learned) LARA -> R1 -> MAiRA -> R1 -> LARA battery-casing handoff loop.

R1 walks on its TRAINED walking policy (model_3999.pt), exactly as in r1_cell_maira.py -- this
is physically-simulated bipedal locomotion, not a kinematic glide. Arm reach reuses the
goal-directed heuristic from r1_record_ik.py (spherical aim from the shoulder toward a world
target point). That file imported isaaclab.controllers.DifferentialIKController and then did not
use it -- a 5-DOF arm (no wrist pitch/yaw) makes full 6-DOF differential IK under-constrained in
practice, so this repo already settled on the heuristic. We reuse it rather than re-relitigating
that decision.

The scene is the real shopfloor + KUKA robotic cell (arena_no_walls.usd + kuka_assembly.usd,
placed at the same transform compose_scene.py already validated for co-registering them), not a
placeholder box. LARA sits inside the KUKA cell, MAiRA sits well outside it on the open floor,
on the west side -- the real fence mesh (tn__Part1_f5) has its door on that wall. R1 crosses
between them through DOOR_PRE/DOOR_MID/DOOR_POST_XY (see "scene layout" below) -- these mark a
SYNTHETIC gap we draw ourselves (build_cell_fence()) at the right wall, not the exact real door's
coordinates, so the gap R1 walks through is guaranteed to match the one drawn on screen.

LARA has no CAD asset in this project yet (only a taught path and AprilTag station id exist for
it). Per your call, this stands LARA in with a second copy of maira7M.usd -- same footprint class
of cobot, not LARA's real geometry. Swap LARA_USD below once real LARA CAD exists; nothing else
in this script depends on which asset it is.

Nothing here is learned: navigation is potential-field steering to hand-placed waypoints (flagged
below where a learned nav policy would go), and grasp is an idealized kinematic attach -- the
casing is teleported to follow the wrist each frame while "held" (flagged below where a real
grasp/force policy, and a real end-effector, would go; the existing fleet/agent/adapters/
unitree_r1.py already documents that R1 has no gripper on the real hardware).

    python r1_lara_maira_loop.py                      # default: 1 full loop, visible window
    python r1_lara_maira_loop.py --cycles 3            # repeat the loop 3 times
    python r1_lara_maira_loop.py --checkpoint <path>   # use a different walking policy
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_flat_v2\model_3999.pt")
parser.add_argument("--cycles", type=int, default=1, help="how many full 10-step loops to run")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False  # success criteria is "visibly show" -- we want to watch

# h5py stub (Isaac Sim's bundled hdf5.dll shadows h5py -> 0xc0000139 in GUI), same guard every
# other script in this repo uses.
import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as _metadata  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from r1_cfg import R1FlatEnvCfg  # noqa: E402
from loop_states import STEP_LABEL, LoopFSM, Step  # noqa: E402

BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
LARA_USD = BASE + r"\maira7M.usd"   # stand-in -- see module docstring
MAIRA_USD = BASE + r"\maira7M.usd"
SHOP_USD = BASE + r"\arena_no_walls.usd"   # the "no walls" variant, same one compose_scene.py used
KUKA_USD = BASE + r"\kuka_assembly.usd"    # -- open floor around the cell, nothing to collide with

# ----------------------------------------------------------------------------------- scene layout
# Real shopfloor + KUKA robotic cell, not a placeholder. Placement numbers below are NOT guessed:
# compose_scene.py already composed these same two assets together (its output is
# imported_cad/scene.usd) and proved that referencing both at the SAME (x, y, Z_LIFT) transform
# keeps them correctly co-registered -- they were authored in one shared native frame. KUKA_X/
# KUKA_Y is compose_scene.py's own validated reference point for "next to the robot inside the
# cell" (it placed R1 at KUKA_X+2.5, KUKA_Y+1.0 for the same reason).
#
# CONFIRMED by an actual run (the printed bbox lines): the KUKA cell spans x=[-13.1,-5.2],
# y=[-18.5,-11.2] in this world frame, and LARA_XY/HOME_XY/MAIRA_XY below all land where they're
# supposed to relative to it -- inside vs. outside, not guesses anymore.
Z_LIFT = 12.1
KUKA_X, KUKA_Y = -9.15, -14.85   # from compose_scene.py -- validated "inside the KUKA cell" anchor
KUKA_CELL_XMIN, KUKA_CELL_XMAX = -13.1, -5.2   # measured from the printed bbox, not guessed
KUKA_CELL_YMIN, KUKA_CELL_YMAX = -18.5, -11.2

LARA_XY = (-11.46287, -14.66537)   # real: marker cube dropped in the GUI and read back via the
                                    # Script Editor -- confirmed inside the cell bbox
MAIRA_XY = (-15.2, -15.33837)      # real, same method -- confirmed ~2.1 m west of the west wall,
                                    # much closer to the cell than my earlier guess

LARA_TRAY_XY = (LARA_XY[0], LARA_XY[1] - 0.7)      # handoff tray in front of each cobot
MAIRA_TRAY_XY = (MAIRA_XY[0], MAIRA_XY[1] - 0.7)

# kuka_assembly.usd DOES model a real fence (tn__Part1_f5) with a real door -- confirmed by
# clicking it in the viewport with the Measure tool, on the west wall. We're still drawing our
# OWN synthetic fence (build_cell_fence()) rather than using the real mesh, per your call, so the
# gap R1 actually walks through is guaranteed to match a gap you can see, without needing to
# depend on exactly where the real door sits within the single fused fence mesh. DOOR_Y is set
# between LARA's and MAiRA's real y (-14.665 and -15.338) now that both are real numbers, not the
# old placeholder halfway point -- move DOOR_Y / widen DOOR_WIDTH to relocate it if needed.
DOOR_Y = -15.0
DOOR_WIDTH = 1.2
DOOR_PRE_XY = (KUKA_CELL_XMIN + 0.8, DOOR_Y)    # just inside the cell
DOOR_MID_XY = (KUKA_CELL_XMIN, DOOR_Y)          # the wall itself
DOOR_POST_XY = (KUKA_CELL_XMIN - 0.9, DOOR_Y)   # just outside, clear of the cell
TRAY_TOP_Z = 0.69
CASING_Z = 0.75                 # world height for both the resting casing and the arm-reach target
                                 # -- kept inside the (0.60-0.95 m) envelope r1_record_ik.py already
                                 # validated the 5-DOF arm can reach, rather than a guessed number.
LARA_STAND_XY = (LARA_XY[0], LARA_XY[1] - 1.1)     # where R1 stops to reach the tray (~0.4 m gap,
MAIRA_STAND_XY = (MAIRA_XY[0], MAIRA_XY[1] - 1.1)  # same order as r1_record_ik.py's proven ~0.38 m)
HOME_XY = (KUKA_X - 8.0, KUKA_Y - 3.0)      # open floor west of the cell -- still a placeholder,
                                             # unlike LARA_XY/MAIRA_XY; checked >3 m clear of both
                                             # real station positions above, so no collision risk
RETREAT_XY = (KUKA_X - 8.0, KUKA_Y + 1.0)   # clear of both cobots' keep-out radius while MAiRA "works"
                                             # (checked >2.4 m clear of both real positions)
STATION_AVOID = 1.15            # keep-out radius around a station when it is NOT the current target

CASING_SIZE = (0.25, 0.18, 0.12)
HOLD_OFFSET = (0.0, 0.0, -0.06)  # casing position relative to the wrist link while "held"

# right arm only -- v1 is a single-arm handoff, matching the only arm this repo has a proven
# reach heuristic for (r1_record_ik.py). Bimanual/left-arm handoffs are future work.
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]
EE_LINK = "right_wrist_roll_link"

REACH_RAMP, REACH_HOLD, REACH_RETRACT = 60, 40, 60
CRUISE, SETTLE, ACCEL = 1.0, 50, 0.03
WORK_HOLD = 90       # steps MAiRA "performs its task" for (timer stand-in, per the task spec)
SIGNAL_HOLD = 30      # steps LARA's "ready" signal is held before R1 moves (timer stand-in)


# --------------------------------------------------------------------------------------- env cfg
@configclass
class R1LoopEnvCfg(R1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.episode_length_s = 100000.0  # effectively never auto-reset on time
        self.terminations.base_contact = None  # our own base_z check handles real falls (below)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # spawn R1 at HOME_XY (not the env origin) so "navigate home -> LARA" and the fall-recovery
        # reset point are the same real place, not two different things that happen to share a name.
        old_pos = self.scene.robot.init_state.pos
        self.scene.robot.init_state.pos = (HOME_XY[0], HOME_XY[1], old_pos[2])


env = ManagerBasedRLEnv(cfg=R1LoopEnvCfg())
env = RslRlVecEnvWrapper(env)

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24, max_iterations=1, save_interval=50, experiment_name="r1_flat",
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128], activation="elu", actor_obs_normalization=False, critic_obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.008, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
)
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

# ------------------------------------------------------------------------------------ build scene
from pxr import Gf, Usd, UsdGeom  # noqa: E402

stage = env.unwrapped.scene.stage
origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
OX, OY = float(origin[0]), float(origin[1])


def _floor_align_ref(path, asset, xy):
    """Reference an asset and lift it so its lowest point sits on z=0. Same helper as
    r1_cell_maira.py -- proven to load maira7M.usd and unitree_r1.usd correctly."""
    p = UsdGeom.Xform.Define(stage, path)
    t = p.AddTranslateOp()
    t.Set(Gf.Vec3d(xy[0], xy[1], 0.0))
    stage.DefinePrim(path + "/asset").GetReferences().AddReference(asset)
    for _ in range(6):
        simulation_app.update()
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    t.Set(Gf.Vec3d(xy[0], xy[1], -mn[2]))
    return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


def _table(path, xy, top_z, color):
    t = UsdGeom.Cube.Define(stage, path)
    t.CreateSizeAttr(1.0)
    tx = UsdGeom.Xformable(t)
    tx.AddTranslateOp().Set(Gf.Vec3d(OX + xy[0], OY + xy[1], top_z / 2.0))
    tx.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, top_z))
    t.CreateDisplayColorAttr([color])


def _add_ref(path, asset, translate):
    """Reference an asset at an EXACT given transform, no auto-align. Shopfloor and Kuka must both
    use this (not _floor_align_ref) because they're only correctly positioned relative to each
    other when given the identical transform compose_scene.py validated -- auto-aligning each to
    its own bounding box independently would break that registration."""
    p = UsdGeom.Xform.Define(stage, path)
    p.AddTranslateOp().Set(Gf.Vec3d(*translate))
    stage.DefinePrim(path + "/asset").GetReferences().AddReference(asset)
    for _ in range(6):
        simulation_app.update()
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return (mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])


shop_mn, shop_mx = _add_ref("/World/Shopfloor", SHOP_USD, (OX, OY, Z_LIFT))
print(f"[LOOP] Shopfloor loaded, world bbox min=({shop_mn[0]:.1f},{shop_mn[1]:.1f},{shop_mn[2]:.1f}) max=({shop_mx[0]:.1f},{shop_mx[1]:.1f},{shop_mx[2]:.1f})", flush=True)
kuka_mn, kuka_mx = _add_ref("/World/Kuka", KUKA_USD, (OX, OY, Z_LIFT))
print(f"[LOOP] KUKA cell loaded, world bbox min=({kuka_mn[0]:.1f},{kuka_mn[1]:.1f},{kuka_mn[2]:.1f}) max=({kuka_mx[0]:.1f},{kuka_mx[1]:.1f},{kuka_mx[2]:.1f}) -- compare against LARA_XY={LARA_XY}, HOME_XY={HOME_XY}", flush=True)


def _wall(name, x0, x1, y0, y1, color=(0.95, 0.75, 0.05)):
    w = UsdGeom.Cube.Define(stage, f"/World/CellFence/{name}")
    w.CreateSizeAttr(1.0)
    wx = UsdGeom.Xformable(w)
    wx.AddTranslateOp().Set(Gf.Vec3d(OX + (x0 + x1) / 2.0, OY + (y0 + y1) / 2.0, 0.6))
    wx.AddScaleOp().Set(Gf.Vec3f(max(x1 - x0, 0.15), max(y1 - y0, 0.15), 1.2))
    w.CreateDisplayColorAttr([color])


def build_cell_fence():
    """A real, visible fence around the CONFIRMED KUKA cell bbox, with an actual gap left in the
    west wall at the door waypoints -- same UsdGeom.Cube wall pattern r1_cell_maira.py already
    used for this exact cell, visual only (no collider), so it can't get R1 physically stuck.
    Since the nav waypoints (DOOR_PRE/MID/POST_XY) and this fence both derive from
    KUKA_CELL_XMIN/DOOR_Y/DOOR_WIDTH, the gap R1 walks through and the gap you can SEE are
    guaranteed to be the same gap."""
    _wall("east", KUKA_CELL_XMAX, KUKA_CELL_XMAX, KUKA_CELL_YMIN, KUKA_CELL_YMAX)
    _wall("north", KUKA_CELL_XMIN, KUKA_CELL_XMAX, KUKA_CELL_YMAX, KUKA_CELL_YMAX)
    _wall("south", KUKA_CELL_XMIN, KUKA_CELL_XMAX, KUKA_CELL_YMIN, KUKA_CELL_YMIN)
    door_y0, door_y1 = DOOR_Y - DOOR_WIDTH / 2.0, DOOR_Y + DOOR_WIDTH / 2.0
    _wall("west_south", KUKA_CELL_XMIN, KUKA_CELL_XMIN, KUKA_CELL_YMIN, door_y0)
    _wall("west_north", KUKA_CELL_XMIN, KUKA_CELL_XMIN, door_y1, KUKA_CELL_YMAX)
    # door-frame posts either side of the gap, so the opening reads as a doorway, not a broken wall
    _wall("door_post_south", KUKA_CELL_XMIN - 0.05, KUKA_CELL_XMIN + 0.05, door_y0 - 0.06, door_y0, (0.85, 0.15, 0.1))
    _wall("door_post_north", KUKA_CELL_XMIN - 0.05, KUKA_CELL_XMIN + 0.05, door_y1, door_y1 + 0.06, (0.85, 0.15, 0.1))


build_cell_fence()
print(f"[LOOP] cell fence built around x=[{KUKA_CELL_XMIN:.1f},{KUKA_CELL_XMAX:.1f}] y=[{KUKA_CELL_YMIN:.1f},{KUKA_CELL_YMAX:.1f}], door gap at y=[{DOOR_Y - DOOR_WIDTH / 2:.1f},{DOOR_Y + DOOR_WIDTH / 2:.1f}] on the west wall", flush=True)

lsize = _floor_align_ref("/World/Lara", LARA_USD, (OX + LARA_XY[0], OY + LARA_XY[1]))
print(f"[LOOP] LARA stand-in loaded at {LARA_XY}, size=({lsize[0]:.2f},{lsize[1]:.2f},{lsize[2]:.2f})", flush=True)
msize = _floor_align_ref("/World/Maira", MAIRA_USD, (OX + MAIRA_XY[0], OY + MAIRA_XY[1]))
print(f"[LOOP] MAiRA loaded at {MAIRA_XY}, size=({msize[0]:.2f},{msize[1]:.2f},{msize[2]:.2f})", flush=True)

_table("/World/LaraTray", LARA_TRAY_XY, TRAY_TOP_Z, (0.35, 0.35, 0.4))
_table("/World/MairaTray", MAIRA_TRAY_XY, TRAY_TOP_Z, (0.35, 0.35, 0.4))

casing = UsdGeom.Cube.Define(stage, "/World/BatteryCasing")
casing.CreateSizeAttr(1.0)
casing_x = UsdGeom.Xformable(casing)
casing_t = casing_x.AddTranslateOp()
casing_x.AddScaleOp().Set(Gf.Vec3f(*CASING_SIZE))  # CreateSizeAttr(1.0) is a full unit cube, so
                                                     # scale = desired full dimensions directly
                                                     # (matches the ground/table/wall cubes above)
casing.CreateDisplayColorAttr([(0.85, 0.5, 0.05)])
casing_xyz = [OX + LARA_TRAY_XY[0], OY + LARA_TRAY_XY[1], CASING_Z]
casing_t.Set(Gf.Vec3d(*casing_xyz))
casing_held = False

_breadcrumb_count = 0
BREADCRUMB_EVERY = 25   # steps (~0.5 s at 50 Hz) -- dense enough to read as a live trail, sparse
                          # enough that a full multi-cycle run doesn't spawn thousands of prims


def drop_breadcrumb(x, y):
    """A small marker at R1's current world (x, y), left on the floor so its path is visible in
    the viewport as it walks -- literal real-time position tracking, just drawn instead of only
    printed. Each call adds ONE new prim (Isaac Sim has no cheap way to move a single marker AND
    keep the trail behind it at the same time), so BREADCRUMB_EVERY caps how many accumulate."""
    global _breadcrumb_count
    m = UsdGeom.Sphere.Define(stage, f"/World/Trail/p{_breadcrumb_count:05d}")
    m.CreateRadiusAttr(0.05)
    UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.05))
    m.CreateDisplayColorAttr([(0.1, 0.8, 0.9)])
    _breadcrumb_count += 1

cam = UsdGeom.Camera.Define(stage, "/World/LoopCam")
# framed on the LARA<->MAiRA span, not the env origin -- the whole scene now lives out near
# KUKA_X/KUKA_Y, ~15 m from (0, 0), so a camera left at the origin would just show empty floor.
_cam_mid_x = OX + (LARA_XY[0] + MAIRA_XY[0]) / 2.0
_cam_mid_y = OY + (LARA_XY[1] + MAIRA_XY[1]) / 2.0
eye = Gf.Vec3d(_cam_mid_x, _cam_mid_y - 16.0, 10.0)
tgt = Gf.Vec3d(_cam_mid_x, _cam_mid_y, 1.0)
UsdGeom.Xformable(cam).MakeMatrixXform().Set(Gf.Matrix4d(1).SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
try:
    import omni.kit.viewport.utility as vpu
    vpu.get_active_viewport().camera_path = "/World/LoopCam"
except Exception as e:
    print(f"[LOOP] cam set skipped: {e}", flush=True)

# ------------------------------------------------------------------------------- command + arm setup
cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
cmd_term.cfg.heading_command = False
cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)
cmd_term.is_standing_env[:] = False

robot = env.unwrapped.scene["robot"]
arm_ids, _ = robot.find_joints(ARM, preserve_order=True)
arm_ids_t = torch.tensor(arm_ids, device=env.device)
ee_id = robot.find_bodies([EE_LINK])[0][0]
default_arm = robot.data.default_joint_pos[:, arm_ids_t].clone()
print(f"[LOOP] arm_ids={arm_ids} ee_body={ee_id}", flush=True)


def reach_targets(target_w3):
    """Right-arm joint targets that aim the hand toward a world 3D point. Verbatim heuristic from
    r1_record_ik.py (shoulder-relative elevation/azimuth aim), not a jacobian-based IK solve --
    see the module docstring for why. Swap for isaaclab.controllers.DifferentialIKController or a
    cuRobo planner here if/when full 6-DOF pose control is needed."""
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    target_w = torch.tensor([target_w3], device=env.device, dtype=root_pos.dtype)
    target_b = subtract_frame_transforms(root_pos, root_quat, target_w)[0][0]
    fx, fy, fz = float(target_b[0]), float(target_b[1]), float(target_b[2])
    sx, sy, sz = 0.05, -0.16, 0.95          # right shoulder, approx, in base frame
    vx_, vy_, vz_ = fx - sx, fy - sy, fz - sz
    horiz = max(1e-3, math.hypot(vx_, vy_))
    elev = math.atan2(vz_, horiz)
    azim = math.atan2(vy_, vx_)
    rdist = math.sqrt(vx_ * vx_ + vy_ * vy_ + vz_ * vz_)
    q_sp = -1.35 - 1.0 * elev
    q_sr = -0.9 * azim
    q_sy = 0.0
    q_el = max(0.05, 0.9 - 1.3 * min(rdist, 0.7))
    q_wr = 0.0
    return torch.tensor([q_sp, q_sr, q_sy, q_el, q_wr], device=env.device, dtype=root_pos.dtype)


def hand_pos_w():
    p = robot.data.body_state_w[0, ee_id, 0:3]
    return float(p[0]), float(p[1]), float(p[2])


# ------------------------------------------------------------------------------------- Navigator
class Navigator:
    """Potential-field walk-to-target, generalised from the inline loop in r1_cell_maira.py:
    attract to the current waypoint, repel from whichever stations are passed in as obstacles.
    NOT a learned navigation policy -- flag here for where RL-based or map-based (occupancy grid /
    A*) navigation would later replace this.

    begin() takes either a single (x, y) target or a LIST of them -- a waypoint chain, e.g. a
    door crossing: [DOOR_PRE_XY, DOOR_MID_XY, DOOR_POST_XY, MAIRA_STAND_XY]. Every waypoint except
    the last is passed through loosely (0.35 m) and doesn't trigger the arrival settle; only the
    final one does. This is the same "pre/mid/post" granularity as the real taught route in
    paths/route.json (see the module docstring) -- that recording isn't numerically reusable here
    (different, drifty odometry frame), but it's the right shape to model."""

    def __init__(self):
        self.active = False
        self.waypoints = [(0.0, 0.0)]
        self.wp_index = 0
        self.obstacles = []
        self.stop_dist = 0.5
        self.arrived_at = None

    def begin(self, waypoints_xy, obstacles_xy, stop_dist=0.5):
        wps = waypoints_xy if isinstance(waypoints_xy, list) else [waypoints_xy]
        self.active = True
        self.waypoints = [(OX + x, OY + y) for x, y in wps]
        self.wp_index = 0
        self.obstacles = [(OX + o[0], OY + o[1]) for o in obstacles_xy]
        self.stop_dist = stop_dist
        self.arrived_at = None

    def tick(self, base_xy, yaw, step):
        bx, by = base_xy
        tx, ty = self.waypoints[self.wp_index]
        is_final = self.wp_index == len(self.waypoints) - 1
        stop_dist = self.stop_dist if is_final else 0.35
        dx, dy = tx - bx, ty - by
        dist = math.hypot(dx, dy)
        if dist > stop_dist:
            ax, ay = dx / dist, dy / dist
            rx = ry = 0.0
            for ox, oy in self.obstacles:
                dm = math.hypot(bx - ox, by - oy)
                if dm < STATION_AVOID and dm > 1e-3:
                    s = (STATION_AVOID - dm) / STATION_AVOID
                    rx += (bx - ox) / dm * s * 1.6
                    ry += (by - oy) / dm * s * 1.6
            des = math.atan2(ay + ry, ax + rx)
            yaw_err = math.atan2(math.sin(des - yaw), math.cos(des - yaw))
            align = max(0.0, math.cos(yaw_err))
            speed = max(0.0, min(CRUISE, 0.9 * (dist - stop_dist) + 0.25))
            vx = speed * (0.45 + 0.55 * align)
            wz = max(-0.6, min(0.6, 0.8 * yaw_err))
            return vx, wz, False
        if not is_final:
            # reached an intermediate waypoint (e.g. DOOR_PRE) -- move on to the next one
            self.wp_index += 1
            self.arrived_at = None
            return 0.0, 0.0, False
        # arrived at the final waypoint: hold still for a short settle before declaring done, so
        # the walker's footing has actually stabilised rather than just crossing the threshold
        # mid-stride.
        if self.arrived_at is None:
            self.arrived_at = step
        settled = (step - self.arrived_at) > 20
        return 0.0, 0.0, settled

    def finish(self):
        self.active = False


# ------------------------------------------------------------------------------------- ArmReacher
class ArmReacher:
    """Ramp the right arm toward a world target, hold, then retract to the default pose. Grasp/
    release is an idealized kinematic attach (the caller toggles casing_held at the ramp->retract
    boundary), not a physics grasp -- there is no modelled end-effector to close, matching that
    the real R1 has no gripper (see fleet/agent/adapters/unitree_r1.py). Flag here for where a
    learned grasp/force-control policy and a real end-effector model would later replace this."""

    def __init__(self):
        self.active = False
        self.target = None
        self.t0 = 0
        self.attach_done = False

    def begin(self, target_w3, step):
        self.active = True
        self.target = target_w3
        self.t0 = step
        self.attach_done = False

    def tick(self, act, step):
        """Returns (grasp_event, done). grasp_event fires exactly once, at max extension, and is
        where the caller should toggle casing_held."""
        t = step - self.t0
        grasp_event = False
        if t < REACH_RAMP:
            frac = t / REACH_RAMP
            q_des = reach_targets(self.target)
            arm_target = default_arm[0] + frac * (q_des - default_arm[0])
        elif t < REACH_RAMP + REACH_HOLD:
            q_des = reach_targets(self.target)
            arm_target = q_des
            if not self.attach_done:
                grasp_event = True
                self.attach_done = True
        elif t < REACH_RAMP + REACH_HOLD + REACH_RETRACT:
            frac = (t - REACH_RAMP - REACH_HOLD) / REACH_RETRACT
            q_des = reach_targets(self.target)
            arm_target = q_des + frac * (default_arm[0] - q_des)
        else:
            self.active = False
            return grasp_event, True
        act[0, arm_ids_t] = 2.0 * (arm_target - default_arm[0])
        return grasp_event, False


navigator = Navigator()
arm = ArmReacher()

# ----------------------------------------------------------------------------------------- main
fsm = LoopFSM(cycles=args_cli.cycles)
print(f"[LOOP] settling ({SETTLE} steps), then starting a {args_cli.cycles}-cycle loop ...", flush=True)

_r = env.reset()
obs = _r[0] if isinstance(_r, (tuple, list)) else _r

step = 0
prev_vx = 0.0
settle_until = SETTLE
step_started: Step | None = None
step_t0 = 0
timer_hold = 0
sub_phase = "nav"  # for MAIRA_WORKING: "nav" (to retreat point) then "hold" (the work timer)

while simulation_app.is_running() and not fsm.done:
    cmd_term.is_standing_env[:] = False

    base = robot.data.root_pos_w[0]
    q = robot.data.root_quat_w[0]
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]).item(), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]).item())
    base_xy = (base[0].item(), base[1].item())
    base_z = base[2].item()

    # ---- fall safety net: log it against the step that was in progress, reset, retry that step ----
    if step < settle_until + 5 and step > 0 and base_z < 0.35:
        pass  # ignore during initial settle; walker hasn't found its footing yet
    elif base_z < 0.35:
        label = STEP_LABEL[fsm.step] if step_started is not None else "settling"
        print(f"[STEP {fsm.step_number}] {label} ... FAILED: R1 fell, resetting and retrying this step", flush=True)
        _rr = env.reset()
        obs = _rr[0] if isinstance(_rr, (tuple, list)) else _rr
        cmd_term.is_standing_env[:] = False
        prev_vx = 0.0
        settle_until = step + SETTLE
        if casing_held:
            casing_t.Set(Gf.Vec3d(*casing_xyz))
            casing_held = False
        navigator.active = False
        arm.active = False
        step_started = None
        step += 1
        continue

    # ---- enter a new step if we haven't started this one yet ----
    if step_started is not fsm.step:
        step_started = fsm.step
        step_t0 = step
        timer_hold = 0
        sub_phase = "nav"
        print(f"[STEP {fsm.step_number}] {fsm.label()} ...", flush=True)
        if fsm.step == Step.NAV_HOME_TO_LARA:
            # entering the cell: cross the wall post -> mid -> pre, then reach the stand point
            navigator.begin([DOOR_POST_XY, DOOR_MID_XY, DOOR_PRE_XY, LARA_STAND_XY], [MAIRA_XY], stop_dist=0.55)
        elif fsm.step == Step.LARA_PICK:
            arm.begin((casing_xyz[0], casing_xyz[1], CASING_Z), step)
        elif fsm.step == Step.NAV_LARA_TO_MAIRA:
            # leaving the cell: pre -> mid -> post, then on to MAiRA
            navigator.begin([DOOR_PRE_XY, DOOR_MID_XY, DOOR_POST_XY, MAIRA_STAND_XY], [LARA_XY], stop_dist=0.55)
        elif fsm.step == Step.MAIRA_PLACE:
            arm.begin((OX + MAIRA_TRAY_XY[0], OY + MAIRA_TRAY_XY[1], CASING_Z), step)
        elif fsm.step == Step.MAIRA_WORKING:
            navigator.begin(RETREAT_XY, [LARA_XY, MAIRA_XY], stop_dist=0.5)  # stays outside the cell, no door needed
        elif fsm.step == Step.NAV_BACK_TO_MAIRA:
            navigator.begin(MAIRA_STAND_XY, [LARA_XY], stop_dist=0.55)  # also stays outside, no door needed
        elif fsm.step == Step.MAIRA_PICK:
            arm.begin((casing_xyz[0], casing_xyz[1], CASING_Z), step)
        elif fsm.step == Step.NAV_MAIRA_TO_LARA:
            # entering the cell again: same crossing as NAV_HOME_TO_LARA
            navigator.begin([DOOR_POST_XY, DOOR_MID_XY, DOOR_PRE_XY, LARA_STAND_XY], [MAIRA_XY], stop_dist=0.55)
        elif fsm.step == Step.LARA_DELIVER:
            arm.begin((OX + LARA_TRAY_XY[0], OY + LARA_TRAY_XY[1], CASING_Z), step)

    # ---- per-frame locomotion: whatever the active navigator wants, else stand still ----
    if step < settle_until:
        target_vx, wz, nav_done = 0.0, 0.0, True
    elif navigator.active:
        target_vx, wz, nav_done = navigator.tick(base_xy, yaw, step)
    else:
        target_vx, wz, nav_done = 0.0, 0.0, True

    vx = prev_vx + max(-ACCEL, min(ACCEL, target_vx - prev_vx))
    prev_vx = vx
    cmd_term.vel_command_b[0, 0] = vx
    cmd_term.vel_command_b[0, 1] = 0.0
    cmd_term.vel_command_b[0, 2] = wz

    with torch.inference_mode():
        act = policy(obs).clone()
        if arm.active:
            grasp_event, arm_done = arm.tick(act, step)
            if grasp_event:
                casing_held = fsm.step in (Step.LARA_PICK, Step.MAIRA_PICK)
                if fsm.step == Step.MAIRA_PLACE:
                    casing_xyz[:] = [OX + MAIRA_TRAY_XY[0], OY + MAIRA_TRAY_XY[1], CASING_Z]
                    casing_t.Set(Gf.Vec3d(*casing_xyz))
                elif fsm.step == Step.LARA_DELIVER:
                    casing_xyz[:] = [OX + LARA_TRAY_XY[0], OY + LARA_TRAY_XY[1], CASING_Z]
                    casing_t.Set(Gf.Vec3d(*casing_xyz))

        if casing_held:
            hx, hy, hz = hand_pos_w()
            casing_t.Set(Gf.Vec3d(hx + HOLD_OFFSET[0], hy + HOLD_OFFSET[1], hz + HOLD_OFFSET[2]))

        _s = env.step(act)
    obs = _s[0] if isinstance(_s, (tuple, list)) else _s

    # ---- step completion checks ----
    if step >= settle_until:
        completed = False
        if fsm.step == Step.LARA_SIGNAL_READY:
            timer_hold += 1
            completed = timer_hold >= SIGNAL_HOLD
        elif fsm.step in (Step.NAV_HOME_TO_LARA, Step.NAV_LARA_TO_MAIRA, Step.NAV_BACK_TO_MAIRA, Step.NAV_MAIRA_TO_LARA):
            completed = navigator.active and nav_done
            if completed:
                navigator.finish()
        elif fsm.step in (Step.LARA_PICK, Step.MAIRA_PLACE, Step.MAIRA_PICK, Step.LARA_DELIVER):
            completed = not arm.active and step > step_t0  # arm.tick() flips active False on done
        elif fsm.step == Step.MAIRA_WORKING:
            if sub_phase == "nav":
                if navigator.active and nav_done:
                    navigator.finish()
                    sub_phase = "hold"
                    timer_hold = 0
            else:
                timer_hold += 1
                completed = timer_hold >= WORK_HOLD

        if completed:
            print(f"[STEP {fsm.step_number}] {fsm.label()} ... SUCCESS", flush=True)
            fsm.advance()
            step_started = None

    if step % BREADCRUMB_EVERY == 0 and step >= settle_until:
        drop_breadcrumb(base_xy[0], base_xy[1])

    if step % 100 == 0:
        phase = "settling" if step < settle_until else fsm.label()
        print(f"[LOOP] step={step:6d} cycle={fsm.cycle_index + 1}/{args_cli.cycles} pos=({base_xy[0]:+.2f},{base_xy[1]:+.2f}) [{phase}]", flush=True)

    step += 1

print(f"[LOOP] completed {args_cli.cycles} cycle(s) without crashing.", flush=True)
simulation_app.close()
