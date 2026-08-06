"""ONE physically-simulated Unitree R1 in the KUKA cell, driven by the TRAINED RL walking
policy (model_999.pt). A simple high-level rule feeds the policy a velocity command toward the
red ball obstacle, then zeroes the command a few cm short -> the RL walker walks up and STANDS.

Layer split:
  high level (this file): distance-to-ball rule  -> commanded base velocity (vx, vy, wz)
  low  level (RL policy):  velocity command       -> joint torques that walk / stand upright
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_flat_v2\model_3999.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False  # we want to watch

# h5py stub (Isaac Sim's bundled hdf5.dll shadows h5py -> 0xc0000139 in GUI)
import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

R1_USD = r"C:\Users\aimat\Desktop\humanoid\imported_cad\unitree_r1.usd"

# ---- R1 articulation (identical to r1_walk_train so the trained policy loads) --------------
R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R1_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
            ".*_shoulder_pitch_joint": 0.20, ".*_elbow_joint": 0.60,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "waist_yaw_joint", "waist_roll_joint"],
            effort_limit_sim=300,
            stiffness={".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0, ".*_hip_pitch_joint": 200.0, ".*_knee_joint": 200.0, "waist_yaw_joint": 200.0, "waist_roll_joint": 200.0},
            damping={".*_hip_.*": 5.0, ".*_knee_joint": 5.0, "waist_.*": 5.0},
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=40, stiffness=20.0, damping=2.0, armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint", "head_pitch_joint", "head_yaw_joint"],
            effort_limit_sim=100, stiffness=40.0, damping=10.0, armature=0.01,
        ),
    },
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg  # noqa: E402


@configclass
class R1CellEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.episode_length_s = 100000.0  # effectively never auto-reset
        # swap G1 -> R1 (same remaps as training)
        self.scene.robot = R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.terminations.base_contact = None   # don't auto-reset on brief contacts; our base_z check handles real falls
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint"])
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]
        )
        # clean deploy: no observation noise, no random pushes
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# --------------------------------------------------------------------------- build & run
import importlib.metadata as _metadata  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

env = ManagerBasedRLEnv(cfg=R1CellEnvCfg())
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

# ------------------------------------------------------------------ KUKA-cell scenery + ball
from pxr import Gf, Usd, UsdGeom  # noqa: E402

BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
stage = env.unwrapped.scene.stage
origin = env.unwrapped.scene.env_origins[0].cpu().numpy()  # env-0 world origin

import random  # noqa: E402

OX, OY = float(origin[0]), float(origin[1])
BALL_R = 0.4
STOP_GAP = 0.60                      # stand this far (m) short of the ball surface
R1_HALF = 0.22                       # R1 body half-depth
STOP_DIST = BALL_R + R1_HALF + STOP_GAP

# ONE KUKA cell: a single fenced area containing the KUKA robot (a back corner) AND the open floor
# the R1 walks on. The red ball appears in the open floor; the R1 walks to it, stops 60 cm short,
# then the ball hops to a new spot in the open floor and the R1 walks to that one, and so on.
# The KUKA cell (kuka_assembly.usd) is one ~7.9x7.3 m robotic cell; its ROBOT sits in the +x half,
# leaving OPEN FLOOR in the -x half. We translate the whole cell so its open floor lands on the R1
# spawn (env origin), then draw ONE fence around the cell's real footprint -> R1 + KUKA share it.
KUKA_T = (OX + 12.0, OY + 15.0)            # asset-local (-12,-15) [open floor] -> env origin (0,0)
CELL_X0, CELL_X1 = OX - 1.3, OX + 7.0      # fence = the KUKA cell footprint (with margin)
CELL_Y0, CELL_Y1 = OY - 3.9, OY + 4.2
# open floor in front of the KUKA robot (its near edge is ~x=+3.3) where the ball may appear
OPEN_X0, OPEN_X1 = OX - 0.5, OX + 2.8
OPEN_Y0, OPEN_Y1 = OY - 2.5, OY + 2.8

# ---- obstacles the R1 must steer around (in addition to walking to the ball) ----
MAIRA_XY = (OX + 1.7, OY + 1.4)     # the MAiRA cobot stands here in the open floor
MAIRA_AVOID = 1.15                  # keep the R1 body-centre at least this far from MAiRA's base
KUKA_FRONT = OX + 3.1               # the KUKA robot's near edge (do not cross into it)
KUKA_MARGIN = 0.6
REPEL_GAIN = 1.6                    # how hard obstacles push the R1 away


def _floor_align_ref(path, asset, xy):
    """Reference an asset and lift it so its lowest point sits on z=0 (env ground). Returns bbox."""
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


def pick_ball(rx, ry):
    """A new ball position in the open floor: >=1.5 m from the robot (so it must walk there) and
    clear of MAiRA (so the goal is reachable without entering the cobot)."""
    bx, by = OX, OY
    for _ in range(80):
        bx = random.uniform(OPEN_X0, OPEN_X1)
        by = random.uniform(OPEN_Y0, OPEN_Y1)
        if math.hypot(bx - rx, by - ry) >= 1.5 and math.hypot(bx - MAIRA_XY[0], by - MAIRA_XY[1]) >= MAIRA_AVOID + 0.4:
            break
    return (bx, by)


ball_xy = (OX + 2.0, OY)   # first ball straight ahead in the open floor -> clean straight walk

# red ball obstacle (visual). keep a handle on its transform so we can move it each round.
ball = UsdGeom.Sphere.Define(stage, "/World/CellBall")
ball.CreateRadiusAttr(BALL_R)
ball_t = UsdGeom.Xformable(ball).AddTranslateOp()
ball_t.Set(Gf.Vec3d(ball_xy[0], ball_xy[1], BALL_R))
ball.CreateDisplayColorAttr([(0.9, 0.15, 0.1)])

# yellow fence = the ONE cell boundary, enclosing both the KUKA and the R1's walking floor
for name, cx, cy, sx, sy in [
    ("xp", CELL_X1, (CELL_Y0 + CELL_Y1) / 2, 0.15, CELL_Y1 - CELL_Y0),
    ("xm", CELL_X0, (CELL_Y0 + CELL_Y1) / 2, 0.15, CELL_Y1 - CELL_Y0),
    ("yp", (CELL_X0 + CELL_X1) / 2, CELL_Y1, CELL_X1 - CELL_X0, 0.15),
    ("ym", (CELL_X0 + CELL_X1) / 2, CELL_Y0, CELL_X1 - CELL_X0, 0.15),
]:
    w = UsdGeom.Cube.Define(stage, f"/World/CellFence/{name}")
    w.CreateSizeAttr(1.0)
    wx = UsdGeom.Xformable(w)
    wx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.6))
    wx.AddScaleOp().Set(Gf.Vec3f(sx, sy, 1.2))
    w.CreateDisplayColorAttr([(0.95, 0.75, 0.05)])

# the KUKA cell asset, translated so its open floor is at the R1 spawn and its robot is ahead (+x)
ksize = _floor_align_ref("/World/CellKuka", BASE + r"\kuka_assembly.usd", KUKA_T)
print(f"[CELL] KUKA cell loaded, size=({ksize[0]:.2f},{ksize[1]:.2f},{ksize[2]:.2f})", flush=True)

# the Neura MAiRA cobot, standing in the open floor as a second obstacle the R1 must avoid
msize = _floor_align_ref("/World/CellMaira", BASE + r"\maira7M.usd", MAIRA_XY)
print(f"[CELL] MAiRA loaded at ({MAIRA_XY[0]:.1f},{MAIRA_XY[1]:.1f}), size=({msize[0]:.2f},{msize[1]:.2f},{msize[2]:.2f})", flush=True)
# small marker discs on the floor around the obstacles (visual footprint of the keep-out zones)
for nm, (ox, oy), rr, col in [("maira", MAIRA_XY, MAIRA_AVOID, (0.15, 0.45, 0.9))]:
    disc = UsdGeom.Cylinder.Define(stage, f"/World/KeepOut/{nm}")
    disc.CreateRadiusAttr(rr)
    disc.CreateHeightAttr(0.02)
    UsdGeom.Xformable(disc).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.01))
    disc.CreateDisplayColorAttr([col])

# oblique side camera: R1 + open floor in front, KUKA robot ahead in the same cell.
# (never point head-on into the dense KUKA CAD at close range -> GPU crash.)
cam = UsdGeom.Camera.Define(stage, "/World/CellCam")
eye = Gf.Vec3d(OX - 3.5, OY - 7.5, 4.0)
tgt = Gf.Vec3d(OX + 2.0, OY, 0.6)
UsdGeom.Xformable(cam).MakeMatrixXform().Set(Gf.Matrix4d(1).SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
try:
    import omni.kit.viewport.utility as vpu
    vpu.get_active_viewport().camera_path = "/World/CellCam"
except Exception as e:
    print(f"[CELL] cam set skipped: {e}", flush=True)

print(f"[CELL] first ball at ({ball_xy[0]:.1f},{ball_xy[1]:.1f}); stops {STOP_GAP*100:.0f} cm short, then ball moves", flush=True)

# ------------------------------------------------------------------ drive the command term
cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
cmd_term.cfg.heading_command = False              # don't let it recompute yaw from a heading target
cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)  # never resample random commands
cmd_term.is_standing_env[:] = False               # not a "stand" env -> won't be zeroed internally

CRUISE = 1.0    # forward walking speed (m/s) matching training
SETTLE = 50     # stand still this many steps first, so the R1 finds its footing before walking
ACCEL = 0.03    # max change in commanded forward speed per step (gentle ramp, avoids toppling)
DWELL = 90      # stand at the ball this many steps, then it hops to a new spot
robot = env.unwrapped.scene["robot"]
_r = env.get_observations()
obs = _r[0] if isinstance(_r, (tuple, list)) else _r
print("[CELL] R1 settling, then walking to the ball ...", flush=True)
stopped_announced = False
dwell_start = 0
rounds = 0
prev_vx = 0.0
settle_until = SETTLE
step = 0
while simulation_app.is_running():
    # a reset re-randomizes the command term -> force it back to "driven, not standing"
    cmd_term.is_standing_env[:] = False

    base = robot.data.root_pos_w[0]
    q = robot.data.root_quat_w[0]  # (w, x, y, z)
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]).item(), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]).item())
    dx = ball_xy[0] - base[0].item()
    dy = ball_xy[1] - base[1].item()
    dist = math.hypot(dx, dy)
    base_z = base[2].item()
    up = robot.data.projected_gravity_b[0, 2].item()  # ~ -1.0 upright, -> 0/positive when tipped

    # safety net: if it ever falls, reset and restart the approach (should not happen with the ramp)
    if base_z < 0.35:
        print(f"[CELL] fell at step {step} — resetting and retrying", flush=True)
        with torch.inference_mode():
            _rr = env.reset()
        obs = _rr[0] if isinstance(_rr, (tuple, list)) else _rr
        cmd_term.is_standing_env[:] = False
        prev_vx = 0.0
        settle_until = step + SETTLE
        continue

    # --- high level: settle, then potential-field steering (attract to ball, repel from MAiRA/KUKA) ---
    if step < settle_until:
        target_vx, wz = 0.0, 0.0
    elif dist > STOP_DIST:
        bx0, by0 = base[0].item(), base[1].item()
        # attraction toward the ball (unit vector)
        ax, ay = dx / dist, dy / dist
        # repulsion from MAiRA (push away, growing as we get inside the keep-out radius)
        rxv = ryv = 0.0
        dm = math.hypot(bx0 - MAIRA_XY[0], by0 - MAIRA_XY[1])
        if dm < MAIRA_AVOID and dm > 1e-3:
            s = (MAIRA_AVOID - dm) / MAIRA_AVOID
            rxv += (bx0 - MAIRA_XY[0]) / dm * s * REPEL_GAIN
            ryv += (by0 - MAIRA_XY[1]) / dm * s * REPEL_GAIN
        # repulsion from the KUKA robot's near edge (never cross into it, +x)
        if bx0 > KUKA_FRONT - KUKA_MARGIN:
            rxv += -((bx0 - (KUKA_FRONT - KUKA_MARGIN)) / KUKA_MARGIN) * REPEL_GAIN
        # combined desired heading
        ddx, ddy = ax + rxv, ay + ryv
        des = math.atan2(ddy, ddx)
        yaw_err = math.atan2(math.sin(des - yaw), math.cos(des - yaw))
        # a biped is stable while WALKING and unstable while pivoting in place, so always keep some
        # forward speed and CURVE toward the target rather than spin in place.
        align = max(0.0, math.cos(yaw_err))          # 1 facing target, 0 perpendicular
        speed = max(0.0, min(CRUISE, 0.9 * (dist - STOP_DIST) + 0.25))
        target_vx = speed * (0.45 + 0.55 * align)    # 45%..100% forward -> keeps the gait stable
        wz = max(-0.6, min(0.6, 0.8 * yaw_err))       # gentle turn
    else:
        target_vx, wz = 0.0, 0.0
        if not stopped_announced:
            gap = dist - BALL_R - R1_HALF
            print(f"[CELL] R1 stopped {gap * 100:.0f} cm from the ball surface — standing", flush=True)
            stopped_announced = True
            dwell_start = step
        elif step - dwell_start > DWELL:
            rounds += 1
            ball_xy = pick_ball(base[0].item(), base[1].item())
            ball_t.Set(Gf.Vec3d(ball_xy[0], ball_xy[1], BALL_R))
            print(f"[CELL] ball moved to ({ball_xy[0]:.1f},{ball_xy[1]:.1f}) -> R1 walking to it (round {rounds})", flush=True)
            stopped_announced = False

    # gentle acceleration limit on forward speed
    vx = prev_vx + max(-ACCEL, min(ACCEL, target_vx - prev_vx))
    prev_vx = vx

    cmd_term.vel_command_b[0, 0] = vx
    cmd_term.vel_command_b[0, 1] = 0.0
    cmd_term.vel_command_b[0, 2] = wz

    if step % 50 == 0:
        state = "UPRIGHT" if (base_z > 0.45 and up < -0.6) else "TIPPED/DOWN"
        phase = "settling" if step < settle_until else ("walking" if target_vx > 0 else "standing")
        print(f"[CELL] step={step:5d} pos=({base[0].item():+.2f},{base[1].item():+.2f}) yaw={yaw:+.2f} ball=({ball_xy[0]:.1f},{ball_xy[1]:.1f}) dist={dist:4.2f} vx={vx:.2f} wz={cmd_term.vel_command_b[0,2].item():+.2f} [{phase}] {state}", flush=True)

    with torch.inference_mode():
        _s = env.step(policy(obs))
        obs = _s[0] if isinstance(_s, (tuple, list)) else _s
    step += 1

simulation_app.close()
