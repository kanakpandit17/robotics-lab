"""R1 hierarchical LEGGED NAVIGATION inside the KUKA cell (milestone M3).

A NEW high-level PPO policy outputs velocity commands (where to go); the FROZEN walking policy
(r1_flat_v2 / policy.pt) turns them into an upright gait (how to move). Everything is confined to
a bounded CELL: the robot spawns inside it, an OBSTACLE sits inside it, goals are sampled only
inside it, and leaving the cell ends the episode with a penalty — so the R1 learns to reach goals
and avoid the obstacle WITHOUT ever leaving the cell.

Adapted from Isaac Lab's manager_based/navigation task (anymal_c) for the Unitree R1.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--play", action="store_true", default=False)
parser.add_argument("--celldemo", action="store_true", default=False, help="GUI: ONE R1 in the visible KUKA cell navigating to a moving goal, avoiding the ball")
parser.add_argument("--eval", action="store_true", default=False, help="headless behavioral eval: goal-reaching, avoidance, in-cell, upright")
parser.add_argument("--resume", action="store_true", default=False, help="load --checkpoint and continue training")
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_nav_v2\model_1499.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not (args_cli.play or args_cli.celldemo)
if args_cli.eval:
    args_cli.headless = True

import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import isaaclab_tasks.manager_based.navigation.mdp as mdp  # noqa: E402
from r1_cfg import R1FlatEnvCfg  # noqa: E402

POLICY_PT = r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_flat_v2\exported\policy.pt"
LOW_LEVEL_ENV_CFG = R1FlatEnvCfg()

# ---- cell geometry (env-relative, metres) ----
CELL_HALF = 3.5        # cell is a 7x7 m box centred on each env origin; robot may not leave it
OBST_OFFSET = (1.5, 0.0)   # obstacle position inside the cell, relative to env origin
# Ball radius is ~0.35 m in the demo; keep the robot's CENTER >= 0.95 m away so it stays a full
# 60 cm clear of the ball SURFACE (robot body ~0.22 m half-width) -> it visibly curves around it.
OBST_KEEPOUT = 1.5     # soft "too close" radius (penalty ramps up inside this)
OBST_HIT = 0.95        # safety radius (= 60 cm from the ball surface) -> episode ends
GOAL_HALF = 3.0        # goals sampled within +/- this (inside the cell, with margin)


# --------------------------------------------------------------- custom cell / obstacle mdp
def _obstacle_pos_w(env):
    off = torch.tensor([OBST_OFFSET[0], OBST_OFFSET[1], 0.0], device=env.device)
    return env.scene.env_origins + off  # (N, 3), one obstacle per env


def obstacle_pos_b(env):
    """Obstacle position relative to the robot base, in the base frame (xy) — the policy's 'eyes'."""
    robot = env.scene["robot"]
    rel_w = _obstacle_pos_w(env) - robot.data.root_pos_w
    rel_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_w)
    return rel_b[:, :2]


def obstacle_avoidance_penalty(env, keepout: float):
    """0 outside the keep-out radius, ramping to 1 at the obstacle centre."""
    robot = env.scene["robot"]
    d = torch.norm(_obstacle_pos_w(env)[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    return torch.clamp(1.0 - d / keepout, min=0.0)


def hit_obstacle(env, hit_radius: float):
    robot = env.scene["robot"]
    d = torch.norm(_obstacle_pos_w(env)[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    return d < hit_radius


def out_of_cell(env, half: float):
    robot = env.scene["robot"]
    rel = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (rel.abs() > half).any(dim=1)


def progress_to_goal(env, command_name: str):
    """Dense reward = speed (m/s) of the base TOWARD the goal, in the base frame. Directly rewards
    making progress each step (not just being at the goal) — breaks the 'stand still and play safe'
    local optimum."""
    cmd = env.command_manager.get_command(command_name)
    goal_b = cmd[:, :2]
    dist = torch.norm(goal_b, dim=1, keepdim=True).clamp(min=0.2)
    goal_dir = goal_b / dist
    vel_b = env.scene["robot"].data.root_lin_vel_b[:, :2]
    return (vel_b * goal_dir).sum(dim=1)


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.5, -0.5), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},  # start on the near side of the obstacle
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )


class ScaledPreTrainedPolicyAction(mdp.PreTrainedPolicyAction):
    """The walker was ONLY trained on commands vx in [0,1], vy in [-0.5,0.5], wz in [-1,1].
    The high-level policy's raw output is unbounded, so squash it into that exact range with tanh
    (smooth -> gradients still flow). Without this the nav policy sends commands the walker has
    never seen, the robot doesn't go where intended, and goal-reaching can never be learned.
    """

    def process_actions(self, actions: torch.Tensor):
        vx = (torch.tanh(actions[:, 0]) + 1.0) * 0.5   # 0 .. 1  m/s forward
        vy = torch.tanh(actions[:, 1]) * 0.5           # -0.5 .. 0.5 m/s lateral
        wz = torch.tanh(actions[:, 2]) * 1.0           # -1 .. 1 rad/s yaw
        self._raw_actions[:] = torch.stack([vx, vy, wz], dim=1)


@configclass
class ScaledPreTrainedPolicyActionCfg(mdp.PreTrainedPolicyActionCfg):
    class_type: type = ScaledPreTrainedPolicyAction


@configclass
class ActionsCfg:
    pre_trained_policy_action: ScaledPreTrainedPolicyActionCfg = ScaledPreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=POLICY_PT,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        obstacle = ObsTerm(func=obstacle_pos_b)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)  # falling, hitting obstacle, leaving cell
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh, weight=0.5, params={"std": 2.0, "command_name": "pose_command"}
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh, weight=0.5, params={"std": 0.2, "command_name": "pose_command"}
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs, weight=-0.05, params={"command_name": "pose_command"}
    )
    progress = RewTerm(
        func=progress_to_goal, weight=1.0, params={"command_name": "pose_command"}
    )
    obstacle_avoidance = RewTerm(
        func=obstacle_avoidance_penalty, weight=-3.0, params={"keepout": OBST_KEEPOUT}
    )


@configclass
class CommandsCfg:
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-GOAL_HALF, GOAL_HALF), pos_y=(-GOAL_HALF, GOAL_HALF), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="waist_yaw_link"), "threshold": 1.0},
    )
    obstacle_collision = DoneTerm(func=hit_obstacle, params={"hit_radius": OBST_HIT})
    left_cell = DoneTerm(func=out_of_cell, params={"half": CELL_HALF})


@configclass
class R1NavEnvCfg(ManagerBasedRLEnvCfg):
    scene = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


# --------------------------------------------------------------------------- train / play
import importlib.metadata as _metadata  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

cfg = R1NavEnvCfg()
if args_cli.eval:
    cfg.scene.num_envs = 256
elif args_cli.celldemo:
    cfg.scene.num_envs = 1
elif args_cli.play:
    cfg.scene.num_envs = 32
else:
    cfg.scene.num_envs = args_cli.num_envs
if args_cli.play or args_cli.eval or args_cli.celldemo:
    cfg.observations.policy.enable_corruption = False

env = ManagerBasedRLEnv(cfg=cfg)
env = RslRlVecEnvWrapper(env)

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=args_cli.max_iterations,
    save_interval=50,
    experiment_name="r1_nav_v4",
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128], activation="elu", actor_obs_normalization=False, critic_obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
)
log_dir = r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_nav_v4"
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=(None if args_cli.play else log_dir), device=env.device)

if args_cli.celldemo:
    from pxr import Gf, Usd, UsdGeom  # noqa: E402

    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    cm = env.unwrapped.command_manager
    stage = env.unwrapped.scene.stage
    origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
    OX, OY = float(origin[0]), float(origin[1])
    obst_xy = (OX + OBST_OFFSET[0], OY + OBST_OFFSET[1])
    BALL_R = 0.35

    # red ball = the obstacle the policy avoids
    ball = UsdGeom.Sphere.Define(stage, "/World/CellBall")
    ball.CreateRadiusAttr(BALL_R)
    UsdGeom.Xformable(ball).AddTranslateOp().Set(Gf.Vec3d(obst_xy[0], obst_xy[1], BALL_R))
    ball.CreateDisplayColorAttr([(0.9, 0.15, 0.1)])

    # yellow fence = the cell boundary the R1 is confined to (7x7 m)
    for name, cx, cy, sx, sy in [
        ("xp", OX + CELL_HALF, OY, 0.12, 2 * CELL_HALF), ("xm", OX - CELL_HALF, OY, 0.12, 2 * CELL_HALF),
        ("yp", OX, OY + CELL_HALF, 2 * CELL_HALF, 0.12), ("ym", OX, OY - CELL_HALF, 2 * CELL_HALF, 0.12),
    ]:
        w = UsdGeom.Cube.Define(stage, f"/World/CellFence/{name}")
        w.CreateSizeAttr(1.0)
        wx = UsdGeom.Xformable(w)
        wx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.5))
        wx.AddScaleOp().Set(Gf.Vec3f(sx, sy, 1.0))
        w.CreateDisplayColorAttr([(0.95, 0.75, 0.05)])

    # green goal marker (moves to the current target each episode)
    goalm = UsdGeom.Sphere.Define(stage, "/World/GoalMarker")
    goalm.CreateRadiusAttr(0.18)
    goal_t = UsdGeom.Xformable(goalm).AddTranslateOp()
    goal_t.Set(Gf.Vec3d(OX, OY, 0.18))
    goalm.CreateDisplayColorAttr([(0.1, 0.8, 0.2)])

    # KUKA robot as cell scenery, floor-aligned, in the background (side-view camera avoids the
    # head-on dense-CAD GPU crash)
    KUKA = r"C:\Users\aimat\Desktop\humanoid\imported_cad\kuka_assembly.usd"
    kp = UsdGeom.Xform.Define(stage, "/World/CellKuka")
    kt = kp.AddTranslateOp()
    kt.Set(Gf.Vec3d(OX + CELL_HALF - 0.5, OY + CELL_HALF - 0.5, 0.0))
    stage.DefinePrim("/World/CellKuka/asset").GetReferences().AddReference(KUKA)
    for _ in range(6):
        simulation_app.update()
    krng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(stage.GetPrimAtPath("/World/CellKuka")).ComputeAlignedRange()
    kt.Set(Gf.Vec3d(OX + CELL_HALF - 0.5, OY + CELL_HALF - 0.5, -krng.GetMin()[2]))

    # side camera on the cell
    cam = UsdGeom.Camera.Define(stage, "/World/CellCam")
    eye = Gf.Vec3d(OX, OY - CELL_HALF - 4.0, 3.2)
    tgt = Gf.Vec3d(OX, OY, 0.6)
    UsdGeom.Xformable(cam).MakeMatrixXform().Set(Gf.Matrix4d(1).SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
    try:
        import omni.kit.viewport.utility as vpu
        vpu.get_active_viewport().camera_path = "/World/CellCam"
    except Exception as e:
        print(f"[CELL-NAV] cam set skipped: {e}", flush=True)

    _r = env.get_observations()
    obs = _r[0] if isinstance(_r, (tuple, list)) else _r
    print("[CELL-NAV] ONE R1 navigating in the KUKA cell (green = goal, red = ball to avoid)", flush=True)
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            _s = env.step(policy(obs))
            obs = _s[0] if isinstance(_s, (tuple, list)) else _s
        # move the green marker to the live goal (goal is given in the robot's base frame)
        goal_b = cm.get_command("pose_command")[0, :3]
        rp = robot.data.root_pos_w[0]
        rq = robot.data.root_quat_w[0]
        goal_w = rp + math_utils.quat_apply(rq.unsqueeze(0), goal_b.unsqueeze(0))[0]
        goal_t.Set(Gf.Vec3d(float(goal_w[0]), float(goal_w[1]), 0.18))
        if step % 60 == 0:
            dgoal = torch.norm(goal_b[:2]).item()
            dobst = ((rp[0].item() - obst_xy[0]) ** 2 + (rp[1].item() - obst_xy[1]) ** 2) ** 0.5
            up = robot.data.projected_gravity_b[0, 2].item()
            print(f"[CELL-NAV] step={step:5d} goal_dist={dgoal:4.2f}m  ball_dist={dobst:4.2f}m  upright={'Y' if up < -0.7 else 'N'}", flush=True)
        step += 1
elif args_cli.eval:
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    cm = env.unwrapped.command_manager
    origins = env.unwrapped.scene.env_origins
    off = torch.tensor([OBST_OFFSET[0], OBST_OFFSET[1], 0.0], device=env.device)
    _r = env.get_observations()
    obs = _r[0] if isinstance(_r, (tuple, list)) else _r
    tot_within = tot_upright = tot_keepout = tot_outcell = 0.0
    sum_dgoal = 0.0
    min_obst = 1e9
    dones_total = 0
    n = 0
    print(f"[NAV-EVAL] {args_cli.checkpoint}  ({cfg.scene.num_envs} envs, {1000} steps)", flush=True)
    for i in range(1000):
        with torch.inference_mode():
            _s = env.step(policy(obs))
        obs = _s[0] if isinstance(_s, (tuple, list)) else _s
        dones = _s[2] if isinstance(_s, (tuple, list)) and len(_s) > 2 else None
        root = robot.data.root_pos_w
        goal_b = cm.get_command("pose_command")[:, :2]
        dgoal = torch.norm(goal_b, dim=1)
        dobst = torch.norm((origins + off)[:, :2] - root[:, :2], dim=1)
        relcell = (root[:, :2] - origins[:, :2]).abs().max(dim=1).values
        up = robot.data.projected_gravity_b[:, 2]
        tot_within += (dgoal < 0.5).float().mean().item()
        sum_dgoal += dgoal.mean().item()
        tot_upright += (up < -0.7).float().mean().item()
        tot_keepout += (dobst < OBST_KEEPOUT).float().mean().item()
        tot_outcell += (relcell > CELL_HALF).float().mean().item()
        min_obst = min(min_obst, dobst.min().item())
        if dones is not None:
            dones_total += int(dones.sum().item())
        n += 1
        if i % 200 == 0:
            print(f"[NAV-EVAL] step {i:4d}  at-goal={tot_within/n*100:4.1f}%  upright={tot_upright/n*100:5.1f}%  mean_goal_dist={sum_dgoal/n:4.2f}m", flush=True)
    print("[NAV-EVAL] ================ RESULT ================", flush=True)
    print(f"[NAV-EVAL] at-goal (<0.5m):      {tot_within/n*100:5.1f}% of the time", flush=True)
    print(f"[NAV-EVAL] mean distance to goal: {sum_dgoal/n:5.2f} m", flush=True)
    print(f"[NAV-EVAL] upright:               {tot_upright/n*100:5.1f}%", flush=True)
    print(f"[NAV-EVAL] time inside keep-out:  {tot_keepout/n*100:5.1f}%  (closest ever {min_obst:.2f} m; hit<{OBST_HIT})", flush=True)
    print(f"[NAV-EVAL] time outside cell:     {tot_outcell/n*100:5.1f}%", flush=True)
    print(f"[NAV-EVAL] resets over run:       {dones_total} (falls / hits / cell-exits / goal-timeouts)", flush=True)
    print("R1_NAV_EVAL_DONE", flush=True)
elif args_cli.play:
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    _r = env.get_observations()
    obs = _r[0] if isinstance(_r, (tuple, list)) else _r
    print("[NAV-PLAY] R1 navigating inside the cell", flush=True)
    while simulation_app.is_running():
        with torch.inference_mode():
            _s = env.step(policy(obs))
            obs = _s[0] if isinstance(_s, (tuple, list)) else _s
else:
    if args_cli.resume:
        runner.load(args_cli.checkpoint)
        print(f"[NAV-TRAIN] resumed from {args_cli.checkpoint}", flush=True)
    print("[NAV-TRAIN] R1 hierarchical in-cell navigation PPO starting ...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("R1_NAV_TRAIN_DONE", flush=True)
simulation_app.close()
