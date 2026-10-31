"""R1 omnidirectional STAIR locomotion via RL -- after arXiv:2603.07928 (Jiang et al., 2026):
"Omnidirectional Humanoid Locomotion on Stairs via Unsafe Stepping Penalty and Sparse LiDAR
Elevation Mapping". Trained here in Isaac Lab, then tested on ARENA2036 staircase 3 in MuJoCo
(eval_mujoco.py) -- the paper trains in Isaac Lab too.

    python stairs-r1/r1_stairs_train.py                           # train (headless)
    python stairs-r1/r1_stairs_train.py --play   --checkpoint ... # watch
    python stairs-r1/r1_stairs_train.py --export --checkpoint ... # policy.pt + r1_stairs_meta.json for MuJoCo

Taken from the paper:
  * proprio obs: base ang vel, projected gravity, command, joint pos, joint vel, last action
    (NO base linear velocity for the actor; the critic gets it)
  * robot-centric elevation map 1.4 x 1.0 m at 0.05 m, 5 frames stacked
  * the dense UNSAFE STEPPING penalty: riser-collision term + unsupported-sole (edge) term
  * reward weights from their table, termination -250
  * 10-level terrain curriculum: stairs up / down with step height 0..0.23 m, treads 0.25..0.6 m,
    slopes up / down to 0.4 rad, flat
Changed, and why:
  * elevation map is ground truth from a ray caster (sim test first; their LiDAR + EGAU pipeline is
    the real-robot step) with +-2 cm noise
  * actor/critic are MLPs (rsl_rl's ActorCritic), not their MLP+CNN; the 5-frame map is flattened
  * an extra 0.22 m tread width in the curriculum: staircase 3 goes 22.35 cm, below their 0.25 m
  * edge term: the paper's sign convention isn't fully specified, so it is implemented as what it
    targets -- the fraction of the sole that is unsupported (> 3 cm drop), weighted by how close the
    foot is to the ground. Joint-torque L2 is on torques normalised by the effort limit (-0.2 on raw
    N.m^2 would swamp everything)
  * effort limits come from unitree_r1/R1.urdf (60 / 50 / 33 N.m), not r1_cfg.py's 300 N.m
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=12000)   # the paper's curves run 12k
parser.add_argument("--play", action="store_true", default=False)
parser.add_argument("--export", action="store_true", default=False, help="headless: TorchScript policy + meta json for eval_mujoco.py")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_stairs\model_11999.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not args_cli.play
if args_cli.export:
    args_cli.headless = True

import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import re  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.envs import mdp  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.sensors import RayCasterCfg, patterns  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.noise import UniformNoiseCfg as Unoise  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg  # noqa: E402
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp  # noqa: E402

HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(HERE.parent))   # r1_cfg.py lives at the repo root
from r1_cfg import R1_CFG  # noqa: E402

NOMINAL_HEIGHT = 0.72          # pelvis above the ground it stands on (r1_cfg.py)
SOLE_BELOW_ANKLE = 0.053       # ankle_roll_link frame -> sole, from the STL
SOLE_HALF = (0.095, 0.036)     # sole half-extents; centre is +0.04 m forward of the ankle frame
MAP_SIZE, MAP_RES, MAP_HISTORY = (1.4, 1.0), 0.05, 5
MAP_FORWARD = 0.2              # map centre this far ahead of the pelvis: more terrain in front
FOOT_GRID, FOOT_RES = (0.40, 0.25), 0.025


# ----------------------------------------------------------------- effort limits from the URDF
def urdf_efforts():
    root = ET.parse(HERE.parent / "unitree_r1" / "R1.urdf").getroot()
    out = {}
    for j in root.iter("joint"):
        lim = j.find("limit")
        if j.get("type") == "revolute" and lim is not None and float(lim.get("effort", 0)) > 0:
            out[j.get("name")] = float(lim.get("effort"))
    return out


EFFORT = urdf_efforts()


def _with_urdf_limits(act_cfg):
    exprs = act_cfg.joint_names_expr
    lim = {n: e for n, e in EFFORT.items() if any(re.fullmatch(x, n) for x in exprs)}
    return act_cfg.replace(effort_limit_sim=lim)


R1_STAIRS_CFG = R1_CFG.replace(actuators={k: _with_urdf_limits(v) for k, v in R1_CFG.actuators.items()})


# ----------------------------------------------------------------- terrain (10 curriculum levels)
def _stairs(width, inv):
    cls = terrain_gen.MeshInvertedPyramidStairsTerrainCfg if inv else terrain_gen.MeshPyramidStairsTerrainCfg
    return cls(proportion=0.08, step_height_range=(0.0, 0.23), step_width=width,
               platform_width=3.0, border_width=1.0, holes=False)


STAIRS_TERRAINS = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.06),
        # robots spawn at the centre: on a pyramid's top platform (walk down) or in an inverted
        # pyramid's pit (walk up), so both directions are trained
        **{f"up_{w:.2f}": _stairs(w, False) for w in (0.22, 0.25, 0.35, 0.45, 0.60)},
        **{f"down_{w:.2f}": _stairs(w, True) for w in (0.22, 0.25, 0.35, 0.45, 0.60)},
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.07, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.07, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
    },
)


# ----------------------------------------------------------------- observations
def height_map(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    """Terrain height below the pelvis, minus the nominal standing height: 0 on flat ground at
    standing height, positive where the ground drops away, negative where it rises (a step up).
    Uses the robot root, not the sensor frame, so eval_mujoco.py can reproduce it exactly."""
    sensor, robot = env.scene.sensors[sensor_cfg.name], env.scene[asset_cfg.name]
    h = robot.data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2] - NOMINAL_HEIGHT
    return torch.nan_to_num(h, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


# ----------------------------------------------------------------- rewards
def track_exp(env, std, kind, command_name="base_velocity", asset_cfg=SceneEntityCfg("robot")):
    robot, cmd = env.scene[asset_cfg.name], env.command_manager.get_command(command_name)
    if kind == "lin":
        err = (cmd[:, :2] - robot.data.root_lin_vel_b[:, :2]).square().sum(dim=1)
    else:
        err = (cmd[:, 2] - robot.data.root_ang_vel_b[:, 2]).square()
    return torch.exp(-err / std**2)


def base_height_exp(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    """exp(-400 e^2) on pelvis height above the terrain right under it (centre of the height map)."""
    sensor, robot = env.scene.sensors[sensor_cfg.name], env.scene[asset_cfg.name]
    hits = sensor.data.ray_hits_w
    under = (hits[..., :2] - robot.data.root_pos_w[:, None, :2]).norm(dim=-1) < 0.11   # rays right below the pelvis
    z = torch.nan_to_num(hits[..., 2], nan=0.0)
    ground = (z * under).sum(dim=1) / under.sum(dim=1).clamp_min(1)
    err = robot.data.root_pos_w[:, 2] - ground - NOMINAL_HEIGHT
    return torch.exp(-400.0 * err.square())


def action_smoothness(env):
    """|| a_t - 2 a_{t-1} + a_{t-2} ||^2"""
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    if not hasattr(env, "_prev2_action"):
        env._prev2_action = torch.zeros_like(a)
    out = (a - 2 * prev + env._prev2_action).square().sum(dim=1)
    env._prev2_action = prev.clone()
    return out


def torques_normalised(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_effort_lim"):
        env._effort_lim = torch.tensor([EFFORT.get(n, 100.0) for n in robot.joint_names], device=env.device)
    return (robot.data.applied_torque / env._effort_lim).square().sum(dim=1)


class _FootGrid:
    """Per-foot ray grid, re-indexed once into (ny, nx) so neighbours are known."""

    def __init__(self, sensor, device):
        rs = sensor.ray_starts[0, :, :2].clone()                 # sensor-frame xy of every ray
        order = torch.argsort(rs[:, 1] * 1e3 + rs[:, 0])         # row-major: y rows, x fastest
        self.order = order
        xs, ys = rs[order, 0], rs[order, 1]
        self.nx = int(torch.unique(torch.round(xs / FOOT_RES)).numel())
        self.ny = xs.numel() // self.nx
        self.sole = ((xs.abs() <= SOLE_HALF[0]) & (ys.abs() <= SOLE_HALF[1])).to(device)


def unsafe_stepping(env, asset_cfg, d_unsafe=0.15, d_min=0.10, cone_deg=30.0, w_colli=1.0, w_edge=1.0):
    """Dense penalty from 2603.07928, per foot, summed (<= 0).

    collision: foot moving horizontally toward a riser closer than d_unsafe (riser = grid cell that
      is above the sole AND a sharp height jump, so slopes are exempt), scaled by approach speed.
    edge: fraction of the sole over a drop (> 3 cm below the highest ground under the sole) -- a foot
      hanging off a step nose -- scaled up as the foot gets within d_min of the ground.
    """
    robot = env.scene[asset_cfg.name]
    total = torch.zeros(env.num_envs, device=env.device)
    cos_cone = torch.cos(torch.deg2rad(torch.tensor(cone_deg)))
    for k, side in enumerate(("left", "right")):
        sensor = env.scene.sensors[f"{side}_foot_scan"]
        key = f"_grid_{side}"
        if not hasattr(env, key):
            setattr(env, key, _FootGrid(sensor, env.device))
        g = getattr(env, key)
        hits = sensor.data.ray_hits_w[:, g.order]                         # (N, R, 3) row-major
        h = torch.nan_to_num(hits[..., 2], nan=-10.0)
        body = asset_cfg.body_ids[k]
        foot_pos = robot.data.body_pos_w[:, body]
        sole_z = foot_pos[:, 2] - SOLE_BELOW_ANKLE
        sole_xy = hits[:, g.sole, :2].mean(dim=1)

        # ---- edge / unsupported sole
        h_sole = h[:, g.sole]
        support = h_sole.max(dim=1).values
        unsupported = (h_sole < support.unsqueeze(1) - 0.03).float().mean(dim=1)
        near = (1.0 - (sole_z - support) / d_min).clamp(0.0, 1.0)
        r_edge = -unsupported * near

        # ---- riser collision
        hg = h.view(env.num_envs, g.ny, g.nx)
        jump = torch.zeros_like(hg)
        jump[:, :, 1:] = torch.maximum(jump[:, :, 1:], (hg[:, :, 1:] - hg[:, :, :-1]).abs())
        jump[:, :, :-1] = torch.maximum(jump[:, :, :-1], (hg[:, :, 1:] - hg[:, :, :-1]).abs())
        jump[:, 1:, :] = torch.maximum(jump[:, 1:, :], (hg[:, 1:, :] - hg[:, :-1, :]).abs())
        jump[:, :-1, :] = torch.maximum(jump[:, :-1, :], (hg[:, 1:, :] - hg[:, :-1, :]).abs())
        sharp = jump.view(env.num_envs, -1) > 0.05                       # > 63 deg over 2.5 cm
        riser = sharp & (h > sole_z.unsqueeze(1) + 0.02)
        v = robot.data.body_lin_vel_w[:, body, :2]
        speed = v.norm(dim=1, keepdim=True).clamp_min(1e-6)
        d = hits[..., :2] - sole_xy.unsqueeze(1)                          # (N, R, 2)
        dist = d.norm(dim=-1).clamp_min(1e-6)
        in_cone = ((d * (v / speed).unsqueeze(1)).sum(-1) / dist) > cos_cone
        cand = riser & in_cone
        dist_c = torch.where(cand, dist, torch.full_like(dist, 1e3))
        nearest = dist_c.argmin(dim=1)
        dn = dist_c.gather(1, nearest[:, None]).squeeze(1)
        dvec = d.gather(1, nearest[:, None, None].expand(-1, 1, 2)).squeeze(1)
        p = ((v * dvec).sum(-1) / dn.clamp_min(1e-6)).clamp_min(0.0)
        r_colli = -p * (1.0 - dn / d_unsafe).clamp_min(0.0)
        r_colli = torch.where(dn < 1e2, r_colli, torch.zeros_like(r_colli))

        total += w_colli * r_colli + w_edge * r_edge
    return total


# ----------------------------------------------------------------- env
def _foot_scanner(side):
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=FOOT_RES, size=list(FOOT_GRID)),
        debug_vis=False, mesh_prim_paths=["/World/ground"],
    )


@configclass
class R1StairsEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # ---- G1 -> R1 (same swaps as r1_cfg.R1FlatEnvCfg, on the ROUGH base)
        self.scene.robot = R1_STAIRS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["waist_yaw_link", "pelvis_link"]
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"), "mass_distribution_params": (-1.0, 2.0), "operation": "add"},
        )
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)
        self.events.push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
                                           params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

        # ---- terrain + sensors
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS
        self.scene.terrain.max_init_terrain_level = 3
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis_link",
            offset=RayCasterCfg.OffsetCfg(pos=(MAP_FORWARD, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=MAP_RES, size=list(MAP_SIZE)),
            debug_vis=False, mesh_prim_paths=["/World/ground"],
        )
        self.scene.left_foot_scan = _foot_scanner("left")
        self.scene.right_foot_scan = _foot_scanner("right")

        # ---- omnidirectional commands (paper evaluates 0.7 fwd/back, 0.5 lateral)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ---- observations
        P = self.observations.policy
        P.base_lin_vel = None                      # actor is blind to base velocity (critic gets it)
        P.height_scan = None
        P.height_map = ObsTerm(func=height_map, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                               noise=Unoise(n_min=-0.02, n_max=0.02), history_length=MAP_HISTORY)

        @configclass
        class CriticCfg(ObsGroup):
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            projected_gravity = ObsTerm(func=mdp.projected_gravity)
            velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)
            height_map = ObsTerm(func=height_map, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, history_length=MAP_HISTORY)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.observations.critic = CriticCfg()

        # ---- rewards: the paper's table. Drop every inherited term first, then add theirs.
        for name in list(vars(self.rewards)):
            if not name.startswith("_"):
                setattr(self.rewards, name, None)
        feet = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        R = self.rewards
        R.track_lin_vel = RewTerm(func=track_exp, weight=5.0, params={"std": 0.5, "kind": "lin"})      # exp(-4 e^2)
        R.track_ang_vel = RewTerm(func=track_exp, weight=5.0, params={"std": 0.5, "kind": "ang"})
        R.base_height = RewTerm(func=base_height_exp, weight=1.0, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        R.feet_air_time = RewTerm(func=loco_mdp.feet_air_time_positive_biped, weight=7.0,
                                  params={"command_name": "base_velocity", "sensor_cfg": feet, "threshold": 1.0})
        R.lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
        R.ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        R.orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-6.0)
        R.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        R.action_smoothness = RewTerm(func=action_smoothness, weight=-0.01)
        R.joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)
        R.joint_torques = RewTerm(func=torques_normalised, weight=-0.2)
        R.feet_slide = RewTerm(func=loco_mdp.feet_slide, weight=-0.1,
                               params={"sensor_cfg": feet, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
        R.joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", "waist_.*", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint", "head_.*"])})
        R.termination = RewTerm(func=mdp.is_terminated, weight=-250.0)
        # asset_cfg must go through params: Isaac Lab only resolves body_ids for SceneEntityCfgs it finds
        # there, and a default-argument one crashes on the first step ('slice' object is not subscriptable)
        R.unsafe_stepping = RewTerm(func=unsafe_stepping, weight=1.0, params={"asset_cfg": SceneEntityCfg(
            "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)})
        # not in the paper's table: keeps the ankles off their stops on steep steps
        R.dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*")})


@configclass
class R1StairsEnvCfg_PLAY(R1StairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 6
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None


# ----------------------------------------------------------------- train / play / export
import dataclasses  # noqa: E402
import importlib.metadata as _metadata  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

cfg = R1StairsEnvCfg_PLAY() if (args_cli.play or args_cli.export) else R1StairsEnvCfg()
if args_cli.export:
    cfg.scene.num_envs = 2
elif not args_cli.play:
    cfg.scene.num_envs = args_cli.num_envs
env = ManagerBasedRLEnv(cfg=cfg)
env = RslRlVecEnvWrapper(env)

runner_kw = dict(
    num_steps_per_env=24, max_iterations=args_cli.max_iterations, save_interval=200, experiment_name="r1_stairs",
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                                  activation="elu", actor_obs_normalization=True, critic_obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
                                   num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                   gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
)
if "obs_groups" in {f.name for f in dataclasses.fields(RslRlOnPolicyRunnerCfg)}:
    runner_kw["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
agent_cfg = handle_deprecated_rsl_rl_cfg(RslRlOnPolicyRunnerCfg(**runner_kw), _metadata.version("rsl-rl-lib"))
log_dir = r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_stairs"
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)


def _obs(r):
    return r[0] if isinstance(r, (tuple, list)) else r


def map_rays_in_pelvis_yaw_frame(u):
    """Where each height-map ray actually lands, relative to the pelvis in its yaw frame, measured
    from the live sensor -- so the layout (offset, ordering) is whatever Isaac really used."""
    robot, hits = u.scene["robot"], u.scene.sensors["height_scanner"].data.ray_hits_w[0]
    q = robot.data.root_quat_w[0]
    yaw = torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    d = hits[:, :2] - robot.data.root_pos_w[0, :2]
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], dim=1).tolist()


def export_meta(path):
    """Everything eval_mujoco.py needs to build the SAME observation and apply the SAME action."""
    u = env.unwrapped
    om, am = u.observation_manager, u.action_manager
    term = am.get_term("joint_pos")
    robot = u.scene["robot"]
    jn = list(term._joint_names)
    jids = [robot.joint_names.index(n) for n in jn]
    scale = term._scale
    meta = {
        "dt": u.step_dt, "decimation": cfg.decimation,
        "obs_terms": list(om.active_terms["policy"]),
        "obs_dims": [list(d) for d in om.group_obs_term_dim["policy"]],
        "joint_names": jn,
        "obs_joint_names": list(robot.joint_names),     # joint_pos_rel / joint_vel_rel use articulation order
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "action_scale": float(scale) if not torch.is_tensor(scale) else scale[0].tolist(),
        "action_joint_ids": jids,
        "map_ray_xy": map_rays_in_pelvis_yaw_frame(u),
        "map_history": MAP_HISTORY, "nominal_height": NOMINAL_HEIGHT,
        "command_ranges": {"lin_vel_x": cfg.commands.base_velocity.ranges.lin_vel_x,
                           "lin_vel_y": cfg.commands.base_velocity.ranges.lin_vel_y,
                           "ang_vel_z": cfg.commands.base_velocity.ranges.ang_vel_z},
    }
    Path(path).write_text(json.dumps(meta, indent=1))


if args_cli.export:
    import os as _os  # noqa: E402
    runner.load(args_cli.checkpoint)
    out = _os.path.join(_os.path.dirname(args_cli.checkpoint), "exported")
    runner.export_policy_to_jit(path=out, filename="policy.pt")
    export_meta(_os.path.join(out, "r1_stairs_meta.json"))
    print(f"R1_STAIRS_EXPORT_DONE {out}  (copy policy.pt + r1_stairs_meta.json to stairs-r1/policy/)", flush=True)
elif args_cli.play:
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = _obs(env.get_observations())
    print("[PLAY] R1 stairs policy running", flush=True)
    while simulation_app.is_running():
        with torch.inference_mode():
            obs = _obs(env.step(policy(obs)))
else:
    if args_cli.resume:
        runner.load(args_cli.checkpoint)
        print(f"[TRAIN] resumed from {args_cli.checkpoint}", flush=True)
    print("[TRAIN] R1 stairs PPO (2603.07928-style) starting ...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("R1_STAIRS_TRAIN_DONE", flush=True)
simulation_app.close()
