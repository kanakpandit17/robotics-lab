"""R1 locomotion (WALKING) via RL — adapts Isaac Lab's proven G1 flat velocity task for the
Unitree R1. PPO learns to track commanded base velocities while staying upright (i.e. to walk).

This is the hard, foundational legged-robot RL milestone. Bipedal walking needs many iterations
(~1000-3000) and may need reward/gain tuning; the first run may only produce a rough gait.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=3072)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--play", action="store_true", default=False)
parser.add_argument("--eval", action="store_true", default=False, help="headless: measure upright fraction + velocity tracking")
parser.add_argument("--export", action="store_true", default=False, help="headless: export --checkpoint as TorchScript policy.pt")
parser.add_argument("--resume", action="store_true", default=False, help="load --checkpoint and continue training")
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_flat\model_999.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not args_cli.play

import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

if args_cli.eval or args_cli.export:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.envs import mdp  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

R1_USD = r"C:\Users\aimat\Desktop\humanoid\imported_cad\unitree_r1.usd"

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
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_shoulder_pitch_joint": 0.20,
            ".*_elbow_joint": 0.60,
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
class R1FlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # swap G1 -> R1
        self.scene.robot = R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # torso_link -> waist_yaw_link (base body for R1)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # R1 has no dexterous fingers -> drop that reward
        self.rewards.joint_deviation_fingers = None
        # torso_joint -> waist joints
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint"])
        # R1 arm joints (elbow_joint / wrist_roll_joint, not G1's elbow_pitch/roll)
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]
        )
        # ---- ANTI-CRAWL: make falling non-viable so PPO must actually walk upright ----
        # end the episode the instant the torso drops or tips over (a belly-crawler can't farm reward)
        self.terminations.root_too_low = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},  # ~46 deg tilt
        )
        # reward staying upright and at standing height
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={"target_height": 0.72, "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class R1FlatEnvCfg_PLAY(R1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# --------------------------------------------------------------------------- train / play
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

cfg = R1FlatEnvCfg_PLAY() if (args_cli.play or args_cli.eval or args_cli.export) else R1FlatEnvCfg()
if args_cli.eval:
    cfg.scene.num_envs = 64
elif args_cli.export:
    cfg.scene.num_envs = 2
elif not args_cli.play:
    cfg.scene.num_envs = args_cli.num_envs
env = ManagerBasedRLEnv(cfg=cfg)
env = RslRlVecEnvWrapper(env)

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=args_cli.max_iterations,
    save_interval=50,
    experiment_name="r1_flat_v2",
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128], activation="elu", actor_obs_normalization=False, critic_obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.008, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
)
log_dir = r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_flat_v2"
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

if args_cli.export:
    import os as _os  # noqa: E402
    runner.load(args_cli.checkpoint)
    out = _os.path.join(_os.path.dirname(args_cli.checkpoint), "exported")
    runner.export_policy_to_jit(path=out, filename="policy.pt")
    print(f"R1_EXPORT_DONE {_os.path.join(out, 'policy.pt')}", flush=True)
elif args_cli.eval:
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    _r = env.get_observations()
    obs = _r[0] if isinstance(_r, (tuple, list)) else _r
    print(f"[EVAL] {args_cli.checkpoint}  ({cfg.scene.num_envs} envs, commanded vx=1.0 m/s)", flush=True)
    for i in range(600):
        with torch.inference_mode():
            _s = env.step(policy(obs))
            obs = _s[0] if isinstance(_s, (tuple, list)) else _s
        base_z = robot.data.root_pos_w[:, 2] - env.unwrapped.scene.env_origins[:, 2]
        up = robot.data.projected_gravity_b[:, 2]          # ~ -1 upright
        vx = robot.data.root_lin_vel_b[:, 0]               # forward speed in body frame
        upright_frac = ((up < -0.7) & (base_z > 0.5)).float().mean().item()
        if i % 50 == 0 or i == 599:
            print(f"[EVAL] step={i:3d}  upright={upright_frac*100:5.1f}%  mean_vx={vx.mean().item():+.2f}  mean_base_z={base_z.mean().item():.2f}", flush=True)
    print("R1_EVAL_DONE", flush=True)
elif args_cli.play:
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    _r = env.get_observations()
    obs = _r[0] if isinstance(_r, (tuple, list)) else _r
    print("[PLAY] R1 walking policy running", flush=True)
    while simulation_app.is_running():
        with torch.inference_mode():
            _s = env.step(policy(obs))
            obs = _s[0] if isinstance(_s, (tuple, list)) else _s
else:
    if args_cli.resume:
        runner.load(args_cli.checkpoint)
        print(f"[TRAIN] resumed from {args_cli.checkpoint}", flush=True)
    print("[TRAIN] R1 walking PPO starting ...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("R1_WALK_TRAIN_DONE", flush=True)
simulation_app.close()
