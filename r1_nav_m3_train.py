"""Milestone 3 — obstacle-avoiding navigation for the Galaxea R1 (Isaac Lab + rsl_rl).

Extends M2: a static pillar (stand-in for the KUKA) sits at each env centre. R1 must reach the
goal WITHOUT hitting it. Observation gains the obstacle-relative vector; reward adds a proximity
penalty and a collision termination. Same 2-D action (steer, wheel).
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=300)
parser.add_argument("--play", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_nav_m3\model_299.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not args_cli.play

import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

R1_USD = r"C:\Users\aimat\Desktop\humanoid\imported_cad\r1.usd"
MAX_WHEEL_VEL = 6.0
GOAL_TOL = 0.6
GOAL_RANGE = 3.0
OBST_RADIUS = 0.7      # obstacle half-extent (collision radius)
OBST_SAFE = 1.2        # start penalising inside this radius
OBST_HIT = 0.95        # closer than this = collision (terminate)

R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R1_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=8),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
    actuators={
        "wheels": ImplicitActuatorCfg(joint_names_expr=["wheel_motor_joint.*"], stiffness=0.0, damping=800.0),
        "steer": ImplicitActuatorCfg(joint_names_expr=["steer_motor_joint.*"], stiffness=800.0, damping=80.0),
        "body": ImplicitActuatorCfg(joint_names_expr=[".*arm.*", "torso.*", ".*gripper.*"], stiffness=1500.0, damping=120.0),
    },
)


@configclass
class R1NavM3Cfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 2
    observation_space = 10
    state_space = 0
    sim = SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    scene = InteractiveSceneCfg(num_envs=512, env_spacing=8.0, replicate_physics=True)
    robot: ArticulationCfg = R1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class R1NavM3Env(DirectRLEnv):
    cfg: R1NavM3Cfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._wheel_ids, _ = self.robot.find_joints("wheel_motor_joint.*")
        self._steer_ids, _ = self.robot.find_joints("steer_motor_joint.*")
        self.goal_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        print(f"[ENV] wheels={self._wheel_ids} steer={self._steer_ids} dofs={self.robot.num_joints}", flush=True)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        g = sim_utils.GroundPlaneCfg()
        g.func("/World/ground", g)
        # static obstacle pillar at each env centre (cloned with the env)
        obst = sim_utils.CuboidCfg(
            size=(2 * OBST_RADIUS, 2 * OBST_RADIUS, 2.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.1)),
        )
        obst.func("/World/envs/env_0/Obstacle", obst, translation=(0.0, 0.0, 1.1))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=1500.0)
        light.func("/World/Light", light)
        self.goal_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/World/GoalMarker",
                markers={"g": sim_utils.SphereCfg(radius=0.3, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)))},
            )
        )

    def _pre_physics_step(self, actions):
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        steer = self.actions[:, 0] * (math.pi / 2.0)
        wheel = self.actions[:, 1] * MAX_WHEEL_VEL
        self.robot.set_joint_position_target(steer.unsqueeze(-1).expand(-1, len(self._steer_ids)), joint_ids=self._steer_ids)
        self.robot.set_joint_velocity_target(wheel.unsqueeze(-1).expand(-1, len(self._wheel_ids)), joint_ids=self._wheel_ids)

    def _local_xy(self):
        return self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]

    def _yaw(self):
        q = self.robot.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _get_observations(self):
        pos = self._local_xy()
        vel = self.robot.data.root_lin_vel_w[:, :2]
        yaw = self._yaw()
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        gvec = self.goal_pos - pos
        gx = cy * gvec[:, 0] + sy * gvec[:, 1]
        gy = -sy * gvec[:, 0] + cy * gvec[:, 1]
        # obstacle is at local (0,0)
        ox = cy * (-pos[:, 0]) + sy * (-pos[:, 1])
        oy = -sy * (-pos[:, 0]) + cy * (-pos[:, 1])
        dist = torch.norm(gvec, dim=1)
        steer = self.robot.data.joint_pos[:, self._steer_ids[0]]
        obs = torch.stack([gx, gy, vel[:, 0], vel[:, 1], torch.cos(yaw), torch.sin(yaw), steer, dist, ox, oy], dim=-1)
        return {"policy": obs}

    def _get_rewards(self):
        pos = self._local_xy()
        dist = torch.norm(self.goal_pos - pos, dim=1)
        progress = self.prev_dist - dist
        self.prev_dist = dist
        reached = (dist < GOAL_TOL).float()
        obst_dist = torch.norm(pos, dim=1)
        near = torch.clamp(OBST_SAFE - obst_dist, min=0.0)  # >0 when inside safe radius
        hit = (obst_dist < OBST_HIT).float()
        up_z = 1 - 2 * (self.robot.data.root_quat_w[:, 1] ** 2 + self.robot.data.root_quat_w[:, 2] ** 2)
        tipped = (up_z < 0.5).float()
        act_pen = torch.sum(self.actions**2, dim=1)
        return 3.0 * progress + 12.0 * reached - 0.02 - 0.002 * act_pen - 5.0 * tipped - 1.0 * near - 3.0 * hit

    def _get_dones(self):
        pos = self._local_xy()
        dist = torch.norm(self.goal_pos - pos, dim=1)
        obst_dist = torch.norm(pos, dim=1)
        up_z = 1 - 2 * (self.robot.data.root_quat_w[:, 1] ** 2 + self.robot.data.root_quat_w[:, 2] ** 2)
        terminated = (dist < GOAL_TOL) | (up_z < 0.4) | (obst_dist < OBST_HIT)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        # start on a ring around the obstacle (radius 2.0-3.0), random angle
        ang = torch.rand(n, device=self.device) * 2 * math.pi
        rad = 2.0 + torch.rand(n, device=self.device)
        root[:, 0] += rad * torch.cos(ang)
        root[:, 1] += rad * torch.sin(ang)
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)
        jp = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
        # goal at a random angle on a ring (>= ~2 m from the obstacle centre, so it's never inside it)
        gang = torch.rand(n, device=self.device) * 2 * math.pi
        grad = 2.0 + torch.rand(n, device=self.device) * 1.5
        self.goal_pos[env_ids] = torch.stack([grad * torch.cos(gang), grad * torch.sin(gang)], dim=-1)
        pos = root[:, :2] - self.scene.env_origins[env_ids][:, :2]
        self.prev_dist[env_ids] = torch.norm(self.goal_pos[env_ids] - pos, dim=1)
        gw = torch.zeros(self.num_envs, 3, device=self.device)
        gw[:, :2] = self.goal_pos + self.scene.env_origins[:, :2]
        gw[:, 2] = 0.4
        self.goal_marker.visualize(translations=gw)


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

cfg = R1NavM3Cfg()
cfg.scene.num_envs = 16 if args_cli.play else args_cli.num_envs
env = R1NavM3Env(cfg)
env = RslRlVecEnvWrapper(env)

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=args_cli.max_iterations,
    save_interval=50,
    experiment_name="r1_nav_m3",
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64], activation="elu", actor_obs_normalization=False, critic_obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.006, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
)

log_dir = r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_nav_m3"
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

if args_cli.play:
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    _r = env.get_observations()
    obs = _r[0] if isinstance(_r, (tuple, list)) else _r
    print("[PLAY] running M3 policy — R1 goes AROUND the orange obstacle", flush=True)
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            _s = env.step(actions)
            obs = _s[0] if isinstance(_s, (tuple, list)) else _s
else:
    print("[TRAIN] starting M3 obstacle-avoidance PPO ...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("R1_NAV_M3_TRAIN_DONE", flush=True)
simulation_app.close()
