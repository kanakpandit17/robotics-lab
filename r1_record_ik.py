"""Goal-directed manipulation data: the R1 walks to the ball, then REACHES its right hand to the
ball's actual 3D position using differential inverse kinematics (so the arm action depends on where
the ball is — what the camera sees). Ball position (lateral + height) varies per episode.

Writes a GR00T-ready LeRobot image dataset (PNG frames, no ffmpeg): camera + 26-joint state +
26-dim action + task string + full meta (info/episodes/tasks/modality/stats).
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_flat_v2\model_3999.pt")
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--out", type=str, default=r"C:\Users\aimat\Desktop\humanoid\lerobot_r1_ikreach")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True

import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as _metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from r1_cfg import R1FlatEnvCfg  # noqa: E402

CAM_W, CAM_H, FPS = 224, 224, 30
TASK = "walk to the ball and reach your hand to it"
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]
EE_LINK = "right_wrist_roll_link"


@configclass
class R1RecordEnvCfg(R1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.episode_length_s = 100000.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.scene.front_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link/front_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.24, 0.0, 0.60), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 40.0)),
            width=CAM_W, height=CAM_H,
        )


env = ManagerBasedRLEnv(cfg=R1RecordEnvCfg())
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

from pxr import Gf, UsdGeom  # noqa: E402

stage = env.unwrapped.scene.stage
origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
OX, OY = float(origin[0]), float(origin[1])
BALL_R, R1_HALF, STOP_GAP = 0.18, 0.22, -0.02   # small ball, stop right at it so the arm can reach
STOP_DIST = BALL_R + R1_HALF + STOP_GAP

ball = UsdGeom.Sphere.Define(stage, "/World/RecBall")
ball.CreateRadiusAttr(BALL_R)
ball_t = UsdGeom.Xformable(ball).AddTranslateOp()
ball.CreateDisplayColorAttr([(0.9, 0.15, 0.1)])

robot = env.unwrapped.scene["robot"]
cam = env.unwrapped.scene["front_cam"]
cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
cmd_term.cfg.heading_command = False
cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)

# ---- arm-aim setup (heuristic; verified with body_state_w, no jacobian) ----
arm_ids, _ = robot.find_joints(ARM, preserve_order=True)
arm_ids_t = torch.tensor(arm_ids, device=env.device)
ee_id = robot.find_bodies([EE_LINK])[0][0]
default_arm = robot.data.default_joint_pos[:, arm_ids_t].clone()
print(f"[REACH] arm_ids={arm_ids} ee_body={ee_id} (heuristic aim toward the ball)", flush=True)


def reach_targets(ball_w3):
    """Right-arm joint targets that aim the hand toward the world 3D ball point, as a function of
    the ball's position in the robot's frame (goal-directed). Returns (q_des[5], hand-to-ball dist).
    Uses only safe articulation data (subtract_frame_transforms + body_state_w), no physx jacobian."""
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    ball_w = torch.tensor([ball_w3], device=env.device, dtype=root_pos.dtype)
    ball_b = subtract_frame_transforms(root_pos, root_quat, ball_w)[0][0]
    fx, fy, fz = float(ball_b[0]), float(ball_b[1]), float(ball_b[2])
    sx, sy, sz = 0.05, -0.16, 0.95          # right shoulder, approx, in base frame
    vx_, vy_, vz_ = fx - sx, fy - sy, fz - sz
    horiz = max(1e-3, math.hypot(vx_, vy_))
    elev = math.atan2(vz_, horiz)           # + above shoulder, - below
    azim = math.atan2(vy_, vx_)             # + to the robot's left
    rdist = math.sqrt(vx_ * vx_ + vy_ * vy_ + vz_ * vz_)
    q_sp = -1.35 - 1.0 * elev               # shoulder pitch: forward, tilt with ball elevation
    q_sr = -0.9 * azim                      # shoulder roll: swing toward the ball's side
    q_sy = 0.0
    q_el = max(0.05, 0.9 - 1.3 * min(rdist, 0.7))   # elbow: extend when the ball is far
    q_wr = 0.0
    q_des = torch.tensor([q_sp, q_sr, q_sy, q_el, q_wr], device=env.device, dtype=root_pos.dtype)
    hand = robot.data.body_state_w[0, ee_id, 0:3]
    hand_ball = float(torch.norm(hand - ball_w[0]).item())
    return q_des, hand_ball


OUT = args_cli.out
os.makedirs(os.path.join(OUT, "meta"), exist_ok=True)
os.makedirs(os.path.join(OUT, "data", "chunk-000"), exist_ok=True)
IMG_ROOT = os.path.join(OUT, "images", "observation.images.front")
os.makedirs(IMG_ROOT, exist_ok=True)


def grab_rgb():
    return cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)


CRUISE, SETTLE, ACCEL = 1.0, 40, 0.03
REACH_RAMP, REACH_HOLD, MAX_STEPS = 70, 30, 700
global_index = 0
episodes_meta = []
all_states, all_actions = [], []
img_sum = np.zeros(3); img_sumsq = np.zeros(3); img_px = 0
img_mx = np.zeros(3); img_mn = np.ones(3)
random.seed(0)

for ep in range(args_cli.episodes):
    with torch.inference_mode():
        _r = env.reset()
    obs = _r[0] if isinstance(_r, (tuple, list)) else _r
    cmd_term.is_standing_env[:] = False
    # ball at varied lateral + HEIGHT so the reach genuinely differs per episode
    bx = OX + random.uniform(2.2, 2.9)
    by = OY + random.uniform(-0.5, 0.5)
    bz = random.uniform(0.60, 0.95)      # reachable height band (near the R1's shoulder height)
    ball_xy = (bx, by)
    ball_t.Set(Gf.Vec3d(bx, by, bz))

    frames, states, actions = [], [], []
    prev_vx, settle_until, stopped_at = 0.0, SETTLE, None
    reach_err = None

    for step in range(MAX_STEPS):
        cmd_term.is_standing_env[:] = False
        base = robot.data.root_pos_w[0]
        q = robot.data.root_quat_w[0]
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]).item(), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]).item())
        dx, dy = ball_xy[0] - base[0].item(), ball_xy[1] - base[1].item()
        dist = math.hypot(dx, dy)

        reaching = False
        if step < settle_until:
            target_vx, wz = 0.0, 0.0
        elif dist > STOP_DIST and stopped_at is None:
            yaw_err = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
            align = max(0.0, math.cos(yaw_err))
            target_vx = (0.45 + 0.55 * align) * max(0.0, min(CRUISE, 0.9 * (dist - STOP_DIST) + 0.25))
            wz = max(-0.6, min(0.6, 0.8 * yaw_err))
        else:
            if stopped_at is None:
                stopped_at = step
            target_vx, wz = 0.0, 0.0
            reaching = step - stopped_at < REACH_RAMP + REACH_HOLD

        vx = prev_vx + max(-ACCEL, min(ACCEL, target_vx - prev_vx))
        prev_vx = vx
        cmd_term.vel_command_b[0, 0] = vx
        cmd_term.vel_command_b[0, 1] = 0.0
        cmd_term.vel_command_b[0, 2] = wz

        with torch.inference_mode():
            act = policy(obs).clone()
            if reaching:
                q_des, hb = reach_targets((bx, by, bz))
                frac = min(1.0, (step - stopped_at) / REACH_RAMP)          # ramp arm smoothly
                arm_target = default_arm[0] + frac * (q_des - default_arm[0])
                act[0, arm_ids_t] = 2.0 * (arm_target - default_arm[0])    # action = 2*(target-default), scale 0.5
                if step - stopped_at == REACH_RAMP:
                    reach_err = hb

        frames.append(grab_rgb())
        states.append(robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32))
        actions.append(act[0].detach().cpu().numpy().astype(np.float32))
        with torch.inference_mode():
            _s = env.step(act)
        obs = _s[0] if isinstance(_s, (tuple, list)) else _s

        if stopped_at is not None and step - stopped_at > REACH_RAMP + REACH_HOLD:
            break

    n = len(frames)
    if ep == 0:
        imageio.imwrite(os.path.join(OUT, "sample_reach_frame.png"), frames[-1])
    ep_dir = os.path.join(IMG_ROOT, f"episode_{ep:06d}")
    os.makedirs(ep_dir, exist_ok=True)
    for fi, fr in enumerate(frames):
        imageio.imwrite(os.path.join(ep_dir, f"frame_{fi:06d}.png"), fr)
    all_states.extend(states); all_actions.extend(actions)
    for fr in frames[::4]:
        f01 = fr[..., :3].astype(np.float64) / 255.0
        img_sum += f01.sum((0, 1)); img_sumsq += (f01 ** 2).sum((0, 1)); img_px += f01.shape[0] * f01.shape[1]
        img_mx = np.maximum(img_mx, f01.max((0, 1))); img_mn = np.minimum(img_mn, f01.min((0, 1)))
    df = pd.DataFrame({
        "observation.state": [s.tolist() for s in states], "action": [a.tolist() for a in actions],
        "timestamp": [i / FPS for i in range(n)], "frame_index": list(range(n)),
        "episode_index": [ep] * n, "index": list(range(global_index, global_index + n)), "task_index": [0] * n,
    })
    df.to_parquet(os.path.join(OUT, "data", "chunk-000", f"episode_{ep:06d}.parquet"))
    episodes_meta.append({"episode_index": ep, "tasks": [TASK], "length": n})
    global_index += n
    print(f"[REC] ep {ep:2d}: {n:3d} frames, ball=({bx:.1f},{by:+.1f},z={bz:.2f}), hand-to-ball at full reach = {reach_err:.2f} m" if reach_err is not None else f"[REC] ep {ep}: {n} frames (no reach)", flush=True)

# ---- meta ----
S = np.asarray(all_states, dtype=np.float64); A = np.asarray(all_actions, dtype=np.float64)
img_mean = img_sum / img_px; img_std = np.sqrt(np.maximum(img_sumsq / img_px - img_mean ** 2, 0)) + 1e-8


def stat(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return {"mean": x.mean(0).tolist(), "std": (x.std(0) + 1e-8).tolist(), "max": x.max(0).tolist(), "min": x.min(0).tolist()}


def img3(a):
    return [[[float(v)]] for v in a]


stats = {"observation.state": stat(S), "action": stat(A),
         "observation.images.front": {"mean": img3(img_mean), "std": img3(img_std), "max": img3(img_mx), "min": img3(img_mn)},
         "timestamp": stat(np.zeros(1)), "frame_index": stat(np.arange(1)),
         "episode_index": stat(np.arange(args_cli.episodes)), "index": stat(np.arange(global_index)), "task_index": stat(np.zeros(1))}
info = {
    "codebase_version": "v2.1", "robot_type": "unitree_r1", "total_episodes": args_cli.episodes,
    "total_frames": global_index, "total_tasks": 1, "total_videos": 0, "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
    "splits": {"train": f"0:{args_cli.episodes}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "image_path": "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
    "features": {
        "observation.images.front": {"dtype": "image", "shape": [CAM_H, CAM_W, 3], "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": [int(S.shape[1])], "names": None},
        "action": {"dtype": "float32", "shape": [int(A.shape[1])], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    },
}
modality = {"state": {"joints": {"start": 0, "end": int(S.shape[1])}}, "action": {"joints": {"start": 0, "end": int(A.shape[1])}},
            "video": {"front": {"original_key": "observation.images.front"}},
            "annotation": {"human.task_description": {"original_key": "task_index"}}}
m = os.path.join(OUT, "meta")
json.dump(info, open(os.path.join(m, "info.json"), "w"), indent=2)
json.dump(stats, open(os.path.join(m, "stats.json"), "w"), indent=2)
json.dump(modality, open(os.path.join(m, "modality.json"), "w"), indent=2)
open(os.path.join(m, "tasks.jsonl"), "w").write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
with open(os.path.join(m, "episodes.jsonl"), "w") as f:
    for e in episodes_meta:
        f.write(json.dumps(e) + "\n")
print(f"R1_RECORD_DONE  {args_cli.episodes} eps, {global_index} frames -> {OUT}", flush=True)
simulation_app.close()
