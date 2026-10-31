"""Run an Isaac-trained R1 stair policy on ARENA2036 staircase 3 in MuJoCo (sim-to-sim).

    mjpython stairs-r1/eval_mujoco.py --policy stairs-r1/policy           # viewer (macOS: mjpython)
    python   stairs-r1/eval_mujoco.py --policy stairs-r1/policy --headless
    python   stairs-r1/eval_mujoco.py --dry-run --headless                # no policy: pipeline check

--policy is a folder holding policy.pt and r1_stairs_meta.json from `r1_stairs_train.py --export`.
The meta file pins everything that must match Isaac: observation terms and order, joint order for
observations and for actions, default pose, action scale, where each height-map ray lands relative
to the pelvis, and the 5-frame history. Nothing here re-derives those from the MuJoCo model.

--dry-run builds a stand-in meta from the MuJoCo model (MuJoCo joint order, the same 29 x 21 grid
1.4 x 1.0 m at 5 cm, centred 0.2 m ahead) and sends zero actions. It checks the plumbing -- obs
size, the height map seeing the stairs -- not a policy.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

import build_r1_mjcf
import staircase3

HERE = Path(__file__).resolve().parent
GROUP0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # world geoms only; robot is in groups 1/3
DOWN = np.array([0.0, 0.0, -1.0])


def load_scene():
    if not (HERE / "r1.xml").exists():
        build_r1_mjcf.main()
    path = HERE / "scene_staircase3.xml"
    path.write_text(staircase3.scene_xml())
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    d.qpos[0:2] = staircase3.START
    mujoco.mj_forward(m, d)
    return m, d


def dry_meta(m):
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
    key = m.key("stand")
    default = [float(key.qpos[m.jnt_qposadr[m.joint(n).id]]) for n in names]
    xs = np.arange(-0.7, 0.7 + 1e-9, 0.05)
    ys = np.arange(-0.5, 0.5 + 1e-9, 0.05)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    rays = np.stack([gx.ravel() + 0.2, gy.ravel()], axis=1)
    nr = len(rays)
    return {"dt": 0.02, "decimation": 4, "joint_names": names, "obs_joint_names": names, "default_joint_pos": default,
            "action_scale": 0.5, "action_joint_ids": list(range(len(names))), "map_ray_xy": rays.tolist(),
            "map_history": 5, "nominal_height": 0.72,
            "obs_terms": ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "height_map"],
            "obs_dims": [[3], [3], [3], [len(names)], [len(names)], [len(names)], [5 * nr]]}


class Bridge:
    """MuJoCo state <-> the policy's observation / action, as Isaac defined them."""

    def __init__(self, m, d, meta):
        self.m, self.d, self.meta = m, d, meta
        jid = lambda n: m.joint(n).id
        self.obs_qadr = np.array([m.jnt_qposadr[jid(n)] for n in meta["obs_joint_names"]])
        self.obs_vadr = np.array([m.jnt_dofadr[jid(n)] for n in meta["obs_joint_names"]])
        self.default_obs = np.array(meta["default_joint_pos"])
        name_to_act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]): i for i in range(m.nu)}
        self.act_idx = np.array([name_to_act[n] for n in meta["joint_names"]])
        default_by_name = dict(zip(meta["obs_joint_names"], meta["default_joint_pos"]))
        self.default_act = np.array([default_by_name[n] for n in meta["joint_names"]])
        self.scale = np.asarray(meta["action_scale"], dtype=float)
        self.rays = np.asarray(meta["map_ray_xy"])
        self.hist = deque(maxlen=meta["map_history"])
        self.last_action = np.zeros(len(meta["joint_names"]))
        self.cmd = np.zeros(3)
        self.pelvis = m.body("pelvis_link").id

    def height_map(self):
        d = self.d
        root = d.qpos[:3]
        R = d.xmat[self.pelvis].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy = root[:2] + self.rays @ np.array([[c, s], [-s, c]])
        gid = np.zeros(1, dtype=np.int32)
        hz = np.empty(len(xy))
        for i, (x, y) in enumerate(xy):
            dist = mujoco.mj_ray(self.m, d, np.array([x, y, 20.0]), DOWN, GROUP0, 1, -1, gid)
            hz[i] = 20.0 - dist if dist >= 0 else -1e3
        return np.clip(root[2] - hz - self.meta["nominal_height"], -1.0, 1.0)

    def observe(self):
        d = self.d
        R = d.xmat[self.pelvis].reshape(3, 3)
        hm = self.height_map()
        if not self.hist:
            self.hist.extend([hm] * self.hist.maxlen)    # Isaac fills the history on reset
        else:
            self.hist.append(hm)
        parts = {
            "base_ang_vel": d.qvel[3:6],                  # free joint: body-frame angular velocity
            "projected_gravity": R.T @ DOWN,
            "velocity_commands": self.cmd,
            "joint_pos": d.qpos[self.obs_qadr] - self.default_obs,
            "joint_vel": d.qvel[self.obs_vadr],
            "actions": self.last_action,
            "height_map": np.concatenate(list(self.hist)),   # oldest -> newest
        }
        obs = []
        for name, dim in zip(self.meta["obs_terms"], self.meta["obs_dims"]):
            if name not in parts:
                raise SystemExit(f"policy expects obs term {name!r}, which this bridge doesn't build")
            v = np.asarray(parts[name], dtype=np.float32).ravel()
            if v.size != int(np.prod(dim)):
                raise SystemExit(f"obs term {name}: built {v.size}, policy expects {dim}")
            obs.append(v)
        return np.concatenate(obs)

    def act(self, a):
        self.last_action = np.asarray(a, dtype=float)
        self.d.ctrl[self.act_idx] = self.default_act + self.scale * self.last_action


def load_policy(folder):
    import torch
    pol = torch.jit.load(str(Path(folder) / "policy.pt"), map_location="cpu").eval()

    def run(obs):
        with torch.inference_mode():
            return pol(torch.from_numpy(obs)[None]).numpy()[0]
    return run


def progress(m, d):
    """Highest step the pelvis is over, and whether it is upright."""
    x, z = d.qpos[0], d.qpos[2]
    up = d.xmat[m.body("pelvis_link").id].reshape(3, 3)[2, 2]
    step = int(np.clip(x // staircase3.GOING + 1, 0, len(staircase3.STEP_TOPS))) if x > 0 else 0
    on_top = x > staircase3.STAIR_LEN + 0.2 and z > staircase3.TOP_Z + 0.5
    return step, up, on_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, help="folder with policy.pt + r1_stairs_meta.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--vx", type=float, default=0.5, help="forward command, m/s (paper tests 0.7)")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    if not args.dry_run and not args.policy:
        ap.error("--policy FOLDER or --dry-run")

    m, d = load_scene()
    if args.dry_run:
        meta, policy = dry_meta(m), None
    else:
        meta = json.loads((Path(args.policy) / "r1_stairs_meta.json").read_text())
        policy = load_policy(args.policy)
    decim = int(meta.get("decimation", 4))
    if abs(m.opt.timestep * decim - meta["dt"]) > 1e-6:
        raise SystemExit(f"timestep mismatch: MuJoCo {m.opt.timestep} x {decim} != Isaac step_dt {meta['dt']}")
    br = Bridge(m, d, meta)
    br.cmd[:] = [args.vx, 0.0, 0.0]
    obs = br.observe()
    print(f"obs size {obs.size} ({', '.join(f'{n}={int(np.prod(s))}' for n, s in zip(meta['obs_terms'], meta['obs_dims']))})")
    hm = br.hist[-1].reshape(-1)
    print(f"height map now: min {hm.min():+.3f}  max {hm.max():+.3f}  (negative = ground higher than under the pelvis)")

    best, fell, t_top = 0, False, None

    def policy_step():
        nonlocal obs, best, fell, t_top
        a = np.zeros(len(meta["joint_names"])) if policy is None else policy(obs)
        br.act(a)
        for _ in range(decim):
            mujoco.mj_step(m, d)
        obs = br.observe()
        step, up, on_top = progress(m, d)
        best = max(best, step)
        if up < 0.5 or d.qpos[2] < 0.35 + (staircase3.STEP_TOPS[step - 1] if step else 0):
            fell = True
        if on_top and t_top is None:
            t_top = d.time

    n = int(args.seconds / meta["dt"])
    if args.headless:
        for _ in range(n):
            policy_step()
            if fell or t_top:
                break
    else:
        from mujoco import viewer as mj_viewer   # a bare `import mujoco.viewer` here would shadow `mujoco` in main()
        with mj_viewer.launch_passive(m, d) as v:
            v.cam.lookat[:] = [1.6, 0.0, 1.2]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 6.0, -35, -18
            for _ in range(n):
                if not v.is_running():
                    break
                t0 = time.time()
                policy_step()
                v.cam.lookat[:] = [d.qpos[0], d.qpos[1], d.qpos[2]]
                v.sync()
                time.sleep(max(0.0, meta["dt"] - (time.time() - t0)))
    total = len(staircase3.STEP_TOPS)
    verdict = f"REACHED THE LANDING in {t_top:.1f} s" if t_top else ("FELL" if fell else "timed out")
    print(f"result: {verdict}; highest step {best}/{total}; t={d.time:.1f} s; pelvis x={d.qpos[0]:+.2f} z={d.qpos[2]:.2f}")


if __name__ == "__main__":
    main()
