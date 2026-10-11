"""Isaac Sim side of the MAiRA sim<->real bridge (runs on Windows).

Loads the MAiRA articulation and exchanges 7 joint angles with the real robot through a shared
JSON file (written/read by the neurapy side in WSL):

  maira_bridge.json = {"real_joints":[...7...], "sim_joints":[...7...], "mode":"real2sim"|"sim2real"}

Modes:
  --mode real2sim : sim MAiRA MIRRORS the real robot (read real_joints, drive the sim).  [SAFE]
  --mode sim2real : you pose the sim MAiRA (here: a slow demo sweep) and it publishes sim_joints
                    for the real robot to follow.
  --mode selftest : no bridge; drive the sim joints with a dummy sweep to verify articulation.
"""
import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["real2sim", "sim2real", "selftest"], default="selftest")
parser.add_argument("--bridge", type=str, default=r"C:\Users\aimat\Desktop\humanoid\maira_bridge.json")
parser.add_argument("--headless", action="store_true", default=False)
args_cli, _ = parser.parse_known_args()

# h5py stub: Isaac Sim's bundled hdf5.dll shadows h5py -> DLL error loading isaaclab_tasks
import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

MAIRA = r"C:\Users\aimat\Desktop\humanoid\imported_cad\maira7M.usd"

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
sim.set_camera_view(eye=(2.2, 2.2, 1.6), target=(0.0, 0.0, 0.6))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1500.0))

maira_cfg = ArticulationCfg(
    prim_path="/World/Maira",
    spawn=sim_utils.UsdFileCfg(usd_path=MAIRA),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=4000.0, damping=200.0)},
)
maira = Articulation(maira_cfg)
sim.reset()

jn = maira.joint_names
NJ = len(jn)
print(f"[MAIRA-SIM] articulation ready: {NJ} joints -> {jn}", flush=True)
default_q = maira.data.default_joint_pos.clone()


def read_bridge():
    try:
        with open(args_cli.bridge) as f:
            return json.load(f)
    except Exception:
        return {}


def write_bridge(d):
    # direct write (os.replace fails with 'Access denied' when the WSL side has the file open)
    try:
        with open(args_cli.bridge, "w") as f:
            json.dump(d, f)
            f.flush()
    except (PermissionError, OSError):
        pass   # transient contention with the reader; try again next tick


def set_sim_joints(angles):
    q = default_q.clone()
    n = min(len(angles), NJ)
    for i in range(n):
        q[0, i] = float(angles[i])
    maira.set_joint_position_target(q)


print(f"[MAIRA-SIM] mode={args_cli.mode}  bridge={args_cli.bridge}", flush=True)
step = 0
last_print = 0.0
sim2real_base = None   # the real robot's pose at sim2real start (so we move relative to it, no jump)
while simulation_app.is_running():
    t = step / 120.0
    if args_cli.mode == "selftest":
        # gentle sweep on the first few joints to confirm the sim arm moves
        ang = [0.0] * NJ
        for i in range(min(NJ, 7)):
            ang[i] = 0.4 * math.sin(0.5 * t + i * 0.6)
        set_sim_joints(ang)
    elif args_cli.mode == "real2sim":
        b = read_bridge()
        rj = b.get("real_joints")
        if rj:
            set_sim_joints(rj)
    elif args_cli.mode == "sim2real":
        # start FROM the real robot's current pose (read once from the bridge) so the real arm
        # never has to make a big jump, then make a SMALL slow move on joint 1 only.
        if sim2real_base is None:
            b = read_bridge()
            rj = b.get("real_joints")
            sim2real_base = list(rj) if rj else [float(default_q[0, i].item()) for i in range(NJ)]
            src = "real pose" if rj else "DEFAULT (no real pose in bridge!)"
            print(f"[MAIRA-SIM] sim2real base ({src}): {[round(x, 3) for x in sim2real_base]}", flush=True)
        ang = list(sim2real_base)
        ang[0] += 0.50 * math.sin(0.35 * t)          # joint 1 (base yaw): +/-0.50 rad (~29 deg) - clearly visible
        if NJ > 3:
            ang[3] += 0.35 * math.sin(0.35 * t + 1.2)  # joint 4: +/-0.35 rad, offset -> a second obvious motion
        set_sim_joints(ang)
        write_bridge({"sim_joints": ang, "real_joints": sim2real_base, "mode": "sim2real"})

    maira.write_data_to_sim()
    sim.step()
    maira.update(1.0 / 120.0)
    step += 1
    if time.time() - last_print > 2.0:
        q_now = [round(float(maira.data.joint_pos[0, i].item()), 3) for i in range(min(NJ, 7))]
        print(f"[MAIRA-SIM] t={t:5.1f}s joints(1-7)={q_now}", flush=True)
        last_print = time.time()

simulation_app.close()
