"""Isaac Sim side of the LARA sim<->real bridge (runs on Windows), mirroring
maira_sim_bridge.py's own structure for the same purpose on MAiRA.

Loads the LARA articulation (imported_cad/lara8.usd, 6 DOF, verified against the real robot's
own joint names -- joint1..joint6, matching exactly) and exchanges joint angles with the real
robot through a shared JSON file (written by lara_real_reader.py, running in WSL against the
neurapy SDK):

  lara_bridge.json = {"real_joints":[...6...], "at": <unix time>}

Modes:
  --mode real2sim : sim LARA MIRRORS the real robot (read real_joints, drive the sim).  [SAFE]
  --mode selftest : no bridge; drive the sim joints with a dummy sweep to verify articulation.

Deliberately no sim2real mode here, unlike maira_sim_bridge.py: nothing in this project has
asked for commanding the real LARA from the simulator, and that is a materially different,
higher-stakes feature (it moves the physical robot) that should be a separate, explicit
decision when it is actually wanted -- not something that comes along for free because the
mirror-only half of the same file exists.
"""
import argparse
import json
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["real2sim", "selftest"], default="selftest")
parser.add_argument("--bridge", type=str, default=r"C:\Users\aimat\Desktop\humanoid\lara_bridge.json")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--max-age", type=float, default=1.0,
                    help="seconds -- freeze the sim rather than follow a bridge reading "
                         "older than this (the WSL reader has stopped or the robot is "
                         "unreachable, and holding the last real pose is safer than either "
                         "snapping to a stale one or silently doing nothing)")
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

LARA = r"C:\Users\aimat\Desktop\humanoid\imported_cad\lara8.usd"

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
sim.set_camera_view(eye=(1.6, 1.6, 1.2), target=(0.0, 0.0, 0.5))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1500.0))

lara_cfg = ArticulationCfg(
    prim_path="/World/Lara",
    spawn=sim_utils.UsdFileCfg(usd_path=LARA),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=4000.0, damping=200.0)},
)
lara = Articulation(lara_cfg)
sim.reset()

jn = lara.joint_names
NJ = len(jn)
print(f"[LARA-SIM] articulation ready: {NJ} joints -> {jn}", flush=True)
if jn != ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]:
    print("[LARA-SIM] WARNING: joint names/order do not match what was verified against the "
         "real robot -- real_joints[i] may not map to joint i. Check before trusting the mirror.",
         flush=True)
default_q = lara.data.default_joint_pos.clone()


def read_bridge():
    try:
        with open(args_cli.bridge) as f:
            return json.load(f)
    except Exception:
        return {}


def set_sim_joints(angles):
    q = default_q.clone()
    n = min(len(angles), NJ)
    for i in range(n):
        q[0, i] = float(angles[i])
    lara.set_joint_position_target(q)


print(f"[LARA-SIM] mode={args_cli.mode} bridge={args_cli.bridge}", flush=True)
step = 0
last_print = 0.0
stale_logged = False
while simulation_app.is_running():
    t = step / 120.0
    if args_cli.mode == "selftest":
        # Gentle sweep on the first three joints, to confirm the sim arm moves at all --
        # matches maira_sim_bridge.py's own selftest shape.
        ang = [0.0] * NJ
        for i in range(min(NJ, 3)):
            ang[i] = 0.4 * math.sin(0.5 * t + i * 0.6)
        set_sim_joints(ang)
    elif args_cli.mode == "real2sim":
        b = read_bridge()
        rj = b.get("real_joints")
        at = b.get("at")
        age = (time.time() - at) if at else None
        if rj and age is not None and age <= args_cli.max_age:
            set_sim_joints(rj)
            stale_logged = False
        elif not stale_logged:
            stale_logged = True
            reason = "no bridge data yet" if not rj else f"bridge reading is {age:.1f}s old"
            print(f"[LARA-SIM] {reason} -- holding last known pose, not following. "
                 f"Is lara_real_reader.py running in WSL?", flush=True)

    lara.write_data_to_sim()
    sim.step()
    lara.update(1.0 / 120.0)
    step += 1
    if time.time() - last_print > 2.0:
        q_now = [round(float(lara.data.joint_pos[0, i].item()), 3) for i in range(NJ)]
        print(f"[LARA-SIM] t={t:5.1f}s joints={q_now}", flush=True)
        last_print = time.time()

simulation_app.close()
