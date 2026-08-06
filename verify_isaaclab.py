"""Minimal end-to-end verification for Isaac Sim + Isaac Lab on GPU.

Boots the simulation app headless, creates a physics scene on cuda:0,
steps it 100 times, then exits. Proves the whole stack works on Blackwell.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab install verification")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.sim import SimulationCfg, SimulationContext

sim_cfg = SimulationCfg(dt=0.01, device="cuda:0")
sim = SimulationContext(sim_cfg)
sim.reset()

print(f"[VERIFY] sim device      = {sim.device}")
print(f"[VERIFY] torch cuda dev  = {torch.cuda.get_device_name(0)}")

for _ in range(100):
    sim.step()

print("[VERIFY] stepped 100 physics steps on GPU")
print("[VERIFY] SUCCESS")

simulation_app.close()
