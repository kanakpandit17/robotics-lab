"""Open a USD file as the stage in the Isaac Sim GUI and keep the window running."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, required=True, help="Path to the USD file to open")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd

print(f"[VIEW] opening stage: {args_cli.usd}", flush=True)
omni.usd.get_context().open_stage(args_cli.usd)

# let it load, then frame the scene
for _ in range(120):
    simulation_app.update()
print("[VIEW] stage loaded — window is live", flush=True)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
