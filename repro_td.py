"""Fast repro: launch Kit headless, then import tensordict + rsl_rl (the crash path)."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app = AppLauncher(args_cli).app
print("[REPRO] Kit launched; importing tensordict ...", flush=True)

import tensordict

print(f"[REPRO] tensordict OK {tensordict.__version__}", flush=True)

import rsl_rl  # noqa: F401

print("[REPRO] rsl_rl OK", flush=True)

app.close()
print("[REPRO] SUCCESS", flush=True)
