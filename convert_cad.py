"""Headless STEP -> USD conversion via Isaac Sim's HOOPS Core CAD converter."""
import argparse
import asyncio
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension

for ext in [
    "omni.kit.converter.common",
    "omni.kit.converter.hoops_core",
    "omni.kit.converter.hoops",
]:
    print(f"[ext] {ext}: {'enabled' if enable_extension(ext) else 'FAILED'}", flush=True)

for _ in range(30):
    simulation_app.update()

from omni.kit.converter.hoops_core import get_instance as get_hoops_converter

OUT = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
os.makedirs(OUT, exist_ok=True)
DL = r"C:\Users\aimat\Downloads"
JOBS = [
    (os.path.join(DL, "KUKA Assembly (1).step"), os.path.join(OUT, "kuka_assembly.usd")),
    (
        os.path.join(DL, "Arena2036 Shopfloor CAD Models (1)", "Arena2036 Shopfloor CAD Models", "Final wihout walls and roof.step"),
        os.path.join(OUT, "arena_no_walls.usd"),
    ),
    (
        os.path.join(DL, "Arena2036 Shopfloor CAD Models (1)", "Arena2036 Shopfloor CAD Models", "Final.step"),
        os.path.join(OUT, "arena_full.usd"),
    ),
]


async def convert_one(hoops, inp, outp):
    print(f"[CAD] converting: {os.path.basename(inp)}", flush=True)
    if not os.path.isfile(inp):
        print(f"[CAD] MISSING INPUT: {inp}", flush=True)
        return
    result_path, status = await hoops.create_converter_task(inp, outp, {})
    if getattr(status, "error_code", -1) == 0:
        mb = os.path.getsize(outp) / 1e6 if os.path.isfile(outp) else 0.0
        print(f"[CAD] OK -> {outp}  ({mb:.1f} MB)", flush=True)
    else:
        print(f"[CAD] FAILED: {os.path.basename(inp)} :: code={getattr(status,'error_code','?')} msg={getattr(status,'error_msg','?')}", flush=True)


async def run_all():
    hoops = get_hoops_converter()
    print(f"[CAD] hoops instance: {hoops}", flush=True)
    for inp, outp in JOBS:
        await convert_one(hoops, inp, outp)


asyncio.get_event_loop().run_until_complete(run_all())
print("CAD_CONVERT_DONE", flush=True)
simulation_app.close()
