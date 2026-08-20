"""List every named sub-prim of a CAD asset with its world bounding box, so you can find things
like a door gap or a named part without clicking through the Stage tree in the GUI.

Plain Isaac Sim app (no Isaac Lab exts -> no h5py crash), same pattern as compose_scene.py.

    python inspect_kuka_hierarchy.py                          # kuka_assembly.usd, all prims
    python inspect_kuka_hierarchy.py --filter door             # only names matching "door"
    python inspect_kuka_hierarchy.py --asset maira7M.usd --filter ""
"""
import argparse
import os
import re

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--asset", default="kuka_assembly.usd", help="filename under imported_cad/")
parser.add_argument("--filter", default="door|gate|gap|open|entry|exit|safety|fence|cage",
                     help="case-insensitive regex on prim name; pass \"\" to list everything")
parser.add_argument("--min-size", type=float, default=0.0, help="skip prims smaller than this (m) in every axis")
parser.add_argument("--headless", action="store_true", default=True)
args_cli, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args_cli.headless})

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
asset_path = os.path.join(BASE, args_cli.asset)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
root = stage.DefinePrim("/World/Asset")
root.GetReferences().AddReference(asset_path)

for _ in range(60):
    simulation_app.update()

pattern = re.compile(args_cli.filter, re.IGNORECASE) if args_cli.filter else None
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

print(f"[HIER] {asset_path}", flush=True)
print(f"[HIER] filter={args_cli.filter!r} min_size={args_cli.min_size}", flush=True)
print("-" * 100, flush=True)

count = 0
for prim in Usd.PrimRange(root):
    name = prim.GetName()
    if pattern is not None and not pattern.search(name):
        continue
    try:
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    except Exception:
        continue
    mn, mx = rng.GetMin(), rng.GetMax()
    size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    if max(size) < args_cli.min_size:
        continue
    if max(abs(v) for v in size) < 1e-6:
        continue  # prim has no geometry of its own (pure Xform grouping node)
    center = ((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0)
    print(f"  {prim.GetPath()!s:<70} center=({center[0]:+7.2f},{center[1]:+7.2f},{center[2]:+7.2f})  size=({size[0]:5.2f},{size[1]:5.2f},{size[2]:5.2f})", flush=True)
    count += 1

print("-" * 100, flush=True)
print(f"[HIER] {count} matching prim(s). Re-run with --filter \"\" to see the full tree if nothing useful matched.", flush=True)
simulation_app.close()
