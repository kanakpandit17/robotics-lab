"""Print metersPerUnit, up-axis, and world bounding box for each asset (to plan scene scale)."""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

from pxr import Usd, UsdGeom

BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
ASSETS = {
    "shopfloor_nowalls": os.path.join(BASE, "arena_no_walls.usd"),
    "kuka": os.path.join(BASE, "kuka_assembly.usd"),
    "r1": os.path.join(BASE, "r1.usd"),
}

for name, path in ASSETS.items():
    if not os.path.isfile(path):
        print(f"[{name}] MISSING {path}", flush=True)
        continue
    stage = Usd.Stage.Open(path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    size = mx - mn
    print(
        f"[{name}] metersPerUnit={mpu}  upAxis={up}\n"
        f"    bbox_min=({mn[0]:.1f},{mn[1]:.1f},{mn[2]:.1f})  bbox_max=({mx[0]:.1f},{mx[1]:.1f},{mx[2]:.1f})\n"
        f"    size(units)=({size[0]:.1f},{size[1]:.1f},{size[2]:.1f})   size(meters)=({size[0]*mpu:.2f},{size[1]*mpu:.2f},{size[2]*mpu:.2f})",
        flush=True,
    )

print("INSPECT_DONE", flush=True)
app.close()
