"""Inspect kuka_assembly.usd: overall bbox + each top-level child's bbox, so we know where the
enclosure, the KUKA robot, and the open floor actually sit within the asset."""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom

KUKA = r"C:\Users\aimat\Desktop\humanoid\imported_cad\kuka_assembly.usd"
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/K")
stage.DefinePrim("/World/K/a").GetReferences().AddReference(KUKA)
for _ in range(40):
    app.update()

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/K")).ComputeAlignedRange()
mn, mx = rng.GetMin(), rng.GetMax()
print(f"[KUKA] OVERALL  x[{mn[0]:.2f},{mx[0]:.2f}]  y[{mn[1]:.2f},{mx[1]:.2f}]  z[{mn[2]:.2f},{mx[2]:.2f}]", flush=True)
print(f"[KUKA] center=({(mn[0]+mx[0])/2:.2f},{(mn[1]+mx[1])/2:.2f})  size=({mx[0]-mn[0]:.2f},{mx[1]-mn[1]:.2f})", flush=True)

root = stage.GetPrimAtPath("/World/K/a")


def walk(prim, depth=0):
    for child in prim.GetChildren():
        try:
            r = cache.ComputeWorldBound(child).ComputeAlignedRange()
            a, b = r.GetMin(), r.GetMax()
            if b[0] >= a[0]:
                print(f"[KUKA] {'  '*depth}{child.GetName():<28} x[{a[0]:6.2f},{b[0]:6.2f}] y[{a[1]:6.2f},{b[1]:6.2f}] z[{a[2]:5.2f},{b[2]:5.2f}]", flush=True)
        except Exception:
            pass
        if depth < 1:
            walk(child, depth + 1)


walk(root)
print("KUKA_INSPECT_DONE", flush=True)
app.close()
