"""Convert lara8's real URDF (verified 6-DOF, from Neura) into a USD asset, the same way
maira7M.usd already exists in this project -- just scripted instead of done through the GUI
importer, so it is reproducible if the URDF is ever updated.

    python import_lara8.py                # headless, writes imported_cad/lara8.usd
    python import_lara8.py --headless=False   # watch it happen in the Isaac Sim window

Deliberately does NOT add a ground plane, physics scene, camera or lights -- those are
scene-level concerns added by whatever later loads this asset (r1_maira_scene.py's pattern,
or an Isaac Lab environment config), not baked into the robot USD itself. maira7M.usd is not
a full scene either; this keeps lara8.usd consistent with it.
"""
import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", type=lambda s: s.lower() != "false", default=True)
parser.add_argument("--urdf", default=r"C:\Users\aimat\Desktop\humanoid\robotics-lab\lara8\lara8.urdf")
parser.add_argument("--out", default=r"C:\Users\aimat\Desktop\humanoid\imported_cad\lara8.usd")
args_cli = parser.parse_args()

simulation_app = SimulationApp({"headless": args_cli.headless})

import sys  # noqa: E402

import omni.kit.app  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402


def log(msg: str) -> None:
    # A bare SimulationApp launch does not guarantee stdout is flushed before
    # the process exits -- confirmed here: an earlier version of this check
    # produced a fully empty log despite running to completion, because every
    # print() was still sitting in Python's buffer when the process closed.
    # This project's own scripts already flush=True for the same reason
    # (r1_lara_maira_loop.py, maira_sim_bridge.py); this just makes the same
    # fix explicit here too.
    sys.stdout.write(f"[IMPORT] {msg}\n")
    sys.stdout.flush()


# The URDF importer commands do not exist until this extension is actually
# enabled -- a bare SimulationApp() launch does not enable it automatically.
# Calling omni.kit.commands.execute("URDFCreateImportConfig") before this
# line does not raise; it just never runs, and the script exits with no
# output and no USD file, which looks identical to a silent hang. Confirmed
# by testing without this line first, not assumed.
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.asset.importer.urdf", True)

# A fresh stage, not whatever happened to be open -- this script's only job is producing
# one clean lara8.usd, and importing into a stage that already has other prims risks a
# path collision or picking up unrelated content by accident.
omni.usd.get_context().new_stage()

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False   # keep the flange/gripper-TCP joints distinct
import_config.fix_base = True              # base_link bolts to the world, not a floating arm
import_config.make_default_prim = True
import_config.create_physics_scene = False  # scene-level concern, not the robot's -- see docstring
import_config.self_collision = False
import_config.distance_scale = 1.0

log(f"parsing {args_cli.urdf} ...")
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=args_cli.urdf,
    import_config=import_config,
)
log(f"result={result} prim_path={prim_path}")
if not result:
    log("FAILED -- see the Isaac Sim log above for the reason")
    simulation_app.close()
    raise SystemExit(1)

stage = omni.usd.get_context().get_stage()
default_prim = stage.GetDefaultPrim()
log(f"default prim: {default_prim.GetPath() if default_prim else 'NONE'}")

# Revolute-joint count is the one number that actually matters here: it is
# the DOF Isaac Lab will see. Confirmed against the real robot earlier
# (6 joints, neurapy) and the URDF (6 <joint type="revolute">) -- this should
# match both, or something was lost in the conversion.
joint_count = sum(
    1 for prim in stage.Traverse() if prim.GetTypeName() == "PhysicsRevoluteJoint")
log(f"revolute joints found in stage: {joint_count} (expect 6)")


# stage.Export(), not omni.usd.get_context().save_as_stage(): the latter is
# asynchronous under Kit's hood, and closing the app right after calling it
# raced the actual write -- confirmed by testing, the log printed "saved"
# while the directory listing showed nothing at all had been written.
# Usd.Stage.Export() is the raw Pixar API and blocks until the file is
# actually on disk.
import os  # noqa: E402

# Belt and braces: the directory demonstrably already existed (maira7M.usd
# and others sit right there) and Export() still raised "could not create
# directory" -- a real pxr.Tf.ErrorException that never reached this script's
# own log, because SimulationApp routes an uncaught Python exception into
# Kit's own log stream instead of crashing normally. The process then sat
# alive with nothing left to do, which from outside looked exactly like a
# hang, not a fast failure -- confirmed only by reading the raw Isaac Sim
# log, not by the CPU/process state, which was misleading. Wrapping Export()
# in try/except is what makes the NEXT failure show up here instead of
# requiring that again.
os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
try:
    ok = stage.Export(args_cli.out)
    log(f"stage.Export() returned {ok}")
except Exception as exc:
    log(f"FAILED -- stage.Export() raised {type(exc).__name__}: {exc}")
    simulation_app.close()
    raise SystemExit(1)

if os.path.exists(args_cli.out):
    log(f"confirmed on disk: {args_cli.out} ({os.path.getsize(args_cli.out)} bytes)")
else:
    log(f"FAILED -- {args_cli.out} does not exist after Export()")
    simulation_app.close()
    raise SystemExit(1)

simulation_app.close()
