#!/usr/bin/env python3
"""Ask the R1 which motion mode owns the joints, and optionally release it.

    python r1_mode.py                 # READ ONLY: report current mode
    python r1_mode.py --release       # switch the built-in controller off
    python r1_mode.py --select ai     # put a named mode back in charge

What release does
-----------------
On Unitree humanoids the motion_switcher decides which controller owns the
joints. ReleaseMode switches the robot's own controller off: every joint goes
limp, the legs included, and the remote no longer controls the robot. A
standing robot falls. Release only with the robot lying flat on a mat or
otherwise supported, never standing and never hanging on a hoist alone
(TELEOP.md). Restore with --select ai.

When it is needed
-----------------
Only for rt/lowcmd, which the arm teleop drives (TELEOP.md). Not for the
camera: quest_stream.py reads RTSP and works in AI mode. Not for rt/arm_sdk:
the R1 blends arm_sdk commands by the weight in mode_pr while its own
controller keeps balancing (r1_arm.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

CODES = {
    0: "success",
    3102: "request send error (service not reachable)",
    3104: "request timeout",
    3202: "internal server error",
    3203: "API not implemented",
    3204: "API parameter error",
    3205: "request rejected",
}


def describe(code: int) -> str:
    return CODES.get(code, f"error {code} (not in the SDK's table)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--release", action="store_true",
                    help="switch the built-in controller off: every joint goes "
                         "limp. Robot lying on a mat or supported only")
    ap.add_argument("--select", metavar="NAME",
                    help="select a mode by name (e.g. ai, normal)")
    ap.add_argument("--settle", type=float, default=3.0)
    args = ap.parse_args()

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )

    from r1vision.unitree_dds import init_channel

    print(f"connecting on {args.iface} ...")
    init_channel(args.iface, peers=None)

    client = MotionSwitcherClient()
    client.SetTimeout(10.0)
    client.Init()
    time.sleep(args.settle)

    # ------------------------------------------------------------ read first
    code, data = client.CheckMode()
    if code != 0:
        print(f"CheckMode failed: {describe(code)}", file=sys.stderr)
        return 1

    print("\ncurrent mode:")
    print(f"  {data}")
    name = (data or {}).get("name", "")
    form = (data or {}).get("form", "")
    if name:
        print(f"\n  A mode named '{name}' (form '{form}') currently owns the joints.")
        print("  While it is active, rt/lowcmd commands are ignored; rt/arm_sdk")
        print("  is blended in by its weight (mode_pr).")
    else:
        print("\n  No mode is active: the built-in controller is off and the")
        print("  joints are limp. Restore with --select ai.")

    # ------------------------------------------------------------- act
    if args.select:
        print(f"\nselecting mode '{args.select}' ...")
        code, _ = client.SelectMode(args.select)
        print(f"  {describe(code)}")
        return 0 if code == 0 else 1

    if args.release:
        if not name:
            print("\nNothing to release.")
            return 0
        print("\n  !! Release switches the robot's own controller off. Every joint")
        print("     goes limp, the legs included, and a standing robot falls.")
        print("     The robot must lie flat on a mat or be otherwise supported.")
        try:
            answer = input("  Type 'release' to continue, anything else aborts: ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip() != "release":
            print("\n  aborted; nothing was sent.")
            return 130

        code, _ = client.ReleaseMode()
        print(f"  ReleaseMode: {describe(code)}")
        if code != 0:
            return 1

        time.sleep(2)
        code, data = client.CheckMode()
        print(f"  mode now: {data}")
        print(f"\n  Restore afterwards with:  python r1_mode.py --select {name}")
        return 0

    print("\nRead-only. --release switches the built-in controller off;")
    print("robot lying on a mat or supported only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
