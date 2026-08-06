#!/usr/bin/env python3
"""Ask the R1 which motion mode owns the joints, and optionally release it.

    python r1_mode.py                 # READ ONLY: report current mode
    python r1_mode.py --release       # hand the arms over to arm_sdk
    python r1_mode.py --select ai     # put a named mode back in charge

Why this exists
---------------
arm_sdk frames are being accepted (mode_machine matches, motors enabled, CRC
valid) yet joint 22 drifts exactly 0.00 deg. A perfect zero is the signature of
commands being discarded, not of weak tracking -- something upstream still owns
the arm.

On Unitree humanoids the motion_switcher decides which controller owns the
joints. While a built-in mode (ai / normal / sport) is active it holds them and
external low-level commands are ignored. Releasing it is what frees the arms.

After --release the robot's own controller is no longer holding the arms, so
they may go limp until arm_sdk engages. Support the robot before running it.
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
    return CODES.get(code, f"error {code}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--release", action="store_true",
                    help="release the current mode so arm_sdk can take the arms")
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
        print("  While it is active, arm_sdk commands are ignored.")
    else:
        print("\n  No mode is active -- the joints should already be free.")
        print("  If arm_sdk still does nothing, the problem is elsewhere.")

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
        print("\n  !! Releasing means the robot's own controller stops holding")
        print("     the arms. Support the robot. Ctrl-C now to abort.")
        try:
            for i in (5, 4, 3, 2, 1):
                print(f"    {i}")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  aborted.")
            return 130

        code, _ = client.ReleaseMode()
        print(f"  ReleaseMode: {describe(code)}")
        if code != 0:
            return 1

        time.sleep(2)
        code, data = client.CheckMode()
        print(f"  mode now: {data}")
        print("\n  Next:  python r1_arm.py --move --degrees -10")
        print(f"  Restore afterwards with:  python r1_mode.py --select {name}")
        return 0

    print("\nRead-only. To free the arms:  python r1_mode.py --release")
    return 0


if __name__ == "__main__":
    sys.exit(main())
