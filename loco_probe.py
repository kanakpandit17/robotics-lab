#!/usr/bin/env python3
"""Find out why the robot is not walking. READ ONLY -- moves nothing.

    python loco_probe.py

Two things have to be true before a thumbstick can move the robot, and when it
does not move you cannot tell which one failed by watching it stand there. This
checks both.

1. Which service implements walking
-----------------------------------
The SDK contradicts itself: `unitree_sdk2py/g1/loco` is hardcoded to the service
name "sport", while `unitree_sdk2py/h1/loco` uses "loco". Your R1 advertises
command topics for both -- `rt/api/loco/request` and `rt/api/sport/request` are
both in control-report.txt -- so the topic list cannot settle it.

Sending to the wrong one fails silently. The call is accepted by DDS, nothing
answers, and the robot simply does not move. This asks each service for its
current state (api 7001, a read) and reports which one replies.

2. Which state the robot is in
------------------------------
Velocity commands are refused unless the robot is in a state that walks. The FSM
id this prints is that state. If a service answers but movement is still
refused, that is what to change -- with `r1_walk.py` key 1, or "Stand & balance"
on the headset page.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

from r1vision.locomotion import (  # noqa: E402
    SERVICE_NAMES, _client_for, describe, probe_services,
)

# The SDK registers these api ids but only wraps 7001 in a method, so the rest
# are called raw. All five are reads.
GETTERS = {
    7001: "fsm id",
    7002: "fsm mode",
    7003: "balance mode",
    7004: "swing height",
    7005: "stand height",
}


def read_all(iface: str, service: str, timeout: float = 2.0) -> dict:
    """Every read-only field the locomotion service exposes, plus who owns the
    joints. Nothing here commands motion."""
    from r1vision.unitree_dds import init_channel, quiet_dds_stderr

    out: dict = {"getters": {}, "mode": None}
    with quiet_dds_stderr():
        init_channel(iface, peers=None)
        client = _client_for(service)
        client.SetTimeout(timeout)
        client.Init()

        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        switcher = MotionSwitcherClient()
        switcher.SetTimeout(timeout)
        switcher.Init()
        time.sleep(2.0)

        for api_id, label in GETTERS.items():
            try:
                code, data = client._Call(api_id, "{}")
                out["getters"][label] = data if code == 0 else describe(code)
            except Exception as exc:
                out["getters"][label] = f"{type(exc).__name__}: {exc}"

        try:
            code, data = switcher.CheckMode()
            out["mode"] = data if code == 0 else describe(code)
        except Exception as exc:
            out["mode"] = f"{type(exc).__name__}: {exc}"
    return out

# Only the ids the SDK names. Anything else is printed as-is rather than guessed
# at, since the numbering is not identical across Unitree humanoids.
KNOWN_FSM = {
    0: "zero torque (limp)",
    1: "damp",
    3: "sit",
    4: "stand up",
    500: "start / main operation",
    702: "lie to stand",
    706: "squat to stand",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto",
                    help="interface facing the robot (ethernet)")
    ap.add_argument("--timeout", type=float, default=2.0)
    args = ap.parse_args()

    print(f"probing {', '.join(SERVICE_NAMES)} on {args.iface} ... "
          "(read only, nothing moves)\n")
    results = probe_services(args.iface, timeout=args.timeout)

    working = []
    for r in results:
        if r["code"] == 0:
            fsm = r["fsm"]
            name = KNOWN_FSM.get(fsm, "unrecognised id")
            print(f"  {r['service']:>6} : ANSWERED   robot FSM {fsm} ({name})")
            working.append(r)
        else:
            print(f"  {r['service']:>6} : {r['detail']}")

    print()
    if not working:
        print("  Nothing answered. The locomotion service is not reachable.")
        print("  Check, in this order:")
        print(f"    - the ethernet link is up and 192.168.123.161 pings")
        print(f"    - --iface {args.iface} is the interface facing the robot")
        print("    - arm control still works (python arm_actions.py --list),")
        print("      which shares the same DDS path and proves it is not the link")
        return 1

    chosen = working[0]
    print(f"  Use service '{chosen['service']}':")
    print(f"    ./vr.sh --walk --loco-service {chosen['service']}")
    print(f"    python r1_walk.py --loco-service {chosen['service']}")

    print("\nreading the rest of the locomotion state ...\n")
    state = read_all(args.iface, chosen["service"], args.timeout)
    for label, value in state["getters"].items():
        print(f"  {label:>14} : {value}")
    print(f"  {'owning mode':>14} : {state['mode']}")

    # Who owns the joints is the thing that bit the arms in this project
    # already: commands were accepted and silently discarded while a built-in
    # mode held them. Same shape of failure applies here.
    mode = state["mode"]
    if isinstance(mode, dict):
        name = mode.get("name") or "(none)"
        print(f"\n  A mode named '{name}' owns the joints.")
        if not mode.get("name"):
            print("  With no mode active nothing is driving the legs, so the")
            print("  robot cannot walk. Select a walking mode in the Unitree")
            print("  app, or with:  python r1_mode.py --select normal")

    fsm = chosen["fsm"]
    if fsm in (0, 1, 3):
        print(f"\n  The robot is in FSM {fsm} ({KNOWN_FSM[fsm]}); it will not")
        print("  walk from there. Stand it up first -- r1_walk.py key 1 or 2,")
        print("  or the Unitree app -- then re-run this.")
    elif fsm not in KNOWN_FSM:
        print(f"\n  FSM {fsm} is not an id the SDK names, so the R1 numbers its")
        print("  states differently from the G1 that locomotion.py's table came")
        print("  from. Re-run this while the robot is standing under the Unitree")
        print("  app: whatever id it reports then is the one that walks, and is")
        print("  what 'Stand & balance' should be sending.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
