#!/usr/bin/env python3
"""Move the R1's right hand up, then back down. High-level and safe.

    python arm_actions.py --list        # READ ONLY: what this robot supports
    python arm_actions.py --wave        # dry run: show the sequence
    python arm_actions.py --wave --go   # actually move

Why this and not arm_wave.py
----------------------------
arm_wave.py drives rt/lowcmd, which is full low-level control of every joint
including the legs -- publish one bad frame and a standing R1 collapses. This
uses the robot's "arm" RPC service instead: we ask for a named action and the
robot's own controller performs it while keeping its balance. It cannot make
the robot fall over the way rt/lowcmd can.

The trade is precision: these are canned actions, so "right hand up" is
whatever Unitree defined it as, not exactly 30 degrees. For a first movement
that is the right trade. Exact angles need rt/lowcmd and a supported robot.

Discovered on this R1: service "arm" is live at rt/api/arm/request.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

from r1vision.unitree_dds import (  # noqa: E402
    default_peers, init_channel, quiet_dds_stderr,
)

# Queried from THIS robot with --list on 2026-07-28. The SDK ships a G1 table
# that disagrees: it lacks 28-36 entirely, lists 20/21 which this robot does not
# have, and misnames several (17 is clamp not "clap", 24 is ultraman_ray not
# "x-ray"). Always trust --list over the SDK.
ACTIONS = {
    "release_arm": 99,
    "blow_kiss_with_both_hands": 11,
    "blow_kiss_with_left_hand": 12,
    "blow_kiss_with_right_hand": 13,
    "both_hands_up": 15,
    "clamp": 17,
    "high_five": 18,
    "hug": 19,
    "refuse": 22,
    "right_hand_up": 23,
    "ultraman_ray": 24,
    "wave_under_head": 25,
    "wave_above_head": 26,
    "shake_hand": 27,
    "box_left_hand_win": 28,
    "box_right_hand_win": 29,
    "box_both_hand_win": 30,
    "extend_right_arm_forward": 31,
    "right_hand_on_heart": 33,
    "both_hands_up_deviate_right": 34,
    "emphasize": 35,
    "forward_push": 36,
}

RIGHT_HAND_UP = 23
RELEASE_ARM = 99


def connect(iface: str, timeout: float, settle: float = 8.0):
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    init_channel(iface, peers=None)
    client = G1ArmActionClient()
    client.SetTimeout(timeout)
    client.Init()

    # DDS discovery is asynchronous. Calling straight after Init() publishes a
    # request nobody is subscribed to yet, which comes back as 3102 "request
    # send error" -- it looks like the service is missing when it is simply not
    # matched yet. Sweeping 94 unicast peers takes a while, so give it time.
    if settle > 0:
        print(f"  waiting {settle:.0f}s for DDS discovery to settle ...")
        time.sleep(settle)
    return client


def call_with_retry(fn, attempts: int = 6, delay: float = 4.0):
    """Retry an RPC while discovery finishes converging.

    Returns (code, data). 3102/3104 mean not-matched-yet or timeout, both of
    which routinely clear on a later attempt; anything else is a real answer
    from the robot and is returned immediately.
    """
    last = (3102, None)
    for i in range(attempts):
        result = fn()
        code, data = result if isinstance(result, tuple) else (result, None)
        if code not in (3102, 3104):
            return code, data
        last = (code, data)
        if i < attempts - 1:
            print(f"  attempt {i + 1}/{attempts}: {describe_code(code)}; "
                  f"retrying in {delay:.0f}s ...")
            time.sleep(delay)
    return last


def describe_code(code: int) -> str:
    known = {
        0: "success",
        3102: "request send error (no server reachable)",
        3103: "API not registered",
        3104: "request timeout",
        3202: "internal server error",
        3203: "API not implemented on the server",
        3204: "API parameter error",
        3205: "request rejected (robot may be in the wrong mode)",
    }
    return known.get(code, f"error {code}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--iface", default="auto",
                    help="interface on 192.168.123.x (ethernet)")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=15.0,
                    help="seconds to let DDS discovery converge before calling")
    ap.add_argument("--list", action="store_true",
                    help="ask the robot which actions it supports (read only)")
    ap.add_argument("--wave", action="store_true",
                    help="right hand up, pause, release back down")
    ap.add_argument("--action", help="run one action by name or id")
    ap.add_argument("--pause", type=float, default=3.0,
                    help="seconds to hold the arm up (default 3)")
    ap.add_argument("--go", action="store_true",
                    help="actually send commands (default is a dry run)")
    args = ap.parse_args()

    if not (args.list or args.wave or args.action):
        ap.print_help()
        return 1

    # DDS logs udp write failures from background threads for the whole run,
    # so the redirect has to stay active for the whole run too. Everything this
    # script prints goes to stdout, so nothing of ours is lost.
    import atexit
    _quiet = quiet_dds_stderr()
    _quiet.__enter__()
    atexit.register(lambda: _quiet.__exit__(None, None, None))

    print(f"connecting on {args.iface} ...")
    try:
        client = connect(args.iface, args.timeout, args.settle)
    except ImportError as exc:
        print(f"error: SDK not importable: {exc}")
        return 1
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 1
    print("client ready\n")

    # ------------------------------------------------------------- list
    if args.list:
        print("asking the robot for its action list (read only) ...")
        try:
            code, data = call_with_retry(client.GetActionList, attempts=10, delay=5.0)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}")
            return 1
        if code != 0:
            print(f"failed after retries: {describe_code(code)}")
            print("\nThe arm service never matched. Either it lives on a "
                  "participant outside the swept port range, or it is not "
                  "running. Try --settle 20 first.")
            return 1
        print("robot reports:\n")
        print(data)
        print("\nNames this script knows (from your robot):")
        for name, aid in sorted(ACTIONS.items(), key=lambda kv: kv[1]):
            mark = "  <-- right hand up" if aid == RIGHT_HAND_UP else ""
            print(f"  {aid:>3}  {name}{mark}")
        return 0

    # ------------------------------------------------------------- resolve
    if args.action:
        if args.action.isdigit():
            sequence = [(int(args.action), args.action)]
        elif args.action in ACTIONS:
            sequence = [(ACTIONS[args.action], args.action)]
        else:
            print(f"unknown action {args.action!r}. Known: "
                  f"{', '.join(sorted(ACTIONS))}")
            return 1
    else:
        sequence = [
            (RIGHT_HAND_UP, "right hand up"),
            (RELEASE_ARM, "release arm (back down)"),
        ]

    print("sequence:")
    for i, (aid, name) in enumerate(sequence):
        print(f"  {i + 1}. action {aid:>3}  {name}")
        if i < len(sequence) - 1:
            print(f"     then wait {args.pause}s")

    if not args.go:
        print("\nDRY RUN -- nothing sent. Add --go to execute.")
        return 0

    # Pre-flight: confirm the service actually answers before counting down.
    # Without this you get a 5s countdown followed by a failure, which is both
    # annoying and misleading about whether anything was sent.
    print("\nchecking the arm service is reachable ...")
    code, _ = call_with_retry(client.GetActionList, attempts=10, delay=5.0)
    if code != 0:
        print(f"  arm service not reachable: {describe_code(code)}")
        print("  Nothing was sent. Run 'python wait_for_services.py' and check")
        print("  whether rt/api/arm/response ever appears as PUB (a server).")
        return 1
    print("  arm service responded")

    print("\nMake sure the space around the robot's right arm is clear.")
    print("Executing in 5s -- Ctrl-C to abort ...")
    try:
        for i in (5, 4, 3, 2, 1):
            print(f"  {i}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\naborted before sending anything.")
        return 130

    for i, (aid, name) in enumerate(sequence):
        print(f"\n-> action {aid} ({name})")
        try:
            code, _ = call_with_retry(lambda a=aid: client.ExecuteAction(a),
                                      attempts=3, delay=3.0)
        except KeyboardInterrupt:
            print("\ninterrupted; sending 'release arm' to return the arm ...")
            try:
                client.ExecuteAction(RELEASE_ARM)
            except Exception:
                pass
            return 130
        except Exception as exc:
            print(f"   error: {type(exc).__name__}: {exc}")
            return 1

        if code == 0:
            print("   accepted")
        else:
            print(f"   FAILED: {describe_code(code)}")
            if code in (3205, 3203):
                print("   The robot may need to be standing/enabled first, or "
                      "this action is not supported on the R1.")
            return 1

        if i < len(sequence) - 1:
            time.sleep(args.pause)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
