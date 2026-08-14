#!/usr/bin/env python3
"""Check that moving your hand a direction moves the arm that direction.

    python axis_check.py                       # against a local dry-run server
    python axis_check.py --host 192.168.94.68  # against the laptop serving VR

Why this exists
---------------
`arm_teleop.AXIS_MAP` converts WebXR controller coordinates into the robot's
frame. It has never been checked against a real controller -- only against
itself, by `arm_teleop_cli.py`, which emits synthetic pre-AXIS_MAP vectors and
therefore agrees with any mapping you give it, right or wrong. A self-
consistent loop cannot catch a wrong mapping. Only a human moving a real
controller in a known direction can.

That matters because a wrong AXIS_MAP is indistinguishable, from inside the
headset, from "the teleop is bad": your hand goes forward and the arm goes
sideways, and no amount of practice fixes it. This is the check that was
missing before the first live VR session.

How to use it
-------------
Run the server in dry-run so nothing can move:

    python quest_stream.py --teleop --teleop-dry-run --teleop-hands right

Put the headset on, enter VR, then run this and follow the prompts. For each
prompt: hold the grip, move your hand ONE way (a slow ~30 cm move), release.
This reads the driver read-only (`/teleop?status=1`) and never commands
anything itself.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("numpy",))

import numpy as np  # noqa: E402

from r1vision.arm_teleop import AXIS_MAP  # noqa: E402

# What the robot frame means, so the verdict can be phrased in words a person
# can check against the robot in front of them rather than axis indices.
ROBOT_AXES = {
    0: ("forward", "backward"),   # +x out of the robot's chest
    1: ("left", "right"),         # +y to the robot's left
    2: ("up", "down"),            # +z up
}

# The moves to ask for, and which robot direction each should produce.
# Wording matters more than it looks. "Move your hand right, across your
# body" was read -- correctly -- as moving the right hand across the midline,
# which is LEFTWARD, and produced a confident WRONG verdict against a mapping
# that was fine. Each prompt now names one direction and says what to hold
# still, because an ambiguous instruction here is indistinguishable from a
# wrong AXIS_MAP in the result.
CHECKS = [
    ("Reach FORWARD, straight out from your chest (same height)", "forward"),
    ("Raise your hand UP toward the ceiling (straight up)", "up"),
    ("Move your hand OUT TO YOUR RIGHT SIDE, away from your body "
     "-- NOT across your chest, and keep it at the same height", "right"),
]

MIN_TRAVEL_M = 0.08   # below this the move is too small to read confidently


def fetch(url: str, timeout: float = 10.0) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode())


def collect_segments(url: str, hand: str, want: int, timeout: float = 180.0):
    """Watch for `want` grip-press-to-release segments and return each one's
    start and end position.

    Deliberately not prompt-driven: the operator is wearing a headset and
    cannot read this terminal. So instead of asking for one move at a time,
    this watches passively while they do all the moves from instructions
    given up front, and segments the stream by grip transitions.
    """
    segments = []
    start = None
    last = None
    # Transient failures must not end the run. Sharing a phone hotspot with
    # an MJPEG video stream, a poll can simply time out; giving up on the
    # first one threw away a whole capture attempt with the operator still
    # standing there in the headset. Only a long unbroken run of failures
    # means the server is actually gone.
    misses = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(segments) < want:
        try:
            st = fetch(url)
            misses = 0
        except Exception as e:
            misses += 1
            if misses % 10 == 1:
                print(f"  (slow link: {e})", flush=True)
            if misses > 60:
                print("  server unreachable for too long, giving up")
                return segments
            time.sleep(0.2)
            continue
        ctrl = (st.get("controller") or {}).get(hand)
        if not ctrl:
            time.sleep(0.1)
            continue
        p = np.array(ctrl["pos"], dtype=float)
        if ctrl.get("grip"):
            if start is None:
                start = p
                print(f"  move {len(segments) + 1}: grip down, tracking ...",
                      flush=True)
            last = p
        elif start is not None:
            travel = float(np.linalg.norm(last - start))
            if travel >= MIN_TRAVEL_M:
                segments.append((start, last))
                print(f"  move {len(segments)}: captured "
                      f"({travel * 100:.0f} cm)", flush=True)
            else:
                print(f"  ignored a {travel * 100:.0f} cm move "
                      f"(under {MIN_TRAVEL_M * 100:.0f} cm)", flush=True)
            start = None
            last = None
        time.sleep(0.05)
    return segments


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--hand", default="right", choices=("left", "right"))
    args = ap.parse_args()

    url = f"https://{args.host}:{args.port}/teleop?status=1"
    try:
        st = fetch(url)
    except Exception as e:
        print(f"error: cannot reach {url}\n  {e}", file=sys.stderr)
        return 1
    if not st.get("ok"):
        print(f"error: {st.get('detail')}", file=sys.stderr)
        return 1
    if args.hand not in (st.get("hands") or []):
        print(f"error: server is not driving the {args.hand} hand "
              f"(it has {st.get('hands')})", file=sys.stderr)
        return 1

    print(f"  checking the {args.hand} controller against AXIS_MAP")
    print(f"  waiting for {len(CHECKS)} moves, in this order:\n")
    for i, (prompt, _) in enumerate(CHECKS, 1):
        print(f"    {i}. {prompt}")
    print("\n  For each: hold GRIP, move ~30 cm, release. Then the next one.\n")

    segments = collect_segments(url, args.hand, want=len(CHECKS))
    if not segments:
        print("\n  nothing captured. Is the headset in VR, and does grip work?")
        return 1

    print()
    results = []
    for (prompt, expect), (start, last) in zip(CHECKS, segments):
        webxr_delta = last - start
        travel = float(np.linalg.norm(webxr_delta))
        robot_delta = AXIS_MAP @ webxr_delta

        ax = int(np.argmax(np.abs(robot_delta)))
        pos_name, neg_name = ROBOT_AXES[ax]
        got_dir = pos_name if robot_delta[ax] > 0 else neg_name

        # Grade the SIGN along the axis under test, not which axis happens to
        # be largest. Demanding dominance was wrong: a human arm pivots at the
        # shoulder, so "move out to your right" traces an arc whose vertical
        # component easily exceeds its sideways one. That failed a correct
        # mapping twice on move 3 while forward and up passed -- the test was
        # measuring how well a person can translate their hand in a straight
        # line, which is not what AXIS_MAP does or needs to do.
        want_ax = next(a for a, (p, n) in ROBOT_AXES.items()
                       if expect in (p, n))
        want_sign = 1.0 if expect == ROBOT_AXES[want_ax][0] else -1.0
        component = float(robot_delta[want_ax]) * want_sign

        print(f"  {prompt}")
        print(f"    hand moved {travel * 100:.0f} cm")
        print(f"    WebXR delta : {np.round(webxr_delta, 3)}")
        print(f"    robot delta : {np.round(robot_delta, 3)}  "
              f"(largest: {got_dir.upper()})")
        print(f"    component along {expect.upper()}: {component:+.3f} m")
        ok = component > 0.02
        print(f"    expected {expect.upper()}: "
              f"{'OK' if ok else 'WRONG (moved the opposite way)'}\n")
        results.append((prompt, expect, got_dir, ok))

    print("  " + "=" * 58)
    graded = results
    for _, expect, got_dir, ok in graded:
        mark = "OK   " if ok else "WRONG"
        print(f"  {mark}  expected {expect:8s} got {got_dir}")
    bad = [r for r in graded if not r[3]]
    print("  " + "=" * 58)
    if not bad:
        print("\n  AXIS_MAP is correct for these directions.\n")
        return 0
    print(f"\n  {len(bad)} of {len(graded)} directions are wrong. AXIS_MAP in\n"
          "  r1vision/arm_teleop.py needs correcting -- the rows/signs that\n"
          "  produced the wrong axes above. Do NOT run live teleop until this\n"
          "  passes: from inside the headset a wrong mapping is\n"
          "  indistinguishable from the teleop simply not working.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
