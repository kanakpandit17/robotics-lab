#!/usr/bin/env python3
"""Raise the left hand as far as the working joints allow, with j15 dead.

    python left_hand_lift.py --plan     # compute only, commands nothing
    python left_hand_lift.py            # MOVES THE LEFT ARM

Why this exists
---------------
Left shoulder pitch (j15) produces no torque, so `r1_lowcmd_arm.py --lift`
raises the right arm and leaves the left where it is. The other four left
joints are healthy, and the hand can still be raised using them -- mostly the
elbow, which lifts the forearm even when the upper arm cannot move.

This is a workaround for a hardware fault, not a fix. It exists so the left
hand can be positioned at all while the joint is being repaired. The height it
reaches is limited by the shoulder being stuck at whatever pitch it failed at,
so the number it prints is the honest ceiling for the current situation, not
the arm's real capability.

It plans against the URDF via arm_kinematics: search the reachable combinations
of the WORKING joints for whichever maximises hand height, holding j15 at its
measured angle. That is preferable to picking an elbow angle by eye, because
which combination lifts most depends on where the shoulder happens to be stuck.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

import numpy as np  # noqa: E402

from r1vision import arm_kinematics  # noqa: E402
from r1_lowcmd_arm import LIMITS, LowCmdArm, deg, rad  # noqa: E402

LEFT = [15, 16, 17, 18, 19]
DEAD = 0            # index within LEFT of the stuck joint (shoulder pitch)
RAMP_S = 3.0


def plan(kin, q_now: np.ndarray, max_change_deg: float) -> tuple[np.ndarray, float, float]:
    """Best pose for hand height, holding the dead joint where it is.

    Bounded by `max_change_deg` per joint, and that bound is not cosmetic. The
    unbounded optimum from a hanging start was shoulder roll +141 deg (1 deg
    off its limit), yaw +104, wrist roll -93 -- a 134 deg sideways swing that
    gains 57 cm and would sweep the arm through whatever the robot is standing
    in, on a shoulder that is already failing. Height is the objective, but it
    is not the only thing that matters, so the search is kept local.
    """
    rng = np.random.default_rng(0)
    best_q, best_z = q_now.copy(), float(kin.fk(q_now)[2])
    start_z = best_z
    m = rad(max_change_deg)
    lo = np.maximum(kin.lower, q_now - m)
    hi = np.minimum(kin.upper, q_now + m)
    # The dead joint cannot move, so pin it. Searching over it would produce a
    # plan the robot silently fails to execute -- exactly the failure mode this
    # whole investigation kept hitting.
    lo[DEAD] = hi[DEAD] = q_now[DEAD]
    for _ in range(60000):
        q = rng.uniform(lo, hi)
        q[DEAD] = q_now[DEAD]
        z = float(kin.fk(q)[2])
        if z > best_z:
            best_q, best_z = q.copy(), z
    return best_q, start_z, best_z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--max-change", type=float, default=35.0,
                    help="max degrees any one joint may move (safety bound)")
    ap.add_argument("--plan", action="store_true", help="compute only, send nothing")
    ap.add_argument("--seconds", type=float, default=RAMP_S)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    kin = arm_kinematics.chain("left")
    arm = LowCmdArm(args.iface)
    print(f"connecting on {args.iface!r} ...")
    if not arm.connect():
        print("no rt/lowstate", file=sys.stderr)
        return 1

    angles = arm.angles()
    q_now = np.array([angles[i] for i in LEFT])
    print(f"  left arm now (deg): {np.round(np.degrees(q_now), 1)}")
    print(f"  j15 stuck at {deg(q_now[DEAD]):.1f} deg -- held there in the plan\n")

    q_goal, z0, z1 = plan(kin, q_now, args.max_change)
    print(f"  hand height now  : {z0*100:+6.1f} cm (relative to shoulder frame)")
    print(f"  best reachable   : {z1*100:+6.1f} cm")
    print(f"  gain             : {(z1-z0)*100:+6.1f} cm")
    print(f"  target pose (deg): {np.round(np.degrees(q_goal), 1)}")

    if args.plan:
        print("\n--plan: nothing commanded.")
        return 0
    if abs(z1 - z0) < 0.01:
        print("\n  Less than 1 cm available -- not worth moving. Nothing sent.")
        return 0

    if not args.yes:
        print(f"\nThis moves the LEFT arm over {args.seconds:.0f}s. Arm clear?")
        try:
            if input("Type 'move' to proceed: ").strip() != "move":
                print("aborted."); return 1
        except (EOFError, KeyboardInterrupt):
            print("\naborted."); return 1

    def ease(p):
        return (1.0 - math.cos(math.pi * p)) / 2.0

    try:
        for k in range(25):                       # take authority where the arm
            arm.send({i: angles[i] for i in LEFT},  # already is, so nothing snaps
                     kp_scale=(k + 1) / 25)
            time.sleep(0.02)

        steps = int(args.seconds / 0.02)
        for s in range(steps + 1):
            f = ease(s / steps)
            tgt = {}
            for n, i in enumerate(LEFT):
                lo, hi = LIMITS[i]
                v = q_now[n] + f * (q_goal[n] - q_now[n])
                tgt[i] = max(lo, min(hi, float(v)))
            arm.send(tgt, kp_scale=1.0)
            time.sleep(0.02)

        end = time.monotonic() + args.hold
        final = {i: float(max(LIMITS[i][0], min(LIMITS[i][1], q_goal[n])))
                 for n, i in enumerate(LEFT)}
        while time.monotonic() < end:
            arm.send(final, kp_scale=1.0)
            time.sleep(0.02)

        reached = np.array([arm.angles()[i] for i in LEFT])
        print(f"\n  reached (deg): {np.round(np.degrees(reached), 1)}")
        print(f"  hand height  : {kin.fk(reached)[2]*100:+.1f} cm "
              f"(target {z1*100:+.1f})")
        for n, i in enumerate(LEFT):
            err = deg(abs(reached[n] - q_goal[n]))
            flag = "   <-- did not reach" if err > 5 else ""
            print(f"    j{i}: {deg(reached[n]):7.1f} vs target "
                  f"{deg(q_goal[n]):7.1f}  ({err:4.1f} deg off){flag}")
    finally:
        try:
            now = arm.angles()
            for k in range(25, -1, -1):   # ramp out; cutting stiffness would
                arm.send({i: now[i] for i in LEFT}, kp_scale=k / 25)
                time.sleep(0.02)
        except Exception as exc:
            print(f"  (ramp-down issue: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

