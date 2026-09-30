#!/usr/bin/env python3
"""Drive both arms to the SAME mirrored pose and compare what each achieves.

    python mirror_pose_test.py --plan     # show the pose, command nothing
    python mirror_pose_test.py            # MOVES BOTH ARMS

Why this exists
---------------
Every other test here drives one joint at a time. This drives a whole natural
posture -- hands raised in front of the chest -- because that is what an
operator actually asks for, and because a side-by-side of the two arms
attempting the identical motion is the most legible evidence there is: one arm
gets there, the other does not, in the same message.

The pose is specified as MIRRORED JOINT ANGLES rather than as a Cartesian
target solved by IK. That matters. If each arm's pose came from its own IK
solve, a difference in outcome could be the solver rather than the robot, and
this repository has already been fooled that way more than once. Mirrored joint
angles remove the solver from the comparison entirely: both arms are commanded
the same numbers, with the signs the URDF's mirrored axes require.

Mirroring convention, verified on hardware 2026-08-24: shoulder pitch and elbow
take the SAME sign on both arms; shoulder roll, shoulder yaw and wrist roll take
OPPOSITE signs. Under that convention the eight healthy joints tracked to
0.06-1.05 deg, so any larger difference is the joint, not the mapping.

SAFETY
------
Drives real joints over rt/lowcmd, bypassing the built-in balancing safeguards.
Robot supported, arms clear, `r1_mode.py --release` already done. Only the ten
arm joints are enabled; everything else stays at mode 0.
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

ARM = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}
JOINT_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll"]
# +1 keeps the sign, -1 flips it, going from the right arm to the left.
MIRROR = np.array([1.0, -1.0, -1.0, 1.0, -1.0])

# "Hands up in front of the chest", written for the RIGHT arm. Shoulder pitch
# raises the arm, the elbow folds it in front, and a little roll brings the hand
# toward the midline -- but only a little, because roll runs out at +13 deg on
# the right and reaching across the body is the one motion this robot cannot do.
POSE_RIGHT_DEG = np.array([-45.0, -20.0, 0.0, 85.0, 0.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--hold", type=float, default=6.0)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    right = np.radians(POSE_RIGHT_DEG)
    left = right * MIRROR
    goal = {"right": right, "left": left}

    print("target pose (deg), mirrored:")
    print(f"  {'joint':10s} {'left':>9s} {'right':>9s}")
    for j, n in enumerate(JOINT_NAMES):
        print(f"  {n:10s} {math.degrees(left[j]):9.1f} {math.degrees(right[j]):9.1f}")

    for side in ("left", "right"):
        kin = arm_kinematics.chain(side)
        q = np.clip(goal[side], kin.lower, kin.upper)
        if not np.allclose(q, goal[side]):
            print(f"  NOTE: {side} pose clipped by joint limits -> "
                  f"{np.round(np.degrees(q),1)}")
        goal[side] = q
        print(f"  {side:5s} hand would sit at {np.round(kin.fk(q), 3)} m")

    if args.plan:
        print("\n--plan: nothing commanded.")
        return 0

    arm = LowCmdArm(args.iface)
    print(f"\nconnecting on {args.iface!r} ...")
    if not arm.connect():
        print("no rt/lowstate", file=sys.stderr)
        return 1

    if not args.yes:
        print("\nThis moves BOTH arms to the pose above. Arms clear?")
        try:
            if input("Type 'move' to proceed: ").strip() != "move":
                print("aborted."); return 1
        except (EOFError, KeyboardInterrupt):
            print("\naborted."); return 1

    def ease(p):
        return (1.0 - math.cos(math.pi * p)) / 2.0

    base = arm.angles()
    peak = {s: np.zeros(5) for s in ARM}
    try:
        for k in range(25):
            arm.send({i: base[i] for idxs in ARM.values() for i in idxs},
                     kp_scale=(k + 1) / 25)
            time.sleep(0.02)

        steps = int(args.seconds / 0.02)
        for s in range(steps + 1):
            f = ease(s / steps)
            tgt = {}
            for side, idxs in ARM.items():
                for n, i in enumerate(idxs):
                    lo, hi = LIMITS[i]
                    v = base[i] + f * (goal[side][n] - base[i])
                    tgt[i] = max(lo, min(hi, float(v)))
            arm.send(tgt, kp_scale=1.0)
            st = arm.state
            for side, idxs in ARM.items():
                for n, i in enumerate(idxs):
                    peak[side][n] = max(peak[side][n],
                                        abs(float(st.motor_state[i].tau_est)))
            time.sleep(0.02)

        final = {}
        for side, idxs in ARM.items():
            for n, i in enumerate(idxs):
                lo, hi = LIMITS[i]
                final[i] = max(lo, min(hi, float(goal[side][n])))
        end = time.monotonic() + args.hold
        while time.monotonic() < end:
            arm.send(final, kp_scale=1.0)
            st = arm.state
            for side, idxs in ARM.items():
                for n, i in enumerate(idxs):
                    peak[side][n] = max(peak[side][n],
                                        abs(float(st.motor_state[i].tau_est)))
            time.sleep(0.02)

        reached = {s: np.array([arm.angles()[i] for i in idxs])
                   for s, idxs in ARM.items()}
    finally:
        try:
            now = arm.angles()
            for k in range(25, -1, -1):
                arm.send({i: now[i] for idxs in ARM.values() for i in idxs},
                         kp_scale=k / 25)
                time.sleep(0.02)
        except Exception as exc:
            print(f"  (ramp-down issue: {exc})")

    print("\n=== result: same mirrored command, both arms ===")
    print(f"  {'joint':10s} | {'L want':>7s} {'L got':>7s} {'L err':>6s} {'L tau':>6s}"
          f" | {'R want':>7s} {'R got':>7s} {'R err':>6s} {'R tau':>6s}")
    for j, n in enumerate(JOINT_NAMES):
        lw, lg = deg(goal["left"][j]), deg(reached["left"][j])
        rw, rg = deg(goal["right"][j]), deg(reached["right"][j])
        le, re_ = abs(lw - lg), abs(rw - rg)
        flag = "  <--" if le > re_ + 10 else ""
        print(f"  {n:10s} | {lw:7.1f} {lg:7.1f} {le:6.1f} {peak['left'][j]:6.2f}"
              f" | {rw:7.1f} {rg:7.1f} {re_:6.1f} {peak['right'][j]:6.2f}{flag}")

    for side in ("left", "right"):
        kin = arm_kinematics.chain(side)
        print(f"  {side:5s} hand ended at {np.round(kin.fk(reached[side]), 3)} m "
              f"(wanted {np.round(kin.fk(goal[side]), 3)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
