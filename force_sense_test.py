#!/usr/bin/env python3
"""Feel a hand push on the arm, using only joint torque -- no F/T sensor exists.

    python force_sense_test.py --hand right

Purely a sensing test: the arm is held stiff at a moderate gain (not the full
strength r1_lowcmd_arm.py uses for lifting), and this script only ever
commands that ONE fixed pose. Nothing new is asked of the arm while running --
the point is to read arm_kinematics.estimate_force() against a real external
push, not to move anything.

Why the arm has to be stiff first
----------------------------------
tau_est reports what the motor is CURRENTLY doing to hold its commanded
position. A limp joint has nothing to resist with, so pushing on it just
moves it -- there is very little torque to read. Held stiff, an external
push displaces the joint slightly, and the motor's response IS the signal
this script reads.

What this cannot do
--------------------
This is a static estimate (see estimate_force()'s docstring in
arm_kinematics.py): valid for a slow, sustained push, not a sharp tap. And it
is one arm, one baseline pose -- it says nothing about the other joints or
about how tau_est's own accuracy compares to a real sensor, which is
Unitree's claim, not something measured here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds", "numpy"))

import numpy as np  # noqa: E402

from r1vision import arm_kinematics  # noqa: E402

RIGHT_ARM = [22, 23, 24, 25, 26]
LEFT_ARM = [15, 16, 17, 18, 19]
ARMS = {"right": RIGHT_ARM, "left": LEFT_ARM}

CONTROL_HZ = 100
# 0.35 was tried first and, on hardware, did not actually feel stiff: a real
# push just moved the arm rather than building up resisting torque, so
# nothing meaningful showed up in tau_est. Matching r1_lowcmd_arm.py's lift
# gain (kp_scale=1.0, i.e. KP=80) instead -- that one is proven stiff on
# this robot: its own hold test explicitly invited nudging and measured
# 0.02 deg of drift. Use this gain, not a softer guess.
KP_SCALE = 1.0
BASELINE_S = 1.5
RUN_S = 20.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--hand", choices=("left", "right"), default="right")
    ap.add_argument("--seconds", type=float, default=RUN_S)
    args = ap.parse_args()

    from r1_lowcmd_arm import LowCmdArm

    joints = ARMS[args.hand]
    kin = arm_kinematics.chain(args.hand)

    arm = LowCmdArm(args.iface)
    print(f"connecting on {args.iface} ...")
    if not arm.connect():
        print("error: no rt/lowstate. Robot powered and on ethernet?", file=sys.stderr)
        return 1
    print(f"connected. mode_machine={arm.mode_machine}")

    held = {j: arm.angles()[j] for j in joints}
    q0 = np.array([held[j] for j in joints])
    print(f"\n  holding the {args.hand} arm's CURRENT pose, gently "
          f"(kp scale {KP_SCALE}). Starting in 3s -- do not push yet.")
    for i in (3, 2, 1):
        print(f"    {i}")
        time.sleep(1)

    period = 1.0 / CONTROL_HZ
    try:
        # ---- baseline: torque to hold this pose with nothing external ----
        # Averaged over the whole window, not a single end-of-window sample:
        # tau_est is a live sensor reading and carries real noise: pushing
        # against that noise as a fixed baseline would make small real
        # pushes indistinguishable from it.
        print(f"\n  capturing baseline for {BASELINE_S:.1f}s -- hands off ...")
        tau_samples = []
        t_end = time.monotonic() + BASELINE_S
        while time.monotonic() < t_end:
            arm.send(held, kp_scale=KP_SCALE)
            tau_samples.append(
                [arm.state.motor_state[j].tau_est for j in joints])
            time.sleep(period)
        tau_baseline = np.mean(np.array(tau_samples), axis=0)
        print(f"  baseline torque: {np.round(tau_baseline, 3)} Nm")

        # ---- live: press now ----
        print(f"\n  PRESS NOW -- reporting for {args.seconds:.0f}s "
              "(push firmly, hold a second, then let go to see it return "
              "toward zero) ...\n")
        t_end = time.monotonic() + args.seconds
        last_print = 0.0
        while time.monotonic() < t_end:
            arm.send(held, kp_scale=KP_SCALE)
            now = time.monotonic()
            if now - last_print > 0.2:
                last_print = now
                tau_now = np.array(
                    [arm.state.motor_state[j].tau_est for j in joints])
                delta = tau_now - tau_baseline
                f_est = kin.estimate_force(q0, delta)
                mag = float(np.linalg.norm(f_est))
                bar = "#" * min(60, int(mag * 3))
                print(f"\r  |F| ~ {mag:5.1f} N   "
                      f"F={np.round(f_est, 1)}  {bar:<60s}", end="", flush=True)
            time.sleep(period)
        print()

    except KeyboardInterrupt:
        print("\n  interrupted.")
    finally:
        print("\n  releasing (joints disabled) ...")
        for _ in range(int(0.5 * CONTROL_HZ)):
            arm.send({})
            time.sleep(period)

    return 0


if __name__ == "__main__":
    sys.exit(main())
