#!/usr/bin/env python3
"""Left-vs-right shoulder pitch check: does j15 actually execute commands?

    python shoulder_sweep.py --read-only     # safe: reads rt/lowstate, moves nothing
    python shoulder_sweep.py                 # MOVES BOTH ARMS -- see safety below

Why this exists
---------------
On 2026-08-24 the R1's left arm "teleoperating badly" was traced past IK,
past the clutch and past the operator to the joint itself: commanded through
nine identical shoulder-pitch targets from +5 deg down to -35 deg, the right
shoulder (j22) tracked every step with tau_est rising -0.66 -> -4.67 Nm as
the arm extended, while the left (j15) sat at 10.77 deg for ALL NINE targets
with tau_est flat at -0.04 Nm. At 45.77 deg of error with kp=80 the loop
should have demanded roughly 63 Nm. The arm was verified mechanically free by
hand afterwards, and no fault flags were set. A joint that accepts commands
and produces no current is a hardware fault, not a kinematics bug.

That test was written ad hoc and thrown away, which is why this file exists:
the fault appeared mid-session and looked progressive (23 deg error on an
earlier replay, 44 deg later), so it may be intermittent, and the question
"is j15 alive today?" has to be answerable in one command before any teleop
session is worth starting.

Commanding BOTH shoulders to the SAME angle is the whole design. One arm's
tracking error means nothing on its own -- position control against gravity
always leaves a steady-state error, and on this robot it is large (TELEOP.md
measures 15 deg commanded -> 3.7 deg achieved at kp=30). Only the comparison
separates "this is what the R1 does" from "this joint is dead", because both
sides see the same command, the same gains and the same gravity.

SAFETY
------
Without --read-only this drives real joints over rt/lowcmd, which the manual
says bypasses the built-in balancing safeguards. Before running it:

  - the robot must be LYING FLAT and supported, never in a hoist -- releasing
    control makes every joint limp, and that already dropped this robot once
  - `python r1_mode.py --release` must have been run, or `ai` mode owns the
    joints and nothing here will move
  - clear the space both arms sweep through

Only the two shoulder pitch joints are ever enabled. Every other joint is
left at mode 0 (disabled), because an unset target in a LowCmd_ defaults to
zero and sending zeros to the legs folds the robot up instantly.
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

from r1_lowcmd_arm import LowCmdArm, deg, rad  # noqa: E402

# Both shoulder PITCH joints. The memo's earlier suspects -- shoulder roll and
# wrist roll -- both tracked fine; pitch is the one that failed.
PITCH = {"left": 15, "right": 22}
# Every arm joint, so a fault that has moved to a different joint since the
# last run is visible instead of being missed by looking only where it was.
ARM = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}
JOINT_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll"]

# The nine targets from the run that found the fault, so results compare
# directly against the numbers in the memo rather than to a new baseline.
TARGETS_DEG = [5.0, 0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0]
SETTLE_S = 1.2
# A joint that is executing shows BOTH: it moves, and it draws current doing
# it. Either one alone is ambiguous -- a joint already at gravity equilibrium
# legitimately needs almost no torque, which is how an earlier read of this
# same fault was misinterpreted as "something is holding it up".
DEAD_TAU_NM = 0.5
DEAD_SPAN_DEG = 2.0
# A joint can move a little and still be failing. These catch the partial case:
# a large standing error that the joint is making no real effort to close.
# Calibrated against the healthy side, which closes to under 2 deg of error and
# reaches 2.5 Nm doing it, versus the failing j15 sitting at 30 deg on 0.8 Nm.
STALL_ERR_DEG = 10.0
STALL_TAU_NM = 1.5


def _hottest(t) -> int:
    """Motor temperature, whether the field is a scalar or a per-motor list."""
    if isinstance(t, (list, tuple)):
        vals = [v for v in t if isinstance(v, (int, float))]
        return int(max(vals)) if vals else 0
    return int(t or 0)


def read_arms(arm) -> dict:
    """Snapshot both arms from the latest rt/lowstate."""
    st = arm.state
    out = {}
    for side, idxs in ARM.items():
        out[side] = [{
            "idx": i,
            "q_deg": round(deg(st.motor_state[i].q), 2),
            "tau": round(float(st.motor_state[i].tau_est), 2),
            # temperature is a LIST on this robot (several sensors per motor),
            # not the scalar the G1 examples show. Report the hottest.
            "temp": _hottest(getattr(st.motor_state[i], "temperature", 0)),
            "mode": int(getattr(st.motor_state[i], "mode", 0) or 0),
        } for i in idxs]
    return out


def report_static(snap: dict) -> None:
    print(f"  {'joint':10s} {'idx':>4s} {'q_deg':>8s} {'tau_Nm':>8s} "
          f"{'temp_C':>7s} {'mode':>5s}")
    for side in ("left", "right"):
        for name, m in zip(JOINT_NAMES, snap[side]):
            print(f"  {side[0]}_{name:8s} {m['idx']:4d} {m['q_deg']:8.2f} "
                  f"{m['tau']:8.2f} {m['temp']:7d} {m['mode']:5d}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--read-only", action="store_true",
                    help="read rt/lowstate and print both arms; commands nothing")
    ap.add_argument("--settle", type=float, default=SETTLE_S,
                    help="seconds to hold each target before measuring")
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="stiffness scale 0..1, as r1_lowcmd_arm uses it")
    ap.add_argument("--yes", action="store_true",
                    help="skip the are-you-sure prompt (robot must be supported)")
    args = ap.parse_args()

    arm = LowCmdArm(args.iface)
    print(f"connecting on iface {args.iface!r} ...")
    if not arm.connect():
        print("no rt/lowstate received -- is the robot powered and on this subnet?",
              file=sys.stderr)
        return 1
    print(f"  connected, mode_machine={arm.mode_machine}\n")

    print("current state, both arms:")
    report_static(read_arms(arm))

    if args.read_only:
        print("\n--read-only: nothing was commanded.")
        print("This shows what the joints report at rest. It CANNOT tell you")
        print("whether j15 executes a command -- a dead joint at rest looks")
        print("exactly like a healthy one. Re-run without --read-only, with")
        print("the robot lying flat and `r1_mode.py --release` already done.")
        return 0

    if not args.yes:
        print("\nThis will DRIVE BOTH SHOULDERS through "
              f"{TARGETS_DEG[0]:+.0f}..{TARGETS_DEG[-1]:+.0f} deg.")
        print("The robot must be lying flat and supported, NOT in a hoist,")
        print("and `python r1_mode.py --release` must already have been run.")
        try:
            if input("Type 'move' to proceed: ").strip() != "move":
                print("aborted -- nothing was sent.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\naborted -- nothing was sent.")
            return 1

    results = []
    try:
        # Take authority at the CURRENT pose first, so the arms stiffen where
        # they already are instead of snapping to the first target.
        here = arm.angles()
        hold = {PITCH[s]: here[PITCH[s]] for s in PITCH}
        for i in range(20):
            arm.send(hold, kp_scale=args.kp_scale * (i + 1) / 20)
            time.sleep(0.02)

        print(f"\n{'target':>8s} | {'left j15':>18s} | {'right j22':>18s}")
        print(f"{'deg':>8s} | {'q_deg':>8s} {'tau_Nm':>9s} | "
              f"{'q_deg':>8s} {'tau_Nm':>9s}")
        for tgt in TARGETS_DEG:
            targets = {PITCH[s]: rad(tgt) for s in PITCH}
            end = time.monotonic() + args.settle
            while time.monotonic() < end:
                arm.send(targets, kp_scale=args.kp_scale)
                time.sleep(0.01)
            snap = read_arms(arm)
            l, r = snap["left"][0], snap["right"][0]
            results.append({"target": tgt, "left": l, "right": r})
            print(f"{tgt:8.1f} | {l['q_deg']:8.2f} {l['tau']:9.2f} | "
                  f"{r['q_deg']:8.2f} {r['tau']:9.2f}")
    finally:
        # Ramp stiffness back down rather than cutting it: dropping kp to zero
        # in one step lets the arm fall from wherever it is.
        try:
            here = arm.angles()
            hold = {PITCH[s]: here[PITCH[s]] for s in PITCH}
            for i in range(20, -1, -1):
                arm.send(hold, kp_scale=args.kp_scale * i / 20)
                time.sleep(0.02)
        except Exception as exc:
            print(f"  (could not ramp down cleanly: {exc})")

    if not results:
        return 1

    print("\n--- verdict ---")
    verdict_bad = False
    for side, idx in (("left", 15), ("right", 22)):
        qs = [r[side]["q_deg"] for r in results]
        taus = [abs(r[side]["tau"]) for r in results]
        span = max(qs) - min(qs)
        peak = max(taus)
        errs = [abs(r[side]["q_deg"] - r["target"]) for r in results]
        dead = span < DEAD_SPAN_DEG and peak < DEAD_TAU_NM
        # Travel alone is not enough, and reporting only it produced a false
        # "both shoulders executed" on a run where j15 tracked the first three
        # targets, stalled at -4.2 deg, and sat there through a further 30 deg
        # of command with its torque flat at 0.8 Nm. The joint had "travelled"
        # 8.6 deg, clearing the span threshold, while plainly failing. A joint
        # that stops moving as the error GROWS is failing under load -- which
        # is the more informative fault, because it says the motor responds
        # until real torque is demanded.
        stalled = (not dead) and max(errs) > STALL_ERR_DEG and peak < STALL_TAU_NM
        note = ""
        if dead:
            note = "   <-- DEAD: accepts commands, produces no current"
        elif stalled:
            note = (f"   <-- STALLS UNDER LOAD: {max(errs):.0f} deg error at "
                    f"only {peak:.2f} Nm")
        print(f"  {side:5s} j{idx}: travelled {span:5.1f} deg over the sweep, "
              f"peak |tau| {peak:5.2f} Nm, max error {max(errs):5.1f} deg{note}")
        verdict_bad |= dead or stalled

    if verdict_bad:
        print("\n  A joint that neither moves NOR draws current is not a")
        print("  kinematics or teleop problem. Do not spend the session on IK.")
        print("  Power-cycle the robot and re-run this before anything else;")
        print("  the fault has previously been intermittent.")
    else:
        print("\n  Both shoulders executed. If teleop still favours one arm,")
        print("  the cause is above the joint level -- run hand_trace.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
