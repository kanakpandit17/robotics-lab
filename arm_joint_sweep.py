#!/usr/bin/env python3
"""Which arm joints actually execute? One joint at a time, both arms together.

    python arm_joint_sweep.py --read-only    # safe, commands nothing
    python arm_joint_sweep.py                # MOVES BOTH ARMS

Why this exists
---------------
`shoulder_sweep.py` answers the question for shoulder pitch only, because that
is the joint that failed. But once a joint has failed twice with an audible
noise and a progressive trend, "is it only that one?" is worth answering before
blaming a single motor -- and it is not answerable from teleop.

Teleop cannot answer it at all. `teleop_diag.py`'s own numbers said j15 was
tracking better than the healthy right shoulder while j15 was seized, because
`ik()` seeds from the previously solved pose: the command chases wherever the
arm actually is, so a dead joint drags its setpoint along and reports near-zero
error. Any test where the command depends on the measurement is worthless for
this. So every joint here gets an OPEN-LOOP command and is judged on whether it
moved and whether it drew current.

Both arms are driven together, one joint index at a time, by the SAME delta
from wherever each arm currently sits. Commanding a delta rather than an
absolute angle sidesteps the left/right mirroring question entirely -- we are
asking "does this motor respond", not "are the two arms symmetric" -- and it
keeps every move small and local rather than swinging to some absolute pose.

SAFETY
------
Drives real joints over rt/lowcmd, which bypasses the built-in balancing
safeguards. The robot must be supported, the arms clear, and `r1_mode.py
--release` already done. Only the ten arm joints are ever enabled; every other
joint stays at mode 0, because an unset target in a LowCmd_ defaults to zero
and zeros sent to the legs fold the robot up instantly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

from r1_lowcmd_arm import LIMITS, LowCmdArm, deg, rad  # noqa: E402

ARM = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}
JOINT_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll"]

# Small and local: enough travel to be unambiguous, little enough that a
# surprise cannot build much speed. The failing joint showed 0.3 deg against a
# 40 deg command, so 12 is far more than needed to separate alive from dead.
AMPLITUDE_DEG = 12.0
SETTLE_S = 1.0
# Alive means BOTH moved and drew current. Either alone is ambiguous: a joint
# already at gravity equilibrium legitimately needs almost no torque, which is
# how this exact fault was once misread as "something is holding it up".
DEAD_TRAVEL_DEG = 2.0
DEAD_TAU_NM = 0.5
# A joint delivering this many times less torque than its mirror under the same
# command is suspect even if it moved. j15 measured 2.1 Nm against j22's 10.9
# (5.2x) while passing on travel alone.
TAU_ASYMMETRY = 3.0


def _hottest(t) -> int:
    if isinstance(t, (list, tuple)):
        vals = [v for v in t if isinstance(v, (int, float))]
        return int(max(vals)) if vals else 0
    return int(t or 0)


def snapshot(arm) -> dict:
    st = arm.state
    return {side: [{"q": deg(st.motor_state[i].q),
                    "tau": float(st.motor_state[i].tau_est),
                    "temp": _hottest(getattr(st.motor_state[i], "temperature", 0))}
                   for i in idxs]
            for side, idxs in ARM.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--read-only", action="store_true")
    ap.add_argument("--amplitude", type=float, default=AMPLITUDE_DEG)
    ap.add_argument("--settle", type=float, default=SETTLE_S)
    ap.add_argument("--only", help="comma-separated joint names to test")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    arm = LowCmdArm(args.iface)
    print(f"connecting on {args.iface!r} ...")
    if not arm.connect():
        print("no rt/lowstate -- robot powered and on this subnet?", file=sys.stderr)
        return 1
    print(f"  connected, mode_machine={arm.mode_machine}\n")

    snap = snapshot(arm)
    print(f"  {'joint':10s} {'L q_deg':>9s} {'L tau':>7s} {'L C':>4s}   "
          f"{'R q_deg':>9s} {'R tau':>7s} {'R C':>4s}")
    for j, name in enumerate(JOINT_NAMES):
        l, r = snap["left"][j], snap["right"][j]
        print(f"  {name:10s} {l['q']:9.2f} {l['tau']:7.2f} {l['temp']:4d}   "
              f"{r['q']:9.2f} {r['tau']:7.2f} {r['temp']:4d}")

    if args.read_only:
        print("\n--read-only: nothing commanded. At-rest values cannot tell a")
        print("dead joint from a healthy one -- run without --read-only.")
        return 0

    todo = list(enumerate(JOINT_NAMES))
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        todo = [(j, n) for j, n in todo if n in want]
        if not todo:
            print(f"error: --only matched nothing; pick from {JOINT_NAMES}",
                  file=sys.stderr)
            return 2

    if not args.yes:
        print(f"\nThis will move BOTH arms, one joint at a time, "
              f"+/-{args.amplitude:.0f} deg from where they are now.")
        print("Robot supported, arms clear, `r1_mode.py --release` already done.")
        try:
            if input("Type 'move' to proceed: ").strip() != "move":
                print("aborted -- nothing sent.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\naborted -- nothing sent.")
            return 1

    results = {}
    try:
        base = arm.angles()
        hold = {i: base[i] for idxs in ARM.values() for i in idxs}
        for k in range(20):                       # ramp stiffness in at the
            arm.send(hold, kp_scale=(k + 1) / 20)  # CURRENT pose, so nothing
            time.sleep(0.02)                       # snaps when authority starts

        for j, name in todo:
            print(f"\n--- {name} ---")
            print(f"  {'target d':>9s} | {'L q':>8s} {'L tau':>7s} | "
                  f"{'R q':>8s} {'R tau':>7s}")
            trav = {"left": [], "right": []}
            taus = {"left": [], "right": []}
            for delta in (args.amplitude, -args.amplitude, 0.0):
                targets = dict(hold)
                for side, idxs in ARM.items():
                    idx = idxs[j]
                    lo, hi = LIMITS[idx]
                    targets[idx] = max(lo, min(hi, base[idx] + rad(delta)))
                end = time.monotonic() + args.settle
                while time.monotonic() < end:
                    arm.send(targets, kp_scale=1.0)
                    time.sleep(0.01)
                s = snapshot(arm)
                l, r = s["left"][j], s["right"][j]
                for side, m in (("left", l), ("right", r)):
                    trav[side].append(m["q"])
                    taus[side].append(abs(m["tau"]))
                print(f"  {delta:+9.1f} | {l['q']:8.2f} {l['tau']:7.2f} | "
                      f"{r['q']:8.2f} {r['tau']:7.2f}")
            results[name] = {
                side: {"travel": max(trav[side]) - min(trav[side]),
                       "tau": max(taus[side])}
                for side in ("left", "right")}
    finally:
        try:
            now = arm.angles()
            hold = {i: now[i] for idxs in ARM.values() for i in idxs}
            for k in range(20, -1, -1):     # ramp out rather than cutting kp,
                arm.send(hold, kp_scale=k / 20)   # which would let the arm drop
                time.sleep(0.02)
        except Exception as exc:
            print(f"  (could not ramp down cleanly: {exc})")

    if not results:
        return 1
    print("\n=== verdict ===")
    print(f"  {'joint':10s} {'L travel':>9s} {'L tau':>7s} | "
          f"{'R travel':>9s} {'R tau':>7s}")
    dead = []
    weak = []
    for name, r in results.items():
        l, rr = r["left"], r["right"]
        note = ""
        if l["travel"] < DEAD_TRAVEL_DEG and l["tau"] < DEAD_TAU_NM:
            note = "  <-- LEFT DEAD"
            dead.append("left " + name)
        if rr["travel"] < DEAD_TRAVEL_DEG and rr["tau"] < DEAD_TAU_NM:
            note += "  <-- RIGHT DEAD"
            dead.append("right " + name)
        # Moving is not the same as being able to move. A joint whose torque
        # is capped executes these small deltas perfectly and still cannot
        # lift the arm -- j15 passed this test twice at 2.1-2.5 Nm while its
        # mirror pulled 10.9-11.5 Nm, then failed a real lift minutes later.
        # Without this check the tool prints a clean bill of health for a
        # joint that is plainly broken.
        if not note and min(l["tau"], rr["tau"]) > 0:
            ratio = max(l["tau"], rr["tau"]) / min(l["tau"], rr["tau"])
            if ratio >= TAU_ASYMMETRY:
                side = "LEFT" if l["tau"] < rr["tau"] else "RIGHT"
                note = f"  <-- {side} produced {ratio:.1f}x less torque"
                weak.append(f"{side.lower()} {name}")
        print(f"  {name:10s} {l['travel']:9.2f} {l['tau']:7.2f} | "
              f"{rr['travel']:9.2f} {rr['tau']:7.2f}{note}")

    print()
    if dead:
        print(f"  Not executing: {', '.join(dead)}")
        print("  A joint that neither moves NOR draws current is a hardware")
        print("  fault. No teleop or IK change will affect it.")
    if weak:
        print(f"  Torque asymmetry: {', '.join(weak)}")
        print("  These joints MOVED, so travel alone would call them healthy.")
        print("  A capped joint passes small, low-load moves like these and")
        print("  still cannot lift the arm. Confirm with r1_lowcmd_arm --lift")
        print("  or j15_torque_probe.py, which demand real torque.")
    if not dead and not weak:
        print("  Every joint moved, drew current, and matched its mirror's")
        print("  torque. Note this test is LOW LOAD by design -- it does not")
        print("  prove a joint can lift the arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
