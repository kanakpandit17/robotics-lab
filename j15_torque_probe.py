#!/usr/bin/env python3
"""Does j15 respond to direct torque, or only fail through the position loop?

    python j15_torque_probe.py            # ramps feedforward torque on ONE joint

Why this exists
---------------
Every test so far has driven j15 through the position loop: set a target, let
kp*error produce the torque. j15 stalls there -- 30.8 deg of error producing
0.8 Nm when kp=80 should demand ~43 Nm.

That leaves one distinction untested, and it decides whether this is repairable
in software at all. rt/lowcmd carries a feedforward torque term (`tau`) that
`r1_lowcmd_arm.send()` always sets to 0.0. If the joint moves when torque is
commanded DIRECTLY, its motor and power stage work and only the position
control path is faulty -- worth reporting, and possibly workable around. If it
does not move under direct torque either, the motor is dead and no software
change will ever lift that arm.

The right shoulder is driven identically as a control, so "did it move" is
judged against a healthy joint under the same command on the same robot.

SAFETY
------
Torque control has no position target: if the joint suddenly works, it
accelerates rather than settling. So the ramp is small, brief, bounded, and
aborts the moment real motion appears -- the goal is a yes/no answer, not a
lift. Every other joint is held under normal position control throughout, and
non-arm joints stay disabled. Robot supported, arms clear, `r1_mode.py
--release` already done.
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

ARM = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}
PITCH = {"left": 15, "right": 22}
MOVED_DEG = 3.0          # enough to be unambiguous, small enough to be safe
MAX_TAU_NM = 6.0         # well under what the healthy side pulls in teleop (18)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--max-tau", type=float, default=MAX_TAU_NM)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    from r1vision.unitree_dds import init_channel

    init_channel(args.iface, peers=None)
    crc = CRC()
    state = {}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: state.__setitem__("m", m), 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and "m" not in state:
        time.sleep(0.1)
    if "m" not in state:
        print("no rt/lowstate", file=sys.stderr)
        return 1

    def q(i):
        return math.degrees(state["m"].motor_state[i].q)

    def tau(i):
        return float(state["m"].motor_state[i].tau_est)

    start = {s: q(i) for s, i in PITCH.items()}
    print(f"start: j15 {start['left']:+.2f} deg, j22 {start['right']:+.2f} deg")
    print(f"ramping feedforward torque to {args.max_tau:.1f} Nm on BOTH "
          f"shoulder pitches over {args.seconds:.0f}s.")
    print("Position control holds every other arm joint. Aborts on real motion.")

    if not args.yes:
        try:
            if input("Type 'move' to proceed: ").strip() != "move":
                print("aborted."); return 1
        except (EOFError, KeyboardInterrupt):
            print("\naborted."); return 1

    base = [m.q for m in state["m"].motor_state]
    t0 = time.monotonic()
    verdict = {}
    try:
        while True:
            el = time.monotonic() - t0
            if el > args.seconds:
                break
            ff = args.max_tau * (el / args.seconds)

            cmd = unitree_hg_msg_dds__LowCmd_()
            cmd.mode_machine = int(getattr(state["m"], "mode_machine", 0))
            cmd.mode_pr = 0
            for side, idxs in ARM.items():
                for i in idxs:
                    m = cmd.motor_cmd[i]
                    m.mode = 1
                    if i == PITCH[side]:
                        # Torque only: no position term at all, so the joint is
                        # asked for effort rather than a place to be.
                        m.q, m.dq, m.kp, m.kd, m.tau = 0.0, 0.0, 0.0, 0.5, ff
                    else:
                        m.q, m.dq, m.kp, m.kd, m.tau = base[i], 0.0, 80.0, 2.0, 0.0
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            moved = {s: q(i) - start[s] for s, i in PITCH.items()}
            if el % 0.5 < 0.02:
                print(f"  t={el:4.1f}s ff={ff:5.2f} Nm | "
                      f"j15 moved {moved['left']:+6.2f} (tau {tau(15):+5.2f}) | "
                      f"j22 moved {moved['right']:+6.2f} (tau {tau(22):+5.2f})")
            if any(abs(v) > MOVED_DEG for v in moved.values()):
                print(f"  -> motion exceeded {MOVED_DEG} deg, stopping")
                break
            time.sleep(0.01)
        verdict = {s: q(i) - start[s] for s, i in PITCH.items()}
    finally:
        # Hand back under position control at the pose actually reached, then
        # ramp stiffness out -- cutting torque dead would drop the arm.
        try:
            now = [m.q for m in state["m"].motor_state]
            for k in range(30, -1, -1):
                cmd = unitree_hg_msg_dds__LowCmd_()
                cmd.mode_machine = int(getattr(state["m"], "mode_machine", 0))
                cmd.mode_pr = 0
                for idxs in ARM.values():
                    for i in idxs:
                        m = cmd.motor_cmd[i]
                        m.mode = 1
                        m.q, m.dq, m.tau = now[i], 0.0, 0.0
                        m.kp, m.kd = 80.0 * k / 30, 2.0 * k / 30
                cmd.crc = crc.Crc(cmd)
                pub.Write(cmd)
                time.sleep(0.02)
        except Exception as exc:
            print(f"  (ramp-down issue: {exc})")

    print("\n=== verdict ===")
    for s, i in PITCH.items():
        print(f"  j{i} {s:5s}: moved {verdict.get(s, 0.0):+.2f} deg under "
              f"{args.max_tau:.1f} Nm feedforward")
    l, r = abs(verdict.get("left", 0)), abs(verdict.get("right", 0))
    if l < 1.0 and r >= 1.0:
        print("\n  j15 does not respond to DIRECT TORQUE either, while j22 does.")
        print("  That is the motor or its power stage, not the control loop.")
        print("  No software change can lift this arm -- it needs service.")
    elif l >= 1.0:
        print("\n  j15 DID respond to direct torque. Its motor works; the")
        print("  position-control path is what fails. Worth reporting to")
        print("  Unitree as a controller fault rather than a dead motor.")
    else:
        print("\n  Neither shoulder moved -- torque too low to overcome gravity.")
        print("  Re-run with a larger --max-tau before concluding anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
