#!/usr/bin/env python3
"""Move the R1's right arm. Staged so nothing surprising happens.

Run over ETHERNET (en6, 192.168.123.222). Wi-Fi cannot reach the control ports.

    python r1_arm.py --joints              # 1. READ ONLY: show live angles
    python r1_arm.py --check               # 2. publish with weight 0 (cannot move)
    python r1_arm.py --engage              # 3. take authority, hold still
    python r1_arm.py --move --degrees -10  # 4. small move, verify direction
    python r1_arm.py --move --degrees -30  # 5. the real thing

Each stage is a superset of the one before, so a failure is caught while the
robot still has no authority handed to it. Do them in order the first time.

WHY rt/arm_sdk AND NOT rt/lowcmd
--------------------------------
Full discovery over ethernet shows the robot subscribes to rt/arm_sdk. That is
the blended path: we command arm joints while the robot keeps running its own
balance controller on the legs. rt/lowcmd would hand us every joint including
the legs, where one bad frame drops a standing robot. We do not need that here.
arm_sdk needs that controller running: use it in AI mode, never after
`r1_mode.py --release`.

JOINT MAP (measured on this robot, 35 slots, 24 real joints)
------------------------------------------------------------
     0-5   left leg          6-11  right leg        12,13 waist roll/yaw
    15-19  left arm         22-26  right arm         29,30 head pitch/yaw
    shoulder pitch/roll/yaw, elbow, wrist roll

    22 = RIGHT SHOULDER PITCH   <- raising the arm
    23 = right shoulder roll    24 = right shoulder yaw
    25 = right elbow            26 = right wrist roll

Confirmed by left/right symmetry in the live data: joint 15 read 12.82 deg and
joint 22 read 13.14 deg with the arms hanging; 18 and 25 both read ~46 deg.

DIRECTION
---------
The sign convention is not documented for the R1, so verify it rather than
trust it: run --move --degrees -10 first. If the arm goes down instead of up,
use +10. Only then go to 30.
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

TOPIC_STATE = "rt/lowstate"
TOPIC_ARM = "rt/arm_sdk"

RIGHT_SHOULDER_PITCH = 22
RIGHT_ARM = [22, 23, 24, 25, 26]
LEFT_ARM = [15, 16, 17, 18, 19]
WAIST_YAW = 13
HEAD = [29, 30]

# The 13 joints arm_sdk takes over. At weight 1.0 each needs a target: Unitree's
# take-over procedure holds every one at its measured position, because a joint
# left out would get an empty command with no stiffness.
ARM_SDK_JOINTS = LEFT_ARM + RIGHT_ARM + [WAIST_YAW] + HEAD

# The R1 reads the arm_sdk blend weight from mode_pr, 0-100 for 0.0-1.0
# (Unitree R1 arm_control_routine). Slot 29, where the G1 carries its weight, is
# head pitch on the R1: writing the weight there left mode_pr at 0, so every
# frame was blended out and the arm moved exactly 0.00 deg. Reported by one
# owner: mode_pr >= 2 switches FSM 811 to 816, which locks walking until reset
# (unitree_sdk2_python #182). First runs on the bracket, with the way back known.

# Unitree's gains for the built-in arm action service (arm_control_routine).
GAINS = {
    15: (50.0, 2.0), 16: (50.0, 2.0), 17: (40.0, 2.0), 18: (40.0, 2.0),
    19: (30.0, 2.0),
    22: (50.0, 2.0), 23: (50.0, 2.0), 24: (40.0, 2.0), 25: (40.0, 2.0),
    26: (30.0, 2.0),
    13: (50.0, 3.0), 29: (15.0, 1.0), 30: (15.0, 1.0),
}

MAX_DEGREES = 60.0
CONTROL_HZ = 100


def deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def rad(degrees: float) -> float:
    return degrees * math.pi / 180.0


def ease(phase: float) -> float:
    """Cosine easing -- no acceleration step through the gearbox."""
    return (1.0 - math.cos(math.pi * phase)) / 2.0


class Arm:
    def __init__(self, iface: str):
        self.iface = iface
        self.state = None
        self._pub = None
        self._crc = None

    @property
    def mode_machine(self) -> int:
        return int(getattr(self.state, "mode_machine", 0)) if self.state else 0

    def connect(self, timeout: float = 8.0) -> bool:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        from r1vision.unitree_dds import init_channel

        # On ethernet we are on the robot's own segment, so plain multicast
        # discovery works and no peer list is needed.
        init_channel(self.iface, peers=None)

        self._crc = CRC()
        # Held on the instance, not in a local: a ChannelSubscriber that goes
        # out of scope is garbage collected, and the subscription dies with it.
        # The failure is quiet and very misleading -- the first sample or two
        # arrive, then `state` freezes at whatever it last saw, so every later
        # reading looks like a joint that is not moving. That cost a whole
        # debugging session: a commanded arm sweep appeared to be ignored when
        # the readings were simply stale.
        self._sub = ChannelSubscriber(TOPIC_STATE, LowState_)
        self._sub.Init(lambda m: setattr(self, "state", m), 10)

        self._pub = ChannelPublisher(TOPIC_ARM, LowCmd_)
        self._pub.Init()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.state is None:
            time.sleep(0.1)
        return self.state is not None

    def angles(self) -> list[float]:
        return [m.q for m in self.state.motor_state] if self.state else []

    def send(self, targets: dict[int, float], weight: float) -> None:
        """Command the listed joints; `weight` scales our authority."""
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        cmd = unitree_hg_msg_dds__LowCmd_()

        # The robot checks this against its own machine mode and drops frames
        # that do not match. It defaults to 0, which is never right -- it has
        # to be echoed back from lowstate.
        cmd.mode_machine = self.mode_machine
        cmd.mode_pr = int(round(100 * min(1.0, max(0.0, weight))))

        for idx, q in targets.items():
            m = cmd.motor_cmd[idx]
            # mode defaults to 0 = DISABLED. Without this the joint silently
            # ignores q/kp/kd and nothing moves at all.
            m.mode = 1
            kp, kd = GAINS[idx]
            m.q, m.dq, m.kp, m.kd, m.tau = q, 0.0, kp, kd, 0.0

        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)


def show_joints(arm: Arm) -> None:
    angles = arm.angles()
    print(f"\n{len(angles)} slots\n")
    print(" idx    deg     joint")
    names = {
        15: "left shoulder pitch", 16: "left shoulder roll",
        17: "left shoulder yaw", 18: "left elbow", 19: "left wrist roll",
        22: "RIGHT SHOULDER PITCH", 23: "right shoulder roll",
        24: "right shoulder yaw", 25: "right elbow", 26: "right wrist roll",
        12: "waist roll", 13: "waist yaw", 29: "head pitch", 30: "head yaw",
    }
    for i, q in enumerate(angles):
        label = names.get(i, "")
        if i < 12:
            label = f"{'left' if i < 6 else 'right'} leg {i % 6}"
        if not label and q == 0.0:
            continue
        print(f" {i:>3}  {deg(q):>7.2f}   {label}")
    print("\nMove the right arm by hand and re-run to confirm these indices.")


def hold_map(arm: Arm, joints: list[int]) -> dict[int, float]:
    angles = arm.angles()
    return {j: angles[j] for j in joints}


def run(arm: Arm, args) -> int:
    angles = arm.angles()
    joint = args.joint
    if joint >= len(angles):
        print(f"error: joint {joint} out of range ({len(angles)} slots)", file=sys.stderr)
        return 1

    period = 1.0 / CONTROL_HZ
    held = hold_map(arm, ARM_SDK_JOINTS)
    start = angles[joint]

    print(f"\n  joint {joint} currently at {deg(start):+.2f} deg")

    # ---------------------------------------------------------------- check
    if args.check:
        print("\n  STAGE: check -- publishing with weight 0 for 3s.")
        print("  Weight 0 means the robot keeps full authority; nothing can move.")
        for _ in range(3 * CONTROL_HZ):
            arm.send(held, weight=0.0)
            time.sleep(period)
        moved = abs(deg(arm.angles()[joint]) - deg(start))
        print(f"  done. Joint {joint} moved {moved:.2f} deg (expect ~0).")
        print("  Publishing works. Next: --engage")
        return 0

    # --------------------------------------------------------------- engage
    print("\n  Clear the space around the right arm.")
    print("  Starting in 5s -- Ctrl-C aborts ...")
    try:
        for i in (5, 4, 3, 2, 1):
            print(f"    {i}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  aborted; nothing was sent.")
        return 130

    def ramp_weight(lo: float, hi: float, seconds: float) -> None:
        steps = int(seconds * CONTROL_HZ)
        for i in range(steps):
            arm.send(held, weight=lo + (hi - lo) * (i / steps))
            time.sleep(period)

    try:
        print("\n  engaging (weight 0 -> 1 over 2s, all arm joints held) ...")
        ramp_weight(0.0, 1.0, 2.0)

        drift = abs(deg(arm.angles()[joint]) - deg(start))
        print(f"  engaged. Joint {joint} drifted {drift:.2f} deg (expect small).")
        if drift > 5.0:
            print("  !! more movement than expected; disengaging.", file=sys.stderr)
            ramp_weight(1.0, 0.0, 1.5)
            return 1

        if args.engage:
            print("\n  holding 3s ...")
            for _ in range(3 * CONTROL_HZ):
                arm.send(held, weight=1.0)
                time.sleep(period)
            print("  disengaging ...")
            ramp_weight(1.0, 0.0, 1.5)
            print("  done. The arm held position. Next: --move --degrees -10")
            return 0

        # ------------------------------------------------------------- move
        target = start + rad(args.degrees)
        travel = int(args.travel * CONTROL_HZ)
        print(f"\n  moving joint {joint}: {deg(start):+.2f} -> {deg(target):+.2f} deg")

        for i in range(travel):
            q = start + (target - start) * ease(i / travel)
            arm.send({**held, joint: q}, weight=1.0)
            time.sleep(period)
        print(f"  holding {args.hold}s at the top ...")
        for _ in range(int(args.hold * CONTROL_HZ)):
            arm.send({**held, joint: target}, weight=1.0)
            time.sleep(period)
        print("  returning ...")
        for i in range(travel):
            q = target + (start - target) * ease(i / travel)
            arm.send({**held, joint: q}, weight=1.0)
            time.sleep(period)

        arm.send({**held, joint: start}, weight=1.0)
        print("  disengaging ...")
        ramp_weight(1.0, 0.0, 1.5)
        print(f"\n  done. Final angle {deg(arm.angles()[joint]):+.2f} deg")
        return 0

    except KeyboardInterrupt:
        print("\n  interrupted -- returning to start and releasing ...")
        try:
            for i in range(int(1.5 * CONTROL_HZ)):
                arm.send({**held, joint: start}, weight=1.0)
                time.sleep(period)
            ramp_weight(1.0, 0.0, 1.5)
        except Exception:
            pass
        print("  released.")
        return 130


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto", help="interface on 192.168.123.x")
    ap.add_argument("--joint", type=int, default=RIGHT_SHOULDER_PITCH)
    ap.add_argument("--degrees", type=float, default=-30.0)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--travel", type=float, default=2.5, help="seconds each way")
    ap.add_argument("--joints", action="store_true", help="print angles and exit")
    ap.add_argument("--check", action="store_true", help="publish at weight 0")
    ap.add_argument("--engage", action="store_true", help="take authority, hold still")
    ap.add_argument("--move", action="store_true", help="perform the movement")
    args = ap.parse_args()

    if not (args.joints or args.check or args.engage or args.move):
        ap.print_help()
        return 1
    if args.joint not in ARM_SDK_JOINTS:
        print(f"error: --joint must be one of the arm_sdk joints {ARM_SDK_JOINTS}",
              file=sys.stderr)
        return 1
    if abs(args.degrees) > MAX_DEGREES:
        print(f"error: --degrees capped at +/-{MAX_DEGREES}", file=sys.stderr)
        return 1

    arm = Arm(args.iface)
    print(f"connecting on {args.iface} ...")
    try:
        ok = arm.connect()
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not ok:
        print(f"error: no {TOPIC_STATE} received on {args.iface}.", file=sys.stderr)
        print("Are you on ethernet with a 192.168.123.x address?", file=sys.stderr)
        return 1
    print(f"connected, {len(arm.angles())} slots reporting")
    print(f"  mode_machine = {arm.mode_machine}  (echoed back on every command)")
    if arm.mode_machine == 0:
        print("  !! mode_machine is 0. The robot may not be in a mode that")
        print("     accepts arm_sdk. If nothing moves, that is the reason.")

    if args.joints:
        show_joints(arm)
        return 0
    return run(arm, args)


if __name__ == "__main__":
    sys.exit(main())
