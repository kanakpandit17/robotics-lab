#!/usr/bin/env python3
"""Lift the R1's arms over rt/lowcmd. Staged, and safe only lying down.

    python r1_lowcmd_arm.py --hold              # stiffen at the current pose
    python r1_lowcmd_arm.py --lift --degrees 15 # then actually move

WHY rt/lowcmd AND NOT rt/arm_sdk
---------------------------------
`r1_arm.py` targets rt/arm_sdk with a blend weight in slot 29, which is the
G1's convention. The R1 user manual's joint table shows slot 29 is
**HEAD_PITCH** on this robot -- a real motor with a +-0.6283 rad limit, not a
spare. There is no weight slot at all: indices 0-30 are entirely real joints
or EMPTY (14, 20, 21, 27, 28). So arm_sdk's weighted blending is a G1 feature
that does not exist here, and every arm_sdk frame we sent was quietly
commanding the head instead of setting a weight.

The manual documents low-level control as commanding joint targets directly
on rt/lowcmd, with no blending -- which is what this does.

WHY THIS IS ONLY SAFE LYING DOWN
---------------------------------
The manual is explicit that low-level commands "bypass built-in balancing
safeguards". Nothing catches a mistake here. Every non-arm joint is therefore
left at mode 0 (disabled) rather than being sent a position: an unset target
in a LowCmd_ defaults to zero, and sending zeros to the legs of a standing
robot folds it up instantly. Disabled means the robot simply is not driven
there.
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
TOPIC_CMD = "rt/lowcmd"

# Manual, page 29. Only these are ever enabled by this script.
RIGHT_ARM = [22, 23, 24, 25, 26]   # shoulder pitch/roll/yaw, elbow, wrist roll
LEFT_ARM = [15, 16, 17, 18, 19]
SHOULDER_PITCH = {"right": 22, "left": 15}

# Limits (rad) for the joints this script drives, from unitree_r1/R1.urdf. The
# left and right shoulder roll mirror each other; the manual's page 29 gives the
# left arm the right arm's roll range, which blocked the left arm beyond 13 deg
# outward and would have allowed 142 deg inward.
LIMITS = {
    22: (-3.1416, 2.0944), 23: (-2.4784, 0.2269), 24: (-1.9199, 1.9199),
    25: (-0.9757, 2.1850), 26: (-1.9199, 1.9199),
    15: (-3.1416, 2.0944), 16: (-0.2269, 2.4784), 17: (-1.9199, 1.9199),
    18: (-0.9757, 2.1850), 19: (-1.9199, 1.9199),
}

CONTROL_HZ = 100
KP = 80.0      # raised again for the lift-to-vertical attempt; gravity torque
KD = 2.0       # grows as the arm rises, so tracking degrades with angle.
MAX_DEGREES = 90.0


def deg(r): return r * 180.0 / math.pi
def rad(d): return d * math.pi / 180.0
def ease(p): return (1.0 - math.cos(math.pi * p)) / 2.0


class LowCmdArm:
    def __init__(self, iface="auto"):
        self.iface = iface
        self.state = None
        self._pub = None
        self._sub = None
        self._crc = None

    @property
    def mode_machine(self):
        return int(getattr(self.state, "mode_machine", 0)) if self.state else 0

    def connect(self, timeout=8.0):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        from r1vision.unitree_dds import init_channel

        init_channel(self.iface, peers=None)
        self._crc = CRC()
        # Held on the instance: a subscriber that goes out of scope is garbage
        # collected and the readings silently freeze.
        self._sub = ChannelSubscriber(TOPIC_STATE, LowState_)
        self._sub.Init(lambda m: setattr(self, "state", m), 10)
        self._pub = ChannelPublisher(TOPIC_CMD, LowCmd_)
        self._pub.Init()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.state is None:
            time.sleep(0.1)
        return self.state is not None

    def angles(self):
        return [m.q for m in self.state.motor_state] if self.state else []

    def send(self, targets: dict[int, float], kp_scale: float = 1.0):
        """Drive only the listed joints. Everything else stays disabled.

        `kp_scale` (0..1) scales stiffness and damping. rt/lowcmd has no blend
        weight -- that was an arm_sdk concept -- so ramping the gains is how
        authority is taken gradually instead of the arm snapping to its target
        the instant control begins. kp_scale=0 means the joints are commanded
        with no stiffness, i.e. effectively free.
        """
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        s = max(0.0, min(1.0, kp_scale))
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_machine = self.mode_machine
        cmd.mode_pr = 0
        for idx, q in targets.items():
            lo, hi = LIMITS[idx]
            m = cmd.motor_cmd[idx]
            m.mode = 1                      # 0 would leave the joint disabled
            m.q = max(lo, min(hi, q))       # never command past the manual's limit
            m.dq, m.kp, m.kd, m.tau = 0.0, KP * s, KD * s, 0.0
        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--hold", action="store_true",
                    help="hold the current pose; proves the motors energise")
    ap.add_argument("--lift", action="store_true", help="raise both shoulders")
    ap.add_argument("--degrees", type=float, default=15.0,
                    help="shoulder pitch change (sign unverified -- try small first)")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    if not (args.hold or args.lift):
        ap.print_help()
        return 1
    if abs(args.degrees) > MAX_DEGREES:
        print(f"error: --degrees capped at +-{MAX_DEGREES}", file=sys.stderr)
        return 1

    arm = LowCmdArm(args.iface)
    print(f"connecting on {args.iface} ...")
    if not arm.connect():
        print(f"error: no {TOPIC_STATE}. Is the robot powered and on ethernet?",
              file=sys.stderr)
        return 1
    print(f"connected. mode_machine={arm.mode_machine}")

    joints = RIGHT_ARM + LEFT_ARM
    print("\n  current arm angles:")
    for side, sp in SHOULDER_PITCH.items():
        print(f"    {side:5s} shoulder pitch = {deg(arm.angles()[sp]):+7.2f} deg")

    print("\n  ROBOT MUST BE LYING DOWN. Arms clear? Starting in 5s, Ctrl-C aborts")
    try:
        for i in (5, 4, 3, 2, 1):
            print(f"    {i}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  aborted; nothing sent.")
        return 130

    # Captured AFTER the countdown, not before: an earlier version sampled the
    # pose first, so anything that shifted the arm during those five seconds
    # became a target the robot then drove back to -- real commanded motion
    # that read as "drift" and made the result impossible to interpret.
    a = arm.angles()
    held = {j: a[j] for j in joints}

    period = 1.0 / CONTROL_HZ
    try:
        # ---- hold: the honest test of whether lowcmd energises anything ----
        print("\n  holding current pose for 3s -- the arms should stiffen ...")
        drift_start = {j: arm.angles()[j] for j in joints}
        for _ in range(3 * CONTROL_HZ):
            arm.send(held)
            time.sleep(period)
        drift = max(abs(deg(arm.angles()[j] - drift_start[j])) for j in joints)
        print(f"  max drift while held: {drift:.2f} deg")
        print("  (try nudging an arm by hand now -- stiff means lowcmd works)")

        if not args.lift:
            print("\n  done. If the arms stiffened, next: --lift --degrees 15")
            return 0

        # ---- lift ----
        targets = {s: a[j] for s, j in SHOULDER_PITCH.items()}
        print(f"\n  lifting both shoulders by {args.degrees:+.1f} deg "
              f"over {args.seconds:.1f}s ...")
        steps = int(args.seconds * CONTROL_HZ)
        for i in range(steps):
            f = ease(i / steps)
            cmd = dict(held)
            for side, j in SHOULDER_PITCH.items():
                cmd[j] = targets[side] + rad(args.degrees) * f
            arm.send(cmd)
            if i % 50 == 0:
                m = arm.angles()
                print(f"    t={i / CONTROL_HZ:4.1f}s  "
                      f"R cmd {deg(targets['right'] + rad(args.degrees) * f):+7.2f} "
                      f"meas {deg(m[22]):+7.2f}   "
                      f"L meas {deg(m[15]):+7.2f}", flush=True)
            time.sleep(period)

        moved = max(abs(deg(arm.angles()[j] - a[j]))
                    for j in SHOULDER_PITCH.values())
        print(f"\n  max shoulder movement: {moved:.2f} deg "
              f"(commanded {abs(args.degrees):.1f})")

        print("  holding 2s, then returning ...")
        for _ in range(2 * CONTROL_HZ):
            cmd = dict(held)
            for side, j in SHOULDER_PITCH.items():
                cmd[j] = targets[side] + rad(args.degrees)
            arm.send(cmd)
            time.sleep(period)
        for i in range(steps):
            f = ease(i / steps)
            cmd = dict(held)
            for side, j in SHOULDER_PITCH.items():
                cmd[j] = targets[side] + rad(args.degrees) * (1.0 - f)
            arm.send(cmd)
            time.sleep(period)

        print(f"\n  VERDICT: {'ARMS MOVED - lowcmd works' if moved > 3.0 else 'no motion'}")
        return 0

    except KeyboardInterrupt:
        print("\n  interrupted -- holding, then releasing.")
        for _ in range(int(0.5 * CONTROL_HZ)):
            arm.send(held)
            time.sleep(period)
        return 130
    finally:
        # Stop driving: publishing nothing leaves the last frame as the robot's
        # most recent instruction, so fade out by disabling the joints.
        print("  releasing (joints disabled) ...")
        for _ in range(int(0.5 * CONTROL_HZ)):
            arm.send({})
            time.sleep(period)


if __name__ == "__main__":
    sys.exit(main())
