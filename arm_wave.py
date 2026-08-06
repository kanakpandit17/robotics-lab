#!/usr/bin/env python3
"""Raise the R1's right arm ~30 degrees, hold, and lower it back.

    python arm_wave.py --identify          # watch joint angles, move nothing
    python arm_wave.py --joint 22          # DRY RUN: print the trajectory
    python arm_wave.py --joint 22 --go     # actually move the arm

Nothing moves without --go. That is deliberate.

WHY --joint IS REQUIRED
-----------------------
Joint indices differ between Unitree models. The published layouts are for the
G1 (29 DOF, right shoulder pitch at index 22); the R1 is a different robot with
a different count, and no public index table. Commanding a guessed index on a
25 kg humanoid could drive a leg or the waist instead of the arm. So this
script will not guess: run --identify, move the right arm by hand, and read off
which index changes.

SAFETY
------
  * clear the space around the right arm before --go
  * keep the e-stop in reach
  * robot in standing lock / damp mode
  * Ctrl-C at any point ramps back to the starting pose rather than cutting
    power, which would let the arm drop

The motion is a slow cosine ramp with a low-gain PD hold, and the amplitude is
capped at 60 degrees regardless of what you pass.
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

MAX_DEGREES = 60.0        # hard cap, whatever the user asks for
CONTROL_HZ = 100          # Unitree low-level loops expect ~100-500 Hz
DEFAULT_KP = 40.0         # low gains: compliant, so a collision does not fight
DEFAULT_KD = 1.0


def deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def rad(degrees: float) -> float:
    return degrees * math.pi / 180.0


class ArmController:
    """Thin wrapper over the Unitree DDS low-level interface."""

    def __init__(self, iface: str, topic: str = "rt/arm_sdk", peer: str | None = None):
        self.iface = iface
        self.topic = topic
        self.peer = peer
        self.publisher = None
        self.state = None
        self._subscriber = None

    def connect(self, timeout: float = 5.0) -> bool:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        from r1vision.unitree_dds import init_channel

        # The stock initialiser is multicast-only; the robot is a routed hop
        # away on 192.168.123.x, so it needs an explicit unicast peer.
        init_channel(self.iface, peers=[self.peer] if self.peer else None)

        def on_state(msg):
            self.state = msg

        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(on_state, 10)

        self.publisher = ChannelPublisher(self.topic, LowCmd_)
        self.publisher.Init()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.state is None:
            time.sleep(0.1)
        return self.state is not None

    def current_angles(self) -> list[float]:
        if self.state is None:
            return []
        return [m.q for m in self.state.motor_state]


def identify(controller: ArmController, seconds: int) -> None:
    """Stream joint angles so the arm joints can be found by moving them."""
    print("Move the RIGHT ARM slowly by hand. Indices that change are the ones.")
    print("Ctrl-C to stop.\n")

    baseline = controller.current_angles()
    if not baseline:
        print("No lowstate data -- cannot identify joints.")
        return

    print(f"{len(baseline)} joints reporting.\n")
    peak = [0.0] * len(baseline)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            now = controller.current_angles()
            for i, (base, current) in enumerate(zip(baseline, now)):
                peak[i] = max(peak[i], abs(current - base))
            movers = sorted(
                (p, i) for i, p in enumerate(peak) if p > rad(3)
            )[::-1][:8]
            line = "  ".join(f"j{i}:{deg(p):+6.1f}" for p, i in movers)
            print(f"\r  moved: {line:<80}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print("\n\nRanked by how far each joint moved:")
    for p, i in sorted(((p, i) for i, p in enumerate(peak)), reverse=True)[:10]:
        if p > rad(1):
            print(f"  joint {i:>3} : {deg(p):6.1f} deg")
    print("\nThe largest mover for a shoulder raise is your --joint value.")


def build_trajectory(start: float, target: float, hold_s: float,
                     travel_s: float) -> list[tuple[float, float]]:
    """(time, angle) pairs: ramp up, hold, ramp down. Cosine easing."""
    points = []
    steps = int(travel_s * CONTROL_HZ)
    for i in range(steps):
        phase = i / steps
        # Cosine ease removes the acceleration step that a linear ramp would
        # put through the gearbox at both ends.
        eased = (1 - math.cos(math.pi * phase)) / 2
        points.append((i / CONTROL_HZ, start + (target - start) * eased))
    t0 = travel_s
    for i in range(int(hold_s * CONTROL_HZ)):
        points.append((t0 + i / CONTROL_HZ, target))
    t0 += hold_s
    for i in range(steps):
        phase = i / steps
        eased = (1 - math.cos(math.pi * phase)) / 2
        points.append((t0 + i / CONTROL_HZ, target + (start - target) * eased))
    points.append((t0 + travel_s, start))
    return points


def run_motion(controller: ArmController, joint: int, amplitude_deg: float,
               hold_s: float, travel_s: float, kp: float, kd: float,
               dry_run: bool) -> int:
    angles = controller.current_angles()
    if not angles:
        print("error: no joint state; refusing to command blind.", file=sys.stderr)
        return 1
    if joint >= len(angles):
        print(f"error: joint {joint} out of range (robot reports {len(angles)})",
              file=sys.stderr)
        return 1

    start = angles[joint]
    target = start + rad(amplitude_deg)
    traj = build_trajectory(start, target, hold_s, travel_s)

    print(f"  joint      : {joint}")
    print(f"  start      : {deg(start):+.2f} deg")
    print(f"  target     : {deg(target):+.2f} deg  ({amplitude_deg:+.1f} relative)")
    print(f"  duration   : {traj[-1][0]:.1f}s over {len(traj)} points at {CONTROL_HZ} Hz")
    print(f"  gains      : kp={kp} kd={kd}")

    if dry_run:
        print("\n  DRY RUN -- nothing sent. Sample of the trajectory:")
        for t, q in traj[:: max(1, len(traj) // 12)]:
            bar = "#" * int((q - start) / max(rad(amplitude_deg), 1e-6) * 40)
            print(f"    t={t:5.2f}s  {deg(q):+7.2f} deg  {bar}")
        print("\n  Add --go to execute.")
        return 0

    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    print("\n  MOVING in 3s -- Ctrl-C aborts ...")
    for i in (3, 2, 1):
        print(f"    {i}")
        time.sleep(1)

    cmd = LowCmd_()
    period = 1.0 / CONTROL_HZ
    last = start
    try:
        for _, q in traj:
            # arm_sdk uses a weight at the tail slot to blend authority in;
            # 1.0 hands the listed joints to us and leaves the rest to the
            # robot's own balance controller.
            try:
                cmd.motor_cmd[len(cmd.motor_cmd) - 1].q = 1.0
            except (IndexError, AttributeError):
                pass
            m = cmd.motor_cmd[joint]
            m.q, m.dq, m.kp, m.kd, m.tau = q, 0.0, kp, kd, 0.0
            controller.publisher.Write(cmd)
            last = q
            time.sleep(period)
        print("  done -- returned to start.")
        return 0
    except KeyboardInterrupt:
        print("\n  aborted; ramping back to start ...")
        steps = int(1.5 * CONTROL_HZ)
        for i in range(steps):
            q = last + (start - last) * (i / steps)
            m = cmd.motor_cmd[joint]
            m.q, m.dq, m.kp, m.kd, m.tau = q, 0.0, kp, kd, 0.0
            controller.publisher.Write(cmd)
            time.sleep(period)
        print("  back at start.")
        return 130


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--peer", default="192.168.123.161",
                    help="robot's locomotion computer (needs the route added)")
    ap.add_argument("--joint", type=int, help="joint index (find it with --identify)")
    ap.add_argument("--degrees", type=float, default=30.0)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds at the top")
    ap.add_argument("--travel", type=float, default=2.0, help="seconds each way")
    ap.add_argument("--kp", type=float, default=DEFAULT_KP)
    ap.add_argument("--kd", type=float, default=DEFAULT_KD)
    ap.add_argument("--topic", default="rt/arm_sdk")
    ap.add_argument("--identify", action="store_true", help="stream angles, move nothing")
    ap.add_argument("--identify-seconds", type=int, default=60)
    ap.add_argument("--go", action="store_true", help="actually move (default is dry run)")
    args = ap.parse_args()

    if abs(args.degrees) > MAX_DEGREES:
        print(f"error: --degrees capped at {MAX_DEGREES}", file=sys.stderr)
        return 1

    controller = ArmController(args.iface, args.topic, peer=args.peer)
    print(f"connecting on {args.iface} ...")
    try:
        connected = controller.connect()
    except ImportError as exc:
        print(f"error: Unitree SDK not importable: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not connected:
        print("error: no lowstate from the robot.", file=sys.stderr)
        print("The SDK control channel is not reachable. Run sdk_probe.py.",
              file=sys.stderr)
        return 1

    print(f"connected; {len(controller.current_angles())} joints reporting.\n")

    if args.identify:
        identify(controller, args.identify_seconds)
        return 0

    if args.joint is None:
        print("error: --joint is required. Run --identify first to find it.",
              file=sys.stderr)
        return 1

    return run_motion(controller, args.joint, args.degrees, args.hold,
                      args.travel, args.kp, args.kd, dry_run=not args.go)


if __name__ == "__main__":
    sys.exit(main())
