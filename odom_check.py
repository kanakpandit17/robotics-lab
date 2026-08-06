"""Is `rt/odommodestate` position usable? Walk a measured straight line.

The first taught route came back inconsistent: over 483 s the reported position
never left a 2.5 x 1.8 m box, while the robot was actually walked between two
stations on opposite sides of a cage. Either the robot was not walking on its
own legs, or the position estimate is not integrating. Those need different
fixes, so measure rather than guess.

Procedure
---------
1. Mark the robot's starting spot on the floor (tape).
2. Mark a spot a known distance away in a straight line -- 5 m is plenty.
3. Run this with that distance, walk the robot from mark to mark, press q.

    .venv/bin/python odom_check.py 5.0

What it compares
----------------
* position delta   -- straight-line move the estimator reports
* integral of |v|  -- distance implied by the reported velocity
* your tape measure

If velocity integrates correctly but position does not, the estimator is
resetting or the frame is being republished. If velocity is also near zero
while the robot visibly walks, the locomotion controller is not feeding the
estimator at all -- which would be the same root cause as the mode-ownership
problem that makes SetVelocity return 127.
"""

from __future__ import annotations

import math
import sys
import time

from teach_path import RawTerminal

TOPIC = "rt/odommodestate"


def main(argv: list[str]) -> int:
    measured = float(argv[0]) if argv else 0.0

    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    from r1vision.unitree_dds import default_peers, init_channel, robot_iface

    init_channel(robot_iface("auto"), peers=default_peers())

    latest: dict = {}
    sub = ChannelSubscriber(TOPIC, SportModeState_)
    sub.Init(lambda m: latest.__setitem__("m", m), 10)

    print(f"waiting for {TOPIC} ...", end="", flush=True)
    deadline = time.monotonic() + 10.0
    while "m" not in latest and time.monotonic() < deadline:
        time.sleep(0.1)
    if "m" not in latest:
        print(" nothing arrived.")
        return 1
    print(" ok\n")

    if measured:
        print(f"  Walk the robot {measured:.2f} m in a straight line, then q.\n")
    else:
        print("  Walk the robot, then press q. "
              "(Pass the tape distance as an argument for a verdict.)\n")

    m0 = latest["m"]
    start = (m0.position[0], m0.position[1])
    start_yaw = m0.imu_state.rpy[2]

    modes: set[int] = set()
    gaits: set[int] = set()
    errors: set[int] = set()
    vel_integral = 0.0
    peak_speed = 0.0
    foot_min = [32767] * 4
    foot_max = [-32768] * 4
    n = 0
    t_prev = time.monotonic()
    t0 = t_prev

    with RawTerminal() as term:
        while True:
            time.sleep(0.05)
            msg = latest.get("m")
            if msg is None:
                continue
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            n += 1

            modes.add(int(msg.mode))
            gaits.add(int(msg.gait_type))
            errors.add(int(msg.error_code))
            vx, vy = msg.velocity[0], msg.velocity[1]
            speed = math.hypot(vx, vy)
            peak_speed = max(peak_speed, speed)
            vel_integral += speed * dt
            for i in range(4):
                f = int(msg.foot_force[i])
                foot_min[i] = min(foot_min[i], f)
                foot_max[i] = max(foot_max[i], f)

            px, py = msg.position[0], msg.position[1]
            moved = math.hypot(px - start[0], py - start[1])
            sys.stdout.write(
                f"\r  mode={int(msg.mode)} gait={int(msg.gait_type)} "
                f"err={int(msg.error_code)}  "
                f"pos=({px:+6.2f},{py:+6.2f})  moved={moved:5.2f}m  "
                f"|v|={speed:4.2f}  vint={vel_integral:5.2f}m  "
                f"h={msg.body_height:4.2f}   ")
            sys.stdout.flush()

            k = term.key()
            if k in ("q", "\x03"):
                break

    msg = latest["m"]
    end = (msg.position[0], msg.position[1])
    delta = math.hypot(end[0] - start[0], end[1] - start[1])
    dyaw = math.degrees(math.atan2(
        math.sin(msg.imu_state.rpy[2] - start_yaw),
        math.cos(msg.imu_state.rpy[2] - start_yaw)))
    elapsed = time.monotonic() - t0

    print("\n\n" + "=" * 66)
    print(f"  elapsed              {elapsed:.1f} s   ({n} samples)")
    print(f"  modes seen           {sorted(modes)}")
    print(f"  gait types seen      {sorted(gaits)}")
    print(f"  error codes seen     {sorted(errors)}")
    print(f"  peak speed           {peak_speed:.2f} m/s")
    print(f"  foot_force min/max   " +
          "  ".join(f"[{foot_min[i]},{foot_max[i]}]" for i in range(4)))
    print("-" * 66)
    print(f"  position delta       {delta:.2f} m")
    print(f"  velocity integral    {vel_integral:.2f} m")
    if measured:
        print(f"  tape measure         {measured:.2f} m")
    print(f"  yaw change           {dyaw:+.1f} deg")
    print("=" * 66 + "\n")

    if not measured:
        return 0

    scale = delta / measured if measured else 0.0
    print(f"  position / tape = {scale:.2f}")
    if scale > 0.85:
        print("  Odometry tracks walking. Teach-and-repeat is viable;\n"
              "  re-record the route and expect a few percent drift.")
    elif vel_integral > 0.5 * measured:
        print("  Velocity is reported but position is not accumulating it.\n"
              "  The estimator is resetting or being republished. Integrating\n"
              "  velocity yourself is a workaround worth testing.")
    elif peak_speed < 0.05:
        print("  The robot reported essentially no velocity. It was not\n"
              "  walking under its own locomotion controller -- carried or\n"
              "  pushed motion is invisible to leg odometry. If it *was*\n"
              "  walking, the controller is not feeding this topic, which is\n"
              "  the same root cause as SetVelocity returning 127.")
    else:
        print("  Motion detected but badly under-reported. Not usable for\n"
              "  navigation. Use camera landmarks instead of position.")
    if modes == {0}:
        print("\n  NOTE: mode stayed 0 the whole time -- the locomotion FSM\n"
              "  never entered a walking state. Confirm R2+A was pressed and\n"
              "  the robot is in operation control mode, not damping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
