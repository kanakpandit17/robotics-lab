"""Record the LARA -> cage -> MAiRA route by walking the robot along it.

Unitree's SLAM service is not available on this machine: it runs on the Jetson
NX at 192.168.123.164 and needs the Mid-360 at 192.168.123.120, and neither
host is on the network (see pose_probe.py). What *is* available is
`rt/odommodestate`, which publishes a fused leg-odometry + IMU position at
~500 Hz. That is dead reckoning, so it drifts -- but it is a real position, and
a route taught by walking is exactly what dead reckoning is good at over a
single short trip.

Usage -- drive with the handheld controller, tap SPACE at each waypoint:

    .venv/bin/python teach_path.py lara_out D_pre D_mid D_post M_appr maira_in

Names are consumed in order, so you never type while holding the controller.
Extra marks past the last name are auto-numbered. Press q to finish and save.

About drift
-----------
Walk the robot back to its exact starting spot before pressing q. The closure
error printed at the end is the honest measure of how much this odometry drifts
over your route; it is the number that decides whether replay is viable or
whether you need visual landmarks. Do not skip it.
"""

from __future__ import annotations

import json
import math
import select
import sys
import termios
import time
import tty
from datetime import datetime

TOPIC = "rt/odommodestate"
SAMPLE_HZ = 20.0        # trajectory log rate; the topic itself runs ~500 Hz
OUT_DEFAULT = "paths/route.json"

# Per-sample movement below this is sensor jitter, not travel. The estimator's
# median step is ~0.14 mm per 50 ms; summing those over a long recording
# manufactured 17.66 m of "path" from a robot that never left a 2.5 m box.
# Walking produces 15-25 mm per 50 ms, so 3 mm separates the two cleanly.
NOISE_FLOOR_M = 0.003


class RawTerminal:
    """Read single keypresses without Enter, and always restore the tty."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def yaw_to_quat(yaw: float) -> tuple[float, float]:
    """Flat-floor quaternion: only rotation about Z survives."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def pose_of(msg) -> dict:
    p = msg.position
    rpy = msg.imu_state.rpy
    qz, qw = yaw_to_quat(rpy[2])
    return {
        "x": round(p[0], 4), "y": round(p[1], 4), "z": round(p[2], 4),
        "yaw": round(rpy[2], 4), "yaw_deg": round(math.degrees(rpy[2]), 1),
        "q_x": 0.0, "q_y": 0.0, "q_z": round(qz, 6), "q_w": round(qw, 6),
    }


def main(argv: list[str]) -> int:
    names = [a for a in argv if not a.startswith("-")]
    out_path = OUT_DEFAULT
    for i, a in enumerate(argv):
        if a in ("-o", "--out") and i + 1 < len(argv):
            out_path = argv[i + 1]
            names = [n for n in names if n != out_path]

    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    from r1vision.unitree_dds import default_peers, init_channel, robot_iface

    iface = robot_iface("auto")
    init_channel(iface, peers=default_peers())

    latest: dict = {}

    def on_msg(msg):
        latest["msg"] = msg

    sub = ChannelSubscriber(TOPIC, SportModeState_)
    sub.Init(on_msg, 10)

    print(f"waiting for {TOPIC} on {iface} ...", end="", flush=True)
    deadline = time.monotonic() + 10.0
    while "msg" not in latest and time.monotonic() < deadline:
        time.sleep(0.1)
    if "msg" not in latest:
        print("\n  nothing arrived. Is the robot powered and past R2+A?")
        return 1
    print(" ok\n")

    print("  SPACE / ENTER  mark a waypoint")
    print("  u              undo last waypoint")
    print("  q              finish and save")
    print("\n  Walk the robot with the handheld controller.")
    print("  Return it to the START spot before pressing q.\n")

    trajectory: list[list[float]] = []
    waypoints: list[dict] = []
    t0 = time.monotonic()
    start_pose = pose_of(latest["msg"])
    last_sample = 0.0
    path_len = 0.0
    prev_xy = (start_pose["x"], start_pose["y"])
    bbox = [start_pose["x"], start_pose["x"], start_pose["y"], start_pose["y"]]

    with RawTerminal() as term:
        while True:
            now = time.monotonic() - t0
            msg = latest.get("msg")
            if msg is None:
                time.sleep(0.05)
                continue
            pose = pose_of(msg)

            if now - last_sample >= 1.0 / SAMPLE_HZ:
                last_sample = now
                trajectory.append([round(now, 2), pose["x"], pose["y"],
                                   pose["z"], pose["yaw"]])
                step = math.hypot(pose["x"] - prev_xy[0],
                                  pose["y"] - prev_xy[1])
                if step > NOISE_FLOOR_M:
                    path_len += step
                prev_xy = (pose["x"], pose["y"])
                bbox[0] = min(bbox[0], pose["x"]); bbox[1] = max(bbox[1], pose["x"])
                bbox[2] = min(bbox[2], pose["y"]); bbox[3] = max(bbox[3], pose["y"])

            drift = math.hypot(pose["x"] - start_pose["x"],
                               pose["y"] - start_pose["y"])
            sys.stdout.write(
                f"\r  x={pose['x']:+7.3f}  y={pose['y']:+7.3f}  "
                f"yaw={pose['yaw_deg']:+7.1f}deg   "
                f"walked={path_len:6.2f}m  from_start={drift:5.2f}m  "
                f"marks={len(waypoints)}   ")
            sys.stdout.flush()

            k = term.key()
            if k is None:
                time.sleep(0.02)
                continue
            if k in (" ", "\r", "\n"):
                idx = len(waypoints)
                name = names[idx] if idx < len(names) else f"wp{idx + 1:02d}"
                wp = {"name": name, "t": round(now, 2), **pose}
                waypoints.append(wp)
                sys.stdout.write(
                    f"\r  marked {name:<12} x={pose['x']:+.3f} "
                    f"y={pose['y']:+.3f} yaw={pose['yaw_deg']:+.1f}deg"
                    + " " * 24 + "\n")
                sys.stdout.flush()
            elif k == "u" and waypoints:
                gone = waypoints.pop()
                sys.stdout.write(f"\r  undid {gone['name']}"
                                 + " " * 56 + "\n")
                sys.stdout.flush()
            elif k in ("q", "\x03"):
                break

    end_pose = pose_of(latest["msg"])
    closure = math.hypot(end_pose["x"] - start_pose["x"],
                         end_pose["y"] - start_pose["y"])
    yaw_closure = math.degrees(
        math.atan2(math.sin(end_pose["yaw"] - start_pose["yaw"]),
                   math.cos(end_pose["yaw"] - start_pose["yaw"])))

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    doc = {
        "recorded": datetime.now().isoformat(timespec="seconds"),
        "source": f"{TOPIC} (leg odometry + IMU -- dead reckoning, drifts)",
        "frame": "odometry origin = wherever the robot booted; NOT a fixed "
                 "map frame. Re-teach after every power cycle.",
        "duration_s": round(time.monotonic() - t0, 1),
        "path_length_m": round(path_len, 2),
        "closure_error_m": round(closure, 3),
        "closure_yaw_deg": round(yaw_closure, 1),
        "start_pose": start_pose,
        "waypoints": waypoints,
        "trajectory": trajectory,
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)

    print("\n" + "=" * 68)
    print(f"  saved            {out_path}")
    print(f"  waypoints        {len(waypoints)}")
    print(f"  path length      {path_len:.2f} m")
    print(f"  trajectory       {len(trajectory)} samples")
    print(f"  closure error    {closure:.3f} m   ({yaw_closure:+.1f} deg yaw)")
    print("=" * 68)

    span = math.hypot(bbox[1] - bbox[0], bbox[3] - bbox[2])
    print(f"  area covered     {bbox[1]-bbox[0]:.2f} x {bbox[3]-bbox[2]:.2f} m"
          f"  (diagonal {span:.2f} m)")

    # A route between two stations must range further than a couple of metres.
    # If it did not, the estimator is not following the robot at all and every
    # number above is meaningless -- say so instead of reporting a drift rate.
    if span < 4.0:
        print("\n  !! The recorded positions never left a "
              f"{span:.1f} m area.")
        print("  If you walked further than that, this estimator is not\n"
              "  tracking the robot and the route is unusable. rt/odommodestate\n"
              "  on this firmware is drifting IMU integration, not leg\n"
              "  odometry -- foot_force and body_height are never populated.\n"
              "  Do not build waypoints on it. See fsm_watch.py.\n")
        return 0

    if path_len > 1.0:
        pct = 100.0 * closure / path_len
        print(f"\n  drift = {pct:.1f}% of distance walked.")
        if pct < 2:
            print("  Good. Odometry replay over this route is realistic.")
        elif pct < 5:
            print("  Usable for coarse motion, but the final approach to each\n"
                  "  station will need a visual landmark to correct against.")
        else:
            print("  Too much to navigate on alone. Use the taught path for\n"
                  "  shape and sequencing, and correct at each station with a\n"
                  "  camera landmark (AprilTag) rather than trusting position.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
