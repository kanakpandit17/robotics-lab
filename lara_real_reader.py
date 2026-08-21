"""WSL side of the LARA sim<->real bridge -- the neurapy half, mirroring maira_sim_bridge.py's
own docstring convention for MAiRA. Runs in WSL, in the existing neura venv:

    source /home/aimat/neura/api_ws/venv/bin/activate
    python3 lara_real_reader.py

Reads LARA's live joint angles via the official SDK (the same call already verified against
the real robot: Robot().get_current_joint_angles()) and writes them to a shared JSON file that
lara_sim_bridge.py (the Windows/Isaac Sim side) polls:

    lara_bridge.json = {"real_joints": [...6...], "at": <unix time>}

This script only ever READS from the robot -- get_current_joint_angles(), is_robot_in_*_mode().
It never calls move_joint or anything else that could command motion, which is what makes it
safe to run continuously alongside whatever else is using the robot.
"""
from __future__ import annotations

import argparse
import json
import time

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bridge", default="/mnt/c/Users/aimat/Desktop/humanoid/lara_bridge.json",
                help="shared bridge file, same one lara_sim_bridge.py reads (WSL path -- "
                     "the Windows side sees this as C:\\Users\\aimat\\Desktop\\humanoid\\lara_bridge.json)")
ap.add_argument("--hz", type=float, default=10.0, help="poll rate")
args = ap.parse_args()


def write_bridge(d: dict) -> None:
    # Direct write, not write-then-os.replace(): maira_sim_bridge.py's own write_bridge()
    # notes os.replace() fails with "Access denied" when the other side (across the
    # WSL/Windows boundary) has the file open for reading at that instant. A transient
    # failure here just means this tick's write is skipped; the next one, ~100ms later,
    # supersedes it, so there is nothing worth retrying immediately.
    try:
        with open(args.bridge, "w") as f:
            json.dump(d, f)
            f.flush()
    except (PermissionError, OSError):
        pass


def main() -> int:
    from neurapy.robot import Robot

    print(f"[LARA-READ] connecting (default socket_address, confirmed 192.168.2.13) ...",
          flush=True)
    robot = Robot()
    print(f"[LARA-READ] connected: {getattr(robot, 'robot_name', '?')}, "
          f"dof={getattr(robot, 'dof', '?')}", flush=True)
    print(f"[LARA-READ] writing to {args.bridge} at {args.hz:.0f} Hz. Ctrl-C to stop.",
          flush=True)

    period = 1.0 / args.hz
    last_print = 0.0
    sent = 0
    errors = 0
    last_joints: list[float] | None = None
    try:
        while True:
            t0 = time.monotonic()
            try:
                last_joints = list(robot.get_current_joint_angles())
                write_bridge({"real_joints": last_joints, "at": time.time()})
                sent += 1
                errors = 0
            except Exception as exc:
                errors += 1
                # A read failing occasionally (a request landing mid-motion-planner-busy,
                # say) is not worth stopping over; failing every single tick for a while
                # is worth knowing about loudly rather than silently spinning forever.
                if errors == 1 or errors % 50 == 0:
                    print(f"[LARA-READ] read failed ({errors} in a row): "
                          f"{type(exc).__name__}: {exc}", flush=True)

            now = time.monotonic()
            if now - last_print > 2.0:
                last_print = now
                shown = [round(j, 3) for j in last_joints] if last_joints else "?"
                print(f"[LARA-READ] sent={sent} last_joints={shown}", flush=True)

            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[LARA-READ] stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
