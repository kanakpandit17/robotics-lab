#!/usr/bin/env python3
"""Talk to the fleet from the laptop. Also the worked example of the API.

    python -m scheduler.cli --config config/laptop.json list
    python -m scheduler.cli --config config/laptop.json state maira-01
    python -m scheduler.cli --config config/laptop.json exec maira-01 PickObject object=battery
    python -m scheduler.cli --config config/laptop.json stop maira-01
    python -m scheduler.cli --config config/laptop.json demo maira-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fleetcore.errors import RobotError                        # noqa: E402
from fleetcore.transport import build_transport                # noqa: E402

from scheduler.fleet import Fleet                              # noqa: E402


def parse_params(pairs: list[str]) -> dict:
    """`object=battery speed=0.2` -> {"object": "battery", "speed": 0.2}."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"parameter must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)      # numbers, booleans, lists
        except json.JSONDecodeError:
            out[key] = value                  # plain string
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show every robot on the bus")
    p_state = sub.add_parser("state"); p_state.add_argument("robot")
    p_stop = sub.add_parser("stop"); p_stop.add_argument("robot")
    p_demo = sub.add_parser("demo"); p_demo.add_argument("robot")
    p_exec = sub.add_parser("exec")
    p_exec.add_argument("robot")
    p_exec.add_argument("command")
    p_exec.add_argument("params", nargs="*")
    p_exec.add_argument("--timeout", type=float, default=120.0)

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    cfg = json.loads(Path(args.config).read_text())
    transport = build_transport(cfg.get("broker", {}), client_id="scheduler-cli")

    with Fleet(transport) as fleet:
        time.sleep(cfg.get("discovery_s", 1.5))   # let retained state arrive

        if args.cmd == "list":
            if not fleet.robots:
                print("no robots on the bus")
                return 1
            print(f"{'ROBOT':<16} {'MODEL':<16} {'STATE':<9} {'TASK':<14} CAPABILITIES")
            for r in sorted(fleet.robots, key=lambda x: x.robot_id):
                mark = "" if r.online else "  (offline)"
                print(f"{r.robot_id:<16} {r.model:<16} {r.state.value:<9} "
                      f"{str(r.current_task or '-'):<14} "
                      f"{','.join(r.capabilities)}{mark}")
            return 0

        try:
            robot = fleet.wait_for(args.robot, timeout=cfg.get("discovery_s", 10))
        except RobotError as exc:
            print(f"error: {exc}")
            return 1

        if args.cmd == "state":
            print(json.dumps(robot.get_state(), indent=2))
            return 0

        if args.cmd == "stop":
            print(json.dumps(robot.stop().result, indent=2))
            return 0

        if args.cmd == "demo":
            # The whole point of the architecture in five lines: no SDK, no
            # ROS, no robot-specific branching.
            for command, params in (("MoveHome", {}),
                                    ("PickObject", {"object": "battery"}),
                                    ("PlaceObject", {"location": "tray"}),
                                    ("MoveHome", {})):
                print(f"  -> {command} {params or ''}")
                try:
                    reply = robot.execute(command, **params)
                    print(f"     ok: {reply.result}")
                except RobotError as exc:
                    print(f"     FAILED: {exc}  (retryable={exc.retryable})")
                    return 1
            return 0

        if args.cmd == "exec":
            try:
                reply = robot.execute(args.command, timeout_s=args.timeout,
                                      **parse_params(args.params))
            except RobotError as exc:
                print(f"FAILED: {exc}   retryable={exc.retryable}")
                if exc.detail:
                    print(json.dumps(exc.detail, indent=2))
                return 1
            print(json.dumps(reply.result, indent=2))
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
