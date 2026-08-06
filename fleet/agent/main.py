#!/usr/bin/env python3
"""Run the Robot Edge Agent. This is what starts on the MAiRA workstation.

    python -m agent.main --config config/maira-agent.json
    python -m agent.main --config config/maira-agent.json --simulate

`--simulate` swaps the vendor SDK for a stub while keeping every other code
path -- transport, executor, state machine, error handling -- identical. Use it
to build the scheduler before the robot is free, then drop the flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fleetcore import topics                                   # noqa: E402
from fleetcore.envelope import Presence                        # noqa: E402
from fleetcore.transport import build_transport                # noqa: E402

from agent.edge_agent import EdgeAgent                         # noqa: E402

ADAPTERS = {
    "maira": "agent.adapters.maira:MairaAdapter",
    "unitree_r1": "agent.adapters.unitree_r1:UnitreeR1Adapter",
    "simulator": "agent.adapters.simulator:SimulatorAdapter",
}


def load_adapter(kind: str, robot_id: str, config: dict):
    """Import by name so an agent only loads the SDK it actually needs.

    The workstation must not import Unitree's DDS stack to run MAiRA, and the
    laptop must not import either.
    """
    import importlib

    spec = ADAPTERS.get(kind)
    if spec is None:
        raise SystemExit(f"unknown adapter {kind!r}; known: {sorted(ADAPTERS)}")
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(robot_id, config)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--simulate", action="store_true",
                    help="run against a stub SDK; nothing physical moves")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S")

    cfg = json.loads(Path(args.config).read_text())
    robot_id = cfg["robot_id"]
    robot_cfg = dict(cfg.get("robot", {}))
    if args.simulate:
        robot_cfg["simulate"] = True

    adapter = load_adapter(cfg.get("adapter", "simulator"), robot_id, robot_cfg)

    # The will is registered at connect time, so the broker can announce this
    # agent's death even if it never gets to run another line of code.
    will = (topics.presence(robot_id), Presence(robot_id, False).to_bytes())
    transport = build_transport(cfg.get("broker", {}),
                                client_id=f"agent-{robot_id}", will=will)

    agent = EdgeAgent(robot_id, adapter, transport, cfg.get("agent", {}))

    def handle_signal(signum, _frame):
        logging.info("signal %s received", signum)
        agent._stop_flag.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    agent.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
