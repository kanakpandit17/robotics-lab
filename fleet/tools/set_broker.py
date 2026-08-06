#!/usr/bin/env python3
"""Point the configs at a broker. Run once per machine after copying.

    python tools/set_broker.py 141.70.221.99            # on the laptop
    python tools/set_broker.py 127.0.0.1 --agent-only   # on the workstation

The broker lives on the workstation, so the two sides disagree about its
address on purpose: the laptop dials the workstation's IP, the agent dials
loopback. Editing JSON by hand is the step people get wrong at 2am, hence a
command for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(path: Path, host: str, port: int, user: str | None,
          password: str | None) -> None:
    if not path.exists():
        print(f"  skip {path.name} (not present)")
        return
    cfg = json.loads(path.read_text())
    broker = cfg.setdefault("broker", {})
    broker["transport"] = "mqtt"
    broker["host"] = host
    broker["port"] = port
    if user:
        broker["username"] = user
        broker["password"] = password
    else:
        broker.pop("username", None)
        broker.pop("password", None)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  {path.name:<22} -> mqtt://{host}:{port}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", help="broker address as THIS machine reaches it")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--agent-only", action="store_true",
                    help="workstation: only touch the agent configs")
    ap.add_argument("--laptop-only", action="store_true",
                    help="laptop: only touch laptop.json")
    args = ap.parse_args()

    cfg_dir = ROOT / "config"
    targets: list[Path] = []
    if not args.agent_only:
        targets.append(cfg_dir / "laptop.json")
    if not args.laptop_only:
        targets += [cfg_dir / "maira-agent.json", cfg_dir / "r1-agent.json"]

    print(f"setting broker to {args.host}:{args.port}")
    for path in targets:
        patch(path, args.host, args.port, args.username, args.password)

    print("\nverify from this machine:")
    print(f"  python -c \"import socket;"
          f"s=socket.create_connection(('{args.host}',{args.port}),3);"
          f"print('broker reachable');s.close()\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
