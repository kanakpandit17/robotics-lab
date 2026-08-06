#!/usr/bin/env python3
"""Watch the robot's DDS services come up live. Read-only.

    python wait_for_services.py

Start this, then wake the robot from the app. Topics appear here as their
services start, so you can see exactly which state change brings the arm
service up -- instead of guessing and re-running a probe each time.

The signal to wait for is `rt/api/arm/response`. A service SERVER publishes
responses and subscribes to requests; a client does the reverse. Every scan so
far found only the client side (the app bridge), which is what a sleeping robot
looks like. When `rt/api/arm/response` appears, the real arm service is running
and `arm_actions.py --list` will work.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

from r1vision.unitree_dds import default_peers, quiet_dds_stderr  # noqa: E402

# Publishing these means the corresponding service SERVER is running.
SERVER_SIGNALS = {
    "rt/api/arm/response": "ARM service  <-- the one we need",
    "rt/api/sport/response": "sport/locomotion service",
    "rt/api/loco/response": "loco service",
    "rt/api/motion_switcher/response": "motion switcher",
    "rt/lowstate": "low-level state (motor feedback)",
}


def configure(iface: str, peers: list[str], verbose: bool = False) -> None:
    from r1vision.unitree_dds import DDS_LOG

    peer_xml = "".join(f"<Peer address='{p}'/>" for p in peers)
    # Unreachable peers log a write failure every cycle -- ~94 lines of noise
    # per few seconds. Send it to a file so the topic list stays readable.
    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS><Domain>"
        "<General>"
        f"<Interfaces><NetworkInterface name='{iface}' priority='default'/></Interfaces>"
        "<AllowMulticast>true</AllowMulticast>"
        "</General>"
        "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
        "<MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>"
        f"<Peers>{peer_xml}</Peers></Discovery>"
        f"<Tracing><Verbosity>{'severe' if verbose else 'none'}</Verbosity>"
        f"<OutputFile>{DDS_LOG}</OutputFile></Tracing>"
        "</Domain></CycloneDDS>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--minutes", type=float, default=10.0)
    args = ap.parse_args()

    configure(args.iface, default_peers())

    # Must be in place before the participant starts its threads, or the first
    # burst of write failures escapes to the terminal.
    import atexit
    _quiet = quiet_dds_stderr()
    _quiet.__enter__()
    atexit.register(lambda: _quiet.__exit__(None, None, None))

    from cyclonedds.builtin import (
        BuiltinDataReader,
        BuiltinTopicDcpsPublication,
        BuiltinTopicDcpsSubscription,
    )
    from cyclonedds.domain import DomainParticipant

    participant = DomainParticipant(0)
    r_pub = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    r_sub = BuiltinDataReader(participant, BuiltinTopicDcpsSubscription)

    print("Watching DDS. Wake the robot from the app now.")
    print("Ctrl-C to stop.\n")
    print(f"  waiting for: rt/api/arm/response\n")

    seen_pub: set[str] = set()
    seen_sub: set[str] = set()
    deadline = time.monotonic() + args.minutes * 60
    arm_up = False

    try:
        while time.monotonic() < deadline:
            for sample in r_pub.take(N=100):
                name = getattr(sample, "topic_name", None)
                if not name or name.startswith("DCPS") or name in seen_pub:
                    continue
                seen_pub.add(name)
                stamp = datetime.now().strftime("%H:%M:%S")
                note = SERVER_SIGNALS.get(name, "")
                flag = "  ***" if note else ""
                print(f"  {stamp}  PUB  {name:<42}{note}{flag}")
                if name == "rt/api/arm/response":
                    arm_up = True

            for sample in r_sub.take(N=100):
                name = getattr(sample, "topic_name", None)
                if not name or name.startswith("DCPS") or name in seen_sub:
                    continue
                seen_sub.add(name)
                stamp = datetime.now().strftime("%H:%M:%S")
                # The server side subscribes to requests.
                note = "  <-- a SERVER is listening" if name.endswith("/request") else ""
                print(f"  {stamp}  SUB  {name:<42}{note}")

            if arm_up:
                print("\n  ARM SERVICE IS UP.")
                print("  Run:  python arm_actions.py --list")
                return 0

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped.")

    print(f"\n  {len(seen_pub)} published / {len(seen_sub)} subscribed topics seen.")
    if not arm_up:
        print("  rt/api/arm/response never appeared -- the arm service did not start.")
    return 0 if arm_up else 1


if __name__ == "__main__":
    sys.exit(main())
