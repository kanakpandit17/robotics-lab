"""Enumerate every topic the R1 actually publishes.

Guessing topic names has cost us twice now: `rt/odommodestate` turned out to be
a partially-filled quadruple message whose `position` is drifting IMU
integration, not leg odometry. DDS already knows the answer -- every publisher
announces itself on the builtin DCPSPublication topic -- so ask it.

    .venv/bin/python dds_topics.py          # 10 s discovery
    .venv/bin/python dds_topics.py 25       # longer, for sleepy publishers

This talks to CycloneDDS directly rather than through the Unitree SDK. Per the
note in r1vision/lowstate_dump.py the two cannot share a process: the SDK
builds its own Domain(0, config) and a second DomainParticipant collides,
silently killing subscriptions. Hence a standalone script.
"""

from __future__ import annotations

import os
import sys
import time

CONFIG = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="{iface}" priority="default" multicast="default"/>
            </Interfaces>
            <AllowMulticast>true</AllowMulticast>
        </General>
        <Discovery>
            <ParticipantIndex>auto</ParticipantIndex>
            <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
            <Peers>
{peers}
            </Peers>
        </Discovery>
        <Tracing><Verbosity>none</Verbosity>
                 <OutputFile>/tmp/dds_topics.log</OutputFile></Tracing>
    </Domain>
</CycloneDDS>"""

# Topics whose name suggests they carry where the robot is.
POSE_HINTS = ("odom", "pose", "slam", "loc", "map", "nav", "state", "tf")


def main(argv: list[str]) -> int:
    seconds = float(argv[0]) if argv else 10.0

    from r1vision.unitree_dds import default_peers, robot_iface

    iface = robot_iface("auto")
    peers = "\n".join(f'                <Peer address="{p}"/>'
                      for p in default_peers())
    os.environ["CYCLONEDDS_URI"] = CONFIG.format(iface=iface, peers=peers)

    from cyclonedds.builtin import (BuiltinDataReader,
                                    BuiltinTopicDcpsPublication)
    from cyclonedds.domain import DomainParticipant

    dp = DomainParticipant(0)
    rd = BuiltinDataReader(dp, BuiltinTopicDcpsPublication)

    print(f"discovering on {iface} for {seconds:.0f}s ...\n")
    found: dict[str, str] = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for s in rd.take(N=200):
            name = getattr(s, "topic_name", None)
            if name and not name.startswith("DCPS"):
                found[name] = getattr(s, "type_name", "?")
        time.sleep(0.25)

    if not found:
        print("  Nothing discovered. Robot powered on? Cable in the R1 switch?")
        return 1

    interesting = {k: v for k, v in found.items()
                   if any(h in k.lower() for h in POSE_HINTS)}
    rest = {k: v for k, v in found.items() if k not in interesting}

    print(f"{len(found)} topics published\n")
    print("POSITION-RELATED (name suggests it may carry a pose)")
    print("-" * 78)
    for name in sorted(interesting):
        print(f"  {name:<42} {interesting[name]}")
    if not interesting:
        print("  (none)")

    print(f"\nEVERYTHING ELSE ({len(rest)})")
    print("-" * 78)
    for name in sorted(rest):
        print(f"  {name:<42} {rest[name]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
