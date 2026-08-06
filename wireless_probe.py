#!/usr/bin/env python3
"""Watch what the real Unitree remote sends. READ ONLY -- moves nothing.

    python wireless_probe.py

Then pick up the Unitree controller and move its sticks and buttons.

Why this is the thing to look at
-------------------------------
`loco_probe.py` found that the R1's `sport` service answers `GetFsmId` but
returns "API not implemented" for every other getter (fsm mode, balance mode,
swing height, stand height). A service that implements one call out of five is
not the G1 locomotion API this code was written against, so `SetVelocity` very
likely is not there either -- which is exactly what "sends fine, robot ignores
it" looks like.

Meanwhile the robot **subscribes** to `rt/wirelesscontroller`
(`unitree_go::msg::dds_::WirelessController_`), and that message is a gamepad:

    lx, ly, rx, ry : float32   stick axes
    keys           : uint16    button bitmask

That is the channel the handheld remote drives, and `ai` mode -- which
`loco_probe.py` shows currently owns the joints -- is the controller listening to
it. Publishing to it is not a workaround; it is how the robot is designed to be
driven, and it is what "walks like the Unitree controller" actually means.

The robot publishes this topic as well as subscribing to it, so this can observe
the genuine remote. Move its sticks and read off the real ranges, signs and key
codes rather than guessing them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

TOPIC = "rt/wirelesscontroller"

# Unitree's remote layout, as used across their robots. Printed as a guide; what
# the robot actually sends is what counts, and this script shows that.
KEY_BITS = {
    0: "R1", 1: "L1", 2: "start", 3: "select",
    4: "R2", 5: "L2", 6: "F1", 7: "F2",
    8: "A", 9: "B", 10: "X", 11: "Y",
    12: "up", 13: "right", 14: "down", 15: "left",
}


def decode(keys: int) -> str:
    names = [n for bit, n in KEY_BITS.items() if keys & (1 << bit)]
    return "+".join(names) if names else "-"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    from r1vision.unitree_dds import (default_peers, init_channel,
                                      quiet_dds_stderr)

    with quiet_dds_stderr():
        # Unicast peers, not multicast-only. Every topic that has actually
        # delivered on this network needed them; peers=None here would report
        # an empty remote as convincingly as a broken one.
        init_channel(args.iface, peers=default_peers())
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

        # Callback rather than sub.Read(): the polling path prints
        # "[Reader] take sample error" once per empty poll, which buries the
        # actual readings under noise and looks like a decode failure when it
        # is only an idle topic.
        inbox: list = []
        sub = ChannelSubscriber(TOPIC, WirelessController_)
        sub.Init(lambda m: inbox.append(m), 50)

        print(f"listening on {TOPIC} for {args.seconds:.0f}s\n")
        print("  Move the Unitree remote's sticks and press its buttons.")
        print("  Nothing here sends anything to the robot.\n")
        print(f"  {'lx':>8} {'ly':>8} {'rx':>8} {'ry':>8}  keys")

        deadline = time.monotonic() + args.seconds
        seen = 0
        last = None
        while time.monotonic() < deadline:
            if not inbox:
                time.sleep(0.02)
                continue
            msg = inbox.pop(0)
            seen += 1
            now = (round(msg.lx, 3), round(msg.ly, 3),
                   round(msg.rx, 3), round(msg.ry, 3), msg.keys)
            # Only changes, because an idle remote publishes the same zeros
            # forever and that scrolls the interesting part off the screen.
            if now != last:
                last = now
                print(f"  {msg.lx:>8.3f} {msg.ly:>8.3f} {msg.rx:>8.3f} "
                      f"{msg.ry:>8.3f}  0x{msg.keys:04x} {decode(msg.keys)}")

    print(f"\n  {seen} messages seen.")
    if seen == 0:
        print("  Nothing published. Either the remote is off, or this robot")
        print("  does not bridge it onto DDS. If the remote does drive the")
        print("  robot while this shows nothing, the walking commands travel")
        print("  by some other path and emulating the remote will not work.")
        return 1
    print("  The channel is live, so publishing to it should drive the robot")
    print("  the same way the remote does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
