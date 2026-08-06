#!/usr/bin/env python3
"""What is the robot actually doing. READ ONLY -- moves nothing.

    python robot_state.py                       # 15s of live state
    python robot_state.py --seconds 60          # while you move the remote

Run it, then pick up the Unitree remote and move its sticks.

Why
---
Two things are unexplained: velocity commands do nothing, and arm actions that
worked earlier now come back `7406`. Both are the robot refusing, and neither
error says why. This reads the state the robot broadcasts about itself.

Where the remote actually is
----------------------------
`rt/wirelesscontroller` is silent on this unit even though the robot subscribes
to it. The remote's real position is carried inside `rt/lowstate` instead, as
`wireless_remote` -- 40 bytes holding the same sticks and key bitmask. Decoding
that shows what the handheld remote is sending, which is the reference for
driving the robot the same way.

    byte 0-1   0xFE 0xEF header      byte 4-7    lx float32
    byte 2-3   keys uint16           byte 8-11   rx float32
                                     byte 12-15  ry float32
                                     byte 16-19  L2 float32
                                     byte 20-23  ly float32

`mode_pr` and `mode_machine` are the robot's own view of which controller owns
it, and are what `7406` is most likely complaining about.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

KEY_BITS = {
    0: "R1", 1: "L1", 2: "start", 3: "select",
    4: "R2", 5: "L2", 6: "F1", 7: "F2",
    8: "A", 9: "B", 10: "X", 11: "Y",
    12: "up", 13: "right", 14: "down", 15: "left",
}


def decode_remote(blob) -> dict | None:
    """Unpack the 40-byte wireless_remote struct out of LowState."""
    raw = bytes(bytearray(blob))
    if len(raw) < 24 or raw[0] != 0xFE or raw[1] != 0xEF:
        return None
    keys = struct.unpack_from("<H", raw, 2)[0]
    lx, rx, ry, l2, ly = struct.unpack_from("<5f", raw, 4)
    return {"keys": keys, "lx": lx, "ly": ly, "rx": rx, "ry": ry, "L2": l2}


def keynames(keys: int) -> str:
    names = [n for bit, n in KEY_BITS.items() if keys & (1 << bit)]
    return "+".join(names) if names else "-"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    from r1vision.unitree_dds import init_channel, quiet_dds_stderr

    latest: dict = {}

    with quiet_dds_stderr():
        init_channel(args.iface, peers=None)
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        # Callback form, not Read(): polling take_one() raises on every timeout
        # and the SDK prints "[Reader] take sample error" for each one, which
        # buries the data.
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(lambda m: latest.update(msg=m), 10)

        print(f"listening {args.seconds:.0f}s on rt/lowstate\n")
        print("  Move the Unitree remote's sticks. Nothing is sent to the robot.\n")

        deadline = time.monotonic() + args.seconds
        seen = 0
        last_line = None
        last_modes = None
        while time.monotonic() < deadline:
            time.sleep(0.1)
            msg = latest.get("msg")
            if msg is None:
                continue
            seen += 1

            modes = (msg.mode_pr, msg.mode_machine)
            if modes != last_modes:
                last_modes = modes
                print(f"  mode_pr={msg.mode_pr}  mode_machine={msg.mode_machine}")

            rc = decode_remote(msg.wireless_remote)
            if rc is None:
                continue
            line = (f"  remote  lx {rc['lx']:+.2f}  ly {rc['ly']:+.2f}  "
                    f"rx {rc['rx']:+.2f}  ry {rc['ry']:+.2f}  "
                    f"L2 {rc['L2']:+.2f}  keys 0x{rc['keys']:04x} "
                    f"{keynames(rc['keys'])}")
            if line != last_line:
                last_line = line
                print(line)

    print(f"\n  {seen} lowstate samples.")
    if seen == 0:
        print("  Nothing arriving on rt/lowstate. Either the ethernet link")
        print("  dropped again (it has been flapping -- check `ifconfig en6`)")
        print("  or DDS discovery did not match this run.")
        return 1
    if last_line is None:
        print("  lowstate arrives but wireless_remote has no valid header, so")
        print("  the remote is not being bridged into it on this unit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
