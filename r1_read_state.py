#!/usr/bin/env python3
"""READ-ONLY Unitree R1 state reader. Run in WSL (default Ubuntu) with the r1bridge venv:

    ~/r1bridge/.venv/bin/python /mnt/c/Users/aimat/Desktop/humanoid/r1_read_state.py --iface eth1

It SUBSCRIBES to the robot's LowState (rt/lowstate) and prints joint angles. It PUBLISHES
NOTHING -> the robot cannot move. This is the safe first test: it proves (1) WiFi/DDS
connectivity, (2) that the R1 uses the unitree_hg message set, and (3) the real joint count.

--iface is the network interface on the R1's subnet. In mirrored WSL you have:
    eth1 = 192.168.2.99   (same subnet the MAiRA used)
    eth4 = 141.70.221.217
Unitree robots are commonly on 192.168.123.x (onboard) -> that may need its own iface/route.
If one iface prints nothing, try the other, or omit --iface to let DDS use all interfaces.
"""
from __future__ import annotations

import argparse
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="", help="network interface on the robot's subnet (e.g. eth1). Empty = all interfaces")
    ap.add_argument("--secs", type=float, default=15.0, help="how long to listen")
    ap.add_argument("--msgset", choices=["hg", "go"], default="hg", help="hg = G1/H1-2/R1 humanoids (default); go = Go2/B2/H1")
    ap.add_argument("--njoints", type=int, default=32, help="how many joint angles to print")
    args = ap.parse_args()

    if args.msgset == "hg":
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    else:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    print(f"[R1-READ] init DDS (iface={args.iface or 'ALL'}, msgset={args.msgset}) ...", flush=True)
    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    state = {"msg": None, "count": 0}

    def on_lowstate(msg: "LowState_") -> None:
        state["msg"] = msg
        state["count"] += 1

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 10)
    print("[R1-READ] subscribed to rt/lowstate. Waiting for robot data ...", flush=True)

    t0 = time.time()
    last = 0.0
    got = False
    while time.time() - t0 < args.secs:
        if state["msg"] is not None and time.time() - last > 0.5:
            got = True
            m = state["msg"]
            ms = list(getattr(m, "motor_state", []) or [])
            n = min(args.njoints, len(ms))
            q = [round(float(ms[i].q), 3) for i in range(n)]
            print(f"[R1-READ] msgs={state['count']:4d}  n_motors={len(ms)}  q[0:{n}]={q}", flush=True)
            last = time.time()
        time.sleep(0.02)

    if not got:
        print("\n[R1-READ] NO DATA received.", flush=True)
        print("  Checklist:", flush=True)
        print("   1. Is the R1 powered on and finished booting?", flush=True)
        print("   2. Is it on the SAME WiFi/LAN as this PC? (ping its IP first)", flush=True)
        print("   3. Try the other interface:  --iface eth4   (or omit --iface)", flush=True)
        print("   4. If the R1 is a Go-family message robot:  --msgset go", flush=True)
        return 2
    print(f"\n[R1-READ] OK - received {state['count']} LowState msgs. The R1 speaks '{args.msgset}'.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
