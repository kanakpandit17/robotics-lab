#!/usr/bin/env python3
"""Log what the arm motors ACTUALLY did, next to what was commanded.

    python arm_track_log.py --seconds 75

Why this exists
---------------
`hand_trace.py` records the teleop driver's own view: the pose it solved and
its IK residual. That is the COMMAND. It cannot tell you whether the joint
executed it, and "jerky" is precisely a question about execution -- a smooth
command that the arm follows in stick-slip steps looks identical, in the
driver's log, to a smooth command that the arm follows perfectly.

So this subscribes to rt/lowstate and records measured q and tau_est per arm
joint. Run it alongside hand_trace.py and the two halves answer different
questions:

  commanded jerky, measured follows   -> the solver/mapping is the problem
  commanded smooth, measured jerky    -> the joint is the problem
  measured barely moves, tau flat     -> the j15 fault from 2026-08-24

Read-only: it creates a SUBSCRIBER only, never a publisher on rt/lowcmd, so it
cannot command anything and is safe to run during live teleop.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

ARM = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}
JOINT_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out", default="arm_track.json")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from r1vision.unitree_dds import init_channel

    init_channel(args.iface, peers=None)
    latest = {}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: latest.__setitem__("m", m), 10)

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and "m" not in latest:
        time.sleep(0.1)
    if "m" not in latest:
        print("no rt/lowstate received", file=sys.stderr)
        return 1

    print(f"logging measured joint state for {args.seconds:.0f}s ...")
    samples = []
    period = 1.0 / args.hz
    t0 = time.monotonic()
    end = t0 + args.seconds
    while time.monotonic() < end:
        cycle = time.monotonic()
        st = latest["m"]
        row = {"t": round(cycle - t0, 3)}
        for side, idxs in ARM.items():
            row[side] = [{
                "q": round(math.degrees(st.motor_state[i].q), 3),
                "tau": round(float(st.motor_state[i].tau_est), 3),
            } for i in idxs]
        samples.append(row)
        rem = period - (time.monotonic() - cycle)
        if rem > 0:
            time.sleep(rem)

    Path(args.out).write_text(json.dumps(samples))
    print(f"wrote {args.out} ({len(samples)} samples)\n")

    # Jerk is the honest measure here: a joint that moves in stick-slip steps
    # has large angular acceleration even when its total travel looks normal,
    # so total range alone would miss exactly the symptom being chased.
    print(f"  {'joint':10s} {'range_deg':>10s} {'|tau|peak':>10s} "
          f"{'jerk_p95':>9s}  (deg/s^2, higher = rougher)")
    summary = {}
    for side in ("right", "left"):
        for j, name in enumerate(JOINT_NAMES):
            q = [s[side][j]["q"] for s in samples]
            tau = [abs(s[side][j]["tau"]) for s in samples]
            ts = [s["t"] for s in samples]
            acc = []
            for k in range(2, len(q)):
                dt1, dt2 = ts[k - 1] - ts[k - 2], ts[k] - ts[k - 1]
                if dt1 > 1e-6 and dt2 > 1e-6:
                    v1 = (q[k - 1] - q[k - 2]) / dt1
                    v2 = (q[k] - q[k - 1]) / dt2
                    acc.append(abs(v2 - v1) / dt2)
            acc.sort()
            p95 = acc[int(len(acc) * 0.95)] if acc else 0.0
            rng = max(q) - min(q)
            summary[f"{side}_{name}"] = {"range": rng, "tau_peak": max(tau),
                                         "jerk_p95": p95}
            print(f"  {side[0]}_{name:8s} {rng:10.2f} {max(tau):10.2f} {p95:9.0f}")

    print("\n  left-vs-right, same joint:")
    for name in JOINT_NAMES:
        l, r = summary[f"left_{name}"], summary[f"right_{name}"]
        note = ""
        if l["range"] < 2.0 and l["tau_peak"] < 0.5 and r["range"] > 5.0:
            note = "  <-- LEFT DEAD: no travel, no current, while right moved"
        elif r["jerk_p95"] > 1.0 and l["jerk_p95"] > r["jerk_p95"] * 2:
            note = f"  <-- LEFT {l['jerk_p95'] / r['jerk_p95']:.1f}x rougher"
        print(f"    {name:9s} range L{l['range']:7.2f} R{r['range']:7.2f} | "
              f"jerk L{l['jerk_p95']:7.0f} R{r['jerk_p95']:7.0f}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
