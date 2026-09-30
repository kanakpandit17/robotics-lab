#!/usr/bin/env python3
"""Commanded vs measured, both arms, triggered by your grip.

    python teleop_diag.py                 # waits until you grip, then records

Why this exists
---------------
Two earlier attempts to capture the left arm's "jerky" behaviour recorded
nothing but a still robot, because they ran on a fixed timer started from the
terminal while the operator was in a headset and had already stopped moving.
Every joint logged 0.00 deg of travel. A capture that has to be started at the
right moment by someone who cannot see the terminal will keep missing the
event, so this one waits for the grip instead and the operator never has to
coordinate with anything.

It also records BOTH halves in one process, which is the point. The teleop
driver's own log (`hand_trace.py`) shows only what was COMMANDED; rt/lowstate
shows what the motors DID. "Jerky" is a question about execution, and a smooth
command executed in stick-slip lurches is indistinguishable, in the driver's
log alone, from a smooth command executed perfectly. Only the pair separates:

  commanded rough, measured follows  -> solver/mapping
  commanded smooth, measured rough   -> the joint
  measured flat, tau flat            -> the j15 no-current fault (2026-08-24)

Read-only with respect to the robot: it opens a rt/lowstate SUBSCRIBER and
never a publisher, so it cannot command anything and is safe during live
teleop.
"""

from __future__ import annotations

import argparse
import json
import math
import ssl
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

ARM = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}
JOINT_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll"]


def fetch(url, timeout=5.0):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode())


def p95_jerk(q, ts):
    """95th-percentile |angular acceleration|, deg/s^2.

    Jerkiness is not visible in total travel -- a stick-slip joint can cover a
    perfectly normal range while getting there in lurches -- so the statistic
    has to be about how abruptly velocity changes, not how far the joint got.
    """
    acc = []
    for k in range(2, len(q)):
        dt1, dt2 = ts[k - 1] - ts[k - 2], ts[k] - ts[k - 1]
        if dt1 > 1e-6 and dt2 > 1e-6:
            acc.append(abs((q[k] - q[k - 1]) / dt2 - (q[k - 1] - q[k - 2]) / dt1) / dt2)
    if not acc:
        return 0.0
    acc.sort()
    return acc[int(len(acc) * 0.95)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--http", action="store_true", default=True)
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--engaged-seconds", type=float, default=30.0,
                    help="how many seconds of ACTUALLY GRIPPED motion to collect")
    ap.add_argument("--wait", type=float, default=300.0,
                    help="how long to wait for the first grip before giving up")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out", default="teleop_diag.json")
    args = ap.parse_args()

    url = f"{'http' if args.http else 'https'}://{args.host}:{args.port}/teleop?status=1"
    try:
        fetch(url)
    except Exception as e:
        print(f"cannot reach {url}: {e}", file=sys.stderr)
        return 1

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
        print("no rt/lowstate -- is the robot powered?", file=sys.stderr)
        return 1

    print(f"ready. Waiting up to {args.wait:.0f}s for you to GRIP a controller.")
    print(f"Once gripped, records {args.engaged_seconds:.0f}s of gripped motion.")
    print("Move BOTH arms the same way: forward, out, up, down.\n")

    samples = []
    period = 1.0 / args.hz
    engaged_time = 0.0
    started = False
    give_up = time.monotonic() + args.wait
    last_note = 0.0
    prev = time.monotonic()

    while engaged_time < args.engaged_seconds:
        now = time.monotonic()
        if not started and now > give_up:
            print("\nno grip seen -- nothing recorded.")
            return 1
        cycle = now
        try:
            st = fetch(url, timeout=2.0)
        except Exception:
            time.sleep(0.2)
            continue

        eng = st.get("engaged", {})
        any_eng = any(eng.values())
        if any_eng and not started:
            started = True
            print("  grip detected -- recording ...")

        if any_eng:
            dt = cycle - prev
            if 0 < dt < 0.5:
                engaged_time += dt
            ls = latest["m"]
            row = {"t": round(cycle, 3), "engaged": dict(eng),
                   "commanded": st.get("solved_deg", {}),
                   "ik_err_mm": st.get("ik_err_mm", {})}
            for side, idxs in ARM.items():
                row["meas_" + side] = [{
                    "q": round(math.degrees(ls.motor_state[i].q), 3),
                    "tau": round(float(ls.motor_state[i].tau_est), 3),
                } for i in idxs]
            samples.append(row)

        prev = cycle
        if started and cycle - last_note >= 3.0:
            print(f"  {engaged_time:5.1f}s gripped, {len(samples)} samples")
            last_note = cycle
        rem = period - (time.monotonic() - cycle)
        if rem > 0:
            time.sleep(rem)

    Path(args.out).write_text(json.dumps(samples))
    print(f"\nwrote {args.out} ({len(samples)} samples)\n")

    ts = [s["t"] for s in samples]
    print(f"  {'joint':10s} {'CMD range':>10s} {'MEAS range':>11s} "
          f"{'follow%':>10s} {'|tau|pk':>8s} {'MEAS jerk':>10s}")
    stats = {}
    for side in ("right", "left"):
        for j, name in enumerate(JOINT_NAMES):
            cmd = [s["commanded"][side][j] for s in samples
                   if s["commanded"].get(side)]
            meas = [s["meas_" + side][j]["q"] for s in samples]
            tau = [abs(s["meas_" + side][j]["tau"]) for s in samples]
            crange = (max(cmd) - min(cmd)) if cmd else 0.0
            mrange = max(meas) - min(meas)
            # RANGE ratio, not instantaneous error -- and this distinction
            # cost a wrong diagnosis, so it is worth the paragraph.
            #
            # Median |command - measured| looks like the rigorous choice and is
            # useless here: `ik()` seeds from the previously solved pose and
            # _clamp_step limits each step, so the command CHASES WHEREVER THE
            # ARM ACTUALLY IS. A seized joint therefore reports a tiny
            # instantaneous error while going nowhere -- measured, on a j15
            # that an open-loop sweep proved dead minutes later: median error
            # 0.04 deg (better than the healthy right shoulder's 0.34) against
            # a commanded range of 52.7 deg and a measured range of 19.0.
            # The range ratio caught it; the median hid it.
            #
            # Neither number is authoritative about joint HEALTH, because both
            # are computed from a closed loop whose command depends on the
            # measurement. Use shoulder_sweep.py -- open-loop, identical
            # targets to both shoulders -- to decide whether a joint works.
            follow = (mrange / crange * 100) if crange > 0.5 else float("nan")
            jerk = p95_jerk(meas, ts)
            stats[f"{side}_{name}"] = dict(crange=crange, mrange=mrange,
                                           follow=follow, tau=max(tau), jerk=jerk)
            fs = "     n/a" if follow != follow else f"{follow:8.0f}%"
            print(f"  {side[0]}_{name:8s} {crange:10.2f} {mrange:11.2f} "
                  f"{fs:>10s} {max(tau):8.2f} {jerk:10.0f}")

    print("\n  left vs right, same joint:")
    for name in JOINT_NAMES:
        l, r = stats[f"left_{name}"], stats[f"right_{name}"]
        note = ""
        if l["crange"] > 3.0 and l["mrange"] < 1.0 and l["tau"] < 0.5:
            note = "  <-- DEAD: commanded to move, no travel, no current"
        elif r["jerk"] > 50 and l["jerk"] > r["jerk"] * 2:
            note = f"  <-- LEFT {l['jerk'] / r['jerk']:.1f}x rougher"
        elif l["follow"] == l["follow"] and r["follow"] == r["follow"] \
                and r['follow'] > 40 and l['follow'] < r['follow'] / 2:
            note = "  <-- LEFT range far short of right - run shoulder_sweep.py"
        print(f"    {name:9s} cmd L{l['crange']:6.1f} R{r['crange']:6.1f} | "
              f"meas L{l['mrange']:6.1f} R{r['mrange']:6.1f} | "
              f"jerk L{l['jerk']:6.0f} R{r['jerk']:6.0f}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

