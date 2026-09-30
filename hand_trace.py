#!/usr/bin/env python3
"""Record both arms at once during teleop, then compare them.

    python quest_stream.py --teleop --teleop-dry-run     # in another terminal
    python hand_trace.py --host 192.168.94.68            # then this

Why this exists
---------------
`left_trace.json` (2026-08-20) shows the left arm missing its IK target by a
median of 68 mm, with wrist roll pinned at its +110 deg limit a third of the
time. What it does NOT show is what the right arm was doing in those same
cycles -- there is no matching right trace, so "the left one is worse" rests
on an impression rather than a measurement, and those numbers could just be
what this teleop does on both sides.

`/teleop?status=1` already reports every field for both hands in a single
snapshot, so one poller captures a matched pair with no chance of the two
samples coming from different moments, different clutch references or
different sessions. That is the whole point: the comparison is only worth
anything if both arms saw the same operator, the same link and the same cycle.

Read-only. It polls status and never commands anything, so it is safe to run
against a live session -- and safer still against `--teleop-dry-run`, which
is how it should be tried first.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("numpy",))

import numpy as np  # noqa: E402

from r1vision import arm_kinematics  # noqa: E402

JOINT_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll"]
# Within this many degrees of a limit counts as sitting on it. The solver
# clamps exactly to the limit, so equality would nearly work -- but the
# status endpoint rounds to 0.1 deg, which exact comparison would then miss.
AT_LIMIT_DEG = 0.5


def fetch(url: str, timeout: float = 10.0) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode())


def collect(url: str, seconds: float, hz: float) -> list[dict]:
    """Poll status into a list of samples.

    Only cycles where a hand is actually engaged tell you anything -- an
    unclutched arm is being commanded to hold still, and its ik_err is
    whatever the last solve left behind, so recording those would dilute
    every statistic below.
    """
    period = 1.0 / hz
    samples: list[dict] = []
    deadline = time.monotonic() + seconds
    t0 = time.monotonic()
    last_note = 0.0
    while time.monotonic() < deadline:
        cycle = time.monotonic()
        try:
            st = fetch(url)
        except Exception as exc:
            print(f"  status fetch failed: {type(exc).__name__}: {exc}")
            time.sleep(0.5)
            continue

        engaged = st.get("engaged", {})
        if any(engaged.values()):
            samples.append({
                "t": round(cycle - t0, 3),
                "engaged": engaged,
                "solved_deg": st.get("solved_deg", {}),
                "ik_err_mm": st.get("ik_err_mm", {}),
                "controller": st.get("controller", {}),
                "weight": st.get("weight"),
            })

        if cycle - last_note >= 2.0:
            live = [h for h, v in engaged.items() if v]
            print(f"  {cycle - t0:5.1f}s  samples={len(samples):4d}  "
                  f"engaged={live or 'none -- grip a controller'}")
            last_note = cycle

        remaining = period - (time.monotonic() - cycle)
        if remaining > 0:
            time.sleep(remaining)
    return samples


def summarise(samples: list[dict], hand: str) -> dict | None:
    """Per-hand statistics, over only the cycles that hand was engaged."""
    rows = [s for s in samples if s["engaged"].get(hand)]
    if not rows:
        return None

    err = np.array([s["ik_err_mm"].get(hand, float("nan")) for s in rows],
                   dtype=float)
    err = err[~np.isnan(err)]
    q = np.array([s["solved_deg"][hand] for s in rows if hand in s["solved_deg"]],
                 dtype=float)
    if err.size == 0 or q.size == 0:
        return None

    kin = arm_kinematics.chain(hand)
    lo, hi = np.degrees(kin.lower), np.degrees(kin.upper)

    joints = []
    for i, name in enumerate(JOINT_NAMES):
        col = q[:, i]
        joints.append({
            "joint": name,
            "lower": round(float(lo[i]), 1),
            "upper": round(float(hi[i]), 1),
            "min": round(float(col.min()), 1),
            "max": round(float(col.max()), 1),
            # How much of the joint's legal travel the arm actually used. A
            # low number with no pinning means the operator never asked for
            # more; a low number WITH pinning means something is driving the
            # joint into a stop and holding it there.
            "span_pct": round(float(col.max() - col.min()) / (hi[i] - lo[i]) * 100, 1),
            "at_lower_pct": round(float(np.mean(np.abs(col - lo[i]) < AT_LIMIT_DEG) * 100), 1),
            "at_upper_pct": round(float(np.mean(np.abs(col - hi[i]) < AT_LIMIT_DEG) * 100), 1),
        })

    return {
        "samples": len(rows),
        "ik_err_mm": {
            "median": round(float(np.median(err)), 1),
            "mean": round(float(err.mean()), 1),
            "p90": round(float(np.percentile(err, 90)), 1),
            "max": round(float(err.max()), 1),
            "over_20mm_pct": round(float((err > 20).mean() * 100), 1),
        },
        "joints": joints,
    }


def report(summary: dict) -> None:
    for hand in ("right", "left"):
        s = summary.get(hand)
        print(f"\n--- {hand} arm ---")
        if s is None:
            print("  never engaged -- grip that controller during the capture")
            continue
        e = s["ik_err_mm"]
        print(f"  {s['samples']} engaged samples")
        print(f"  ik_err: median {e['median']} mm, mean {e['mean']} mm, "
              f"p90 {e['p90']} mm, max {e['max']} mm")
        print(f"  cycles missing target by >20 mm: {e['over_20mm_pct']}%")
        print(f"  {'joint':9s} {'limits':>17s} {'used':>17s} {'span':>6s} "
              f"{'@lower':>7s} {'@upper':>7s}")
        for j in s["joints"]:
            print(f"  {j['joint']:9s} [{j['lower']:7.1f},{j['upper']:7.1f}] "
                  f"[{j['min']:7.1f},{j['max']:7.1f}] {j['span_pct']:5.1f}% "
                  f"{j['at_lower_pct']:6.1f}% {j['at_upper_pct']:6.1f}%")

    left, right = summary.get("left"), summary.get("right")
    print("\n--- verdict ---")
    if not (left and right):
        # Without both, there is no comparison -- and reporting one side's
        # numbers as if they meant something on their own is the exact
        # mistake left_trace.json invites.
        print("  only one arm was engaged, so there is nothing to compare.")
        print("  Re-run and grip BOTH controllers, ideally making the same")
        print("  motion with each, or the difference you measure is just the")
        print("  difference between the two motions you made.")
        return

    lm, rm = left["ik_err_mm"]["median"], right["ik_err_mm"]["median"]
    print(f"  median ik_err: left {lm} mm vs right {rm} mm", end="")
    if rm > 0.5 and lm > rm * 2:
        print(f"  -- left is {lm / rm:.1f}x worse")
    elif lm > rm * 1.3 or rm > lm * 1.3:
        print("  -- some difference, but not a clear one")
    else:
        print("  -- comparable; whatever is wrong is NOT left-specific")

    for i, name in enumerate(JOINT_NAMES):
        lj, rj = left["joints"][i], right["joints"][i]
        lpin = max(lj["at_lower_pct"], lj["at_upper_pct"])
        rpin = max(rj["at_lower_pct"], rj["at_upper_pct"])
        if lpin > 10 and lpin > rpin * 2:
            edge = "lower" if lj["at_lower_pct"] >= lj["at_upper_pct"] else "upper"
            print(f"  {name}: left sits on its {edge} limit {lpin}% of cycles "
                  f"(right {rpin}%) -- that joint is being driven into a stop")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--http", action="store_true", help="plain HTTP (dry-run server)")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--out", default="hand_trace.json")
    args = ap.parse_args()

    scheme = "http" if args.http else "https"
    url = f"{scheme}://{args.host}:{args.port}/teleop?status=1"

    print(f"Polling {url} for {args.seconds:.0f}s at {args.hz:.0f} Hz.")
    print("Grip BOTH controllers and make the SAME motion with each -- reach")
    print("forward, out to the side, up, and down. Only engaged cycles count.\n")

    samples = collect(url, args.seconds, args.hz)
    if not samples:
        print("\nNo engaged samples. Was a grip ever held? Is --teleop running?")
        return 1

    summary = {h: summarise(samples, h) for h in ("left", "right")}
    report(summary)

    Path(args.out).write_text(json.dumps(
        {"samples": samples, "summary": summary}, indent=1))
    print(f"\nwrote {args.out} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
