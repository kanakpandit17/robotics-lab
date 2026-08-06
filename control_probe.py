#!/usr/bin/env python3
"""Find the R1's control channel. Read-only -- commands nothing, moves nothing.

Run while joined to the R1's Wi-Fi.

    python control_probe.py

The earlier deep probe found no DDS, but it looked for it via multicast, and
Wi-Fi APs routinely filter multicast. That would hide a perfectly healthy DDS
stack. This retries discovery over UNICAST, aimed straight at the robot, which
does not depend on multicast working at all.

If DDS turns up, this also dumps the joint layout -- how many joints, and each
one's current angle. That is what a motion program needs in order to address
the right arm rather than guess at an index.
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

# Must happen before any cyclonedds import, and before we could possibly
# mistake a missing library for a robot that does not speak DDS.
ensure_venv(("cyclonedds", "cv2"))

REPORT = Path("control-report.txt")
_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def section(title: str) -> None:
    log()
    log("=" * 68)
    log(title)
    log("=" * 68)


def default_peers(hosts: tuple[str, ...] = ("192.168.123.161", "192.168.12.1"),
                  domain: int = 0, max_participant: int = 45) -> list[str]:
    """Every plausible SPDP locator on the robot.

    Each robot process is its own DDS participant on its own unicast port
    (7400 + 250*domain + 10 + 2*index). We found one participant on the
    multicast port 7400, publishing only a status string -- the motion
    controller will be a different process on a different port. Sweeping the
    participant index range finds it without needing to know the index.
    """
    base = 7400 + 250 * domain
    ports = [base] + [base + 10 + 2 * i for i in range(max_participant + 1)]
    return [f"{host}:{port}" for host in hosts for port in ports]


def configure_dds(iface: str, peers: list[str], multicast: bool) -> None:
    """Point CycloneDDS at the robot directly.

    A peer entry may carry an explicit port ("192.168.12.1:7400"). That matters
    here: the robot has 7400 bound but every unicast SPDP port (7410+2i) is
    closed, so discovery aimed at the default ports reaches nothing. Naming
    7400 explicitly sends SPDP where something is actually listening.

    Must be set before cyclonedds is imported.
    """
    peer_xml = "".join(f"<Peer address='{p}'/>" for p in peers)
    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS><Domain>"
        "<General>"
        f"<Interfaces><NetworkInterface name='{iface}' priority='default'/></Interfaces>"
        f"<AllowMulticast>{'true' if multicast else 'false'}</AllowMulticast>"
        "</General>"
        "<Discovery>"
        "<ParticipantIndex>auto</ParticipantIndex>"
        "<MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>"
        f"<Peers>{peer_xml}</Peers>"
        "</Discovery>"
        "<Tracing><Verbosity>none</Verbosity>"
        "<OutputFile>/tmp/r1vision-cyclonedds.log</OutputFile></Tracing>"
        "</Domain></CycloneDDS>"
    )


def enumerate_topics(domain: int, seconds: int) -> tuple[list, list, list]:
    from cyclonedds.builtin import (
        BuiltinDataReader,
        BuiltinTopicDcpsParticipant,
        BuiltinTopicDcpsPublication,
        BuiltinTopicDcpsSubscription,
    )
    from cyclonedds.domain import DomainParticipant

    participant = DomainParticipant(domain)
    r_part = BuiltinDataReader(participant, BuiltinTopicDcpsParticipant)
    r_pub = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    r_sub = BuiltinDataReader(participant, BuiltinTopicDcpsSubscription)

    parts, pubs, subs = {}, {}, {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for s in r_part.take(N=50):
            parts[str(getattr(s, "key", s))] = True
        for s in r_pub.take(N=100):
            name = getattr(s, "topic_name", None)
            if name and not name.startswith("DCPS"):
                pubs[name] = getattr(s, "type_name", "?")
        for s in r_sub.take(N=100):
            name = getattr(s, "topic_name", None)
            # DCPS* are our own builtin readers. Counting them as robot topics
            # made the probe report a control channel that does not exist.
            if name and not name.startswith("DCPS"):
                subs[name] = getattr(s, "type_name", "?")
        time.sleep(0.4)
    return sorted(parts), sorted(pubs.items()), sorted(subs.items())


def dump_lowstate(iface: str, seconds: int = 8, peers: list[str] | None = None) -> bool:
    """Print the joint layout by running the dumper in a clean subprocess.

    This process already built a cyclonedds participant for topic enumeration.
    Letting the Unitree SDK create its own Domain(0, config) on top of that
    collides and the subscriber receives nothing -- which is exactly what the
    previous run showed. A fresh interpreter avoids it entirely.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "r1vision.lowstate_dump",
         iface, ",".join(peers or []), str(seconds)],
        capture_output=True, text=True, timeout=seconds + 25,
        cwd=str(Path(__file__).resolve().parent),
    )
    out = (result.stdout or "").strip()
    if not out or "NO_DATA" in out:
        log("  no lowstate data arrived")
        if result.stderr.strip():
            log(f"  stderr: {result.stderr.strip()[:300]}")
        return False
    for line in out.splitlines():
        log("  " + line)
    return True


def _unused_dump_lowstate_inprocess(iface: str, seconds: int = 6,
                                    peers: list[str] | None = None) -> bool:
    from unitree_sdk2py.core.channel import ChannelSubscriber

    from r1vision.unitree_dds import init_channel

    init_channel(iface, peers=peers)

    # G1/H1-class humanoids use unitree_hg; Go2-class uses unitree_go. Which one
    # the R1 speaks is exactly what we are trying to establish, so try both.
    attempts = []
    try:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HgLowState
        attempts.append(("unitree_hg", "rt/lowstate", HgLowState))
    except ImportError:
        pass
    try:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as GoLowState
        attempts.append(("unitree_go", "rt/lowstate", GoLowState))
    except ImportError:
        pass

    for family, topic, msg_type in attempts:
        log(f"\n  trying {family} on {topic} ...")
        got = {}

        def on_msg(msg, _store=got):
            _store["msg"] = msg

        try:
            sub = ChannelSubscriber(topic, msg_type)
            sub.Init(on_msg, 10)
        except Exception as exc:
            log(f"    subscribe failed: {type(exc).__name__}: {exc}")
            continue

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and "msg" not in got:
            time.sleep(0.2)

        if "msg" not in got:
            log("    no data")
            continue

        msg = got["msg"]
        motors = getattr(msg, "motor_state", None)
        if motors is None:
            log("    message received but has no motor_state field")
            continue

        live = [(i, m) for i, m in enumerate(motors) if getattr(m, "q", 0.0) != 0.0]
        log(f"    GOT DATA on {family}: {len(motors)} motor slots,"
            f" {len(live)} reporting non-zero angle")
        log()
        log("    idx     q (rad)      q (deg)     temperature")
        for i, m in enumerate(motors):
            q = getattr(m, "q", 0.0)
            if q == 0.0 and i > 30:
                continue
            temp = getattr(m, "temperature", "")
            log(f"    {i:>3}   {q:>9.4f}   {q * 57.2958:>9.2f}    {temp}")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--peer", default="auto",
                    help="comma-separated DDS peers (entries may name a port), "
                         "or 'auto' to sweep the whole participant index range")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=25)
    ap.add_argument("--route", action="store_true",
                    help="add a route to 192.168.123.0/24 first (needs sudo)")
    ap.add_argument("--gateway", default="192.168.12.1")
    args = ap.parse_args()

    log(f"r1vision control probe - {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"  interface {args.iface}, peer {args.peer}")
    log("  READ ONLY - this sends no commands and moves nothing.")

    if args.route:
        # The AP hands out no default route, so the internal 192.168.123.x
        # segment where the SDK lives is unreachable until we add one.
        import subprocess
        log(f"\n  adding route 192.168.123.0/24 via {args.gateway} (sudo) ...")
        result = subprocess.run(
            ["sudo", "route", "-n", "add", "-net", "192.168.123.0/24", args.gateway],
            capture_output=True, text=True,
        )
        out = (result.stdout + result.stderr).strip()
        log(f"  {out or 'ok'}")
        if "File exists" in out:
            log("  (route was already present -- fine)")
        log(f"  remove later with: sudo route -n delete -net 192.168.123.0/24 {args.gateway}")

    # ------------------------------------------------------------------ 1
    section("1. Unicast DDS discovery (multicast-independent)")
    if args.peer == "auto":
        peers = default_peers(domain=args.domain)
        log(f"  peers: {len(peers)} locators across "
            f"{min(int(p.rsplit(':', 1)[1]) for p in peers)}-"
            f"{max(int(p.rsplit(':', 1)[1]) for p in peers)} on 2 hosts")
    else:
        peers = [p.strip() for p in args.peer.split(",") if p.strip()]
        log(f"  peers: {', '.join(peers)}")
    # Multicast stays on: 7400 is the multicast discovery port, and if the AP
    # does forward multicast that is the intended path anyway.
    configure_dds(args.iface, peers, multicast=True)
    env_broken = False
    try:
        parts, pubs, subs = enumerate_topics(args.domain, args.seconds)
    except ImportError as exc:
        env_broken = True
        parts, pubs, subs = [], [], []
        log(f"  ENVIRONMENT ERROR: {exc}")
        log("  The DDS library is not importable, so this run says NOTHING")
        log("  about the robot. Re-run with ./.venv/bin/python.")
    except Exception as exc:
        log(f"  failed: {type(exc).__name__}: {exc}")
        parts, pubs, subs = [], [], []

    log(f"  participants seen : {len(parts)}")
    log(f"  topics published  : {len(pubs)}")
    log(f"  topics subscribed : {len(subs)}")

    if pubs:
        log("\n  PUBLISHED (robot -> us):")
        for name, type_name in pubs:
            log(f"    {name:<40} {type_name}")
    if subs:
        log("\n  SUBSCRIBED (us -> robot, i.e. command topics):")
        for name, type_name in subs:
            log(f"    {name:<40} {type_name}")

    # ------------------------------------------------------------------ 2
    if not pubs and not subs:
        section("2. Retry with multicast enabled, other domains")
        for domain in (0, 1, 2):
            configure_dds(args.iface, peers, multicast=True)
            try:
                p, pu, su = enumerate_topics(domain, 5)
                log(f"  domain {domain}: {len(pu)} published, {len(su)} subscribed")
                if pu or su:
                    for name, type_name in pu + su:
                        log(f"    {name:<40} {type_name}")
                    pubs = pu
                    break
            except Exception as exc:
                log(f"  domain {domain}: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ 3
    section("3. Joint layout")
    if pubs or subs:
        try:
            ok = dump_lowstate(args.iface, peers=peers)
        except Exception as exc:
            ok = False
            log(f"  failed: {type(exc).__name__}: {exc}")
        if not ok:
            log("\n  DDS is present but lowstate did not arrive.")
    else:
        log("  Skipped -- no DDS to read from.")

    # ------------------------------------------------------------------ 4
    section("4. Verdict")
    if env_broken:
        log("  INCONCLUSIVE. The DDS library never loaded, so nothing here")
        log("  describes the robot. This is a laptop problem, not a robot one.")
        log("  Re-run as:  ./.venv/bin/python control_probe.py")
    elif pubs or subs:
        log("  A DDS control channel EXISTS. The earlier 'no DDS' result was a")
        log("  multicast artefact, not the truth. Send me this report and I will")
        log("  write the arm motion program against these exact topics.")
    else:
        log("  No DDS -- multicast and unicast, domains 0/1/2, both the AP")
        log("  address and the internal 192.168.123.161.")
        log()
        log("  Combined with the full port scan, this robot exposes only:")
        log("    :8551  RTSP video (one-way, carries no commands)")
        log("    :9991  Unitree app channel (proprietary, binary)")
        log()
        log("  So SDK motion control is not reachable from this laptop on the")
        log("  current firmware. This is a robot configuration question now,")
        log("  not a networking one. See README 'Motion control: not available'.")

    REPORT.write_text("\n".join(_lines) + "\n")
    print(f"\n--- written to {REPORT.resolve()} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
