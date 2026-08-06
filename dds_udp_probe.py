#!/usr/bin/env python3
"""Scan for DDS on UDP. Every previous scan was TCP-only.

Run while joined to the R1's Wi-Fi.

    python dds_udp_probe.py

DDS runs entirely over UDP. The "full 65535 port scan" that found only 8551
and 9991 was a TCP scan, so it could never have seen a DDS stack no matter
what was running. This closes that gap.

Method: send a UDP datagram and watch for ICMP port-unreachable. A closed port
produces ECONNREFUSED on the next socket operation; a listening one stays
silent. Silence is not proof, but a wall of ECONNREFUSED across the whole RTPS
range is solid evidence that nothing is listening.

Ports follow the RTPS formula for domain d, participant p:
    discovery multicast  7400 + 250*d
    discovery unicast    7400 + 250*d + 10 + 2*p
    user traffic unicast 7400 + 250*d + 11 + 2*p
"""

from __future__ import annotations

import argparse
import errno
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

REPORT = Path("udp-report.txt")
_lines: list[str] = []

# A syntactically valid RTPS 2.1 header with no submessages. Harmless, and
# enough that a real DDS stack will at least not reject it as malformed.
RTPS_PING = (
    b"RTPS"
    + bytes([2, 1])            # protocol version
    + bytes([0x01, 0x10])      # vendor: Eclipse Cyclone DDS
    + b"\x00" * 12             # GUID prefix
)


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def section(title: str) -> None:
    log()
    log("=" * 68)
    log(title)
    log("=" * 68)


def udp_state(host: str, port: int, timeout: float = 1.2) -> str:
    """'closed' | 'open|filtered' | 'error'."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.send(RTPS_PING)
        time.sleep(0.05)
        try:
            sock.send(RTPS_PING)   # second send surfaces a queued ICMP error
        except OSError as exc:
            if exc.errno == errno.ECONNREFUSED:
                return "closed"
            raise
        try:
            data = sock.recv(2048)
            return f"REPLIED {data[:16].hex()}"
        except socket.timeout:
            return "open|filtered"
        except OSError as exc:
            if exc.errno == errno.ECONNREFUSED:
                return "closed"
            return f"error {exc.errno}"
    except OSError as exc:
        return f"error {exc}"
    finally:
        try:
            sock.close()
        except Exception:
            pass


def confirm_silent(host: str, port: int, attempts: int = 5,
                   spacing: float = 0.5) -> tuple[bool, list[str]]:
    """Re-test a port several times, spaced out.

    A single sample is unreliable: ICMP unreachable replies get dropped or
    rate-limited, so a closed port intermittently looks silent. Only a port
    that stays silent across every spaced attempt is worth believing.
    """
    results = []
    for i in range(attempts):
        if i:
            time.sleep(spacing)
        results.append(udp_state(host, port, timeout=1.5))
    return all(r == "open|filtered" for r in results), results


def ports_for_domain(domain: int, participants: int = 6) -> list[int]:
    base = 7400 + 250 * domain
    ports = [base]                                   # SPDP multicast port
    for p in range(participants):
        ports.append(base + 10 + 2 * p)              # SPDP unicast
        ports.append(base + 11 + 2 * p)              # user traffic
    return sorted(set(ports))


PEERS_FILE = Path("dds-peers.txt")


def sweep_mode(hosts: list[str], args) -> int:
    """Find every bound SPDP port, then write them out as a peer list.

    Guessing at participant-index ranges has failed twice. Ports that are
    genuinely bound are directly measurable, so measure them and aim discovery
    at exactly those instead.
    """
    if args.route:
        log("\n  adding route 192.168.123.0/24 via 192.168.12.1 (sudo) ...")
        out = subprocess.run(
            ["sudo", "route", "-n", "add", "-net", "192.168.123.0/24", "192.168.12.1"],
            capture_output=True, text=True,
        )
        log(f"  {(out.stdout + out.stderr).strip() or 'ok'}")

    log(f"r1vision DDS port sweep - {datetime.now():%Y-%m-%d %H:%M:%S}")
    # SPDP unicast ports are even (7400 + 10 + 2i); scanning odd ports as well
    # would only find user-traffic ports, which discovery does not need.
    ports = [7400] + list(range(7410, 7702, 2))
    log(f"  scanning {len(ports)} ports per host on {', '.join(hosts)}")

    confirmed: list[str] = []
    for host in hosts:
        if subprocess.run(["ping", "-c", "1", "-W", "800", host],
                          capture_output=True).returncode != 0:
            log(f"\n  {host}: unreachable, skipping")
            continue
        log(f"\n  {host}: pass 1 ...")
        silent = [p for p in ports if udp_state(host, p, 0.8) == "open|filtered"]
        log(f"    {len(silent)} silent of {len(ports)}")
        if not silent:
            continue
        log(f"    confirming each 3x ...")
        for port in silent:
            steady, _ = confirm_silent(host, port, attempts=3, spacing=0.4)
            if steady:
                confirmed.append(f"{host}:{port}")
                log(f"    {host}:{port}  BOUND")

    section("Result")
    if confirmed:
        PEERS_FILE.write_text("\n".join(confirmed) + "\n")
        log(f"  {len(confirmed)} bound port(s) written to {PEERS_FILE}")
        for entry in confirmed:
            log(f"    {entry}")
        log("\n  The other tools pick this file up automatically. Run:")
        log("    python arm_actions.py --list")
    else:
        log("  No bound SPDP ports found at all. If the robot is awake this")
        log("  means its DDS participants are not reachable from this network.")
    REPORT.write_text("\n".join(_lines) + "\n")
    return 0 if confirmed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", default="192.168.12.1,192.168.123.161")
    ap.add_argument("--domains", default="0,1,2")
    ap.add_argument("--route", action="store_true",
                    help="add the 192.168.123.0/24 route first (sudo)")
    ap.add_argument("--sweep", action="store_true",
                    help="scan every SPDP port 7400-7700 and write the bound "
                         "ones to dds-peers.txt for the other tools to use")
    args = ap.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    domains = [int(d) for d in args.domains.split(",")]

    if args.sweep:
        return sweep_mode(hosts, args)

    log(f"r1vision UDP/DDS probe - {datetime.now():%Y-%m-%d %H:%M:%S}")

    if args.route:
        log("\n  adding route 192.168.123.0/24 via 192.168.12.1 (sudo) ...")
        out = subprocess.run(
            ["sudo", "route", "-n", "add", "-net", "192.168.123.0/24", "192.168.12.1"],
            capture_output=True, text=True,
        )
        log(f"  {(out.stdout + out.stderr).strip() or 'ok'}")

    interesting = []
    for host in hosts:
        section(f"{host}")
        reachable = subprocess.run(
            ["ping", "-c", "1", "-W", "800", host], capture_output=True
        ).returncode == 0
        log(f"  ping: {'reply' if reachable else 'NO REPLY'}")
        if not reachable:
            log("  skipping port scan -- host unreachable")
            continue

        for domain in domains:
            states = {}
            for port in ports_for_domain(domain):
                states[port] = udp_state(host, port)
            closed = sum(1 for v in states.values() if v == "closed")
            quiet = [p for p, v in states.items() if v == "open|filtered"]
            replied = {p: v for p, v in states.items() if v.startswith("REPLIED")}

            log(f"\n  domain {domain} (ports {min(states)}-{max(states)}):")
            log(f"    closed (ICMP unreachable) : {closed}/{len(states)}")
            if quiet:
                log(f"    silent (open or filtered) : {quiet}")
                interesting.extend((host, domain, p) for p in quiet)
            if replied:
                for port, val in replied.items():
                    log(f"    *** {port} ANSWERED: {val}")
                    interesting.append((host, domain, port))

    # ---------------------------------------------------------------- confirm
    if interesting:
        section("Confirmation pass")
        log("  Re-testing each silent port 5x, spaced 0.5s, to weed out")
        log("  dropped/rate-limited ICMP replies.\n")
        confirmed = []
        for host, domain, port in interesting:
            steady, results = confirm_silent(host, port)
            summary = " ".join("S" if r == "open|filtered" else "c" for r in results)
            log(f"  {host}:{port:<6} [{summary}]  {'CONFIRMED SILENT' if steady else 'noise'}")
            if steady:
                confirmed.append((host, domain, port))
        interesting = confirmed

    section("Verdict")
    if not interesting:
        log("  Every RTPS port answered ICMP port-unreachable. Nothing is")
        log("  listening for DDS on any tested host or domain. The absence of")
        log("  DDS is now confirmed on UDP as well as TCP.")
        log()
        log("  The SDK is not running, or not exposed to this network path.")
        log("  Next: the «quick_development» page of the R1 docs, which covers")
        log("  connection setup -- that is the piece we are missing.")
    else:
        log("  Some ports did NOT refuse the datagram:")
        for host, domain, port in interesting:
            log(f"    {host}:{port} (domain {domain})")
        log()
        log("  Something may be listening. Re-run:")
        log("    python control_probe.py --route --peer <that host>")
        log("  and if it still finds no topics, the DDS stack is there but")
        log("  not publishing -- which points at an enable step in the app.")

    REPORT.write_text("\n".join(_lines) + "\n")
    print(f"\n--- written to {REPORT.resolve()} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
