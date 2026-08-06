#!/usr/bin/env python3
"""Can we reach the R1's internal 192.168.123.x segment from its Wi-Fi AP?

Run while joined to the R1's Wi-Fi. Needs sudo to add a route.

    python sdk_probe.py

Unitree's SDK (DDS, motion control) lives on the locomotion computer at
192.168.123.x. The Wi-Fi AP at 192.168.12.1 is the app-facing side. Our DHCP
lease came with no router option, so nothing currently routes between the two
and 192.168.123.x is simply unreachable -- which would explain the total
absence of DDS without meaning the SDK is missing.

This adds an explicit route via the AP and retests. It also fingerprints
port 9991, which is almost certainly how the phone app drives the robot.

The route is removed again on exit unless --keep is passed.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

from r1vision.discover import PORT_HINTS, scan_ports, sweep  # noqa: E402

REPORT = Path("sdk-report.txt")
_lines: list[str] = []

INTERNAL = "192.168.123"
# Addresses Unitree puts the locomotion computer and motor boards on.
INTERNAL_TARGETS = ["192.168.123.161", "192.168.123.164", "192.168.123.10",
                    "192.168.123.15", "192.168.123.1"]


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def section(title: str) -> None:
    log()
    log("=" * 68)
    log(title)
    log("=" * 68)


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except (subprocess.SubprocessError, OSError) as exc:
        return f"<failed: {exc}>"


def ping(host: str, count: int = 2) -> bool:
    return subprocess.run(
        ["ping", "-c", str(count), "-W", "800", host],
        capture_output=True,
    ).returncode == 0


def fingerprint(host: str, port: int) -> None:
    """Poke a port with several dialects and report what comes back."""
    probes = {
        "raw connect": b"",
        "newline": b"\n",
        "HTTP": b"GET / HTTP/1.1\r\nHost: %s\r\n\r\n" % host.encode(),
        "WebSocket": (b"GET / HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\n"
                      b"Connection: Upgrade\r\nSec-WebSocket-Key: "
                      b"x3JJHMbDL1EzLkh9GBhXDw==\r\nSec-WebSocket-Version: 13\r\n\r\n"
                      % host.encode()),
    }
    for label, payload in probes.items():
        try:
            with socket.create_connection((host, port), 3) as s:
                s.settimeout(3)
                if payload:
                    s.sendall(payload)
                data = s.recv(400)
        except OSError as exc:
            log(f"    {label:<12}: {exc}")
            continue
        if not data:
            log(f"    {label:<12}: connected, no response")
            continue
        printable = data.decode("latin-1", "replace").replace("\r", "")
        first = printable.split("\n")[0][:90]
        log(f"    {label:<12}: {first!r}")
        log(f"    {'':<12}  hex: {data[:24].hex()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="192.168.12.1")
    ap.add_argument("--keep", action="store_true", help="leave the route in place")
    ap.add_argument("--skip-route", action="store_true")
    args = ap.parse_args()

    log(f"r1vision SDK-reachability probe - {datetime.now():%Y-%m-%d %H:%M:%S}")

    my_ip = sh(["ipconfig", "getifaddr", "en0"]).strip()
    log(f"  this Mac : {my_ip or '(none)'}")
    if not my_ip.startswith("192.168.12."):
        log("\n  Not on the robot's AP. Join R1_03302_c33840b2 and re-run.")
        REPORT.write_text("\n".join(_lines) + "\n")
        return 1

    # ------------------------------------------------------------------ 1
    section("1. Current routing")
    log(sh(["netstat", "-rn", "-f", "inet"]).strip()[:1500])

    # ------------------------------------------------------------------ 2
    section(f"2. Is {INTERNAL}.x reachable right now?")
    reachable = [t for t in INTERNAL_TARGETS if ping(t, 1)]
    log(f"  before adding a route: {reachable or 'nothing'}")

    # ------------------------------------------------------------------ 3
    added = False
    if not reachable and not args.skip_route:
        section(f"3. Adding route {INTERNAL}.0/24 via {args.gateway}")
        log("  (needs sudo)")
        out = sh(["sudo", "route", "-n", "add", "-net", f"{INTERNAL}.0/24", args.gateway], 30)
        log(f"  {out.strip()}")
        added = "add net" in out or not out.strip()
        time.sleep(1)

        log("\n  retesting ...")
        reachable = [t for t in INTERNAL_TARGETS if ping(t, 2)]
        log(f"  after route: {reachable or 'still nothing'}")

        if not reachable:
            log("\n  Sweeping the whole /24 in case it uses other addresses ...")
            found = sweep(INTERNAL)
            log(f"  {len(found)} host(s): {found or '(none)'}")
            reachable = found

    # ------------------------------------------------------------------ 4
    if reachable:
        section(f"4. What is on {INTERNAL}.x")
        for host in reachable:
            ports = scan_ports(host, list(PORT_HINTS))
            log(f"  {host}: {ports or '(no common ports)'}")
            for p in ports:
                log(f"      :{p:<6} {PORT_HINTS.get(p, '')}")
        log("\n  --> Re-run control_probe.py with --peer set to the locomotion")
        log("      computer's address to look for DDS there.")

    # ------------------------------------------------------------------ 5
    section("5. Fingerprinting the app control channel (192.168.12.1:9991)")
    fingerprint("192.168.12.1", 9991)

    section("6. Port 53 (the AP's DNS) - may reveal internal hostnames")
    for name in ("unitree", "robot", "r1", "localhost", "gateway"):
        out = sh(["dig", f"@192.168.12.1", "+short", "+time=2", "+tries=1", name], 8)
        if out.strip():
            log(f"  {name} -> {out.strip()}")

    # ------------------------------------------------------------------ cleanup
    if added and not args.keep:
        section("7. Removing the route")
        log(sh(["sudo", "route", "-n", "delete", "-net", f"{INTERNAL}.0/24",
                args.gateway], 30).strip())

    section("Verdict")
    if reachable:
        log("  The internal segment IS reachable. The SDK may well be there.")
        log("  Next: control_probe.py --peer <that address>")
    else:
        log("  The robot does not route from its Wi-Fi AP to the internal")
        log("  192.168.123.x segment. From this laptop, over Wi-Fi, the SDK")
        log("  control surface cannot be reached at all.")
        log()
        log("  That leaves three real options:")
        log("    1. Ethernet into the locomotion computer -- but this R1 has no")
        log("       RJ45, so this needs the EDU expansion or an internal port.")
        log("    2. A developer/SDK toggle in the Unitree app, if one exists.")
        log("    3. Ask Unitree support directly whether R1 SDK motion control")
        log("       is exposed over the AP on your firmware, and how.")

    REPORT.write_text("\n".join(_lines) + "\n")
    print(f"\n--- written to {REPORT.resolve()} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
