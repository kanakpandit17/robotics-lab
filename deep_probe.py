#!/usr/bin/env python3
"""Second pass: find what the R1 exposes when a normal port scan finds nothing.

Run this while joined to the R1's Wi-Fi. No internet needed.

    python deep_probe.py

A TCP scan of the common ports came back empty on 192.168.12.1, which is what
you would expect from a robot that talks DDS -- DDS runs over UDP and is
invisible to TCP scanning. So this does three things a normal scan does not:

  * scans all 65535 TCP ports, not just the usual suspects
  * listens for RTPS/SPDP multicast announcements, which every DDS participant
    broadcasts on a timer -- this confirms DDS and reveals the robot's unicast
    ports without needing the Unitree SDK installed
  * TCP-probes every address on the subnet, catching hosts that ignore ping

Results go to deep-report.txt.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds", "cv2"))

REPORT = Path("deep-report.txt")
_lines: list[str] = []

# DDS domain 0 discovery. Port formula is 7400 + 250*domain, and Unitree's SDK
# initialises on domain 0, so 7400 is where announcements land.
SPDP_MULTICAST = "239.255.0.1"
SPDP_PORT = 7400

VENDORS = {
    b"\x01\x01": "RTI Connext",
    b"\x01\x02": "OpenSplice",
    b"\x01\x03": "OCI",
    b"\x01\x04": "MilSOFT",
    b"\x01\x05": "InterCOM/Kongsberg",
    b"\x01\x06": "TwinOaks",
    b"\x01\x07": "eProsima Fast-DDS",
    b"\x01\x08": "Eclipse Cyclone DDS",
    b"\x01\x09": "GurumDDS",
    b"\x01\x10": "Eclipse Cyclone DDS",
}


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def section(title: str) -> None:
    log()
    log("=" * 68)
    log(title)
    log("=" * 68)


def local_ip_on(prefix: str) -> str:
    import subprocess
    out = subprocess.run(["ipconfig", "getifaddr", "en0"],
                         capture_output=True, text=True).stdout.strip()
    return out


def tcp_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def full_tcp_scan(host: str, workers: int = 600) -> list[int]:
    ports = range(1, 65536)
    open_ports: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for port, ok in zip(ports, pool.map(lambda p: tcp_open(host, p), ports)):
            if ok:
                open_ports.append(port)
    return open_ports


def banner(host: str, port: int) -> str:
    """Peek at whatever a port says when you connect, to identify the service."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            if s.connect_ex((host, port)) != 0:
                return ""
            try:
                data = s.recv(200)
                if data:
                    return data.decode("latin-1", "replace").strip()[:120]
            except socket.timeout:
                pass
            # Silent port: nudge it with an HTTP request.
            s.sendall(b"GET / HTTP/1.0\r\n\r\n")
            data = s.recv(200)
            return data.decode("latin-1", "replace").strip()[:120]
    except OSError:
        return ""


def listen_rtps(seconds: int, bind_ip: str) -> list[tuple[str, int, str]]:
    """Join the SPDP multicast group and collect RTPS announcements.

    Every DDS participant multicasts these on a timer, so a few seconds of
    listening is enough to prove DDS is running and identify the stack.
    """
    found: dict[tuple[str, int], str] = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    try:
        sock.bind(("", SPDP_PORT))
    except OSError as exc:
        log(f"  could not bind udp/{SPDP_PORT}: {exc}")
        return []
    try:
        mreq = struct.pack("4s4s", socket.inet_aton(SPDP_MULTICAST),
                           socket.inet_aton(bind_ip or "0.0.0.0"))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except OSError as exc:
        log(f"  could not join {SPDP_MULTICAST}: {exc}")

    sock.settimeout(1.0)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break
        if data[:4] == b"RTPS":
            vendor = VENDORS.get(data[6:8], f"unknown {data[6:8].hex()}")
            version = f"{data[4]}.{data[5]}"
            found[addr] = f"RTPS v{version}, {vendor}"
    sock.close()
    return [(ip, port, info) for (ip, port), info in found.items()]


def dds_topic_scan(iface: str, seconds: int = 15) -> list[tuple[str, str]]:
    """Enumerate every topic published on DDS domain 0.

    DDS keeps a set of builtin topics describing its own participants and
    writers, so reading DCPSPublication tells us exactly what the R1 offers --
    including whatever the camera topic turns out to be named. This beats
    guessing at API names from Go2 documentation.
    """
    import os

    # Pin Cyclone to the robot-facing interface; otherwise it may bind to a
    # different one and hear nothing.
    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS><Domain><General>"
        f"<Interfaces><NetworkInterface name='{iface}'/></Interfaces>"
        "<AllowMulticast>true</AllowMulticast>"
        "</General></Domain></CycloneDDS>"
    )
    from cyclonedds.builtin import BuiltinTopicDcpsPublication, BuiltinDataReader
    from cyclonedds.domain import DomainParticipant

    participant = DomainParticipant(0)
    reader = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)

    seen: dict[str, str] = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for sample in reader.take(N=100):
            name = getattr(sample, "topic_name", None)
            if name:
                seen[name] = getattr(sample, "type_name", "?")
        time.sleep(0.5)
    return sorted(seen.items())


def subnet_tcp_probe(prefix: str, ports=(22, 80, 554, 8080, 8000, 9000)) -> dict[str, list[int]]:
    """TCP-knock every host; finds machines that drop ICMP but answer TCP."""
    targets = [(f"{prefix}.{i}", p) for i in range(1, 255) for p in ports]
    hits: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=400) as pool:
        for (host, port), ok in zip(targets, pool.map(lambda t: tcp_open(*t, 0.4), targets)):
            if ok:
                hits.setdefault(host, []).append(port)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="192.168.12.1", help="robot IP")
    ap.add_argument("--rtps-seconds", type=int, default=25)
    ap.add_argument("--dds-seconds", type=int, default=15)
    ap.add_argument("--skip-full-scan", action="store_true")
    args = ap.parse_args()

    log(f"r1vision deep probe - {datetime.now():%Y-%m-%d %H:%M:%S}")
    my_ip = local_ip_on("")
    prefix = args.target.rsplit(".", 1)[0]
    log(f"  target : {args.target}")
    log(f"  this Mac: {my_ip or '(no address!)'}")
    if not my_ip:
        log("\n  Not on the robot's Wi-Fi. Join it and re-run.")
        REPORT.write_text("\n".join(_lines) + "\n")
        return 1

    # ------------------------------------------------------------------ 1
    section("1. RTPS / DDS discovery (the important one)")
    log(f"  Listening {args.rtps_seconds}s on {SPDP_MULTICAST}:{SPDP_PORT} ...")
    participants = listen_rtps(args.rtps_seconds, my_ip)
    if participants:
        for ip, port, info in participants:
            mine = "  (this Mac)" if ip == my_ip else ""
            log(f"  DDS participant: {ip}:{port}  {info}{mine}")
        robots = [p for p in participants if p[0] != my_ip]
        if robots:
            log(f"\n  --> The R1 IS running DDS. Use the SDK backend:")
            log(f"      python -m r1vision.view --source dds --interface en0")
        else:
            log("\n  Only this Mac announced. The robot is not on DDS domain 0,")
            log("  or its DDS is bound to an interface that does not reach here.")
    else:
        log("  No RTPS traffic at all.")

    # ------------------------------------------------------------------ 1b
    section("1b. DDS topic enumeration")
    try:
        topics = dds_topic_scan("en0", args.dds_seconds)
    except ImportError as exc:
        topics = []
        log(f"  SDK not available: {exc}")
    except Exception as exc:
        topics = []
        log(f"  topic scan failed: {type(exc).__name__}: {exc}")

    if topics:
        log(f"  {len(topics)} topic(s) published on domain 0:\n")
        for name, type_name in topics:
            log(f"    {name:<44} {type_name}")
        cam = [n for n, _ in topics
               if any(k in n.lower() for k in ("video", "image", "camera", "frame", "rgb"))]
        if cam:
            log(f"\n  --> camera-looking topics: {', '.join(cam)}")
    else:
        log("  No topics seen. Either the robot is not on DDS domain 0, or its")
        log("  SDK services are not started (some units need the app to enable them).")

    # ------------------------------------------------------------------ 1c
    section("1c. Unitree VideoClient RPC")
    try:
        from r1vision.sources import UnitreeDdsSource
        src = UnitreeDdsSource("en0")
        src.open()
        frame = src.read()
        src.close()
        if frame is not None and frame.size:
            log(f"  videohub ANSWERED - got a {frame.shape[1]}x{frame.shape[0]} frame")
            log("  --> python -m r1vision.view --source dds --interface en0")
        else:
            log("  videohub opened but returned no frame.")
    except Exception as exc:
        log(f"  videohub did not answer: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ 2
    open_ports: list[int] = []
    if not args.skip_full_scan:
        section(f"2. Full TCP scan of {args.target} (all 65535 ports)")
        log("  This takes 1-3 minutes ...")
        open_ports = full_tcp_scan(args.target)
        if open_ports:
            for port in open_ports:
                text = banner(args.target, port)
                log(f"  :{port:<6} {text or '(no banner)'}")
        else:
            log("  No TCP port on the whole range is open.")
            log("  Consistent with a UDP-only (DDS) control surface.")

    # ------------------------------------------------------------------ 3
    section(f"3. TCP probe across {prefix}.0/24")
    log("  Catching hosts that ignore ping ...")
    hits = subnet_tcp_probe(prefix)
    for host, ports in sorted(hits.items()):
        mine = "  (this Mac)" if host == my_ip else ""
        log(f"  {host}: {', '.join(str(p) for p in ports)}{mine}")
    if not any(h != my_ip for h in hits):
        log("  Nothing but this Mac answered TCP anywhere on the subnet.")

    # ------------------------------------------------------------------ 4
    section("4. Summary")
    # Consult the full scan too -- an earlier version concluded "no services"
    # while the section directly above it was listing an RTSP server.
    if participants and any(p[0] != my_ip for p in participants):
        log("  DDS is live -> use --source dds --interface en0.")
    elif open_ports:
        log(f"  No DDS, but {len(open_ports)} TCP port(s) open on {args.target}:")
        log(f"    {', '.join(str(p) for p in open_ports)}")
        log("  Run rtsp_probe.py to walk the mount points on any RTSP server.")
    else:
        log("  No DDS and no TCP services anywhere. The R1's AP is likely")
        log("  serving only the Unitree phone app over a proprietary protocol.")

    REPORT.write_text("\n".join(_lines) + "\n")
    print(f"\n--- written to {REPORT.resolve()} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
