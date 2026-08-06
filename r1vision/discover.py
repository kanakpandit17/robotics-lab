"""Find the R1 on the wire and work out how to get video off it.

Run this once after plugging in the ethernet cable:

    python -m r1vision.discover

It checks the link, sweeps the Unitree subnet, port-scans whatever answers,
then actively probes each plausible video transport and prints the exact
`r1vision.view` command for anything that produced a real frame.
"""

from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import net

# Ports worth knocking on, and what finding one open would mean.
PORT_HINTS = {
    22: "SSH (onboard computer)",
    80: "HTTP",
    443: "HTTPS",
    554: "RTSP",
    1935: "RTMP",
    5000: "HTTP alt / dev server",
    5001: "HTTP alt",
    7447: "Zenoh router",
    8000: "HTTP alt",
    8080: "HTTP alt (MJPEG common)",
    8081: "HTTP alt (MJPEG common)",
    8550: "RTSP alt",
    8551: "RTSP (GStreamer -- Unitree R1 video)",
    8552: "RTSP alt",
    8553: "RTSP alt",
    8554: "RTSP alt",
    8888: "HTTP alt",
    9000: "HTTP alt",
    9090: "rosbridge websocket",
    9991: "Unitree app control channel",
    11311: "ROS 1 master",
}

# The R1 serves video on 8551, which is nobody's idea of a standard RTSP port.
# Probe the whole neighbourhood rather than just 554/8554.
RTSP_PORTS = (554, 8550, 8551, 8552, 8553, 8554)

HTTP_PATHS = [
    "/stream", "/video", "/video_feed", "/videostream.cgi",
    "/mjpg/video.mjpg", "/?action=stream", "/stream.mjpg", "/cam.mjpg",
    "/live", "/image.jpg", "/snapshot", "/",
]

RTSP_PATHS = [
    "/", "/live", "/stream", "/video", "/h264", "/main", "/ch0",
    "/cam/realmonitor?channel=1&subtype=0", "/Streaming/Channels/101",
]


@dataclass
class Host:
    ip: str
    label: str = ""
    open_ports: list[int] = field(default_factory=list)


@dataclass
class StreamHit:
    url: str
    kind: str
    detail: str = ""


def ping(ip: str, timeout_ms: int = 300) -> bool:
    try:
        res = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout_ms), "-t", "1", ip],
            capture_output=True, timeout=3,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def arp_table() -> set[str]:
    """Hosts the kernel already knows about -- catches ICMP-silent devices."""
    out = net._run(["arp", "-an"])
    found = set()
    for line in out.splitlines():
        if "(" in line and ")" in line and "incomplete" not in line:
            found.add(line.split("(")[1].split(")")[0])
    return found


def sweep(prefix: str, workers: int = 96) -> list[str]:
    targets = [f"{prefix}.{i}" for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        alive = list(pool.map(ping, targets))
    hits = {ip for ip, ok in zip(targets, alive) if ok}
    hits |= {ip for ip in arp_table() if ip.startswith(prefix + ".")}
    return sorted(hits, key=lambda ip: int(ip.rsplit(".", 1)[1]))


def scan_ports(ip: str, ports: list[int], workers: int = 32) -> list[int]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda p: (p, net.tcp_open(ip, p, 0.5)), ports))
    return [p for p, ok in results if ok]


def probe_rtsp(ip: str, port: int, path: str, timeout: float = 2.0) -> StreamHit | None:
    """Send an RTSP DESCRIBE and see whether we get a session description back."""
    url = f"rtsp://{ip}:{port}{path}"
    request = (
        f"DESCRIBE {url} RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "User-Agent: r1vision\r\n"
        "Accept: application/sdp\r\n\r\n"
    )
    try:
        with socket.create_connection((ip, port), timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request.encode())
            reply = sock.recv(2048).decode("latin-1", "replace")
    except OSError:
        return None
    first = reply.split("\r\n", 1)[0]
    if "RTSP/1.0 200" in first:
        return StreamHit(url, "rtsp", "DESCRIBE 200 OK")
    # 401 means the stream exists but wants credentials -- still a real find.
    if "401" in first or "Unauthorized" in reply:
        return StreamHit(url, "rtsp", "needs credentials (401)")
    return None


def probe_http(ip: str, port: int, path: str, timeout: float = 2.0) -> StreamHit | None:
    """Fetch a little and decide whether it smells like MJPEG or a JPEG still."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    url = f"http://{ip}:{port}{path}"
    try:
        resp = urlopen(Request(url, headers={"User-Agent": "r1vision"}), timeout=timeout)
    except (HTTPError, URLError, OSError, ValueError):
        return None
    try:
        ctype = resp.headers.get("Content-Type", "")
        if "multipart" in ctype:
            return StreamHit(url, "mjpeg", ctype)
        head = resp.read(4096)
        if head.startswith(b"\xff\xd8"):
            kind = "mjpeg" if "image" in ctype else "jpeg-still"
            return StreamHit(url, kind, ctype or "jpeg magic bytes")
        if b"\xff\xd8" in head and b"boundary" in head.lower():
            return StreamHit(url, "mjpeg", "boundary + JPEG in body")
    finally:
        resp.close()
    return None


def probe_streams(host: Host) -> list[StreamHit]:
    hits: list[StreamHit] = []
    rtsp_ports = [p for p in host.open_ports if p in RTSP_PORTS]
    http_ports = [p for p in host.open_ports if p in (80, 5000, 5001, 8000, 8080, 8081, 8888, 9000)]

    jobs = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        for port in rtsp_ports:
            for path in RTSP_PATHS:
                jobs.append(pool.submit(probe_rtsp, host.ip, port, path))
        for port in http_ports:
            for path in HTTP_PATHS:
                jobs.append(pool.submit(probe_http, host.ip, port, path))
        for job in jobs:
            hit = job.result()
            if hit:
                hits.append(hit)

    # One good stream per port is enough; extra paths are usually aliases.
    seen: set[tuple[str, str]] = set()
    unique = []
    for hit in sorted(hits, key=lambda h: (h.kind != "mjpeg", h.url)):
        key = (hit.url.split("/")[2], hit.kind)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def try_dds(interface: str) -> StreamHit | None:
    from .sources import SourceError, UnitreeDdsSource

    src = UnitreeDdsSource(interface)
    try:
        src.open()
    except SourceError as exc:
        print(f"  dds        : no  ({exc})")
        return None
    except Exception as exc:  # SDK raises assorted DDS errors
        print(f"  dds        : no  ({type(exc).__name__}: {exc})")
        return None
    finally:
        src.close()
    print("  dds        : YES (videohub answered)")
    return StreamHit("dds", "dds", f"interface {interface}")


def candidate_subnets(args) -> list[str]:
    """Which /24s to sweep.

    Always the Unitree default, plus any subnet this Mac sits directly on --
    but never a wide one. On eduroam the Mac holds a /19, and sweeping that
    would touch thousands of unrelated university machines.
    """
    if args.subnet != "auto":
        return [args.subnet]

    subnets = [net.UNITREE_SUBNET]
    for itf in net.usable_candidates():
        if not itf.is_up or not itf.ipv4 or itf.ipv4.startswith("127."):
            continue
        prefix = itf.ipv4.rsplit(".", 1)[0]
        if prefix in subnets:
            continue
        if not itf.is_direct_link:
            print(f"\n  skipping {prefix}.0 on {itf.name} (netmask {itf.netmask}): wider"
                  f" than /24, so it is a shared network, not a direct robot link.")
            print("  Pass --subnet explicitly if you really want to scan it.")
            continue
        subnets.append(prefix)
    return subnets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find the Unitree R1 and its video stream")
    ap.add_argument("-i", "--interface", help="ethernet interface, e.g. en5")
    ap.add_argument("--subnet", default="auto",
                    help="first three octets to sweep, or 'auto' (default) to use "
                         "192.168.123 plus any /24 this Mac is directly on")
    ap.add_argument("--skip-dds", action="store_true", help="do not try the SDK backend")
    ap.add_argument("--host", help="skip the sweep and probe just this IP")
    args = ap.parse_args(argv)

    print("=" * 66)
    print("1. Network interfaces")
    print("=" * 66)
    candidates = net.usable_candidates()
    if not candidates:
        print("  No usable network interface found.")
        return 1
    for cand in candidates:
        kind = "wifi " if cand.is_wifi else "wired"
        print(f"  [{kind}] {cand.describe()}")

    itf = net.pick_interface(args.interface)
    if itf is None:
        return 1
    print(f"\n  using: {itf.name}" + ("  (Wi-Fi)" if itf.is_wifi else ""))

    if not itf.is_up and not args.host:
        print(f"\n  !! {itf.name} has no link.")
    if itf.is_wifi and not itf.on_unitree_subnet:
        print(f"\n  note: {itf.name} is on {itf.ipv4}/{itf.netmask}.")
        print("        The R1 has no ethernet port -- it is a Wi-Fi 6 device. Either join")
        print("        the robot's own access point, or put the robot on this same network")
        print("        with the Unitree app, then re-run.")
    elif not itf.is_wifi and not itf.on_unitree_subnet:
        # Port names contain spaces ("Ethernet Adapter (en5)"), so quote them
        # or the printed command is not copy-pasteable.
        cmd = shlex.join(net.static_ip_command(itf))
        print(f"\n  !! {itf.name} is not on {net.UNITREE_SUBNET}.x.")
        print(f"     If this really is a wired Unitree link, fix it with:\n       {cmd}")

    subnets = candidate_subnets(args)

    print()
    print("=" * 66)
    print(f"2. Sweeping {', '.join(s + '.0/24' for s in subnets)}")
    print("=" * 66)
    if args.host:
        ips = [args.host]
    else:
        ips = []
        for prefix in subnets:
            found = sweep(prefix)
            print(f"  {prefix}.0/24 -> {len(found)} host(s)")
            ips.extend(found)
    if not ips:
        print("\n  Nothing answered. Most likely causes:")
        print("    - this Mac and the R1 are not on the same network")
        print("    - robot still booting (give it ~60s after power-on)")
        print("    - the R1 is on its own Wi-Fi AP that you have not joined yet")
        return 1

    hosts: list[Host] = []
    for ip in ips:
        host = Host(ip=ip, label=net.KNOWN_HOSTS.get(ip, ""))
        host.open_ports = scan_ports(ip, list(PORT_HINTS))
        hosts.append(host)
        label = f"  {ip}" + (f"  <- {host.label}" if host.label else "")
        print(label)
        for port in host.open_ports:
            print(f"      :{port:<6} {PORT_HINTS.get(port, '')}")
        if not host.open_ports:
            print("      (no common ports open)")

    print()
    print("=" * 66)
    print("3. Probing video transports")
    print("=" * 66)
    found: list[StreamHit] = []
    for host in hosts:
        hits = probe_streams(host)
        for hit in hits:
            print(f"  {hit.kind:<10} : {hit.url}   [{hit.detail}]")
        found.extend(hits)
    if not found:
        print("  No HTTP/RTSP video endpoint answered.")

    if not args.skip_dds:
        dds_hit = try_dds(itf.name)
        if dds_hit:
            found.append(dds_hit)

    print()
    print("=" * 66)
    print("4. What to run")
    print("=" * 66)
    if not found:
        print("  Nothing streamed. Next step is to look from the inside:")
        ssh_hosts = [h.ip for h in hosts if 22 in h.open_ports]
        for ip in ssh_hosts or ["192.168.123.161"]:
            print(f"    ssh unitree@{ip}       # password is usually 123 or Unitree0408")
        print("    then on the robot:  ls /dev/video*  &&  v4l2-ctl --list-devices")
        print("  Once you know the device, push a stream to this Mac:")
        mac_ip = net.local_ip_for("192.168.123.161") or net.HOST_IP_DEFAULT
        print(f"    (robot) ffmpeg -f v4l2 -i /dev/video0 -vcodec libx264 -tune zerolatency \\")
        print(f"              -f mpegts udp://{mac_ip}:5000")
        print(f"    (mac)   python -m r1vision.view --source udp://@:5000")
        return 1

    for hit in found:
        if hit.kind == "dds":
            print(f"    python -m r1vision.view --source dds --interface {itf.name}")
        elif hit.kind == "jpeg-still":
            print(f"    python -m r1vision.view --source {hit.url}   # still endpoint, will poll")
        else:
            print(f"    python -m r1vision.view --source {hit.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
