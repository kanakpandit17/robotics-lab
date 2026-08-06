#!/usr/bin/env python3
"""Run this while joined to the R1's Wi-Fi. No internet needed.

Works out why there is no IP, gets one if it can, then finds the robot and its
video stream. Everything is written to field-report.txt so it can be read back
later from a normal network.

    python field_probe.py

May ask for your sudo password: if the robot's AP runs no DHCP server, the only
way onto its subnet is to assign a static address.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

from r1vision import net  # noqa: E402
from r1vision.discover import PORT_HINTS, probe_streams, scan_ports, sweep, Host

REPORT = Path("field-report.txt")
_lines: list[str] = []

# Subnets Unitree APs are known to hand out on, tried in order if DHCP is dead.
# .1 is the robot itself on an AP; we claim .222 to stay clear of its pool.
STATIC_FALLBACKS = [
    ("192.168.12", "192.168.12.222"),    # Go1-style AP
    ("192.168.123", "192.168.123.222"),  # Unitree onboard segment
    ("192.168.4", "192.168.4.222"),      # generic embedded AP
]


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def run(cmd: list[str], timeout: int = 15) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (res.stdout or "") + (res.stderr or "")
    except (subprocess.SubprocessError, OSError) as exc:
        return f"<failed: {exc}>"


def current_ssid(iface: str = "en0") -> str:
    out = run(["ipconfig", "getsummary", iface])
    for line in out.splitlines():
        if " SSID : " in line:
            return line.split(" SSID : ", 1)[1].strip()
    return ""


def iface_ip(iface: str = "en0") -> str:
    return run(["ipconfig", "getifaddr", iface]).strip()


def section(title: str) -> None:
    log()
    log("=" * 68)
    log(title)
    log("=" * 68)


def main() -> int:
    log(f"r1vision field probe - {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ---------------------------------------------------------------- 1
    section("1. Wi-Fi association")
    ssid = current_ssid()
    ip = iface_ip()
    log(f"  SSID : {ssid or '(not associated)'}")
    log(f"  IP   : {ip or '(none)'}")

    if not ssid:
        log()
        log("  NOT ASSOCIATED. The password was rejected or the AP went away.")
        log("  An SSID appears here only after the handshake succeeds, so this is")
        log("  a credentials/range problem, not a DHCP one.")
        log("  Check the sticker on the robot for the real AP password.")
        write_report()
        return 1

    if "R1" not in ssid.upper():
        log(f"\n  WARNING: joined '{ssid}', which does not look like the robot's AP.")

    # ---------------------------------------------------------------- 2
    section("2. DHCP")
    if not ip:
        log("  No address yet. Waiting up to 30s for a lease ...")
        run(["ipconfig", "set", "en0", "DHCP"], timeout=10)
        for i in range(15):
            time.sleep(2)
            ip = iface_ip()
            if ip:
                log(f"  Got {ip} after ~{(i + 1) * 2}s")
                break
    if ip:
        router = run(["ipconfig", "getoption", "en0", "router"]).strip()
        log(f"  address : {ip}")
        log(f"  router  : {router or '(none given)'}")
    else:
        log("  DHCP produced nothing. The AP likely runs no DHCP server,")
        log("  so a static address is the only way onto its subnet.")

    # ---------------------------------------------------------------- 3
    if not ip:
        section("3. Static address fallback (needs sudo)")
        for prefix, host_ip in STATIC_FALLBACKS:
            log(f"\n  Trying {host_ip}/24 ...")
            out = run(["sudo", "networksetup", "-setmanual",
                       "Wi-Fi", host_ip, "255.255.255.0", f"{prefix}.1"], timeout=25)
            if out.strip():
                log(f"    {out.strip()[:200]}")
            time.sleep(2)
            gateway = f"{prefix}.1"
            alive = subprocess.run(
                ["ping", "-c", "2", "-W", "500", gateway],
                capture_output=True,
            ).returncode == 0
            log(f"    ping {gateway}: {'REPLY' if alive else 'no reply'}")
            if alive:
                ip = host_ip
                log(f"  -> {prefix}.0/24 is live. Keeping this address.")
                break
        else:
            log("\n  No candidate subnet answered. Returning Wi-Fi to DHCP.")
            run(["sudo", "networksetup", "-setdhcp", "Wi-Fi"], timeout=20)

    if not ip:
        log("\n  Could not get onto the robot's network at all.")
        write_report()
        return 1

    prefix = ip.rsplit(".", 1)[0]

    # ---------------------------------------------------------------- 4
    section(f"4. Sweeping {prefix}.0/24")
    hosts_ips = sweep(prefix)
    log(f"  {len(hosts_ips)} host(s): {', '.join(hosts_ips) or '(none)'}")

    # This Mac answers its own ping and runs SSH/Bonjour; without this it looks
    # exactly like a promising robot and sends you off to ssh into yourself.
    if ip in hosts_ips:
        hosts_ips.remove(ip)
        log(f"  excluding {ip} (this Mac)")

    gateway = run(["ipconfig", "getoption", "en0", "router"]).strip()
    if gateway and gateway not in hosts_ips:
        log(f"  adding gateway {gateway} (did not answer ping but is the AP)")
        hosts_ips.append(gateway)

    if not hosts_ips:
        log("\n  Associated and addressed, but nothing on the subnet responds.")
        write_report()
        return 1

    # ---------------------------------------------------------------- 5
    section("5. Open ports")
    hosts: list[Host] = []
    for host_ip in hosts_ips:
        host = Host(ip=host_ip, label=net.KNOWN_HOSTS.get(host_ip, ""))
        host.open_ports = scan_ports(host_ip, list(PORT_HINTS))
        hosts.append(host)
        log(f"  {host_ip}" + (f"  <- {host.label}" if host.label else ""))
        for port in host.open_ports:
            log(f"      :{port:<6} {PORT_HINTS.get(port, '')}")
        if not host.open_ports:
            log("      (none of the common ports open)")

    # ---------------------------------------------------------------- 6
    section("6. Video transports")
    found = []
    for host in hosts:
        for hit in probe_streams(host):
            log(f"  {hit.kind:<10} : {hit.url}   [{hit.detail}]")
            found.append(hit)
    if not found:
        log("  No HTTP/RTSP video endpoint answered.")

    # ---------------------------------------------------------------- 7
    section("7. mDNS / Bonjour")
    try:
        proc = subprocess.Popen(["dns-sd", "-B", "_services._dns-sd._udp", "local"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        time.sleep(6)
        proc.terminate()
        out = proc.communicate(timeout=5)[0] or ""
        services = {line.split()[-1] for line in out.splitlines()
                    if line.strip() and "_tcp" in line or "_udp" in line}
        log("  " + (", ".join(sorted(services)) if services else "(nothing advertised)"))
    except Exception as exc:
        log(f"  (mDNS browse failed: {exc})")

    # ---------------------------------------------------------------- 8
    section("8. Next")
    if found:
        for hit in found:
            log(f"    python -m r1vision.view --source {hit.url}")
    else:
        ssh_hosts = [h.ip for h in hosts if 22 in h.open_ports]
        if ssh_hosts:
            log("  SSH is open. Log in and look for the camera:")
            for h in ssh_hosts:
                log(f"    ssh unitree@{h}      # try password 123 or Unitree0408")
            log("    ls /dev/video*  &&  v4l2-ctl --list-devices")
        else:
            log("  No video and no SSH. The R1's AP may only serve the Unitree app.")
            log("  Pair with the app over Bluetooth and put the robot on a private")
            log("  router you also control -- not eduroam (client isolation).")

    write_report()
    return 0 if found else 1


def write_report() -> None:
    REPORT.write_text("\n".join(_lines) + "\n")
    print(f"\n--- written to {REPORT.resolve()} ---")


if __name__ == "__main__":
    sys.exit(main())
