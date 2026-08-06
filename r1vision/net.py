"""macOS network helpers for talking to a Unitree R1 over ethernet.

Unitree's onboard wired network lives on 192.168.123.0/24, so the Mac needs a
static address on that subnet before anything else will work.
"""

from __future__ import annotations

import re
import socket
import subprocess
from dataclasses import dataclass, field

UNITREE_SUBNET = "192.168.123"
HOST_IP_DEFAULT = "192.168.123.222"
HOST_NETMASK = "255.255.255.0"

# Addresses Unitree ships on across its fleet. The R1's onboard computer is the
# interesting one; the rest are motor-control boards and are listed so discovery
# can label what it finds instead of just printing bare IPs.
KNOWN_HOSTS = {
    "192.168.123.10": "motion control board",
    "192.168.123.15": "onboard switch / gateway",
    "192.168.123.161": "onboard computer (Go2/R1 typical)",
    "192.168.123.162": "onboard computer (alt)",
    "192.168.123.164": "Jetson Orin (G1/EDU typical)",
    "192.168.123.220": "user PC (Unitree default)",
}


@dataclass
class Interface:
    name: str
    mac: str = ""
    ipv4: str = ""
    netmask: str = ""
    status: str = ""
    service: str = ""  # networksetup service name, e.g. "USB 10/100/1000 LAN"

    @property
    def is_up(self) -> bool:
        return self.status == "active"

    @property
    def is_wifi(self) -> bool:
        return "Wi-Fi" in self.service or "Wireless" in self.service

    @property
    def is_direct_link(self) -> bool:
        """True when the subnet is /24 or narrower.

        A robot's own AP hands out a /24. A campus network like eduroam is a
        /19 or wider -- sweeping that would scan thousands of unrelated
        machines, so discovery refuses to.
        """
        parts = self.netmask.split(".")
        return len(parts) == 4 and parts[:3] == ["255", "255", "255"]

    @property
    def on_unitree_subnet(self) -> bool:
        return self.ipv4.startswith(UNITREE_SUBNET + ".")

    def describe(self) -> str:
        bits = [self.name]
        if self.service:
            bits.append(f"({self.service})")
        bits.append(f"[{self.status or 'unknown'}]")
        if self.ipv4:
            bits.append(f"ip={self.ipv4}")
        else:
            bits.append("ip=none")
        return " ".join(bits)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _hardware_ports() -> dict[str, str]:
    """Map BSD device name (en5) -> networksetup service name."""
    out = _run(["networksetup", "-listallhardwareports"])
    ports: dict[str, str] = {}
    service = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            service = line.split(":", 1)[1].strip()
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            if device:
                ports[device] = service
    return ports


def list_interfaces() -> list[Interface]:
    """Parse `ifconfig -a` into structured interfaces."""
    out = _run(["ifconfig", "-a"])
    ports = _hardware_ports()
    interfaces: list[Interface] = []
    current: Interface | None = None

    for line in out.splitlines():
        head = re.match(r"^([a-zA-Z0-9_.]+):\s+flags=", line)
        if head:
            current = Interface(name=head.group(1))
            current.service = ports.get(current.name, "")
            interfaces.append(current)
            continue
        if current is None:
            continue
        line = line.strip()
        if line.startswith("ether "):
            current.mac = line.split()[1]
        elif line.startswith("inet ") and not line.startswith("inet6"):
            parts = line.split()
            current.ipv4 = parts[1]
            if "netmask" in parts:
                raw = parts[parts.index("netmask") + 1]
                current.netmask = _hex_netmask(raw)
        elif line.startswith("status:"):
            current.status = line.split(":", 1)[1].strip()

    return interfaces


def _hex_netmask(raw: str) -> str:
    if raw.startswith("0x"):
        value = int(raw, 16)
        return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))
    return raw


def _rank(itf: Interface) -> tuple:
    """Sort key: lower is a better candidate for reaching the robot."""
    service = itf.service.lower()
    # Macs expose "Thunderbolt 1/2/3" bridge ports that look like ethernet but
    # have no jack. A real adapter says "Ethernet" or "LAN" in its port name.
    is_bridge = service.startswith("thunderbolt") and "ethernet" not in service
    is_real_ethernet = ("ethernet" in service or "lan" in service) and not is_bridge
    return (
        not itf.on_unitree_subnet,  # already talking to the robot wins outright
        not itf.is_up,              # then anything with a live link
        itf.is_wifi,                # a live wired link beats a live wireless one
        not is_real_ethernet,       # then a genuine ethernet adapter
        is_bridge,                  # bridge ports last
        itf.name,
    )


def usable_candidates() -> list[Interface]:
    """Every interface that could reach the robot, best first.

    Includes Wi-Fi: the R1 has no ethernet port, so wireless is the normal
    path to it. Virtual interfaces (bridge/utun/awdl/llw) are excluded.
    """
    skip_prefixes = ("lo", "gif", "stf", "utun", "awdl", "llw", "bridge", "ap", "anpi")
    out = [
        itf for itf in list_interfaces()
        if itf.name.startswith("en") and not itf.name.startswith(skip_prefixes)
    ]
    out.sort(key=_rank)
    return out


def ethernet_candidates() -> list[Interface]:
    """Wired interfaces only, best first."""
    return [itf for itf in usable_candidates() if not itf.is_wifi]


def pick_interface(explicit: str | None = None) -> Interface | None:
    """Choose the interface to use, preferring one already on the robot subnet."""
    candidates = usable_candidates()
    if explicit:
        for itf in candidates:
            if itf.name == explicit:
                return itf
        return Interface(name=explicit)  # trust the user; may be a fresh adapter
    return candidates[0] if candidates else None  # already ranked best-first


def static_ip_command(itf: Interface, host_ip: str = HOST_IP_DEFAULT) -> list[str]:
    """The command that puts `itf` on the robot subnet.

    Prefers `networksetup` (persists across reboots and survives the GUI) and
    falls back to `ifconfig` when the interface has no network service.
    """
    if itf.service:
        return [
            "sudo", "networksetup", "-setmanual",
            itf.service, host_ip, HOST_NETMASK,
        ]
    return ["sudo", "ifconfig", itf.name, "inet", host_ip, "netmask", HOST_NETMASK]


def tcp_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """True if a TCP connect to host:port succeeds."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def local_ip_for(target: str) -> str:
    """Which local address the kernel would use to reach `target`."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.5)
            sock.connect((target, 9))
            return sock.getsockname()[0]
    except OSError:
        return ""
