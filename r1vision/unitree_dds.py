"""Make the Unitree SDK talk to a robot on another subnet.

`ChannelFactoryInitialize` hardcodes a CycloneDDS config containing only an
interface name -- no <Peers> -- and passes it straight to Domain(), which
overrides CYCLONEDDS_URI. Discovery is therefore multicast-only.

That is fine over ethernet on a flat 192.168.123.x network, which is what
Unitree assume. It fails here: we sit on the robot's Wi-Fi AP (192.168.12.x)
and reach the locomotion computer (192.168.123.161) through a route, where
multicast does not follow.

Patching the template in before initialisation adds unicast peer discovery,
which does cross a routed hop.
"""

from __future__ import annotations

CONFIG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
            </Interfaces>
            <AllowMulticast>{multicast}</AllowMulticast>
        </General>
        <Discovery>
            <ParticipantIndex>auto</ParticipantIndex>
            <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
            <Peers>
{peers}
            </Peers>
        </Discovery>
        <Tracing>
            <Verbosity>{verbosity}</Verbosity>
            <OutputFile>{logfile}</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>"""

# Sweeping ~94 unicast peers means ~94 "ddsi_udp_conn_write failed" lines every
# few seconds whenever one is unreachable -- which is normal and expected, since
# most of those ports have nothing behind them. At severe verbosity that noise
# buries the actual output, so it goes to a file instead of the terminal.
DDS_LOG = "/tmp/r1vision-cyclonedds.log"


import contextlib
import os
import sys
from pathlib import Path


@contextlib.contextmanager
def quiet_dds_stderr(path: str = "/tmp/r1vision-cyclonedds.log"):
    """Send CycloneDDS's chatter to a file instead of the terminal.

    `<Tracing><Verbosity>none</Verbosity></Tracing>` does NOT suppress
    "ddsi_udp_conn_write ... failed with retcode -1"; those are emitted below
    the verbosity control. Sweeping ~94 peers, most of which have nothing
    listening, yields one such line per port per cycle and buries real output.
    Redirecting fd 2 is the only thing that reliably silences it.

    Our own messages go to stdout, so they survive this.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    handle = open(path, "a")
    try:
        os.dup2(handle.fileno(), 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        handle.close()


def peers_from_file(path: str = "dds-peers.txt") -> list[str]:
    """Measured peer list from `dds_udp_probe.py --sweep`, if it exists.

    Beats guessing at a participant-index range: these are ports observed to be
    bound on the robot.
    """
    candidate = Path(path)
    if not candidate.exists():
        candidate = Path(__file__).resolve().parent.parent / path
    if not candidate.exists():
        return []
    entries = [
        line.strip() for line in candidate.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return entries


def default_peers(hosts: tuple[str, ...] = ("192.168.123.161", "192.168.12.1"),
                  domain: int = 0, max_participant: int = 99) -> list[str]:
    """Every plausible SPDP locator on the robot.

    The R1 runs several DDS processes, each its own participant on its own
    unicast port (7400 + 250*domain + 10 + 2*index), and discovery lands on a
    different subset run to run. Sweeping the index range finds whichever one
    hosts the service we want.
    """
    measured = peers_from_file()
    if measured:
        return measured
    base = 7400 + 250 * domain
    ports = [base] + [base + 10 + 2 * i for i in range(max_participant + 1)]
    return [f"{host}:{port}" for host in hosts for port in ports]


def render_config(iface_placeholder: str, peers: list[str], multicast: bool = True,
                  verbose: bool = False) -> str:
    """Build a CycloneDDS config with peers, quiet by default."""
    peer_xml = "\n".join(f'                <Peer address="{p}"/>' for p in peers)
    return CONFIG_TEMPLATE.format(
        peers=peer_xml,
        multicast="true" if multicast else "false",
        verbosity="severe" if verbose else "none",
        logfile=DDS_LOG,
    ).replace("$__IF_NAME__$", iface_placeholder)


ROBOT_SUBNET = "192.168.123."


def interfaces() -> list[tuple[str, str]]:
    """(name, ipv4) for every interface that currently has an address."""
    import subprocess

    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    found, name = [], ""
    for line in out.splitlines():
        if line and not line[0].isspace():
            name = line.split(":")[0]
        elif line.strip().startswith("inet ") and "127.0.0.1" not in line:
            found.append((name, line.split()[1]))
    return found


def robot_iface(preferred: str = "auto") -> str:
    """The interface facing the robot, found by address rather than by name.

    The USB-ethernet adapter does not keep a stable name: it has come back as
    en5, en6 and en7 across replugs on this machine, and every tool hardcoding
    en6 then fails with an error that says nothing about why. What is stable is
    the robot's subnet, so look for that instead.
    """
    if preferred != "auto":
        return preferred
    for name, ip in interfaces():
        if ip.startswith(ROBOT_SUBNET):
            return name
    raise RuntimeError(
        "no interface on " + ROBOT_SUBNET + "x -- the robot's ethernet adapter "
        "is not connected.\n  Interfaces with addresses right now: "
        + ", ".join(f"{n} ({ip})" for n, ip in interfaces()))


def init_channel(iface: str, peers: list[str] | None = None,
                 domain: int = 0, multicast: bool = True) -> None:
    """Initialise the Unitree channel factory, optionally with unicast peers.

    Call this instead of ChannelFactoryInitialize. Safe to call once per
    process; the factory is a singleton and ignores later calls.
    """
    iface = robot_iface(iface)

    # CycloneDDS reports a missing interface as "channel factory init error",
    # having already written the real reason to stderr -- which is redirected to
    # a log file here. Checking first turns a recurring mystery into a sentence.
    live = {name for name, _ in interfaces()}
    if iface not in live:
        raise RuntimeError(
            f"interface {iface!r} does not exist. Present: "
            + ", ".join(f"{n} ({ip})" for n, ip in interfaces())
            + "\n  The robot's USB-ethernet adapter has probably dropped; "
              "reseat it, then re-run.")

    from unitree_sdk2py.core import channel as _channel

    if peers:
        peer_xml = "\n".join(
            f'                <Peer address="{p}"/>' for p in peers
        )
        # channel.py did `from .channel_config import ChannelConfigHasInterface`,
        # so the name lives in the channel module's globals and is looked up at
        # call time -- patch it there, not in channel_config.
        _channel.ChannelConfigHasInterface = CONFIG_TEMPLATE.format(
            peers=peer_xml,
            multicast="true" if multicast else "false",
            verbosity="none",
            logfile=DDS_LOG,
        )

    _channel.ChannelFactoryInitialize(domain, iface)
