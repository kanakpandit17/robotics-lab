#!/usr/bin/env python3
"""Root-free DDS presence sniffer. Joins the RTPS SPDP discovery multicast group and reports
any senders. If the robot publishes DDS on this interface, we WILL see its participant
announcements here, regardless of topic or message type.

Usage:  dds_sniff.py <iface_ip> <seconds>
        (domain 0 -> multicast 239.255.0.1 : 7400)
"""
import socket
import struct
import sys
import time

iface_ip = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
GROUP = "239.255.0.1"

seen = {}
for port in (7400, 7401):          # SPDP (7400) + user-traffic mcast (7401) for domain 0
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))
        mreq = struct.pack("4s4s", socket.inet_aton(GROUP), socket.inet_aton(iface_ip))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        s.setblocking(False)
        seen[port] = s
    except OSError as e:
        print(f"  port {port}: join failed ({e})", flush=True)

print(f"[SNIFF] listening {GROUP}:7400/7401 on {iface_ip} for {secs}s ...", flush=True)
senders = {}
t0 = time.time()
while time.time() - t0 < secs:
    for port, s in seen.items():
        try:
            while True:
                data, addr = s.recvfrom(65535)
                senders[addr[0]] = senders.get(addr[0], 0) + 1
        except (BlockingIOError, OSError):
            pass
    time.sleep(0.05)

if senders:
    print("[SNIFF] DDS/RTPS multicast senders DETECTED:", flush=True)
    for ip, c in sorted(senders.items()):
        print(f"    {ip}: {c} packets", flush=True)
    print("[SNIFF] -> robot (or some node) IS speaking DDS on this interface.", flush=True)
else:
    print("[SNIFF] NO DDS/RTPS multicast seen -> nothing is publishing DDS on this WiFi interface.", flush=True)
