#!/usr/bin/env bash
# DDS reachability test to the R1 over its WiFi AP. Run from WSL while the PC is on the robot AP.
echo "== WSL sees robot subnet (192.168.12.x)? =="
ip -4 addr show | grep -E 'inet 192\.168\.12' || echo "  NO 192.168.12.x on any WSL interface (mirrored networking didn't pick up the WiFi)"
echo "== routes to robot subnet =="
ip route | grep -E '192\.168\.12' || echo "  no route"
echo "== ping robot gateway .1 from WSL =="
ping -c 2 -W 2 192.168.12.1 2>&1 | tail -2
echo
echo "== DDS test A: default/multicast discovery, 8s =="
~/r1bridge/.venv/bin/python /mnt/c/Users/aimat/Desktop/humanoid/r1_read_state.py --secs 8 2>&1 | tail -8
echo
echo "== DDS test B: unicast peer -> 192.168.12.1, 8s =="
export CYCLONEDDS_URI='<CycloneDDS><Domain><Discovery><Peers><Peer address="192.168.12.1"/></Peers></Discovery></Domain></CycloneDDS>'
~/r1bridge/.venv/bin/python /mnt/c/Users/aimat/Desktop/humanoid/r1_read_state.py --secs 8 2>&1 | tail -8
echo "== dds test done =="
