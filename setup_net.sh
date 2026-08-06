#!/usr/bin/env bash
# Put this Mac on the Unitree wired subnet (192.168.123.0/24).
#
#   ./setup_net.sh            # auto-pick the ethernet interface
#   ./setup_net.sh en5        # or name it
#   ./setup_net.sh en5 --undo # hand the interface back to DHCP
#
# Unitree's onboard network is a fixed 192.168.123.x segment with no DHCP
# server, so the Mac has to be given a static address on it manually.

set -euo pipefail

HOST_IP="192.168.123.222"   # avoids Unitree's own .10/.15/.161/.164/.220
NETMASK="255.255.255.0"

iface="${1:-}"
mode="${2:-}"

# Ethernet interfaces only: skip Wi-Fi, Thunderbolt bridges and virtual links.
find_iface() {
  networksetup -listallhardwareports \
    | awk '/Hardware Port:/{port=$0} /^Device:/{print port"|"$2}' \
    | grep -iE 'ethernet|LAN' \
    | head -1 | cut -d'|' -f2
}

if [[ -z "$iface" ]]; then
  iface="$(find_iface || true)"
  if [[ -z "$iface" ]]; then
    echo "No ethernet interface found. Plug in your USB-C/Thunderbolt adapter." >&2
    echo "Interfaces present:" >&2
    networksetup -listallhardwareports >&2
    exit 1
  fi
  echo "Auto-detected interface: $iface"
fi

# networksetup works on the human-readable service name, not the BSD device.
service="$(networksetup -listallhardwareports \
  | awk -v dev="$iface" '/Hardware Port:/{port=substr($0,16)} /^Device:/{if ($2==dev) print port}')"

if [[ -z "$service" ]]; then
  echo "No network service maps to $iface; falling back to ifconfig." >&2
  echo "This does NOT survive a reboot or an unplug." >&2
  sudo ifconfig "$iface" inet "$HOST_IP" netmask "$NETMASK"
  echo "Set $iface to $HOST_IP/$NETMASK"
  exit 0
fi

if [[ "$mode" == "--undo" ]]; then
  echo "Returning '$service' to DHCP ..."
  sudo networksetup -setdhcp "$service"
  echo "Done."
  exit 0
fi

echo "Setting '$service' ($iface) to $HOST_IP / $NETMASK ..."
sudo networksetup -setmanual "$service" "$HOST_IP" "$NETMASK"

echo
echo "Current address:"
ipconfig getifaddr "$iface" || echo "  (no address yet - check the cable and that the R1 is powered on)"

echo
echo "Next:  python -m r1vision.discover"
