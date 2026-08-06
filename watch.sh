#!/usr/bin/env bash
# See what the R1 is looking at. Join the robot's Wi-Fi first.
#
#   ./watch.sh              # live view
#   ./watch.sh --grid       # any r1vision.view flag passes through
#
# Confirmed on this unit 2026-07-27: GStreamer RTSP on :8551, mount /video,
# H.264 High@4.0, 1280x720, already upright (no rotation needed).

set -euo pipefail
cd "$(dirname "$0")"

HOST="${R1_HOST:-192.168.12.1}"
PORT="${R1_PORT:-8551}"
URL="rtsp://${HOST}:${PORT}/video"

if ! ipconfig getifaddr en0 >/dev/null 2>&1; then
  echo "No Wi-Fi address. Join the robot's AP (R1_03302_c33840b2) first." >&2
  exit 1
fi

ssid="$(ipconfig getsummary en0 2>/dev/null | awk -F' SSID : ' '/ SSID : /{print $2; exit}')"
if [[ "$ssid" != *R1_* ]]; then
  echo "Warning: joined '${ssid}', not the robot's AP. Trying anyway ..." >&2
fi

echo "Streaming ${URL}"
exec ./.venv/bin/python -m r1vision.view --source "$URL" "$@"
