#!/usr/bin/env bash
# Serve the robot's camera to the Quest on the default HTTPS port.
#
#   ./vr.sh
#
# Port 443 means the URL has no ":8080" to type in the headset -- just
# https://<ip>. Ports below 1024 need root, hence the sudo.
#
# Any quest_stream.py flag passes through:  ./vr.sh --quality 60
#
# Video and arm gestures only by default. To drive the robot from the
# thumbsticks, ./vr.sh --walk -- see WALKING.md, and run r1_walk.py first.

set -euo pipefail
cd "$(dirname "$0")"

echo "Serving on port 443 so the headset URL is just  https://<ip>"
echo "(sudo is only needed because ports under 1024 are privileged)"
echo

exec sudo ./.venv/bin/python quest_stream.py --port 443 "$@"
