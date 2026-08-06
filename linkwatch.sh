#!/usr/bin/env bash
# Live ethernet link status. Leave it running while you try ports/cables.
#
#   ./linkwatch.sh
#
# "active" means a real electrical link. Anything else means nothing is
# driving the far end of the cable, and no IP configuration will help.

set -uo pipefail

echo "Watching wired interfaces. Ctrl-C to stop."
echo "Plug/unplug and watch the status change."
echo

prev=""
while true; do
  line=""
  for d in en1 en2 en3 en4 en5 en6 en7 en8; do
    st=$(ifconfig "$d" 2>/dev/null | awk '/status:/{print $2}')
    [ -z "$st" ] && continue
    ip=$(ipconfig getifaddr "$d" 2>/dev/null)
    if [ "$st" = "active" ]; then
      line+="  $d=ACTIVE${ip:+ ip=$ip}"
    else
      line+="  $d=$st"
    fi
  done

  if [ "$line" != "$prev" ]; then
    printf "%s %s\n" "$(date +%H:%M:%S)" "$line"
    if [[ "$line" == *ACTIVE* ]]; then
      echo
      echo "  >>> LINK UP. Next:  ./setup_net.sh"
      echo
    fi
    prev="$line"
  fi
  sleep 1
done
