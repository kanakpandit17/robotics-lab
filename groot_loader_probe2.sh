#!/usr/bin/env bash
cd ~/Isaac-GR00T || exit 1
echo "=== LeRobotEpisodeLoader.__init__ ==="
awk '/class LeRobotEpisodeLoader/{f=1} f&&/def __init__/{p=1} p{print} p&&/def [a-z]/&&!/__init__/{c++} c>1{exit}' gr00t/data/dataset/lerobot_episode_loader.py | head -45
echo
echo "=== DatasetFactory (head) ==="
sed -n '1,90p' gr00t/data/dataset/factory.py
echo
echo "=== ModalityConfig definition (types.py) ==="
awk '/class ModalityConfig/{f=1} f{print} f&&/^class [A-Z]/&&!/ModalityConfig/{exit}' gr00t/data/types.py | head -30
echo
echo "=== getting_started examples referencing the loader/factory ==="
grep -rlnE "LeRobotEpisodeLoader|DatasetFactory|register_modality_config" getting_started tests 2>/dev/null | head
