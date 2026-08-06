#!/usr/bin/env bash
cd ~/Isaac-GR00T || exit 1
echo "=== gr00t/data/dataset/__init__.py ==="
sed -n '1,40p' gr00t/data/dataset/__init__.py
echo
echo "=== classes across gr00t/data/ ==="
grep -rnE "^class [A-Za-z0-9_]+" gr00t/data/ | grep -iE "dataset|loader" | head -30
echo
echo "=== ModalityConfig / DataConfig locations ==="
grep -rnE "class ModalityConfig|class .*DataConfig|EmbodimentTag" gr00t/ | head -20
echo
echo "=== how the finetune script builds the dataset ==="
f=$(ls scripts/gr00t_finetune.py 2>/dev/null || find . -name '*finetune*.py' | head -1)
echo "finetune file: $f"
grep -nE "Dataset\(|modality_config|ModalityConfig|EmbodimentTag|video_backend|data_config|get_data_config" "$f" 2>/dev/null | head -30
echo
echo "=== available data configs ==="
grep -rnE "= .*DataConfig\(|DATA_CONFIG_MAP|register|\"single_panda|new_embodiment" gr00t/experiment/data_config.py 2>/dev/null | head -20
