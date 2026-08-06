#!/usr/bin/env bash
cd ~/Isaac-GR00T || exit 1
echo "===== finetune_new_embodiment.md: Step 3 (launch command) ====="
sed -n '120,210p' getting_started/finetune_new_embodiment.md
echo
echo "===== examples/SO100 contents ====="
ls -R examples/SO100 2>/dev/null | head -30
echo
echo "===== SO100 registration / finetune script (grep) ====="
grep -rnE "register_modality_config|EmbodimentTag|dataset_path|launch_finetune|--|python" examples/SO100/*.sh examples/SO100/*.py 2>/dev/null | head -30
