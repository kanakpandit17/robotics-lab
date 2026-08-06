#!/usr/bin/env bash
cd ~/Isaac-GR00T || exit 1
echo "===== finetune entry points ====="
ls scripts/*finetune* 2>/dev/null; find . -name "*finetune*.py" -not -path "*/tests/*" 2>/dev/null | head
echo
echo "===== finetune_new_embodiment.md (key steps) ====="
sed -n '1,120p' getting_started/finetune_new_embodiment.md
echo
echo "===== EmbodimentTag NEW / custom options ====="
grep -nE "NEW_EMBODIMENT|new_embodiment|CUSTOM|register" gr00t/data/embodiment_tags.py | head
