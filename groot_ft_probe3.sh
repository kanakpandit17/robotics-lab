#!/usr/bin/env bash
cd ~/Isaac-GR00T || exit 1
echo "===== the concrete finetune command in the doc ====="
grep -nE "launch_finetune.py|--dataset-path|--embodiment-tag|--modality-config-path|--output-dir|--max-steps|--batch-size|--data-config" getting_started/finetune_new_embodiment.md | head -20
echo
echo "===== launch_finetune.py argparse (arg names) ====="
grep -nE "add_argument|dataset_path|embodiment|modality_config|output_dir|max_steps|batch_size|data_config|lora|dataset-path" gr00t/experiment/launch_finetune.py 2>/dev/null | head -35
echo
echo "===== SO100 README finetune command ====="
grep -nE "launch_finetune|--dataset|--embodiment|--modality|--output|--max|uv run" examples/SO100/README.md 2>/dev/null | head -15
