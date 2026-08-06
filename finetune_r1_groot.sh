#!/usr/bin/env bash
# Fine-tune NVIDIA GR00T on the Unitree R1 goal-directed reach dataset.
#
# LOCAL (WSL, single small GPU — will likely OOM on 16 GB, use for a smoke test only):
#   cd ~/Isaac-GR00T && uv run bash /mnt/c/Users/aimat/Desktop/humanoid/finetune_r1_groot.sh
#
# HPC (Vulcan A100 / HammerHAI Blackwell): copy the GR00T repo + dataset + this config to the
# cluster, load the container/modules, then run the same python command below inside a Slurm job.
set -e

DATASET=${1:-/mnt/c/Users/aimat/Desktop/humanoid/lerobot_r1_ikreach}
CONFIG=${2:-/mnt/c/Users/aimat/Desktop/humanoid/r1_groot_config.py}
OUT=${3:-/tmp/r1_groot}

echo "dataset : $DATASET"
echo "config  : $CONFIG"
echo "output  : $OUT"

uv run python gr00t/experiment/launch_finetune.py \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$CONFIG" \
    --output-dir "$OUT" \
    --max-steps 5000 \
    --save-steps 500

# ---- after training, open-loop sanity eval (predicted vs ground-truth on a training traj) ----
# uv run python gr00t/eval/open_loop_eval.py \
#     --dataset-path "$DATASET" \
#     --embodiment-tag NEW_EMBODIMENT \
#     --model-path "$OUT/checkpoint-2000" \
#     --traj-ids 0 --action-horizon 16 --steps 400 --modality-keys joints \
#     --save-plot-path "$OUT/eval_plots"
