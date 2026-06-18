#!/bin/bash
# Phase-1 pretrain RLT_a encoder on 1traj basebone.
#
# Required first time only — output is shared across all 3 RLT_a + PPO 1traj
# task runs (t0/t1/t3). After this is done, run.sh rlt_a <task_id> can find
# the encoder automatically (because run_rlt_pretrain.sh writes to
# results/rlt_training/rlt_a_<...>/pretrain/checkpoints/pretrain_best/).
#
# Usage:
#   bash pretrain_rlt_a_1traj.sh [gpu_id]  (default GPU 0)
#
# Time: ~30 minutes on one A100.
set -euo pipefail

GPU_ID="${1:-0}"

source /share/zhanghe/miniconda3/etc/profile.d/conda.sh
conda activate vla
cd /share/zhanghe/AlphaBrain-zh

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma
unset DISPLAY
mkdir -p logs

LOG="logs/rlt_a_pretrain_1traj_$(date +%m%d_%H%M).log"

echo "============================================================"
echo "  Phase-1 RLT_a pretrain  (1traj basebone, multi-task)"
echo "  ckpt:  results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
echo "  GPU:   ${GPU_ID}"
echo "  log:   ${LOG}"
echo "============================================================"

CKPT_PATH=results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model \
TRACK=rlt_a \
    bash scripts/run_rl_scripts/run_rlt_pretrain.sh "${GPU_ID}" 2>&1 | tee "${LOG}"

# After completion, the encoder will be at:
#   results/rlt_training/rlt_a_<TS>/pretrain/checkpoints/pretrain_best/encoder.pt
# run.sh will look for the most recent rlt_a_*1traj* pretrain dir automatically.
