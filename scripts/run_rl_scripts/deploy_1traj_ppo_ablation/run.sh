#!/bin/bash
# Dispatcher to launch one RL run for the 1traj × PPO ablation.
#
# 6 cells to backfill in table 1c (1traj basebone):
#   RLT   + PPO 1traj × {t0, t1, t3}
#   RLT_a + PPO 1traj × {t0, t1, t3}
#
# Usage:
#   bash run.sh <track> <task_id> [gpu_id]
#
# Args:
#   track    one of: rlt | rlt_a
#   task_id  one of: 0 | 1 | 3
#   gpu_id   GPU index on this machine (default 0)
#
# Examples:
#   bash run.sh rlt 0
#   bash run.sh rlt_a 3 5
#
# Output:
#   logs/<track>_ppo_1traj_t<task>_<MMDD_HHMM>.log
#   results/rlt_training/rlt_<...>ppo_qwen_t<task>_<TS>/...
#
# Notes:
#   - For RLT track: uses existing 1traj encoder at
#     results/rlt_training/1traj_libero_goal_step30k_0423_0545/pretrain/checkpoints/pretrain_best/encoder.pt
#   - For RLT_a track: auto-discovers the latest pretrain_rlt_a_1traj output.
#     Run pretrain_rlt_a_1traj.sh first if it hasn't been done yet.
#   - 5traj basebone -> 1traj basebone: only ckpt + encoder differ, everything
#     else (G=16, 300 iter, in-train 20-ep eval) matches table 1 protocol.
set -euo pipefail

TRACK="${1:?usage: $0 <track: rlt|rlt_a> <task_id: 0|1|3> [gpu_id]}"
TASK_ID="${2:?usage: $0 <track> <task_id> [gpu_id]}"
GPU_ID="${3:-0}"

case "${TRACK}" in
    rlt|rlt_a) ;;
    *) echo "ERROR: track must be 'rlt' or 'rlt_a' (got '${TRACK}')" >&2; exit 1 ;;
esac
case "${TASK_ID}" in
    0|1|3) ;;
    *) echo "ERROR: task_id must be 0, 1, or 3 (got '${TASK_ID}')" >&2; exit 1 ;;
esac

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

# 1traj VLA ckpt
export CKPT_PATH="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: 1traj VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1
fi

# Encoder per track
case "${TRACK}" in
    rlt)
        # Existing 1traj RLT (full-token) pretrain from 4/23.
        export ENCODER_PATH="results/rlt_training/1traj_libero_goal_step30k_0423_0545/pretrain/checkpoints/pretrain_best/encoder.pt"
        ;;
    rlt_a)
        # Auto-discover latest rlt_a 1traj pretrain dir (from pretrain_rlt_a_1traj.sh).
        # The pretrain dir tag is "rlt_a_0324-zh-QwenOFT-1traj-libero_goal_<TS>".
        _PRETRAIN_DIR=$(ls -td results/rlt_training/rlt_a_0324-zh-QwenOFT-1traj-libero_goal_*/pretrain 2>/dev/null | head -1 || true)
        if [ -z "${_PRETRAIN_DIR}" ]; then
            echo "ERROR: no rlt_a 1traj pretrain dir found." >&2
            echo "       Run first:  bash pretrain_rlt_a_1traj.sh [gpu_id]" >&2
            exit 1
        fi
        export ENCODER_PATH="${_PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt"
        ;;
esac
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: encoder.pt not found: ${ENCODER_PATH}" >&2; exit 1
fi

LOG="logs/${TRACK}_ppo_1traj_t${TASK_ID}_$(date +%m%d_%H%M).log"

echo "============================================================"
echo "  ${TRACK^^} + PPO  1traj  task ${TASK_ID}  GPU ${GPU_ID}"
echo "  ckpt:    ${CKPT_PATH}"
echo "  encoder: ${ENCODER_PATH}"
echo "  log:     ${LOG}"
echo "============================================================"

# Single-task, 300 iter, in-train 20-ep eval (matches table 1 protocol).
case "${TRACK}" in
    rlt)
        TASK_ID="${TASK_ID}" \
            bash scripts/run_rl_scripts/run_rlt_ppo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    rlt_a)
        TASK_ID="${TASK_ID}" \
            bash scripts/run_rl_scripts/run_rlt_a_ppo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
esac
