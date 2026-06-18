#!/bin/bash
# Helper to launch one RLT_a Phase-2 RL run for the report table 1 backfill.
# All 8 remaining cells (GRPO/TD3/PPO × task 0/1/3, minus GRPO task0 already
# started on GPU 6) share the same Phase-1 encoder + 5traj basebone + 300 iter
# + in-train 20-ep eval terminal protocol, matching the existing RLT rows.
#
# Usage (inside tmux on any server):
#   bash scripts/run_rl_scripts/run_rlt_a_table1.sh <algo> <task_id> [gpu_id]
#
# Args:
#   algo     one of: grpo | td3 | ppo
#   task_id  one of: 0 | 1 | 3
#   gpu_id   GPU index on this machine (default 0)
#
# Examples:
#   bash scripts/run_rl_scripts/run_rlt_a_table1.sh grpo 1       # GPU 0
#   bash scripts/run_rl_scripts/run_rlt_a_table1.sh td3 3 5      # GPU 5
#   bash scripts/run_rl_scripts/run_rlt_a_table1.sh ppo 0 7      # GPU 7
#
# Output:
#   logs/rlt_a_<algo>_t<task>_remote_<MMDD_HHMM>.log
#   results/rlt_training/rlt_a_<algo>_qwen_t<task>_<TS>/...
set -euo pipefail

ALGO="${1:?usage: $0 <algo: grpo|td3|ppo> <task_id: 0|1|3> [gpu_id]}"
TASK_ID="${2:?usage: $0 <algo> <task_id> [gpu_id]}"
GPU_ID="${3:-0}"

case "${ALGO}" in
    grpo|td3|ppo) ;;
    *) echo "ERROR: algo must be 'grpo', 'td3', or 'ppo' (got '${ALGO}')" >&2; exit 1 ;;
esac
case "${TASK_ID}" in
    0|1|3) ;;
    *) echo "ERROR: task_id must be 0, 1, or 3 (got '${TASK_ID}')" >&2; exit 1 ;;
esac

# ── env (same as Baige / GPU 6 successful runs) ──────────────────────────────
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

# ── Phase-1 encoder (multi-task 5traj, pretrained 5/26) ──────────────────────
export ENCODER_PATH="results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt"
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: encoder.pt not found at ${ENCODER_PATH}" >&2; exit 1
fi

LOG="logs/rlt_a_${ALGO}_t${TASK_ID}_remote_$(date +%m%d_%H%M).log"

echo "============================================================"
echo "  RLT_a + ${ALGO^^}  task ${TASK_ID}  GPU ${GPU_ID}"
echo "  encoder: ${ENCODER_PATH}"
echo "  log:     ${LOG}"
echo "============================================================"

# ── dispatch to per-algo launcher ────────────────────────────────────────────
case "${ALGO}" in
    grpo)
        TASK_ID="${TASK_ID}" bash scripts/run_rl_scripts/run_rlt_a_grpo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    td3)
        TRACK=rlt_a MULTI_TASK=0 MAX_ITER=300 TASK_ID="${TASK_ID}" \
            bash scripts/run_rl_scripts/run_rlt_rl.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    ppo)
        TASK_ID="${TASK_ID}" bash scripts/run_rl_scripts/run_rlt_a_ppo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
esac
