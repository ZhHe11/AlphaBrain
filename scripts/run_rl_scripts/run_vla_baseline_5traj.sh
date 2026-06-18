#!/bin/bash
# Helper to launch one vanilla VLA-baseline RL run (VLA+PPO or VLA+GRPO) on
# the 5traj backbone for table 1 backfill. Sibling of run_rlt_a_table1.sh,
# same protocol: 5traj basebone + single-task + 300 iter + in-train 20-ep
# eval terminal — so VLA+{PPO,GRPO} cells compare apples-to-apples with
# RLT+{PPO,GRPO} cells (same basebone, same iter count, same eval).
#
# Existing VLA+PPO/GRPO data in the report was on 1traj (see table 1c).
# This helper produces the 5traj numbers that table 1 actually needs.
#
# Usage:
#   bash scripts/run_rl_scripts/run_vla_baseline_5traj.sh <algo> <task_id> [gpu_id]
#
# Args:
#   algo     one of: ppo | grpo
#   task_id  one of: 0 | 1 | 3
#   gpu_id   GPU index on this machine (default 0)
#
# Examples:
#   bash scripts/run_rl_scripts/run_vla_baseline_5traj.sh ppo 0
#   bash scripts/run_rl_scripts/run_vla_baseline_5traj.sh grpo 3 7
#
# Output:
#   logs/vla_<algo>_5traj_t<task>_<MMDD_HHMM>.log
#   results/rlt_training/vla_<algo>_qwen_t<task>_<TS>/...
set -euo pipefail

ALGO="${1:?usage: $0 <algo: ppo|grpo> <task_id: 0|1|3> [gpu_id]}"
TASK_ID="${2:?usage: $0 <algo> <task_id> [gpu_id]}"
GPU_ID="${3:-0}"

case "${ALGO}" in
    ppo|grpo) ;;
    *) echo "ERROR: algo must be 'ppo' or 'grpo' (got '${ALGO}')" >&2; exit 1 ;;
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

# Force 5traj basebone (launcher would default to 1traj otherwise)
export CKPT_PATH="results/training/QwenOFT-5traj-libero_goal/final_model"
if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: 5traj VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1
fi

LOG="logs/vla_${ALGO}_5traj_t${TASK_ID}_$(date +%m%d_%H%M).log"

echo "============================================================"
echo "  VLA + ${ALGO^^}  (vanilla full-finetune)  5traj  task ${TASK_ID}  GPU ${GPU_ID}"
echo "  ckpt:    ${CKPT_PATH}"
echo "  log:     ${LOG}"
echo "============================================================"

# 300 iter to match table 1 RLT+* protocol; eval every 20.
case "${ALGO}" in
    ppo)
        MAX_ITER=300 EVAL_INTERVAL=20 TASK_ID="${TASK_ID}" \
            bash scripts/run_rl_scripts/run_qwen_vla_ppo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    grpo)
        MAX_ITER=300 EVAL_INTERVAL=20 TASK_ID="${TASK_ID}" \
            bash scripts/run_rl_scripts/run_qwen_vla_grpo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
esac
