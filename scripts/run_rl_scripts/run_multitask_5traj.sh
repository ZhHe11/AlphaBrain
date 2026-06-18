#!/bin/bash
# Helper to launch one multi-task (all 10 libero_goal tasks) RL run for the
# table 1 "全 10 任务" column backfill. 5traj basebone, encoder shared across
# tasks (matches the original release multi-task protocol).
#
# Usage:
#   bash scripts/run_rl_scripts/run_multitask_5traj.sh <algo> <track> [gpu_id]
#
# Args:
#   algo     one of: td3 | grpo | ppo
#   track    one of: rlt | rlt_a
#   gpu_id   GPU index on this machine (default 0)
#
# Examples:
#   bash scripts/run_rl_scripts/run_multitask_5traj.sh grpo rlt 0
#   bash scripts/run_rl_scripts/run_multitask_5traj.sh ppo rlt_a 3
#
# Output:
#   logs/<track>_<algo>_alltasks_5traj_<MMDD_HHMM>.log
#   results/rlt_training/{<track>_<algo>_qwen_alltasks,rlt_rl_qwen_alltasks}_<TS>/...
set -euo pipefail

ALGO="${1:?usage: $0 <algo: td3|grpo|ppo> <track: rlt|rlt_a> [gpu_id]}"
TRACK="${2:?usage: $0 <algo> <track> [gpu_id]}"
GPU_ID="${3:-0}"

case "${ALGO}" in
    td3|grpo|ppo) ;;
    *) echo "ERROR: algo must be 'td3', 'grpo', or 'ppo' (got '${ALGO}')" >&2; exit 1 ;;
esac
case "${TRACK}" in
    rlt|rlt_a) ;;
    *) echo "ERROR: track must be 'rlt' or 'rlt_a' (got '${TRACK}')" >&2; exit 1 ;;
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

# Encoder per track (multi-task pretrained, 5traj basebone)
case "${TRACK}" in
    rlt)
        export ENCODER_PATH="results/rlt_training/t1rlt_0521_0921/pretrain/checkpoints/pretrain_best/encoder.pt"
        ;;
    rlt_a)
        export ENCODER_PATH="results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt"
        ;;
esac
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: encoder.pt not found: ${ENCODER_PATH}" >&2; exit 1
fi

LOG="logs/${TRACK}_${ALGO}_alltasks_5traj_$(date +%m%d_%H%M).log"

echo "============================================================"
echo "  ${TRACK^^} + ${ALGO^^}  alltasks (10 tasks multi-task)  5traj  GPU ${GPU_ID}"
echo "  encoder: ${ENCODER_PATH}"
echo "  log:     ${LOG}"
echo "============================================================"

# Dispatch to per-(algo,track) launcher. All set MULTI_TASK=1 for --all_tasks.
# Multi-task hyperparams aligned to the original release (G_per_task=30,
# num_envs_per_task=10) — bigger than single-task defaults (16/8) so the
# rollout signal is strong enough across 10 tasks.
# MAX_ITER=300 to match table 1 protocol (release was 400 iter).
COMMON_BIG="MULTI_TASK=1 MAX_ITER=300 G_PER_TASK=30 NUM_ENVS_PER_TASK=10"

case "${TRACK}-${ALGO}" in
    rlt-td3)
        env TRACK=rlt $COMMON_BIG \
            bash scripts/run_rl_scripts/run_rlt_rl.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    rlt-grpo)
        env $COMMON_BIG GROUP_SIZE=5 \
            bash scripts/run_rl_scripts/run_rlt_grpo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    rlt-ppo)
        env $COMMON_BIG \
            bash scripts/run_rl_scripts/run_rlt_ppo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    rlt_a-td3)
        env TRACK=rlt_a $COMMON_BIG \
            bash scripts/run_rl_scripts/run_rlt_rl.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    rlt_a-grpo)
        env $COMMON_BIG GROUP_SIZE=5 \
            bash scripts/run_rl_scripts/run_rlt_a_grpo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
    rlt_a-ppo)
        env $COMMON_BIG \
            bash scripts/run_rl_scripts/run_rlt_a_ppo.sh "${GPU_ID}" 2>&1 | tee "${LOG}"
        ;;
esac
