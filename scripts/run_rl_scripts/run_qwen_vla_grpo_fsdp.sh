#!/bin/bash
# Vanilla VLA + GRPO (full finetune) — FSDP via torchrun, RLinf-aligned. [scale]
#
# Each rank = 1 GPU: trainable VLM FSDP-sharded (FULL_SHARD across all ranks);
# action_model replicated per rank with manual NCCL all-reduce; the frozen
# reference VLA is replicated (bf16) per rank. Rollout: step-lock batched
# collector over a per-rank persistent env pool. Launcher: torchrun (no Ray).
#
# GRPO groups (same init state × group_size rollouts) form WITHIN each rank's
# shard → keep G_local = G/NGPU a multiple of GROUP_SIZE.
#
# Defaults are RLinf-GRPO numbers (see RLINF_VLA_PPO_GRPO.md):
#   group_size=8, micro_batch=40, lr_vla=1e-5, kl_beta=0.0
#
# Usage:
#   GPUS=1,2,3,4,5,7 MULTI_TASK=1 MAX_ITER=400 bash scripts/run_rl_scripts/run_qwen_vla_grpo_fsdp.sh
#
# Env:
#   GPUS            comma-separated GPU ids (default 0,1,2,3,4,5,6,7)
#   G               TOTAL episodes/iter/task (default 144; G/NGPU must be ÷ GROUP_SIZE)
#   GROUP_SIZE      rollouts per initial state (default 8, RLinf-GRPO)
#   MULTI_TASK      1 = --all_tasks (default 0 = single task)
#   TASKS_PER_ITER  tasks cycled per iter under --all_tasks (default 0 = all)
#   TASK_ID         libero_goal task index (default 0; ignored if MULTI_TASK=1)
#   CKPT_PATH       Qwen VLA ckpt (default 5traj)
#   PPO_EPOCHS      GRPO epochs per iter (default 1)
#   MICRO_BATCH     VLA re-forward batch in update (default 40, RLinf-GRPO)
#   LR_VLA          VLA full-FT LR (default 1e-5, RLinf-GRPO)
#   KL_COEF         KL-to-ref penalty (default 0.0, RLinf-GRPO)
#   MAX_ITER        total iterations (default 4 — verification; pass 400 for real)
#   EVAL_INTERVAL   eval cadence (default 4)
#   MASTER_PORT     torchrun rendezvous port (default 29521)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=${GPUS}
NGPU=$(echo "${GPUS}" | tr ',' '\n' | grep -c .)
MASTER_PORT=${MASTER_PORT:-29521}

TASK_ID=${TASK_ID:-0}
MULTI_TASK=${MULTI_TASK:-0}
TASKS_PER_ITER=${TASKS_PER_ITER:-0}
G=${G:-144}
GROUP_SIZE=${GROUP_SIZE:-8}
PPO_EPOCHS=${PPO_EPOCHS:-1}
MICRO_BATCH=${MICRO_BATCH:-40}
LR_VLA=${LR_VLA:-1e-5}
KL_COEF=${KL_COEF:-0.0}
REF_UPD_INT=${REF_UPD_INT:-0}
MAX_ITER=${MAX_ITER:-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-4}
SEED=${SEED:-42}

if [ "${G}" -lt "${NGPU}" ]; then
    echo "ERROR: G (${G}) must be >= number of GPUs (${NGPU})" >&2
    exit 1
fi
if [ "${MULTI_TASK}" = "1" ]; then
    TASK_FLAG="--all_tasks"; RUN_SUFFIX="alltasks"
else
    TASK_FLAG="--task_id ${TASK_ID}"; RUN_SUFFIX="t${TASK_ID}"
fi

if [ -d "results/training/QwenOFT-5traj-libero_goal/final_model" ]; then
    DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
else
    DEFAULT_CKPT="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
fi
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
[ -d "${CKPT_PATH}" ] || { echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1; }

TIMESTAMP=$(date +%m%d_%H%M)
RUN_TAG="vla_grpo_fsdp_qwen_${RUN_SUFFIX}"
OUTPUT_DIR="results/rlt_training/${RUN_TAG}_${TIMESTAMP}/vla_grpo"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " Vanilla VLA + GRPO (FULL FT) — FSDP torchrun [RLinf-aligned]"
echo "   GPUS:          ${GPUS}  (${NGPU} ranks, FULL_SHARD)"
echo "   ckpt:          ${CKPT_PATH}"
echo "   G total:       ${G}   (= ${NGPU} ranks × $((G / NGPU))/rank), group_size ${GROUP_SIZE}"
echo "   GRPO epochs:   ${PPO_EPOCHS}    micro_batch: ${MICRO_BATCH}"
echo "   lr_vla:        ${LR_VLA}   kl_coef: ${KL_COEF}"
echo "   tasks:         ${TASK_FLAG} (tasks_per_iter ${TASKS_PER_ITER})"
echo "   max_iter:      ${MAX_ITER}    eval_interval: ${EVAL_INTERVAL}"
echo "   output:        ${OUTPUT_DIR}"
echo "============================================================"

python -m torch.distributed.run --nproc_per_node=${NGPU} --master_port=${MASTER_PORT} \
    AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase vla_grpo \
    --ckpt_path ${CKPT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal ${TASK_FLAG} --tasks_per_iter ${TASKS_PER_ITER} \
    --G ${G} --group_size ${GROUP_SIZE} \
    --reward_coef 5.0 \
    --lr_vla ${LR_VLA} \
    --fixed_std 0.1 \
    --ppo_epochs ${PPO_EPOCHS} --micro_batch ${MICRO_BATCH} \
    --clip_eps 0.2 --grpo_kl_coef ${KL_COEF} \
    --ref_update_interval ${REF_UPD_INT} \
    --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 1.0 \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 32 \
    --save_interval 50 --num_steps_wait 10 \
    --train_gpu 0 --seed ${SEED} \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME:-${RUN_TAG}}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
