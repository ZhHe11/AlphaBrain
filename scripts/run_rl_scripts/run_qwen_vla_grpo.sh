#!/bin/bash
# Vanilla VLA + GRPO (full finetune) — Qwen-OFT only.
#
# Trains the ENTIRE VLA via group-relative PG with KL-to-reference penalty.
# Memory: ~58 GB on 80 GB GPU (current VLA + Adam + ref VLA bf16).
#
# IMPORTANT: GRPO needs G >= group_size and group_size >= 2 for a usable
# relative signal. Default G=8, group_size=2 → 4 distinct init states,
# 2 rollouts each.
#
# Usage:
#   bash scripts/run_rl_scripts/run_qwen_vla_grpo.sh [GPU_ID]
#   TASK_ID=1 bash scripts/run_rl_scripts/run_qwen_vla_grpo.sh 0
#   MULTI_TASK=1 bash scripts/run_rl_scripts/run_qwen_vla_grpo.sh 0       # all 10 libero_goal tasks
#
# Env:
#   TASK_ID         libero_goal task index (default 0; ignored if MULTI_TASK=1)
#   MULTI_TASK      1 = --all_tasks (default 0 = single task). NOTE: with
#                   --all_tasks, --G / --num_envs are PER-TASK (trainer dest is
#                   G_per_task / num_envs_per_task), so keep them small.
#   CKPT_PATH       Qwen VLA ckpt (default 1traj if exists, else 5traj)
#   PPO_EPOCHS      GRPO epochs per iter (default 2)
#   G               episodes per iter (default 8)
#   GROUP_SIZE      rollouts per initial state (default 2; min 2 for GRPO)
#   NUM_ENVS        parallel envs per rollout wave (default 4)
#   MICRO_BATCH     VLA re-forward batch size (default 2; OOM-bound)
#   LR_VLA          VLA full-FT LR (default 1e-5)
#   KL_COEF         KL-to-ref penalty (default 0.04, DeepSeek default)
#   REF_UPD_INT     refresh ref VLA every N iters (default 0 = never)
#   MAX_ITER        total iterations (default 30)
#   EVAL_INTERVAL   eval cadence (default 5)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_ID=${1:-0}
TASK_ID=${TASK_ID:-0}
MULTI_TASK=${MULTI_TASK:-0}
TASKS_PER_ITER=${TASKS_PER_ITER:-0}
PPO_EPOCHS=${PPO_EPOCHS:-2}
G=${G:-8}
GROUP_SIZE=${GROUP_SIZE:-2}
NUM_ENVS=${NUM_ENVS:-4}
MICRO_BATCH=${MICRO_BATCH:-2}
LR_VLA=${LR_VLA:-1e-5}
KL_COEF=${KL_COEF:-0.04}
REF_UPD_INT=${REF_UPD_INT:-0}
MAX_ITER=${MAX_ITER:-30}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}

if [ -d "results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model" ]; then
    DEFAULT_CKPT="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
else
    DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
fi
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"

[ -d "${CKPT_PATH}" ] || { echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1; }

if [ "${MULTI_TASK}" = "1" ]; then
    TASK_FLAG="--all_tasks"; RUN_TAG="vla_grpo_qwen_alltasks"
else
    TASK_FLAG="--task_id ${TASK_ID}"; RUN_TAG="vla_grpo_qwen_t${TASK_ID}"
fi
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_training/${RUN_TAG}_${TIMESTAMP}/vla_grpo"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " Vanilla VLA + GRPO (FULL FT)  — Qwen, ${TASK_FLAG}"
echo "   GPU:           ${GPU_ID}"
echo "   ckpt:          ${CKPT_PATH}"
echo "   GRPO epochs:   ${PPO_EPOCHS}     micro_batch: ${MICRO_BATCH}"
echo "   G/iter:        ${G}              group_size: ${GROUP_SIZE}"
echo "   envs:          ${NUM_ENVS}"
echo "   lr_vla:        ${LR_VLA}         kl_coef: ${KL_COEF}"
echo "   ref_upd_int:   ${REF_UPD_INT}    max_iter: ${MAX_ITER}"
echo "   output:        ${OUTPUT_DIR}"
echo "============================================================"
echo "WARN: full-VLA GRPO is mem-heavy (~58 GB GPU; trainable + ref VLA)."
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -u AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase vla_grpo \
    --ckpt_path ${CKPT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal ${TASK_FLAG} --tasks_per_iter ${TASKS_PER_ITER} \
    --G ${G} --num_envs ${NUM_ENVS} --group_size ${GROUP_SIZE} \
    --reward_coef 5.0 \
    --lr_vla ${LR_VLA} \
    --fixed_std 0.1 \
    --ppo_epochs ${PPO_EPOCHS} --micro_batch ${MICRO_BATCH} \
    --clip_eps 0.2 --grpo_kl_coef ${KL_COEF} \
    --ref_update_interval ${REF_UPD_INT} \
    --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 1.0 \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 20 \
    --save_interval 50 --num_steps_wait 10 \
    --train_gpu 0 --seed ${SEED:-42} \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME:-${RUN_TAG}}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
