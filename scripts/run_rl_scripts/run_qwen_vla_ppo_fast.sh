#!/bin/bash
# Vanilla VLA + PPO (full finetune) — STEP-LOCK rollout, single GPU. [Phase 1]
#
# RLinf-style rollout kernel: a persistent env pool of G envs advances in
# lockstep — ONE batched VLA forward per chunk over all active envs, all envs
# step concurrently in a thread pool. Replaces the legacy vla_ppo_collect
# (serial per-env stepping + per-iter subprocess respawn).
#
# Goal of this phase: confirm the batched rollout actually feeds the GPU and
# measure the rollout speedup. Multi-GPU (torchrun + FSDP) is Phase 2.
#
# Usage:
#   bash scripts/run_rl_scripts/run_qwen_vla_ppo_fast.sh [GPU_ID]
#
# Env:
#   GPU_ID          physical GPU for training + EGL render (default 0; also $1)
#   TASK_ID         libero_goal task index (default 0)
#   CKPT_PATH       Qwen VLA ckpt (default 1traj if exists, else 5traj)
#   G               episodes/iter = persistent env-pool size (default 32)
#   PPO_EPOCHS      PPO epochs per iter (default 2)
#   MICRO_BATCH     VLA re-forward batch size in PPO update (default 2)
#   LR_VLA          VLA full-FT LR (default 1e-5)
#   MAX_ITER        total iterations (default 4 — short verification run)
#   EVAL_INTERVAL   eval cadence (default 4)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Reduce allocator fragmentation during the micro-batched PPO re-forward.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID=${1:-${GPU_ID:-0}}
TASK_ID=${TASK_ID:-0}
G=${G:-32}
PPO_EPOCHS=${PPO_EPOCHS:-2}
MICRO_BATCH=${MICRO_BATCH:-2}
LR_VLA=${LR_VLA:-1e-5}
MAX_ITER=${MAX_ITER:-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-4}

# Do NOT remap CUDA_VISIBLE_DEVICES — keep cuda:${GPU_ID} == physical GPU so
# MuJoCo EGL rendering (egl_gpu_id) lands on the same card as the VLA.

# Prefer 1traj ckpt (faster experiments); fall back to 5traj.
if [ -d "results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model" ]; then
    DEFAULT_CKPT="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
else
    DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
fi
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"

[ -d "${CKPT_PATH}" ] || { echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1; }

TIMESTAMP=$(date +%m%d_%H%M)
RUN_TAG="vla_ppo_fast_qwen_t${TASK_ID}"
OUTPUT_DIR="results/rlt_training/${RUN_TAG}_${TIMESTAMP}/vla_ppo"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " Vanilla VLA + PPO (FULL FT) — STEP-LOCK rollout [Phase 1]"
echo "   GPU:           ${GPU_ID}   task: ${TASK_ID}"
echo "   ckpt:          ${CKPT_PATH}"
echo "   G (envs/iter): ${G}        (step-lock batch = up to ${G})"
echo "   PPO epochs:    ${PPO_EPOCHS}    micro_batch: ${MICRO_BATCH}"
echo "   lr_vla:        ${LR_VLA}"
echo "   max_iter:      ${MAX_ITER}    eval_interval: ${EVAL_INTERVAL}"
echo "   output:        ${OUTPUT_DIR}"
echo "============================================================"

python -u AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase vla_ppo \
    --ckpt_path ${CKPT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal --task_id ${TASK_ID} \
    --G ${G} --group_size 1 \
    --reward_coef 5.0 \
    --lr_vla ${LR_VLA} --lr_critic 3e-4 \
    --critic_hidden_dim 256 \
    --fixed_std 0.1 \
    --ppo_epochs ${PPO_EPOCHS} --micro_batch ${MICRO_BATCH} \
    --clip_eps 0.2 --vf_coef 0.5 \
    --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 1.0 \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 20 \
    --save_interval 50 --num_steps_wait 10 \
    --train_gpu ${GPU_ID} --seed 42 \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_TAG}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
