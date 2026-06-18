#!/bin/bash
# Vanilla VLA + PPO (full finetune) — DATA-PARALLEL via torchrun. [Phase 2]
#
# Each rank = 1 GPU: loads an identical VLA, owns a persistent env pool,
# collects its shard of G episodes (step-lock rollout), runs the PPO update
# on that shard; gradients are all-reduced (AVG) across ranks before the
# optimizer step. torchrun is the launcher — no Ray.
#
# Usage:
#   bash scripts/run_rl_scripts/run_qwen_vla_ppo_ddp.sh
#   GPUS=0,1,2,3 G=32 bash scripts/run_rl_scripts/run_qwen_vla_ppo_ddp.sh
#
# Env:
#   GPUS            comma-separated GPU ids (default 0,1,2,3,4,5,6,7)
#   G               TOTAL episodes/iter across all ranks (default 64; must be >= #GPUs)
#   TASK_ID         libero_goal task index (default 0)
#   CKPT_PATH       Qwen VLA ckpt (default 1traj if exists, else 5traj)
#   PPO_EPOCHS      PPO epochs per iter (default 2)
#   MICRO_BATCH     VLA re-forward batch in PPO update (default 2)
#   LR_VLA          VLA full-FT LR (default 1e-5)
#   MAX_ITER        total iterations (default 4 — short verification run)
#   EVAL_INTERVAL   eval cadence (default 4)
#   MASTER_PORT     torchrun rendezvous port (default 29501)
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
MASTER_PORT=${MASTER_PORT:-29501}

TASK_ID=${TASK_ID:-0}
G=${G:-64}
PPO_EPOCHS=${PPO_EPOCHS:-2}
MICRO_BATCH=${MICRO_BATCH:-2}
LR_VLA=${LR_VLA:-1e-5}
LR_CRITIC=${LR_CRITIC:-3e-4}
MAX_ITER=${MAX_ITER:-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-4}

if [ "${G}" -lt "${NGPU}" ]; then
    echo "ERROR: G (${G}) must be >= number of GPUs (${NGPU})" >&2
    exit 1
fi

# Prefer 1traj ckpt (faster experiments); fall back to 5traj.
if [ -d "results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model" ]; then
    DEFAULT_CKPT="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
else
    DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
fi
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
[ -d "${CKPT_PATH}" ] || { echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1; }

TIMESTAMP=$(date +%m%d_%H%M)
RUN_TAG="vla_ppo_ddp_qwen_t${TASK_ID}"
OUTPUT_DIR="results/rlt_training/${RUN_TAG}_${TIMESTAMP}/vla_ppo"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " Vanilla VLA + PPO (FULL FT) — DATA-PARALLEL torchrun [Phase 2]"
echo "   GPUS:          ${GPUS}  (${NGPU} ranks)"
echo "   ckpt:          ${CKPT_PATH}"
echo "   G total:       ${G}   (= ${NGPU} ranks × $((G / NGPU))/rank)"
echo "   PPO epochs:    ${PPO_EPOCHS}    micro_batch: ${MICRO_BATCH}"
echo "   lr_vla:        ${LR_VLA}"
echo "   max_iter:      ${MAX_ITER}    eval_interval: ${EVAL_INTERVAL}"
echo "   output:        ${OUTPUT_DIR}"
echo "============================================================"

# Use `python -m torch.distributed.run` (not the system `torchrun` binary) so
# worker processes inherit this env's Python — the project deps live here.
python -m torch.distributed.run --nproc_per_node=${NGPU} --master_port=${MASTER_PORT} \
    AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase vla_ppo \
    --ckpt_path ${CKPT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal --task_id ${TASK_ID} \
    --G ${G} --group_size 1 \
    --reward_coef 5.0 \
    --lr_vla ${LR_VLA} --lr_critic ${LR_CRITIC} \
    --critic_hidden_dim 256 --fixed_std 0.1 \
    --ppo_epochs ${PPO_EPOCHS} --micro_batch ${MICRO_BATCH} \
    --clip_eps 0.2 --vf_coef 0.5 \
    --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 1.0 \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 32 \
    --save_interval 50 --num_steps_wait 10 \
    --seed 42 \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_TAG}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
