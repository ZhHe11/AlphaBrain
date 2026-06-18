#!/bin/bash
# Vanilla VLA + PPO (full finetune) — MULTI-GPU rollout, Qwen-OFT.
#
# Same trainer as run_qwen_vla_ppo.sh, but rollout collection is sharded
# across all GPUs in GPU_LIST: the train GPU collects with the live policy,
# every other GPU runs a frozen VLA copy resynced after each PPO update.
# The PPO gradient update stays single-GPU on the train GPU.
#
# Usage:
#   bash scripts/run_rl_scripts/run_qwen_vla_ppo_mgpu.sh                 # all 8 GPUs
#   bash scripts/run_rl_scripts/run_qwen_vla_ppo_mgpu.sh 0,1,2,3         # 4 GPUs
#
# Env:
#   GPU_LIST        comma-separated physical GPU ids (default 0,1,2,3,4,5,6,7;
#                   also accepted as $1)
#   TASK_ID         libero_goal task index (default 0)
#   CKPT_PATH       Qwen VLA ckpt (default 1traj if exists, else 5traj)
#   PPO_EPOCHS      PPO epochs per iter (default 2)
#   G               episodes per iter (default 8 — kept = single-GPU baseline)
#   NUM_ENVS        parallel envs per rollout wave PER GPU (default 4)
#   MICRO_BATCH     VLA re-forward batch size in PPO update (default 2)
#   LR_VLA          VLA full-FT LR (default 1e-5)
#   MAX_ITER        total iterations (default 8 — short timing experiment)
#   EVAL_INTERVAL   eval cadence (default 5)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_LIST=${1:-${GPU_LIST:-0,1,2,3,4,5,6,7}}
TASK_ID=${TASK_ID:-0}
PPO_EPOCHS=${PPO_EPOCHS:-2}
G=${G:-8}
NUM_ENVS=${NUM_ENVS:-4}
MICRO_BATCH=${MICRO_BATCH:-2}
LR_VLA=${LR_VLA:-1e-5}
MAX_ITER=${MAX_ITER:-8}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}

# Make exactly GPU_LIST visible; the trainer then addresses them as the
# remapped indices 0..N-1, so --rollout_gpus / --train_gpu use that space.
export CUDA_VISIBLE_DEVICES=${GPU_LIST}
N_GPUS=$(echo "${GPU_LIST}" | tr ',' '\n' | grep -c .)
ROLLOUT_GPUS=$(seq -s, 0 $((N_GPUS - 1)))
TRAIN_GPU=0

# Prefer 1traj ckpt (faster experiments); fall back to 5traj.
if [ -d "results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model" ]; then
    DEFAULT_CKPT="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
else
    DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
fi
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"

[ -d "${CKPT_PATH}" ] || { echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1; }

TIMESTAMP=$(date +%m%d_%H%M)
RUN_TAG="vla_ppo_mgpu_qwen_t${TASK_ID}"
OUTPUT_DIR="results/rlt_training/${RUN_TAG}_${TIMESTAMP}/vla_ppo"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " Vanilla VLA + PPO (FULL FT) — MULTI-GPU rollout, Qwen task ${TASK_ID}"
echo "   GPU_LIST:      ${GPU_LIST}  (${N_GPUS} GPUs)"
echo "   rollout GPUs:  ${ROLLOUT_GPUS}    train GPU: ${TRAIN_GPU}"
echo "   ckpt:          ${CKPT_PATH}"
echo "   PPO epochs:    ${PPO_EPOCHS}     micro_batch: ${MICRO_BATCH}"
echo "   G/iter:        ${G}              envs/GPU: ${NUM_ENVS}"
echo "   lr_vla:        ${LR_VLA}"
echo "   max_iter:      ${MAX_ITER}       eval_interval: ${EVAL_INTERVAL}"
echo "   output:        ${OUTPUT_DIR}"
echo "============================================================"
echo "NOTE: ${G} episodes are sharded across ${N_GPUS} GPUs; the train GPU"
echo "      also collects. Each iter logs rollout/update/sync seconds."
echo "============================================================"

python -u AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase vla_ppo \
    --ckpt_path ${CKPT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal --task_id ${TASK_ID} \
    --G ${G} --num_envs ${NUM_ENVS} --group_size 1 \
    --reward_coef 5.0 \
    --lr_vla ${LR_VLA} --lr_critic 3e-4 \
    --critic_hidden_dim 256 \
    --fixed_std 0.1 \
    --ppo_epochs ${PPO_EPOCHS} --micro_batch ${MICRO_BATCH} \
    --clip_eps 0.2 --vf_coef 0.5 \
    --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 1.0 \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 20 \
    --save_interval 50 --num_steps_wait 10 \
    --rollout_gpus ${ROLLOUT_GPUS} --train_gpu ${TRAIN_GPU} --seed 42 \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_TAG}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
