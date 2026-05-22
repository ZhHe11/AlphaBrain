#!/bin/bash
# RLT (full-token RL Token track) Phase-2 GRPO launcher.
#
# Sibling of run_rlt_a_grpo.sh — same trainer / loss / rollout, but the
# encoder is the full-VLM-token RLTokenEncoderDecoder (--encoder_mode rlt)
# instead of the action-token ActionTokenEncoderDecoder (action_token).
#
# RLT x GRPO was wired up via the encoder_mode dispatch added to
# train_rl_grpo.py + action_token_collect_group + _eval_distributed.
#
# The RLT encoder must be pretrained on the *rlt* track first:
#   TRACK=rlt bash scripts/run_rl_scripts/run_rlt_pretrain.sh [GPU_ID]
#
# Usage:
#   ENCODER_PATH=<rlt encoder.pt> bash scripts/run_rl_scripts/run_rlt_grpo.sh [GPU_ID]
#   ENCODER_PATH=... TASK_ID=3   bash scripts/run_rl_scripts/run_rlt_grpo.sh 1
#   ENCODER_PATH=... MULTI_TASK=1 bash scripts/run_rl_scripts/run_rlt_grpo.sh 1
#
# Env overrides:
#   ENCODER_PATH       Phase-1 rlt encoder.pt — REQUIRED (no auto-discovery:
#                      rlt and rlt_a encoders are indistinguishable by path)
#   TASK_ID            libero_goal task index (default 0)
#   MULTI_TASK         1 = --all_tasks (default 0 = single task)
#   CKPT_PATH          VLA ckpt (default Qwen 5traj)
#   ENCODER_HEADS      encoder attention heads — MUST match the pretrained encoder (default 8)
#   ENCODER_LAYERS     encoder layers          — MUST match (default 2)
#   DECODER_LAYERS     decoder layers          — MUST match (default 2)
#   MAX_LEN            decoder positional length — MUST match (default 4096)
#   GRPO_EPOCHS        epochs per iter (default 4; reuses --ppo_epochs flag)
#   GRPO_KL_COEF       KL-to-ref coefficient (default 0.04)
#   REF_UPDATE_INTERVAL  refresh reference actor every N iters (default 0 = never)
#   G_PER_TASK         episodes per iter per task (default 16; >= 8 for group signal)
#   GROUP_SIZE         episodes per initial state (default 4; needs >= 2)
#   NUM_ENVS_PER_TASK  parallel envs (default 8)
#   MAX_ITER           total iterations (default 300)
#   EVAL_INTERVAL      eval cadence (default 20)
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
ENCODER_HEADS=${ENCODER_HEADS:-8}
ENCODER_LAYERS=${ENCODER_LAYERS:-2}
DECODER_LAYERS=${DECODER_LAYERS:-2}
MAX_LEN=${MAX_LEN:-4096}
GRPO_EPOCHS=${GRPO_EPOCHS:-4}
GRPO_KL_COEF=${GRPO_KL_COEF:-0.04}
REF_UPDATE_INTERVAL=${REF_UPDATE_INTERVAL:-0}
G_PER_TASK=${G_PER_TASK:-16}
GROUP_SIZE=${GROUP_SIZE:-4}
NUM_ENVS_PER_TASK=${NUM_ENVS_PER_TASK:-8}
MAX_ITER=${MAX_ITER:-300}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}

DEFAULT_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
ENCODER_PATH="${ENCODER_PATH:?Set ENCODER_PATH to an rlt-track encoder.pt — pretrain one with: TRACK=rlt bash scripts/run_rl_scripts/run_rlt_pretrain.sh}"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}" >&2; exit 1
fi
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: rlt encoder not found: ${ENCODER_PATH}" >&2
    echo "       Pretrain one: TRACK=rlt bash scripts/run_rl_scripts/run_rlt_pretrain.sh" >&2
    exit 1
fi

if [ "${MULTI_TASK}" = "1" ]; then
    TASK_FLAG="--all_tasks"; RUN_TAG="rlt_grpo_qwen_alltasks"
else
    TASK_FLAG="--task_id ${TASK_ID}"; RUN_TAG="rlt_grpo_qwen_t${TASK_ID}"
fi
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_training/${RUN_TAG}_${TIMESTAMP}/rl_grpo"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

echo "============================================================"
echo " RLT Phase-2 GRPO  (Qwen, ${TASK_FLAG})"
echo "   GPU:          ${GPU_ID}"
echo "   ckpt:         ${CKPT_PATH}"
echo "   encoder:      ${ENCODER_PATH}  (rlt track)"
echo "   epochs/iter:  ${GRPO_EPOCHS}    kl_coef: ${GRPO_KL_COEF}"
echo "   G/task:       ${G_PER_TASK}    group_size: ${GROUP_SIZE}    envs/task: ${NUM_ENVS_PER_TASK}"
echo "   max_iter:     ${MAX_ITER}    eval_interval: ${EVAL_INTERVAL}"
echo "   output:       ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -u AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase grpo --encoder_mode rlt \
    --ckpt_path ${CKPT_PATH} --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal ${TASK_FLAG} \
    --encoder_layers ${ENCODER_LAYERS} --encoder_heads ${ENCODER_HEADS} \
    --decoder_layers ${DECODER_LAYERS} --max_len ${MAX_LEN} \
    --actor_hidden_dim 512 --critic_hidden_dim 512 \
    --ref_dropout 0.5 --fixed_std 0.1 \
    --G_per_task ${G_PER_TASK} --group_size ${GROUP_SIZE} --num_envs_per_task ${NUM_ENVS_PER_TASK} \
    --reward_coef 5.0 \
    --lr_actor 3e-4 --lr_critic 3e-4 --gamma 0.99 --max_grad_norm 1.0 \
    --ppo_epochs ${GRPO_EPOCHS} --clip_eps 0.2 \
    --grpo_kl_coef ${GRPO_KL_COEF} --ref_update_interval ${REF_UPDATE_INTERVAL} \
    --max_iter ${MAX_ITER} --eval_interval ${EVAL_INTERVAL} --eval_n_episodes 20 \
    --save_interval 50 --save_video_interval 100 \
    --seed 42 \
    --use_wandb --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_TAG}" --log_interval 1 \
    2>&1 | tee "${TRAIN_LOG}"
