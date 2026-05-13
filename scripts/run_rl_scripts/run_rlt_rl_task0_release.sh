#!/bin/bash
# Single-task RLT release config — port of
# /share/zhanghe/VLA-Engine-Developer/scripts/run_rlt_1traj_task0_release.sh
# (TD3 off-policy hyperparams), but with the RLT encoder family
# (--encoder_mode rlt, bottleneck=2048, heads=8).
#
# Phase-1 (encoder pretrain) is NOT auto-run here — produce the encoder
# first via scripts/run_rl_scripts/run_rlt_pretrain.sh and point
# ENCODER_PATH at it (override below or via env var).
#
# Usage:
#   bash scripts/run_rl_scripts/run_rlt_rl_task0_release.sh [GPU_ID]
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

CKPT_PATH="results/training/QwenOFT-5traj-libero_goal/final_model"
ENCODER_PATH="results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt"

RUN_NAME="rlt_rl_t${TASK_ID}_release"
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_training/${RUN_NAME}_${TIMESTAMP}/rl_offpolicy"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}"
    exit 1
fi
if [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: RLT encoder not found: ${ENCODER_PATH}"
    echo "       Run scripts/run_rl_scripts/run_rlt_pretrain.sh first."
    exit 1
fi

echo "============================================================"
echo " RLT Phase-2 TD3 (release config, libero_goal task ${TASK_ID})"
echo "   GPU:        ${GPU_ID}"
echo "   ckpt:       ${CKPT_PATH}"
echo "   encoder:    ${ENCODER_PATH}"
echo "   output:     ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --encoder_mode rlt \
    --ckpt_path ${CKPT_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --suite libero_goal \
    --task_id ${TASK_ID} \
    --rollout_gpus 0 \
    --train_gpu 0 \
    --bottleneck_dim 2048 \
    --encoder_layers 2 \
    --encoder_heads 8 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 512 \
    --ref_dropout 0.5 \
    --fixed_std 0.1 \
    --G 64 \
    --group_size 8 \
    --num_envs 64 \
    --reward_coef 5.0 \
    --lr_actor 1e-3 \
    --lr_critic 1e-3 \
    --gamma 0.99 \
    --max_grad_norm 1.0 \
    --buffer_capacity 1000000 \
    --buffer_warmup 256 \
    --warmup_iters 5 \
    --td_updates_per_iter 10000 \
    --utd_ratio 10.0 \
    --td_batch_size 1024 \
    --tau 0.005 \
    --beta 1.0 \
    --actor_update_freq 2 \
    --target_noise_std 0.2 \
    --target_noise_clip 0.5 \
    --max_iter 300 \
    --eval_interval 10 \
    --eval_n_episodes 20 \
    --save_interval 25 \
    --save_video_interval 999 \
    --seed 42 \
    --use_wandb \
    --wandb_project AlphaBrain_RLT \
    --run_name "${RUN_NAME}" \
    --log_interval 1 \
    --use_steplock \
    2>&1 | tee "${TRAIN_LOG}"
