#!/bin/bash
# Pi05 (PaliGemmaPi05) RLT_ori Phase-2 TD3 launcher.
# Same hyperparams + steplock for both 1traj and 5traj variants; only the
# VLA backbone and Phase-1 encoder differ.
#
# Phase-2 RL is wired through the Pi05 inference adapter
# (algos/RLT_ori/pi05_inference_zhanghe.py) which runs PaliGemma's prefix
# Gemma forward + flow-matching diffusion in a single fused call per
# rollout step. Trainer/rollout/eval all dispatch on framework type
# (is_pi05) — no behavior change for Qwen runs.
#
# Usage:
#   VARIANT=1traj TASK_ID=0 bash scripts/run_rl_scripts/run_rlt_ori_rl_task0_release_pi05.sh [GPU_ID]
#   VARIANT=5traj TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_ori_rl_task0_release_pi05.sh [GPU_ID]
#
# Env overrides:
#   VARIANT      1traj|5traj  (default 1traj)
#   TASK_ID      libero_goal task index (default 0)
#   CKPT_PATH    override VLA ckpt directory
#   ENCODER_PATH override Phase-1 encoder.pt
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
VARIANT=${VARIANT:-1traj}

if [[ "${VARIANT}" != "1traj" && "${VARIANT}" != "5traj" ]]; then
    echo "ERROR: VARIANT must be '1traj' or '5traj' (got '${VARIANT}')"
    exit 1
fi

# Latest pretrain dir for this variant (timestamp-suffixed); pick the most
# recent one matching the prefix unless caller overrides ENCODER_PATH.
DEFAULT_CKPT="results/training/Pi05-goal-${VARIANT}-openpi/checkpoints/steps_30000"
DEFAULT_PRETRAIN_DIR=$(ls -td results/rlt_ori_training/pi05_${VARIANT}_openpi_strict_*/pretrain 2>/dev/null | head -1 || true)
DEFAULT_ENCODER="${DEFAULT_PRETRAIN_DIR:+${DEFAULT_PRETRAIN_DIR}/checkpoints/pretrain_best/encoder.pt}"

CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
ENCODER_PATH="${ENCODER_PATH:-${DEFAULT_ENCODER}}"

RUN_NAME="rlt_ori_rl_t${TASK_ID}_release_pi05_${VARIANT}"
TIMESTAMP=$(date +%m%d_%H%M)
OUTPUT_DIR="results/rlt_ori_training/${RUN_NAME}_${TIMESTAMP}/rl_offpolicy"
mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

if [ ! -d "${CKPT_PATH}" ]; then
    echo "ERROR: VLA ckpt not found: ${CKPT_PATH}"
    exit 1
fi
if [ -z "${ENCODER_PATH}" ] || [ ! -f "${ENCODER_PATH}" ]; then
    echo "ERROR: RLT_ori encoder not found: ${ENCODER_PATH:-<unset>}"
    echo "       (Pi05 ${VARIANT} Phase-1 may still be running.)"
    exit 1
fi

echo "============================================================"
echo " RLT_ori Phase-2 TD3 (release-pi05-${VARIANT}, libero_goal task ${TASK_ID})"
echo "   GPU:        ${GPU_ID}"
echo "   ckpt:       ${CKPT_PATH}"
echo "   encoder:    ${ENCODER_PATH}"
echo "   output:     ${OUTPUT_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase rl_offpolicy \
    --encoder_mode rlt_ori \
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
