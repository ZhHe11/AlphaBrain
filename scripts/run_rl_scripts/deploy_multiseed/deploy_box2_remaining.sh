#!/bin/bash
# ============================================================================
# RUN ON box2 (192.168.33.5) — deploys the 2 genuinely-remaining report cells.
# box1 (the pod) cannot route to box2, so this must be launched from box2.
#
#   #19  RLT+TD3 1traj task0  → 50-ep offline eval of the FINISHED iter300 ckpt
#                               (light, single-GPU, ~30min) → table 1c
#   #21  Pi05-1traj all-10 TD3 → RESUME from iter50 (OOM-killed) → iter300
#                               (heavy multitask; cross-backbone table 4)
#
# Safety: #19 eval runs immediately (light). #21 is HEAVY (~100 env workers);
# it launches ONLY when active multitask trainings drop below MAX (default 2)
# so it does not over-pack box2 while the two s43 TD3 runs are still going
# (over-pack is exactly what OOM-killed it last time). The script waits.
#
# Usage (on box2):
#   bash scripts/run_rl_scripts/deploy_multiseed/deploy_box2_remaining.sh
#   EVAL_GPU=2 TRAIN_GPU=3 bash .../deploy_box2_remaining.sh   # pin GPUs
# ============================================================================
set -u
ROOT=/share/zhanghe/AlphaBrain-zh
cd "$ROOT"
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla
[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export ALPHABRAIN_ROOT=$ROOT TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_WORKER_TIMEOUT=300
mkdir -p logs results/eval_remaining_0612

MAX=${MAX:-2}   # max concurrent multitask trainings before launching the heavy #21
pick_free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | awk -F, 'NR==1{gsub(/ /,"",$1);print $1}'; }
EVAL_GPU=${EVAL_GPU:-$(pick_free_gpu)}
mt_count() { ps -eo cmd | grep "train.py --phase" | grep -v grep \
  | grep -cE "alltasks|all_tasks|_mt_|pi05_1traj_mt"; }

VLA_1TRAJ=$ROOT/results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model
RLT_1TRAJ_ENC=$ROOT/results/rlt_training/1traj_libero_goal_step30k_0423_0545/pretrain/checkpoints/pretrain_best/encoder.pt
TD3_T0_CKPT=$ROOT/results/rlt_training/rlt_td3_1traj_t0_0609b_0610_1127/rl_offpolicy/checkpoints/rl_offpolicy_iter_00300

# ── #19: 50-ep offline eval of the finished RLT+TD3 1traj task0 ckpt ──────────
#   RLT full-token: bottleneck 2048, heads 8, prop_dim 8 (td3, no residual), --max_len 4096
echo "[$(date +%H:%M)] #19 eval RLT+TD3 1traj t0 (iter300, 50-ep, task0) on GPU $EVAL_GPU"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python $ROOT/AlphaBrain/training/reinforcement_learning/eval/eval_libero_rlt.py \
    --vla_ckpt $VLA_1TRAJ --action_token_ckpt "$TD3_T0_CKPT" \
    --suite libero_goal --task_ids 0 --n_eps_per_task 50 --gpu 0 --num_workers 4 --seed 42 \
    --bottleneck_dim 2048 --encoder_layers 2 --encoder_heads 8 --max_len 4096 \
    --actor_hidden_dim 512 --ref_dropout 0.5 --fixed_std 0.1 --prop_dim 8 \
    --results_json results/eval_remaining_0612/rlt_td3_1traj_t0.json \
    > logs/eval_rlt_td3_1traj_t0.log 2>&1 &
echo "       -> logs/eval_rlt_td3_1traj_t0.log ; result -> results/eval_remaining_0612/rlt_td3_1traj_t0.json"

# ── #21: resume Pi05-1traj all-10 TD3 from iter50 — wait for a free slot ──────
TRAIN_GPU=${TRAIN_GPU:-}
echo "[$(date +%H:%M)] #21 Pi05-1traj all-10 TD3 resume: waiting for multitask slot (MAX=$MAX)..."
while [ "$(mt_count)" -ge "$MAX" ]; do
  echo "  [$(date +%H:%M)] $(mt_count) multitask runs active (>= $MAX) — waiting 5min for a TD3 to finish..."
  sleep 300
done
[ -z "$TRAIN_GPU" ] && TRAIN_GPU=$(pick_free_gpu)
echo "[$(date +%H:%M)] slot free — launching #21 on GPU $TRAIN_GPU (RESUME from iter50)"
export MUJOCO_EGL_DEVICE_ID=$TRAIN_GPU
SEED=42 TRACK=rlt BACKBONE=pi05 VARIANT=1traj MULTI_TASK=1 RESUME=1 \
  RUN_NAME=rlt_td3_pi05_1traj_mt_0609 \
  ENCODER_PATH=$ROOT/results/rlt_training/pi05_1traj_openpi_strict_0427_0721/pretrain/checkpoints/pretrain_best/encoder.pt \
  EVAL_INTERVAL=9999 MAX_ITER=300 \
  bash $ROOT/scripts/run_rl_scripts/run_rlt_rl.sh $TRAIN_GPU \
  > logs/box2_rlt_td3_pi05_1traj_mt_resume.log 2>&1 &
echo "       -> logs/box2_rlt_td3_pi05_1traj_mt_resume.log"
echo "[$(date +%H:%M)] both deployed. Monitor with: tail -f logs/box2_rlt_td3_pi05_1traj_mt_resume.log"
