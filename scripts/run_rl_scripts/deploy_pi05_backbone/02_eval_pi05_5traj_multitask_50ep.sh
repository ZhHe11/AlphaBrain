#!/bin/bash
# 50-ep offline eval of the Pi05-5traj multitask TD3 run (after 01 finishes).
# Evals iter300 (and all iters) across ALL 10 libero_goal tasks → fills the
# table-4 Pi05-5traj 全10任务 cell. Uses run_eval_rlt.sh (rl_offpolicy ckpts).
#
# Pi05 RLT arch = full-token bottleneck=2048, TD3 → prop_dim=8, no residual.
#
# Usage:  GPUS="2 3 5 7" bash 02_eval_pi05_5traj_multitask_50ep.sh
#   ITER=00300  to eval a single iter (default: all iters in the run)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

RUN_NAME="${RUN_NAME:-rlt_td3_pi05_5traj_mt}"
RUN_DIR=$(ls -dt results/rlt_training/${RUN_NAME}_*/rl_offpolicy 2>/dev/null | head -1)
[ -z "${RUN_DIR}" ] && { echo "ERROR: no run dir for ${RUN_NAME}_* — did 01 finish?" >&2; exit 1; }

RUN_DIR="${RUN_DIR}" \
VLA_CKPT="results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000" \
GPUS="${GPUS:-2 3 5 7}" \
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}" \
N_EPS="${N_EPS:-50}" \
ITER="${ITER:-}" \
BOTTLENECK_DIM=2048 ENCODER_LAYERS=2 ENCODER_HEADS=8 \
ACTOR_HIDDEN_DIM=512 REF_DROPOUT=0.5 FIXED_STD=0.1 \
PROP_DIM=8 \
NUM_WORKERS="${NUM_WORKERS:-4}" \
  bash scripts/run_rl_scripts/run_eval_rlt.sh
