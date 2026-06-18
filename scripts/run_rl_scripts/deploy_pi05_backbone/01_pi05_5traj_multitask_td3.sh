#!/bin/bash
# KEY cross-backbone experiment: Pi0.5 (PaliGemmaPi05) as RLT backbone, 5traj,
# MULTI-TASK (全10任务) TD3. Fills the table-4 Pi05-5traj 全10任务 cell (currently
# n/a) so we get a true backbone-vs-backbone multi-task comparison against
# QwenOFT-5traj (RLT+TD3 multitask = 0.83, table 1).
#
# Why TD3: run_rlt_rl.sh wires BACKBONE=pi05 for TRACK=rlt (RLT_a × Pi05 not
# wired; PPO/GRPO launchers don't take pi05). TD3 is the released RLT algo and
# matches the QwenOFT-5traj multitask number we compare to.
#
# Safety: BUFFER_CAPACITY=300000 (Pi0.5 full-token bottleneck=2048 has the same
# memory profile that OOM'd RLT+TD3 multitask at 1M buffer — buf300k fixed it).
# Picks up the 0603 rollout per-env soft-fail, so a single env-pool timeout no
# longer kills the run.
#
# Usage:  bash 01_pi05_5traj_multitask_td3.sh <GPU_ID>
#   GPU_ID  single GPU index (default 0)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

GPU_ID="${1:-0}"

BACKBONE=pi05 VARIANT=5traj TRACK=rlt MULTI_TASK=1 \
  BUFFER_CAPACITY="${BUFFER_CAPACITY:-300000}" \
  MAX_ITER="${MAX_ITER:-300}" \
  EVAL_INTERVAL="${EVAL_INTERVAL:-20}" \
  G_PER_TASK="${G_PER_TASK:-10}" \
  NUM_ENVS_PER_TASK="${NUM_ENVS_PER_TASK:-10}" \
  RUN_NAME="${RUN_NAME:-rlt_td3_pi05_5traj_mt}" \
  bash scripts/run_rl_scripts/run_rlt_rl.sh "${GPU_ID}"
