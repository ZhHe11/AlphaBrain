#!/bin/bash
# RE-RUN the 3 table-3 ablations lost when the remote server crashed 2026-06-03
# (all died with ckpt=NONE, i.e. before iter-50, zero usable data).
# Run on the remote box (or any free GPUs) once it's back. Each is single-task
# task0, unique RUN_NAME (avoids the earlier rlt_rl_qwen_t0 collision), and
# picks up the 0603 per-env soft-fail so env-pool timeouts won't kill them.
#
# Usage:  bash 00_rerun_remote_ablations.sh        # uses GPUs 0,1,2
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Job 1: RLT_a 瓶颈 D=128  (table-3 D-sweep) — GPU 0
TRACK=rlt_a MULTI_TASK=0 TASK_ID=0 BOTTLENECK_DIM=128 \
  ENCODER_PATH=results/rlt_training/rlt_a_dim128_5traj_0601_1600/pretrain/checkpoints/pretrain_best/encoder.pt \
  RUN_NAME=rlt_a_td3_abl_dim128_t0 \
  bash scripts/run_rl_scripts/run_rlt_rl.sh 0 &

# Job 2: RLT_a 瓶颈 D=512  (table-3 D-sweep) — GPU 1
TRACK=rlt_a MULTI_TASK=0 TASK_ID=0 BOTTLENECK_DIM=512 \
  ENCODER_PATH=results/rlt_training/rlt_a_dim512_5traj_0601_1600/pretrain/checkpoints/pretrain_best/encoder.pt \
  RUN_NAME=rlt_a_td3_abl_dim512_t0 \
  bash scripts/run_rl_scripts/run_rlt_rl.sh 1 &

# Job 3: RLT ref_dropout=0  (table-3 ref_dropout cell) — GPU 2
TRACK=rlt MULTI_TASK=0 TASK_ID=0 REF_DROPOUT=0 \
  RUN_NAME=rlt_td3_abl_refdrop0_t0 \
  bash scripts/run_rl_scripts/run_rlt_rl.sh 2 &

wait
echo "[done] 3 ablation runs finished — now 50-ep eval each (singletask, task0)."
