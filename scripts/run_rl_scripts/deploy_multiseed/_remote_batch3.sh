#!/bin/bash
# Run ON 192.168.33.5. CLEAN conservative batch: 6 light single-task seeds,
# STAGGERED 30s (avoid NFS model-load I/O storm). 6 is the sustainable count —
# single-task PPO/GRPO turned out to be ~19 load each (10-epoch PPO update is
# compute-heavy), so 6 ≈ load 115 on 128 cores. Do NOT exceed 6 here.
set -u
cd /share/zhanghe/AlphaBrain-zh
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma
unset DISPLAY
mkdir -p logs
RLT_ENC=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt

launch_st() {  # algo task seed gpu
  local algo=$1 task=$2 seed=$3 gpu=$4
  local rn="rlt_${algo}_qwen_t${task}_s${seed}"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" SEED="${seed}" TASK_ID="${task}" MULTI_TASK=0 \
    RUN_NAME="${rn}" ENCODER_PATH="${RLT_ENC}" \
    setsid bash "scripts/run_rl_scripts/run_rlt_${algo}.sh" "${gpu}" \
      > "logs/${rn}_remote.log" 2>&1 < /dev/null &
  echo "  + ${rn} -> GPU ${gpu}  ($(date +%H:%M:%S))"
}

# 6 single-task seeds, one per GPU 0-5, staggered 30s. (avoid t3 = main box covers it)
launch_st ppo  0 43 0; sleep 30
launch_st ppo  1 43 1; sleep 30
launch_st grpo 0 43 2; sleep 30
launch_st grpo 1 43 3; sleep 30
launch_st ppo  0 44 4; sleep 30
launch_st ppo  1 44 5; sleep 5

echo "=== batch3 launched. train procs: $(pgrep -fc 'train.py --phase') | load: $(cat /proc/loadavg|cut -d' ' -f1) ==="
