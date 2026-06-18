#!/bin/bash
# Run ON 192.168.33.5. Fill remaining capacity with MORE light single-task
# seeds, STAGGERED (sleep between each) to avoid the NFS model-load I/O storm.
# Fills idle GPU1 + doubles a 2nd light run onto under-utilized GPUs (each light
# run ~8-10 workers; GPUs have 80GB so 2 runs/GPU fit memory; while one rolls
# out on CPU the other uses GPU → better util). Watch load; soft-fail covers bursts.
set -u
cd /share/zhanghe/AlphaBrain-zh
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma
unset DISPLAY
mkdir -p logs
RLT_ENC=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt
RLTA_ENC=results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt

# args: launcher_suffix(ppo|grpo|a_ppo|a_grpo) task seed gpu encoder
launch_st() {
  local algo=$1 task=$2 seed=$3 gpu=$4 enc=$5
  local rn="rlt_${algo}_qwen_t${task}_s${seed}"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" SEED="${seed}" TASK_ID="${task}" MULTI_TASK=0 \
    RUN_NAME="${rn}" ENCODER_PATH="${enc}" \
    setsid bash "scripts/run_rl_scripts/run_rlt_${algo}.sh" "${gpu}" \
      > "logs/${rn}_remote.log" 2>&1 < /dev/null &
  echo "  + ${rn} -> GPU ${gpu}"
}

# Fill idle GPU1 with the one missing RLT single-task seed.
launch_st grpo 1 44 1 "$RLT_ENC"; sleep 25

# Second layer: RLT_a single-task seeds (also need mean±std), doubled onto the
# least-busy GPUs. Staggered 25s each to keep NFS reads serial.
launch_st a_ppo  0 43 0 "$RLTA_ENC"; sleep 25
launch_st a_grpo 0 43 2 "$RLTA_ENC"; sleep 25
launch_st a_ppo  1 43 3 "$RLTA_ENC"; sleep 25
launch_st a_grpo 1 43 4 "$RLTA_ENC"; sleep 25

echo "=== batch2 done. train procs: $(pgrep -fc 'train.py --phase') | load: $(cat /proc/loadavg|cut -d' ' -f1) ==="
