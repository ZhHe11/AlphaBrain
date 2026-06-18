#!/bin/bash
# Run ON 192.168.33.5 (shared /share). Cleans broken partial launches, then
# starts a clean batch: keep the 1 healthy RLT_a+PPO multitask (GPU1), fill the
# other 7 GPUs with LIGHT single-task PPO/GRPO seeds (prioritized per user).
# Each single-task run ~8 env workers; all 7 ≈ 56 workers, load stays low.
set -u
cd /share/zhanghe/AlphaBrain-zh
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma
unset DISPLAY
mkdir -p logs
RLT_ENC=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt

# 1) kill broken/partial launches (the malformed run_name=+ one and any half-dead
#    rlt_ppo multitask). KEEP rlt_a_ppo_qwen_alltasks_s44 (healthy, GPU1).
for bad in "run_name +" "rlt_ppo_qwen_alltasks_s4" "rlt_a_ppo_qwen_alltasks_s43"; do
  pkill -9 -f "$bad" 2>/dev/null
done
sleep 4

# 2) launch one single-task seed on a GPU.  args: algo task seed gpu
launch_st() {
  local algo=$1 task=$2 seed=$3 gpu=$4
  local rn="rlt_${algo}_qwen_t${task}_s${seed}"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" SEED="${seed}" TASK_ID="${task}" MULTI_TASK=0 \
    RUN_NAME="${rn}" ENCODER_PATH="${RLT_ENC}" \
    setsid bash scripts/run_rl_scripts/run_rlt_${algo}.sh "${gpu}" \
      > "logs/${rn}_remote.log" 2>&1 < /dev/null &
  echo "  launched ${rn} -> GPU ${gpu} (bg pid $!)"
}

# 7 light single-task seeds on GPU 0,2,3,4,5,6,7 (GPU1 = healthy RLT_a+PPO mt)
launch_st ppo  0 43 0
launch_st ppo  1 43 2
launch_st grpo 0 43 3
launch_st grpo 1 43 4
launch_st ppo  0 44 5
launch_st ppo  1 44 6
launch_st grpo 0 44 7

sleep 5
echo "=== launched. current train.py procs: $(pgrep -fc 'train.py --phase') ==="
