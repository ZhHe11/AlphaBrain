# ============================================================================
# 多种子 mean±std — 5traj 多任务 headline 格,seed 43 & 44。共 12 个 run。
# 网页启动格式:每条 = 1 个独立节点(本地 GPU 0)。整行复制一条到一个节点。
# 每个 ~80-100 env workers, ~3-4h。launcher 已 patch(SEED= / RUN_NAME=)。
# 优先级:#7/#8 (RLT_a+PPO 榜首 0.936) > #1/#2 (RLT+PPO 0.916) > 其余。
# 跑完每个 → 50-ep 全10任务 eval 记 overall_sr,凑齐 seed 42/43/44 算 mean±std。
# ============================================================================


### 1  RLT + PPO  multitask  seed43   [seed42 基线 = 0.916]
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=43 MULTI_TASK=1 RUN_NAME=rlt_ppo_qwen_alltasks_s43 ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_ppo.sh 0 2>&1 | tee logs/rlt_ppo_mt_s43_$(date +%H%M).log


### 2  RLT + PPO  multitask  seed44
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=44 MULTI_TASK=1 RUN_NAME=rlt_ppo_qwen_alltasks_s44 ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_ppo.sh 0 2>&1 | tee logs/rlt_ppo_mt_s44_$(date +%H%M).log


### 3  RLT + GRPO  multitask  seed43   [seed42 基线 = 0.720]
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=43 MULTI_TASK=1 RUN_NAME=rlt_grpo_qwen_alltasks_s43 ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_grpo.sh 0 2>&1 | tee logs/rlt_grpo_mt_s43_$(date +%H%M).log


### 4  RLT + GRPO  multitask  seed44
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=44 MULTI_TASK=1 RUN_NAME=rlt_grpo_qwen_alltasks_s44 ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_grpo.sh 0 2>&1 | tee logs/rlt_grpo_mt_s44_$(date +%H%M).log


### 5  RLT + TD3  multitask  seed43   [seed42 基线 = 0.830]  (buf300k 防 OOM)
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=43 TRACK=rlt MULTI_TASK=1 BUFFER_CAPACITY=300000 RUN_NAME=rlt_td3_qwen_alltasks_s43 ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_rl.sh 0 2>&1 | tee logs/rlt_td3_mt_s43_$(date +%H%M).log


### 6  RLT + TD3  multitask  seed44   (buf300k 防 OOM)
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=44 TRACK=rlt MULTI_TASK=1 BUFFER_CAPACITY=300000 RUN_NAME=rlt_td3_qwen_alltasks_s44 ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_rl.sh 0 2>&1 | tee logs/rlt_td3_mt_s44_$(date +%H%M).log


### 7  RLT_a + PPO  multitask  seed43   [seed42 基线 = 0.936  ← 榜首,最高优先]
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=43 MULTI_TASK=1 RUN_NAME=rlt_a_ppo_qwen_alltasks_s43 ENCODER_PATH=results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_a_ppo.sh 0 2>&1 | tee logs/rlt_a_ppo_mt_s43_$(date +%H%M).log


### 8  RLT_a + PPO  multitask  seed44   [← 榜首,最高优先]
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=44 MULTI_TASK=1 RUN_NAME=rlt_a_ppo_qwen_alltasks_s44 ENCODER_PATH=results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_a_ppo.sh 0 2>&1 | tee logs/rlt_a_ppo_mt_s44_$(date +%H%M).log


### 9  RLT_a + GRPO  multitask  seed43   [seed42 基线 = 0.704]
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=43 MULTI_TASK=1 RUN_NAME=rlt_a_grpo_qwen_alltasks_s43 ENCODER_PATH=results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_a_grpo.sh 0 2>&1 | tee logs/rlt_a_grpo_mt_s43_$(date +%H%M).log


### 10  RLT_a + GRPO  multitask  seed44
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=44 MULTI_TASK=1 RUN_NAME=rlt_a_grpo_qwen_alltasks_s44 ENCODER_PATH=results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_a_grpo.sh 0 2>&1 | tee logs/rlt_a_grpo_mt_s44_$(date +%H%M).log


### 11  RLT_a + TD3  multitask  seed43   [seed42 基线 = 0.920 release]
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=43 TRACK=rlt_a MULTI_TASK=1 RUN_NAME=rlt_a_td3_qwen_alltasks_s43 ENCODER_PATH=results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_rl.sh 0 2>&1 | tee logs/rlt_a_td3_mt_s43_$(date +%H%M).log


### 12  RLT_a + TD3  multitask  seed44
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla && cd /share/zhanghe/AlphaBrain-zh && export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PALIGEMMA_TOKENIZER_PATH=/datasets/peligemma && unset DISPLAY && mkdir -p logs && SEED=44 TRACK=rlt_a MULTI_TASK=1 RUN_NAME=rlt_a_td3_qwen_alltasks_s44 ENCODER_PATH=results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt bash scripts/run_rl_scripts/run_rlt_rl.sh 0 2>&1 | tee logs/rlt_a_td3_mt_s44_$(date +%H%M).log
