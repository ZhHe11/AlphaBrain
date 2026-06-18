#!/usr/bin/env bash
# Eval ALL ckpts of a VLA-baseline run (iter 50..300) at offline 50-ep / 10-task,
# to plot the VLA-baseline process curve (shows training instability vs RLT).
# Usage:  ALGO=ppo GPUS="1 2 3" bash vla_curve_eval.sh
#         ALGO=grpo GPUS="1 3 5" bash vla_curve_eval.sh   (on box2)
set -u
ROOT=/share/zhanghe/AlphaBrain-zh
cd "$ROOT"
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla
[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl ALPHABRAIN_ROOT=$ROOT
unset DISPLAY
mkdir -p "$ROOT/logs" "$ROOT/results/eval_vla_curve_0607"

ALGO=${ALGO:-ppo}
GPUS=(${GPUS:-1 2 3})
NGPU=${#GPUS[@]}
VLA=results/training/QwenOFT-5traj-libero_goal/final_model
case $ALGO in
  ppo)  RUN=vla_ppo_qwen_alltasks_0531_2052/vla_ppo ;;
  grpo) RUN=vla_grpo_qwen_alltasks_0531_2112/vla_grpo ;;
esac
CKDIR=$ROOT/results/rlt_training/$RUN/checkpoints

echo "[$(date +%H:%M)] VLA-$ALGO curve eval start; GPUs=${GPUS[*]}; ckdir=$CKDIR"
i=0
for it in 00050 00100 00150 00200 00250 00300; do
  ckpt=$CKDIR/vla_${ALGO}_iter_${it}/vla_state_dict.pt
  out=$ROOT/results/eval_vla_curve_0607/${ALGO}_iter${it}.json
  [ -f "$out" ] && { echo "  skip $ALGO iter$it (json exists)"; continue; }
  [ -f "$ckpt" ] || { echo "  MISSING $ckpt"; continue; }
  gpu=${GPUS[$((i % NGPU))]}
  echo "  [$(date +%H:%M)] launch $ALGO iter$it on GPU $gpu"
  # CUDA_VISIBLE_DEVICES must be set BEFORE python starts: an import inits CUDA
  # with the default device, so eval_libero.py's internal --gpu set is too late.
  # Mask to the target GPU here and pass --gpu 0 (the only visible device).
  CUDA_VISIBLE_DEVICES=$gpu python AlphaBrain/training/reinforcement_learning/eval/eval_libero.py \
    --vla_ckpt $VLA --base_vla --vla_state_dict "$ckpt" \
    --suite libero_goal --task_ids 0,1,2,3,4,5,6,7,8,9 \
    --n_eps_per_task 50 --gpu 0 --num_workers 4 --seed 42 \
    --results_json "$out" > $ROOT/logs/vla_curve_${ALGO}_iter${it}.log 2>&1 &
  i=$((i + 1))
  # bound concurrency to NGPU: every NGPU launches, wait for them
  [ $((i % NGPU)) -eq 0 ] && { echo "    (waiting for batch of $NGPU)"; wait; }
  sleep 8
done
wait
echo "[$(date +%H:%M)] VLA-$ALGO curve eval DONE"
