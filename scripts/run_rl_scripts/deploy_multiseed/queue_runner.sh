#!/bin/bash
# ============================================================================
# Self-driving multitask seed queue runner (per box).
# Maintains EXACTLY <=3 concurrent multitask runs (hard cap — never thrash).
# When a slot frees (a run hits "Done. Metrics"), evals the finished run
# (50-ep all tasks) then launches the next queued seed. Exits when its whole
# queue is trained + eval'd.
#
# Usage:  BOX=box1 nohup bash queue_runner.sh > logs/queue_box1.log 2>&1 &
#   (box2: run via tmux new-session -d with BOX=box2)
#
# Safety: hard MAX=3; only launches when running_count < MAX; staggers 40s.
# ============================================================================
set -u
ROOT=/share/zhanghe/AlphaBrain-zh
cd "$ROOT"
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla
[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export ALPHABRAIN_ROOT=$ROOT TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"   # eval_libero_rlt.py needs repo root on path
mkdir -p logs results/eval_mt_seeds_0607

RLT_ENC=$ROOT/results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt
RLTA_ENC=$ROOT/results/rlt_training/rlt_a_QwenOFT-5traj-libero_goal_0526_1243/pretrain/checkpoints/pretrain_best/encoder.pt
VLA=$ROOT/results/training/QwenOFT-5traj-libero_goal/final_model
MAX=${MAX:-3}   # env-overridable; lower to 2 on a box with external interference (e.g. someone else's debug.py)
BOX=${BOX:-box1}

# Queue entries: "run_name|kind|seed|gpu"
#   kind ∈ {rlt_ppo, rlt_grpo, rlt_td3, rlta_ppo, rlta_grpo, rlta_td3}
# box1 handles the s43 half of the 3 not-yet-covered cells (+ already-running covered ones are skipped if present);
# box2 handles s44 half. Already-running seeds (RLT_a+PPO, RLT+PPO, RLT+GRPO) are NOT in the queue (handled separately) —
# this runner only drives the 3 REMAINING cells per box.
# Full list of 6 runs this box owns: 3 already-running covered cells (runner
# only EVALs them — run_dir exists so launch() is skipped) + 3 to-launch.
# Clean split: box1 = all 6 cells seed43, box2 = all 6 cells seed44 (v2 names
# so a fresh relaunch isn't blocked by the old eval-bound dirs). EVAL_INTERVAL
# is forced high in launch() — these seed runs need only iter300 + offline eval;
# the per-20-iter multitask in-train eval (10 tasks × 20ep ≈ 2.5h each) was the
# dominant cost (~40h wasted/run) — skipping it ~4× speeds each run.
if [ "$BOX" = "box1" ]; then
  QUEUE=(
    "rlt_a_ppo_qwen_alltasks_s43v2|rlta_ppo|43|0"
    "rlt_ppo_qwen_alltasks_s43v2|rlt_ppo|43|1"
    "rlt_grpo_qwen_alltasks_s43v2|rlt_grpo|43|2"
    "rlt_td3_qwen_alltasks_s43v2|rlt_td3|43|3"
    "rlt_a_grpo_qwen_alltasks_s43v2|rlta_grpo|43|4"
    "rlt_a_td3_qwen_alltasks_s43v2|rlta_td3|43|5"
  )
else
  QUEUE=(
    "rlt_a_ppo_qwen_alltasks_s44v2|rlta_ppo|44|0"
    "rlt_ppo_qwen_alltasks_s44v2|rlt_ppo|44|1"
    "rlt_grpo_qwen_alltasks_s44v2|rlt_grpo|44|2"
    "rlt_td3_qwen_alltasks_s44v2|rlt_td3|44|3"
    "rlt_a_grpo_qwen_alltasks_s44v2|rlta_grpo|44|4"
    "rlt_a_td3_qwen_alltasks_s44v2|rlta_td3|44|5"
  )
fi

running_count() { ps -eo cmd | grep "train.py --phase" | grep -v grep | grep -cE "alltasks_s4[34]"; }
is_done() { grep -q "Done. Metrics" "$1" 2>/dev/null; }
run_dir() { ls -dt "$ROOT"/results/rlt_training/${1}_*/ 2>/dev/null | head -1; }

launch() {  # run_name kind seed gpu
  local name=$1 kind=$2 seed=$3 gpu=$4
  export MUJOCO_EGL_DEVICE_ID=$gpu
  export EVAL_INTERVAL=9999   # skip the slow multitask in-train eval; final offline eval only
  case $kind in
    rlt_ppo)  SEED=$seed MULTI_TASK=1 RUN_NAME=$name ENCODER_PATH=$RLT_ENC  bash $ROOT/scripts/run_rl_scripts/run_rlt_ppo.sh  $gpu > logs/qr_${name}.log 2>&1 & ;;
    rlt_grpo) SEED=$seed MULTI_TASK=1 RUN_NAME=$name ENCODER_PATH=$RLT_ENC  bash $ROOT/scripts/run_rl_scripts/run_rlt_grpo.sh $gpu > logs/qr_${name}.log 2>&1 & ;;
    rlt_td3)  SEED=$seed TRACK=rlt    MULTI_TASK=1 BUFFER_CAPACITY=300000 RUN_NAME=$name ENCODER_PATH=$RLT_ENC bash $ROOT/scripts/run_rl_scripts/run_rlt_rl.sh $gpu > logs/qr_${name}.log 2>&1 & ;;
    rlta_ppo) SEED=$seed MULTI_TASK=1 RUN_NAME=$name ENCODER_PATH=$RLTA_ENC bash $ROOT/scripts/run_rl_scripts/run_rlt_a_ppo.sh $gpu > logs/qr_${name}.log 2>&1 & ;;
    rlta_grpo) SEED=$seed MULTI_TASK=1 RUN_NAME=$name ENCODER_PATH=$RLTA_ENC bash $ROOT/scripts/run_rl_scripts/run_rlt_a_grpo.sh $gpu > logs/qr_${name}.log 2>&1 & ;;
    rlta_td3) SEED=$seed TRACK=rlt_a  MULTI_TASK=1 RUN_NAME=$name ENCODER_PATH=$RLTA_ENC bash $ROOT/scripts/run_rl_scripts/run_rlt_rl.sh $gpu > logs/qr_${name}.log 2>&1 & ;;
  esac
  echo "[$(date +%H:%M)] launched $name (kind=$kind seed=$seed gpu=$gpu)"
}

eval_run() {  # run_name kind gpu
  local name=$1 kind=$2 egpu=${3:-0}
  local d=$(run_dir "$name"); [ -z "$d" ] && return
  local sub=$(ls "$d" 2>/dev/null | grep -E "rl_onpolicy|rl_ppo|rl_grpo|rl_offpolicy" | head -1)
  local ck=$(ls -d "$d$sub"/checkpoints/*00300* 2>/dev/null | head -1); [ -z "$ck" ] && return
  local prop res bn hd script extra
  case $kind in
    # RLT (full-token): eval_libero_rlt.py + bottleneck 2048, heads 8, --max_len
    rlt_ppo|rlt_grpo)     prop=0; res="--residual"; bn=2048; hd=8; script=eval/eval_libero_rlt.py; extra="--max_len 4096" ;;
    rlt_td3)              prop=8; res="";           bn=2048; hd=8; script=eval/eval_libero_rlt.py; extra="--max_len 4096" ;;
    # RLT_a (action-token, cls_token+bottleneck_proj): eval_libero.py + bottleneck 256, heads 4
    rlta_ppo|rlta_grpo)   prop=0; res="--residual"; bn=256;  hd=4; script=eval/eval_libero.py; extra="" ;;
    rlta_td3)             prop=8; res="";           bn=256;  hd=4; script=eval/eval_libero.py; extra="" ;;
  esac
  echo "[$(date +%H:%M)] eval $name (kind=$kind, 50-ep all tasks) on GPU $egpu"
  CUDA_VISIBLE_DEVICES=$egpu python $ROOT/AlphaBrain/training/reinforcement_learning/$script \
    --vla_ckpt $VLA --action_token_ckpt "$ck" --suite libero_goal --task_ids 0,1,2,3,4,5,6,7,8,9 \
    --n_eps_per_task 50 --gpu 0 --num_workers 4 --seed 42 \
    --bottleneck_dim $bn --encoder_layers 2 --encoder_heads $hd $extra --actor_hidden_dim 512 \
    --ref_dropout 0.5 --fixed_std 0.1 --prop_dim $prop $res \
    --results_json results/eval_mt_seeds_0607/${name}.json > logs/qr_eval_${name}.log 2>&1 &
}

# ── main loop ──
declare -A LAUNCHED EVALED
echo "[$(date +%H:%M)] queue_runner start (BOX=$BOX, queue=${#QUEUE[@]}, MAX=$MAX)"
while true; do
  all_done=1
  for entry in "${QUEUE[@]}"; do
    IFS='|' read -r name kind seed gpu <<< "$entry"
    d=$(run_dir "$name")
    if [ -z "$d" ]; then
      # not launched yet — launch if slot free
      all_done=0
      if [ "$(running_count)" -lt "$MAX" ]; then launch "$name" "$kind" "$seed" "$gpu"; sleep 40; fi
    else
      sub=$(ls "$d" 2>/dev/null | grep -E "rl_onpolicy|rl_ppo|rl_grpo|rl_offpolicy" | head -1)
      if is_done "$d$sub/train.log"; then
        if [ -z "${EVALED[$name]:-}" ] && [ ! -f results/eval_mt_seeds_0607/${name}.json ]; then
          eval_run "$name" "$kind" "$gpu"; EVALED[$name]=1
        fi
      else
        all_done=0
      fi
    fi
  done
  [ "$all_done" = "1" ] && { echo "[$(date +%H:%M)] all queue runs done+eval'd, exit"; break; }
  sleep 300
done
