#!/bin/bash
# 50-ep offline re-eval of the RLT (full-token) single-task 1traj runs
# (PPO / GRPO × task 0/1/3, QwenOFT-1traj base) → fills table 1c with
# protocol-matched 50-ep numbers. Mirrors run_eval_rlt_singletask_50ep.sh
# but points at the 1traj base VLA + the 0602 run ckpts (some runs died
# from env-pool timeout at iter 150-250, so each job evals its LAST ckpt).
#
# Architecture (same trainer as 5traj → same dims):
#   PPO / GRPO : prop_dim=0, residual=True
#
# Usage: GPUS="2 3 5 7" bash scripts/run_rl_scripts/run_eval_rlt_1traj_50ep.sh
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

VLA_CKPT="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
RT="results/rlt_training"
OUT_ROOT="${OUT_ROOT:-results/eval_rlt_1traj_50ep_0603}"
N_EPS="${N_EPS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
# Concurrency cap — 0603 lesson: launching all 6 evals at once (×workers) on an
# already-loaded box starved every env worker (all resets timed out → crash).
# Run at most CONCURRENCY evals simultaneously so total env-workers stay bounded.
CONCURRENCY="${CONCURRENCY:-2}"
read -r -a GPUS <<< "${GPUS:-2 3 5 7}"
mkdir -p "${OUT_ROOT}"

# job: "label|task_id|ckpt_dir|prop_dim|residual_flag"
JOBS=(
  "ppo_t0|0|${RT}/rlt_ppo_qwen_t0_0602_1612/rl_ppo/checkpoints/rl_iter_00200|0|--residual"
  "ppo_t1|1|${RT}/rlt_ppo_qwen_t1_0602_1612/rl_ppo/checkpoints/rl_iter_00200|0|--residual"
  "ppo_t3|3|${RT}/rlt_ppo_qwen_t3_0602_1612/rl_ppo/checkpoints/rl_iter_00150|0|--residual"
  "grpo_t0|0|${RT}/rlt_grpo_qwen_t0_0602_1612/rl_grpo/checkpoints/grpo_iter_00200|0|--residual"
  "grpo_t1|1|${RT}/rlt_grpo_qwen_t1_0602_1612/rl_grpo/checkpoints/grpo_iter_00250|0|--residual"
  "grpo_t3|3|${RT}/rlt_grpo_qwen_t3_0602_1612/rl_grpo/checkpoints/grpo_iter_00150|0|--residual"
)

echo "============================================================"
echo " RLT 1traj single-task 50-ep eval | ${#JOBS[@]} jobs on GPUs [${GPUS[*]}]"
echo "   base VLA: ${VLA_CKPT}"
echo "   out:      ${OUT_ROOT}"
echo "============================================================"

for j in "${JOBS[@]}"; do
  IFS='|' read -r label tid ckpt prop res <<< "$j"
  [ -d "$ckpt" ] || { echo "ERROR: missing ckpt for ${label}: ${ckpt}" >&2; exit 1; }
done

SHARD_PIDS=(); TAIL_PIDS=()
_cleanup () { trap - EXIT INT TERM
  for pid in "${SHARD_PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true; }
trap _cleanup EXIT INT TERM

echo "[concurrency] running at most ${CONCURRENCY} evals at a time"
fail=0; i=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r label tid ckpt prop res <<< "$j"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  out="${OUT_ROOT}/${label}.json"; log="${OUT_ROOT}/${label}.log"; : > "${log}"
  rm -f "${out}"
  CUDA_VISIBLE_DEVICES=${gpu} setsid python AlphaBrain/training/reinforcement_learning/eval/eval_libero_rlt.py \
      --vla_ckpt "${VLA_CKPT}" \
      --action_token_ckpt "${ckpt}" \
      --suite libero_goal --task_ids "${tid}" \
      --n_eps_per_task ${N_EPS} --gpu 0 --num_workers ${NUM_WORKERS} --seed ${SEED} \
      --bottleneck_dim 2048 --encoder_layers 2 --encoder_heads 8 --max_len 4096 \
      --actor_hidden_dim 512 --ref_dropout 0.5 --fixed_std 0.1 \
      --prop_dim ${prop} ${res} \
      --results_json "${out}" \
      > "${log}" 2>&1 &
  pid=$!; SHARD_PIDS+=(${pid})
  echo "[launch] ${label} (task ${tid}) → GPU ${gpu} pid=${pid} ckpt=$(basename ${ckpt})"
  i=$(( i + 1 ))
  # Throttle: once CONCURRENCY jobs are in flight, wait for the oldest to finish.
  if [ "$(jobs -rp | wc -l)" -ge "${CONCURRENCY}" ]; then
    if ! wait -n; then fail=1; fi
  fi
  sleep 3
done

echo "[all launched] waiting for remaining jobs..."
for pid in "${SHARD_PIDS[@]}"; do
  if ! wait "${pid}"; then echo "ERROR: pid ${pid} FAILED — check ${OUT_ROOT}/*.log" >&2; fail=1; fi
done

echo "============================================================"
echo " RESULTS"
for j in "${JOBS[@]}"; do
  IFS='|' read -r label tid ckpt prop res <<< "$j"
  out="${OUT_ROOT}/${label}.json"
  if [ -f "$out" ]; then
    sr=$(python3 -c "import json;d=json.load(open('$out'));x=d[-1] if isinstance(d,list) else d;print(f\"{x['overall_sr']:.3f}\")" 2>/dev/null)
    echo "  ${label}: SR=${sr}  (ckpt $(basename ${ckpt}))"
  else
    echo "  ${label}: NO OUTPUT"
  fi
done
echo "============================================================"
exit ${fail}
