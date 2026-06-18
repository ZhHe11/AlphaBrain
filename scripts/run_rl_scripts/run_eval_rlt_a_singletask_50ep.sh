#!/bin/bash
# Uniform 50-ep offline re-eval of the RLT_a single-task runs (action-token
# bottleneck, eval_libero.py). Companion to run_eval_rlt_singletask_50ep.sh.
# Covers 8 cells: TD3/GRPO × t0/t1/t3 + PPO × t0/t1.  (PPO t3 has no clean
# iter300 ckpt — retrain stalled at iter200 — so it is omitted here.)
#
# Arch (verified from actor.pt input dims, bottleneck=256):
#   TD3 : prop_dim=8, residual=False  (actor input 256+8+56=320)
#   GRPO: prop_dim=0, residual=True   (actor input 256+0+56=312)
#   PPO : prop_dim=0, residual=True
#
# Usage: GPUS="1 2 3 4 5" bash scripts/run_rl_scripts/run_eval_rlt_a_singletask_50ep.sh
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

VLA_CKPT="results/training/QwenOFT-5traj-libero_goal/final_model"
RT="results/rlt_training"
OUT_ROOT="${OUT_ROOT:-results/eval_rlt_a_50ep_0529}"
N_EPS="${N_EPS:-50}"; NUM_WORKERS="${NUM_WORKERS:-4}"; SEED="${SEED:-42}"
read -r -a GPUS <<< "${GPUS:-1 2 3 4 5}"
mkdir -p "${OUT_ROOT}"

# job: "label|task_id|ckpt_dir|prop_dim|residual_flag"
JOBS=(
  "a_td3_t0|0|${RT}/rlt_a_rl_qwen_t0_0527_1945/rl_offpolicy/checkpoints/rl_offpolicy_iter_00300|8|"
  "a_td3_t1|1|${RT}/rlt_a_rl_qwen_t1_0527_1946/rl_offpolicy/checkpoints/rl_offpolicy_iter_00300|8|"
  "a_td3_t3|3|${RT}/rlt_a_rl_qwen_t3_0527_1946/rl_offpolicy/checkpoints/rl_offpolicy_iter_00300|8|"
  "a_grpo_t0|0|${RT}/rlt_a_grpo_qwen_t0_0527_1846/rl_grpo/checkpoints/grpo_iter_00300|0|--residual"
  "a_grpo_t1|1|${RT}/rlt_a_grpo_qwen_t1_0527_1943/rl_grpo/checkpoints/grpo_iter_00300|0|--residual"
  "a_grpo_t3|3|${RT}/rlt_a_grpo_qwen_t3_0527_1944/rl_grpo/checkpoints/grpo_iter_00300|0|--residual"
  "a_ppo_t0|0|${RT}/rlt_a_ppo_qwen_t0_0527_1947/rl_onpolicy/checkpoints/rl_iter_00300|0|--residual"
  "a_ppo_t1|1|${RT}/rlt_a_ppo_qwen_t1_0527_2340/rl_onpolicy/checkpoints/rl_iter_00300|0|--residual"
)

echo "=== RLT_a single-task 50-ep re-eval | ${#JOBS[@]} jobs on GPUs [${GPUS[*]}] ==="
for j in "${JOBS[@]}"; do IFS='|' read -r l t c p r <<< "$j"; [ -d "$c" ] || { echo "ERROR missing $l: $c" >&2; exit 1; }; done

SHARD_PIDS=(); TAIL_PIDS=()
_cleanup () { trap - EXIT INT TERM
  for pid in "${SHARD_PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true; }
trap _cleanup EXIT INT TERM

i=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r label tid ckpt prop res <<< "$j"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  out="${OUT_ROOT}/${label}.json"; log="${OUT_ROOT}/${label}.log"; : > "${log}"; rm -f "${out}"
  CUDA_VISIBLE_DEVICES=${gpu} setsid python AlphaBrain/training/reinforcement_learning/eval/eval_libero.py \
      --vla_ckpt "${VLA_CKPT}" --action_token_ckpt "${ckpt}" \
      --suite libero_goal --task_ids "${tid}" \
      --n_eps_per_task ${N_EPS} --gpu 0 --num_workers ${NUM_WORKERS} --seed ${SEED} \
      --bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4 \
      --actor_hidden_dim 512 --ref_dropout 0.5 --fixed_std 0.1 \
      --prop_dim ${prop} ${res} \
      --results_json "${out}" > "${log}" 2>&1 &
  pid=$!; SHARD_PIDS+=(${pid})
  echo "[launch] ${label} (task ${tid}, prop=${prop} ${res:-no-res}) → GPU ${gpu} pid=${pid}"
  (tail -F -q -n +1 "${log}" 2>/dev/null | sed -u "s/^/[${label}] /") & TAIL_PIDS+=($!)
  i=$(( i + 1 )); sleep 3
done

echo "[all launched] waiting for ${#SHARD_PIDS[@]} jobs..."
fail=0
for pid in "${SHARD_PIDS[@]}"; do if ! wait "${pid}"; then echo "ERROR pid ${pid} FAILED" >&2; fail=1; fi; done
sleep 1; kill "${TAIL_PIDS[@]}" 2>/dev/null || true

python3 - <<PY
import json, os, glob
out_root="${OUT_ROOT}"; rows={}
for f in sorted(glob.glob(os.path.join(out_root,"*.json"))):
    label=os.path.splitext(os.path.basename(f))[0]
    try:
        d=json.load(open(f)); e=d[-1] if isinstance(d,list) else d
        pt=e.get("per_task_sr",{}); rows[label]=list(pt.values())[0] if pt else e.get("overall_sr")
    except Exception as ex: rows[label]=f"ERR:{ex}"
print(" RLT_a 50-ep results:"); [print(f"   {k:10s} {rows[k]}") for k in sorted(rows)]
json.dump(rows, open(os.path.join(out_root,"all_50ep_summary.json"),"w"), indent=2)
print("saved", os.path.join(out_root,"all_50ep_summary.json"))
PY
[ "${fail}" -eq 0 ] && echo "Done." || { echo "(some jobs failed)"; exit 1; }
