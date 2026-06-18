#!/bin/bash
# P0: uniform offline 50-ep re-eval of the 4 multi-task (all-10-task) RL runs
# + the RLT_a+PPO task3 single-task run, to replace the §-marked in-train
# 200-ep numbers in RL_REPORT_TABLES.md table 1 / table 2 with protocol-matched
# 50-ep numbers (same seed 42 / max_steps 320 as the T1 base anchors and the
# RLT_a+TD3 release 0.92).
#
# Run identities verified from actor.pt first-layer dims + train.log final eval:
#   RLT    (full-token):  net.0.weight=(512, 2104) = 2048+0+56  -> rlt eval,  bn2048 heads8 maxlen4096
#   RLT_a  (action-token):net.0.weight=(512, 312)  = 256 +0+56  -> libero eval, bn256  heads4
#   All four GRPO/PPO actors are residual + prop_dim=0.
#
# Report-cited timestamps were typos; the on-disk dirs with a clean iter_00300 are:
#   RLT+GRPO   alltasks -> rlt_grpo_qwen_alltasks_0529_2029  (report said 2029  OK)
#   RLT+PPO    alltasks -> rlt_ppo_qwen_alltasks_0529_1830   (report said 1836)
#   RLT_a+GRPO alltasks -> rlt_a_grpo_qwen_alltasks_0529_1551 (report said 1526 = empty)
#   RLT_a+PPO  alltasks -> rlt_a_ppo_qwen_alltasks_0529_1829  (report said 1836)
#   RLT_a+PPO  task3     -> rlt_a_ppo_qwen_t3_0529_1109       (report said 1109  OK)
#
# Usage: GPUS="1 2 3 4 5" bash scripts/run_rl_scripts/run_eval_p0_multitask_50ep.sh
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
OUT_ROOT="${OUT_ROOT:-results/eval_p0_50ep_0529}"
N_EPS="${N_EPS:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
ALL_TASKS="${ALL_TASKS:-0,1,2,3,4,5,6,7,8,9}"
read -r -a GPUS <<< "${GPUS:-1 2 3 4 5}"
mkdir -p "${OUT_ROOT}"

RLT_GRPO="${RT}/rlt_grpo_qwen_alltasks_0529_2029/rl_grpo/checkpoints/grpo_iter_00300"
RLT_PPO="${RT}/rlt_ppo_qwen_alltasks_0529_1830/rl_ppo/checkpoints/rl_iter_00300"
RLTA_GRPO="${RT}/rlt_a_grpo_qwen_alltasks_0529_1551/rl_grpo/checkpoints/grpo_iter_00300"
RLTA_PPO="${RT}/rlt_a_ppo_qwen_alltasks_0529_1829/rl_onpolicy/checkpoints/rl_iter_00300"
RLTA_PPO_T3="${RT}/rlt_a_ppo_qwen_t3_0529_1109/rl_onpolicy/checkpoints/rl_iter_00300"

# job: "label|track|task_ids|ckpt"   (all use prop_dim=0 --residual)
#   track=rlt   -> eval_libero_rlt.py, --bottleneck_dim 2048 --encoder_heads 8 --max_len 4096
#   track=rlt_a -> eval_libero.py,     --bottleneck_dim 256  --encoder_heads 4
JOBS=(
  "mt_rlt_grpo|rlt|${ALL_TASKS}|${RLT_GRPO}"
  "mt_rlt_ppo|rlt|${ALL_TASKS}|${RLT_PPO}"
  "mt_rlta_grpo|rlt_a|${ALL_TASKS}|${RLTA_GRPO}"
  "mt_rlta_ppo|rlt_a|${ALL_TASKS}|${RLTA_PPO}"
  "rlta_ppo_t3|rlt_a|3|${RLTA_PPO_T3}"
)

echo "============================================================"
echo " P0 multitask 50-ep re-eval | ${#JOBS[@]} jobs on GPUs [${GPUS[*]}]"
echo "   n_eps=${N_EPS}  workers=${NUM_WORKERS}  seed=${SEED}  out=${OUT_ROOT}"
echo "============================================================"

# Pre-flight: every ckpt dir must exist.
for j in "${JOBS[@]}"; do
  IFS='|' read -r label track tids ckpt <<< "$j"
  [ -d "$ckpt" ] || { echo "ERROR: missing ckpt for ${label}: ${ckpt}" >&2; exit 1; }
done

SHARD_PIDS=(); TAIL_PIDS=()
_cleanup () { trap - EXIT INT TERM
  for pid in "${SHARD_PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true; }
trap _cleanup EXIT INT TERM

i=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r label track tids ckpt <<< "$j"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  out="${OUT_ROOT}/${label}.json"; log="${OUT_ROOT}/${label}.log"; : > "${log}"; rm -f "${out}"
  if [ "${track}" = "rlt" ]; then
    SCRIPT="AlphaBrain/training/reinforcement_learning/eval/eval_libero_rlt.py"
    ARCH="--bottleneck_dim 2048 --encoder_layers 2 --encoder_heads 8 --max_len 4096"
  else
    SCRIPT="AlphaBrain/training/reinforcement_learning/eval/eval_libero.py"
    ARCH="--bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4"
  fi
  CUDA_VISIBLE_DEVICES=${gpu} setsid python ${SCRIPT} \
      --vla_ckpt "${VLA_CKPT}" \
      --action_token_ckpt "${ckpt}" \
      --suite libero_goal --task_ids "${tids}" \
      --n_eps_per_task ${N_EPS} --gpu 0 --num_workers ${NUM_WORKERS} --seed ${SEED} \
      ${ARCH} \
      --actor_hidden_dim 512 --ref_dropout 0.5 --fixed_std 0.1 \
      --prop_dim 0 --residual \
      --results_json "${out}" \
      > "${log}" 2>&1 &
  pid=$!; SHARD_PIDS+=(${pid})
  echo "[launch] ${label} (${track}, tasks ${tids}) -> GPU ${gpu} pid=${pid}"
  (tail -F -q -n +1 "${log}" 2>/dev/null | sed -u "s/^/[${label}] /") & TAIL_PIDS+=($!)
  i=$(( i + 1 )); sleep 3
done

echo "[all launched] waiting for ${#SHARD_PIDS[@]} jobs..."
fail=0
for pid in "${SHARD_PIDS[@]}"; do
  if ! wait "${pid}"; then echo "ERROR: pid ${pid} FAILED — check ${OUT_ROOT}/*.log" >&2; fail=1; fi
done
sleep 1; kill "${TAIL_PIDS[@]}" 2>/dev/null || true

echo "============================================================"
python3 - <<PY
import json, os, glob
out_root="${OUT_ROOT}"
rows={}
for f in sorted(glob.glob(os.path.join(out_root,"*.json"))):
    label=os.path.splitext(os.path.basename(f))[0]
    try:
        d=json.load(open(f)); e=d[-1] if isinstance(d,list) else d
        rows[label]={"overall_sr": e.get("overall_sr"), "per_task_sr": e.get("per_task_sr")}
    except Exception as ex:
        rows[label]={"error": str(ex)}
print(" P0 50-ep results (label -> overall | per-task):")
for k in sorted(rows):
    r=rows[k]
    if "error" in r: print(f"   {k:14s} ERR {r['error']}"); continue
    print(f"   {k:14s} overall={r['overall_sr']}")
    print(f"   {'':14s} per_task={r['per_task_sr']}")
json.dump(rows, open(os.path.join(out_root,"all_p0_50ep_summary.json"),"w"), indent=2)
print("saved", os.path.join(out_root,"all_p0_50ep_summary.json"))
PY
[ "${fail}" -eq 0 ] && echo "Done." || { echo "(some jobs failed)"; exit 1; }
