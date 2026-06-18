#!/bin/bash
# T1 base anchor: evaluate a frozen base VLA (no RL actor) under the RL eval
# protocol. 10 libero_goal tasks split across the given GPUs, then aggregated
# into <OUT_DIR>/summary.json. Uses eval_libero.py --base_vla.
#
# Usage:
#   VLA_CKPT=results/training/QwenOFT-5traj-libero_goal/final_model \
#   OUT_DIR=results/eval_base_vla/qwen_5traj \
#   GPU_IDS="1,2,3" bash scripts/run_rl_scripts/run_eval_base_vla.sh
#
# Env overrides:
#   VLA_CKPT     base VLA checkpoint dir (required)
#   OUT_DIR      output dir for shard_*.json + summary.json (required)
#   GPU_IDS      comma-separated GPU ids (default "0,1,2")
#   SUITE        default libero_goal
#   N_EPS        episodes/task (default 50)
#   TASK_IDS     comma-separated task ids (default 0..9, the full suite)
#   NUM_WORKERS  parallel LIBERO env workers per shard (default 4)
#   SEED         eval seed (default 42, matches RL offline eval)
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

VLA_CKPT="${VLA_CKPT:-}"
OUT_DIR="${OUT_DIR:-}"
[ -z "${VLA_CKPT}" ] && { echo "ERROR: VLA_CKPT is required" >&2; exit 1; }
[ -z "${OUT_DIR}" ]  && { echo "ERROR: OUT_DIR is required" >&2; exit 1; }
[ ! -d "${VLA_CKPT}" ] && { echo "ERROR: VLA_CKPT not found: ${VLA_CKPT}" >&2; exit 1; }

GPU_IDS="${GPU_IDS:-0,1,2}"
SUITE="${SUITE:-libero_goal}"
N_EPS="${N_EPS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
read -r -a GPUS <<< "$(echo "${GPU_IDS}" | tr ',' ' ')"

# Default task set: all 10 libero_goal tasks, round-robin across GPUs.
if [ -n "${TASK_IDS:-}" ]; then
    read -r -a ALL_TASKS <<< "$(echo "${TASK_IDS}" | tr ',' ' ')"
else
    ALL_TASKS=(0 1 2 3 4 5 6 7 8 9)
fi

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/shard_*.json "${OUT_DIR}"/summary.json

# Round-robin assign tasks to GPUs → per-GPU comma list.
declare -a SHARD_TASKS
for i in "${!ALL_TASKS[@]}"; do
    g=$(( i % ${#GPUS[@]} ))
    SHARD_TASKS[$g]="${SHARD_TASKS[$g]:+${SHARD_TASKS[$g]},}${ALL_TASKS[$i]}"
done

echo "============================================================"
echo " Base-VLA eval (T1) | ${N_EPS} eps/task, seed ${SEED}"
echo "   vla:   ${VLA_CKPT}"
echo "   out:   ${OUT_DIR}"
echo "   suite: ${SUITE}"
for g in "${!GPUS[@]}"; do
    echo "   GPU ${GPUS[$g]}: tasks [${SHARD_TASKS[$g]:-}]"
done
echo "============================================================"

SHARD_PIDS=()
TAIL_PIDS=()
_cleanup () {
    trap - EXIT INT TERM
    for pid in "${SHARD_PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
    [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

for g in "${!GPUS[@]}"; do
    tasks="${SHARD_TASKS[$g]:-}"
    [ -z "${tasks}" ] && continue
    gpu="${GPUS[$g]}"
    out="${OUT_DIR}/shard_${g}.json"
    log="${OUT_DIR}/shard_${g}.log"
    : > "${log}"
    CUDA_VISIBLE_DEVICES=${gpu} setsid python AlphaBrain/training/reinforcement_learning/eval/eval_libero.py \
        --vla_ckpt "${VLA_CKPT}" \
        --base_vla \
        --suite "${SUITE}" \
        --n_eps_per_task ${N_EPS} \
        --gpu 0 \
        --task_ids "${tasks}" \
        --results_json "${out}" \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED} \
        > "${log}" 2>&1 &
    pid=$!
    SHARD_PIDS+=(${pid})
    echo "[launch] GPU ${gpu} tasks [${tasks}] pid=${pid} log=${log}"
    (tail -F -q -n +1 "${log}" 2>/dev/null | sed -u "s/^/[g${g}] /") &
    TAIL_PIDS+=($!)
    sleep 2
done

fail=0
for pid in "${SHARD_PIDS[@]}"; do
    if ! wait "${pid}"; then echo "ERROR: shard pid ${pid} FAILED — see ${OUT_DIR}/shard_*.log" >&2; fail=1; fi
done
sleep 1
kill "${TAIL_PIDS[@]}" 2>/dev/null || true
[ "${fail}" -eq 0 ] || { echo "(some shards failed)" >&2; exit 1; }

echo "============================================================"
echo " Aggregating per-task SR → ${OUT_DIR}/summary.json"
python - <<PY
import json, glob, os
out_dir = "${OUT_DIR}"
per_task = {}
for sf in sorted(glob.glob(os.path.join(out_dir, "shard_*.json"))):
    with open(sf) as f:
        data = json.load(f)
    entry = data[-1] if isinstance(data, list) else data
    for k, v in entry.get("per_task_sr", {}).items():
        per_task[int(k)] = float(v)
overall = sum(per_task.values()) / len(per_task) if per_task else 0.0
summary = {
    "vla_ckpt": "${VLA_CKPT}",
    "base_vla": True,
    "suite": "${SUITE}",
    "n_eps_per_task": ${N_EPS},
    "seed": ${SEED},
    "per_task_sr": {str(k): per_task[k] for k in sorted(per_task)},
    "overall_sr": overall,
}
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
echo "Done."
