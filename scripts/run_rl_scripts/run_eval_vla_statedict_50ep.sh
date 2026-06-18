#!/bin/bash
# Offline 50-ep re-eval of a VLA-RL-finetuned checkpoint saved as a RAW
# state_dict (vla_state_dict.pt) rather than an HF dir. The full-VLA trainers'
# save_pretrained() fails on a transformers version mismatch
# (`_get_non_default_generation_parameters`) and falls back to torch.save of a
# bare state_dict, leaving the `vla/` subdir empty. eval_libero.py supports
# this via: load base VLA (HF skeleton, for framework_config.yaml) then overlay
# the trained weights with --vla_state_dict.
#
# Usage:
#   STATE_DICT=.../vla_xxx_iter_00300_final/vla_state_dict.pt \
#   OUT_DIR=results/eval_vla_mt_50ep_0601/ppo \
#   GPUS="1 2" bash scripts/run_rl_scripts/run_eval_vla_statedict_50ep.sh
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

BASE_VLA="${BASE_VLA:-results/training/QwenOFT-5traj-libero_goal/final_model}"
STATE_DICT="${STATE_DICT:?Set STATE_DICT to the trained vla_state_dict.pt}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR}"
SUITE="${SUITE:-libero_goal}"
N_EPS="${N_EPS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
read -r -a GPUS <<< "${GPUS:-1 2}"
[ -f "${STATE_DICT}" ] || { echo "ERROR: state_dict not found: ${STATE_DICT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"; rm -f "${OUT_DIR}"/shard_*.json "${OUT_DIR}"/summary.json

if [ -n "${TASK_IDS:-}" ]; then
    read -r -a ALL_TASKS <<< "$(echo "${TASK_IDS}" | tr ',' ' ')"
else
    ALL_TASKS=(0 1 2 3 4 5 6 7 8 9)
fi
declare -a SHARD_TASKS
for i in "${!ALL_TASKS[@]}"; do
    g=$(( i % ${#GPUS[@]} ))
    SHARD_TASKS[$g]="${SHARD_TASKS[$g]:+${SHARD_TASKS[$g]},}${ALL_TASKS[$i]}"
done

echo "=== VLA state_dict 50-ep eval | base=${BASE_VLA} | sd=${STATE_DICT} ==="
SHARD_PIDS=(); TAIL_PIDS=()
_cleanup () { trap - EXIT INT TERM
  for pid in "${SHARD_PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true; }
trap _cleanup EXIT INT TERM

for g in "${!GPUS[@]}"; do
    tasks="${SHARD_TASKS[$g]:-}"; [ -z "${tasks}" ] && continue
    gpu="${GPUS[$g]}"; out="${OUT_DIR}/shard_${g}.json"; log="${OUT_DIR}/shard_${g}.log"; : > "${log}"
    CUDA_VISIBLE_DEVICES=${gpu} setsid python AlphaBrain/training/reinforcement_learning/eval/eval_libero.py \
        --vla_ckpt "${BASE_VLA}" --base_vla --vla_state_dict "${STATE_DICT}" \
        --suite "${SUITE}" --n_eps_per_task ${N_EPS} --gpu 0 \
        --task_ids "${tasks}" --results_json "${out}" \
        --num_workers ${NUM_WORKERS} --seed ${SEED} > "${log}" 2>&1 &
    pid=$!; SHARD_PIDS+=(${pid})
    echo "[launch] GPU ${gpu} tasks [${tasks}] pid=${pid}"
    (tail -F -q -n +1 "${log}" 2>/dev/null | sed -u "s/^/[g${g}] /") & TAIL_PIDS+=($!)
    sleep 2
done

fail=0
for pid in "${SHARD_PIDS[@]}"; do
    if ! wait "${pid}"; then echo "ERROR: shard pid ${pid} FAILED" >&2; fail=1; fi
done
sleep 1; kill "${TAIL_PIDS[@]}" 2>/dev/null || true
[ "${fail}" -eq 0 ] || { echo "(some shards failed)" >&2; exit 1; }

python - <<PY
import json, glob, os
out_dir="${OUT_DIR}"; per_task={}
for sf in sorted(glob.glob(os.path.join(out_dir,"shard_*.json"))):
    d=json.load(open(sf)); e=d[-1] if isinstance(d,list) else d
    for k,v in e.get("per_task_sr",{}).items(): per_task[int(k)]=float(v)
overall=sum(per_task.values())/len(per_task) if per_task else 0.0
summary={"vla_state_dict":"${STATE_DICT}","suite":"${SUITE}","n_eps_per_task":${N_EPS},
         "seed":${SEED},"per_task_sr":{str(k):per_task[k] for k in sorted(per_task)},"overall_sr":overall}
json.dump(summary, open(os.path.join(out_dir,"summary.json"),"w"), indent=2)
print(json.dumps(summary, indent=2))
PY
echo "Done."
