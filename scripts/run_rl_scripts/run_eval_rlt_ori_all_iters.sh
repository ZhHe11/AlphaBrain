#!/bin/bash
# Offline rlt_ori eval across ALL ckpts in one run_dir.
# Parallelizes across the GPU list, one ckpt per shard.
#
# Usage (defaults work for Qwen-1traj-task0; override via env vars for
# other runs):
#   bash scripts/run_rl_scripts/run_eval_rlt_ori_all_iters.sh
#
#   RUN_DIR=results/rlt_ori_training/rlt_ori_rl_t0_release_pi05_5traj_0429_0924/rl_offpolicy \
#   VLA_CKPT=results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000 \
#   GPUS="0 1 2" TASK_IDS=0 \
#   bash scripts/run_rl_scripts/run_eval_rlt_ori_all_iters.sh
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ── Defaults (override via env) ──────────────────────────────
RUN_DIR="${RUN_DIR:-results/rlt_ori_training/rlt_ori_rl_t0_release_0423_1216/rl_offpolicy}"
VLA_CKPT="${VLA_CKPT:-results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model}"
read -r -a GPUS <<< "${GPUS:-5 6}"
N_EPS="${N_EPS:-50}"
TASK_IDS="${TASK_IDS:-0}"
SUITE="${SUITE:-libero_goal}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Arch — matches run_rlt_ori_rl_task0_release.sh
BOTTLENECK_DIM="${BOTTLENECK_DIM:-2048}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
ENCODER_HEADS="${ENCODER_HEADS:-8}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
REF_DROPOUT="${REF_DROPOUT:-0.5}"
FIXED_STD="${FIXED_STD:-0.1}"
# ─────────────────────────────────────────────────────────────

if [ ! -d "${VLA_CKPT}" ]; then
    echo "ERROR: VLA ckpt not found: ${VLA_CKPT}" >&2
    exit 1
fi
if [ ! -d "${RUN_DIR}/checkpoints" ]; then
    echo "ERROR: no checkpoints dir at ${RUN_DIR}/checkpoints" >&2
    exit 1
fi

# Collect all iter ckpts, sorted.
mapfile -t CKPTS < <(ls -1 "${RUN_DIR}/checkpoints" | grep -E '^rl_offpolicy_iter_[0-9]+$' | sort)
if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "ERROR: no rl_offpolicy_iter_* dirs under ${RUN_DIR}/checkpoints" >&2
    exit 1
fi

AGG_DIR="${RUN_DIR}/eval_all_iters_rlt_ori"
mkdir -p "${AGG_DIR}"

echo "============================================================"
echo " rlt_ori eval ALL iters"
echo "   run_dir:  ${RUN_DIR}"
echo "   ckpts:    ${#CKPTS[@]}  (${CKPTS[*]})"
echo "   GPUs:     ${GPUS[*]}"
echo "   suite:    ${SUITE}   task_ids: ${TASK_IDS}   n_eps: ${N_EPS}"
echo "   agg_dir:  ${AGG_DIR}"
echo "============================================================"

SHARD_PIDS=()
TAIL_PIDS=()
_cleanup () {
    trap - EXIT INT TERM
    for pid in "${SHARD_PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true
    pkill -TERM -u "$(id -u)" -f "AlphaBrain/training/reinforcement_learning/envs/libero_env_worker" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

launch_one () {
    local iter_name=$1        # e.g. rl_offpolicy_iter_00025
    local gpu=$2
    local ckpt="${RUN_DIR}/checkpoints/${iter_name}"
    local iter_tag="${iter_name#rl_offpolicy_}"      # iter_00025
    local out_dir="${RUN_DIR}/eval_${iter_tag}_rlt_ori"
    local log="${AGG_DIR}/${iter_tag}.log"

    mkdir -p "${out_dir}"
    rm -f "${out_dir}/summary.json"

    CUDA_VISIBLE_DEVICES=${gpu} setsid python AlphaBrain/training/reinforcement_learning/eval/eval_libero_rlt_ori.py \
        --vla_ckpt "${VLA_CKPT}" \
        --action_token_ckpt "${ckpt}" \
        --suite "${SUITE}" \
        --n_eps_per_task ${N_EPS} \
        --gpu 0 \
        --task_ids "${TASK_IDS}" \
        --results_json "${out_dir}/summary.json" \
        --num_workers ${NUM_WORKERS} \
        --bottleneck_dim ${BOTTLENECK_DIM} \
        --encoder_layers ${ENCODER_LAYERS} \
        --encoder_heads ${ENCODER_HEADS} \
        --actor_hidden_dim ${ACTOR_HIDDEN_DIM} \
        --ref_dropout ${REF_DROPOUT} \
        --fixed_std ${FIXED_STD} \
        > "${log}" 2>&1 &
    local pid=$!
    SHARD_PIDS+=(${pid})
    echo "[launch] ${iter_tag} → GPU ${gpu}  pid=${pid}  log=${log}"
    # Mirror log to main stdout with a tag prefix.
    (tail -F -q -n +1 "${log}" 2>/dev/null | sed -u "s/^/[${iter_tag}] /") &
    TAIL_PIDS+=($!)
}

# Round-robin ckpts across GPU list.
i=0
for ckpt_name in "${CKPTS[@]}"; do
    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
    launch_one "${ckpt_name}" "${gpu}"
    i=$(( i + 1 ))
    sleep 2    # stagger start so log headers don't interleave unreadably
done

echo "[all launched] waiting for ${#SHARD_PIDS[@]} shards..."

fail=0
for pid in "${SHARD_PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "ERROR: shard pid ${pid} FAILED — check ${AGG_DIR}/*.log" >&2
        fail=1
    fi
done
sleep 1
kill "${TAIL_PIDS[@]}" 2>/dev/null || true

echo "============================================================"
echo " Aggregating per-iter SR:"
python - <<PY
import json, os, re, glob
run_dir = "${RUN_DIR}"
agg_dir = "${AGG_DIR}"
rows = []
for d in sorted(glob.glob(os.path.join(run_dir, "eval_iter_*_rlt_ori"))):
    m = re.search(r"eval_(iter_\d+)_rlt_ori", d)
    if not m: continue
    it = m.group(1)
    summary = os.path.join(d, "summary.json")
    if not os.path.isfile(summary):
        rows.append((it, None, "NO SUMMARY"))
        continue
    with open(summary) as f:
        data = json.load(f)
    entry = data[-1] if isinstance(data, list) else data
    rows.append((it, entry.get("overall_sr"), entry.get("per_task_sr")))
print(f"{'iter':>12}  {'overall_sr':>10}  per_task_sr")
for it, sr, ptsr in rows:
    sr_s = f"{sr:>10.3f}" if isinstance(sr, float) else f"{str(sr):>10}"
    print(f"{it:>12}  {sr_s}  {ptsr}")

with open(os.path.join(agg_dir, "all_iters_summary.json"), "w") as f:
    json.dump([{"iter": it, "overall_sr": sr, "per_task_sr": ptsr} for it, sr, ptsr in rows], f, indent=2)
print(f"\nsaved: {os.path.join(agg_dir, 'all_iters_summary.json')}")
PY

[ "${fail}" -eq 0 ] || { echo "(one or more shards failed; check logs)" >&2; exit 1; }
echo "Done."
