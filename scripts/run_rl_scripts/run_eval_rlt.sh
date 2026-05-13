#!/bin/bash
# Offline eval for rlt runs.
#
# Why this exists:
#   In-training eval_sr in metrics.json is computed via
#   eval_helpers._eval_deterministic_local, which hard-codes the
#   action-token encoder input (encoder.encode(action_queries)). For
#   --encoder_mode rlt that's the wrong input — rollout/training used
#   compacted image hidden states — so the logged eval_sr is not a valid
#   measurement of the trained policy. This script re-evals each ckpt
#   through the correct rlt path (eval_libero_rlt.py).
#
# Usage:
#   bash scripts/run_rl_scripts/run_eval_rlt.sh          # GPU 0
#   bash scripts/run_rl_scripts/run_eval_rlt.sh 1        # GPU 1
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

[ -f .env ] && { set -a; source .env; set +a; }
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPU_ID=${1:-0}

# ── Edit these before running ────────────────────────────────
VLA_CKPT="results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model"
ITER="iter_00300"
N_EPS=50
TASK_IDS="0"           # rlt release script trains only task 0
SUITE="libero_goal"
NUM_WORKERS=4

RUN_DIRS=(
    "results/rlt_training/rlt_rl_t0_release_0423_1216/rl_offpolicy"
    "results/rlt_training/rlt_rl_t0_G16_1_0423_1234/rl_offpolicy"
)

# Arch — matches run_rlt_rl_task0_release.sh
BOTTLENECK_DIM=2048
ENCODER_LAYERS=2
ENCODER_HEADS=8
ACTOR_HIDDEN_DIM=512
REF_DROPOUT=0.5
FIXED_STD=0.1
# ─────────────────────────────────────────────────────────────

if [ ! -d "${VLA_CKPT}" ]; then
    echo "ERROR: VLA ckpt not found: ${VLA_CKPT}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}

eval_one () {
    local run_dir=$1
    local ckpt="${run_dir}/checkpoints/rl_offpolicy_${ITER}"
    local out_dir="${run_dir}/eval_${ITER}_rlt"
    local results_json="${out_dir}/summary.json"
    local log="${out_dir}/eval.log"

    if [ ! -d "${ckpt}" ]; then
        echo "SKIP: ckpt not found: ${ckpt}" >&2
        return 0
    fi
    mkdir -p "${out_dir}"
    # Start clean so a failed re-run never aggregates with stale results.
    rm -f "${results_json}"

    echo "============================================================"
    echo " rlt offline eval"
    echo "   run_dir:  ${run_dir}"
    echo "   ckpt:     ${ckpt}"
    echo "   vla:      ${VLA_CKPT}"
    echo "   suite:    ${SUITE}   task_ids: ${TASK_IDS}   n_eps: ${N_EPS}"
    echo "   gpu:      ${GPU_ID}   workers: ${NUM_WORKERS}"
    echo "   out:      ${out_dir}"
    echo "============================================================"

    python AlphaBrain/training/reinforcement_learning/eval/eval_libero_rlt.py \
        --vla_ckpt "${VLA_CKPT}" \
        --action_token_ckpt "${ckpt}" \
        --suite "${SUITE}" \
        --n_eps_per_task ${N_EPS} \
        --gpu 0 \
        --task_ids "${TASK_IDS}" \
        --results_json "${results_json}" \
        --num_workers ${NUM_WORKERS} \
        --bottleneck_dim ${BOTTLENECK_DIM} \
        --encoder_layers ${ENCODER_LAYERS} \
        --encoder_heads ${ENCODER_HEADS} \
        --actor_hidden_dim ${ACTOR_HIDDEN_DIM} \
        --ref_dropout ${REF_DROPOUT} \
        --fixed_std ${FIXED_STD} \
        2>&1 | tee "${log}"
}

for run_dir in "${RUN_DIRS[@]}"; do
    eval_one "${run_dir}"
done

echo "============================================================"
echo " All evals finished. Summaries:"
for run_dir in "${RUN_DIRS[@]}"; do
    echo "   ${run_dir}/eval_${ITER}_rlt/summary.json"
done
echo "============================================================"
