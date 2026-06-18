#!/bin/bash
# ============================================================================
# box2_agent.sh — /share-based job agent. RUN ONCE ON box2 (192.168.33.5).
#
# Why: the box1 pod cannot route to box2 (firewalled), but BOTH mount /share.
# This agent polls a shared job dir; box1 drops *.job files there and this
# agent runs them on box2 — an auto-deploy channel that needs no network.
#
# Start (on box2, once):
#   cd /share/zhanghe/AlphaBrain-zh
#   nohup bash scripts/run_rl_scripts/deploy_multiseed/box2_agent.sh \
#         > logs/box2_agent.log 2>&1 &
#
# Job protocol (box1 side just writes files here):
#   QUEUE dir: /share/zhanghe/AlphaBrain-zh/.box2_jobs/
#   <name>.job          a bash script to run. HEAVY (multitask training):
#                       launched only when active multitask runs < MAX, staggered.
#   <name>.light.job    LIGHT (eval etc.): launched immediately, ignores MAX.
#   lifecycle: <name>.job -> <name>.running -> <name>.done  (+ <name>.outlog)
#
# Safety: never exceeds MAX concurrent multitask trainings (default 2) — the
# over-pack guard that prevents the OOM kills seen before.  Stop with:
#   touch /share/zhanghe/AlphaBrain-zh/.box2_jobs/STOP
# ============================================================================
set -u
ROOT=/share/zhanghe/AlphaBrain-zh
cd "$ROOT"
source /share/zhanghe/miniconda3/etc/profile.d/conda.sh && conda activate vla
[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export ALPHABRAIN_ROOT=$ROOT TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_WORKER_TIMEOUT=300

Q="$ROOT/.box2_jobs"
mkdir -p "$Q" logs
MAX=${MAX:-2}

mt_count() { ps -eo cmd | grep "train.py --phase" | grep -v grep \
  | grep -cE "alltasks|all_tasks|_mt_|pi05_1traj_mt"; }
host_ip() { hostname -I 2>/dev/null | awk '{print $1}'; }

echo "[$(date +%H:%M)] box2_agent start on $(hostname) ($(host_ip)); queue=$Q MAX=$MAX"
# Refuse to run on the box1 pod by accident (it can't be the executor).
if [ "$(host_ip)" = "192.168.80.229" ]; then
  echo "ERROR: this is the box1 pod — the agent must run ON box2. Aborting." >&2; exit 1
fi

while true; do
  [ -f "$Q/STOP" ] && { echo "[$(date +%H:%M)] STOP file present — exiting."; rm -f "$Q/STOP"; break; }

  # 1) LIGHT jobs: run immediately, no capacity gate.
  for j in "$Q"/*.light.job; do
    [ -e "$j" ] || continue
    base="${j%.job}"; mv "$j" "$base.running" 2>/dev/null || continue
    echo "[$(date +%H:%M)] LIGHT launch: $(basename "$base")"
    ( bash "$base.running" > "$base.outlog" 2>&1; mv "$base.running" "$base.done" 2>/dev/null
      echo "[$(date +%H:%M)] LIGHT done: $(basename "$base")" ) &
    sleep 5
  done

  # 2) HEAVY jobs: one per loop, only when below MAX; stagger to let it register.
  if [ "$(mt_count)" -lt "$MAX" ]; then
    for j in "$Q"/*.job; do
      [ -e "$j" ] || continue
      case "$j" in *.light.job) continue;; esac
      base="${j%.job}"; mv "$j" "$base.running" 2>/dev/null || continue
      echo "[$(date +%H:%M)] HEAVY launch: $(basename "$base") (mt_count=$(mt_count))"
      ( bash "$base.running" > "$base.outlog" 2>&1; mv "$base.running" "$base.done" 2>/dev/null
        echo "[$(date +%H:%M)] HEAVY done: $(basename "$base")" ) &
      sleep 120   # let it register before the next capacity check
      break
    done
  fi

  sleep 60
done
