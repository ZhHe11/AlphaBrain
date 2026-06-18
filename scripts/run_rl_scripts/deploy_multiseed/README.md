# Multi-seed (mean±std) queue — single box, no remote (2026-06-04)

The remaining report gap is **multi-seed mean±std** (T3, a publication requirement —
all current numbers are seed 42 only). This box is env-worker saturated
(~120/128 with Pi05 multitask + 2 dims), and there's no second server, so seed
runs must be **queued and run as capacity frees**, NOT stacked (stacking → env
starvation → soft-drops → corrupted data, proven 0604 with refdrop0).

`run_rlt_ppo.sh` / `run_rlt_grpo.sh` now accept `SEED=` and `RUN_NAME=` env
overrides (added 0604) so seed variants get distinct dirs.

## Worker budget (128-core box)

| run type | env workers |
|---|---|
| RLT multitask PPO/GRPO (10 tasks × 8 envs) | ~80 |
| RLT multitask TD3 (10 × 10) | ~100 |
| RLT single-task PPO/GRPO (8 envs) | ~8 |
| RLT single-task TD3 (64) | ~64 |
| RLT_a single-task (10) | ~10 |

Healthy ceiling ≈ 120 workers. **One multitask run at a time**; light single-task
seeds (~8w) can backfill the gaps.

## Queue (priority order — highest-value error bars first)

**Tier A — light single-task seeds (~8w, fit when a dim frees ~10w):**
The headline difficult-task claim is PPO t3 = 0.96 (5traj) — give it error bars.
```bash
# RLT+PPO task3, seeds 43 & 44  (5traj base)
SEED=43 TASK_ID=3 MULTI_TASK=0 RUN_NAME=rlt_ppo_qwen_t3_s43 \
  ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt \
  bash scripts/run_rl_scripts/run_rlt_ppo.sh <GPU>
SEED=44 TASK_ID=3 MULTI_TASK=0 RUN_NAME=rlt_ppo_qwen_t3_s44 \
  ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt \
  bash scripts/run_rl_scripts/run_rlt_ppo.sh <GPU>
# RLT+GRPO task3 seeds 43/44 (same pattern, run_rlt_grpo.sh)
```

**Tier B — heavy multitask seeds (~80w, only after Pi05 frees its 100w):**
The two headline overall numbers (RLT+PPO 0.916, RLT_a+PPO 0.936) need error bars.
```bash
SEED=43 MULTI_TASK=1 RUN_NAME=rlt_ppo_qwen_alltasks_s43 \
  ENCODER_PATH=results/rlt_training/5traj_libero_goal_0425_1322/pretrain/checkpoints/pretrain_best/encoder.pt \
  bash scripts/run_rl_scripts/run_rlt_ppo.sh <GPU>
# + seed 44; + RLT_a+PPO multitask seeds (action-token launcher)
```

**Also queued (table 3, env-heavy):** refdrop0 (RLT full-token single-task TD3,
64w) — run after Pi05 finishes.

## Auto-advance

Driven by the session's scheduled health checks: at each wake-up, if a run
finished (`Done. Metrics`) and freed workers, launch the next queued item that
fits the freed budget. After each seed run completes → 50-ep eval, then report
mean±std once ≥3 seeds exist for a cell.
