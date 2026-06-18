# Pi0.5 backbone deployment + remote re-runs (2026-06-03)

Cross-backbone story: prove the RLT pipeline + the algo findings hold on a 2nd
VLM (Pi0.5 / PaliGemmaPi05), not just QwenOFT. The one cell that makes Pi0.5 a
*complete* backbone row is the **5traj multi-task (全10任务)** number — currently
n/a in table 4 (Pi0.5 only has single-task t0/t1/t3 release runs).

## Key experiment — Pi05-5traj multi-task TD3

```bash
bash 01_pi05_5traj_multitask_td3.sh <GPU>      # ~300 iter, fills table-4 全10任务
GPUS="2 3 5 7" bash 02_eval_pi05_5traj_multitask_50ep.sh   # after 01 finishes
```

- TD3 only: `run_rlt_rl.sh` wires `BACKBONE=pi05` for `TRACK=rlt`. RLT_a × Pi05
  and PPO/GRPO × Pi05 are NOT wired (would need launcher work).
- `BUFFER_CAPACITY=300000` — Pi0.5 full-token bottleneck=2048 has the memory
  profile that OOM'd RLT+TD3 multitask at 1M; buf300k is the proven-safe config.
- Compare against QwenOFT-5traj RLT+TD3 multitask = **0.83** (table 1).

## Remote re-runs (table 3 ablations lost to the 0603 server crash)

```bash
bash 00_rerun_remote_ablations.sh    # dim128 / dim512 / refdrop0, GPUs 0,1,2
```

All 3 died with ckpt=NONE (before first save) → zero data, need full re-run.
Each uses a unique RUN_NAME and the 0603 per-env soft-fail.

## Notes

- All scripts assume the shared `/share/zhanghe/AlphaBrain-zh` checkout.
- New runs pick up the 0603 robustness fixes (env-pool retry/backoff in
  `persistent_env_pool.py` + rollout per-env soft-fail in
  `action_token_rollout_fast.py`), so a single env-worker timeout no longer
  kills the whole run — important when sharing a box.
- **Concurrency discipline**: a 5traj multitask run uses ~100 libero env
  workers (≈ nproc). Do NOT stack multitask runs + many evals on one 128-core
  box — that starves env workers and times everything out (0603 lesson). Run
  evals with `CONCURRENCY≤2` and don't co-locate two multitask trainings.

## Phase 2 (not scripted yet)

- Pi05-1traj 50-ep eval: 12 `rlt_ori_rl_t*_release_pi05_1traj_*` runs have
  ckpts but NO_SUMMARY; needs triage to pick canonical t0/t1/t3 runs, then
  eval with `VLA_CKPT=results/training/Pi05-goal-1traj-openpi/checkpoints/steps_30000`.
