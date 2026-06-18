"""Fast VLA-PPO rollout — step-lock + ROBUST env handling.

Step-lock structure (RLinf-style):
  1. ONE batched VLA forward over ALL active envs → action_mean + features
  2. sample (fixed-std Gaussian) + value head        → action, log_prob, value
  3. ALL envs execute the action chunk in parallel threads (step_chunk)
  4. repeat until every env terminates

Per-step it records (images, instruction, prop) into ``VLAPPOStepRecord`` so
``vla_ppo_loss`` can re-forward the VLA unchanged.

ROBUSTNESS: every env IPC point (reset / warmup / per-chunk step) is bounded
by ``concurrent.futures.wait(timeout=...)``. A hung env subprocess is
``SIGKILL``ed so its socket closes and the worker thread unblocks; that env
is marked dead for the remainder of the rollout and skipped. A wall-clock
budget further caps total rollout time. The contract is: each rank's
``vla_ppo_collect_steplock`` ALWAYS returns within bounded time, so the
downstream NCCL collectives never wait > the env timeouts → no NCCL watchdog
SIGABRT from a single rank's hung subprocess.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

from AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool import PersistentEnvPool
from AlphaBrain.training.reinforcement_learning.common.rollout import (
    _unnormalize, _postprocess_action, DUMMY_ACTION,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_ppo_rollout import (
    VLAPPOStepRecord, VLAPPOEpisode,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_policy import (
    VLAPolicy, VLAValueHead,
)

logger = logging.getLogger(__name__)

_ZERO_OBS = {
    "primary_image": np.zeros((256, 256, 3), dtype=np.uint8),
    "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
    "state": np.zeros(8, dtype=np.float32),
}


# ── env IPC helpers (all bounded-failure: never raise) ────────────────


def _kill_env_subprocess(env_pool, env_idx: int) -> None:
    """SIGKILL the env's worker subprocess. After this the env's socket
    closes and any hung thread blocked on it unblocks. Best-effort, swallows
    every error — caller has already marked the env dead."""
    try:
        env = env_pool.envs[env_idx]
        proc = getattr(env, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
    except Exception:
        pass


def _safe_reset_env(env_pool, env_idx, suite_name, task_id, state_idx, seed):
    """Reset that returns None on any failure (instead of raising).

    If the env subprocess was previously killed (poll() != None), revives it
    via the pool worker's _restart_worker() before issuing the reset.
    """
    try:
        env = env_pool.envs[env_idx]
        # If the subprocess died (e.g. we SIGKILLed it last iter), bring up a
        # fresh worker before sending the reset message — otherwise sendall
        # would BrokenPipeError on the dead socket.
        proc = getattr(env, "_proc", None)
        if proc is None or proc.poll() is not None:
            try:
                env._restart_worker()
            except Exception as e:
                print(f"  [WARN] env {env_idx} restart failed: {e}", flush=True)
                return None
        return env_pool.reset_env(env_idx, suite_name, task_id, state_idx, seed)
    except Exception as e:
        print(f"  [WARN] env {env_idx} reset failed: {e}", flush=True)
        _kill_env_subprocess(env_pool, env_idx)
        return None


def _env_step_chunk(env_pool, env_idx, action_chunk_unnorm, chunk_len):
    """Execute chunk_len env steps in ONE round-trip. Any failure → (zero_obs, 0, done=True, 0)."""
    actions = [_postprocess_action(action_chunk_unnorm[s]) for s in range(chunk_len)]
    try:
        return env_pool.envs[env_idx].step_chunk(actions)
    except Exception as e:
        print(f"  [WARN] env {env_idx} step_chunk failed: {e}; marking done", flush=True)
        return dict(_ZERO_OBS), 0.0, True, 0


def _env_dummy_steps(env_pool, env_idx, n_steps):
    """Warmup dummy steps. Any failure → zero-obs placeholder."""
    obs = None
    for _ in range(n_steps):
        try:
            obs, _, _ = env_pool.step_env(env_idx, DUMMY_ACTION)
        except Exception as e:
            print(f"  [WARN] env {env_idx} dummy step failed: {e}", flush=True)
            return dict(_ZERO_OBS)
    return obs


@torch.no_grad()
def vla_ppo_collect_steplock(
    *,
    env_pool: PersistentEnvPool,
    policy: VLAPolicy,
    value_head: VLAValueHead,
    suite_name: str,
    task_id: int,
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G: int = 64,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    group_idx: int = 0,
    group_size: int = 1,
    reward_coef: float = 5.0,
    deterministic: bool = False,
    env_offset: int = 0,
    reset_timeout: float = 300.0,    # max 5 min for parallel reset of G envs
    chunk_timeout: float = 120.0,    # max 2 min for one chunk's parallel env step
    wall_budget: float = 900.0,      # max 15 min total rollout (overall budget)
) -> List[VLAPPOEpisode]:
    """Collect up to G episodes for one task in step-lock; bounded env waits.

    Returns a list of episodes that actually produced ≥1 step record. May be
    shorter than G if env subprocesses died / hung and were dropped — but
    ALWAYS returns within ~``wall_budget`` seconds.
    """
    policy.vla.eval()
    value_head.eval()

    num_unique = max(1, G // group_size)
    _rng = np.random.RandomState(seed + group_idx)
    unique_states = _rng.randint(0, n_initial_states, size=num_unique)
    state_ids = np.repeat(unique_states, group_size)[:G]

    # ── Phase 1: bounded parallel reset ──────────────────────────────
    obs_list: List[Optional[dict]] = [None] * G
    with ThreadPoolExecutor(max_workers=G) as pool:
        futs = {pool.submit(_safe_reset_env, env_pool, env_offset + g, suite_name,
                            task_id, int(state_ids[g]), seed + g): g
                for g in range(G)}
        done_set, pending = wait(futs, timeout=reset_timeout)
        for f in done_set:
            obs_list[futs[f]] = f.result()
        for f in pending:
            g = futs[f]
            print(f"  [WARN] env {g} reset hung > {reset_timeout}s; killing subprocess",
                  flush=True)
            _kill_env_subprocess(env_pool, env_offset + g)
            obs_list[g] = None
        # Pool exit: killed subprocesses → hung sockets close → threads finish.

    dead = [obs_list[g] is None for g in range(G)]
    n_dead_reset = sum(dead)
    # Do NOT early-return on all-dead: this rank must stay in the chunk loop
    # to fire FSDP collectives in lockstep with peers (they may still be
    # running real rollouts). The chunk loop's per-chunk all-reduce will
    # naturally exit when ALL ranks are done.
    if n_dead_reset:
        logger.warning(f"[steplock g{group_idx}] {n_dead_reset}/{G} envs failed reset")

    task_descriptions = [
        env_pool.envs[env_offset + g].task_description if not dead[g] else ""
        for g in range(G)
    ]

    # ── Phase 2: bounded warmup (skip dead envs) ─────────────────────
    if num_steps_wait > 0:
        alive_idx = [g for g in range(G) if not dead[g]]
        if alive_idx:
            with ThreadPoolExecutor(max_workers=len(alive_idx)) as pool:
                futs = {pool.submit(_env_dummy_steps, env_pool, env_offset + g, num_steps_wait): g
                        for g in alive_idx}
                done_set, pending = wait(futs, timeout=reset_timeout)
                for f in done_set:
                    g = futs[f]
                    obs_list[g] = f.result()
                for f in pending:
                    g = futs[f]
                    print(f"  [WARN] env {g} warmup hung > {reset_timeout}s; killing",
                          flush=True)
                    _kill_env_subprocess(env_pool, env_offset + g)
                    dead[g] = True

    # ── Phase 3: step-lock main loop with wall-clock budget ──────────
    episodes = [VLAPPOEpisode(task_id=task_id, state_idx=int(state_ids[g])) for g in range(G)]
    active = [not dead[g] for g in range(G)]
    env_steps = [0] * G
    max_chunks = max_steps // chunk_len + 1

    n_dead_step = 0
    n_dummy_chunks = 0
    _t_vla = _t_env = 0.0
    _n_chunks = 0
    _t_start = time.time()

    # Constant dummy batch for FSDP-sync forwards on idle ranks (batch=1 zero
    # data, generic instruction to keep tokenizer happy).
    _dummy_imgs = [[_ZERO_OBS["primary_image"], _ZERO_OBS["wrist_image"]]]
    _dummy_instrs = ["task"]

    for _chunk in range(max_chunks):
        # Wall-budget: kill THIS rank's still-alive envs but stay in the loop
        # so we keep firing FSDP collectives in lockstep with peers.
        elapsed = time.time() - _t_start
        if elapsed > wall_budget and any(active):
            n_killed = sum(active)
            logger.warning(f"[steplock g{group_idx}] wall budget {wall_budget}s exceeded; "
                           f"force-ending {n_killed} envs (continuing dummy forwards for FSDP sync)")
            for g in range(G):
                if active[g]:
                    active[g] = False
                    ep = episodes[g]
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
                    ep.reward = 0.0

        # Cross-rank: is ANY rank still working? If not, break in lockstep.
        local_active = sum(active)
        if dist.is_initialized():
            _at = torch.tensor([local_active], device=device, dtype=torch.int64)
            dist.all_reduce(_at, op=dist.ReduceOp.SUM)
            global_active = int(_at.item())
        else:
            global_active = local_active
        if global_active == 0:
            break  # all ranks done; safe to exit together

        active_ids = [g for g in range(G) if active[g]]

        # ── This rank idle but peers still working → dummy VLA forward only ──
        # FSDP per-layer all-gather still fires (same collective sequence as a
        # real forward), keeping NCCL in sync. No record, no env step.
        if not active_ids:
            _t0 = time.time()
            try:
                _ = policy.forward_mean_and_features(_dummy_imgs, _dummy_instrs)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  [WARN] dummy forward failed: {e}", flush=True)
            _t_vla += time.time() - _t0
            _n_chunks += 1
            n_dummy_chunks += 1
            continue

        # ── 1. ONE batched VLA forward over all active envs ──
        _t0 = time.time()
        batch_images = [[obs_list[g]["primary_image"], obs_list[g]["wrist_image"]]
                        for g in active_ids]
        batch_instrs = [task_descriptions[g] for g in active_ids]
        batch_props = [np.array(obs_list[g]["state"], dtype=np.float32) for g in active_ids]

        mean, features = policy.forward_mean_and_features(batch_images, batch_instrs)
        sampled, log_prob = policy.sample(mean)
        value = value_head(features)
        torch.cuda.synchronize()
        _t_vla += time.time() - _t0

        mean_cpu = mean.cpu()
        sampled_cpu = sampled.cpu()
        lp_cpu = log_prob.cpu()
        value_cpu = value.cpu()

        # ── 2. record per-env BEFORE stepping ──
        for j, g in enumerate(active_ids):
            episodes[g].step_records.append(VLAPPOStepRecord(
                images=[obs_list[g]["primary_image"].copy(),
                        obs_list[g]["wrist_image"].copy()],
                instruction=task_descriptions[g],
                prop_state=torch.tensor(batch_props[j]),
                action_taken=sampled_cpu[j].clone(),
                action_mean=mean_cpu[j].clone(),
                old_log_prob=float(lp_cpu[j].item()),
                value=float(value_cpu[j].item()),
            ))

        # ── 3. unnormalize ──
        exec_cpu = mean_cpu if deterministic else sampled_cpu
        action_chunks = [_unnormalize(exec_cpu[j].numpy(), action_norm_stats)
                         for j in range(len(active_ids))]

        # ── 4. all envs execute chunk in parallel — BOUNDED ──
        _t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(active_ids)) as pool:
            futs = {pool.submit(_env_step_chunk, env_pool, env_offset + g,
                                action_chunks[j], chunk_len): g
                    for j, g in enumerate(active_ids)}
            done_set, pending = wait(futs, timeout=chunk_timeout)
            for f in done_set:
                g = futs[f]
                obs, reward, env_done, steps_taken = f.result()
                obs_list[g] = obs
                env_steps[g] += steps_taken
                if env_done or env_steps[g] >= max_steps:
                    active[g] = False
                    ep = episodes[g]
                    ep.success = bool(env_done and reward > 0.5)
                    ep.reward = reward_coef if ep.success else 0.0
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
                    ep.done_cache_idx = steps_taken
            for f in pending:
                g = futs[f]
                print(f"  [WARN] chunk {_chunk}: env {g} step hung > {chunk_timeout}s; killing",
                      flush=True)
                _kill_env_subprocess(env_pool, env_offset + g)
                active[g] = False
                ep = episodes[g]
                ep.finish_step = len(ep.step_records)
                ep.env_steps = env_steps[g]
                ep.reward = 0.0
                n_dead_step += 1
            # Pool exit: killed subprocesses unblock any still-pending threads.
        _t_env += time.time() - _t0
        _n_chunks += 1

    # Finalize episodes that hit the chunk-loop ceiling
    for g in range(G):
        ep = episodes[g]
        if ep.finish_step == 0:
            ep.finish_step = len(ep.step_records)
            ep.env_steps = env_steps[g]
            ep.reward = 0.0

    # Drop episodes with zero step records (envs that died at reset/warmup)
    episodes = [ep for ep in episodes if ep.finish_step > 0]

    if _n_chunks > 0:
        total = time.time() - _t_start
        logger.info(
            f"[steplock g{group_idx}] G={G} → {len(episodes)} valid eps, "
            f"chunks={_n_chunks} (dummy={n_dummy_chunks}) "
            f"dead_reset={n_dead_reset} dead_step={n_dead_step} "
            f"total={total:.1f}s vla_fwd={_t_vla:.1f}s ({100*_t_vla/total:.0f}%) "
            f"env_step={_t_env:.1f}s ({100*_t_env/total:.0f}%)"
        )
    return episodes


def vla_ppo_collect_multitask_steplock(
    *,
    env_pool: PersistentEnvPool,
    policy: "VLAPolicy",
    value_head,
    suite_name: str,
    task_ids: List[int],
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G_per_task: int = 8,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    group_idx: int = 0,
    group_size: int = 1,
    reward_coef: float = 5.0,
    deterministic: bool = False,
    reset_timeout: float = 300.0,
    chunk_timeout: float = 120.0,
    wall_budget: float = 1200.0,
) -> List[VLAPPOEpisode]:
    """MERGED multi-task step-lock: all ``task_ids`` collected in ONE wave with
    one batched VLA forward per chunk over the mixed-task batch — replaces the
    trainer's ``len(task_ids)`` sequential per-task collects (the rollout
    bottleneck). Single-GPU only (no FSDP collectives). Each of the
    ``G_per_task * len(task_ids)`` env slots is reset to its own (task, state);
    the batched forward already consumes per-env instructions, so mixed tasks
    need no special handling. GRPO groups (group_size rollouts of one state) are
    formed within each task block, so the per-task group structure is preserved.
    """
    policy.vla.eval()
    value_head.eval()

    n_tasks = len(task_ids)
    _rng = np.random.RandomState(seed + group_idx)
    env_tasks: List[int] = []
    state_ids: List[int] = []
    num_unique = max(1, G_per_task // group_size)
    for tid in task_ids:
        st = _rng.randint(0, n_initial_states, size=num_unique)
        st = np.repeat(st, group_size)[:G_per_task]
        state_ids.extend(int(s) for s in st)
        env_tasks.extend([tid] * G_per_task)
    G = G_per_task * n_tasks

    # ── Phase 1: parallel reset, each env to its own task ──
    obs_list: List[Optional[dict]] = [None] * G
    with ThreadPoolExecutor(max_workers=G) as pool:
        futs = {pool.submit(_safe_reset_env, env_pool, g, suite_name,
                            env_tasks[g], state_ids[g], seed + g): g for g in range(G)}
        done_set, pending = wait(futs, timeout=reset_timeout)
        for f in done_set:
            obs_list[futs[f]] = f.result()
        for f in pending:
            g = futs[f]
            _kill_env_subprocess(env_pool, g)
            obs_list[g] = None
    dead = [obs_list[g] is None for g in range(G)]
    if sum(dead) == G:
        raise RuntimeError(f"All {G} envs failed to reset — env pool unrecoverable")
    task_descriptions = [env_pool.envs[g].task_description if not dead[g] else ""
                         for g in range(G)]

    # ── Phase 2: parallel warmup (skip dead) ──
    if num_steps_wait > 0:
        alive_idx = [g for g in range(G) if not dead[g]]
        with ThreadPoolExecutor(max_workers=max(1, len(alive_idx))) as pool:
            futs = {pool.submit(_env_dummy_steps, env_pool, g, num_steps_wait): g
                    for g in alive_idx}
            done_set, pending = wait(futs, timeout=reset_timeout)
            for f in done_set:
                obs_list[futs[f]] = f.result()
            for f in pending:
                g = futs[f]
                _kill_env_subprocess(env_pool, g)
                dead[g] = True

    # ── Phase 3: merged step-lock loop ──
    episodes = [VLAPPOEpisode(task_id=env_tasks[g], state_idx=state_ids[g]) for g in range(G)]
    active = [not dead[g] for g in range(G)]
    env_steps = [0] * G
    max_chunks = max_steps // chunk_len + 1
    _t_vla = _t_env = 0.0
    _n_chunks = 0
    _t_start = time.time()

    for _chunk in range(max_chunks):
        if time.time() - _t_start > wall_budget and any(active):
            for g in range(G):
                if active[g]:
                    active[g] = False
                    ep = episodes[g]
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
                    ep.reward = 0.0
        active_ids = [g for g in range(G) if active[g]]
        if not active_ids:
            break

        _t0 = time.time()
        batch_images = [[obs_list[g]["primary_image"], obs_list[g]["wrist_image"]] for g in active_ids]
        batch_instrs = [task_descriptions[g] for g in active_ids]
        batch_props = [np.array(obs_list[g]["state"], dtype=np.float32) for g in active_ids]
        with torch.no_grad():   # rollout collection — no grad (single-GPU: trainable VLA params require_grad)
            mean, features = policy.forward_mean_and_features(batch_images, batch_instrs)
            sampled, log_prob = policy.sample(mean)
            value = value_head(features)
            torch.cuda.synchronize()
        _t_vla += time.time() - _t0

        mean_cpu = mean.cpu(); sampled_cpu = sampled.cpu()
        lp_cpu = log_prob.cpu(); value_cpu = value.cpu()
        for j, g in enumerate(active_ids):
            episodes[g].step_records.append(VLAPPOStepRecord(
                images=[obs_list[g]["primary_image"].copy(), obs_list[g]["wrist_image"].copy()],
                instruction=task_descriptions[g],
                prop_state=torch.tensor(batch_props[j]),
                action_taken=sampled_cpu[j].clone(),
                action_mean=mean_cpu[j].clone(),
                old_log_prob=float(lp_cpu[j].item()),
                value=float(value_cpu[j].item()),
            ))

        exec_cpu = mean_cpu if deterministic else sampled_cpu
        action_chunks = [_unnormalize(exec_cpu[j].numpy(), action_norm_stats)
                         for j in range(len(active_ids))]
        _t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(active_ids)) as pool:
            futs = {pool.submit(_env_step_chunk, env_pool, g, action_chunks[j], chunk_len): g
                    for j, g in enumerate(active_ids)}
            done_set, pending = wait(futs, timeout=chunk_timeout)
            for f in done_set:
                g = futs[f]
                obs, reward, env_done, steps_taken = f.result()
                obs_list[g] = obs
                env_steps[g] += steps_taken
                if env_done or env_steps[g] >= max_steps:
                    active[g] = False
                    ep = episodes[g]
                    ep.success = bool(env_done and reward > 0.5)
                    ep.reward = reward_coef if ep.success else 0.0
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
                    ep.done_cache_idx = steps_taken
            for f in pending:
                g = futs[f]
                _kill_env_subprocess(env_pool, g)
                active[g] = False
                ep = episodes[g]
                ep.finish_step = len(ep.step_records)
                ep.env_steps = env_steps[g]
                ep.reward = 0.0
        _t_env += time.time() - _t0
        _n_chunks += 1

    for g in range(G):
        ep = episodes[g]
        if ep.finish_step == 0:
            ep.finish_step = len(ep.step_records)
            ep.env_steps = env_steps[g]
            ep.reward = 0.0
    episodes = [ep for ep in episodes if ep.finish_step > 0]
    if _n_chunks > 0:
        total = time.time() - _t_start
        logger.info(
            f"[merged-mt g{group_idx}] {n_tasks} tasks × {G_per_task} = {G} envs → "
            f"{len(episodes)} eps, chunks={_n_chunks} total={total:.1f}s "
            f"vla_fwd={_t_vla:.1f}s ({100*_t_vla/max(total,1e-6):.0f}%) "
            f"env_step={_t_env:.1f}s ({100*_t_env/max(total,1e-6):.0f}%)"
        )
    return episodes
