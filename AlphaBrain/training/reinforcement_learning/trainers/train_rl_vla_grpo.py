"""Vanilla VLA + GRPO trainer — full-parameter fine-tune via group-relative PG.

Differences vs train_rl_vla_ppo.py:
  * loads a second frozen VLA as the reference policy (KL penalty target)
  * no value head, no GAE — advantage = group-relative (R - μ_grp) / σ_grp
  * group_size ≥ 2 required for the relative signal to be non-zero;
    RLinf-scale launcher sets group_size=8.

Training: **FSDP** across GPUs via ``torchrun`` (RLinf-aligned), mirroring
``train_rl_vla_ppo.py``. The trainable VLM backbone is FULL_SHARD-sharded; the
small action_model is replicated per rank with manual NCCL grad all-reduce. The
reference VLA is kept replicated (frozen, bf16) on every rank — it never gets
gradients so it needs no sharding. Run without torchrun → world_size=1, plain
single-GPU (no FSDP), exactly as before.

GRPO groups (same initial state × group_size rollouts) are formed **within each
rank's local shard**, so the advantage baseline is rank-local — keep
``G_local = G / world_size`` an integer multiple of ``group_size``.

Memory per rank under FSDP-6: trainable VLM shard (~9 GB) + ref VLA full
(~8 GB) + Adam shard ≈ fits comfortably on 80 GB.
"""
import functools
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLVisionBlock,
)

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool import PersistentEnvPool
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO import (
    VLAPolicy,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_ppo_rollout_fast import (
    vla_ppo_collect_steplock,
    vla_ppo_collect_multitask_steplock,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_grpo_loss import (
    vla_grpo_loss,
)

logger = logging.getLogger(__name__)


class _NoopValueHead(nn.Module):
    """Zero-value head so we can reuse vla_ppo_collect (which records value).

    GRPO ignores recorded values entirely; we just need the collector to run.
    """
    def forward(self, features):
        if features.dim() == 3:
            return torch.zeros(features.shape[0], device=features.device)
        return torch.zeros(features.shape[0], device=features.device)


def _dist_setup(args):
    """Init torch.distributed when launched via torchrun; else single-process.

    Returns (rank, local_rank, world_size, is_dist).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return rank, local_rank, world_size, True
    local_rank = args.train_gpu if args.train_gpu is not None else 0
    torch.cuda.set_device(local_rank)
    return 0, local_rank, 1, False


def run_rl_vla_grpo(args):
    """Vanilla VLA-GRPO entry point — data-parallel (torchrun) or single-GPU."""
    rank, local_rank, world_size, is_dist = _dist_setup(args)
    is_main = (rank == 0)
    device = f"cuda:{local_rank}"
    if not is_main:
        logging.getLogger("AlphaBrain.training.reinforcement_learning").setLevel(logging.WARNING)

    # Per-rank seed → diverse rollouts; params are made identical below.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    logger.info(f"=== Vanilla VLA + GRPO — rank {rank}/{world_size}, {device} ===")

    # Sanity: GRPO needs group_size >= 2 for a usable relative signal
    if args.group_size < 2:
        logger.warning(
            f"group_size={args.group_size} < 2 — GRPO advantage will be zero "
            "for every episode. Set --group_size 2 (or higher)."
        )

    # ── Trainable VLA (every rank loads the same ckpt → identical weights) ──
    logger.info(f"[rank {rank}] Loading trainable VLA from {args.ckpt_path} (bf16)")
    vla = BaseFramework.from_pretrained(args.ckpt_path)
    vla = vla.to(torch.bfloat16).to(device).train()
    if hasattr(vla, "qwen_vl_interface") and hasattr(vla.qwen_vl_interface, "model"):
        try:
            vla.qwen_vl_interface.model.gradient_checkpointing_enable()
            logger.info("gradient_checkpointing enabled on trainable VLA")
        except Exception as e:
            logger.warning(f"gradient_checkpointing_enable failed: {e}")
    n_train = 0
    for p in vla.parameters():
        p.requires_grad_(True)
        n_train += p.numel()
    logger.info(f"trainable VLA params: {n_train / 1e9:.3f}B")

    # ── Reference VLA (frozen snapshot of init; replicated per rank) ────────
    logger.info(f"[rank {rank}] Loading reference VLA from {args.ckpt_path} (bf16, frozen)")
    ref_vla = BaseFramework.from_pretrained(args.ckpt_path)
    ref_vla = ref_vla.to(torch.bfloat16).to(device).eval()
    for p in ref_vla.parameters():
        p.requires_grad_(False)
    logger.info("reference VLA loaded and frozen")

    chunk_len = vla.chunk_len
    action_dim = vla.config.framework.action_model.action_dim
    _norm = vla.norm_stats
    action_norm_stats = _norm[next(iter(_norm.keys()))]["action"]
    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    max_steps = MAX_STEPS[args.suite]

    # ── Multi-task: --all_tasks / --task_ids ─────────────────────────────
    if getattr(args, "task_ids", None):
        _sel_tasks = [int(x) for x in args.task_ids.split(",")]
        args.all_tasks = True
    else:
        _sel_tasks = None
    task_list = ((_sel_tasks if _sel_tasks else list(range(n_tasks)))
                 if getattr(args, "all_tasks", False) else None)
    tasks_per_iter = getattr(args, "tasks_per_iter", 0) or (
        len(task_list) if task_list is not None else 0)
    if task_list is not None and is_main:
        logger.info(f"[multi-task] training across tasks {task_list} "
                    f"(cycle {tasks_per_iter} task/iter, {args.G} ep/task, "
                    f"eval covers all {len(task_list)})")

    # ── FSDP-shard the trainable VLM backbone (RLinf-aligned) ──────────────
    # Only the heavy Qwen2.5-VL transformer is sharded; action_model stays
    # replicated (manual grad all-reduce). The frozen ref VLA is NOT wrapped.
    fsdp_vlm = None
    if is_dist:
        bf16_mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock},
        )
        vla.qwen_vl_interface.model = FSDP(
            vla.qwen_vl_interface.model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=bf16_mp,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
            use_orig_params=True,
        )
        fsdp_vlm = vla.qwen_vl_interface.model
        logger.info(f"[rank {rank}] FSDP-wrapped trainable VLM (FULL_SHARD); "
                    f"action_model replicated; ref VLA replicated (frozen).")

    policy = VLAPolicy(vla, fixed_std=args.fixed_std)
    ref_policy = VLAPolicy(ref_vla, fixed_std=args.fixed_std)
    dummy_vh = _NoopValueHead().to(device)

    # Optimizer: VLA only (no value head, no critic)
    optimizer = torch.optim.AdamW(
        vla.parameters(), lr=args.lr_vla,
        betas=(0.9, 0.95), weight_decay=1e-8,
    )

    # ── Per-rank episode shard + persistent env pool ──
    def _shard(total):
        return total // world_size + (1 if rank < total % world_size else 0)

    G_local = _shard(args.G)
    eval_local = _shard(args.eval_n_episodes)
    # Single-GPU MERGED multi-task rollout: collect all tasks in ONE step-lock
    # wave (one batched VLA forward) instead of len(task_list) sequential
    # collects — needs G_local×n_tasks env slots in the pool. (Multi-GPU FSDP
    # keeps the sequential per-task path: a merged per-rank pool ×world_size
    # would blow the env-worker RAM ceiling.)
    use_merged = (not is_dist) and (task_list is not None) and len(task_list) > 1
    if use_merged:
        pool_size = max(G_local * len(task_list), eval_local, 1)
    else:
        pool_size = max(G_local, eval_local, 1)
    if is_dist and G_local % args.group_size != 0 and is_main:
        logger.warning(f"G_local ({G_local}) is not a multiple of group_size "
                       f"({args.group_size}); last group on each rank is truncated.")
    logger.info(f"[rank {rank}] G_local={G_local} eval_local={eval_local} "
                f"pool={pool_size} (G total={args.G})")
    _phys = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    egl_gpu = _phys[local_rank] if local_rank < len(_phys) else local_rank
    env_pool = PersistentEnvPool(
        num_envs=pool_size,
        libero_python=os.environ.get("LIBERO_PYTHON"),
        egl_gpu_id=egl_gpu,
    )

    def _collect(*, task_id, n_local, group_idx, group_size, deterministic):
        if n_local <= 0:
            return []
        return vla_ppo_collect_steplock(
            env_pool=env_pool, policy=policy, value_head=dummy_vh,
            suite_name=args.suite, task_id=task_id, n_initial_states=50,
            action_norm_stats=action_norm_stats, max_steps=max_steps,
            chunk_len=chunk_len, G=n_local, seed=args.seed,
            num_steps_wait=args.num_steps_wait, device=device,
            group_idx=group_idx, group_size=group_size,
            reward_coef=args.reward_coef, deterministic=deterministic,
        )

    def _reduce(x, op):
        """All-reduce a python scalar across ranks (no-op when single-process)."""
        if not is_dist:
            return x
        t = torch.tensor([float(x)], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=op)
        return float(t.item())

    # action_model is NOT FSDP-managed → manual cross-rank grad averaging.
    unwrapped_params = list(vla.action_model.parameters())

    def _allreduce_unwrapped_grads():
        if not is_dist:
            return
        for p in unwrapped_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _save_full_ckpt(ckpt_dir):
        """Save vla. Under FSDP, gathers FULL_STATE_DICT on rank 0."""
        if fsdp_vlm is not None:
            with FSDP.state_dict_type(
                fsdp_vlm, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_sd = vla.state_dict()
                if is_main:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        vla.save_pretrained(str(ckpt_dir / "vla"), state_dict=full_sd)
                    except Exception as e:
                        logger.warning(f"vla.save_pretrained failed: {e}; state_dict fallback")
                        torch.save(full_sd, ckpt_dir / "vla_state_dict.pt")
                    logger.info(f"Saved ckpt → {ckpt_dir}")
            if is_dist:
                dist.barrier()
        else:
            if is_main:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                try:
                    vla.save_pretrained(str(ckpt_dir / "vla"))
                except Exception as e:
                    logger.warning(f"vla.save_pretrained failed: {e}; state_dict fallback")
                    torch.save(vla.state_dict(), ckpt_dir / "vla_state_dict.pt")
                logger.info(f"Saved ckpt → {ckpt_dir}")

    if args.use_wandb and is_main:
        run_name = args.run_name or f"vla_grpo_qwen_task{args.task_id}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "chunk_len": chunk_len,
                           "action_dim": action_dim,
                           "algo": "vla_grpo_full", "rollout": "steplock",
                           "world_size": world_size})

    metrics_history = []
    best_sr = 0.0
    running_sr = []
    total_env_steps = 0

    try:
        for iteration in range(1, args.max_iter + 1):
            # Multi-task: this iteration's rotating window of tasks.
            if task_list is not None:
                _start = ((iteration - 1) * tasks_per_iter) % len(task_list)
                iter_tasks = [task_list[(_start + k) % len(task_list)]
                              for k in range(tasks_per_iter)]
            else:
                iter_tasks = [args.task_id if args.task_id >= 0
                              else random.randint(0, n_tasks - 1)]
            task_id = iter_tasks[0]  # for logging

            if is_main:
                logger.info("=" * 60)
                logger.info(
                    f"[iter {iteration}/{args.max_iter}] collecting {args.G} ep "
                    f"({world_size}×{G_local}) on tasks {iter_tasks} "
                    f"(group_size={args.group_size})"
                )

            # ── Rollout — each rank step-locks its own env-pool shard ──
            # GRPO needs intra-group relative signal, so each task is collected
            # as its own group block (group_idx offset per task); the blocks are
            # concatenated for the rank-local update.
            t_roll = time.time()
            if use_merged:
                # ONE merged wave over all tasks (single GPU, batched forward).
                local_episodes = vla_ppo_collect_multitask_steplock(
                    env_pool=env_pool, policy=policy, value_head=dummy_vh,
                    suite_name=args.suite, task_ids=iter_tasks, n_initial_states=50,
                    action_norm_stats=action_norm_stats, max_steps=max_steps,
                    chunk_len=chunk_len, G_per_task=G_local, seed=args.seed,
                    num_steps_wait=args.num_steps_wait, device=device,
                    group_idx=iteration * 100, group_size=args.group_size,
                    reward_coef=args.reward_coef, deterministic=False,
                )
            else:
                local_episodes = []
                for _ti, _t in enumerate(iter_tasks):
                    local_episodes += _collect(
                        task_id=_t, n_local=G_local,
                        group_idx=(iteration * world_size + rank) * 100 + _ti,
                        group_size=args.group_size, deterministic=False,
                    )
            roll_sec = _reduce(time.time() - t_roll, dist.ReduceOp.MAX)

            # Stats (local → global). local_episodes may be EMPTY if every env
            # on this rank failed — still participate in collectives below.
            if local_episodes:
                ep_rewards = np.array([ep.reward for ep in local_episodes])
                sr_local = float(np.mean(ep_rewards > 0.5))
                mean_r_local = float(np.mean(ep_rewards))
                mean_steps_local = float(np.mean([ep.finish_step for ep in local_episodes]))
                env_steps_local = sum(ep.env_steps for ep in local_episodes)
            else:
                sr_local = mean_r_local = mean_steps_local = 0.0
                env_steps_local = 0
            sr = _reduce(sr_local, dist.ReduceOp.AVG)
            mean_r = _reduce(mean_r_local, dist.ReduceOp.AVG)
            mean_steps = _reduce(mean_steps_local, dist.ReduceOp.AVG)
            iter_env_steps = int(_reduce(env_steps_local, dist.ReduceOp.SUM))
            total_env_steps += iter_env_steps
            running_sr.append(sr)
            if len(running_sr) > 20: running_sr.pop(0)
            best_sr = max(best_sr, sr)

            if is_main:
                logger.info(
                    f"[iter {iteration}] SR={sr:.2f} (best={best_sr:.2f}, "
                    f"avg={np.mean(running_sr):.2f}) reward={mean_r:.2f} "
                    f"steps={mean_steps:.1f} env_steps={iter_env_steps}"
                )

            # ── GRPO update — each rank backwards its shard; grads all-reduced ──
            if is_main:
                logger.info(
                    f"[iter {iteration}] GRPO update "
                    f"({args.ppo_epochs} epochs, micro_batch={args.micro_batch}, "
                    f"kl_coef={args.grpo_kl_coef})"
                )
            vla.train()
            epoch_stats = []
            t_upd = time.time()
            for grpo_epoch in range(args.ppo_epochs):
                optimizer.zero_grad()
                # vla_grpo_loss re-forwards + backward()s per micro-batch
                # internally, in FSDP lockstep (all-reduce MIN(has_data) +
                # MAX(n_batches) with loss×0 padding) so every rank fires the
                # same FSDP collective sequence — no NCCL desync on uneven
                # per-rank episode counts. Returns stats only (grads local).
                stats = vla_grpo_loss(
                    policy=policy, ref_policy=ref_policy,
                    episodes=local_episodes,
                    clip_eps=args.clip_eps,
                    clip_eps_high=args.clip_eps_high,
                    dual_clip_c=args.dual_clip_c,
                    token_level=args.grpo_token_level,
                    temporal_credit=args.grpo_temporal_credit,
                    gamma=args.gamma,
                    kl_coef=args.grpo_kl_coef,
                    micro_batch=args.micro_batch,
                    device=device,
                )
                _allreduce_unwrapped_grads()   # FSDP-VLM reduces its own
                if args.max_grad_norm > 0:
                    if fsdp_vlm is not None:
                        fsdp_vlm.clip_grad_norm_(args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(
                            unwrapped_params, args.max_grad_norm,
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            vla.parameters(), args.max_grad_norm,
                        )
                optimizer.step()
                if stats.get("n_steps", 0) > 0:
                    epoch_stats.append(stats)
            upd_sec = _reduce(time.time() - t_upd, dist.ReduceOp.MAX)
            if is_main:
                logger.info(f"[iter {iteration}] timing: rollout={roll_sec:.1f}s "
                            f"update={upd_sec:.1f}s")

            # ── Periodic deterministic eval (held-out states) ──
            eval_sr = None
            eval_per_task = {}
            if args.eval_interval > 0 and iteration % args.eval_interval == 0:
                _eval_tasks = task_list if task_list is not None else [task_id]
                try:
                    vla.eval()
                    for _t in _eval_tasks:
                        eval_eps = _collect(
                            task_id=_t, n_local=eval_local,
                            group_idx=rank, group_size=1, deterministic=True,
                        )
                        local_succ = float(np.mean([ep.success for ep in eval_eps])) if eval_eps else 0.0
                        eval_per_task[_t] = _reduce(local_succ, dist.ReduceOp.AVG)
                    eval_sr = float(np.mean(list(eval_per_task.values()))) if eval_per_task else None
                    if is_main and eval_sr is not None:
                        if len(eval_per_task) > 1:
                            logger.info(f"[iter {iteration}] [eval] all-task mean "
                                        f"SR={eval_sr:.2%} | " +
                                        " ".join(f"t{t}:{s:.2f}" for t, s in sorted(eval_per_task.items())))
                        else:
                            logger.info(f"[iter {iteration}] [eval] deterministic "
                                        f"SR={eval_sr:.2%}")
                except Exception as e:
                    logger.warning(f"[iter {iteration}] eval failed: {e}")
                    eval_sr = None
                vla.train()

            if not epoch_stats:
                # No usable update this iter (e.g. every group all-success /
                # all-fail). Still record SR so the curve is continuous.
                metrics_history.append({
                    "iter": iteration, "total_env_steps": total_env_steps,
                    "success_rate": sr, "best_success_rate": best_sr,
                    "running_avg_sr": float(np.mean(running_sr)), "mean_reward": mean_r,
                    "eval_sr": eval_sr, "eval_per_task": eval_per_task or None,
                    "iter_tasks": iter_tasks,
                })
                if args.use_wandb and is_main:
                    wlog = {"rollout/success_rate": sr, "rollout/best": best_sr,
                            "rollout/mean_reward": mean_r,
                            "rollout/total_env_steps": total_env_steps}
                    if eval_sr is not None:
                        wlog["eval/success_rate"] = eval_sr
                    wandb.log(wlog, step=iteration)
                if iteration % args.save_interval == 0:
                    _save_full_ckpt(Path(args.output_dir) / "checkpoints" / f"vla_grpo_iter_{iteration:05d}")
                continue

            avg = lambda k: float(np.mean([s[k] for s in epoch_stats if k in s]))
            log_entry = {
                "iter": iteration, "total_env_steps": total_env_steps,
                "success_rate": sr, "best_success_rate": best_sr,
                "running_avg_sr": float(np.mean(running_sr)), "mean_reward": mean_r,
                "loss": _reduce(avg("loss"), dist.ReduceOp.AVG),
                "pg_loss": _reduce(avg("pg_loss"), dist.ReduceOp.AVG),
                "kl": _reduce(avg("kl"), dist.ReduceOp.AVG),
                "ratio_mean": _reduce(avg("ratio_mean"), dist.ReduceOp.AVG),
                "clip_frac": _reduce(avg("clip_frac"), dist.ReduceOp.AVG),
                "advantage_mean": _reduce(avg("advantage_mean"), dist.ReduceOp.AVG),
                "advantage_std": _reduce(avg("advantage_std"), dist.ReduceOp.AVG),
                "n_groups": _reduce(avg("n_groups"), dist.ReduceOp.SUM),
                "n_groups_with_signal": _reduce(avg("n_groups_with_signal"), dist.ReduceOp.SUM),
                "n_steps": int(_reduce(avg("n_steps"), dist.ReduceOp.SUM)),
                "eval_sr": eval_sr,
                "eval_per_task": eval_per_task or None,
                "iter_tasks": iter_tasks,
            }
            metrics_history.append(log_entry)
            if is_main:
                logger.info(
                    f"  loss={log_entry['loss']:.4f} pg={log_entry['pg_loss']:.4f} "
                    f"kl={log_entry['kl']:.4f} ratio={log_entry['ratio_mean']:.3f} "
                    f"clip_frac={log_entry['clip_frac']:.3f} "
                    f"adv_std={log_entry['advantage_std']:.3f} "
                    f"groups_w_signal={log_entry['n_groups_with_signal']:.0f}/"
                    f"{log_entry['n_groups']:.0f}"
                )

            if args.use_wandb and is_main:
                wandb_log = {
                    "rollout/success_rate": sr, "rollout/best": best_sr,
                    "rollout/mean_reward": mean_r,
                    "rollout/total_env_steps": total_env_steps,
                    "train/loss": log_entry["loss"], "train/pg_loss": log_entry["pg_loss"],
                    "train/kl": log_entry["kl"],
                    "train/ratio_mean": log_entry["ratio_mean"],
                    "train/clip_frac": log_entry["clip_frac"],
                    "train/advantage_mean": log_entry["advantage_mean"],
                    "train/advantage_std": log_entry["advantage_std"],
                    "train/n_groups": log_entry["n_groups"],
                    "train/n_groups_with_signal": log_entry["n_groups_with_signal"],
                    "train/n_steps": log_entry["n_steps"],
                    "time/rollout_sec": roll_sec, "time/update_sec": upd_sec,
                }
                if eval_sr is not None:
                    wandb_log["eval/success_rate"] = eval_sr
                    for _t, _s in eval_per_task.items():
                        wandb_log[f"eval/task_{_t}_sr"] = _s
                wandb.log(wandb_log, step=iteration)

            # Optional: refresh reference policy every K iters. Under FSDP the
            # trainable params are sharded, so an in-place copy to the replicated
            # ref is not well-defined — skip (default ref_update_interval=0).
            if args.ref_update_interval > 0 and iteration % args.ref_update_interval == 0:
                if fsdp_vlm is not None:
                    if is_main:
                        logger.warning("ref_update_interval>0 is unsupported under "
                                       "FSDP; skipping reference refresh.")
                else:
                    logger.info(f"[iter {iteration}] refreshing reference VLA")
                    with torch.no_grad():
                        for ref_p, cur_p in zip(ref_vla.parameters(), vla.parameters()):
                            ref_p.data.copy_(cur_p.data)

            if iteration % args.save_interval == 0:
                _save_full_ckpt(Path(args.output_dir) / "checkpoints" / f"vla_grpo_iter_{iteration:05d}")
    finally:
        try:
            env_pool.close()
        except Exception:
            pass

    # Final checkpoint + metrics
    _save_full_ckpt(Path(args.output_dir) / "checkpoints" / f"vla_grpo_iter_{args.max_iter:05d}_final")
    if is_main:
        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        logger.info(f"Done. Metrics → {metrics_path}")
        if args.use_wandb:
            wandb.finish()

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
