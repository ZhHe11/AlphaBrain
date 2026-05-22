"""Vanilla VLA + GRPO trainer — full-parameter fine-tune via group-relative PG.

Differences vs train_rl_vla_ppo.py:
  * loads a second frozen VLA as the reference policy (KL penalty target)
  * no value head, no GAE — advantage = group-relative (R - μ_grp) / σ_grp
  * group_size ≥ 2 required for the relative signal to be non-zero;
    launcher sets group_size=2 by default.

Memory: ~50 GB (trainable VLA + Adam) + ~8 GB (ref VLA bf16) ≈ 58 GB
on 80 GB GPU.
"""
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO import (
    VLAPolicy, vla_ppo_collect,
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
        # features: (B, T, D) or (B, D) → return zeros over batch dim
        if features.dim() == 3:
            return torch.zeros(features.shape[0], device=features.device)
        return torch.zeros(features.shape[0], device=features.device)


def run_rl_vla_grpo(args):
    """Vanilla VLA-GRPO entry. Single-GPU."""
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    device = f"cuda:{args.train_gpu}" if args.train_gpu is not None else "cuda:0"
    logger.info(f"=== Vanilla VLA + GRPO (full finetune) on {device} ===")

    # Sanity: GRPO needs group_size >= 2 for a usable relative signal
    if args.group_size < 2:
        logger.warning(
            f"group_size={args.group_size} < 2 — GRPO advantage will be zero "
            "for every episode. Set --group_size 2 (or higher)."
        )

    # ── Trainable VLA ─────────────────────────────────────────────────
    logger.info(f"Loading trainable VLA from {args.ckpt_path} (bf16)")
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

    # ── Reference VLA (frozen snapshot of init) ────────────────────────
    logger.info(f"Loading reference VLA from {args.ckpt_path} (bf16, frozen)")
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

    policy = VLAPolicy(vla, fixed_std=args.fixed_std)
    ref_policy = VLAPolicy(ref_vla, fixed_std=args.fixed_std)
    dummy_vh = _NoopValueHead().to(device)

    # Optimizer: VLA only (no value head, no critic)
    optimizer = torch.optim.AdamW(
        vla.parameters(), lr=args.lr_vla,
        betas=(0.9, 0.95), weight_decay=1e-8,
    )

    if args.use_wandb:
        run_name = args.run_name or f"vla_grpo_qwen_task{args.task_id}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "chunk_len": chunk_len,
                           "action_dim": action_dim,
                           "algo": "vla_grpo_full"})

    metrics_history = []
    best_sr = 0.0
    running_sr = []
    total_env_steps = 0

    for iteration in range(1, args.max_iter + 1):
        logger.info("=" * 60)
        logger.info(
            f"[iter {iteration}/{args.max_iter}] collecting {args.G} ep "
            f"on task {args.task_id} (group_size={args.group_size})"
        )

        task_id = args.task_id if args.task_id >= 0 else random.randint(0, n_tasks - 1)
        group_seed = args.seed + iteration * 1000

        # ── Rollout (reuse VLA-PPO collector w/ noop value head) ──────
        local_episodes = vla_ppo_collect(
            policy=policy, value_head=dummy_vh,
            suite_name=args.suite, task_id=task_id,
            n_initial_states=50,
            action_norm_stats=action_norm_stats,
            max_steps=max_steps, chunk_len=chunk_len, G=args.G,
            libero_python=os.environ.get("LIBERO_PYTHON"),
            seed=group_seed,
            num_steps_wait=args.num_steps_wait,
            device=device, num_envs=args.num_envs,
            group_idx=iteration, group_size=args.group_size,
            reward_coef=args.reward_coef,
        )

        # Stats
        ep_rewards = np.array([ep.reward for ep in local_episodes])
        sr = float(np.mean(ep_rewards > 0.5))
        mean_r = float(np.mean(ep_rewards))
        mean_steps = float(np.mean([ep.finish_step for ep in local_episodes]))
        iter_env_steps = sum(ep.env_steps for ep in local_episodes)
        total_env_steps += iter_env_steps
        running_sr.append(sr)
        if len(running_sr) > 20: running_sr.pop(0)
        best_sr = max(best_sr, sr)

        logger.info(
            f"[iter {iteration}] SR={sr:.2f} (best={best_sr:.2f}, "
            f"avg={np.mean(running_sr):.2f}) reward={mean_r:.2f} "
            f"steps={mean_steps:.1f} env_steps={iter_env_steps}"
        )

        # ── GRPO update ───────────────────────────────────────────────
        logger.info(
            f"[iter {iteration}] GRPO update "
            f"({args.ppo_epochs} epochs, micro_batch={args.micro_batch}, "
            f"kl_coef={args.grpo_kl_coef})"
        )
        vla.train()
        epoch_stats = []
        for grpo_epoch in range(args.ppo_epochs):
            optimizer.zero_grad()
            loss, stats = vla_grpo_loss(
                policy=policy, ref_policy=ref_policy,
                episodes=local_episodes,
                clip_eps=args.clip_eps,
                kl_coef=args.grpo_kl_coef,
                micro_batch=args.micro_batch,
                device=device,
            )
            if stats.get("n_steps", 0) == 0:
                logger.warning(f"  epoch {grpo_epoch}: 0 transitions to update on")
                break
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(vla.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_stats.append(stats)

        if not epoch_stats:
            continue

        # ── Periodic deterministic eval (held-out states, real-time monitoring) ──
        eval_sr = None
        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
            try:
                vla.eval()
                eval_eps = vla_ppo_collect(
                    policy=policy, value_head=dummy_vh,
                    suite_name=args.suite, task_id=task_id,
                    n_initial_states=50, action_norm_stats=action_norm_stats,
                    max_steps=max_steps, chunk_len=chunk_len,
                    G=args.eval_n_episodes,
                    libero_python=os.environ.get("LIBERO_PYTHON"),
                    seed=args.seed, num_steps_wait=args.num_steps_wait,
                    device=device, num_envs=args.num_envs,
                    group_idx=0, group_size=1, reward_coef=args.reward_coef,
                    deterministic=True,
                )
                eval_sr = float(np.mean([ep.success for ep in eval_eps])) if eval_eps else 0.0
                logger.info(f"[iter {iteration}] [eval] deterministic SR={eval_sr:.2%} "
                            f"({len(eval_eps)} ep)")
            except Exception as e:
                logger.warning(f"[iter {iteration}] eval failed: {e}")
                eval_sr = None
            vla.train()

        avg = lambda k: float(np.mean([s[k] for s in epoch_stats if k in s]))
        log_entry = {
            "iter": iteration, "total_env_steps": total_env_steps,
            "success_rate": sr, "best_success_rate": best_sr,
            "running_avg_sr": float(np.mean(running_sr)), "mean_reward": mean_r,
            "loss": avg("loss"), "pg_loss": avg("pg_loss"), "kl": avg("kl"),
            "ratio_mean": avg("ratio_mean"), "clip_frac": avg("clip_frac"),
            "advantage_mean": avg("advantage_mean"),
            "advantage_std": avg("advantage_std"),
            "n_groups": avg("n_groups"),
            "n_groups_with_signal": avg("n_groups_with_signal"),
            "n_steps": avg("n_steps"),
            "eval_sr": eval_sr,
        }
        metrics_history.append(log_entry)
        logger.info(
            f"  loss={log_entry['loss']:.4f} pg={log_entry['pg_loss']:.4f} "
            f"kl={log_entry['kl']:.4f} ratio={log_entry['ratio_mean']:.3f} "
            f"clip_frac={log_entry['clip_frac']:.3f} "
            f"adv_std={log_entry['advantage_std']:.3f} "
            f"groups_w_signal={log_entry['n_groups_with_signal']:.0f}/"
            f"{log_entry['n_groups']:.0f}"
        )

        if args.use_wandb:
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
            }
            if eval_sr is not None:
                wandb_log["eval/success_rate"] = eval_sr
            wandb.log(wandb_log, step=iteration)

        # Optional: refresh reference policy every K iters
        if args.ref_update_interval > 0 and iteration % args.ref_update_interval == 0:
            logger.info(f"[iter {iteration}] refreshing reference VLA from current VLA")
            with torch.no_grad():
                for ref_p, cur_p in zip(ref_vla.parameters(), vla.parameters()):
                    ref_p.data.copy_(cur_p.data)

        # Checkpoint
        if iteration % args.save_interval == 0:
            ckpt_dir = Path(args.output_dir) / "checkpoints" / f"vla_grpo_iter_{iteration:05d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            try:
                vla.save_pretrained(str(ckpt_dir / "vla"))
            except Exception as e:
                logger.warning(f"vla.save_pretrained failed: {e}; falling back to state_dict")
                torch.save(vla.state_dict(), ckpt_dir / "vla_state_dict.pt")
            logger.info(f"Saved ckpt → {ckpt_dir}")

    # Final
    final_dir = Path(args.output_dir) / "checkpoints" / f"vla_grpo_iter_{args.max_iter:05d}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    try:
        vla.save_pretrained(str(final_dir / "vla"))
    except Exception:
        torch.save(vla.state_dict(), final_dir / "vla_state_dict.pt")

    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"Done. Metrics → {metrics_path}")

    if args.use_wandb:
        wandb.finish()
