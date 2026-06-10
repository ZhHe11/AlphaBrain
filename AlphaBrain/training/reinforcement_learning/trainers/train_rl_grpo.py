"""Phase 2 (on-policy GRPO): group-relative policy optimization.

GRPO drops the value critic entirely. Per-episode advantage is the
group-normalized return where a "group" = episodes sharing the same
initial state (state_idx). A frozen reference actor regularizes policy
drift through a KL penalty (k3 estimator).

Compared to train_rl_onpolicy.run_rl (PPO):
  - No ActionTokenCritic, no V/GAE, no value loss
  - Reference actor maintained as a deepcopy (no grad)
  - Uses action_token_grpo_loss

Reuses the rest: collector, encoder, actor, eval, ckpt I/O.
"""
import copy
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.common.ckpt_io import save_rlt_checkpoint, maybe_resume
from AlphaBrain.training.reinforcement_learning.eval.eval_helpers import _eval_distributed
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
    ActionTokenActor, ActionTokenCritic,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import (
    ActionTokenEncoderDecoder,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import (
    action_token_collect_group, action_token_grpo_loss,
)

logger = logging.getLogger(__name__)


def _bc_warmstart_actor(actor, episodes, n_steps, lr, device):
    """BC-pretrain the actor to reproduce the VLA reference action ã.

    Cold-start fix for on-policy RL on the bottleneck actor: a fresh
    ActionTokenActor's near-zero-init head outputs ≈0, so under sparse
    binary reward the policy never lands a success and GRPO's
    group-relative advantage stays 0 → no gradient (the run dies at SR 0).
    We first BC-fit the actor to ã (≈ an SFT step) on the first rollout
    batch's (z_rl, ã, s_p) tuples, so the GRPO loop starts from a working
    ≈VLA policy. Returns (bc_before, bc_after) or None if no data.
    """
    rl, vla, prop = [], [], []
    for ep in episodes:
        for t in range(ep.finish_step):
            s = ep.step_records[t]
            rl.append(s.rl_token)
            vla.append(s.vla_action)
            prop.append(s.prop_state if s.prop_state is not None else torch.zeros(8))
    if not rl:
        return None
    rl = torch.stack(rl).to(device)
    vla = torch.stack(vla).to(device)
    prop = torch.stack(prop).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    actor.train()
    bc_before = None
    for step in range(n_steps):
        opt.zero_grad()
        mean, _ = actor(rl, vla, prop, deterministic=True)
        bc = ((mean - vla) ** 2).sum(dim=(-2, -1)).mean()
        if step == 0:
            bc_before = float(bc.item())
        bc.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
    return bc_before, float(bc.item())


def run_rl_grpo(args):
    """GRPO trainer entry point. Mirrors run_rl (PPO) but with group-relative
    advantage, KL-to-ref penalty, and no value function."""
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    logger.info(f"[rank {rank}/{world_size}] Loading frozen VLA from {args.ckpt_path}")
    frozen_vla = BaseFramework.from_pretrained(args.ckpt_path)
    frozen_vla = frozen_vla.to(torch.bfloat16).to(device).eval()
    for p in frozen_vla.parameters():
        p.requires_grad_(False)

    hidden_dim = frozen_vla.qwen_vl_interface.model.config.hidden_size
    chunk_len = frozen_vla.chunk_len
    action_dim = frozen_vla.config.framework.action_model.action_dim

    _norm = frozen_vla.norm_stats
    action_norm_stats = _norm[next(iter(_norm.keys()))]["action"]

    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    max_steps = MAX_STEPS[args.suite]

    # ── Multi-task: --all_tasks / --task_ids ─────────────────────────────
    # On-policy trainers historically ignored --all_tasks and silently trained
    # only args.task_id. When set, collect args.G episodes from EVERY task each
    # iteration (mirrors the off-policy trainer) so one shared policy learns all
    # tasks; eval then covers all tasks. task_list=None ⇒ original single-task.
    if getattr(args, "task_ids", None):
        _sel_tasks = [int(x) for x in args.task_ids.split(",")]
        args.all_tasks = True
    else:
        _sel_tasks = None
    task_list = ((_sel_tasks if _sel_tasks else list(range(n_tasks)))
                 if getattr(args, "all_tasks", False) else None)
    if task_list is not None:
        logger.info(f"[multi-task] training across tasks {task_list} "
                    f"({args.G} ep/task/iter, eval covers all)")

    # Encoder — action_token (RLT_a) or rlt (full-token RLT) per --encoder_mode
    encoder_mode = getattr(args, "encoder_mode", "action_token")
    if encoder_mode == "rlt":
        # RLT reference track: z_rl is kept at the VLA hidden dim (no extra
        # bottleneck projection). --bottleneck_dim is repurposed as the
        # encoder hidden dim and must equal the VLA hidden_size.
        from AlphaBrain.training.reinforcement_learning.algos.RLT import (
            RLTokenEncoderDecoder,
        )
        if args.bottleneck_dim != hidden_dim:
            logger.warning(
                f"  --bottleneck_dim={args.bottleneck_dim} != VLA hidden_dim="
                f"{hidden_dim}; RLT encoder uses the VLA hidden dim. "
                f"Overriding bottleneck_dim."
            )
            args.bottleneck_dim = hidden_dim
        enc_dec = RLTokenEncoderDecoder(
            hidden_dim=hidden_dim,
            num_heads=args.encoder_heads,
            encoder_layers=args.encoder_layers,
            decoder_layers=getattr(args, "decoder_layers", args.encoder_layers),
            max_len=getattr(args, "max_len", 4096),
        ).to(device)
    else:
        enc_dec = ActionTokenEncoderDecoder(
            input_dim=hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
            chunk_len=chunk_len,
            num_heads=args.encoder_heads,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.encoder_layers,
        ).to(device)
    if is_main:
        logger.info(f"Encoder mode: {encoder_mode}  (bottleneck_dim={args.bottleneck_dim})")
    if args.encoder_path:
        logger.info(f"[rank {rank}] Loading pretrained encoder from {args.encoder_path}")
        enc_dec.load_state_dict(torch.load(args.encoder_path, map_location=device))

    # Actor (stochastic Gaussian, fixed_std)
    actor = ActionTokenActor(
        bottleneck_dim=args.bottleneck_dim,
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=args.actor_hidden_dim,
        ref_dropout=args.ref_dropout,
        residual=True,  # μ = ã + Δ — fresh actor = VLA pass-through, no on-policy cold-start
    ).to(device)

    # Reference actor — frozen snapshot for KL penalty
    ref_actor = copy.deepcopy(actor).eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    # A dummy critic kept only so collect_group / ckpt-save signatures stay
    # the same as PPO. Its value head is unused by GRPO loss.
    dummy_critic = ActionTokenCritic(
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.critic_hidden_dim,
    ).to(device).eval()
    for p in dummy_critic.parameters():
        p.requires_grad_(False)

    if is_main:
        actor_params = sum(p.numel() for p in actor.parameters())
        enc_params = sum(p.numel() for p in enc_dec.parameters())
        logger.info(f"Frozen VLA: {sum(p.numel() for p in frozen_vla.parameters()) / 1e9:.2f}B × {world_size} GPU")
        logger.info(f"GRPO trainable: encoder={enc_params / 1e6:.2f}M  actor={actor_params / 1e6:.2f}M  (no critic)")
        logger.info(f"Rollout: {world_size} ranks × {args.G} ep/rank = {world_size * args.G} ep/iter; "
                    f"group_size={args.group_size} → ~{args.G // max(args.group_size, 1)} groups/rank")

    # Optimizer — no critic params
    param_groups = [{"params": actor.parameters(), "lr": args.lr_actor}]
    if args.lr_encoder > 0:
        param_groups.append({"params": enc_dec.parameters(), "lr": args.lr_encoder})
    else:
        for p in enc_dec.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=1e-8)

    if args.use_wandb and is_main:
        run_name = args.run_name or f"grpo_{args.suite}_task{args.task_id}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "chunk_len": chunk_len,
                           "hidden_dim": hidden_dim, "action_dim": action_dim,
                           "world_size": world_size, "algo": "grpo"})

    video_dir = Path(args.output_dir) / "videos"
    metrics_history = []
    best_sr = 0.0
    best_eval_sr = 0.0
    running_sr = []
    total_env_steps = 0

    # Multi-task fast path: persistent merged-batch step-lock env pool (mirrors
    # the off-policy release rollout). All tasks' envs are created ONCE and
    # stepped in lockstep with a single batched VLA forward — far faster than a
    # naive per-task collect loop that rebuilds envs every iter.
    _mt_pool = None
    if task_list is not None:
        from AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool import PersistentEnvPool
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_rollout_fast import (
            action_token_collect_multitask_steplock,
        )
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        _phys = [int(x) for x in _cvd.split(",") if x.strip()]
        _egl_gpu = _phys[0] if _phys else 0
        _mt_pool = PersistentEnvPool(
            num_envs=args.num_envs * len(task_list),
            libero_python=os.environ.get("LIBERO_PYTHON"),
            egl_gpu_id=_egl_gpu,
        )
        if is_main:
            logger.info(f"[multi-task] step-lock pool: {args.num_envs * len(task_list)} envs "
                        f"({len(task_list)} tasks × {args.num_envs} envs/task)")

    # ── Resume from latest checkpoint of a prior same-named run (if --resume) ──
    start_iter = maybe_resume(
        args, args.run_name, args.output_dir,
        encoder=enc_dec, actor=actor, critic=None,  # GRPO is critic-free (dummy_critic)
        optimizers={"opt": optimizer}, map_location="cpu")

    for iteration in range(start_iter, args.max_iter + 1):
        if is_main:
            logger.info("=" * 60)
            logger.info(f"[iter {iteration}/{args.max_iter}] collecting "
                        f"{args.G}×{world_size}={args.G * world_size} ep")

        save_video = (args.save_video_interval > 0 and
                      (iteration == 1 or iteration % args.save_video_interval == 0))
        iter_video_dir = (str(video_dir / f"iter_{iteration:05d}")
                         if save_video and is_main else None)

        group_seed = args.seed + iteration * 1000 + rank * 100
        if task_list is not None:
            # Merged-batch step-lock over ALL tasks at once (fast path). Auto-chunk
            # into ceil(G/num_envs) passes if G_per_task exceeds per-task env count.
            _n_passes = max(1, (args.G + args.num_envs - 1) // args.num_envs)
            _G_pass = min(args.G, args.num_envs)
            local_episodes = []
            for _p in range(_n_passes):
                local_episodes += action_token_collect_multitask_steplock(
                    env_pool=_mt_pool, frozen_vla=frozen_vla, encoder=enc_dec,
                    actor=actor, critic=dummy_critic, suite_name=args.suite,
                    task_ids=task_list, n_initial_states=50,
                    action_norm_stats=action_norm_stats, max_steps=max_steps,
                    chunk_len=chunk_len, G_per_task=_G_pass,
                    seed=group_seed + _p * 50000, num_steps_wait=args.num_steps_wait,
                    device=str(device), group_idx=(iteration * 100 + _p),
                    group_size=args.group_size, reward_coef=args.reward_coef,
                    encoder_mode=encoder_mode,
                )
            task_id = task_list[0]  # for logging
        else:
            task_id = args.task_id if args.task_id >= 0 else random.randint(0, n_tasks - 1)
            local_episodes = action_token_collect_group(
                frozen_vla=frozen_vla, encoder=enc_dec, actor=actor, critic=dummy_critic,
                suite_name=args.suite, task_id=task_id,
                n_initial_states=50,
                action_norm_stats=action_norm_stats,
                max_steps=max_steps, chunk_len=chunk_len, G=args.G,
                libero_python=os.environ.get("LIBERO_PYTHON"),
                seed=group_seed,
                num_steps_wait=args.num_steps_wait,
                device=str(device), video_dir=iter_video_dir,
                num_envs=args.num_envs,
                group_idx=iteration * world_size + rank,
                group_size=args.group_size,
                reward_coef=args.reward_coef,
                encoder_mode=encoder_mode,
            )

        local_rewards = torch.tensor([ep.reward for ep in local_episodes],
                                     device=device, dtype=torch.float32)
        global_rewards = accelerator.gather(local_rewards).cpu().numpy()
        success_rate = float(np.mean(global_rewards > 0.5))
        mean_reward = float(np.mean(global_rewards))
        mean_steps = float(np.mean([ep.finish_step for ep in local_episodes]))
        local_env_steps = torch.tensor(sum(ep.env_steps for ep in local_episodes),
                                       device=device, dtype=torch.long)
        global_env_steps = accelerator.reduce(local_env_steps, reduction="sum").item()
        total_env_steps += int(global_env_steps)
        running_sr.append(success_rate)
        if len(running_sr) > 20:
            running_sr.pop(0)
        running_sr_avg = float(np.mean(running_sr))
        best_sr = max(best_sr, success_rate)

        if is_main:
            logger.info(f"[iter {iteration}] SR={success_rate:.2f} (best={best_sr:.2f}, "
                        f"avg={running_sr_avg:.2f}) reward={mean_reward:.2f} "
                        f"steps={mean_steps:.1f} ({len(global_rewards)} ep)")

        # ── BC warm-start (cold-start fix; first iteration only) ──────
        # A fresh on-policy actor outputs ≈0 → sparse reward gives no
        # signal. BC-pretrain it to ≈VLA on iter-1's rollout data before
        # any GRPO update, then re-anchor the KL reference to the warmed
        # actor (else KL fights the warm-up by pulling toward random init).
        bc_warmup_steps = getattr(args, "bc_warmup_steps", 0)
        if iteration == 1 and args.beta > 0.0 and bc_warmup_steps > 0:
            if is_main:
                logger.info(f"[iter 1] BC warm-start: {bc_warmup_steps} steps "
                            f"(fresh actor → ≈VLA before GRPO)")
            bc_res = _bc_warmstart_actor(actor, local_episodes,
                                         n_steps=bc_warmup_steps, lr=1e-3,
                                         device=str(device))
            _ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
            if _ddp:
                for p in actor.parameters():
                    torch.distributed.broadcast(p.data, src=0)
            ref_actor.load_state_dict(actor.state_dict())
            if is_main and bc_res is not None:
                logger.info(f"[iter 1] BC warm-start done: "
                            f"‖μ−ã‖² {bc_res[0]:.3f} → {bc_res[1]:.3f}")

        # ── GRPO update ─────────────────────────────────────────
        if is_main:
            logger.info(f"[iter {iteration}] GRPO update ({args.ppo_epochs} epochs, "
                        f"kl_coef={args.grpo_kl_coef})")
        actor.train()
        if args.lr_encoder > 0:
            enc_dec.train()

        epoch_stats = []
        for grpo_epoch in range(args.ppo_epochs):
            optimizer.zero_grad()
            loss, stats = action_token_grpo_loss(
                encoder=enc_dec, actor=actor, ref_actor=ref_actor,
                episodes=local_episodes,
                clip_eps=args.clip_eps, kl_coef=args.grpo_kl_coef,
                bc_coef=args.beta,
                device=str(device),
            )
            loss.backward()
            _ddp_active = torch.distributed.is_available() and torch.distributed.is_initialized()
            if _ddp_active:
                for p in actor.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
                if args.lr_encoder > 0:
                    for p in enc_dec.parameters():
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            if args.max_grad_norm > 0:
                all_params = list(actor.parameters())
                if args.lr_encoder > 0:
                    all_params += list(enc_dec.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            optimizer.step()
            epoch_stats.append(stats)

        # Update reference actor periodically (snapshot of current actor)
        ref_update = getattr(args, "ref_update_interval", 0)
        if ref_update > 0 and iteration % ref_update == 0:
            if is_main:
                logger.info(f"[iter {iteration}] refreshing reference actor")
            ref_actor.load_state_dict(actor.state_dict())

        # ── Eval ────────────────────────────────────────────────
        eval_sr = None
        eval_result = None
        do_eval = (args.eval_interval > 0 and iteration % args.eval_interval == 0)
        if do_eval:
            if is_main:
                logger.info(f"[iter {iteration}] distributed eval ({args.eval_n_episodes} ep)")
            eval_video_dir = str(video_dir / f"eval_iter_{iteration:05d}") if save_video else None
            _eval_tasks = task_list if task_list is not None else [task_id]
            _eval_srs = {}
            for _t in _eval_tasks:
                _r = _eval_distributed(
                    accelerator=accelerator, frozen_vla=frozen_vla,
                    encoder=enc_dec, actor=actor,
                    suite_name=args.suite, task_id=_t,
                    action_norm_stats=action_norm_stats,
                    max_steps=max_steps, chunk_len=chunk_len,
                    n_episodes=args.eval_n_episodes,
                    num_steps_wait=args.num_steps_wait, seed=args.seed,
                    device=str(device),
                    video_dir=(eval_video_dir if _t == _eval_tasks[0] else None),
                    encoder_mode=encoder_mode,
                )
                if is_main and _r:
                    _eval_srs[_t] = _r["eval_sr"]
            if is_main and _eval_srs:
                eval_sr = float(np.mean(list(_eval_srs.values())))  # all-task mean
                best_eval_sr = max(best_eval_sr, eval_sr)
                if len(_eval_srs) > 1:
                    logger.info(f"  [eval] all-task mean SR={eval_sr:.2%} "
                                f"(best={best_eval_sr:.2%}) | " +
                                " ".join(f"t{t}:{s:.2f}" for t, s in sorted(_eval_srs.items())))
                else:
                    logger.info(f"  [eval] SR={eval_sr:.2%} (best={best_eval_sr:.2%})")

        if iteration % args.log_interval == 0 and is_main:
            avg = lambda k: float(np.mean([s[k] for s in epoch_stats if k in s]))
            entry = {
                "iter": iteration, "total_env_steps": total_env_steps,
                "success_rate": success_rate, "best_success_rate": best_sr,
                "running_avg_sr": running_sr_avg, "mean_reward": mean_reward,
                "loss": avg("loss"), "pg_loss": avg("pg_loss"), "kl": avg("kl"),
                "bc_penalty": avg("bc_penalty"),
                "ratio_mean": avg("ratio_mean"), "clip_frac": avg("clip_frac"),
                "advantage_mean": avg("advantage_mean"),
                "advantage_std": avg("advantage_std"),
                "n_groups_with_signal": avg("n_groups_with_signal"),
                "n_steps": avg("n_steps"),
            }
            if eval_sr is not None:
                entry["eval_sr"] = eval_sr
                entry["best_eval_sr"] = best_eval_sr
            metrics_history.append(entry)
            logger.info(f"  loss={entry['loss']:.4f} pg={entry['pg_loss']:.4f} "
                        f"kl={entry['kl']:.4f} bc={entry['bc_penalty']:.3f} ratio={entry['ratio_mean']:.3f} "
                        f"clip_frac={entry['clip_frac']:.3f} "
                        f"groups_signal={entry['n_groups_with_signal']:.1f}")

            if args.use_wandb:
                wandb_log = {
                    "rollout/success_rate": success_rate,
                    "rollout/best_success_rate": best_sr,
                    "rollout/running_avg_sr": running_sr_avg,
                    "rollout/mean_reward": mean_reward,
                    "rollout/total_env_steps": total_env_steps,
                    "rollout/iter_env_steps": int(global_env_steps),
                    "train/loss": entry["loss"], "train/pg_loss": entry["pg_loss"],
                    "train/kl": entry["kl"], "train/bc_penalty": entry["bc_penalty"],
                    "train/ratio_mean": entry["ratio_mean"],
                    "train/clip_frac": entry["clip_frac"],
                    "train/advantage_mean": entry["advantage_mean"],
                    "train/advantage_std": entry["advantage_std"],
                    "train/n_groups_with_signal": entry["n_groups_with_signal"],
                    "train/n_steps": entry["n_steps"],
                }
                if eval_sr is not None:
                    wandb_log["eval/success_rate"] = eval_sr
                    wandb_log["eval/best_success_rate"] = best_eval_sr
                wandb.log(wandb_log, step=iteration)

        if iteration % args.save_interval == 0 and is_main:
            save_rlt_checkpoint(enc_dec, actor, dummy_critic,
                                iteration, args.output_dir, phase="grpo",
                                optimizers={"opt": optimizer})
        accelerator.wait_for_everyone()

    if is_main:
        save_rlt_checkpoint(enc_dec, actor, dummy_critic,
                            args.max_iter, args.output_dir, phase="grpo",
                            optimizers={"opt": optimizer})
        metrics_path = Path(args.output_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        logger.info(f"Done. Metrics -> {metrics_path}")

    if args.use_wandb and is_main:
        wandb.finish()
