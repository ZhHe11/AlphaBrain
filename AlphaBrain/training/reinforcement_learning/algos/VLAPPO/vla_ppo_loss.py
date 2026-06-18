"""GAE + PPO loss for vanilla VLA-PPO.

Key cost driver: each PPO epoch must re-forward the VLA over every
transition in the rollout (because the policy IS the VLA). We mini-batch
to keep peak memory bounded.

Gradient accumulation: each micro-batch re-forwards the VLA, computes its
share of the loss, and immediately ``backward()``s — so only ONE
micro-batch's autograd graph is alive at a time. Peak memory is
O(micro_batch), independent of how many episodes were collected (this is
what lets G scale to RLinf-style values without OOM).
"""
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_policy import (
    VLAPolicy, VLAValueHead,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_ppo_rollout import (
    VLAPPOEpisode,
)


def compute_vla_gae(
    episode: VLAPPOEpisode,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Per-step GAE advantage + return for one episode.

    Sparse-reward convention: episode reward is delivered at the last
    step (terminal). All earlier rewards are 0.
    """
    n = episode.finish_step
    if n == 0:
        return [], []
    values = [sr.value for sr in episode.step_records[:n]]
    rewards = [0.0] * n
    rewards[-1] = episode.reward

    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    next_value = 0.0  # bootstrap from 0 at terminal (sparse, no future reward)
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
    return advantages, returns


def vla_ppo_loss(
    *,
    policy: VLAPolicy,
    value_head: VLAValueHead,
    episodes: List[VLAPPOEpisode],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    micro_batch: int = 4,
    device: str = "cuda",
) -> dict:
    """Compute the PPO loss over all transitions and run ``backward()``.

    Each transition is re-forwarded through the VLA in mini-batches of
    ``micro_batch``. Every micro-batch is backward'd immediately, so its
    autograd graph is freed before the next — peak memory does not grow
    with the number of episodes.

    The caller owns ``optimizer.zero_grad()`` (before) and grad-clip +
    ``optimizer.step()`` (after). Returns a stats dict only — no loss
    tensor (the backward has already happened).
    """
    # ── 1. Flatten transitions + compute GAE per episode ─────────────
    flat_images: List = []
    flat_instrs: List[str] = []
    flat_props: List[torch.Tensor] = []
    flat_actions: List[torch.Tensor] = []
    flat_old_lp: List[float] = []
    flat_old_values: List[float] = []
    flat_advantages: List[float] = []
    flat_returns: List[float] = []

    for ep in episodes:
        adv, ret = compute_vla_gae(ep, gamma, gae_lambda)
        for t in range(ep.finish_step):
            sr = ep.step_records[t]
            flat_images.append(sr.images)
            flat_instrs.append(sr.instruction)
            flat_props.append(sr.prop_state)
            flat_actions.append(sr.action_taken)
            flat_old_lp.append(sr.old_log_prob)
            flat_old_values.append(sr.value)
            flat_advantages.append(adv[t])
            flat_returns.append(ret[t])

    N = len(flat_actions)
    empty = {"loss": 0.0, "pg_loss": 0.0, "vf_loss": 0.0, "ratio_mean": 1.0,
             "clip_frac": 0.0, "advantage_mean": 0.0, "return_mean": 0.0,
             "n_steps": 0, "n_batches": 0}
    # Distributed: every rank must take the same path (return-empty vs run-loop)
    # or FSDP collectives go out of sync. All-reduce MIN(has_data): if ANY
    # rank failed to collect anything, everyone skips this update — that
    # rank can't supply local data to pad FSDP collectives, so the only
    # safe action is for all ranks to bail in lockstep.
    if dist.is_initialized():
        _ht = torch.tensor([1 if N > 0 else 0], device=device, dtype=torch.int64)
        dist.all_reduce(_ht, op=dist.ReduceOp.MIN)
        all_have_data = (int(_ht.item()) == 1)
    else:
        all_have_data = (N > 0)
    if not all_have_data:
        return empty

    actions_t = torch.stack(flat_actions).to(device)             # (N, C, A)
    old_lp_t = torch.tensor(flat_old_lp, device=device, dtype=torch.float32)
    old_values_t = torch.tensor(flat_old_values, device=device, dtype=torch.float32)
    adv_t = torch.tensor(flat_advantages, device=device, dtype=torch.float32)
    ret_t = torch.tensor(flat_returns, device=device, dtype=torch.float32)

    # Normalize advantages globally
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n_batches_local = (N + micro_batch - 1) // micro_batch

    # FSDP-aware: each rank must fire the SAME sequence of FSDP collectives.
    # Variable N (episode lengths differ across ranks) → fast ranks otherwise
    # finish early and call `_allreduce_unwrapped_grads` while slow ranks are
    # still mid-FSDP → NCCL deadlocks on call-order mismatch (this is exactly
    # what RLinf avoids by redistributing rollout data to a fixed per-rank
    # global batch). Fix: all-reduce MAX → every rank loops n_batches_global
    # times; extra iters re-process the last real micro-batch with loss × 0
    # (zero gradient contribution, but FSDP all-gather/reduce-scatter still
    # fire in lockstep).
    if dist.is_initialized():
        _t = torch.tensor([n_batches_local], device=device, dtype=torch.int64)
        dist.all_reduce(_t, op=dist.ReduceOp.MAX)
        n_batches_global = int(_t.item())
    else:
        n_batches_global = n_batches_local

    inv_real = 1.0 / max(n_batches_local, 1)
    last_real_start = max(0, (n_batches_local - 1) * micro_batch)

    # ── 2. Mini-batched VLA re-forward — backward per micro-batch ────
    sum_pg = 0.0
    sum_vf = 0.0
    sum_ratio = 0.0
    sum_clip = 0.0

    for i_mb in range(n_batches_global):
        is_real = i_mb < n_batches_local
        if is_real:
            start = i_mb * micro_batch
            end = min(start + micro_batch, N)
        else:
            # Padding: reuse the last real micro-batch — loss × 0 → no grad
            # contribution; FSDP collectives still fire to keep lockstep.
            start = last_real_start
            end = N
        idx = list(range(start, end))

        batch_images = [flat_images[j] for j in idx]
        batch_instrs = [flat_instrs[j] for j in idx]
        batch_actions = actions_t[start:end]

        # VLA re-forward (with grad)
        new_mean, new_features = policy.forward_mean_and_features(batch_images, batch_instrs)
        new_lp = policy.log_prob_of_with_mean(new_mean, batch_actions)
        new_value = value_head(new_features)

        old_lp_b = old_lp_t[start:end]
        old_v_b = old_values_t[start:end]
        adv_b = adv_t[start:end]
        ret_b = ret_t[start:end]

        ratio = torch.exp(new_lp - old_lp_b)
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
        pg = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        v_clipped = old_v_b + torch.clamp(new_value - old_v_b, -10.0, 10.0)
        vf = torch.max((new_value - ret_b) ** 2, (v_clipped - ret_b) ** 2).mean()

        # Real iters average over n_batches_local; padding iters contribute 0.
        scale = inv_real if is_real else 0.0
        loss_mb = (pg + vf_coef * vf) * scale
        loss_mb.backward()

        if is_real:
            sum_pg += float(pg.detach().item())
            sum_vf += float(vf.detach().item())
            sum_ratio += float(ratio.mean().detach().item())
            sum_clip += float(((ratio - 1.0).abs() > clip_eps).float().mean().detach().item())

    n_batches = n_batches_local  # local count for stats — trainer aggregates

    pg_loss = sum_pg / n_batches
    vf_loss = sum_vf / n_batches
    return {
        "loss": pg_loss + vf_coef * vf_loss,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "ratio_mean": sum_ratio / n_batches,
        "clip_frac": sum_clip / n_batches,
        "advantage_mean": float(adv_t.mean().item()),
        "return_mean": float(ret_t.mean().item()),
        "n_steps": N,
        "n_batches": n_batches,
    }
