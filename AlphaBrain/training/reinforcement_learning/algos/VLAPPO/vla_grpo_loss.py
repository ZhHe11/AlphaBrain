"""GRPO loss for vanilla VLA + full FT.

Differs from vla_ppo_loss in three places:
  - advantage = (R_ep - μ_group) / σ_group,  group = same state_idx
    (no GAE, no value function — episode-level signal broadcast to all
    its steps)
  - KL penalty to a frozen reference VLA (k3 estimator, like DeepSeek's GRPO)
  - no value loss

Otherwise reuses the same mini-batched VLA re-forward pattern.
"""
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.distributed as dist

from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_policy import (
    VLAPolicy,
)
from AlphaBrain.training.reinforcement_learning.algos.VLAPPO.vla_ppo_rollout import (
    VLAPPOEpisode,
)


def vla_grpo_loss(
    *,
    policy: VLAPolicy,          # current VLA policy (grad on)
    ref_policy: VLAPolicy,      # frozen reference VLA (no grad)
    episodes: List[VLAPPOEpisode],
    clip_eps: float = 0.2,
    clip_eps_high: float = 0.0, # DAPO clip-higher: asymmetric upper bound (0=symmetric=clip_eps)
    dual_clip_c: float = 0.0,   # DAPO dual-clip: cap on negative-adv loss (0=disabled; DAPO=3.0)
    kl_coef: float = 0.04,
    micro_batch: int = 2,
    device: str = "cuda",
) -> dict:
    """Compute GRPO loss over all transitions and backward() per micro-batch.

    Mirrors vla_ppo_loss: the caller owns optimizer.zero_grad() (before) and
    grad-clip + optimizer.step() (after); this fn does the backward internally
    in FSDP lockstep. Returns a stats dict only (no loss tensor).
    """
    # ── 1. Group-relative episode-level advantages ───────────────────
    # Group key is (task_id, state_idx): the merged multi-task collector
    # flattens all tasks into one episode list, and state_idx only runs 0..49
    # PER TASK, so grouping by state_idx alone would mix episodes of the same
    # init-state index across DIFFERENT tasks → contaminated group baseline.
    groups = defaultdict(list)
    for ep_idx, ep in enumerate(episodes):
        groups[(ep.task_id, ep.state_idx)].append((ep_idx, ep))

    # Success-rate dynamic filter (DAPO/RLinf): a group whose episodes are all
    # success or all failure has σ=0 → zero learning signal. We drop it: its
    # advantages stay 0 AND its transitions are excluded from the loss entirely
    # (no wasted forward passes on uninformative samples).
    ep_advantages = [0.0] * len(episodes)
    ep_has_signal = [False] * len(episodes)
    n_groups_with_signal = 0
    for _key, group in groups.items():
        rewards = [ep.reward for _, ep in group]
        if len(rewards) < 2:
            continue  # single-ep group → no relative signal
        if len(set(rewards)) < 2:
            continue  # all-success or all-fail → σ=0, drop the group
        n_groups_with_signal += 1
        mu = sum(rewards) / len(rewards)
        sigma = max((sum((r - mu) ** 2 for r in rewards) / len(rewards)) ** 0.5, 1e-8)
        for (ep_idx, ep) in group:
            ep_advantages[ep_idx] = (ep.reward - mu) / sigma
            ep_has_signal[ep_idx] = True

    # ── 2. Flatten transitions; broadcast ep adv to each of its steps ──
    # Skip episodes in dropped (zero-signal) groups.
    flat_images: List = []
    flat_instrs: List[str] = []
    flat_actions: List[torch.Tensor] = []
    flat_old_lp: List[float] = []
    flat_advantages: List[float] = []

    for ep_idx, ep in enumerate(episodes):
        if not ep_has_signal[ep_idx]:
            continue
        adv = ep_advantages[ep_idx]
        for t in range(ep.finish_step):
            sr = ep.step_records[t]
            flat_images.append(sr.images)
            flat_instrs.append(sr.instruction)
            flat_actions.append(sr.action_taken)
            flat_old_lp.append(sr.old_log_prob)
            flat_advantages.append(adv)

    N = len(flat_actions)
    empty = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "ratio_mean": 1.0, "clip_frac": 0.0,
             "advantage_mean": 0.0, "advantage_std": 0.0, "n_groups": len(groups),
             "n_groups_with_signal": n_groups_with_signal,
             "n_steps": 0, "n_batches": 0}
    # Distributed FSDP lockstep (mirrors vla_ppo_loss): every rank must fire the
    # SAME sequence of FSDP collectives or NCCL deadlocks. (1) all-reduce
    # MIN(has_data): if ANY rank collected nothing, all bail together. (2)
    # all-reduce MAX(n_batches): every rank loops n_batches_global times; padding
    # iters re-process the last real micro-batch with loss×0 (no grad, but FSDP
    # all-gather/reduce-scatter still fire in lockstep). Per-micro-batch backward
    # (not one big graph) keeps peak memory O(micro_batch).
    if dist.is_initialized():
        _ht = torch.tensor([1 if N > 0 else 0], device=device, dtype=torch.int64)
        dist.all_reduce(_ht, op=dist.ReduceOp.MIN)
        if int(_ht.item()) == 0:
            return empty
    elif N == 0:
        return empty

    actions_t = torch.stack(flat_actions).to(device)
    old_lp_t = torch.tensor(flat_old_lp, device=device, dtype=torch.float32)
    adv_t = torch.tensor(flat_advantages, device=device, dtype=torch.float32)

    n_batches_local = (N + micro_batch - 1) // micro_batch
    if dist.is_initialized():
        _t = torch.tensor([n_batches_local], device=device, dtype=torch.int64)
        dist.all_reduce(_t, op=dist.ReduceOp.MAX)
        n_batches_global = int(_t.item())
    else:
        n_batches_global = n_batches_local
    inv_real = 1.0 / max(n_batches_local, 1)
    last_real_start = max(0, (n_batches_local - 1) * micro_batch)

    # ── 3. Mini-batched re-forward (current + ref VLA), backward per micro-batch
    sum_pg = 0.0
    sum_kl = 0.0
    sum_ratio = 0.0
    sum_clip = 0.0

    for i_mb in range(n_batches_global):
        is_real = i_mb < n_batches_local
        if is_real:
            start = i_mb * micro_batch
            end = min(start + micro_batch, N)
        else:
            start = last_real_start
            end = N
        idx = list(range(start, end))

        batch_images = [flat_images[j] for j in idx]
        batch_instrs = [flat_instrs[j] for j in idx]
        batch_actions = actions_t[start:end]

        new_mean = policy.forward_mean(batch_images, batch_instrs)
        new_lp = policy.log_prob_of_with_mean(new_mean, batch_actions)
        with torch.no_grad():
            ref_mean = ref_policy.forward_mean(batch_images, batch_instrs)
            ref_lp = ref_policy.log_prob_of_with_mean(ref_mean, batch_actions)

        old_lp_b = old_lp_t[start:end]
        adv_b = adv_t[start:end]

        ratio = torch.exp(new_lp - old_lp_b)
        surr1 = ratio * adv_b
        # DAPO clip-higher: decouple lower/upper clip bounds. Upper bound
        # (1+eps_high) > lower (1-eps_low) lets low-prob good actions grow more
        # aggressively. eps_high=0 falls back to the symmetric PPO clip.
        eps_high = clip_eps_high if clip_eps_high > 0 else clip_eps
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + eps_high) * adv_b
        clipped = torch.min(surr1, surr2)
        if dual_clip_c > 1.0:
            # DAPO dual-clip: for negative-advantage samples, lower-bound the
            # objective at dual_clip_c * adv so a single huge ratio can't blow
            # up the update. (For adv>=0, standard min-clip is the cap.)
            dual = dual_clip_c * adv_b
            obj = torch.where(adv_b < 0, torch.max(clipped, dual), clipped)
        else:
            obj = clipped
        pg = -obj.mean()
        log_ratio_ref = ref_lp - new_lp
        kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()

        scale = inv_real if is_real else 0.0   # padding iters contribute 0 grad
        loss_mb = (pg + kl_coef * kl) * scale
        loss_mb.backward()

        if is_real:
            sum_pg += float(pg.detach().item())
            sum_kl += float(kl.detach().item())
            sum_ratio += float(ratio.mean().detach().item())
            sum_clip += float(((ratio - 1.0).abs() > clip_eps).float().mean().detach().item())

    nb = n_batches_local
    return {
        "loss": sum_pg / nb + kl_coef * sum_kl / nb,
        "pg_loss": sum_pg / nb,
        "kl": sum_kl / nb,
        "ratio_mean": sum_ratio / nb,
        "clip_frac": sum_clip / nb,
        "advantage_mean": float(adv_t.mean().item()),
        "advantage_std": float(adv_t.std().item()) if adv_t.numel() > 1 else 0.0,
        "n_groups": len(groups),
        "n_groups_with_signal": n_groups_with_signal,
        "n_steps": N,
        "n_batches": nb,
    }
