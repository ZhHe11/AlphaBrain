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
    kl_coef: float = 0.04,
    micro_batch: int = 2,
    device: str = "cuda",
) -> Tuple[torch.Tensor, dict]:
    """Compute GRPO loss summed over all transitions in the rollout.

    Returns: (loss, stats). loss is a scalar tensor with grad on the
    current policy's VLA parameters.
    """
    # ── 1. Group-relative episode-level advantages ───────────────────
    groups = defaultdict(list)
    for ep_idx, ep in enumerate(episodes):
        groups[ep.state_idx].append((ep_idx, ep))

    ep_advantages = [0.0] * len(episodes)
    for _state_idx, group in groups.items():
        rewards = [ep.reward for _, ep in group]
        if len(rewards) < 2:
            continue  # single-ep group → no relative signal
        mu = sum(rewards) / len(rewards)
        sigma = max((sum((r - mu) ** 2 for r in rewards) / len(rewards)) ** 0.5, 1e-8)
        for (ep_idx, ep) in group:
            ep_advantages[ep_idx] = (ep.reward - mu) / sigma

    # ── 2. Flatten transitions; broadcast ep adv to each of its steps ──
    flat_images: List = []
    flat_instrs: List[str] = []
    flat_actions: List[torch.Tensor] = []
    flat_old_lp: List[float] = []
    flat_advantages: List[float] = []

    for ep_idx, ep in enumerate(episodes):
        adv = ep_advantages[ep_idx]
        for t in range(ep.finish_step):
            sr = ep.step_records[t]
            flat_images.append(sr.images)
            flat_instrs.append(sr.instruction)
            flat_actions.append(sr.action_taken)
            flat_old_lp.append(sr.old_log_prob)
            flat_advantages.append(adv)

    N = len(flat_actions)
    if N == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"n_steps": 0}

    actions_t = torch.stack(flat_actions).to(device)
    old_lp_t = torch.tensor(flat_old_lp, device=device, dtype=torch.float32)
    adv_t = torch.tensor(flat_advantages, device=device, dtype=torch.float32)

    # ── 3. Mini-batched re-forward (current + ref VLA) ───────────────
    total_pg = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    n_batches = 0
    sum_ratio = 0.0
    sum_clip = 0.0

    for start in range(0, N, micro_batch):
        end = min(start + micro_batch, N)
        idx = list(range(start, end))

        batch_images = [flat_images[i] for i in idx]
        batch_instrs = [flat_instrs[i] for i in idx]
        batch_actions = actions_t[start:end]

        # current policy (with grad)
        new_mean = policy.forward_mean(batch_images, batch_instrs)
        new_lp = policy.log_prob_of_with_mean(new_mean, batch_actions)

        # reference policy (no grad)
        with torch.no_grad():
            ref_mean = ref_policy.forward_mean(batch_images, batch_instrs)
            ref_lp = ref_policy.log_prob_of_with_mean(ref_mean, batch_actions)

        old_lp_b = old_lp_t[start:end]
        adv_b = adv_t[start:end]

        ratio = torch.exp(new_lp - old_lp_b)
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
        pg = -torch.min(surr1, surr2).mean()

        # KL(π || π_ref) via k3 estimator: exp(ref - new) - (ref - new) - 1
        log_ratio_ref = ref_lp - new_lp
        kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()

        total_pg = total_pg + pg
        total_kl = total_kl + kl
        n_batches += 1
        sum_ratio += float(ratio.mean().detach().item())
        sum_clip += float(((ratio - 1.0).abs() > clip_eps).float().mean().detach().item())

    pg_loss = total_pg / max(n_batches, 1)
    kl_loss = total_kl / max(n_batches, 1)
    loss = pg_loss + kl_coef * kl_loss

    stats = {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl": kl_loss.item(),
        "ratio_mean": sum_ratio / max(n_batches, 1),
        "clip_frac": sum_clip / max(n_batches, 1),
        "advantage_mean": float(adv_t.mean().item()),
        "advantage_std": float(adv_t.std().item()) if adv_t.numel() > 1 else 0.0,
        "n_groups": len(groups),
        "n_groups_with_signal": sum(1 for _s, g in groups.items() if len(g) >= 2),
        "n_steps": N,
        "n_batches": n_batches,
    }
    return loss, stats
