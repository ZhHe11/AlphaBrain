"""Checkpoint save/resume helpers shared by all RLT training phases.

Resume (added 2026-06-10): crashes (OOM-kill / EGL / preemption) used to throw
away all training progress. Now a run can be relaunched with the same RUN_NAME
and ``--resume``; it finds the latest checkpoint across that run's dir(s), loads
encoder/actor/critic weights (+ optimizer state if present) and continues from
the next iteration instead of restarting at iter 0.
"""
import glob
import logging
import os
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_rlt_checkpoint(encoder, actor, critic, iteration, output_dir, phase="rl",
                        optimizers=None):
    """Save a checkpoint. ``optimizers`` is an optional dict {name: optimizer};
    their state_dicts are bundled into optim.pt so ``--resume`` can restore the
    exact Adam moments (cheap: actor/critic optimizers are a few MB)."""
    ckpt_dir = Path(output_dir) / "checkpoints" / f"{phase}_iter_{iteration:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), str(ckpt_dir / "encoder.pt"))
    if actor is not None:
        torch.save(actor.state_dict(), str(ckpt_dir / "actor.pt"))
    if critic is not None:
        torch.save(critic.state_dict(), str(ckpt_dir / "critic.pt"))
    if optimizers:
        torch.save({k: o.state_dict() for k, o in optimizers.items() if o is not None},
                   str(ckpt_dir / "optim.pt"))
    logger.info(f"Saved RLT checkpoint -> {ckpt_dir}")
    return str(ckpt_dir)


def find_latest_rlt_checkpoint(run_name, base="results/rlt_training", exclude_dir=None):
    """Find the highest-iteration checkpoint across all run dirs whose name starts
    with ``run_name`` (reruns get a fresh timestamp suffix, so a crashed run's
    progress lives under an older sibling dir). Returns (ckpt_dir, iteration) or
    (None, 0). A checkpoint counts only if it has actor.pt (i.e. fully written).

    ``exclude_dir``: skip checkpoints under this dir (the current run's own output)
    so a fresh relaunch resumes from the PREVIOUS attempt, not from itself.
    """
    if not run_name:
        return (None, 0)
    best_dir, best_iter = None, 0
    pattern = os.path.join(base, f"{run_name}_*", "*", "checkpoints", "*_iter_*")
    excl = os.path.abspath(exclude_dir) if exclude_dir else None
    for d in glob.glob(pattern):
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "actor.pt")):
            continue
        if excl and os.path.abspath(d).startswith(excl):
            continue
        m = re.search(r"_iter_(\d+)$", os.path.basename(d))
        if not m:
            continue
        it = int(m.group(1))
        if it > best_iter:
            best_dir, best_iter = d, it
    return (best_dir, best_iter)


def load_rlt_checkpoint(ckpt_dir, encoder=None, actor=None, critic=None,
                        optimizers=None, map_location="cpu", strict=True):
    """Load weights (+ optimizer state if optim.pt present) into the given modules
    in place. Returns the checkpoint's iteration number."""
    p = Path(ckpt_dir)
    def _load(mod, fname):
        f = p / fname
        if mod is not None and f.exists():
            mod.load_state_dict(torch.load(str(f), map_location=map_location), strict=strict)
            return True
        return False
    _load(encoder, "encoder.pt")
    _load(actor, "actor.pt")
    _load(critic, "critic.pt")
    if optimizers and (p / "optim.pt").exists():
        states = torch.load(str(p / "optim.pt"), map_location=map_location)
        for k, o in optimizers.items():
            if o is not None and k in states:
                try:
                    o.load_state_dict(states[k])
                except Exception as e:  # noqa: BLE001 — optimizer resume is best-effort
                    logger.warning(f"[resume] optimizer '{k}' state load failed ({e}); "
                                   f"continuing with fresh moments")
    m = re.search(r"_iter_(\d+)$", os.path.basename(str(p).rstrip("/")))
    return int(m.group(1)) if m else 0


def maybe_resume(args, run_name, output_dir, encoder=None, actor=None, critic=None,
                 optimizers=None, map_location="cpu"):
    """Convenience wrapper for trainers. If args.resume is set, find the latest
    prior checkpoint for run_name (excluding the current output_dir), load it, and
    return start_iter = last_iter + 1. Otherwise return 1. Never raises on a
    missing/partial checkpoint — falls back to a clean start."""
    if not getattr(args, "resume", False):
        return 1
    try:
        ck, last = find_latest_rlt_checkpoint(run_name, exclude_dir=output_dir)
        if not ck:
            logger.info("[resume] --resume set but no prior checkpoint found; starting fresh")
            return 1
        it = load_rlt_checkpoint(ck, encoder=encoder, actor=actor, critic=critic,
                                 optimizers=optimizers, map_location=map_location)
        logger.info(f"[resume] loaded {ck} (iter {it}); continuing from iter {it + 1}")
        return it + 1
    except Exception as e:  # noqa: BLE001 — resume is best-effort; never block training
        logger.warning(f"[resume] failed ({e}); starting fresh from iter 1")
        return 1
