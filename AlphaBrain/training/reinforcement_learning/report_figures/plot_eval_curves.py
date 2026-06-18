#!/usr/bin/env python3
"""In-train eval-SR curves for single-task RLT / RLT_a runs.

Auto-discovers run dirs under results/rlt_training[ _TD3]/, reads each
metrics.json, detects (encoder, algo, task) from the dir name + metric keys,
and plots eval_sr (20-ep in-loop eval) vs iter. One subplot per task.

    python AlphaBrain/training/reinforcement_learning/report_figures/plot_eval_curves.py

PNG -> report_figures/fig_eval_curves.png . No GPU; reads metrics.json only.
Labels in English (matplotlib CJK font).
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

TASKS = [0, 1, 3]
# (encoder, algo) -> colour
STYLE = {
    ("RLT", "TD3"): ("#3b6fb0", "-"),
    ("RLT", "GRPO"): ("#c0654a", "-"),
    ("RLT", "PPO"): ("#5a9367", "-"),
    ("RLT_a", "TD3"): ("#3b6fb0", "--"),
    ("RLT_a", "GRPO"): ("#c0654a", "--"),
    ("RLT_a", "PPO"): ("#5a9367", "--"),
}


def detect_algo(metric_keys):
    k = set(metric_keys)
    if {"td_loss", "critic_loss"} & k or "buffer_size" in k:
        return "TD3"
    if "vf_loss" in k or "vf" in k or "gae" in " ".join(k):
        return "PPO"
    if "n_groups_with_signal" in k or ("kl" in k and "pg_loss" in k):
        return "GRPO"
    if "pg_loss" in k:
        return "PPO"
    return None


def parse_dir(name):
    enc = "RLT_a" if "rlt_a_" in name else "RLT"
    m = re.search(r"_t(\d+)_", name)
    task = int(m.group(1)) if m else None
    return enc, task


def load_runs():
    runs = []
    pats = [
        "results/rlt_training/rlt_*qwen_t*_*",
        "results/rlt_training_TD3/*qwen_t*_*",
    ]
    seen = {}
    for pat in pats:
        for d in glob.glob(os.path.join(ROOT, pat)):
            name = os.path.basename(d)
            enc, task = parse_dir(name)
            if task not in TASKS:
                continue
            inner = sorted(glob.glob(os.path.join(d, "rl_*")))
            mpath = os.path.join(inner[0] if inner else d, "metrics.json")
            if not os.path.isfile(mpath):
                continue
            try:
                data = json.load(open(mpath))
            except Exception:
                continue
            if not isinstance(data, list) or not data:
                continue
            algo = detect_algo(data[0].keys())
            if algo is None:
                continue
            curve = [(e["iter"], e["eval_sr"]) for e in data
                     if e.get("eval_sr") is not None]
            if len(curve) < 2:
                continue
            # keep most-recent run per (enc, algo, task)
            key = (enc, algo, task)
            ts = name.split("_")[-2] + name.split("_")[-1]
            if key not in seen or ts > seen[key]:
                seen[key] = ts
                runs.append((key, curve, name))
    # dedup: keep latest per key
    best = {}
    for key, curve, name in runs:
        ts = name.split("_")[-2] + name.split("_")[-1]
        if key not in best or ts >= best[key][2]:
            best[key] = (curve, name, ts)
    return {k: (v[0], v[1]) for k, v in best.items()}


def main():
    runs = load_runs()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 4.2),
                             sharey=True)
    for ax, task in zip(axes, TASKS):
        any_line = False
        for (enc, algo), (color, ls) in STYLE.items():
            key = (enc, algo, task)
            if key not in runs:
                continue
            curve, name = runs[key]
            xs = [c[0] for c in curve]
            ys = [c[1] for c in curve]
            ax.plot(xs, ys, color=color, linestyle=ls, marker="o", ms=3,
                    label=f"{enc}+{algo}  (final {ys[-1]:.2f})")
            any_line = True
        ax.set_title(f"libero_goal task {task}")
        ax.set_xlabel("iter")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        if not any_line:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="0.6")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("eval SR (20-ep in-loop)")
    fig.suptitle("Single-task in-train eval-SR curves (RLT vs RLT_a x TD3/GRPO/PPO)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_eval_curves.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    print(f"runs plotted: {len(runs)}")
    for (enc, algo, task), (curve, name) in sorted(runs.items()):
        print(f"  {enc:6s}+{algo:4s} t{task}: {len(curve)} pts, "
              f"final={curve[-1][1]:.2f}  ({name})")


if __name__ == "__main__":
    main()
