#!/usr/bin/env python3
"""fig5 — VLA-baseline offline-50-ep process curve (SR vs training iter).

Reads results/eval_vla_curve_0607/{ppo,grpo}_iter*.json (each = a full
10-task 50-ep offline eval of one VLA-baseline checkpoint) and plots the
offline success-rate trajectory for VLA+PPO and VLA+GRPO. The point is to
show the *instability* of full-VLA fine-tuning (RQ-B): unlike the RLT
bottleneck route, the full-VLA curve swings across iterations.

Run:  python AlphaBrain/training/reinforcement_learning/report_figures/plot_vla_curve.py
PNG -> next to this script (fig5_vla_curve.png). Missing iters are skipped.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
CURVE_DIR = os.path.join(ROOT, "results/eval_vla_curve_0607")

C = {"ppo": "#5a9367", "grpo": "#c0654a"}


def load_curve(algo):
    pts = {}
    for f in glob.glob(os.path.join(CURVE_DIR, f"{algo}_iter*.json")):
        it = int(os.path.basename(f).split("iter")[1].split(".")[0])
        try:
            d = json.load(open(f))
            x = d[-1] if isinstance(d, list) else d
            sr = x.get("overall_sr")
            if sr is not None:
                pts[it] = sr
        except Exception:
            continue
    return dict(sorted(pts.items()))


def main():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    summary = []
    for algo in ("ppo", "grpo"):
        pts = load_curve(algo)
        if not pts:
            continue
        its = list(pts)
        srs = [pts[i] for i in its]
        rng = max(srs) - min(srs)
        ax.plot(its, srs, "-o", color=C[algo], lw=1.9,
                label=f"VLA + {algo.upper()}  (range {min(srs):.2f}–{max(srs):.2f}, Δ{rng:.2f})")
        summary.append(f"VLA+{algo.upper()}: {len(pts)} pts, term={srs[-1]:.3f}, "
                       f"peak={max(srs):.3f}, swing={rng:.3f}")
    # RLT reference band (stable terminal values, for contrast)
    ax.axhspan(0.916, 0.936, color="#3b6fb0", alpha=0.12,
               label="RLT route terminal band (0.916–0.936)")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Offline 50-ep all-task SR")
    ax.set_ylim(0, 1.0)
    ax.set_title("VLA-baseline process curve — full-VLA fine-tuning is unstable\n"
                 "(offline 50-ep at each ckpt; cf. RLT route's flat high band)", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "fig5_vla_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    for s in summary:
        print(" ", s)


if __name__ == "__main__":
    main()
