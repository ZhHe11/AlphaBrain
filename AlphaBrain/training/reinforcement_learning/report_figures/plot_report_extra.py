#!/usr/bin/env python3
"""Extra report figures: (fig7) main-results grouped bar, (fig8) RLT_a parameter
efficiency. Brand palette per STYLE_GUIDE. Numbers from the multi-seed table /
checkpoint param counts (see RL_REPORT_TABLES.md, tab:rlt_main / tab:rlt_encoder).

Run: python AlphaBrain/training/reinforcement_learning/report_figures/plot_report_extra.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ALGO_C = {"TD3": "#143C5C", "PPO": "#5DBB46", "GRPO": "#E8A33D", "base": "#9CA3AF", "VLA": "#6B7280"}


def fig_main_bar():
    """All-task (10-task) SR per method, grouped; error bars where n>=2."""
    # (label, algo, mean, std_or_None)
    # point estimates (match tab:rlt_main); std lives only in the convergence
    # curve, so no error bars here.
    rows = [
        ("Base VLA", "base", 0.704, None),
        ("RLT+TD3", "TD3", 0.830, None),
        ("RLT+GRPO", "GRPO", 0.786, None),
        ("RLT+PPO", "PPO", 0.953, None),
        ("RLT$_a$+TD3", "TD3", 0.920, None),
        ("RLT$_a$+GRPO", "GRPO", 0.753, None),
        ("RLT$_a$+PPO", "PPO", 0.951, None),
        ("VLA+PPO\n(small)", "VLA", 0.718, None),
        ("VLA+GRPO\n(small)", "VLA", 0.756, None),
        ("VLA+PPO\n(scaled)", "VLA", 0.922, None),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    xs = range(len(rows))
    for i, (lab, algo, m, sd) in enumerate(rows):
        hatch = "//" if lab.startswith("RLT$_a$") else None
        ax.bar(i, m, color=ALGO_C[algo], width=0.74, alpha=0.95,
               yerr=sd, capsize=4 if sd else 0,
               edgecolor="white", hatch=hatch, linewidth=0.6)
        ax.text(i, m + (sd or 0) + 0.012, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8, rotation=0)
    ax.axhline(0.704, color="#9CA3AF", ls=":", lw=1.0, alpha=0.8)
    ax.text(8.4, 0.704 + 0.008, "base 0.704", fontsize=7.5, color="#6B7280", ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Offline 50-ep all-task success rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Main results (LIBERO-Goal, all-10 tasks): PPO routes lead; "
                 "RLT$_a$ (hatched) matches RLT at far fewer params", fontsize=10.5)
    # legend for algo colors
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=ALGO_C[a], label=a) for a in ["PPO", "GRPO", "TD3"]] + \
          [Patch(facecolor="#6B7280", label="VLA baseline"),
           Patch(facecolor="white", edgecolor="#444", hatch="//", label="RLT$_a$ (action-token)")]
    ax.legend(handles=leg, loc="lower left", fontsize=8, ncol=2)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(HERE, "fig7_main_bar.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_rlta_efficiency():
    """RLT vs RLT_a: trainable params (bars) with all-task PPO SR annotated."""
    names = ["RLT\n(full-token, bn2048)", "RLT$_a$\n(action-token, bn256)"]
    params = [2.68, 0.85]          # M trainable (actor+critic)
    sr = [0.953, 0.951]            # all-task PPO, 3-seed
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    bars = ax.bar(names, params, color=["#143C5C", "#16A89B"], width=0.6, alpha=0.95,
                  edgecolor="white", linewidth=0.8)
    for b, p, s in zip(bars, params, sr):
        ax.text(b.get_x() + b.get_width()/2, p + 0.06, f"{p:.2f}M params",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, p/2, f"SR {s:.3f}\n(PPO, 3-seed)",
                ha="center", va="center", fontsize=9, color="white")
    ax.annotate("", xy=(1, 0.95), xytext=(0, 2.78),
                arrowprops=dict(arrowstyle="->", color="#E8A33D", lw=2))
    ax.text(0.5, 2.0, "3.2$\\times$ fewer\ntrainable params\n(SR tied)",
            ha="center", fontsize=9.5, color="#B5740A", fontweight="bold")
    ax.set_ylabel("Trainable parameters (actor + critic), millions")
    ax.set_ylim(0, 3.1)
    ax.set_title("RLT$_a$ compactness: same accuracy,\n${\\sim}3\\times$ fewer trained parameters",
                 fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(HERE, "fig8_rlta_efficiency.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_architecture():
    """Method overview: frozen VLA + frozen encoder -> bottleneck token -> tiny
    trained actor/critic. Drawn as a block diagram (PNG, no TikZ risk)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.set_xlim(0, 102); ax.set_ylim(0, 34); ax.axis("off")
    FROZEN = "#DCE9F2"; FROZEN_E = "#143C5C"; TRAIN = "#D6F0CB"; TRAIN_E = "#3E8E2E"; IO = "#F0F0F0"
    def box(x, y, w, h, text, fc, ec, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                                     fc=fc, ec=ec, lw=1.6))
        ax.text(x+w/2, y+h/2 + (1.4 if sub else 0), text, ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        if sub:
            ax.text(x+w/2, y+h/2-2.4, sub, ha="center", va="center", fontsize=7.8, color="#444")
    def arrow(x0, x1, y=17):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.8, color="#555"))
    box(1, 11, 13, 12, "Image +\ninstruction", IO, "#999")
    box(17, 11, 16, 12, "Frozen VLA", FROZEN, FROZEN_E, "4B params (no grad)")
    box(36, 11, 16, 12, "Light encoder", FROZEN, FROZEN_E, "frozen after Phase-1")
    box(55, 11, 14, 12, "Bottleneck\ntoken $z_{rl}$", "#EAF2F5", "#16A89B", "dim 256 / 2048")
    box(72, 11, 16, 12, "Actor / Critic", TRAIN, TRAIN_E, "TRAINED  ~0.8M")
    box(91, 11, 10, 12, "Action\nchunk", IO, "#999")
    for x0, x1 in [(14,17),(33,36),(52,55),(69,72),(88,91)]:
        arrow(x0, x1)
    # phase annotations
    ax.add_patch(FancyBboxPatch((17, 4), 35, 4.5, boxstyle="round,pad=0.2",
                                 fc="none", ec="#143C5C", ls="--", lw=1.0))
    ax.text(34.5, 6.2, "Phase 1: imitation-pretrain encoder, then freeze",
            ha="center", fontsize=8, color="#143C5C")
    ax.add_patch(FancyBboxPatch((55, 4), 33, 4.5, boxstyle="round,pad=0.2",
                                 fc="none", ec="#3E8E2E", ls="--", lw=1.0))
    ax.text(71.5, 6.2, "Phase 2: RL only here (TD3 / GRPO / PPO), single GPU",
            ha="center", fontsize=8, color="#3E8E2E")
    ax.text(51, 31, "RLT bottleneck: freeze the VLA, run RL on a tiny actor/critic over a compact token",
            ha="center", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "fig0_architecture.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    fig_architecture()
    fig_main_bar()
    fig_rlta_efficiency()
