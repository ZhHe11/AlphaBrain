#!/usr/bin/env python3
"""Figures for the RL technical report.

Regenerate whenever new experiments land — edit the DATA block below
(or later wire it to read results/*/summary.json) and re-run:

    python AlphaBrain/training/reinforcement_learning/report_figures/plot_rl_report.py

PNGs are written next to this script. Series / cells with no data yet
are drawn as explicit "TBD" placeholders so the report layout is
visible before every number is in. All figure labels are English on
purpose (avoids the matplotlib CJK-font issue); report prose stays zh.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

C_TD3   = "#3b6fb0"   # RLT + TD3   (the only mature cell)
C_GRPO  = "#c0654a"   # RLT + GRPO
C_PPO   = "#5a9367"   # RLT + PPO
C_WEAK  = "#c0654a"   # highlight colour for the weak task
C_TBD   = "0.82"      # placeholder grey

# ─────────────────────────────────────────────────────────────────────
# DATA — fill in as experiments complete.  None  ==  not measured yet.
# ─────────────────────────────────────────────────────────────────────

# Base VLA (pre-RL) — per-task SR.  source: results/evaluation/libero_goal/*/results.json
# NOTE: provisional — these existing evals use a slightly different protocol
# from the RL eval (1traj: seed42/max_steps512/steps_10000; 5traj:
# seed0/max_steps320/steps_20000). A clean re-eval (T1) supersedes them.
BASE_VLA_5TRAJ = {0: 0.52, 1: 0.92, 2: 0.96, 3: 0.50, 4: 0.82,
                  5: 0.44, 6: 0.34, 7: 0.94, 8: 0.88, 9: 0.54}   # overall 0.686
BASE_VLA_1TRAJ = {0: 0.32, 1: 0.82, 2: 0.02, 3: 0.22, 4: 0.70,
                  5: 0.12, 6: 0.10, 7: 0.88, 8: 0.06, 9: 0.02}   # overall 0.326

# RLT + TD3 — per-task offline eval SR (50-ep).
# source: results/eval_rlt_release_0415/summary.json
RLT_TD3_1TRAJ = {0: 0.86, 1: 0.96, 3: 0.08}          # QwenOFT-1traj, single-task runs
RLT_TD3_5TRAJ = {0: 1.00, 1: 1.00, 2: 0.94, 3: 0.68,  # QwenOFT-5traj, all-10-task run
                 4: 1.00, 5: 0.86, 6: 0.80, 7: 1.00,
                 8: 1.00, 9: 0.92}                     # overall 0.92

# RLT + TD3 — online SR training curve, libero_goal task 0 (20-ep in-loop eval).
# source: results/eval_rlt_release_0415/woshare_t0_iters/summary.json
RLT_TD3_T0_CURVE = {25: 0.36,  50: 0.26,  75: 0.58, 100: 0.48,
                    125: 0.64, 150: 0.84, 175: 0.86, 200: 0.84,
                    225: 0.74, 250: 0.84, 275: 0.74, 300: 0.92}
# Placeholders — fill once the runs unlocked this session finish.
RLT_GRPO_T0_CURVE = {}   # RLT + GRPO  (code unlocked; run pending)
RLT_PPO_T0_CURVE  = {}   # RLT + PPO   (pending E2)

# Headline SR per method on libero_goal (overall, QwenOFT).  None = TBD.
# RLT+TD3 cell uses the 5-traj all-10-task overall (0.92).
MAIN_RESULTS = {
    "Base VLA\n(pre-RL)":     0.686,  # QwenOFT-5traj base, libero_goal (provisional, see note above)
    "RLT + TD3":              0.92,
    "RLT + GRPO":             None,   # unlocked this session — run pending
    "RLT + PPO":              None,   # pending E2 (small-actor PPO)
    "RLT_a + TD3":            None,
    "VLA + PPO\n(baseline)":  None,   # pending E3 (run to convergence)
    "VLA + GRPO\n(baseline)": None,   # pending E3
}


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def fig_main_results():
    """Bar — SR by method. Known cells solid; unknown drawn as hatched TBD."""
    methods = list(MAIN_RESULTS)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, m in enumerate(methods):
        v = MAIN_RESULTS[m]
        if v is None:
            ax.bar(i, 1.0, color=C_TBD, hatch="//", edgecolor="0.6", alpha=0.55)
            ax.text(i, 0.50, "TBD", ha="center", va="center",
                    fontsize=10, color="0.35", rotation=90)
        else:
            ax.bar(i, v, color=C_TD3)
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Success rate")
    ax.set_title("Main results — success rate by method on libero_goal\n"
                 "(QwenOFT; RLT+TD3 cell = 5-traj all-task overall)", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(HERE, "fig1_main_results.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_training_curve():
    """Line — phase-2 training curve. RLT+TD3 real; GRPO/PPO placeholders."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    it = sorted(RLT_TD3_T0_CURVE)
    ax.plot(it, [RLT_TD3_T0_CURVE[i] for i in it],
            "-o", color=C_TD3, lw=1.8, label="RLT + TD3")

    for curve, color, mk, name in [
        (RLT_GRPO_T0_CURVE, C_GRPO, "s", "RLT + GRPO"),
        (RLT_PPO_T0_CURVE,  C_PPO,  "^", "RLT + PPO"),
    ]:
        if curve:
            xs = sorted(curve)
            ax.plot(xs, [curve[i] for i in xs], "-" + mk, color=color, lw=1.8, label=name)
        else:
            ax.plot([], [], "--" + mk, color=color, label=f"{name} (pending)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Online success rate (20-ep in-loop eval)")
    ax.set_ylim(0, 1.0)
    ax.set_title("RLT phase-2 training curve — libero_goal task 0 (QwenOFT-1traj)", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "fig2_training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_per_task():
    """Bar — per-task SR for RLT+TD3 (QwenOFT-5traj). Highlights the weak task."""
    tasks = sorted(RLT_TD3_5TRAJ)
    sr = [RLT_TD3_5TRAJ[t] for t in tasks]
    colors = [C_WEAK if t == 3 else C_TD3 for t in tasks]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(tasks, sr, color=colors)
    for t, v in zip(tasks, sr):
        ax.text(t, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    mean_sr = float(np.mean(sr))
    ax.axhline(mean_sr, color="0.4", ls="--", lw=1, label=f"mean = {mean_sr:.2f}")
    ax.set_xticks(tasks)
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Success rate (50-ep)")
    ax.set_ylim(0, 1.24)
    ax.set_title("Per-task SR — RLT+TD3, QwenOFT-5traj all-task run\n"
                 "(task 3 is the weak point; cf. 1-traj task 3 = 0.08)", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "fig3_per_task.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_rl_gain():
    """Grouped bar — base VLA vs RLT+TD3 per task (QwenOFT-5traj). The headline."""
    tasks = sorted(set(BASE_VLA_5TRAJ) & set(RLT_TD3_5TRAJ))
    base = [BASE_VLA_5TRAJ[t] for t in tasks]
    rlt = [RLT_TD3_5TRAJ[t] for t in tasks]
    x = np.arange(len(tasks))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar(x - w / 2, base, w, color="0.62", label=f"Base VLA  (overall {np.mean(base):.2f})")
    ax.bar(x + w / 2, rlt,  w, color=C_TD3,  label=f"RLT + TD3  (overall {np.mean(rlt):.2f})")
    for i, (b, r) in enumerate(zip(base, rlt)):
        d = (r - b) * 100
        ax.text(i, max(b, r) + 0.02, f"{d:+.0f}", ha="center", fontsize=8,
                color="#2e7d32" if d >= 0 else "#c62828")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{t}" for t in tasks])
    ax.set_xlabel("Task")
    ax.set_ylabel("Success rate (50-ep)")
    ax.set_ylim(0, 1.20)
    ax.set_title("RL gain — base VLA vs RLT+TD3 per task (QwenOFT-5traj, libero_goal)\n"
                 "labels = SR points gained by RL", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(HERE, "fig4_rl_gain.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (fig_main_results, fig_training_curve, fig_per_task, fig_rl_gain):
        print("wrote", fn())
