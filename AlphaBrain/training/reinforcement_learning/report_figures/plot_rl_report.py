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

# Base VLA (pre-RL) — per-task SR. T1 clean re-eval (offline 50-ep, seed42,
# max_steps320, final_model). source: RL_REPORT_TABLES.md 表 1b-①.
BASE_VLA_5TRAJ = {0: 0.78, 1: 0.92, 2: 0.98, 3: 0.42, 4: 0.84,
                  5: 0.34, 6: 0.34, 7: 1.00, 8: 0.90, 9: 0.52}   # overall 0.704
BASE_VLA_1TRAJ = {0: 0.96, 1: 0.96, 2: 0.18, 3: 0.06, 4: 0.96,
                  5: 0.04, 6: 0.38, 7: 1.00, 8: 0.92, 9: 0.02}   # 1traj_alltasks_v3 (RL after); base 1traj overall 0.346

# RLT_a + TD3 release — per-task offline eval SR (50-ep). source: 表 1b-①.
RLT_TD3_1TRAJ = {0: 0.86, 1: 0.96, 3: 0.08}          # QwenOFT-1traj, single-task runs
RLT_TD3_5TRAJ = {0: 1.00, 1: 1.00, 2: 0.94, 3: 0.68,  # RLT_a+TD3 release, all-10-task
                 4: 1.00, 5: 0.86, 6: 0.80, 7: 1.00,
                 8: 1.00, 9: 0.92}                     # overall 0.92

# Pi0.5-5traj multitask RLT+TD3 — per-task (0606). source: 表 4. overall 0.868
PI05_TD3_5TRAJ = {0: 0.94, 1: 1.00, 2: 0.94, 3: 0.48, 4: 1.00,
                  5: 0.96, 6: 0.78, 7: 0.96, 8: 0.98, 9: 0.64}   # overall 0.868

# RLT + TD3 — online SR training curve, libero_goal task 0 (20-ep in-loop eval).
RLT_TD3_T0_CURVE = {25: 0.36,  50: 0.26,  75: 0.58, 100: 0.48,
                    125: 0.64, 150: 0.84, 175: 0.86, 200: 0.84,
                    225: 0.74, 250: 0.84, 275: 0.74, 300: 0.92}
RLT_GRPO_T0_CURVE = {}
RLT_PPO_T0_CURVE  = {}

# Headline SR per method on libero_goal 全10任务 (overall, QwenOFT-5traj,
# 离线 50-ep, iter300). source: RL_REPORT_TABLES.md 表 1.
MAIN_RESULTS = {
    "Base VLA\n(pre-RL)":     0.704,
    "RLT + TD3":              0.830,
    "RLT + GRPO":             0.720,
    "RLT + PPO":              0.916,
    "RLT_a + TD3":            0.920,
    "RLT_a + GRPO":           0.704,
    "RLT_a + PPO":            0.936,
    "VLA + PPO\n(baseline)":  0.718,
    "VLA + GRPO\n(baseline)": 0.756,
}


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def fig_main_results():
    """Bar — SR by method. Known cells solid; unknown drawn as hatched TBD."""
    methods = list(MAIN_RESULTS)
    # colour by family: base=grey, RLT=blue, RLT_a=green, VLA baseline=orange
    def _fam_color(m):
        if "Base" in m: return "0.55"
        if m.startswith("RLT_a"): return C_PPO
        if m.startswith("RLT"):   return C_TD3
        return C_GRPO            # VLA baselines
    best = max((m for m in methods if MAIN_RESULTS[m] is not None),
               key=lambda m: MAIN_RESULTS[m])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, m in enumerate(methods):
        v = MAIN_RESULTS[m]
        if v is None:
            ax.bar(i, 1.0, color=C_TBD, hatch="//", edgecolor="0.6", alpha=0.55)
            ax.text(i, 0.50, "TBD", ha="center", va="center",
                    fontsize=10, color="0.35", rotation=90)
        else:
            ax.bar(i, v, color=_fam_color(m),
                   edgecolor=("gold" if m == best else "none"),
                   lw=(2.5 if m == best else 0))
            ax.text(i, v + 0.02, f"{v:.3f}" + ("  *" if m == best else ""),
                    ha="center", va="bottom", fontsize=9,
                    fontweight=("bold" if m == best else "normal"))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Success rate (all-10-task, offline 50-ep)")
    ax.set_title("Main results — all-task overall SR by method (QwenOFT-5traj, libero_goal)\n"
                 "grey=base, blue=RLT, green=RLT_a, orange=VLA-baseline, star=best", fontsize=11)
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
