#!/usr/bin/env python3
"""Efficiency / convergence curves for the RL chapter. TWO figures.

fig6a_rlt_routes.png  -- the full 2x3 grid: encoder {RLT full-token, RLT_a
    action-token} x algorithm {TD3, PPO, GRPO}, single GPU, frozen 4B VLA.
    Encoding: COLOR = algorithm (TD3 navy / PPO green / GRPO amber),
              LINESTYLE = encoder (RLT solid-o / RLT_a dashed-^).
    All curves start at iter 0 = 0.0 -- the residual actor is untrained at
    init (knows nothing), so SR ~ 0; the curve shows learning from a cold
    start. RLT_a+TD3 retained no intermediate checkpoints, so it appears as a
    single end-point marker (0.92, from the released all-task eval).
    End-point error bars = multi-seed spread where n>=2.

fig6b_rlt_vs_vla.png  -- best RLT route (RLT_a+PPO, 1 GPU, ~0.8M trained
    params) vs full-VLA fine-tuning (VLA+PPO / VLA+GRPO, 4B params). RLT
    starts at 0 yet overtakes the full-VLA baselines, which start from the
    0.704 SFT head-start and then stagnate / decay.

Run: python AlphaBrain/training/reinforcement_learning/report_figures/plot_efficiency_curve.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))

# color = algorithm (STYLE_GUIDE brand palette)
ALGO_C = {"TD3": "#143C5C", "PPO": "#5DBB46", "GRPO": "#E8A33D"}
GRID = [0, 50, 100, 150, 200, 250, 300]
BASE_VLA = 0.704   # SFT base = pre-RL anchor for the full-VLA baselines


def _overall(path):
    try:
        d = json.load(open(path))
        x = d[-1] if isinstance(d, list) else d
        if x.get("overall_sr") is not None:
            return float(x["overall_sr"])
        pt = x.get("per_task_sr") or x.get("per_task")
        if isinstance(pt, dict) and pt:
            return sum(float(v) for v in pt.values()) / len(pt)
    except Exception:
        pass
    return None


def _glob_curve(pattern, anchor300=None):
    pts = {0: 0.0}
    for f in glob.glob(os.path.join(ROOT, pattern)):
        it = int("".join(ch for ch in os.path.basename(f).split("iter")[-1] if ch.isdigit()))
        sr = _overall(f)
        if sr is not None:
            pts[it] = sr
    if anchor300 is not None:
        v = _overall(os.path.join(ROOT, anchor300))
        if v is not None:
            pts[300] = v
    return pts


def _td3_curve():
    pts = {0: 0.0}
    for f in glob.glob(os.path.join(
            ROOT, "results/rlt_training/rlt_td3_multitask_0602_1433/rl_offpolicy/eval_iter_*/summary.json")):
        it = int("".join(ch for ch in os.path.basename(os.path.dirname(f)) if ch.isdigit()))
        if it in GRID:
            sr = _overall(f)
            if sr is not None:
                pts[it] = sr
    return pts


def _seed_std(headline, seed_paths):
    vals = [headline]
    for p in seed_paths:
        v = _overall(os.path.join(ROOT, p))
        if v is not None:
            vals.append(v)
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
    return mean, std, n


# ---- the six RLT routes ----
def routes():
    return {
        "RLT+TD3":    dict(algo="TD3",  enc="RLT",   pts=_td3_curve()),
        "RLT+PPO":    dict(algo="PPO",  enc="RLT",   pts=_glob_curve(
            "results/eval_effcurve_0608/rlt_ppo_iter*.json", "results/eval_p0_50ep_0529/mt_rlt_ppo.json")),
        "RLT+GRPO":   dict(algo="GRPO", enc="RLT",   pts=_glob_curve(
            "results/eval_effcurve_0608/rlt_grpo_iter*.json", "results/eval_p0_50ep_0529/mt_rlt_grpo.json")),
        "RLT_a+PPO":  dict(algo="PPO",  enc="RLT_a", pts=_glob_curve(
            "results/eval_effcurve_0608/rlta_ppo_iter*.json", "results/eval_p0_50ep_0529/mt_rlta_ppo.json")),
        "RLT_a+GRPO": dict(algo="GRPO", enc="RLT_a", pts=_glob_curve(
            "results/eval_effcurve_0608/rlta_grpo_iter*.json", "results/eval_p0_50ep_0529/mt_rlta_grpo.json")),
        # no intermediate checkpoints -> end-point only
        "RLT_a+TD3":  dict(algo="TD3",  enc="RLT_a", pts={0: 0.0, 300: 0.920}, endpoint_only=True),
    }


STD = {
    "RLT_a+PPO": _seed_std(0.936, ["results/eval_mt_seeds_0607/rlt_a_ppo_qwen_alltasks_s43v2.json",
                                   "results/eval_mt_seeds_0607/rlt_a_ppo_qwen_alltasks_s44v2.json"]),
    "RLT+PPO":   _seed_std(0.916, ["results/eval_mt_seeds_0607/rlt_ppo_qwen_alltasks_s44v2.json"]),
    "RLT+GRPO":  _seed_std(0.720, ["results/eval_mt_seeds_0607/rlt_grpo_qwen_alltasks_s44v2.json"]),
}


def _axfmt(ax):
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Offline 50-ep all-task success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-5, 312)
    ax.set_xticks(GRID)
    ax.grid(ls=":", alpha=0.5)


def fig_grid():
    R = routes()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    print("== fig6a (2x3 grid) ==")
    for name, r in R.items():
        pts = r["pts"]
        color = ALGO_C[r["algo"]]
        solid = r["enc"] == "RLT"
        ls = "-" if solid else "--"
        mk = "o" if solid else "^"
        its = sorted(pts)
        srs = [pts[i] for i in its]
        if r.get("endpoint_only"):
            ax.plot([its[-1]], [srs[-1]], marker="*", color=color, markersize=15,
                    markeredgecolor="white", markeredgewidth=0.8, linestyle="none",
                    label=f"{name}  (final only; no per-iter ckpt)")
            print(f"  {name}: endpoint {srs[-1]:.3f} (no curve)")
            continue
        ax.plot(its, srs, ls + mk, color=color, lw=2.0, markersize=5.5, alpha=0.95,
                label=f"{name}")
        if name in STD and STD[name][1] > 0:
            ax.errorbar([its[-1]], [srs[-1]], yerr=[STD[name][1]], fmt="none",
                        ecolor=color, elinewidth=1.6, capsize=4, alpha=0.85)
        tag = f"   end={STD[name][0]:.3f}+/-{STD[name][1]:.3f} (n={STD[name][2]})" if name in STD else ""
        print(f"  {name}: " + " ".join(f"{i}:{pts[i]:.3f}" for i in its) + tag)
    _axfmt(ax)
    ax.set_title("Frozen-VLA bottleneck RL: encoder x algorithm convergence (cold start)\n"
                 "single GPU, frozen 4B VLA, ~0.8M trained params  |  "
                 "solid=RLT (full-token), dashed=RLT_a (action-token)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    fig.tight_layout()
    out = os.path.join(HERE, "fig6a_rlt_routes.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_vs_vla():
    R = routes()
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    print("== fig6b (RLT vs full-VLA) ==")
    # best RLT route
    best = R["RLT_a+PPO"]["pts"]
    its = sorted(best); ax.plot(its, [best[i] for i in its], "-o", color="#16A89B",
                                lw=2.4, markersize=5.5, label="RLT_a+PPO  (RLT, 1 GPU)")
    if STD["RLT_a+PPO"][1] > 0:
        ax.errorbar([its[-1]], [best[its[-1]]], yerr=[STD["RLT_a+PPO"][1]], fmt="none",
                    ecolor="#16A89B", elinewidth=1.8, capsize=4)
    print("  RLT_a+PPO:", " ".join(f"{i}:{best[i]:.3f}" for i in its))
    # full-VLA baselines (start at SFT base)
    for name, col in [("VLA+PPO", "#E8A33D"), ("VLA+GRPO", "#6B7280")]:
        tag = "ppo" if name.endswith("PPO") else "grpo"
        pts = {0: BASE_VLA}
        for f in glob.glob(os.path.join(ROOT, f"results/eval_vla_curve_0607/{tag}_iter*.json")):
            it = int("".join(ch for ch in os.path.basename(f).split("iter")[-1] if ch.isdigit()))
            sr = _overall(f)
            if sr is not None:
                pts[it] = sr
        xs = sorted(pts)
        ax.plot(xs, [pts[i] for i in xs], "--s", color=col, lw=1.6, markersize=5,
                alpha=0.8, label=f"{name}  (full-VLA finetune, ours, small-scale)")
        print(f"  {name}:", " ".join(f"{i}:{pts[i]:.3f}" for i in xs))
    ax.axhline(BASE_VLA, color="#9CA3AF", ls=":", lw=1.0, alpha=0.7)
    ax.text(150, BASE_VLA - 0.05, "SFT base 0.704 (full-VLA RL head-start)",
            fontsize=8, color="#6B7280", ha="center")
    _axfmt(ax)
    ax.set_title("1-GPU frozen-VLA RL vs full-VLA RL\n"
                 "RLT learns from 0 and overtakes full fine-tuning's 0.70 head-start", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "fig6b_rlt_vs_vla.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def main():
    fig_grid()
    fig_vs_vla()


if __name__ == "__main__":
    main()
