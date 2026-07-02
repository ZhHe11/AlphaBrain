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
# per-route distinct colors: hue = algorithm family, dark = RLT / light = RLT_a,
# so all six routes are visually separable while still grouping by algorithm.
ROUTE_C = {
    # Okabe-Ito colour-blind-safe palette (Nature house style); hue = algorithm,
    # dark = RLT (full-token), lighter = RLT_a (action-token).
    "RLT+TD3":    "#0072B2",  # blue
    "RLT_a+TD3":  "#56B4E9",  # sky blue
    "RLT+PPO":    "#009E73",  # bluish green
    "RLT_a+PPO":  "#66C2A5",  # light green
    "RLT+GRPO":   "#D55E00",  # vermillion
    "RLT_a+GRPO": "#E69F00",  # orange
}

# Nature-journal figure aesthetic: sans-serif, hairline axes, no top/right
# spines, outward ticks, frameless legend, 300 dpi.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150,
})
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


# per-route metrics.json -> iter:total_env_steps (for the sample-efficiency axis)
METRICS = {
    "RLT+PPO":    "results/rlt_training/rlt_ppo_qwen_alltasks_0529_1830/rl_ppo/metrics.json",
    "RLT+GRPO":   "results/rlt_training/rlt_grpo_qwen_alltasks_0529_2029/rl_grpo/metrics.json",
    "RLT_a+PPO":  "results/rlt_training/rlt_a_ppo_qwen_alltasks_0529_1829/rl_onpolicy/metrics.json",
    "RLT_a+GRPO": "results/rlt_training/rlt_a_grpo_qwen_alltasks_0529_1551/rl_grpo/metrics.json",
    "RLT+TD3":    "results/rlt_training/rlt_td3_multitask_0602_1433/rl_offpolicy/metrics.json",
}


def _env_steps(route):
    """iter -> total_env_steps from the run's metrics.json."""
    p = os.path.join(ROOT, METRICS.get(route, ""))
    if not os.path.exists(p):
        return {}
    rows = json.load(open(p))
    rows = rows if isinstance(rows, list) else rows.get("history", [])
    m = {}
    for r in rows:
        if isinstance(r, dict) and "iter" in r:
            es = r.get("total_env_steps") or r.get("env_steps")
            if es is not None:
                m[int(r["iter"])] = int(es)
    return m


def _seed_std(headline, seed_paths):
    vals = [headline]
    for p in seed_paths:
        v = _overall(os.path.join(ROOT, p))
        if v is not None and v > 0.05:   # skip broken/empty evals (e.g. EGL-failed -> 0.0)
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
        # per-iter curve from the 0610 re-train (#20), iter50-300; iter300=0.874
        # is #20's own offline 50-ep (consistent with the other routes' iter300,
        # vs the older release run's iter-400 value of 0.92).
        "RLT_a+TD3":  dict(algo="TD3",  enc="RLT_a",
                           pts=_glob_curve("results/eval_effcurve_0610/rlta_td3_iter*.json")),
    }


STD = {
    "RLT_a+PPO": _seed_std(0.936, ["results/eval_mt_seeds_0607/rlt_a_ppo_qwen_alltasks_s43v2.json",
                                   "results/eval_mt_seeds_0607/rlt_a_ppo_qwen_alltasks_s44v2.json"]),
    "RLT+PPO":   _seed_std(0.916, ["results/eval_mt_seeds_0607/rlt_ppo_qwen_alltasks_s43v3.json",
                                   "results/eval_mt_seeds_0607/rlt_ppo_qwen_alltasks_s44v2.json"]),
    "RLT+GRPO":  _seed_std(0.720, ["results/eval_mt_seeds_0607/rlt_grpo_qwen_alltasks_s43v3.json",
                                   "results/eval_mt_seeds_0607/rlt_grpo_qwen_alltasks_s44v2.json"]),
    "RLT_a+GRPO": _seed_std(0.704, ["results/eval_mt_seeds_0607/rlt_a_grpo_qwen_alltasks_s43v3.json",
                                    "results/eval_mt_seeds_0607/rlt_a_grpo_qwen_alltasks_s44v3.json"]),
    "RLT+TD3":   _seed_std(0.830, ["results/eval_mt_seeds_0607/rlt_td3_qwen_alltasks_s44v3.json",
                                   "results/eval_remaining_0612/rlt_td3_s43v4.json"]),
    "RLT_a+TD3": _seed_std(0.874, ["results/eval_remaining_0612/rlt_a_td3_s43v5.json"]),
}


# route display name -> eval_band_seeds file key
NAME2KEY = {"RLT+TD3": "rlt_td3", "RLT_a+TD3": "rlta_td3", "RLT+PPO": "rlt_ppo",
            "RLT_a+PPO": "rlta_ppo", "RLT+GRPO": "rlt_grpo", "RLT_a+GRPO": "rlta_grpo"}


def _band_std_by_iter(name, pts):
    """REAL per-iter std at iters {100,200,300}: std over seed42 (curve) +
    seed43/44 (eval_band_seeds). Returns sorted [(iter, std), ...] (>=2 seeds)."""
    key = NAME2KEY.get(name)
    out = []
    for it in (50, 100, 200, 300):
        vals = []
        if pts.get(it) is not None:
            vals.append(pts[it])                       # seed 42 (main curve)
        for s in (43, 44):
            v = _overall(os.path.join(ROOT, f"results/eval_band_seeds/{key}_s{s}_iter{it}.json"))
            if v is not None and v > 0.05:             # skip broken/empty evals
                vals.append(v)
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            out.append((it, (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5))
    return out


def _interp_std(band, x):
    """std at x by clamp + linear interpolation over measured (iter, std) points."""
    if not band:
        return 0.0
    xs = [i for i, _ in band]; ss = [s for _, s in band]
    if x <= xs[0]:
        return ss[0]
    if x >= xs[-1]:
        return ss[-1]
    for k in range(1, len(xs)):
        if x <= xs[k]:
            t = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
            return ss[k - 1] + t * (ss[k] - ss[k - 1])
    return ss[-1]


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
        color = ROUTE_C[name]
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
            sd = STD[name][1]
            lo = [max(0.0, s - sd) for s in srs]
            hi = [min(1.0, s + sd) for s in srs]
            ax.fill_between(its, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)
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
        sd = STD["RLT_a+PPO"][1]
        bs = [best[i] for i in its]
        ax.fill_between(its, [max(0.0, b - sd) for b in bs], [min(1.0, b + sd) for b in bs],
                        color="#16A89B", alpha=0.15, linewidth=0, zorder=1)
    print("  RLT_a+PPO:", " ".join(f"{i}:{best[i]:.3f}" for i in its))
    # full-VLA small-scale baselines (start at SFT base)
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
                alpha=0.72, label=f"{name} small-scale: {pts[xs[-1]]:.3f} final")
        print(f"  {name}:", " ".join(f"{i}:{pts[i]:.3f}" for i in xs))
    # Our trained large-scale/full-VLA checkpoints. Only draw checkpoints that
    # have offline 50-episode evals; do not connect them back to the SFT base.
    scaled_points = [
        ("VLA+PPO scaled FSDP-6 (iter300): 0.922",
         "results/eval_remaining_0612/vla_ppo_scale_iter300.json", 300, "#5B4BAA", "*", 125),
        ("VLA+GRPO tok+temp FSDP-5 (iter150): 0.836",
         "results/eval_remaining_0612/grpo_fsdp_toktemp_iter150/summary.json", 150, "#D55E00", "D", 70),
        ("VLA+GRPO tok+temp 1 GPU (iter300): 0.796",
         "results/eval_remaining_0612/grpo_toktemp_iter300/summary.json", 300, "#F28E2B", "D", 58),
    ]
    for label, rel, x, col, marker, size in scaled_points:
        y = _overall(os.path.join(ROOT, rel))
        if y is None:
            print("  missing", label, rel)
            continue
        ax.scatter([x], [y], s=size, marker=marker, color=col, edgecolor="white",
                   linewidth=0.6, zorder=5, label=label)
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(4, 6), textcoords="offset points",
                    fontsize=7.0, color=col, ha="left", va="bottom")
        print(f"  {label} {x}:{y:.3f}")
    ax.axhline(BASE_VLA, color="#9CA3AF", ls=":", lw=1.0, alpha=0.7)
    ax.text(138, BASE_VLA - 0.055, "QwenOFT SFT base 0.704",
            fontsize=8, color="#6B7280", ha="center")
    ax.annotate("our large-scale\nfull-VLA checkpoints", xy=(300, 0.922), xytext=(205, 0.985),
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.9),
                fontsize=8.2, ha="left", va="top", color="#333")
    _axfmt(ax)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-5, 330)
    ax.set_title("Frozen-VLA RLT vs our full-VLA RL: small-scale curves plus trained checkpoints", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=7.0, frameon=True)
    fig.tight_layout()
    out = os.path.join(HERE, "fig6b_rlt_vs_vla.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_combined():
    """Merge the two all-task multitask curves into one 2-panel figure:
    (a) the six encoder x algorithm RLT routes; (b) best RLT vs full-VLA."""
    R = routes()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.4))
    print("== fig6_combined ==")
    # ---- Panel (a): six RLT routes, one distinct color each ----
    for name, r in R.items():
        pts = r["pts"]; color = ROUTE_C[name]
        solid = r["enc"] == "RLT"
        ls, mk = ("-", "o") if solid else ("--", "^")
        its = sorted(pts); srs = [pts[i] for i in its]
        ax1.plot(its, srs, ls + mk, color=color, lw=2.0, markersize=5.5, alpha=0.95, label=name)
        if name in STD and STD[name][1] > 0:
            sd = STD[name][1]
            ax1.fill_between(its, [max(0.0, s - sd) for s in srs], [min(1.0, s + sd) for s in srs],
                             color=color, alpha=0.15, linewidth=0, zorder=1)
            print(f"  (a) {name}: end {srs[-1]:.3f} band +/-{sd:.3f} (n={STD[name][2]})")
        else:
            print(f"  (a) {name}: end {srs[-1]:.3f} (no band)")
    _axfmt(ax1)
    ax1.set_title("(a) Encoder $\\times$ algorithm convergence (cold start, 1 GPU)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=8.5, ncol=2)
    # ---- Panel (b): best RLT vs full-VLA baselines ----
    best = R["RLT_a+PPO"]["pts"]; its = sorted(best); bs = [best[i] for i in its]
    ax2.plot(its, bs, "-o", color="#16A89B", lw=2.4, markersize=5.5, label="RLT$_a$+PPO  (RLT, 1 GPU)")
    if STD["RLT_a+PPO"][1] > 0:
        sd = STD["RLT_a+PPO"][1]
        ax2.fill_between(its, [max(0.0, b - sd) for b in bs], [min(1.0, b + sd) for b in bs],
                         color="#16A89B", alpha=0.15, linewidth=0, zorder=1)
    for name, col in [("VLA+PPO", "#E8A33D"), ("VLA+GRPO", "#6B7280")]:
        tag = "ppo" if name.endswith("PPO") else "grpo"
        pts = {0: BASE_VLA}
        for f in glob.glob(os.path.join(ROOT, f"results/eval_vla_curve_0607/{tag}_iter*.json")):
            it = int("".join(c for c in os.path.basename(f).split("iter")[-1] if c.isdigit()))
            sr = _overall(f)
            if sr is not None:
                pts[it] = sr
        xs = sorted(pts)
        ax2.plot(xs, [pts[i] for i in xs], "--s", color=col, lw=1.6, markersize=5,
                 alpha=0.8, label=f"{name}  (full-VLA, small-scale)")
    ax2.axhline(BASE_VLA, color="#9CA3AF", ls=":", lw=1.0, alpha=0.7)
    ax2.text(150, BASE_VLA - 0.05, "SFT base 0.704", fontsize=8, color="#6B7280", ha="center")
    _axfmt(ax2)
    ax2.set_title("(b) 1-GPU frozen-VLA RL vs full-VLA fine-tuning", fontsize=11)
    ax2.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Frozen-VLA bottleneck RL on LIBERO-Goal (all-10-task)  |  "
                 "solid $=$ RLT (full-token), dashed $=$ RLT$_a$ (action-token); shaded $=$ multi-seed std",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig6_combined.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_one():
    """Everything on ONE axes: the six RLT routes + the full-VLA baselines,
    all all-task SR vs iteration. Single plot (not panels, not two figures)."""
    R = routes()
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    print("== fig_one (single merged axes) ==")
    for name, r in R.items():
        pts = r["pts"]; color = ROUTE_C[name]
        solid = r["enc"] == "RLT"
        ls, mk = ("-", "o") if solid else ("--", "^")
        its = sorted(pts); srs = [pts[i] for i in its]
        ax.plot(its, srs, ls + mk, color=color, lw=1.6, markersize=3.6,
                markeredgewidth=0, alpha=0.97, label=name, zorder=3)
        # REAL per-iter std band (varying width) from seed42/43/44 at iter 100/200/300.
        band = _band_std_by_iter(name, pts)
        if band:
            lo = [max(0.0, s - _interp_std(band, i)) for i, s in zip(its, srs)]
            hi = [min(1.0, s + _interp_std(band, i)) for i, s in zip(its, srs)]
            ax.fill_between(its, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=1)
            print(f"  band {name}: " + " ".join(f"{it}:±{sd:.3f}" for it, sd in band))
    # full-VLA baselines (start at the SFT base 0.704)
    for nm, col in [("VLA+PPO", "#6A51A3"), ("VLA+GRPO", "#737373")]:
        tag = "ppo" if nm.endswith("PPO") else "grpo"
        pts = {0: BASE_VLA}
        for f in glob.glob(os.path.join(ROOT, f"results/eval_vla_curve_0607/{tag}_iter*.json")):
            it = int("".join(c for c in os.path.basename(f).split("iter")[-1] if c.isdigit()))
            sr = _overall(f)
            if sr is not None:
                pts[it] = sr
        xs = sorted(pts)
        ax.plot(xs, [pts[i] for i in xs], ":s", color=col, lw=1.2, markersize=3.0,
                alpha=0.85, label=f"{nm} (full-VLA)", zorder=2)
    ax.axhline(BASE_VLA, color="#9CA3AF", ls=(0, (4, 3)), lw=0.8, alpha=0.7)
    ax.text(305, BASE_VLA + 0.010, "SFT base 0.704", fontsize=7.0, color="#6B7280", ha="right")
    _axfmt(ax)
    ax.grid(False)
    ax.grid(axis="y", color="#000000", lw=0.4, alpha=0.10, zorder=0)
    ax.legend(loc="lower right", ncol=2, handlelength=1.9, columnspacing=1.0,
              borderaxespad=0.3, labelspacing=0.35)
    fig.tight_layout()
    out = os.path.join(HERE, "fig6_one.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def fig_panels():
    """Nature small multiples split by ENCODER: (a) RLT full-token, (b) RLT_a
    action-token; each panel overlays the three algorithms (TD3/PPO/GRPO) with
    per-iter std bands. Comparing the two panels shows the encoder swap barely
    moves the curves; within a panel the algorithm ordering (PPO>GRPO~TD3) is
    clear. The RLT-vs-full-VLA efficiency point lives in its own figure."""
    R = routes()
    ALGO_OI = {"TD3": "#0072B2", "PPO": "#009E73", "GRPO": "#D55E00"}  # Okabe-Ito
    ALGO_MK = {"TD3": "o", "PPO": "^", "GRPO": "s"}
    panels = [("a", "RLT (full-token)", "RLT", ["RLT+TD3", "RLT+PPO", "RLT+GRPO"]),
              ("b", "RLT$_a$ (action-token)", "RLT_a", ["RLT_a+TD3", "RLT_a+PPO", "RLT_a+GRPO"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)
    print("== fig_panels (by encoder) ==")
    for ax, (lab, title, enc, names) in zip(axes, panels):
        for name in names:
            r = R[name]; pts = r["pts"]; algo = r["algo"]
            color = ALGO_OI[algo]; mk = ALGO_MK[algo]
            its = sorted(pts); srs = [pts[i] for i in its]
            ax.plot(its, srs, "-" + mk, color=color, lw=1.7, markersize=3.6,
                    markeredgewidth=0, alpha=0.97, label=algo, zorder=3)
            band = _band_std_by_iter(name, pts)
            if band:
                lo = [max(0.0, s - _interp_std(band, i)) for i, s in zip(its, srs)]
                hi = [min(1.0, s + _interp_std(band, i)) for i, s in zip(its, srs)]
                ax.fill_between(its, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.axhline(BASE_VLA, color="#CCCCCC", ls=":", lw=0.8, zorder=0)
        ax.text(307, BASE_VLA + 0.012, "SFT base", fontsize=6.8, color="#999999", ha="right")
        ax.set_title(f"({lab}) {title}", fontsize=9.5, loc="left")
        ax.set_xlim(-5, 312); ax.set_ylim(0.0, 1.0); ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        ax.set_xlabel("Training iteration")
        ax.grid(axis="y", color="#000000", lw=0.4, alpha=0.10, zorder=0)
        ax.legend(loc="lower right", fontsize=7.8, handlelength=1.7, labelspacing=0.35)
    axes[0].set_ylabel("Offline 50-ep all-task SR")
    fig.tight_layout()
    out = os.path.join(HERE, "fig6_panels.png")
    fig.savefig(out, dpi=300); plt.close(fig); print("wrote", out)


def fig_sample_efficiency():
    """SR vs cumulative environment steps -- the proper sample-efficiency axis
    that makes off-policy (TD3) and on-policy (PPO/GRPO) comparable."""
    R = routes()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    print("== fig6c (sample efficiency: SR vs env-steps) ==")
    ax.axhline(0.80, color="#9CA3AF", ls=":", lw=1.0, alpha=0.8)
    ax.text(0.2, 0.815, "80% SR", fontsize=8, color="#6B7280")
    for name, r in R.items():
        if r.get("endpoint_only"):
            continue
        es = _env_steps(name)
        if not es:
            continue
        color = ALGO_C[r["algo"]]
        solid = r["enc"] == "RLT"
        ls, mk = ("-", "o") if solid else ("--", "^")
        # pair each evaluated iter with its cumulative env-steps (anchor at 0,0)
        xy = [(0.0, 0.0)]
        for it in sorted(r["pts"]):
            if it == 0:
                continue
            if it in es:
                xy.append((es[it] / 1e6, r["pts"][it]))
        if len(xy) < 2:
            continue
        xs, ys = zip(*xy)
        ax.plot(xs, ys, ls + mk, color=color, lw=2.0, markersize=5.5, alpha=0.95, label=name)
        # env-steps to first reach 80%
        hit = next((x for x, y in xy if y >= 0.80), None)
        tag = f"   ->80% at {hit:.1f}M" if hit else "   never reaches 80%"
        print(f"  {name}: " + " ".join(f"{x:.1f}M:{y:.3f}" for x, y in xy) + tag)
    ax.set_xlabel("Cumulative environment steps (millions)")
    ax.set_ylabel("Offline 50-ep all-task success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.3, 15.2)
    ax.grid(ls=":", alpha=0.5)
    ax.set_title("Sample efficiency: success rate vs cumulative environment steps\n"
                 "PPO crosses 80% at ${\\sim}0.5$M steps; TD3 needs ${\\sim}20\\times$ more "
                 "(${\\sim}10$M); GRPO stays below 80%", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    out = os.path.join(HERE, "fig6c_sample_efficiency.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def main():
    fig_grid()
    fig_vs_vla()
    fig_one()
    fig_panels()
    # fig_sample_efficiency() removed 2026-06-10: the env-step axis unfairly
    # penalizes off-policy TD3 (it collects ~5.7x more env-steps/iter to fill its
    # replay buffer), making the "sample efficiency" claim a collection-rate
    # artifact rather than a data-efficiency result. Iteration-axis fig6a is the
    # clean comparison. Function kept for reference but no longer emitted.


if __name__ == "__main__":
    main()
