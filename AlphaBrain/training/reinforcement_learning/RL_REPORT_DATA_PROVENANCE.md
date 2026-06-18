# 报告数据溯源与核验(Data Provenance & Verification)

> 目的:确认技术报告主表(表 1)每个数字都能追溯到真实 eval 产出,不是编造/估计/过时值。
> 核验日期 2026-06-08。方法:对每个 headline SR,定位其源 eval JSON 并比对 `overall_sr`。

## 表 1 全 10 任务 SR 核验

| 方法 | 报告值 | 源文件 | 源 overall_sr | 状态 |
|:--|:--|:--|:--|:--|
| 基座 VLA(5traj) | 0.704 | `results/eval_base_vla/qwen_5traj/summary.json` | 0.704 | ✅ 逐字吻合 |
| RLT + GRPO | 0.720 | `results/eval_p0_50ep_0529/mt_rlt_grpo.json` | 0.72 | ✅ |
| RLT + PPO | 0.916 | `results/eval_p0_50ep_0529/mt_rlt_ppo.json` | 0.916 | ✅ |
| RLT_a + GRPO | 0.704 | `results/eval_p0_50ep_0529/mt_rlta_grpo.json` | 0.704 | ✅ |
| RLT_a + PPO | 0.936 | `results/eval_p0_50ep_0529/mt_rlta_ppo.json` | 0.936 | ✅ |
| RLT_a + PPO (t3) | 0.96 | `results/eval_p0_50ep_0529/rlta_ppo_t3.json` | 0.96 | ✅ |
| VLA + PPO（基线） | 0.718 | `results/eval_vla_mt_50ep_0601/ppo/summary.json` | 0.718 | ✅ |
| VLA + GRPO（基线） | 0.756 | `results/eval_vla_mt_50ep_0601/grpo/summary.json` | 0.756 | ✅ |
| 基座 VLA(1traj,表1c) | 0.346 | `results/eval_base_vla/qwen_1traj/summary.json` | 0.346 | ✅ |
| **RLT + TD3** | **0.830** | `results/eval_verify_0608/rlt_td3_mt_recheck.json`(0608 重评) | **0.834** | ✅ 重评吻合(0.834≈0.830,rounding) |
| **RLT_a + TD3** | **0.920** | `eval_rlt_release_0415/5traj_alltasks_merged.json` | **0.92** | ✅ overall+逐任务吻合 |
| **Pi0.5 + TD3** | **0.868** | `rlt_td3_pi05_5traj_mt_0604_1010/.../eval_all_iters_rlt/all_iters_summary.json` | **0.868** | ✅ overall+逐任务吻合 |

**说明**:9/11 数字逐字吻合源 `summary.json` 的 `overall_sr`。3 个 TD3/Pi05 数字的 in-train eval summary 用别的 key(20-ep 监控),其 0603/0604 的 iter300/400 50-ep 复评以 per-task 形式记录,报告内的逐任务分解平均值精确等于报告 overall(内部自洽),但缺一个独立的 `overall_sr` JSON。**建议:重评这 3 个 ckpt(50-ep)生成干净 JSON 以完成 100% 溯源。**

## 已查出并修正的错误
- **资源元数据(非 SR 结果)**:早前误报 RLT 多任务训练显存 ~18GB(实为 eval 进程快照),实测逐 PID 核实 = **~55–69GB/卡**;可训参数 ~0.8M(actor.pt 1.7MB+critic.pt 1.5MB)。已在 `RLINF_VS_OURS.md` / draft / summary 全部更正。

## 进行中(多种子 n=3,表 1-①)
- seed 43/44 的 12 个 run + eval 收集中,源 `results/eval_mt_seeds_0607/*.json`,每个直接读 `overall_sr`,落地即填表(同 9/11 的逐字核验口径)。

## VLA-baseline 过程曲线(fig5)
- 源 `results/eval_vla_curve_0607/{ppo,grpo}_iter*.json`,每点 = 一次完整 10 任务 50-ep 离线 eval 的 `overall_sr`。PPO 6 点已核(0.746/0.774/0.798/0.782/0.766/0.710),GRPO 收集中。
