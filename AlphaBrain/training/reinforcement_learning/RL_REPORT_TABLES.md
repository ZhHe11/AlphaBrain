# RL 技术报告 — 表格

> 配套:`TECHNICAL_REPORT.md`(代码层)、`RL_PAPER_WRITING_GUIDE.md`(写作+实验)、`report_figures/`(配图)。
> 本文是报告正文要用的表格,**已测数据填入、未测留空**。

## 说明

- **评测协议**:离线 50-ep(除非另注);成功率 = LIBERO reward ≥ 0.5。
- **填充图例**:数值 = 已测;`—` = 待测/待跑(括注依赖项)。
- **数据来源**:`results/eval_rlt_release_0415/summary.json`、`results/eval_rlt_release_0415/woshare_t0_iters/summary.json`。
- **配图**:`report_figures/fig1_main_results.png`、`fig2_training_curve.png`、`fig3_per_task.png`(由 `plot_rl_report.py` 生成,数据更新后重跑即可)。
- ⚠️ **基座 SR 不要跨模型借数**(PI0.5 ≠ QwenOFT)——每行"基座 VLA"必须用该行 RL 实际加载的 ckpt 自测。

---

## 表 1 — 主结果:RL 前后成功率(libero_goal,QwenOFT)

> 离线评测,每任务 50-ep。task0/1/3 + 全 10 任务均值。配图 `fig1_main_results.png`、`fig4_rl_gain.png`。

| 方法 | 编码器 | 基座 | task0 | task1 | task3 | 全 10 任务 | 说明 |
|:-----|:-------|:-----|:------|:------|:------|:-----------|:-----|
| 基座 VLA(RL 前) | — | Qwen-1traj | 0.32 | 0.82 | 0.22 | 0.33 | 现有 eval † |
| 基座 VLA(RL 前) | — | Qwen-5traj | 0.52 | 0.92 | 0.50 | **0.69** | 现有 eval † |
| RLT_a + TD3 | RLT_a | Qwen-1traj | 0.86 | 0.96 | **0.08** | — | 现有 release run(单任务) |
| RLT_a + TD3 | RLT_a | Qwen-5traj | 1.00 | 1.00 | 0.68 | **0.92** | 现有 release run(全任务);逐任务见表 1b |
| RLT + TD3 | RLT | Qwen-5traj | — | — | — | — | 运行中 — 全 token 轨道,本会话新增(GPU2) |
| RLT + GRPO | RLT | Qwen-5traj | — | — | — | — | 运行中 `run_rlt_grpo.sh`(GPU1) |
| VLA + PPO(基线) | — | Qwen-5traj | — | — | — | — | 运行中 `run_qwen_vla_ppo.sh`(GPU4) |
| VLA + GRPO(基线) | — | Qwen-5traj | — | — | — | — | 运行中 `run_qwen_vla_grpo.sh`(GPU5) |
| RLT + PPO | RLT | Qwen | — | — | — | — | ⏸ 阻塞:需 E2 |

**关键对比(5traj)**:基座 0.69 → RLT_a+TD3 **0.92**,RL 净增 **+0.23**。(RLT 全 token 轨道结果运行中)

† 基座数字取自现有 eval,协议与 RL eval 未完全对齐(1traj:seed 42 / max_steps 512 / ckpt `steps_10000`;5traj:seed 0 / max_steps 320 / ckpt `steps_20000`)。**T1 干净重测**(与 RL eval 同协议、同 `final_model` ckpt)后替换。多种子(≥3 seed)后改报 `mean±std`。

---

## 表 1b — 逐任务:基座 vs RLT_a+TD3(QwenOFT-5traj,全 10 任务)

> 全部实测。配图 `fig4_rl_gain.png`。

| 任务 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | **overall** |
|:-----|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:------------|
| 基座 VLA | 0.52 | 0.92 | 0.96 | 0.50 | 0.82 | 0.44 | 0.34 | 0.94 | 0.88 | 0.54 | **0.686** |
| RLT_a + TD3 | 1.00 | 1.00 | 0.94 | 0.68 | 1.00 | 0.86 | 0.80 | 1.00 | 1.00 | 0.92 | **0.920** |
| **Δ** | +0.48 | +0.08 | −0.02 | +0.18 | +0.18 | +0.42 | +0.46 | +0.06 | +0.12 | +0.38 | **+0.234** |

观察:RL 增益集中在基座弱任务(t0/t5/t6/t9,基座 0.34–0.54);基座已强的任务(t1/t2/t7/t8)接近饱和,t2 轻微 −0.02(噪声级)。task3 在 5traj 下 0.50→0.68 仍是最弱任务;在 1traj 下更崩——基座 0.22 → RL **0.08**,RL 反而变差,见 RQ-C 失败分析。

---

## 表 2 — RLT × 不同 RL 算法(报告重点)

> 固定同一 RLT 编码器、相同任务集、相同 env-step 预算、≥3 seed。横轴对齐 env steps。

| 算法 | 类型 | critic | 终态 SR (mean±std) | 样本效率 (env-steps→80%SR) | 稳定性 (seed 方差) | 代码状态 |
|:-----|:-----|:-------|:-------------------|:---------------------------|:-------------------|:---------|
| TD3 | off-policy AC | 双 Q + 目标平滑 | — | — | — | ✅ 可跑 |
| GRPO | on-policy PG | 无(组相对优势) | — | — | — | ✅ 本会话解锁 |
| PPO | on-policy AC | V(s) + GAE | — | — | — | ⚠️ 待 E2 |

说明:这张表是"重点"叙事的承载表——同一 RLT 底座下,off-policy / on-policy、有无 critic、有无 KL 约束的对照。当前仅 TD3 有单种子数据(见表 1),其余待 Tier-1 实验。

---

## 表 3 — 消融(固定 RLT + TD3,libero_goal task0)

| 变体 | 改动 | SR | 相对默认 Δ |
|:-----|:-----|:---|:-----------|
| 默认(RLT + 预训练编码器) | — | — | 0 |
| 编码器随机初始化 | 去 Phase-1 预训练 | — | — |
| RLT_a 编码器 | 全 token → 动作 token 瓶颈 | — | — |
| `β = 0` | 去掉 BC 正则 | — | — |
| `ref_dropout = 0` | 关闭参考动作 dropout | — | — |
| 瓶颈维度 `D` | 128 / 512(仅 RLT_a) | — | — |

注:`β`、`ref_dropout` 等部分变体可从 `results/rlt_training_TD3/`(`*beta3*`、`*groupsize4*`、`*chunk4*` 等 dev run)整理,不必全部重跑。

---

## 表 4 — 跨 VLM 骨干(RLT + TD3,证明仓库多骨干支持)

| 骨干 | task0 | 全 10 任务 | 数据来源 / 备注 |
|:-----|:------|:-----------|:----------------|
| QwenOFT-5traj | 1.00 | 0.92 | 见表 1 |
| Pi05-1traj | — | — | `results/rlt_training/rlt_ori_rl_t0_release_pi05_1traj_*` 有 run,待汇总 |
| Pi05-5traj | — | — | `results/rlt_training/rlt_ori_rl_t0_release_pi05_5traj_*` 有 run,待汇总 |

---

## 表 5 — RLT + TD3 训练曲线(libero_goal task0,QwenOFT-1traj)

> 在线 20-ep 内嵌评测(监控用,方差大;论文图以离线 50-ep 为准)。源:`woshare_t0_iters/summary.json`。配图 `fig2_training_curve.png`。

| iter | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 |
|:-----|:---|:---|:---|:----|:----|:----|:----|:----|:----|:----|:----|:----|
| SR | 0.36 | 0.26 | 0.58 | 0.48 | 0.64 | 0.84 | 0.86 | 0.84 | 0.74 | 0.84 | 0.74 | 0.92 |

注:iter150–300 在 0.74–0.92 间抖动 → model selection 用末 3 个 ckpt 均值,不取峰值(见写作指南陷阱表)。

---

## 待回填清单(对应实验编号见 `RL_PAPER_WRITING_GUIDE.md`)

| 空缺 | 依赖实验 |
|:-----|:---------|
| 表 1「基座 VLA」行 | T1(基座锚定) |
| 表 1 各 RL 行的多种子 mean±std | T3(≥3 seed) |
| 表 2 全部 | RLT×GRPO/PPO 跑通 + E2 |
| 表 2 样本效率列 | T9(样本效率曲线) |
| 表 3 全部 | T7 / T8 / T10(消融) |
| 表 4 Pi05 行 | T5(跨骨干汇总) |
| RLT+TD3 1-traj 全 10 任务 | 补 1-traj all-task 评测 |
