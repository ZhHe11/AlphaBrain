# RL 技术报告 — 表格

> 配套:`TECHNICAL_REPORT.md`(代码层)、`RL_PAPER_WRITING_GUIDE.md`(写作+实验)、`report_figures/`(配图)。
> 本文是报告正文要用的表格,**已测数据填入、未测留空**。

## 说明

- **评测协议**:离线 50-ep(除非另注);成功率 = LIBERO reward ≥ 0.5。
- **填充图例**:数值 = 已测;`—` = 待测/待跑(括注依赖项)。
- **eval 口径**:成功率统一取**最终 ckpt 的最后一次评测**(终值),**不取训练峰值、不用均值平滑**;论文级数字以训练后的统一离线 50-ep 复评为准(in-train 20-ep eval 仅监控,方差大)。
- **数据来源**:`results/eval_rlt_release_0415/summary.json`、`results/eval_rlt_release_0415/woshare_t0_iters/summary.json`。
- **配图**:`report_figures/fig1_main_results.png`、`fig2_training_curve.png`、`fig3_per_task.png`(由 `plot_rl_report.py` 生成,数据更新后重跑即可)。
- ⚠️ **基座 SR 不要跨模型借数**(PI0.5 ≠ QwenOFT)——每行"基座 VLA"必须用该行 RL 实际加载的 ckpt 自测。

---

## 表 1 — 主结果(5traj 基座,libero_goal,QwenOFT)

> task0/1/3 + 全 10 任务均值。**基座行 = 训练前**,其余行 = RL 后终值。配图 `fig1_main_results.png`、`fig4_rl_gain.png`。

| 方法 | 编码器 | task0 | task1 | task3 | 全 10 任务 | 说明 |
|:-----|:-------|:------|:------|:------|:-----------|:-----|
| **基座 VLA(训练前)** | — | **0.78** | **0.92** | **0.42** | **0.704** | T1 干净重测(离线 50-ep,seed 42,max_steps 320,`final_model`);全 10 任务逐项见表 1b-① |
| RLT + TD3 | RLT | **0.98** | **0.92** | **0.64** | **0.830** | 离线 50-ep 终值(iter 300,seed 42);run=`rlt_rl_qwen_t{0,1,3}_0521`。全 10 任务 = **0.830**(离线 50-ep,iter300,`rlt_td3_multitask_0602_1433`,0603 完成全 iter 50-ep 复评;per-task t0–t9 = 1.00/.96/.92/.68/.98/.36/.48/1.00/.98/.94,弱点 t5/t6 ≈ .36/.48;末3 ckpt overall .80/.86/.83;前 3 次 0601_1701/1705/1716 反复 OOM/崩，buf300k 配置后跑通) |
| RLT + GRPO | RLT | **0.96** | **0.96** | **0.71±0.06** | **0.720** | 离线 50-ep 终值(iter 300);run=`rlt_grpo_qwen_t{0,1,3}_0522`;冷启动经残差 actor(μ=ã+Δ)修复。**t3 = 0.71±0.06(3-seed 42/43/44 = 0.64/0.74/0.74)**。全 10 任务 = **0.720**(离线 50-ep,iter300,`rlt_grpo_qwen_alltasks_0529_2029`,steplock + `--all_tasks` fix 后重跑;per-task t0–t9 = .68/.94/.98/.46/.84/.34/.32/1.00/.92/.72,弱点 t5/t6 ≈ .33) |
| RLT + PPO | RLT | **0.94** | **0.92** | **0.95±0.01** | **0.916** | 离线 50-ep 终值(iter 300);run=`rlt_ppo_qwen_t{0,1,3}`;**t3 = 0.95±0.01(3-seed 42/43/44 = 0.96/0.94/0.94)是 RLT × 任意 algo 最高 task3,且方差极小**。全 10 任务 = **0.916**(离线 50-ep,iter300,`rlt_ppo_qwen_alltasks_0529_1830`,steplock + fix 后重跑;per-task t0–t9 = .98/.98/.94/.80/.96/.94/.58/1.00/.98/1.00,唯一弱点 t6=.58) |
| RLT_a + TD3 | RLT_a | **0.82** | **0.94** | **0.46** | 0.92‡ | 离线 50-ep 终值(iter 300,seed 42);run=`rlt_a_rl_qwen_t{0,1,3}_0527`;t3=0.46(**task3 难任务 RLT_a 慢半拍**,vs RLT+TD3 0.64)。‡全 10 任务取 **release run 0.92**(表 1b,iter 400,离线 50-ep) |
| RLT_a + GRPO | RLT_a | **0.88** | **0.96** | **0.70** | **0.704** | 离线 50-ep 终值(iter 300,seed 42);**0608 RLT_a 单任务 50-ep 复评**(`results/eval_rlta_single_50ep_0608/grpo_t{0,1,3}.json` = 0.88/0.96/0.70,替换早前 in-train 口径 0.84/0.96/0.64);vs RLT+GRPO(0.96/0.96/0.64):t3 RLT_a 反高 +0.06,t0 RLT_a 低 8pp。全 10 任务 = **0.704**(离线 50-ep,iter300,`rlt_a_grpo_qwen_alltasks_0529_1551`,steplock + `--all_tasks` fix 后重跑;per-task t0–t9 = .76/.94/.96/.50/.80/.26/.34/.98/.92/.58,弱点 t5/t6 ≈ .30) |
| RLT_a + PPO | RLT_a | **1.00** | **0.98** | **0.96** | **0.936** 🏆 | 离线 50-ep 终值(iter 300,seed 42):t0/t1=**1.00/0.98**;t3=**0.96**(离线 50-ep,run `rlt_a_ppo_qwen_t3_0529_1109` iter300;in-train best=1.00,50-ep 复评 0.96)。**全 10 任务 = 0.936(榜首)**(离线 50-ep,iter300,`rlt_a_ppo_qwen_alltasks_0529_1829`,steplock + fix;per-task t0–t9 = .90/.96/.98/.96/.98/.94/.72/1.00/.96/.96,**全表无 <.72 任务**;**反超 RLT_a+TD3 release 0.92**) |
| VLA + PPO(基线) | — | **0.96** | **0.94** | **0.80** | **0.718** | 离线 50-ep 终值(iter 300,seed 42,state_dict overlay 复评,源 `results/eval_vla_singletask_50ep_0601/`);t0 run=`vla_ppo_qwen_t0_0530`(in-train range 80-100,std~7,**抖**);t1 run=`vla_ppo_qwen_t1_0601`;t3 run=`vla_ppo_qwen_t3_0530`(in-train range 35-80,std~12,**严重抖**)。**vs RLT+PPO t3=0.96 → RLT 高 +0.16**;**vs RLT_a+PPO t3=0.96 → RLT_a 高 +0.16**。**全 10 任务 = 0.718**(离线 50-ep,iter300,`vla_ppo_qwen_alltasks_0531_2052`,task-cycling 4/iter;in-train 终值 0.700;per-task t0–t9 = .80/.96/.86/.30/.88/.72/.36/.98/.94/.38,弱点 t3/t6/t9;**< RLT_a+PPO 0.936 / RLT+PPO 0.916,全量微调多任务显著弱于 RLT 路线**) |
| VLA + GRPO(基线) | — | **0.80** | **0.92** | **0.62** | **0.756** | 离线 50-ep 终值(iter 300,seed 42,state_dict overlay 复评,源 `results/eval_vla_singletask_50ep_0601/`);t0 in-train range 80-100;t1 run=`vla_grpo_qwen_t1_0601`;t3 in-train range 20-60。**vs RLT_a+GRPO t3=0.64**:GRPO 在 t3 上两条路线接近(0.62 vs 0.64),但 VLA+GRPO **训练抖动远大于 RLT 路线**(std~14 vs <5),终值非峰值。**全 10 任务 = 0.756**(离线 50-ep,iter300,`vla_grpo_qwen_alltasks_0531_2112`,task-cycling 4/iter;in-train 终值 0.775;per-task t0–t9 = .82/.96/.96/.52/.82/.52/.38/1.00/.92/.66;注:RLT_a+GRPO 0.704 / RLT+GRPO 0.720,**VLA+GRPO 0.756 实际略高于两条 RLT-GRPO 路线**——GRPO 下全量微调不输瓶颈结构) |

> **RLT_a + TD3** 的 **4/14-4/16 release runs**(全任务 + 1traj 单任务,离线 50-ep)单独放在 **表 1b**。新加的 RLT_a 单任务行(下表)用于和 RLT 单任务做 **encoder ablation** 对照(RLT_a 行仍为 in-train 20-ep,待统一 50-ep 复评)。

**关键对比(5traj,口径统一为离线 50-ep / seed 42 / iter 300 终值;RLT 单任务与基座现已同口径可比)**:基座 task0/1/3 = **0.78/0.92/0.42**(T1 干净重测)。
- RLT+TD3:0.78→**0.98**、0.92→**0.92**、0.42→**0.64**
- RLT+GRPO:0.78→**0.96**、0.92→**0.96**、0.42→**0.64**
- RLT+PPO:0.78→**0.94**、0.92→**0.92**、0.42→**0.96** ← **task3 几近满分**
- VLA+PPO / VLA+GRPO:**5traj 未跑**(已有数据全部为 1traj 基座,见表 1c)

参照:**RLT_a + TD3 release alltasks**(同 5traj,离线 50-ep,iter 400) → task0/1/3 = **1.00/1.00/0.68**,overall **0.92**(+0.23 over baseline),详见表 1b。

- **task0 同基座横比(5traj,离线 50-ep)**:基座 0.78 < RLT+PPO 0.94 < RLT+GRPO 0.96 < **RLT+TD3 0.98**。三算法都接近饱和,差异在噪声级。
- **task3(难任务)同基座横比(5traj,离线 50-ep)**:基座 0.42 → RLT+TD3 **0.64** / RLT+GRPO **0.64** / **RLT+PPO 0.96 ← 一骑绝尘**。**PPO 在难任务上是 RLT 路线最优 algo**,大幅领先 TD3/GRPO(+0.32)。50-ep 复评后该结论依然稳固(20-ep 时 PPO=1.00,50-ep=0.96,仍是唯一突破难任务的算法)。
- **"VLA 全量微调在难任务上失败"** 的叙事目前仅由 **1traj** 数据支撑(见表 1c);要在 5traj 上同样成立,需要补 5traj VLA+PPO / VLA+GRPO 单任务 run。

口径说明:本表 RLT 单任务行已由 9-run 统一离线 50-ep 复评(`results/eval_rlt_50ep_0529/`,seed 42 / max_steps 320 / 4 workers)产出,替换早前的 in-train 20-ep 数。**「全 10 任务」列已全部转为离线 50-ep**(GRPO/PPO × RLT/RLT_a 4 个 multi-task run + RLT_a+PPO task3,源 `results/eval_p0_50ep_0529/`,seed 42 / max_steps 320 / 8 workers,iter300 ckpt),与基座 0.704、RLT_a+TD3 release 0.92 完全同口径;替换早前的 in-train 200-ep 监控值(原 GRPO/PPO/RLT_a-GRPO/RLT_a-PPO = 0.74/0.885/0.755/0.95 → 50-ep 0.720/0.916/0.704/0.936)。多种子(≥3 seed)后改报 `mean±std`。

### 表 1-① — 全 10 任务多种子 mean±std(收集中,0607)

> 每个 cell 跑 seed 42/43/44 三种子,离线 50-ep / iter300 / 同口径。seed42 = 上表已报值。
> s44 由 box2 跑(健康主力),s43 由 box1(MAX=2,受外部 debug.py 干扰偏慢)+ box2 空闲 GPU 补跑。
> 🔄 = 训练/eval 进行中。**收集进度随 `results/eval_mt_seeds_0607/*.json` 落地更新。**

| 方法 | seed42 | seed43 | seed44 | mean±std |
|---|---|---|---|---|
| RLT + TD3   | 0.830 | 🔄(box2 s43v3) | 🔄(box2 s44v3) | 🔄 待齐 |
| RLT + GRPO  | 0.720 | **0.806** | **0.832** | **0.786 ± 0.048(n=3)** ✅ |
| RLT + PPO   | 0.916 | **0.978** | **0.966** | **0.953 ± 0.027(n=3)** ✅ |
| RLT_a + TD3 | 0.920 | 🔄(box2 s42v3 重训给曲线) | 🔄 | 🔄 待齐 |
| RLT_a + GRPO| 0.704 | 🔄(box2 s43v3) | 🔄(box2 s44v3) | 🔄 待齐 |
| RLT_a + PPO | 0.936 | **0.96** | **0.956** | **0.951 ± 0.010(n=3)** ✅ |

> **0610 第一波齐 3 格 n=3**:RLT+PPO **0.953±0.027**(42/43/44=.916/.978/.966)、RLT+GRPO **0.786±0.048**(.720/.806/.832)、RLT_a+PPO **0.951±0.010**(.936/.96/.956)。**两条 PPO 路线统计打平(0.953 vs 0.951)**,均为榜首;GRPO 显著低。s43 由 box1 重训(0609v3,thread-pin),box2 在跑剩余 TD3×2 + RLT_a+GRPO×2 的 s43/s44。已更新 overleaf 表1 + fig6a/6c error bar。

> **已落地(0607,3/12)**:RLT+PPO s44=**0.966** / RLT+GRPO s44=**0.832** / RLT_a+PPO s44=**0.956**(均 seed44)。
> 三个 n=2 cell 跨种子均稳健:PPO 两路(RLT 0.941、RLT_a 0.946)均 >0.94 且方差极小,**主结论 PPO>GRPO 跨种子保持**;RLT_a+PPO 0.946±0.010 印证榜首稳健。其余 9 run 训练中(~15-20h),齐后转 n=3。

---

## 表 1b — RLT_a + TD3 release runs(4/14-4/16,离线 50-ep)

> 主结果之外**作为参照**的 release 训练 + 离线 eval 数据。算法 = **RLT_a + TD3**(action-token 瓶颈编码器,`ActionTokenEncoderDecoder`:`bottleneck_proj (256, 2048)` + `cls_token (1,1,2048)`,encoder.pt 135M params)。原始命名 "RLT_a" 正确(注:`cls_token` 两个 track 都有,**真正区分是 `bottleneck_proj`**)。配图 `fig4_rl_gain.png`。

### 表 1b-① — 5traj 全 10 任务(multi-task 单 run,iter 400)

> 源:`rlt_5traj_alltasks_v3_release_0414_1727`,eval = `eval_rlt_release_0415/5traj_alltasks_merged.json`。

| 任务 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | **overall** |
|:-----|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:------------|
| 基座 VLA (T1,50-ep) | 0.78 | 0.92 | 0.98 | 0.42 | 0.84 | 0.34 | 0.34 | 1.00 | 0.90 | 0.52 | **0.704** |
| RLT_a + TD3 (release) | 1.00 | 1.00 | 0.94 | 0.68 | 1.00 | 0.86 | 0.80 | 1.00 | 1.00 | 0.92 | **0.920** |
| **Δ** | +0.22 | +0.08 | −0.04 | +0.26 | +0.16 | +0.52 | +0.46 | 0.00 | +0.10 | +0.40 | **+0.216** |

观察:基座行已用 **T1 干净重测**(离线 50-ep,与 release eval 同协议)替换早前的 0.686 锚定。RL 增益集中在基座弱任务(t3/t5/t6/t9,基座 0.34–0.42);基座已强的任务(t1/t2/t7/t8)接近饱和,t2 轻微 −0.04(噪声级),t7 已满分无增益。task3 在 5traj 下 0.42→0.68 仍是最弱任务;1traj 场景的负迁移见下表 1b-②。

### 表 1b-② — 1traj 单任务 release(每个任务独立 run)

> 源:`rlt_1traj_t0_release_0415_1010` / `rlt_1traj_t1_release_0414_1829` / `rlt_1traj_t3_release_0414_1838`,eval = `eval_rlt_release_0415/task{0,1,3}_release.json`。

| 任务 | t0 | t1 | t3 |
|:---|:---|:---|:---|
| 基座 VLA (1traj) | 0.32 | 0.82 | 0.22 |
| RLT_a + TD3 (release) | **0.86** | **0.96** | **0.08** |
| **Δ** | +0.54 | +0.14 | **−0.14** |

观察:task0/1 1traj 弱基座下 RL 大幅提升(+0.14, +0.54);**task3 出现负迁移**(0.22 → 0.08),与 1traj_alltasks_v3 中 task3=0.06 一致(见 `1traj_alltasks_merged.json`)。这是 1traj × 难任务的稳定模式,而非随机抖动。

---

## 表 1c — 主结果(1traj 基座,libero_goal,QwenOFT)

> 单轨基座弱、协方差小,RL 训练更易崩。同样 task0/1/3。配图 `fig1_main_results.png`(1traj 子图)。

| 方法 | 编码器 | task0 | task1 | task3 | 全 10 任务 | 说明 |
|:-----|:-------|:------|:------|:------|:-----------|:-----|
| **基座 VLA(训练前)** | — | **0.34** | **0.88** | **0.00** | 0.33 | **0602 严格 50-ep 重测**(seed 42,`results/eval_base_vla/qwen_1traj_t013_0601/`)替换早前 0.32/0.82/0.22;锚定 |
| RLT_a + TD3 (release) | RLT_a | 0.86 | 0.96 | **0.08** | — | release run(单任务,4/15);**离线 50-ep**;task3 负迁移(见表 1b-②) |
| RLT + TD3 | RLT | — 🔄 | **0.92** | **0.12** | — | task1/3 **0602 50-ep 复评完成**(`rlt_td3_1traj_t{1,3}_0601_1014`,iter300);**vs 基座 0.88/0.00:t1 +0.04(已饱和),t3 +0.12(基座 0)** — RLT+TD3 单 traj 上仍未突破 task3 但比 RLT_a release −0.14 好。task0 还未启 1traj run |
| RLT + GRPO | RLT | **0.46** | **0.92** | **0.08** | — | 离线 50-ep(seed 42,0604,`results/eval_rlt_1traj_50ep_0603/`);run=`rlt_grpo_qwen_t{0,1,3}_0602_1612`。⚠️ ckpt 非 iter300:t0/t1=grpo_iter_200/250,t3=iter150(run 在 env-pool 超载 bug 下中途崩,取最后可用 ckpt;t1 崩前已收敛 SR=1.00→50-ep 0.92)。**vs 基座 0.34/0.88/0.00:t0 +0.12,t1 +0.04(饱和),t3 +0.08(几乎没学起来)**;**难任务 t3 GRPO 失败**(in-train 全程卡 ≤0.125) |
| RLT + PPO | RLT | **1.00** | **0.88** | **0.80** | — | 离线 50-ep(seed 42,0604,`results/eval_rlt_1traj_50ep_0603/`);run=`rlt_ppo_qwen_t{0,1,3}_0602_1612`。⚠️ ckpt 非 iter300:t0/t1=rl_iter_200,t3=iter150(同上 bug 中途崩;t0 崩前已收敛 SR=1.00)。**vs 基座 0.34/0.88/0.00:t0 +0.66,t1 0.00(已饱和),t3 +0.80** ← **PPO 在 1traj 难任务上也突破(0.00→0.80),远超 GRPO 的 0.08**,与 5traj 结论一致(PPO 难任务最优) |
| VLA + PPO(基线) | — | **0.85** | — | **(0.45末峰)** | — | task0:300 iter,15 evals 范围 55–100,终值 0.85(抖动剧烈);task3:8 evals 范围 0–15%,**末次 45% 是孤立单峰**,随后 rollout SR 跌回 0.12 被杀(iter 161),**实际难任务失败** |
| VLA + GRPO(基线) | — | **0.65** | — | **0.00** | — | task0:300 iter,15 evals 范围 35–75,终值 0.65;task3:7 evals 全部 ≤15%,终值 0,iter 143 被杀;**难任务失败** |

**1traj 关键现象(口径均为终值;RLT_a+TD3 数据来自表 1b-② release)**:
- **task0**(基座 0.32):RLT_a+TD3 release **0.86**(+0.54,离线 50-ep)/ VLA+PPO **0.85**(+0.53,in-train 20-ep)/ VLA+GRPO **0.65**(+0.33)。RLT_a+TD3 与 VLA+PPO **几乎打平** —— 弱基座下 RL 增益空间大,瓶颈结构(RLT_a)与全量微调(VLA)在 task0 上差异不显著(注:口径不同,RLT_a 是离线 50-ep,VLA 是 in-train 20-ep)。
- **task3**(难任务,基座 0.22):RLT_a+TD3 release **0.08**(−0.14,负迁移)/ VLA+PPO 末值 0.45 但前 7 次 eval ≤15%(均值 ~0.05),**实际崩** / VLA+GRPO **0.00**(7 次全 ≤15%)。**三种 RL 算法在 1traj 难任务上全部失败** —— 负迁移**是方法无关的**,单轨数据 + 难任务 + 稀疏奖励 = 探索方向被噪声主导,跟用 RLT_a 瓶颈结构还是 VLA 全量微调无关。这是 RQ-C 的核心证据。
- **VLA+PPO/GRPO 训练稳定性显著差于 RLT 路线**:VLA+PPO task0 15 次 eval **范围 55–100**(均值 ~84,std ~12)/ VLA+GRPO 范围 **35–75**(均值 ~56,std ~11)/ 对比 RLT+PPO 5traj task0 范围 85–100(std ~5)/ RLT+GRPO 5traj task0 范围 90–100(std ~3)—— **RLT 瓶颈结构降低了训练方差**,这是与 VLA 全量微调对比的核心定性优势(不仅是终值)。
- RLT × 三个算法 1traj **in-train** 数据全空:**1traj 上 RLT+TD3/GRPO/PPO 单任务 in-train 未跑**(只有 release multi-task 数据),无法在同基座、同协议下做 RLT-vs-VLA 同算法对比。建议 T6 实验:在 1traj 上跑一组 RLT+PPO 与 VLA+PPO 单任务对照,验证 RLT 稳定性优势在难基座上是否保留。
- 训练曲线(task0,1traj):见表 5(iter 25→300:0.36→0.92,中段抖动)。

---

## 表 2 — RLT × 不同 RL 算法(报告重点)

> 固定同一 RLT 编码器、相同任务集、相同 env-step 预算、≥3 seed。横轴对齐 env steps。

| 算法 | 类型 | critic | 终态 SR(终值,task0/1/3) | 样本效率 (env-steps→80% all-task SR) | 稳定性 (seed 方差) | 代码状态 |
|:-----|:-----|:-------|:---------------------------|:---------------------------|:-------------------|:---------|
| TD3 | off-policy AC | 双 Q + 目标平滑 | 0.98 / 0.92 / 0.64 | **~10.2M**(iter300 累计 14.6M,终值仅 0.83) | — | ✅ 可跑 |
| GRPO | on-policy PG | 无(组相对优势) | 0.96 / 0.96 / **0.71±0.06** | **从未达 80%**(plateau ~0.72-0.75;iter300 累计 ~3.1M) | — | ✅ 已跑通;t3 **3-seed mean±std**(42/43/44 = 0.64/0.74/0.74) |
| PPO | on-policy AC | V(s) + GAE | 0.94 / 0.92 / **0.95±0.01** | **~0.5M**(首个 eval 点即 >0.8;~2.5M 到 plateau >0.9) | — | ✅ task0/1/3 已跑通(`run_rlt_ppo.sh`,iter 300);**task3 = 0.95±0.01(3-seed:0.96/0.94/0.94)是 RLT × 任意 algo 最高 task3**,且方差极小(稳定突破难任务) |

> **样本效率(0608 新增,all-task SR vs 累计 env-step,源各 run `metrics.json` 的 `total_env_steps` + `eval_effcurve_0608/` SR)**:横轴 env-step 对齐后,**PPO ~0.5M 到 80% / TD3 ~10.2M(20× 更多)/ GRPO 永不达 80%**。off-policy replay 让 TD3 跑到 iter300 消耗 14.6M env-step(vs PPO 2.5M,~6×),换来更低的终值。配图 `fig6c_sample_efficiency.png`(已接 overleaf `fig:rlt_sample_eff`)。注:此 env-step 数来自**多任务 all-10 run**,样本效率指标为 all-task 口径(非单任务 task0/1/3)。

说明:这张表是"重点"叙事的承载表——同一 RLT 底座下,off-policy / on-policy、有无 critic、有无 KL 约束的对照。SR 均为 iter 300 ckpt 的**离线 50-ep 复评**终值(seed 42 / max_steps 320,`results/eval_rlt_50ep_0529/`;不取峰值、不做均值平滑),与基座 T1 同口径可比。当前 TD3 / GRPO / PPO **三 algo × 三 task 全部齐**(总 9 cell)。**PPO task3 = 0.96 是难任务上 RLT 路线的最强结果**,显著优于 TD3(0.64) 和 GRPO(0.64) —— PPO 的 critic 信号在难任务上对 RLT 底座最有利。多种子 + 样本效率/稳定性指标待 Tier-1 实验。

---

## 表 3 — 消融(固定 RLT + TD3,libero_goal task0)

| 变体 | 改动 | SR | 相对默认 Δ |
|:-----|:-----|:---|:-----------|
| 默认(RLT + 预训练编码器) | — | **0.98** | 0 |
| 编码器随机初始化 | 去 Phase-1 预训练 | **0.66** | −0.32 | 离线 50-ep,iter300,`rlt_td3_randenc_t0_0601_1544`(`ENCODER_PATH=""` 随机初始化 frozen encoder);**预训练 +0.32** vs 默认 0.98 |
| RLT_a 编码器 | 全 token → 动作 token 瓶颈 | **0.82** | −0.16 |
| **`β = 0`** | 去掉 BC 正则(无 anchor) | **0.00** | −0.98 | 离线 50-ep,iter300,`rlt_td3_abl_beta0_t0_0601_0940`(unique RUN_NAME 干净重跑,非 `rlt_rl_qwen_t0` 污染版);in-train 6000 ep 全 0 → 50-ep **确认 0.00**,去 BC anchor 直接崩 |
| `ref_dropout = 0` | 关闭 RL 特征屏蔽(去 VLA-anchor 正则) | **1.00** | +0.02 | 离线 50-ep,iter300,`rlt_td3_abl_refdrop0_t0_0605_1634`(unique RUN_NAME 干净重跑);**vs 默认 0.98 略升(噪声级)→ ref_dropout 对 task0 终值无负面影响**(去 anchor 后训练稳定性可能稍差,但终值不降) |
| 瓶颈维度 `D=128` | RLT_a 瓶颈 256→128 | **0.84** | +0.02 | 离线 50-ep,iter300,`rlt_a_td3_abl_dim128_t0_0604_1015`(task0,vs RLT_a 默认 D=256=0.82) |
| 瓶颈维度 `D=512` | RLT_a 瓶颈 256→512 | **0.82** | 0.00 | 离线 50-ep,iter300,`rlt_a_td3_abl_dim512_t0_0604_1015`(task0);**D-sweep 128/256/512 = 0.84/0.82/0.82 → 瓶颈维度鲁棒,128-512 几无差异** |

**关键观察**:
- ✅ **β=0 已解决**:用 unique RUN_NAME 干净重跑 `rlt_td3_abl_beta0_t0_0601_0940`(iter300),0603 离线 50-ep 复评 = **0.00**,已填表。原污染版(`rlt_rl_qwen_t0` 同分钟撞名互相覆盖)弃用。
- ✅ **ref_dropout=0 完成(0606)**:`rlt_td3_abl_refdrop0_t0_0605_1634`(iter300,离线 50-ep)= **1.00**(vs 默认 0.98,+0.02 噪声级)→ **ref_dropout 对 task0 终值无影响**。**表 3 现已全满**(默认 0.98 / 随机初始化 0.66 / RLT_a 0.82 / β=0 0.00 / D=128 0.84 / D=512 0.82 / ref_dropout=0 1.00)。
- ✅ **瓶颈维度 D-sweep 完成(0605)**:RLT_a 单任务 t0,D=128/256/512 = **0.84/0.82/0.82**(离线 50-ep,`rlt_a_td3_abl_dim{128,512}_t0_0604_1015`)→ **瓶颈维度鲁棒,128-512 几无差异**,D=128 略优(+0.02)。
- **`β = 0` 强 prediction**:in-train 6000 ep 全 0 → 50-ep ≈ 0.00 几乎确定(任何 ckpt 都 0)
- **`ref_dropout = 0` 强 prediction**:in-train 末段 0.85-1.00,**预计 50-ep 接近默认 0.98 或略低**(去 anchor 后稳定性差)
- `RLT_a 编码器`:0.82 < 默认 0.98(差 16pp),task0 上 full-token RLT 略优于 action-token RLT_a 瓶颈;但 RLT_a 多任务 0.936 反超 → 两条路线在不同场景各擅

注:`β`、`ref_dropout` 等部分变体可从 `results/rlt_training_TD3/`(`*beta3*`、`*groupsize4*`、`*chunk4*` 等 dev run)整理,不必全部重跑。

---

## 表 4 — 跨 VLM 骨干(RLT + TD3,证明仓库多骨干支持)

| 骨干 | task0 | 全 10 任务 | 数据来源 / 备注 |
|:-----|:------|:-----------|:----------------|
| QwenOFT-5traj | 1.00 | 0.92 | 见表 1b-① (RLT_a+TD3 release alltasks,离线 50-ep) |
| QwenOFT-1traj | 0.86 | **0.548** | task0 见表 1b-②;全 10 任务 = **0.548**(`1traj_alltasks_merged.json`,release `1traj_alltasks_v3`,离线 50-ep);逐任务见表 4-① |
| Pi05-1traj | — | — | 只有 in-train 20-ep,无 50-ep 离线 eval(`rlt_ori_rl_t{0,1,3}_release_pi05_1traj_*` 8 个 run 全部 NO_SUMMARY) |
| Pi05-5traj(单任务) | **0.98** | — | task0=0.98(`rlt_ori_rl_t0_release_pi05_5traj_0429_0924`,全 iter 50-ep,iter300 终值 0.98 / peak 1.00);**task1=1.00**(0602 补,`rlt_ori_rl_t1_release_pi05_5traj_0429_0522` iter300 50-ep);task3=0.82(`rlt_ori_rl_t3_release_pi05_5traj_0429_0923`,iter300 终值,peak 0.84)。**vs QwenOFT-5traj 同任务:task0=1.00/1.00 打平;task3 = 0.82 (Pi0.5) > 0.64 (Qwen) → Pi0.5 在难任务上 +0.18**;全 10 任务多任务见下行 |
| **Pi05-5traj 多任务(RLT+TD3)** | **0.94** | **0.868** | **0605 新跑完**:`rlt_td3_pi05_5traj_mt_0604_1010`(RLT full-token + TD3,全 10 任务 multi-task,iter300,离线 50-ep);per-task t0–t9 = .94/1.00/.94/.48/1.00/.96/.78/.96/.98/.64,弱点 t3=.48。**vs QwenOFT-5traj RLT+TD3 多任务 0.830 → Pi0.5 +0.038**(跨骨干:Pi0.5 多任务整体略优于 Qwen);弱点都在 t3(Pi0.5 .48 vs Qwen .68) |

### 表 4-① — QwenOFT 5traj vs 1traj 逐任务(RLT_a + TD3 release alltasks,离线 50-ep)

> 同一 RLT_a+TD3 算法、同骨干家族,仅训练数据量不同(5traj vs 1traj)。源:`5traj_alltasks_merged.json` / `1traj_alltasks_merged.json`(release `*_alltasks_v3`)。展示数据量对 RL 后多任务 SR 的影响。

| 骨干 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | **overall** |
|:-----|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:------------|
| QwenOFT-5traj | 1.00 | 1.00 | 0.94 | 0.68 | 1.00 | 0.86 | 0.80 | 1.00 | 1.00 | 0.92 | **0.920** |
| QwenOFT-1traj | 0.96 | 0.96 | 0.18 | 0.06 | 0.96 | 0.04 | 0.38 | 1.00 | 0.92 | 0.02 | **0.548** |
| **Δ (5traj−1traj)** | +0.04 | +0.04 | +0.76 | +0.62 | +0.04 | +0.82 | +0.42 | 0.00 | +0.08 | +0.90 | **+0.372** |

观察:1traj 在**简单/已饱和任务**(t0/t1/t4/t7/t8)与 5traj 几乎打平(Δ ≤ 0.08),但在**难任务/长程任务**(t2/t3/t5/t9,5traj 已是弱项)**断崖式崩塌**(1traj t9=0.02、t5=0.04、t3=0.06、t2=0.18)。**Δ 完全集中在难任务**——数据量不足时 RL 无法在难任务上建立有效策略,与表 1c「1traj × 难任务负迁移」叙事一致,且证明该现象是**数据量驱动**而非算法/骨干驱动。

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
| ~~表 1「基座 VLA」行~~ | ✅ **T1 完成**(5traj 0.704 / 1traj 0.346,逐任务齐全,离线 50-ep) |
| ~~表 1/表 2 RLT 单任务「离线 50-ep」复评~~ | ✅ **完成**(9-run:TD3/GRPO/PPO×t0/t1/t3,`results/eval_rlt_50ep_0529/`) |
| 表 1 各 RL 行的多种子 mean±std | T3(≥3 seed) |
| ~~表 1 VLA+PPO / VLA+GRPO 5traj 全 10 任务~~ | ✅ **完成**(5traj,task-cycling 4/iter,iter300,离线 50-ep:PPO 0.718 / GRPO 0.756,源 `results/eval_vla_mt_50ep_0601/`)。需 VLA trainer 多任务改造已完成(`train_rl_vla_{ppo,grpo}.py` 加 `--all_tasks`/`--tasks_per_iter` + GRPO collector 升级修死锁,见 `VLA_MULTITASK_STATUS.md`)。**单任务 t1 仍 🔄** |
| 表 1c RLT × {TD3, GRPO, PPO} 1traj | 🔄 T6 进行中:编码器预训练中(GPU0),随后跑 task3 × 3 算法 |
| ~~表 2 收敛/效率曲线~~ | ✅ **完成(0608)**:6 条 RLT route(RLT/RLT_a × TD3/PPO/GRPO)逐 iter(50–300)离线 50-ep 收敛曲线,源 `results/eval_effcurve_0608/{rlt,rlta}_{ppo,grpo}_iter*.json` + TD3 `rlt_td3_multitask_0602_1433/eval_iter_*`;均从 iter0=0 冷启动锚定。两张图 `report_figures/fig6a_rlt_routes.png`(2×3 grid,color=algo/style=encoder)、`fig6b_rlt_vs_vla.png`(最强 RLT vs 小规模 VLA baseline),已接入 overleaf 章节(`fig:rlt_convergence` / `fig:rlt_vs_vla`)。RLT_a+TD3 无中间 ckpt → 仅终点 0.92。**逐 iter overall_sr**:RLT+PPO 0/.856/.892/.920/.932/_/.916;RLT_a+PPO 0/.828/.910/.918/.906/.912/.936;RLT+GRPO 0/.728/.742/.746/.748/.744/.720;RLT_a+GRPO 0/.714/.720/.724/.742/.704/.704;RLT+TD3 0/.706/.724/.770/.808/.800/.830 |
| 表 2 样本效率列(env-steps→80%) | T9(env-step 对齐样本效率,off/on-policy 需统一横轴) |
| 表 2 多种子 mean±std + seed 方差 | T3(≥3 seed) |
| ~~表 3 消融~~ | ✅ **全满(0606)**:默认 0.98 / 随机初始化 0.66 / RLT_a 0.82 / β=0 0.00 / D=128 0.84 / D=512 0.82 / ref_dropout=0 1.00 |
| 表 1/2 多种子 mean±std | 🔄 **进行中**:t3 单任务已 3-seed(PPO 0.95±0.01 / GRPO 0.71±0.06);**多任务 headline 格**(0.936/0.916/…)seed 43/44 两台机并行跑中(`WEB_LAUNCH_multitask_seeds.sh`),凑齐后填 |
| RLT_a 单任务行离线 50-ep | 本轮只复评了 RLT 9 run;RLT_a 单任务仍为 in-train 20-ep,待补 |
| ~~表 4 QwenOFT-1traj 逐任务~~ | ✅ **完成**(表 4-①,5traj vs 1traj 逐任务对照,0603 整理;1traj overall 0.548,Δ 集中难任务)。Pi05-1traj 逐任务仍缺(8 run 全 NO_SUMMARY,需重 eval) |
| ~~表 1 全 10 任务 multi-task 离线 50-ep~~ | ✅ **完成**(GRPO/PPO × RLT/RLT_a + RLT_a-PPO-t3,iter300 0529 重跑 run,`results/eval_p0_50ep_0529/`;0.720/0.916/0.704/0.936,替换 in-train 200-ep)。**RLT+TD3 multitask 0603 补齐 = 0.830**(`rlt_td3_multitask_0602_1433`)→ 5traj 全 10 任务列**全满,无 🔄** |
