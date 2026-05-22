# RL 技术报告:写作指南与实验补充清单

> 主线:**RLT(瓶颈-token 编码器)× 不同 RL 算法**,以刻画 RLT 的能力与局限。
> 基线:**直接对整个 VLA 做 RL(VLA+PPO / VLA+GRPO)**,代表主流方法。
> 基座:**QwenOFT 为主**,Pi05 补充若干以证明仓库支持多种 VLM。
> 配套文档:`TECHNICAL_REPORT.md`(代码层技术报告)。

---

## Part 0 — 报告的组织方法

本报告把 **RLT 当作研究对象**,把 **不同 RL 算法当作探针**:在固定的 RLT 瓶颈底座上分别接入 TD3 / GRPO / PPO,通过算法间的差异来回答"这个瓶颈底座能做什么、在哪里失效"。直接 RL 整个 VLA(VLA+PPO / VLA+GRPO)是主流做法,作为基线对照。

由此,报告要回答的研究问题(后文按此组织实验):

- **RQ-A**:同一个 RLT 底座,接 TD3 / GRPO / PPO,表现有何差异?(主线)
- **RQ-B**:RLT 路线相比直接 RL 整个 VLA,值不值?(vs 基线)
- **RQ-C**:RLT 在哪里失效?(失败案例 = 能力边界)
- **RQ-D**:RLT 能否跨 VLM 骨干移植?(Qwen vs Pi05)
- **RQ-E**:瓶颈/编码器的设计选择是否必要?(消融)

---

## Part 1 — ⚠️ 关键代码前置(必须先解决)

**主线"RLT × 不同算法"目前在代码层还没打通。** 经核实:

| 编码器 \ 算法 | TD3(`train_rl_offpolicy.py`) | GRPO(`train_rl_grpo.py`) | PPO 小 actor(`train_rl_onpolicy.py`) |
|:--------------|:------------------------------|:--------------------------|:--------------------------------------|
| **RLT** | ✅ 支持 `--encoder_mode rlt`(L135–161 条件构造 `RLTokenEncoderDecoder`) | ❌ L73 硬编码 `ActionTokenEncoderDecoder`,**无 `encoder_mode` 分发** | ❌ 同样硬编码;且文件头标注 **legacy**,损失是 "placeholder" |
| **RLT_a** | ✅ | ✅ | ⚠️ 可跑,但损失为 legacy 占位 |

**结论**:只有 `RLT × TD3` 这一格是成熟的。要让主线成立,先做下面三件工程任务(均在 `trainers/` 下,**不涉及** `model/framework/`,可放心改):

| 工程任务 | 内容 | 工作量 | 推荐做法 |
|:---------|:-----|:-------|:---------|
| **E1 ✅ 已完成** | 给 `train_rl_grpo.py` 加 `--encoder_mode {action_token, rlt}` 分发 | 小 | 已实现:`train_rl_grpo.py` 按 `encoder_mode` 选编码器;`action_token_collect_group`、`_eval_distributed` 加 `encoder_mode` 透传。GRPO 损失用缓存的 `rl_token`、本身编码器无关。**RLT×GRPO 现可跑** |
| **E2** | 解决小 actor PPO | 中 | `train_rl_grpo.py` 的损失本身就是 **PPO 截断代理 + 组相对优势**。建议在该 trainer 里加一个开关:用 `GAE + critic` 替换组相对优势,即得"小 actor PPO",**复用同一套已验证代码**,绕开 legacy 的 `train_rl_onpolicy.py` |
| **E3** | 把 VLA+PPO / VLA+GRPO 跑到收敛 | 算力 | 当前只有 2 个 iter 的 smoke;基线必须收敛才能进对比表 |

> 如果 E2 暂时不做:主线退化为 **RLT × {TD3, GRPO}**(off-policy vs on-policy 的对照仍然成立,叙事完整),PPO 仅以 `VLA+PPO` 基线身份出现。这是可接受的最小可行方案。

---

## Part 2 — 现状盘点(按新矩阵)

`results/` 下已有产出与成熟度(QwenOFT 为主):

| 单元格 | 代码 | 已有数据 | 成熟度 |
|:-------|:-----|:---------|:-------|
| RLT × TD3 | ✅ | 1-traj task0=86%/task1=96%/task3=8%、5-traj 全 10 任务=92%、task0 曲线 iter25→300:36%→92%;Pi05 亦有 run | **高** |
| RLT_a × TD3 | ✅ | `results/rlt_training_TD3/` 大量 dev run(beta3/groupsize4/chunk4/steplock…) | 中(散乱,无干净对照) |
| RLT × GRPO | ❌(待 E1) | 无 | — |
| RLT × PPO | ❌(待 E2) | 无 | — |
| RLT_a × GRPO | ✅ | `rlt_a_grpo_qwen_t0_0519` smoke | 低 |
| RLT_a × PPO | ⚠️ legacy | `rlt_a_ppo_qwen_t0_0519` smoke | 低 |
| VLA+PPO(基线) | ✅ | `vla_ppo_qwen_t1_*` smoke | 低 |
| VLA+GRPO(基线) | ✅ | `vla_grpo_qwen_t*` 仅 2 iter | 低 |

**一句话**:主线现在只有 1/3~1/4 个数据点(RLT×TD3)。Part 5 的 Tier 1 就是把这张矩阵填满。

---

## Part 3 — 论文章节骨架

```
第 N 节  VLA 在线强化学习:RLT 瓶颈底座与多算法研究

  N.1  动机
       — VLA 为何需要在线 RL;直接 RL 整个 VLA 的代价(显存、稳定性、遗忘)
       — RLT 思路:冻结 VLA → 压成瓶颈 token → 其上做轻量 RL
       — 本节研究问题(RQ-A…E)

  N.2  问题形式化
       — MDP:状态 x=(z_rl, s_p) / 动作 chunk / 稀疏二值奖励 / 折扣 γ^C / 终止

  N.3  方法
       N.3.1  RLT 瓶颈编码器(Phase-1)— RLT(全 token)与 RLT_a(动作 token)两变体
       N.3.2  瓶颈状态上的 RL 算法(Phase-2)
              · TD3 —— off-policy,双 Q,目标平滑,BC 正则
              · GRPO —— on-policy,组相对优势,无 critic,KL 约束
              · PPO —— on-policy,GAE + critic
              · 核心论点:RLT 提供算法无关的紧凑状态接口,RL 算法即插即用
       N.3.3  基线:直接 VLA RL(VLA+PPO / VLA+GRPO,整个 VLA 即策略)

  N.4  实验设置

  N.5  主结果
       — 表 1:RLT×{算法} vs VLA-RL 基线 vs 纯基座(QwenOFT)
       — 图 1:样本效率曲线(SR vs env steps)

  N.6  研究:RLT 的能力与局限
       — RQ-A 算法间差异(on/off-policy、有无 critic、KL 约束)
       — RQ-C 失败案例:task3 崩溃 —— 各算法是否同崩
       — 多任务与长程任务下的退化
       — RQ-D 跨 VLM 骨干(Qwen vs Pi05)

  N.7  消融(RQ-E)
       — RLT vs RLT_a 编码器;编码器预训练 on/off;瓶颈维度 / BC β / 参考 dropout

  N.8  讨论与局限
```

---

## Part 4 — 写作建议

### 4.1 核心论断

把贡献写成一句可证伪的话:

> **RLT 把冻结 VLA 的隐藏状态压缩为一个信息瓶颈 token,形成一个算法无关的紧凑 RL 接口;在其上 TD3 / GRPO / PPO 均可稳定训练并提升 LIBERO 成功率。我们借多算法的对照刻画该瓶颈底座的能力与局限,并与直接 RL 整个 VLA 的主流做法对比,后者计算昂贵且更易不稳。**

注意这是一个"**研究/分析**"型论断,不是"我们的方法 SOTA"型。好处:`task3` 崩溃这类负面结果是**研究发现**而非缺陷,可以大方写进 N.6。

### 4.2 各小节要点

- **N.2 形式化**:务必明说**奖励稀疏二值**(只有任务成功给奖励)。这是后文一切的根:解释了为什么 TD3 需要 BC 正则、为什么 GRPO 在单任务下可能"整组无信号"、为什么 `task3` 会崩。
- **N.3 方法**:公式优先。编码器 Eq.1、重构损失 Eq.2、TD3 critic 目标与 actor 损失(含 `β‖a−ã‖²`)、GRPO 组相对优势与 KL、PPO 截断代理 + GAE,可直接搬 `TECHNICAL_REPORT.md` 第 6–8 节。每条公式后跟一句设计动机。
- **偏差要显式写**:`RLT_a` 用动作查询切片输入、额外 `H→256` 投影、随机 rollout 预训练数据,都是相对参考论文(RL Token, Physical Intelligence 2026)的已知偏差。单列一小段 "Deviations from the reference recipe" 写清楚——主动披露反而加可信度。各算法目录 `README.md` 已有完整 delta。
- **N.4 设置**:给可复现表——基座 VLA 名称与来源、suite、每任务初始状态数、**env-step 预算**、评测协议(离线 50 ep)、各算法超参。
- **N.6 研究**:这是报告的"重点"。围绕 RQ-A 写算法间差异的**机理分析**,不只是贴数字:off-policy(TD3,复用 replay)为何样本效率高于 on-policy;无 critic 的 GRPO 在稀疏奖励下靠组相对优势,稳定性如何;KL 约束对漂移的作用。`task3` 单独成段诚实分析。

### 4.3 写作陷阱(务必规避)

| 陷阱 | 说明 | 对策 |
|:-----|:-----|:-----|
| **基座张冠李戴** | 曾把 **71.4% 的 "1-traj baseline" 当成 QwenOFT,实际那是 PI0.5**。RL 提升表里的"RL 前"基座 SR,必须是该实验**实际加载的那个 ckpt 自测**的数 | 每个 RL 实验对应的基座单独 eval 一遍;表里写明 ckpt 路径 |
| **算法对比未对齐预算** | TD3 off-policy 复用数据,GRPO/PPO on-policy 每步丢弃;按 *iteration* 比较不公平 | 算法对比的横轴统一用 **env steps**(或 wall-clock),不用 iteration |
| **算法对比未对齐调参** | 只调了 TD3、其余用默认,比的是调参投入不是算法 | 每个算法各自做一轮合理超参搜索,在 setup 里说明 |
| **用训练内嵌 eval 数字** | 训练中 20-ep eval 仅供监控,方差大 | 论文数字一律用 `run_eval_rlt{,_a}.sh` 的 50-ep 离线评测 |
| **峰值 cherry-picking** | task0 曲线 iter150→300 在 74%–92% 抖动,只报 92% 是选择性报告 | 固定 model-selection:报 last iter 或末 3 ckpt 均值,setup 写明 |
| **单种子当定论** | 当前所有 run 单种子;LIBERO 单任务 SR 方差可达 ±10% | 主结果 ≥3 seed,报 mean±std |
| **GRPO 无信号未披露** | 稀疏二值奖励下,一组 episode 全成功/全失败则优势为 0 | 报告 `n_groups_with_signal` 占比,说明 `group_size` 取值 |
| **smoke run 当结果** | VLA+PPO/GRPO 现仅 1–2 iter | 跑到收敛(E3)再进表,否则只在讨论里一句话带过 |
| **混用 1-traj / 5-traj 基座** | 二者基线差异大 | 表格按基座分块或单列标注 |

---

## Part 5 — 实验补充清单

按研究问题组织,分 4 档优先级。Tier 0 是工程前置,不做则后面无从谈起。

### Tier 0 — 工程前置(见 Part 1)

- **E1 ✅ 已完成** `train_rl_grpo.py` + `action_token_collect_group` + `_eval_distributed` 已支持 `--encoder_mode rlt` → RLT×GRPO 已解锁
- **E2** 在 GRPO trainer 加 `GAE+critic` 开关得到小 actor PPO →(解锁 RLT×PPO);或暂缓,主线退化为 RLT×{TD3,GRPO}
- **E3** VLA+PPO / VLA+GRPO 跑到收敛 →(解锁 RQ-B 基线)

### Tier 1 — 主结果(必须;不补则论文核心不成立)

| 编号 | 实验 | 服务 RQ | 说明 |
|:-----|:-----|:--------|:-----|
| **T1** | **基座锚定**:每个用于 RL 的 VLA ckpt(QwenOFT-1traj/5traj、Pi05)在对应任务跑纯 VLA 评测,每任务 50 ep | RQ-B | 现有曲线缺 iter0,无法说明 RL 提升了多少 |
| **T2** | **RLT × {TD3, GRPO, (PPO)} 主矩阵**:固定同一 RLT 编码器(同 Phase-1 ckpt),固定 3–5 个 `libero_goal` 任务(含易/中/难,难的含 task3),**相同 env-step 预算**,**≥3 seed** | RQ-A | 报告的重点;横轴用 env steps |
| **T3** | **基线对照**:纯基座 vs RLT×最佳算法 vs VLA+PPO vs VLA+GRPO,同任务同预算 | RQ-B | 含计算成本(GPU-hour、显存) |
| **T4** | **task3 崩溃归因**:查(a)基座在 task3 的 SR 是否本就极低 →稀疏奖励 rollout 拿不到正样本 →无梯度;(b)buffer 中 reward>0 占比 / `n_groups_with_signal`;(c)各算法在 task3 是否**同崩**还是某算法能救 | RQ-C | 负面结果,诚实分析;算法×task3 的交互是好发现 |

### Tier 2 — 强烈建议(支撑研究与消融)

| 编号 | 实验 | 服务 RQ |
|:-----|:-----|:--------|
| **T5** | **跨骨干**:RLT×TD3 的 Qwen vs Pi05 对比表(已有数据,主要是整理)+ 至少补一个 RLT×GRPO on Pi05 | RQ-D |
| **T6** | **多任务**:RLT×{算法} 在全 10 任务 vs 单任务,看多任务干扰下是否退化 | RQ-A/C |
| **T7** | **RLT vs RLT_a 编码器**:固定算法(如 TD3),对比全 token 编码器与动作 token 瓶颈 | RQ-E |
| **T8** | **编码器预训练 on/off**:随机初始化编码器 vs Phase-1 预训练编码器 | RQ-E |
| **T9** | **样本效率曲线**:SR vs 累计 env steps,RLT×各算法 + VLA-RL 基线同图 | RQ-A/B |

### Tier 3 — 加分项

- **T10** 瓶颈维度 `D`∈{128,256,512}、BC `β`∈{0,1,3}、`ref_dropout`∈{0,0.5} 细消融——多数可从 `results/rlt_training_TD3/` 已有 dev run **整理**而非重跑(先盘点已有 run 配置)。
- **T11** 长程任务(`libero_10`)、跨 suite(`spatial`/`object`)泛化。
- **T12** 计算成本表:各算法 wall-clock / GPU-hour / 显存——呼应 RLT 小 actor 的"轻量"卖点。
- **T13** 朴素基线 filtered-BC:把 RL rollout 的成功轨迹直接 BC/SFT,对比 RL——若 RL 打不过 filtered-BC,这个发现本身值得写。
- **T14** 定性可视化:同一初始状态 RL 前失败 / RL 后成功的轨迹帧序列。

---

## Part 6 — 主结果表 / 图模板

### 表 1:主结果(T2 + T3 产出)

> QwenOFT,每任务 50 ep,≥3 seed,mean±std;model selection = 末 3 ckpt 均值。

| 方法 | 编码器 | 策略对象 | task(易) | task(中) | task(难/t3) | 全 10 任务 | env-steps→80%SR |
|:-----|:-------|:---------|:---------|:---------|:------------|:-----------|:----------------|
| 基座 VLA(RL 前) | — | — | __±__ | __±__ | __±__ | __±__ | — |
| RLT + TD3 | RLT | 小 actor | | | | | |
| RLT + GRPO | RLT | 小 actor | | | | | |
| RLT + PPO | RLT | 小 actor | | | | | |
| RLT_a + TD3 | RLT_a | 小 actor | | | | | |
| VLA + PPO(基线) | — | 整个 VLA | | | | | |
| VLA + GRPO(基线) | — | 整个 VLA | | | | | |

要点:每行"基座"用该行 RL 实际加载的 ckpt 自测值(规避 4.3 基座混淆陷阱)。

### 表 2:消融(T7/T8 产出)

| 变体 | 改动 | task SR | 相对默认 Δ |
|:-----|:-----|:--------|:-----------|
| 默认(RLT + 预训练编码器) | — | __ | 0 |
| 编码器随机初始化 | 去 Phase-1 | __ | __ |
| RLT_a 编码器 | 全 token → 动作 token 瓶颈 | __ | __ |
| `β=0` / `ref_dropout=0`(TD3) | 去 BC / 去参考 dropout | __ | __ |

### 图 1:样本效率曲线

横轴**累计 env steps**,纵轴 SR(50-ep 离线评测点 + std 阴影),RLT×{TD3,GRPO,PPO} 与 VLA-RL 基线同图——这是研究算法差异的核心图。

### 图 2:训练曲线

横轴 iteration,纵轴在线 SR,**画全程**(含 iter150→300 的抖动),不截峰值——抖动本身支撑"末 N 轮均值做 model selection"的协议。

---

## Part 7 — 落地顺序

1. ~~**E1**~~ ✅ 已完成 —— RLT×GRPO 已解锁,主线从 1 个数据点变 2 个;下一步直接跑 T2 矩阵的 RLT×GRPO 行。
2. **T1 基座锚定**(算力小)→ 决定整个"RL 提升多少"的故事是否成立。
3. **E2 决策**:做小 actor PPO 还是退化为 RLT×{TD3,GRPO}。建议先按退化方案推进,PPO 用 VLA+PPO 基线占位,有余力再补。
4. **T2 主矩阵 + T4 task3 归因**并行——T4 可能改主表数字,要早做。
5. **E3 + T3**:VLA-RL 基线跑收敛并对照。
6. Tier 2 消融:多数能从 `results/rlt_training_TD3/` 已有 run 整理,先盘点缺哪个补哪个。

---

*本指南基于 `TECHNICAL_REPORT.md` 与 `results/` 现有产出撰写。E1 的代码改动我可以直接帮你做——需要的话告诉我。*
