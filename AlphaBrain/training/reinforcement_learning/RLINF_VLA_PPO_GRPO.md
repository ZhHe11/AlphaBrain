# RLinf 训练 VLA + PPO / GRPO 的逻辑

> 调研对象：RLinf-VLA(arXiv:2510.06710)+ RLinf 系统(arXiv:2509.15965)+ 官方文档与示例配置
> 目的：把"RLinf 到底是怎么训 VLA+PPO/GRPO 的"讲清楚,作为我们复刻的参照
> 数据来源：官方 quickstart、`examples/embodiment/config/maniskill_{ppo,grpo}_openvla.yaml`
> 日期：2026-05-22

---

## 0. 一句话总览

RLinf 把"VLA 强化学习"拆成**三个可独立放置的组件**——`env`(模拟器)/ `rollout`(生成)/ `actor`(训练)——rollout 阶段用**几百个向量化环境**喂一次**大批量 VLA 前向**,训练阶段用 **FSDP** 做 PPO/GRPO 梯度更新。

> **吞吐来自:大 batch(几百环境一次前向) + 组件解耦 + 流水线重叠。不是来自"一卡一 episode"。**
> 这正是我们之前那版多卡实现搞反的地方。

---

## 1. 三个组件与 GPU 放置

RLinf 把一次 VLA-RL 迭代拆成三类 worker:

| 组件 | 角色 | 后端 | 说明 |
|:-----|:-----|:-----|:-----|
| **`env`** | 模拟器 | ManiSkill / LIBERO / RoboTwin | ManiSkill 是 GPU 向量化(占 GPU);LIBERO 是 CPU 向量化 |
| **`rollout`** | 生成(Generation) | `huggingface`(也支持 vLLM/SGLang) | rollout 阶段跑 VLA 前向,产 action chunk |
| **`actor`** | 训练(Training) | `fsdp`(或 Megatron) | 持有可训 VLA,做 PPO/GRPO 梯度更新 |

每个组件都 `enable_offload: True`——不工作时把权重/显存卸到 CPU,需要时再加载。

**GPU 放置由一段 YAML 决定**(`examples/embodiment/config/maniskill_ppo_openvla.yaml`):

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor: 0-7      # 训练:FSDP 占满 8 卡
    env: 0-3        # 模拟器:GPU 0-3
    rollout: 4-7    # 生成:GPU 4-7
```

三种模式就是这段配置的不同写法:

- **collocated**:`actor,env,rollout: 0-3` —— 三者同一组卡,靠 offload 换入换出(quickstart 小配置用这个)。
- **disaggregated**:`env` 与 `rollout` 占互不重叠的卡。
- **hybrid**:上面那个 + `pipeline_stage_num: 2` 做流水重叠(见第 7 节)。

> 注意:`actor: 0-7` 表示训练阶段 FSDP 用满 8 卡;`env/rollout` 分 0-3 / 4-7 是 rollout 阶段的事。两阶段靠 offload 错峰,同一批卡复用。

---

## 2. Rollout 逻辑(关键)

### 2.1 向量化环境——几百个

| 配置 | PPO | GRPO |
|:-----|:----|:-----|
| `env.train.total_num_envs` | **128** | **256** |
| `env.train.max_episode_steps` | 80 | 80 |
| `env.eval.total_num_envs` | 16 | 32 |

**所有环境的观测拼成一个大 batch,一次 VLA 前向。** GPU 利用率来自这个 batch size——128~256 的批量前向才喂得饱 GPU。LIBERO/RoboTwin 的环境数随节点线性扩展(1/2/4 节点 = 64/128/256)。

### 2.2 动作分块(action chunk)

VLA 输出的是**动作块**而非单步动作:

```
chunk  c_t = (a_{t,1}, a_{t,2}, …, a_{t,C})      # C 个原子动作
```

交互循环:**Generation 根据当前观测推一个 chunk → Simulator 一次执行整个 chunk → 返回最后一帧观测**。OpenVLA-OFT 的一个 chunk 跨约 15 个仿真步(generation:execution ≈ 1:15)——即一次 VLA 前向换十几步仿真,这是 rollout 的成本结构。

### 2.3 Partial Reset(对 PPO 明显更好)

两种 reset 策略:

- **Fixed Episode Length**:所有子环境一起,到最大步数才统一 reset。
- **Partial Reset**:每个子环境一旦终止**立刻单独 reset**,不等别人。

论文实测 **Partial Reset 一致优于 Fixed**,显著提升样本效率——因为先完成的环境不必空等。

### 2.4 奖励

**稀疏二值**:基于任务成功(`success_once`)。chunk 级给奖励,在 action-level 形式下再分摊到每个原子动作。指标 `rollout/env_info/return`、`rollout/env_info/success_once`。

---

## 3. 核心抽象:Chunk → Action → Token 三级粒度

RLinf 算法可配置性的核心——**任何 RL 算法只需指定 reward 函数与 logprob 函数的"粒度",不写冗余代码**。三级层次:

```
Chunk c_t ─┬─ 原子动作 a_{t,1} ─┬─ token d_{t,1,1}
           │                    ├─ token d_{t,1,2}
           │                    └─ … d_{t,1,M}     （M 个 token 维）
           ├─ 原子动作 a_{t,2}
           └─ … a_{t,C}                            （C 个原子动作）
```

**log-probability 三档**:

| 粒度 | 公式 |
|:-----|:-----|
| chunk-level  | `π(c_t∣o_t) = ∏_{i=1}^{C} π(a_{t,i}∣o_t, a_{t,:i-1})` |
| action-level | `π(a_{t,i}∣o_t, a_{t,:i-1}) = ∏_{j=1}^{M} π(d_{t,i,j}∣o_t, d_{t,i,:j-1})` |
| token-level  | `π(d_{t,i,j}∣o_t, d_{t,i,:j-1})` |

配置项:`reward_type` / `logprob_type` / `entropy_type`。
- **PPO** 用 `action_level`(reward / logprob / entropy 都 action 级)。
- **GRPO** 用 `reward_type: action_level`、`logprob_type: token_level`、`entropy_type: token_level`。

---

## 4. PPO 逻辑

`algorithm` 段(`maniskill_ppo_openvla.yaml`)实际配置:

```yaml
algorithm:
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"
  normalize_advantages: True
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  value_clip: 0.2
  entropy_bonus: 0
  kl_beta: 0.0
  rollout_epoch: 1
  reward_type: action_level
  logprob_type: action_level
  entropy_type: action_level
```

### 4.1 Critic:参数共享的轻量值头

- 独立 critic 网络太贵 → **actor / critic 参数共享**。
- `add_value_head: True` —— 在 VLA 的**语言模型**上挂一个轻量 value head(沿用 RL4VLA 的做法)。
- **值估计粒度**:
  - chunk-level `V: 𝒮 → ℝ`(整块一个标量)
  - action-level `V: 𝒮 → ℝ^C`(每个原子动作一个值)—— **论文实测 action-level 一致更优**,是推荐做法。

### 4.2 优势与损失

- 优势:**GAE**,`γ=0.99`、`λ=0.95`,`normalize_advantages: True`。
- 损失:`actor_critic` —— clipped surrogate(策略)+ clipped value loss(值),`clip 0.2`、`value_clip 0.2`。
- `entropy_bonus 0`、`kl_beta 0`(PPO 默认不加熵 / KL)。
- 聚合 `loss_agg_func: token-mean`。

### 4.3 训练规模

`micro_batch_size 80` / `global_batch_size 640`(→ 8 个 micro-batch);actor `lr 1e-4`、`value_lr 3e-3`、`clip_grad 10`;FSDP + gradient_checkpointing。

---

## 5. GRPO 逻辑

`algorithm` 段(`maniskill_grpo_openvla.yaml`)实际配置:

```yaml
algorithm:
  adv_type: grpo
  loss_type: actor           # ← 无 critic
  loss_agg_func: "token-mean"
  group_size: 8
  normalize_advantages: True
  reward_type: action_level
  logprob_type: token_level
  entropy_type: token_level
  kl_penalty: kl
  kl_beta: 0.0
  clip_ratio_high: 0.28      # ← clip-higher
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0          # ← dual-clip
  gamma: 0.99
  gae_lambda: 0.95
```

### 5.1 无 critic + 分组采样

- `loss_type: actor` —— **不训 critic**,优势直接由组统计算。
- `group_size: 8` —— 同一初始状态采 **8 条轨迹**为一组。`total_num_envs 256 / group_size 8 = 32` 个不同初始状态,各 8 条。
- **组相对优势**(标准 GRPO):`A_i = (R_i − mean(组内 R)) / std(组内 R)`。

### 5.2 GRPO 的三个 VLA 专属改进

1. **Valid Action Mask**:rollout 按固定时长跑,但任务常提前完成 → **屏蔽"完成之后"的步**,不让它们贡献梯度。论文实测开启后 GRPO 更好。
2. **按有效步数归一 loss**:轨迹 `τ_i` 有 `T_i^{succ}` 个有效步,则其每步贡献乘 **`1/T_i^{succ}`** —— 防止长轨迹主导梯度。
3. **Success-rate Filter**(借鉴 DAPO 动态采样):**丢掉一组里"全成功"或"全失败"的组**——这种组优势全为 0、没有学习信号。论文实测加速收敛并提升性能。

### 5.3 dual-clip 与 KL

- 双侧 clip:`clip_ratio_high 0.28`(DAPO 的 clip-higher,鼓励探索)、`clip_ratio_low 0.2`、`clip_ratio_c 3.0`(dual-clip,负优势侧再加一层 clip)。
- `kl_penalty: kl`、`kl_beta 0.0`(KL 惩罚可用,默认权重为 0)。
- `micro_batch 40` / `global_batch 640`;actor `lr 1e-5`。

---

## 6. 训练步与权重同步

一次完整迭代(`runner.max_epochs: 1000` 个外层 epoch,每个 epoch):

```
1. Rollout    ──  几百个向量化 env，partial reset，批量 generation 采轨迹
                  （rollout_epoch: 1 = 每 epoch 采一次）
2. 算优势      ──  PPO: GAE / value head；  GRPO: 组相对归一 + valid mask + 过滤
3. Actor 更新  ──  FSDP，global_batch 640 拆成 micro_batch（80 / 40）做梯度更新
4. 权重同步    ──  actor → rollout，把更新后的 VLA 推到 generation 端
5. save_interval: 40 存档
```

- 训练后端 **FSDP + gradient_checkpointing**(占满 `actor` 的全部卡)。
- **权重同步**:论文称"权重更新充当 generation 与 training 之间的同步 barrier";走 NCCL 集合通信(不是 Ray)。配合 `enable_offload`——actor 训练时加载、训完卸出,rollout 反之。

---

## 7. Hybrid 流水线(吞吐关键)

`pipeline_stage_num: 2`(PPO/GRPO 配置都设了):

- 把一个模拟器实例切成 **子模拟器** `S⁽¹⁾, S⁽²⁾, …`,**错相位流水**:`S⁽¹⁾` 出观测送 Generation 算动作时,`S⁽²⁾` 同时在产下一帧观测 → **Simulator 与 Generation 重叠,填掉 GPU 气泡**。
- **两个数据队列**解耦生产者/消费者速率。
- rollout 跑完,同组 GPU 立刻 offload 切换去做训练,不留空窗。
- 收益:hybrid 相对 disaggregated **1.61–1.88×**(ManiSkill);README 称具身 hybrid 最高 **2.434×** 吞吐。

---

## 8. PPO vs GRPO 配置对照

| 项 | PPO | GRPO |
|:---|:----|:-----|
| `adv_type` | `gae` | `grpo` |
| `loss_type` | `actor_critic` | `actor`(无 critic) |
| critic / value head | 有(参数共享,action-level `V:𝒮→ℝ^C`) | 无 |
| 优势 | GAE,全局归一 | 组相对 `(R−μ)/σ`,`group_size 8` |
| `logprob_type` | `action_level` | `token_level` |
| clip | 单侧 `0.2` | dual-clip `high 0.28 / low 0.2 / c 3.0` |
| VLA 专属技巧 | Partial Reset | Valid Action Mask + `1/T^{succ}` 归一 + Success-rate Filter |
| `total_num_envs` | 128 | 256 |
| `micro/global batch` | 80 / 640 | 40 / 640 |
| actor `lr` | 1e-4 | 1e-5 |

---

## 9. 对我们(LIBERO + QwenOFT)的映射

**RLinf-VLA 在 LIBERO 上其实用的是 collocated 模式**(LIBERO 是 CPU 模拟器,不和 GPU 抢算力),提速来自"**向量化环境 + 批量 generation**",不是多卡 disaggregated(那主要给 ManiSkill 的 GPU 模拟器)。

我们当前 VLA-PPO vs RLinf 的差距:

| 维度 | 我们现状(`train_rl_vla_ppo.py`) | RLinf |
|:-----|:--------------------------------|:------|
| 并行环境数 | G=8,num_envs=4 | 128–256 向量化 |
| rollout 内核 | `vla_ppo_collect`,**环境串行 step**,每轮重 spawn | step-lock 批量前向 + partial reset |
| 单次前向 batch | 1–4 | 128–256 |
| 训练 | 单卡 + AdamW | FSDP 多卡 |
| 优势 | episode 级 GAE | action-level value head / 组相对 |

**复刻 RLinf 逻辑的要点**(给 LIBERO):

1. **rollout 改大批量 step-lock**:对所有活跃环境一次批量 VLA 前向 + 线程并行 step + `PersistentEnvPool` 常驻。我们仓库的 `algos/RLT_a/...rollout_fast.py` 已是这个模式(TD3 路径在用),VLA-PPO 直接借。
2. **把环境数拉到几十~上百**(机器有 128 核 CPU),G 相应放大——大 batch 才喂得饱 GPU。
3. **Partial Reset**:终止即单独 reset。
4. **PPO 值估计用 action-level value head**(`V:𝒮→ℝ^C`)。
5. **GRPO** 加 valid action mask、`1/T^{succ}` 归一、success-rate filter。
6. hybrid 流水线(`pipeline_stage_num` 思路)作为二期。

→ 先把"大批量 step-lock rollout"做对(单卡就能看到 GPU 被喂满),再谈多卡。顺序是 **batch → 流水线 → 多卡**。

---

## 10. 参考资料

- RLinf-VLA 论文:<https://arxiv.org/abs/2510.06710>
- RLinf 系统论文:<https://arxiv.org/abs/2509.15965>
- Quickstart — PPO Training of VLAs on ManiSkill3:<https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html>
- 示例配置:`RLinf/RLinf` 仓库 `examples/embodiment/config/maniskill_{ppo,grpo}_openvla.yaml`
- 配套权重:HuggingFace `RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood`、`RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood`
- 姊妹文档:`RLINF_STUDY.md`(去 Ray 可行性分析)

---

*本文基于 RLinf 公开论文/文档/示例配置整理,数值以其仓库为准;部分实现细节(PPO 内层 epoch 数等)未在公开材料明确,以代码为准。*
