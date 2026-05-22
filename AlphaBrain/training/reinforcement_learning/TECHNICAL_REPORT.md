# AlphaBrain 在线强化学习模块技术报告

> 适用范围：`AlphaBrain/training/reinforcement_learning/`
> 任务环境：LIBERO 仿真套件(`libero_spatial / object / goal / 10 / 90`)
> 基座模型(VLA)：`Qwenvl_OFT`(Qwen2.5-VL-3B + MLP 动作头)与 `PaliGemmaPi05`(SigLIP + Gemma 2B + 流匹配动作专家)

---

## 1. 概述

本模块在一个**已经过监督微调(SFT)的视觉-语言-动作模型(VLA)** 之上做在线强化学习,目标是在 LIBERO 仿真任务上进一步提升任务成功率(Success Rate, SR)。整套代码围绕一个核心思想组织:

> **把 VLA 庞大的隐藏状态压缩进一个信息瓶颈 token,再让一个轻量策略在这个紧凑状态上做 RL;或者直接把整个 VLA 当作策略端到端微调。**

围绕这个思想,模块同时提供了**两条技术路线**和**四类 RL 算法**,并配套了一整套面向大规模并行仿真的环境基础设施与评测流水线。

---

## 2. 整体架构与设计理念

### 2.1 两条技术路线

| 路线 | 编码器输入 | 瓶颈宽度 | 定位 |
|:-----|:-----------|:---------|:-----|
| **`RLT_a`**(首发) | VLA 的动作查询切片 `(B, M=chunk_len, H)` | 投影到 `D=256` | 工程默认路线,刻意偏离参考论文以支持多任务、跨 VLM 骨干 |
| **`RLT`**(后加) | VLA 完整 token 序列 `(B, L, H)` | 保持 VLA 隐藏维 `H`(如 2048) | 参考论文 Eq.1/2 的忠实复现路线,用于研究侧对照 |

两条路线**共享同一套** trainer / rollout / 评测基础设施,仅编码器/解码器类与 `--encoder_mode {action_token, rlt}` 开关不同。

`RLT_a` 偏离论文的两个核心动机:

1. **语言进入回路 → 多任务潜力**:参考论文从编码器输入中丢弃语言 token(假设单任务固定指令);`RLT_a` 喂的动作查询切片已经过 VLA 的图文注意力,语言信息被隐式烘焙进每个隐藏状态,同一个 actor 因此能覆盖多任务。
2. **跨 VLM 更易训练**:动作查询几乎每个 VLA 都以统一方式暴露(一次 `get_vla_action` 风格调用),`RLT_a` 编码器只需小适配器即可移植;`RLT` 编码器消费完整 token 序列,每接入一个新 VLA 都要写懂其 token 布局的特征抽取代码。

### 2.2 训练流水线

```
[Phase-1 编码器预训练]  →  [Phase-2 在线 RL]  →  [离线评测]
 run_rlt_pretrain.sh        run_rlt_rl.sh        run_eval_rlt{,_a}.sh
```

VLA 本身的微调不属于本目录流程,是一次性的上游步骤。

### 2.3 算法矩阵

模块区分两类策略对象:

- **小 actor RL**:VLA 冻结,在瓶颈状态上训练一个百万参数级的小 actor/critic。
- **整 VLA RL**:整个 VLA 就是策略 `π(a|s)`,端到端微调全部参数。

| 算法 | 入口 trainer | 策略对象 | critic | 在/离线 |
|:-----|:-------------|:---------|:-------|:--------|
| TD3(off-policy) | `train_rl_offpolicy.py` | 小 actor | 双 Q | 离线 |
| PPO(小 actor) | `train_rl_onpolicy.py` *(legacy)* | 小 actor | V(s) | 在线 |
| GRPO(小 actor) | `train_rl_grpo.py` | 小 actor | 无(组相对) | 在线 |
| VLA + PPO | `train_rl_vla_ppo.py` | 整个 VLA | 值头 | 在线 |
| VLA + GRPO | `train_rl_vla_grpo.py` | 整个 VLA | 无(组相对) | 在线 |

---

## 3. 代码结构

```
reinforcement_learning/
├── trainers/                    # 各 phase / 算法的入口
│   ├── train.py                 # 统一 CLI 入口,按 --phase 分发
│   ├── train_args.py            # 全部 CLI 参数定义
│   ├── train_pretrain.py        # Phase-1:RLT_a 编码器预训练
│   ├── train_rlt_pretrain.py    # Phase-1:RLT 参考路线预训练(含联合 VLA 微调)
│   ├── train_rl_offpolicy.py    # Phase-2:off-policy TD3(生产路径,1109 行)
│   ├── train_rl_onpolicy.py     # Phase-2:on-policy PPO(legacy)
│   ├── train_rl_grpo.py         # Phase-2:GRPO(小 actor)
│   ├── train_rl_vla_ppo.py      # 整 VLA + PPO
│   └── train_rl_vla_grpo.py     # 整 VLA + GRPO
├── algos/
│   ├── RLT/                     # 参考路线:全 token 编码器-解码器 + Pi05 推理适配器
│   ├── RLT_a/                   # 工程路线:动作-token 编码器 + TD3 actor/critic + 快速 rollout
│   └── VLAPPO/                  # 整 VLA 策略包装 + PPO/GRPO loss + rollout
├── common/                      # replay buffer / rollout / ckpt I/O / 孤儿进程治理
├── envs/                        # LIBERO 环境封装与并行 worker 池
└── eval/                        # 离线评测与分片聚合
```

---

## 4. CLI 入口与参数

`train.py` 是统一入口,按 `--phase` 分发到 7 个目标:

| `--phase` | 分发函数 | 用途 |
|:----------|:---------|:-----|
| `pretrain` | `run_pretrain()` | Phase-1:`RLT_a` 编码器-解码器重构预训练 |
| `pretrain_rlt` | `run_rlt_pretrain()` | Phase-1:`RLT` 参考路线预训练(可选联合 VLA 微调) |
| `rl` | `run_rl()` | 在线多卡 PPO(legacy) |
| `rl_offpolicy` | `run_rl_offpolicy()` | 离线 TD3,rollout/训练分卡(生产) |
| `grpo` | `run_rl_grpo()` | 组相对策略优化 |
| `vla_ppo` | `run_rl_vla_ppo()` | 整 VLA + PPO |
| `vla_grpo` | `run_rl_vla_grpo()` | 整 VLA + GRPO |

关键参数按功能分组(默认值见第 11 节超参表):

- **I/O**:`--ckpt_path`(必填,SFT 基座)、`--encoder_path`(预训练好的编码器)、`--output_dir`、`--seed`、`--use_wandb`。
- **任务**:`--suite`、`--task_id` / `--task_ids` / `--all_tasks`。
- **编码器**:`--encoder_mode {action_token, rlt}`、`--bottleneck_dim`、`--encoder_layers`、`--encoder_heads`、`--decoder_layers`、`--max_len`、`--image_only` / `--all_tokens`、`--drop_action_tokens`。
- **Phase-1**:`--pretrain_n_obs`、`--pretrain_epochs`、`--pretrain_lr`、`--pretrain_batch_size`、`--demo_config`、`--alpha_vla`、`--lr_vla`。
- **Phase-2 通用**:`--G_per_task`、`--group_size`、`--num_envs_per_task`、`--use_steplock`、`--max_iter`、`--gamma`、`--eval_interval`、`--save_interval`。
- **TD3 专用**:`--buffer_capacity`、`--buffer_warmup`、`--warmup_iters`、`--utd_ratio`、`--td_batch_size`、`--tau`、`--beta`、`--actor_update_freq`、`--target_noise_std`、`--target_noise_clip`、`--fixed_std`、`--ref_dropout`。
- **PPO/GRPO 专用**:`--ppo_epochs`、`--clip_eps`、`--gae_lambda`、`--vf_coef`、`--grpo_kl_coef`、`--ref_update_interval`、`--micro_batch`。
- **多卡切分**:`--rollout_gpus`、`--train_gpu`。

---

## 5. Phase-1:编码器预训练

Phase-1 把冻结 VLA 的隐藏状态压缩进一个 RL token,并用重构损失保证瓶颈"信息无损"。两条路线各有一个 trainer。

### 5.1 `RLT_a` 路线 — `train_pretrain.py`

三段式流程:

1. **快速观测采集**(纯环境,无 VLA 前向):并行 reset 环境 + 每次 reset 走 `pretrain_steps_per_reset` 步随机动作,收集 `(image, instruction)`。
2. **批量抽取动作查询**(一次性 GPU 工作):VLA 前向 `image + instruction → action_queries (B, chunk_len, H)`,抽完即把冻结 VLA 移出显存。
3. **在缓存张量上训练编码器-解码器**:对缓存的动作查询做若干 epoch 的 MSE 重构训练,损失下降即保存最优编码器到 `checkpoints/pretrain_best/encoder.pt`。

此路线全程冻结 VLA,无联合微调。

### 5.2 `RLT` 路线 — `train_rlt_pretrain.py`

参考论文 Algorithm 1 line 3 的联合目标:

```
ϕ, θ_vla = argmin  L_ro(ϕ) + α · L_vla(θ_vla)
```

- `L_ro`:RL token 的重构损失(见 6.2 节)。
- `L_vla`:VLA 自身的模仿损失(`Qwenvl_OFT.forward` 返回的 L1 动作回归损失)。
- `α = --alpha_vla`:`α = 0` 保持 VLA 冻结(默认);`α > 0` 解冻 VLA、加入其模仿损失,两个参数组分别用 `pretrain_lr`(编码器-解码器)与 `lr_vla`(VLA)优化。

数据路径有两种:

- **路径 A — 示范数据(论文忠实)**:`--demo_config` 指向一个 LeRobot 混合数据集 YAML,复用 SFT 流水线的 `LeRobotMixtureDataset`;`α > 0` 时用其中的动作标签算 `L_vla`。
- **路径 B — 随机 rollout 回退**:未给 `--demo_config` 时退回 `RLT_a` 的随机动作 rollout 观测采集,此时 `α` 必须为 0(无动作标签)。这是相对论文的一个偏差,help 文本中已注明。

---

## 6. 编码器/解码器架构

### 6.1 `RLT` 参考编码器-解码器(`algos/RLT/encoder_decoder.py`)

**`RLTokenEncoder`(Eq.1)**:`z_rl = g_φ([z_{1:M}, e_rl])_{M+1}`

- 输入 VLA 末层 token 嵌入 `(B, M, H)`,拼接一个可学习的 `e_rl`(形状 `(1,1,H)`,初始化尺度 0.02)得到 `(B, M+1, H)`。
- 过一个 `nn.TransformerEncoder`(默认 2 层,pre-norm,GELU,`dim_feedforward = 4H`),取 `e_rl` 位置的输出 → `z_rl (B, 1, H)`。
- **无逐 token 投影**:`z_rl` 保持 VLA 隐藏维 `H`,瓶颈完全来自 "M 个 token → 1 个 token" 的坍缩。
- `key_padding_mask` 一路透传,使自注意力忽略分词器右侧 padding。

**`RLTokenDecoder`(Eq.2)**:

```
L_ro = E_D[ Σ_{i=1}^M ‖ h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i) ‖² ]
```

- `nn.TransformerDecoder`:`memory = z_rl`,目标流为右移、因果掩码的 `[BOS, sg(z_1), …, sg(z_{M-1})]`,叠加可学习位置嵌入(在 `--max_len` 处分配,默认 4096)。
- `h_φ` 是末端 `Linear(H, H)`。
- 重构损失对**停梯度(stop-gradient)** 的 VLA 嵌入做 MSE,梯度只经 `z_rl` 回流入编码器;padding 位置由掩码排除:`loss = Σ(valid·‖ẑ−z‖²) / (Σ(valid)·H)`。

### 6.2 `RLT_a` 工程编码器-解码器(`algos/RLT_a/action_token_encoder_decoder.py`)

- **`ActionTokenEncoder`**:输入动作查询 `(B, M, H)`,拼一个 cls token 后过 2 层自注意力,再经 `Linear(H → D)` 投影到瓶颈 `D=256`,输出 `z_rl (B, 1, 256)`。**额外的瓶颈投影**是相对论文的有意偏差(论文保持 `1×2048`),目的是缩小下游小 actor/critic 的 MLP。
- **`ActionTokenDecoder`**:`Linear(D → H)` 扩回隐藏维,作为前缀拼到 teacher-forcing 的目标序列前,加位置嵌入后过因果掩码自注意力栈;`L_ro = MSE(reconstructed, sg(action_queries))`。这是"前缀 + 因果自注意力",架构上不同于论文的编码器-解码器交叉注意力,但功能等价。

### 6.3 VLA 隐藏状态抽取

- **Qwen(`vla_features.py`)**:`get_vla_hidden_states` 组装带动作占位符的提示词,VLM 前向取 `hidden_states[-1]`;`image_only=True`(严格参考模式,默认)只保留图像 token,否则保留图文 token 并可丢弃动作占位符。变长序列经 `compact_by_mask` 左对齐打包成稠密 `(B, M_max, H)` + `key_padding_mask`。
- **Pi05(`vla_features_pi05.py`)**:Pi05 无在流动作 token,隐藏状态取 Gemma 对图文前缀的末层输出;由于 4D 加性掩码与 Flash-Attention 不兼容,前向强制 `eager` 注意力实现。
- **Pi05 推理适配器(`pi05_inference.py`)**:`get_pi05_rl_state_and_action` 在一次调用里融合 "Gemma 前缀前向(产 `z_rl`)+ 流匹配扩散循环(产参考动作)",避免对 VLM 做两次前向。

---

## 7. Phase-2:Off-Policy TD3(生产路径)

`train_rl_offpolicy.py`(1109 行)是生产级的 Phase-2 训练器,实现 **rollout/训练分卡的分布式 off-policy TD3**。

### 7.1 系统结构

- **Rollout GPU(如 0–4)**:各持一份冻结 VLA,通过持久环境池并行采集 episode。
- **训练 GPU(如 5)**:跑带梯度的 actor/critic TD3 更新。
- **中心化 replay buffer(CPU)**:所有 rollout GPU 写入,训练 GPU 采样。

这一解耦把慢速仿真与 GPU 密集的训练更新分离;rollout 在后台线程持续运行,权重每若干次更新同步一次。

### 7.2 训练循环

```python
for iteration in range(1, max_iter + 1):
    all_episodes = rollout_stats_queue.get()      # 阻塞拿 rollout 数据
    if iteration <= warmup_iters:                 # 纯 VLA rollout 预热填 buffer
        ...
    if replay_buffer.is_ready(buffer_warmup):     # buffer 就绪后开始 TD 更新
        n_updates = max(1, int(n_pushed * utd_ratio / batch_size))
        for td_step in range(n_updates):
            critic_update()                       # 每步更新 critic
            if (td_step + 1) % actor_update_freq == 0:
                actor_update(); soft_update_targets()   # 延迟更新 actor + 目标网
```

### 7.3 TD3 三大要素

**双 Q critic + 目标策略平滑**:

```python
with torch.no_grad():
    next_action = target_actor(next_state, deterministic=True)
    noise = (torch.randn_like(next_action) * target_noise_std)   # σ=0.2
                 .clamp(-target_noise_clip, target_noise_clip)   # ±0.5
    next_action = (next_action + noise).clamp(-1, 1)
    next_q = torch.min(*target_q_critic(next_state, next_action))
    target = reward + gamma * next_q * (1 - done)                # γ 按 actor_chunk 折算
critic_loss = MSE(q1, target) + MSE(q2, target)
```

**延迟 actor 更新 + BC 正则**(参考论文 Eq.5):

```python
action  = actor(state, deterministic=False)        # 重参数采样,固定 std=0.1
q_val   = q_critic.q1_forward(state, action)
bc_pen  = ((action - vla_action) ** 2).sum(dim=(-2,-1)).mean()
actor_loss = -q_val.mean() + beta * bc_pen          # β 默认 1.0
```

actor 每 `actor_update_freq=2` 次 critic 更新才更新一次;目标网用 Polyak 软更新 `θ' ← (1−τ)θ' + τθ`,`τ=0.005`。

### 7.4 Actor / Critic 架构(`action_token_actor_critic.py`)

- **`ActionTokenActor`**:输入 `(z_rl 256, vla_action C×7, prop_state 8)`,3 层 MLP(隐藏 256)直接输出整段动作 chunk(**非残差**);BC 约束放在 loss 里而非结构里。带 **50% 参考动作 dropout**(`ref_dropout`),防止 actor 退化为"照抄 VLA"。输出固定 std=0.1 的高斯分布。
- **`ActionTokenQCritic`**:TD3 双 Q,两套独立权重的 3 层 MLP(隐藏 256);目标用 `min(Q1, Q2)` 抑制高估。
- 状态构造 `x = (z_rl, s_p)`,`s_p` 为 8 维本体感(eef 位置 3 + 轴角 3 + 夹爪 2)。

### 7.5 快速 Rollout(`action_token_rollout_fast.py`)

**Step-lock(步锁)批量 rollout** 是速度关键:所有并行环境锁步推进,每个 chunk 内——

1. **一次** 批量 VLA 前向(覆盖全部活跃环境);
2. 批量编码器 + actor 前向;
3. 所有环境在并行线程里执行该 chunk;

典型耗时:VLA 前向 ~50%、编码器+actor ~10%、环境步进 ~30%。多任务时把各任务的环境合并进同一批 VLA 前向,有效 batch = `任务数 × 每任务环境数`。

**Chunk 子采样(stride=2)**:每个 8 步 chunk 在位置 `[0,2,4,6]` 各构造一条 transition,跨 chunk 拼接动作切片后压入 replay buffer,显著放大数据利用率。

### 7.6 Replay Buffer(`common/replay_buffer.py`)

定容环形缓冲。每条 transition 存 `(rl_token, vla_action, action_taken, reward, next_rl_token, next_vla_action, done, prop_state, next_prop_state, task_id)`。支持均匀采样与**按任务分层均衡采样**(`sample_balanced`,借助 `task_id → 位置列表` 索引保证各任务对梯度的等量贡献)。

---

## 8. 在线算法:PPO 与 GRPO

### 8.1 小 actor 的 PPO / GRPO

二者复用同一套小 actor + 编码器,rollout 用 `action_token_collect_group` 按初始状态分组采集。

- **PPO(`train_rl_onpolicy.py`,legacy)**:带 `ActionTokenCritic` 值头,用 GAE 算优势,clip 替代目标 + 值损失。
- **GRPO(`train_rl_grpo.py`)**:**去掉 critic**,优势改为**组相对回报归一化**——同一初始状态的一组 episode,`A_ep = (R_ep − μ_group) / σ_group`。维护一个冻结的参考 actor(深拷贝,每 `ref_update_interval` 刷新)做 KL 惩罚。

### 8.2 整 VLA 的 PPO / GRPO(`algos/VLAPPO/`)

此处**整个 VLA 就是策略**,动作头即 `π(a|s)`,全部参数可训。

- **`VLAPolicy`**:无状态包装器,暴露 `forward_mean / sample / log_prob_of_with_mean`;策略为各向同性固定 std(默认 0.1)高斯,`log π(a) = −½[Σ(a−μ)²/σ² + D·log2π + D·logσ²]`。
- **`VLAValueHead`**:对动作查询 `(B, C, H)` 沿 chunk 维均值池化后过 2 层 SiLU MLP → 标量 `V(s)`(仅 PPO 用)。
- **`vla_ppo_loss`**:稀疏奖励(只在终止步给),逐 episode 算 GAE → 优势归一化 → 微批(`micro_batch`)重新前向 VLA → PPO 截断代理损失 + 截断值损失,`L = L_pg + vf_coef·L_vf`。
- **`vla_grpo_loss`**:按 `state_idx` 分组算组相对优势(组内 < 2 条 episode 则优势置 0),PPO 截断代理 + 对参考 VLA 的 KL 惩罚(k3 估计:`KL ≈ E[exp(Δ) − Δ − 1]`),`L = L_pg + kl_coef·KL`,`kl_coef` 默认 0.04。
- **`vla_ppo_collect`**:`no_grad` 下批量分块采集;每步**先记录再执行**,保证 VLA 输入与产生该 transition 的动作对齐。

整 VLA RL 是单卡路径,显存开销大(可训 VLA + Adam ~50GB,GRPO 还需 ~8GB 参考 VLA),因此 PPO 更新阶段必须微批重新前向。

### 8.3 两类策略对象对比

| 维度 | 小 actor RL(`RLT_a`) | 整 VLA RL(`VLAPPO`) |
|:-----|:----------------------|:---------------------|
| VLA | 冻结 | 可训 |
| 策略 | `ActionTokenActor`(百万级) | VLA 动作头(十亿级) |
| 显存 | 每卡 ~2–3GB | ~50GB(+GRPO 参考 ~8GB) |
| 并行 | 多卡(Accelerate) | 单卡 |
| 速度 | 快 | 慢(每个 PPO epoch 重前向 VLA) |

---

## 9. 环境与基础设施

### 9.1 LIBERO 环境与 IPC

LIBERO 装在独立 conda 环境,通过子进程封装。两种 IPC worker:

| Worker | 协议 | 图像编码 | 开销 |
|:-------|:-----|:---------|:-----|
| `libero_env_worker.py`(pipe-IPC) | stdin/stdout + msgpack | PNG(变长) | 高(PNG 压缩/解压) |
| `libero_env_worker_fast.py`(socket-IPC) | Unix socket + msgpack | 裸 numpy 字节(256×256×3=196KB 定长) | 低(仅 memcpy) |

socket-fast worker 还有两个关键优化:**同任务环境复用**(同任务只 `reset + set_init_state`,换任务才重建 MuJoCo 环境)与 **`step_chunk` 批量步进**(一个 chunk 的多步动作一次往返完成)。`PersistentEnvPool` 管理 N 个 socket worker 贯穿整个训练生命周期,带 socket 超时崩溃自愈(重启子进程并自动重放上次 reset)。

环境观测为 `{primary_image, wrist_image, state(8 维本体感)}`;**奖励稀疏二值**——仅任务成功时给 1.0;各套件最大步数不同(`libero_goal` 320 步等)。动作 7 维,执行前夹爪维做二值化与符号变换。

### 9.2 孤儿进程治理(`common/parent_death.py`)

历史问题:bash 对 `train.py` 发 SIGKILL → 训练进程被杀但 LIBERO 环境 worker 孤儿化、长期占用 GPU EGL 上下文死锁。`set_die_with_parent` 用 Linux `prctl(PR_SET_PDEATHSIG)` 让 worker 在父进程死亡时收到信号自杀,在 `train.py` 与两个 worker 启动早期调用。

### 9.3 检查点 I/O(`common/ckpt_io.py`)

`save_rlt_checkpoint` 把 encoder / actor / critic 分别存为独立 `.pt`,目录按 `{phase}_iter_{iteration:05d}` 命名,只保存非 None 的模块。

---

## 10. 评测流程

`eval/` 提供离线评测,与训练分离以拿到论文级数字。

- **`eval_libero.py`**:`RLT_a`(action-token)路径,编码器消费动作查询;确定性评测(固定种子、无探索噪声)。
- **`eval_libero_rlt.py`**:`RLT` 路径,编码器消费压缩后的图像隐藏状态(`compact_by_mask` + `key_padding_mask`),因此单独走 `run_rlt_inference`。
- **分片与聚合**:评测任务在多 GPU 间轮询切分,各分片产 `shard_*.json`,`aggregate_shards.py` 合并各任务 SR 得 `summary.json`。
- **指标**:逐任务成功率 `SR_task = 成功 episode 数 / 总 episode 数`(LIBERO 奖励 ≥ 0.5 判成功),总体 SR 为各任务 SR 的均值;训练内嵌评测用 20 episode 监控,离线评测默认每任务 50 episode。

---

## 11. 关键超参数汇总

> 下表为 `train_args.py` 中的 CLI 默认值;具体启动脚本可能覆盖部分项。

| 类别 | 参数 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| 编码器(RLT) | `hidden_dim` | 2048 | 取自 VLA 骨干 |
| | `encoder_layers` / `decoder_layers` | 2 / 2 | Transformer 层数 |
| | `encoder_heads` | 4 | 注意力头数 |
| | `max_len` | 4096 | 解码器位置嵌入长度 |
| 编码器(RLT_a) | `bottleneck_dim` | 256 | 瓶颈宽度 `D` |
| Phase-1 | `pretrain_batch_size` | 32 | |
| | `pretrain_lr` | 1e-4 | 编码器-解码器学习率 |
| | `pretrain_epochs` | 50 | |
| | `pretrain_n_obs` | 2000 | 预训练观测数 |
| | `alpha_vla` | 0.0 | >0 启用联合 VLA 微调 |
| | `lr_vla` | 5e-6 | VLA 微调学习率 |
| TD3 | `buffer_capacity` | 100000 | replay buffer 容量 |
| | `buffer_warmup` | 512 | 起训最小 buffer 大小 |
| | `warmup_iters` | 5 | 纯 VLA rollout 预热轮数 |
| | `td_batch_size` | 256 | TD 采样批大小 |
| | `utd_ratio` | 2.0 | 更新-数据比 |
| | `tau` | 0.005 | 目标网软更新系数 |
| | `beta` | 1.0 | BC 正则系数 `‖a−ã‖²` |
| | `actor_update_freq` | 2 | actor 延迟更新间隔 |
| | `target_noise_std` / `clip` | 0.2 / 0.5 | 目标策略平滑噪声 |
| | `fixed_std` | 0.1 | actor 探索固定 std |
| | `ref_dropout` | 0.5 | 参考动作 dropout |
| PPO/GRPO | `gamma` | 0.99 | 折扣因子 |
| | `gae_lambda` | 0.95 | GAE λ |
| | `clip_eps` | 0.2 | PPO 截断 ε |
| | `vf_coef` | 0.5 | 值损失权重 |
| | `grpo_kl_coef` | 0.04 | GRPO KL 惩罚权重 |
| | `ppo_epochs` | 10 | 每迭代更新轮数 |
| 通用 | `max_iter` | 500 | 最大迭代数 |
| | `eval_interval` / `save_interval` | 10 / 50 | 评测 / 存档间隔 |
| 环境 | LIBERO 分辨率 | 256×256 | |
| | chunk_len(LIBERO) | 8 | VLA 动作 chunk |
| | actor_chunk_len | 4 | actor 重规划 chunk |
| | prop 维度 / 动作维度 | 8 / 7 | |

---

## 12. 已知偏差与局限

相对参考论文(RL Token, Physical Intelligence, 2026)与生产可用性,模块有意识地保留若干偏差:

- **`RLT_a` 的偏差**:编码器输入用动作查询切片而非完整图像 token;额外 `H→256` 瓶颈投影;Phase-1 用随机 rollout 观测而非任务示范;无 Phase-1 联合 VLA 微调。`RLT` 路线修正了前三点,但仍在 Phase-1 数据、基座 VLA、无人在回路上与论文不同。
- **基座差异**:论文用 `π0.6`(SigLIP + Gemma 4B + 860M 扩散动作专家),本模块在 QwenOFT / Pi05 上构建。
- **环境差异**:论文为 14 维双臂真实世界 50Hz,本模块为 7 维 LIBERO 仿真。
- **缺失能力**:人在回路、关键阶段切换、策略交接学习均为论文独有,当前发布为纯自主仿真配方。
- **`train_rl_onpolicy.py`** 标注为 legacy,生产路径是 off-policy TD3。
- **`RLT_a × Pi05`** 尚未打通(Pi05 流匹配头无"动作查询切片"等价物),已在路线图上。

后续计划:扩展在线 RL 算法覆盖(GRPO、PPO 等)、改进正样本采集/筛选/清洗工具、发布更强且更可复现的基线。

---

*本报告基于代码现状撰写,API、配置与数值可能随版本变化。详细每条偏差见各算法目录下的 `README.md`。*
