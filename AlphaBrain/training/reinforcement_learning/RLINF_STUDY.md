# RLinf 调研报告：多卡 Rollout 架构借鉴与"去 Ray"可行性分析

> **调研对象**：RLinf — Reinforcement Learning Infrastructure for Embodied & Agentic AI
> （GitHub: `RLinf/RLinf`；系统论文 arXiv:2509.15965；VLA 论文 arXiv:2510.06710）
> **要回答的两个问题**：
> 1. RLinf 是怎么用多卡做 RL 训练 / 数据采集的，有什么可借鉴；
> 2. 我们能否在**不引入 Ray** 的前提下复用它的方法。
> **落点**：本目录 VLA-PPO（`train_rl_vla_ppo.py`）当前是单卡，需要多卡化。
> **日期**：2026-05-21

---

## 0. 结论速览（TL;DR）

1. **瓶颈实测确认**：当前单卡 VLA-PPO 一个 iter ≈ **244 s**，其中 rollout 采集 **214 s（88%）**、PPO 更新仅 30 s；采集期间训练卡 GPU 利用率约 **1%**。问题 100% 在数据采集，不在梯度更新。

2. **RLinf 的方法值得借鉴**：它把 RL 拆成 **Rollout（Simulator + Generation）** 与 **Training** 两类组件，提供 **collocated / disaggregated / hybrid** 三种 GPU 放置模式；多卡采集的本质是"按 worker 切分任务 + NCCL/FIFO 队列通信"。其 hybrid 模式用"子模拟器流水线"把仿真与推理重叠，消除 GPU 气泡。

3. **可以不用 Ray**：在 RLinf 自己的设计里，Ray **只负责**"跨节点拉起进程 + 派发函数调用"。设备分配（RLinf 明确"不依赖 Ray"）、通信（NCCL / cudaIPC / Gloo）、数据通道（FIFO Channel）、权重同步**本来就不经过 Ray**。单节点 8 卡场景，Ray 的那点职责可被 `torchrun` / `torch.multiprocessing` 完全替代。

4. **我们已经有现成实现**：仓库里的 `train_rl_offpolicy.py`（TD3 生产路径）**本身就是一个"去 Ray 的 disaggregated RLinf"**——它已经做了 rollout/训练分卡、每卡一份 VLA、step-lock 批量 rollout、权重同步。它甚至是**单进程多线程**模型，连多进程都没用，更没用 Ray。

→ **建议**：照搬 `train_rl_offpolicy.py` 的单进程多线程分卡模式，把 VLA-PPO 的 rollout 铺到多卡，全程不引入 Ray。8 卡预计把 iter 从 244 s 压到约 65 s（≈ 3.7×）。

---

## 1. 调研背景

`train_rl_vla_ppo.py` 把整个 4.1B 的 VLA 当策略做端到端 PPO，是单卡路径。实测一个迭代的时间结构（task 0，G=8，num_envs=4，1traj ckpt）：

| 阶段 | 实测耗时 | 占比 | GPU 利用率 |
|:-----|:---------|:-----|:-----------|
| rollout 采集（8 episode） | 214 s | **88%** | ~1% |
| PPO 更新（2 epoch，micro_batch=2） | 30 s | 12% | 高（峰值显存 ~33 GB） |

rollout 是 `@torch.no_grad()` 的纯推理，瓶颈在 LIBERO 子进程仿真；采集期间训练卡几乎闲置。**梯度更新只占 12%，单卡完全够用——要打的就是采集。** 因此本调研聚焦：如何把数据采集铺到多卡，并评估能否绕开 Ray。

---

## 2. RLinf 是什么

RLinf 是面向具身/智能体 AI 的开源 RL 基础设施，目标是"在不改代码的前提下把 RL 训练扩展到大量 GPU 节点"。两篇配套论文：

- **系统论文**（arXiv:2509.15965）："Flexible and Efficient Large-scale RL via **Macro-to-Micro Flow Transformation**"——讲调度抽象与执行引擎。
- **VLA 论文**（arXiv:2510.06710）："RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training"——讲它在 VLA 上的具体配方。

它支持的算法很广（PPO、Async PPO、GRPO、Reinforce++、SAC、CrossQ、IQL、RLPD 等），后端可选 FSDP+HF/vLLM/SGLang（快速原型）或 Megatron+vLLM/SGLang（大规模）；具身侧支持 OpenVLA、OpenVLA-OFT、π₀、π₀.₅、GR00T，仿真器支持 ManiSkill、LIBERO、IsaacLab、RoboTwin。

**和我们的相关性**：RLinf-VLA 跑的正是 "LIBERO + OpenVLA-OFT + PPO/GRPO"——和我们 `QwenOFT + LIBERO + VLA-PPO` 几乎同构。它报告的最终成绩：LIBERO 98.11% SR、ManiSkill 97.66%、RoboTwin 84.63%。

---

## 3. RLinf 的系统架构

### 3.1 逻辑/物理解耦：Macro-to-Micro Flow（M2Flow）

RLinf 的核心抽象是 **M2Flow**：开发者用"宏观"的命令式接口描述 RL 工作流（组件之间怎么通信、怎么同步），系统再"自动把这个逻辑流转换成针对具体负载和硬件的细粒度执行计划"——在空间和时间两个维度上做调度（时间复用、空间流水、混合调度）。

对我们的意义：**"工作流描述"和"GPU 放置/调度"是两层**。同一份 PPO 工作流，可以被映射成单卡、分卡、流水线等不同物理执行。我们要借的就是后一层——物理执行的几种模式。

### 3.2 三种 GPU 放置模式

RLinf 把 GPU 资源在 **Rollout 阶段（Generation 推理动作 chunk + Simulator 执行返回观测）** 和 **Training 阶段（消费轨迹、更新参数）** 之间分配，有三种模式：

| 模式 | 做法 | 问题 |
|:-----|:-----|:-----|
| **Collocated（同置）** | 所有组件共享同一组 GPU，按时间片轮流独占；空闲组件 offload 到 CPU 内存 | 组件互相等待 → 资源浪费、扩展性受限 |
| **Disaggregated（分离）** | 每个组件独占互不重叠的 GPU 分区，可并发执行 | 组件间有依赖 → 某些 GPU 阶段性闲置（如训练卡在 rollout 期间完全空闲） |
| **Hybrid（混合）** | 部分组件跨 GPU 流水线执行；某阶段做完后该组 GPU 被换出给后继任务复用 | RLinf 的主推方案，下节详述 |

> 我们当前的单卡 VLA-PPO 本质是 collocated 的极端退化（1 张卡，串行）；下面要做的多卡采集对应 **disaggregated**。

### 3.3 Hybrid 模式：子模拟器流水线（RLinf 的核心贡献）

朴素的 disaggregated 有个浪费：Generation（推理）和 Simulator（仿真）有严格的先后依赖——仿真出观测 → 推理出动作 → 喂回仿真——天然串行，互相等待时对方的 GPU 是气泡。

RLinf 的 hybrid 模式把**单个模拟器实例切成多个子模拟器** S⁽¹⁾, S⁽²⁾, …, S⁽ᵏ⁾，错相位流水：

```
t0:  S⁽¹⁾ 产 o₀⁽¹⁾ ─→ Generation 算 a₀⁽¹⁾
     S⁽²⁾ 同时产 o₀⁽²⁾（并行）
t1:  a₀⁽¹⁾ 就绪 ─→ 喂回 S⁽¹⁾ 产 o₁⁽¹⁾
     与此同时 o₀⁽²⁾ 进 Generation 算 a₀⁽²⁾
...  Simulator 和 Generation 始终都有活干，气泡被填平
```

关键机制：

- **两个数据队列**解耦生产者/消费者速率——"平滑流水线、均衡负载、几乎消除性能瓶颈"。
- **GPU 复用**：rollout 跑完后，原本做仿真/推理的那组 GPU 立刻转去做训练，不留空窗。
- 配置上具身 RL 用 `pipeline_stage_num = 2`，在 rollout 和 env 之间开两级流水重叠。

效果：hybrid 相对 disaggregated 基线有 **1.61×–1.88×** 加速（ManiSkill+OpenVLA），README 称具身 hybrid 最高 **2.434×** 吞吐。

> 对我们的启示：disaggregated 多卡采集是第一步（确定收益）；子模拟器流水线（VLA 前向 ↔ 环境步进重叠）是可选的进阶优化。

### 3.4 编程模型：Worker / Cluster / Channel

RLinf 的代码抽象（`rlinf` 包下有 `workers/`、`scheduler/`、`runners/`、`hybrid_engines/` 等）：

- **Worker**：一个"远程进程 / 计算单元"。继承 `Worker` 即获得跨节点运行、与其它 worker 通信的能力，并**自动拿到** `MASTER_ADDR / MASTER_PORT / RANK / LOCAL_RANK / WORLD_SIZE` 等环境变量。
- **Cluster + Placement**：`MyWorker.create_group().launch(cluster, placement)`；`Cluster.allocate(...)` 决定每个 worker 落在哪个节点、哪些 GPU 上。
- **Channel**：worker 之间的 **FIFO 队列**（`create_channel` / `connect_channel`，`put/get/get_batch`），用于生产者-消费者式数据交换。**注意 worker 间通信不经过主控进程**。
- **CollectiveGroup**：Worker 本身不直接做通信，而是委托给 `CollectiveGroup`——它才是真正搬数据的两端点。

### 3.5 通信层：NCCL / cudaIPC / Gloo（与 Ray 无关）

这是回答"去 Ray"问题的关键。RLinf 的数据通信**完全不走 Ray**：

- **自适应后端选择**：根据通信双方的 worker/数据放置，自动选 **NCCL（GPU↔GPU）**、**零拷贝 cudaIPC（节点内/同 GPU）**、**Gloo（CPU）**。
- 为收、发各建独立的 process group（NCCL 给 GPU、Gloo 给 CPU），形成单向通道；用 **TCP rendezvous** 协调端口与同步；每个方向有独立 CUDA stream 的工作队列保序。
- 收发 API 支持张量、张量 list/dict、任意可 pickle 对象。

**一句话**：RLinf 里搬模型权重、搬轨迹的活，是 **torch.distributed（NCCL/Gloo）+ cudaIPC + FIFO Channel** 干的，**不是 Ray 的 object store**。

---

## 4. Ray 在 RLinf 里到底承担什么

把上面拼起来，Ray 的职责面其实**很窄**。系统论文原话：

> Ray handles "**cluster management, launching worker processes on remote nodes, and dispatching worker function executions**."

而且 RLinf 还**主动绕开了 Ray 的一部分**：

> "Ray only supports rigid packed-style ... or spread-style ... resource allocation ... which are not flexible enough" — 因此 "**RLinf does not rely on Ray for device allocation to workers**"（设备分配是 RLinf 自己实现的）。

归纳 Ray 在 RLinf 里**只做三件事**：

| Ray 实际负责 | Ray **不**负责（RLinf 自己做） |
|:-------------|:-------------------------------|
| ① 集群管理（把多台机器组成一个 Ray 集群） | 设备分配（哪个 worker 用哪些卡）—— RLinf 自己实现 |
| ② 在**远程节点**上拉起 worker 进程 | 通信 —— NCCL / cudaIPC / Gloo |
| ③ 派发 worker 的函数调用（控制面 RPC） | 数据通道 —— 自建 FIFO Channel |
|  | 权重同步 —— torch.distributed 集合通信 + 同步 barrier |

**所有性能关键路径（放置灵活性、通信、Channel、权重同步）在 RLinf 内部本来就是非 Ray 的。** Ray 剩下的就是一个"多节点进程启动器 + 控制面 RPC"。

---

## 5. 去 Ray 可行性分析（核心问题）

### 5.1 结论：可以，单节点下 Ray 完全可被替代

我们的场景是**单节点 8×80GB**。Ray 在 RLinf 里的三件事，逐项都有标准等价物：

| RLinf 用 Ray 做的事 | 单机去 Ray 等价物 | 说明 |
|:--------------------|:------------------|:-----|
| 集群管理（多节点组集群） | **不需要** | 单机无集群可管 |
| 在远程节点拉起 worker 进程 | `torchrun --nproc_per_node=N` 或 `torch.multiprocessing.spawn` | 单机只需在本机拉 N 个进程并各绑一张卡 |
| 自动注入 `RANK/WORLD_SIZE/MASTER_*` | `torchrun` 原生注入；`mp.spawn` 手动设 | Worker 拿到的就是这套标准 env |
| 派发函数调用（控制面 RPC） | 一个 rank 同步的主循环 + `mp.Queue` / `dist.barrier` | on-policy PPO 控制流是规整的"采集→更新"，不需要通用 RPC |
| FIFO Channel（worker 间数据） | `torch.multiprocessing.Queue` 或 `dist.send/recv` | 语义一致 |
| 通信 / 权重同步 | `torch.distributed`（NCCL/Gloo）、cudaIPC | **和 RLinf 内部用的是同一套**，本就与 Ray 无关 |

**核心论据**：我们要"借鉴的方法"——disaggregated/hybrid 的 rollout-训练分卡、子模拟器流水线、NCCL 权重同步——**这些东西在 RLinf 自己的实现里就不依赖 Ray**。Ray 只是它的多节点启动器。去掉 Ray = 把"多节点启动器"换成"单机启动器（torchrun/mp.spawn）"，方法本身一点不损失。

### 5.2 我们会失去什么（Ray 的真实价值）

Ray 不是没用，它的价值在我们**当前用不到**的维度：

- **多节点编排**：跨机器组集群、统一调度——我们单机用不到。
- **弹性伸缩 / 容错**：worker 挂了自动重启、动态增减节点——我们单机短作业用不到（崩了重跑即可）。
- **异构工作流的通用 RPC**：复杂 agentic pipeline 里任意 worker 互调——我们的 PPO 控制流是规整的同步循环，不需要。

**取舍**：去 Ray 我们放弃的是"多节点 + 弹性容错"，换来的是"少一个重依赖、少一层抽象、调试直观"。对单机 8 卡的 VLA-PPO，这个取舍明显划算。**如果将来要上多机**，再引入 Ray（或 `torchrun` 的多节点模式）也不迟——届时工作流代码不用大改，这正是 M2Flow"逻辑/物理解耦"思想的好处，值得我们在设计时保留。

### 5.3 甚至可以连"多进程"都不用

RLinf 用多进程（Ray actor）是因为它**必须**支持多节点——跨节点只能多进程。但单机多卡有更轻的选择：**单进程 + 多线程 + 每卡一份模型副本**。CUDA kernel 和子进程 IPC 都会释放 GIL，所以多线程在多卡上能真正重叠。这条路连进程间权重同步（pickle/IPC）都省了——权重在进程内跨卡 `load_state_dict` 即可。

**而我们仓库里已经有人这么干了**——见下一节。

---

## 6. 关键发现：`train_rl_offpolicy.py` 已经是"去 Ray 的 disaggregated RLinf"

调研中最重要的发现：本目录的 TD3 生产路径 `train_rl_offpolicy.py` **已经实现了 RLinf disaggregated 模式的全部要件，且全程没有 Ray、没有多进程**。它的 docstring 自述：

> "off-policy TD3 with split rollout/training GPUs … Rollout GPUs: each loads a frozen VLA copy, collects episodes in parallel … Train GPU: runs actor-critic updates … Replay buffer: centralized on CPU."

对照 RLinf 概念：

| RLinf 概念 | `train_rl_offpolicy.py` 里的对应实现 |
|:-----------|:-------------------------------------|
| Rollout / Training 组件分离 | `--rollout_gpus 0,1,2,3,4 --train_gpu 5` |
| 每个 Generation worker 一份模型 | 每张 rollout 卡一份**冻结 VLA 副本** + 一份 actor/critic 副本 |
| Simulator worker | `PersistentEnvPool`——每卡常驻一组 LIBERO 子进程，`egl_gpu_id` 把 MuJoCo 渲染绑到对应物理卡 |
| 批量推理的 step-lock rollout | `rlt_rollout_fast.py`，其 docstring 明写 **"step-lock architecture (like SimpleVLA-RL/RLinf)"** |
| 权重同步 barrier | `_sync_rollout_weights()`：训练卡权重 → CPU → 每张 rollout 卡 `load_state_dict` |
| FIFO Channel | `rollout_stats_queue`（`queue.Queue`）+ 带锁的中心化 replay buffer |
| Worker 放置 | `task_id % n_rollout_gpus` 把任务摊到各卡 |

**它用的不是 Ray，是 `threading.Thread` + `ThreadPoolExecutor` + `queue.Queue`**——单进程多线程多卡。也就是说，"用 RLinf 的方法、不用 Ray"这件事，**我们自己的代码库里已经验证可行并在生产用了**。

缺口很明确：

- ✅ TD3（off-policy）：已多卡采集。
- ❌ **VLA-PPO（`train_rl_vla_ppo.py`）：单卡**——本次要补的。
- ⚠️ `train_rl_onpolicy.py`（小 actor PPO，legacy）：用 HF Accelerate + `all_reduce`，是"每卡各采各的、再同步梯度"的 DDP 式，不是 rollout/训练分卡。

---

## 7. 落地建议：把 RLinf 思路用到 VLA-PPO（全程不用 Ray）

### 7.1 推荐方案：复用 offpolicy 的"单进程多线程分卡"

把 `train_rl_offpolicy.py` 的分卡机制移植到 VLA-PPO，对应 RLinf 的 **disaggregated 模式**：

1. **每张采集卡一份 VLA 推理副本**（VLA-PPO 里策略就是 VLA，rollout 用当前策略的 eval 态副本，`no_grad`，~8–11 GB/卡，8×80GB 随便放）。
2. **G 个 episode 按初始状态切片**，每张卡用 `vla_ppo_collect` 采自己那份；训练卡在 rollout 阶段也参与采集（它持有的可训 VLA 直接当采集策略，对应 hybrid 的"GPU 复用"）。
3. 用线程池并发驱动各卡，episode 是**纯 CPU 可 pickle 数据**（`VLAPPOStepRecord` 里是 numpy uint8 图 + CPU 张量），汇总零成本。
4. **PPO 更新仍在单训练卡**（只占 12%，无需动）。
5. 更新后 `_sync_rollout_weights()` 式地把新 VLA 权重 `load_state_dict` 到各采集副本——**进程内跨卡拷贝，无 IPC、无 Ray**。

可直接复用的现成件：`PersistentEnvPool`、`rlt_rollout_fast.py` 的 step-lock、`--rollout_gpus / --train_gpu / --num_envs_per_task / --use_steplock` 等 CLI、offpolicy 的线程编排与权重同步函数。**真正新写的只有"VLA-PPO 版的多卡采集封装"，估计 ~150–250 行。**

### 7.2 备选方案：`torchrun` 多进程

若日后要上多机，或想严格对齐 RLinf 的多进程模型：用 `torchrun --nproc_per_node=8` 起 8 进程，每进程绑一卡，`dist.broadcast` 同步 VLA 权重（NCCL）、`dist.gather` 收轨迹。这才是"最像 RLinf 但不用 Ray"的形态——因为 RLinf 的多进程通信本就是 torch.distributed。代价是多进程调试更重、8 GB 权重要走 NCCL 广播。**单机当前不推荐，留作多机预案。**

### 7.3 进阶：hybrid 子模拟器流水线（可选）

disaggregated 落地、确认收益后，可再上 RLinf 的 hybrid 流水线：当前 step-lock 在一个 chunk 内"VLA 前向（~50%）→ 环境步进（~30%）"是串行的；把环境分成 2 个错相位子组（`pipeline_stage_num=2`），让 A 组步进时 B 组的 VLA 前向同时跑，可再叠一层加速。属于二期优化，不阻塞一期。

### 7.4 预期收益估算

rollout 214 s，N 卡 disaggregated 后 ≈ `214/N + 权重同步`（线程内跨卡同步 8 GB，约 5–10 s）：

| 采集卡数 | rollout | + 更新 30 s | 整 iter | 相对单卡 |
|:--------:|:-------:|:-----------:|:-------:|:--------:|
| 1（现状） | 214 s | 30 s | 244 s | 1.0× |
| 4 卡 | ~54 s + 8 | 30 s | ~92 s | **~2.6×** |
| 8 卡 | ~27 s + 8 | 30 s | ~65 s | **~3.7×** |

附带收益：周期评测（`eval_n_episodes=20` 的确定性 rollout，单卡要 ~535 s）同样并行化，8 卡可压到 ~70 s。

> 注意 batch 取舍：G=8 摊到 8 卡则每卡仅 1 episode，VLA 前向 batch 偏小。建议多卡采集**配合调大 G**（如 G=16–32，用 `--G_per_task / --group_size`），既喂饱每卡的 batch，也给 PPO 更多样本——RLinf-VLA 也用大 group 采集。

---

## 8. 风险与取舍

| 项 | 评估 |
|:---|:-----|
| 去 Ray 是否损失"方法" | **否**。要借鉴的 disaggregated/hybrid/NCCL 同步在 RLinf 内部本就不依赖 Ray |
| 去 Ray 损失什么 | 多节点编排、弹性容错——单机短作业用不到；未来上多机再补 |
| 多线程 GIL | CUDA 与子进程 IPC 均释放 GIL，offpolicy 生产已验证多卡线程可重叠 |
| 显存 | 每采集卡一份 VLA 推理副本 ~8–11 GB，远低于 80 GB；训练卡峰值 ~33–50 GB 不变 |
| 权重同步开销 | 进程内跨卡 `load_state_dict` ~8 GB，估 5–10 s，相对 214 s rollout 可忽略 |
| 与现有代码一致性 | 复用 offpolicy 的分卡/同步/step-lock，风格统一，新增面小 |
| 孤儿进程 | 沿用 `common/parent_death.py` 的 `PR_SET_PDEATHSIG` 治理，多卡更要注意 |

---

## 9. 参考资料

- RLinf GitHub：<https://github.com/RLinf/RLinf>
- RLinf 系统论文（M2Flow）：<https://arxiv.org/abs/2509.15965>
- RLinf-VLA 论文：<https://arxiv.org/abs/2510.06710>
- RLinf 文档 — Worker 编程模型：<https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/worker.html>
- RLinf 文档 — Hybrid 模式：<https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/hybrid.html>
- RLinf 文档 — 通信层（Adaptive P2P）：<https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/communication/collective.html>
- 本目录现有实现：`train_rl_offpolicy.py`、`algos/RLT_a/`（step-lock rollout）、`envs/persistent_env_pool.py`、`TECHNICAL_REPORT.md` §7
- 瓶颈实测数据：`results/rlt_training/vla_ppo_qwen_t0_0522_1029/vla_ppo/train.log`（本次调研期间运行）

---

*本报告基于 RLinf 公开论文/文档与本仓库代码现状撰写；RLinf 实现细节以其源码为准。*
