# RLinf-VLA vs 我们的 VLA+RL —— 数字对照与诚实评估

> 调研日期 2026-06-08。来源:RLinf-VLA 论文 arXiv:2510.06710(Table 3)+ 官方 LIBERO 文档。
> 触发:我们的 VLA+PPO/GRPO baseline(0.718/0.756)看起来远低于"别人"的报告值。**核实结论:用户直觉正确,差距真实且重要。**

## 1. RLinf-VLA 在 LIBERO 的确切成绩(OpenVLA-OFT,50-ep/task,success_once)

| LIBERO suite | base(SFT) | RLinf-GRPO | 增益 |
|:--|:--|:--|:--|
| Spatial | 72.18 | 99.40 | +27.2 |
| Object | 71.48 | 99.80 | +28.3 |
| **Goal**(我们用的) | **64.06** | **98.79** | **+34.7** |
| LIBERO-10 (Long) | 48.44 | 93.95 | +45.5 |
| LIBERO-90 | 70.97 | 98.59 | +27.6 |
| **平均(130 tasks)** | **65.43** | **98.11** | **+32.7** |

注:论文 Table 3 只报 GRPO(无单独 PPO 列)。eval 判据 = **success_once**(一局内成功过一次即算)。

## 2. 对照我们(QwenOFT,LIBERO-Goal,50-ep,reward≥0.5 判据)

| | base | VLA+GRPO | VLA+PPO | 最强 RLT_a+PPO |
|:--|:--|:--|:--|:--|
| **我们** | 0.704 | 0.756 (+5.2) | 0.718 (+1.4) | 0.936 |
| **RLinf(Goal)** | 0.641 | **0.988 (+34.7)** | — | — |

**判据对照实测(2026-06-08,success_once 重评)**:用 RLinf 的 success_once 判据重评我们的 VLA baseline → VLA+PPO 0.718→**0.74**(+0.022)、VLA+GRPO 0.756→**0.75**(−0.006)。**判据只值 ~±2pp,RLinf 的 +23pp 差距几乎全是规模/训练有效性,不是测量假象。** 这反而强化了"差距真实、RLT 卖效率"的定位。

**关键观察**:
1. **base 可比甚至我们更高**:我们 QwenOFT base 0.704 > RLinf OpenVLA-OFT base(Goal)0.641。**起点不是问题。**
2. **差距 100% 在 RL 有效性**:RLinf-GRPO 增益 +34.7pp(冲到 0.988),我们 VLA+GRPO 仅 +5.2pp。
3. **连我们最强的 RLT_a+PPO(0.936)都低于 RLinf 全量微调 GRPO(0.988)**。

## 3. 为什么差这么多(setting 差异)

| 维度 | RLinf | 我们的 VLA baseline |
|:--|:--|:--|
| 并行环境数 | **128–256**(向量化,一次大 batch 前向) | ~4–8 envs |
| batch | global_batch **640**,micro 80 | 小(micro_batch 2) |
| GRPO group_size | **8**(同初态 8 轨迹比较) | 基础版,组小 |
| GRPO 配方 | **DAPO 式**:clip-higher 0.28、dual-clip 3.0、normalize advantages | 基础 clip 0.2 |
| 训练吞吐 | 大规模、跑到收敛 | 单卡/少卡、迭代少 |
| eval 判据 | **success_once**(宽松) | reward≥0.5(可能更严) |

**主因:规模 + GRPO 调参**。RLinf 的吞吐来自"几百环境一次大前向 + 大 group + DAPO",我们的 VLA-PPO/GRPO 是最小可用实现,严重 under-scaled。eval 判据(success_once vs reward≥0.5)能解释一部分绝对值差。

## 3.5 资源对比 —— RLT 的核心卖点(效率)

> **实测(2026-06-08 核实,逐 PID 对应 nvidia-smi)**:RLT 多任务单 run GPU 显存 **~55–69GB/卡**(box1 G5=69GB/G6=56GB、box2 G0/G2/G4=51–61GB),iter **~3.2min**(EVAL_INTERVAL=9999,单卡)。
> 可训参数由 ckpt 文件大小反推:**actor.pt 1.7MB + critic.pt 1.5MB ≈ ~0.8M 参数(fp32)**;encoder.pt 516MB(~129M)与 VLA 7.7GB(~4B)在 Phase-2 **全冻结**。
> ⚠️ 早前误报的"~18GB"实为 eval 进程显存,非训练——已更正。

| 维度 | **我们 (RLT)** | **RLinf (full-VLA)** | 倍数 |
|:--|:--|:--|:--|
| 可训参数 | **~0.8M**(actor+critic;encoder 129M + VLA 4B 全冻结) | **~4.1B**(全参) | **~5000× 少** |
| 可训占比 | **~0.02%** | 100% | — |
| 训练 GPU 数 | **1 张**(单卡即可跑;多卡仅加速 rollout) | **8 卡/节点起**(FSDP 分片),可扩 32 卡 | **≥8× 少** |
| GPU 显存 | **~55–69GB(单卡,冻结 4B VLA + 大 batch 多任务 rollout 激活)** | **~80GB/卡 × 8 ≈ 640GB**(FSDP 分片 4B + 激活 + 优化器态) | **~10× 少(总量)** |
| 并行环境 | 4–8 | 128–256 | ~32× 少 |
| batch | global ~40 / micro 2 | global 640 / micro 80 | ~16–40× 少 |
| 训练后端 | 单进程多线程,**无 FSDP/Megatron** | FSDP+HF / Megatron+vLLM | — |
| **LIBERO-Goal SR** | **0.936**(RLT_a+PPO) | **0.988**(GRPO) | — |

**一句话效率论断(已用实测修正)**:
> **RLT 在单张 80GB 卡上(占 ~55–69GB)、只训 ~0.8M 参数(<0.02%,4B VLA + encoder 全冻结、无需 FSDP),达到 0.936;RLinf 用 8+ 卡 FSDP、~640GB 总显存、全 4B 参数梯度、256 环境,达到 0.988。RLT 以 ~8× 更少 GPU、~10× 更少总显存、~5000× 更少可训参数,换到 ~95% 的性能。**

**注意**:RLT 的省**不在"单卡显存小"**(冻结的 4B VLA + 大 batch rollout 仍占满大半张卡 ~60GB),而在 **(1) 单卡即可、无需 8 卡 FSDP;(2) 只训 <0.02% 的参数、不碰 4B 全参梯度与优化器态**。这才是诚实的效率卖点。

## 4. 对我们报告 RQ-B 的诚实影响(必须修正)

报告原结论:**"RLT 路线(0.916/0.936)显著优于全量微调(0.718/0.756)"**。

**问题**:这是拿我们**强的 RLT** 对比我们**弱的、under-scaled 的 VLA baseline**——不是公平对比。RLinf 证明全量微调 GRPO 在 LIBERO-Goal 能到 **0.988**,远超我们 RLT 的 0.936。

**因此 RQ-B 必须改为**(三选一或组合):
- (a)**加 caveat**:明确"全量微调 baseline 为小规模实现(few-env / 基础 GRPO),非 SOTA;大规模全量微调(如 RLinf)可达 0.98+。本文 RLT 优势仅相对此小规模 baseline,且 RLT 的价值主张应改为**计算效率/训练稳定性**而非绝对 SR 上限"。
- (b)**对齐 eval 判据**:用 success_once 重评我们所有 run,看绝对值能否抬升、与 RLinf 可比。
- (c)**做公平对比**:把 VLA baseline 按 RLinf 配方放大(group_size 8、多环境、DAPO)重训,再比。

**最低限度必须做 (a)**——否则报告把 under-scaled baseline 当"全量微调的代表",结论站不住、且 reviewer 一查 RLinf 就破。

## 5. RLT 路线真正还站得住的卖点(重新定位)
- **计算/显存效率**:冻结 VLA、只训小 actor/critic,无需 FSDP 训 4B 全参 —— 这是真优势,与 SR 上限无关。
- **训练稳定性**:RLT 路线 in-train SR std<5 vs 我们 VLA 路线 std~12(且 fig5 显示 VLA+PPO 先升后降)。**但注意**:RLinf 大规模全量微调未必不稳,这点也需 caveat。
- **不宜再主张**:"RLT 在 SR 上优于全量微调"——RLinf 反例在先。

来源:[RLinf-VLA arXiv:2510.06710](https://arxiv.org/html/2510.06710v1) · [RLinf LIBERO docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)

## DAPO reproduction attempt (2026-06-18) — NEGATIVE RESULT

Implemented 3 of DAPO's 4 pieces in vla_grpo_loss.py (commit 9c20d9a), verified
active in the running process (clip_eps_high=0.28, dual_clip_c=3.0, success filter,
fixed_std=0.2, kl=0):

| variant | all-task SR |
|---|---|
| base VLA | 0.704 |
| vanilla GRPO (single-GPU / FSDP-scale / kl0) | 0.740–0.744 |
| + exploration std=0.2 | ~0.77 |
| + DAPO clip-higher + dual-clip + filter | 0.740 (iter10–50: .705/.75/.735/.725/.74) |
| VLA+PPO (same stack) | 0.922 |
| RLinf DAPO-GRPO (OpenVLA-OFT) | 0.988 |

DAPO clip tricks are INERT on our continuous-action VLA. Root cause isolated: the
4th DAPO piece, **token-level credit assignment**, is inseparable from a **discrete
action-token head** (OpenVLA-OFT). QwenOFT uses a continuous fixed-std Gaussian — no
tokens to credit, and clip-higher has no entropy to preserve (std fixed). The gap to
RLinf is the action parameterization, not a GRPO hyperparameter or bug.

### CORRECTION (2026-06-19): the first DAPO test was invalid; properly-powered redo

The 2026-06-18 run used ppo_epochs=1 + lr=1e-5 → policy ratio stayed ~1.0,
clip_frac ~1.5%, so clip-higher (1.28) / dual-clip NEVER activated. That run did
NOT test DAPO — the clips were no-ops because nothing reached the bound, not
because of the action representation.

Valid redo (ppo_epochs=4 so ratios drift; separate output dirs after a same-minute
collision bug, fixed in run_qwen_vla_grpo.sh):
| run | clip_frac | det. eval iter10/20/30 |
|---|---|---|
| lr=5e-5, 4ep | 0.17–0.23 (clips ENGAGE) | 72 / 75 / 72 |
| lr=1e-5, 4ep | ~0.015 (still inert, lr too low) | 71.5 / 64.5 / 71.5 |

CONCLUSION (now valid): with clip-higher + dual-clip genuinely firing on ~20% of
samples, SR stays ~72–75% — NO improvement over vanilla 0.74 / std0.2 0.77. The
lr=5e-5 run actively HURTS the hard task (t3 0.30→0.10) while easy tasks saturate
— the coarse episode-level credit's tug-of-war. Clips regulate step SIZE; they
cannot fix step DIRECTION, which is set by the (coarse) advantage. Confirms the
limiter is token-level credit (needs discrete action head), not the clip recipe.
