# Table 1 实验 Runbook

> 目标:填满 `RL_REPORT_TABLES.md` 表 1 的空行。
> 你后台跑这些脚本,我读日志盯进度。
> 第一轮口径:**libero_goal task 0,QwenOFT-5traj 基座,单种子**(最快、各方法可比;`全 10 任务` / 多种子留作第二轮)。

---

## 现状

| Table-1 行 | 命令 | GPU | 状态 |
|:-----------|:-----|:----|:-----|
| 基座 VLA(1traj / 5traj) | 已用现有 eval 填(0.33 / 0.69) | — | ✅ 已填(协议待对齐) |
| RLT + TD3 | 已有数据(0.92) | — | ✅ 已完成 |
| **RLT + GRPO** | ① pretrain ② `run_rlt_grpo.sh` | 1 | ⬜ 待跑 |
| **RLT_a + TD3** | ① pretrain ② `run_rlt_rl.sh TRACK=rlt_a` | 2 | ⬜ 待跑 |
| **VLA + PPO** | `run_qwen_vla_ppo.sh` | 4 | ⬜ 待跑 |
| **VLA + GRPO** | `run_qwen_vla_grpo.sh` | 5 | ⬜ 待跑 |
| RLT + PPO | — | — | ⏸ 阻塞:需 E2(小 actor PPO 代码) |

GPU 0–5 空闲,6 号有任务。下面 4 个 run 分到 GPU 1/2/4/5,可同时后台启动。

---

## A — RLT + GRPO  (GPU 1)

本会话刚解锁(E1)。需先预训练一个 **RLT 轨道**编码器。

```bash
# ① 预训练 RLT 编码器(前台或后台均可,~0.5–1.5h)
TRACK=rlt bash scripts/run_rl_scripts/run_rlt_pretrain.sh 1

#    跑完记下编码器路径,形如:
#    results/rlt_training/<tag>_<ts>/pretrain/checkpoints/pretrain_best/encoder.pt

# ② GRPO(后台)
nohup env ENCODER_PATH="<①的 encoder.pt 路径>" \
    bash scripts/run_rl_scripts/run_rlt_grpo.sh 1 >/dev/null 2>&1 &
```

- 产物:`results/rlt_training/rlt_grpo_qwen_t0_<ts>/rl_grpo/{train.log, metrics.json}`
- `run_rlt_grpo.sh` 是本会话新建的脚本(`run_rlt_a_grpo.sh` 的 RLT 轨道版)。
- 编码器超参默认 `ENCODER_HEADS=8 ENCODER_LAYERS=2 DECODER_LAYERS=2 MAX_LEN=4096`,**必须与 ① 预训练时一致**(默认值一致,不动即可)。

## B — RLT_a + TD3  (GPU 2)

```bash
# ① 预训练 RLT_a 编码器
TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_pretrain.sh 2

# ② TD3(后台);run_rlt_rl.sh 默认 TRACK=rlt,这里切 rlt_a
nohup env TRACK=rlt_a bash scripts/run_rl_scripts/run_rlt_rl.sh 2 >/dev/null 2>&1 &
```

- `run_rlt_rl.sh` 会自动发现 ① 产出的编码器(`run_rlt_a_grpo.sh` 同款发现逻辑)。
- 产物:`results/rlt_training/<rlt_a tag>_<ts>/rl_offpolicy/`

## C — VLA + PPO 基线  (GPU 4)

```bash
nohup bash scripts/run_rl_scripts/run_qwen_vla_ppo.sh 4 >/dev/null 2>&1 &
```

- 直接全量微调整个 VLA,**显存 ~50GB**,务必独占一张卡。
- 产物:`results/rlt_training/vla_ppo_qwen_t0_<ts>/vla_ppo/`

## D — VLA + GRPO 基线  (GPU 5)

```bash
nohup bash scripts/run_rl_scripts/run_qwen_vla_grpo.sh 5 >/dev/null 2>&1 &
```

- 全量微调 + 参考 VLA,**显存 ~58GB**,独占一张卡。
- 产物:`results/rlt_training/vla_grpo_qwen_t0_<ts>/vla_grpo/`

---

## 监督进度(我来做)

每个 run 都把日志 tee 到 `<output_dir>/train.log`,SR/loss 写进 `metrics.json`。
你启动后告诉我一声,我会定期读这些文件并汇报。我盯的 glob:

```
results/rlt_training/rlt_grpo_qwen_t0_*/rl_grpo/train.log
results/rlt_training/*_*/rl_offpolicy/train.log          # RLT_a+TD3
results/rlt_training/vla_ppo_qwen_t0_*/vla_ppo/metrics.json
results/rlt_training/vla_grpo_qwen_t0_*/vla_grpo/metrics.json
```

关注信号:`success_rate` 是否随 iter 上升;GRPO 看 `n_groups_with_signal`(=0 说明组内无奖励差,无学习信号);崩溃迹象(loss NaN、SR 长期 0、env worker 报错)。

---

## 填表流程

1. 训练中每 20 iter 有 20-ep 内嵌 eval → 可先填**临时数**进 Table 1。
2. 跑完后用 `run_eval_rlt.sh` 做 **50-ep 离线评测** → 替换为正式数(RLT/RLT_a 行)。VLA 基线行用各自 ckpt 的 VLA 评测。
3. `RL_REPORT_TABLES.md` 表 1 / 表 2 回填;`plot_rl_report.py` 的 `MAIN_RESULTS` 同步更新后重跑出图。

## 第二轮(可选,扩展)

- `全 10 任务`列:各命令加 `MULTI_TASK=1`(VLA 基线的多任务支持需先确认)。
- 多种子:`SEED` / `--seed` 改 3 个值,Table 1 改报 `mean±std`。
- 基座行 T1 干净重测:与 RL eval 同协议(50-ep / max_steps 320 / `final_model` ckpt)重测,替换现有 † 数。

---

## 阻塞项

- **RLT + PPO**:小 actor 的 on-policy PPO trainer(`train_rl_onpolicy.py`)是 legacy 占位损失,且不支持 `--encoder_mode rlt`。需 E2:在 `train_rl_grpo.py` 加 `GAE+critic` 开关复用其代码。要做的话我可以接手。
