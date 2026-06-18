# 1traj × PPO encoder ablation — deployment

填表 1c(1traj 基座)中 RLT + PPO 和 RLT_a + PPO 各 3 个 task 的 6 个空 cell。

**关键问题:RLT + PPO 在 5traj task3 拿到了 1.00(满分),在 1traj 难任务上是否同样强?** 如果是,PPO + RLT 就是难任务的稳定通解,直接破 RLT_a + TD3 release 1traj task3 = 0.08 那个负迁移结论。

## 6 个 cell

| Track | task0 | task1 | task3 |
|:---|:---|:---|:---|
| RLT + PPO 1traj | (run) | (run) | (run) |
| RLT_a + PPO 1traj | (run) | (run) | (run) |

## ⚠️ 依赖顺序(重要)

- **RLT track(`run.sh rlt ...`)**:1traj encoder 已存在,**现在就能跑,无需等待**。
- **RLT_a track(`run.sh rlt_a ...`)**:**必须先跑下面的 pretrain**,否则报错
  `ERROR: no rlt_a 1traj pretrain dir found`。pretrain 完成(~30 min)后才能起 rlt_a 三个 task。

推荐:RLT 三个(r1/r2/r3)和 pretrain 同时起;pretrain 完再起 RLT_a 三个(r4/r5/r6)。

## 准备(RLT_a 必需,只需做一次)

RLT 已有 1traj encoder(`results/rlt_training/1traj_libero_goal_step30k_0423_0545/pretrain/...`),**RLT_a 没有**,需要先 pretrain:

```bash
# 在某张 GPU(默认 0)上跑 ~30 分钟
bash scripts/run_rl_scripts/deploy_1traj_ppo_ablation/pretrain_rlt_a_1traj.sh 0
```

完成后 `run.sh` 会自动找到新 pretrain 出的 encoder。

## 跑 6 个 cell(每个一个 tmux session,各 ~10h)

```bash
# RLT + PPO × 3 任务
bash scripts/run_rl_scripts/deploy_1traj_ppo_ablation/run.sh rlt 0          # GPU 0
bash scripts/run_rl_scripts/deploy_1traj_ppo_ablation/run.sh rlt 1 1        # GPU 1
bash scripts/run_rl_scripts/deploy_1traj_ppo_ablation/run.sh rlt 3 2        # GPU 2

# RLT_a + PPO × 3 任务(必须等 pretrain_rlt_a_1traj.sh 完成后才能起!)
bash scripts/run_rl_scripts/deploy_1traj_ppo_ablation/run.sh rlt_a 0 3      # GPU 3
bash scripts/run_rl_scripts/deploy_1traj_ppo_ablation/run.sh rlt_a 1 4      # GPU 4
bash scripts/run_rl_scripts/deploy_1traj_ppo_ablation/run.sh rlt_a 3 5      # GPU 5
```

如果远端机器只有 N 张卡,起 N 个 tmux session,每个跑一个 cell;剩下 cell 等先完成的 GPU 释放再跑。

## tmux 用法回顾

```bash
tmux new -s rlt-ppo-t0
# 粘贴对应命令,enter
# Ctrl-B D 退出(任务继续跑)
# 想看进度: tmux a -t rlt-ppo-t0
```

## 监督

我这边(主服务器 Claude)挂了 persistent monitor,会自动扫描 `logs/{rlt,rlt_a}_ppo_1traj_t{0,1,3}_*.log` 的:
- 每个 formal 20-ep eval(每 20 iter 一次)
- 完成 / 失败信号

跑完 6 个 cell,我会自动回填表 1c。

## 配置细节

- VLA ckpt: `results/training/0324-zh-QwenOFT-1traj-libero_goal/final_model`(1traj 基座)
- 协议: 单任务 in-train 20-ep eval terminal (iter 300) — 与表 1 5traj RLT+PPO 一致
- RLT encoder: existing 1traj pretrain(4/23,full-token)
- RLT_a encoder: fresh pretrain on 1traj basebone(本文件夹 pretrain_rlt_a_1traj.sh 产出)

## 常见问题

**Q: 跑起来报 "encoder.pt not found"?**
A: 多半是 RLT_a 没 pretrain。先跑 `pretrain_rlt_a_1traj.sh`,完成后再起 RLT_a + PPO 三个 task。

**Q: 跑到一半 hang 住?**
A: 检查 `MUJOCO_EGL_DEVICE_ID` 是否跟该 GPU_ID 一致。`run.sh` 自动设置,但如果你自己改了 GPU_ID 后 export 没刷新,可能 mismatch。重起 tmux session 即可。

**Q: 是否会跟主服务器跑的 multi-task 冲突?**
A: 不会。主服务器和你这台是不同机器,不共享 GPU,只共享 NFS log 目录。
