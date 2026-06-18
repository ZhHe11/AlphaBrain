# VLA 多任务 RL — 进度交接 (2026-06-01)

## 已完成 & 已验证

### 1. RL_REPORT_TABLES.md 表1「全10任务」列回填（已改，未 commit）
P0 离线 50-ep 复评完成，4 个 § + t3 已换实测值（源 `results/eval_p0_50ep_0529/`）：
- RLT+GRPO 0.720 / RLT+PPO 0.916 / RLT_a+GRPO 0.704 / RLT_a+PPO 0.936🏆 / RLT_a+PPO t3=0.96
- 时间戳 typo 已修正（`_1836`→`_1830`、`_1526`→`_1551`、`_1836`→`_1829`）

### 2. VLA 多任务 trainer 支持（已改，已 smoke 验证，未 commit）
**根因**：VLA trainer 原本不读 `--all_tasks`（`task_id = args.task_id if >=0 else random`），传了被静默忽略只训 task0。
**修法**：移植 RLT 的 task-cycling（每 iter 轮 `tasks_per_iter` 个任务）+ 全 task_list per-task eval。
**GRPO 额外修复**：原用老 collector `vla_ppo_collect`（每次 reset 重握手 libero worker → eval reset×10 时管道死锁，py-spy 栈 `_read_msg`@libero_env.py:38 坐实）。已换成 `vla_ppo_collect_steplock` + `PersistentEnvPool`（与 PPO trainer 对齐）。

改动文件（7 个，全部 parse OK + smoke 通过）：
- `trainers/train_args.py` — 加 `--tasks_per_iter`
- `trainers/train_rl_vla_ppo.py` — task-cycling + per-task eval（双 smoke 通过）
- `trainers/train_rl_vla_grpo.py` — 同上 + collector 升级（fix2 smoke 通过，deadlock 消失）
- `scripts/run_rl_scripts/run_qwen_vla_ppo.sh` / `run_qwen_vla_grpo.sh` — `MULTI_TASK` + `TASKS_PER_ITER` 透传

### 3. 真实 300-iter run 完成（in-train 20-ep 终值）
- **VLA+PPO** `results/rlt_training/vla_ppo_qwen_alltasks_0531_2052/vla_ppo`
  - final iter300 mean=**0.700**；per-task t0–t9 = .75/.95/.95/.20/.95/.70/.25/1.0/1.0/.25
  - best iter120=0.790；last-3 mean=0.737
- **VLA+GRPO** `results/rlt_training/vla_grpo_qwen_alltasks_0531_2112/vla_grpo`  ⚠️ 是 `_2112` 不是 `_2059`（`_2059` 是 NameError 崩溃的 3-iter 残run，忽略）
  - final iter300 mean=**0.775**；per-task t0–t9 = .90/.85/1.0/.55/.80/.55/.40/1.0/.95/.75
  - best iter200=0.800；last-3 mean=0.757

## 待办（下一步）

### A. VLA 终ckpt 离线 50-ep 复评（口径对齐其他行）— 卡住，需修
**问题**：final ckpt 的 `vla/` 子目录是**空的**！`save_pretrained` 失败（warning: `Missing key _get_non_default_generation_parameters`），只 fallback 存了 `vla_state_dict.pt`（8.2GB 裸 state_dict）。
- 路径：`.../vla_ppo_iter_00300_final/vla_state_dict.pt`（vla/ 空）
- `run_eval_base_vla.sh` 走 `BaseFramework.from_pretrained(vla_ckpt)` 需 HF 目录，加载不了裸 state_dict
- `eval_libero.py` 第 46 行有 `--vla_state_dict` flag（"not an HF-format dir, combine with --base_vla"）→ **可加载 state_dict，但 run_eval_base_vla.sh 没暴露这个 flag**
- **我已 kill 掉之前误指向空 vla/ 的两个 eval（GPU1,2,3,4）**

**修法选项**：
1. 给 `run_eval_base_vla.sh` 加 `VLA_STATE_DICT` 透传 → `eval_libero.py --base_vla --vla_state_dict <path>`（需先确认 eval_libero.py state_dict 加载逻辑：base_vla 时仍需一个 HF 配置目录做骨架？查第 78-90 行）
2. 或直接 in-train 20-ep 终值入表（0.700/0.775），注脚标"in-train 20-ep，待 50-ep 对齐"

### B. 回填表1 VLA+PPO/GRPO 两行
拿到 50-ep（或暂用 in-train）数后，填 `RL_REPORT_TABLES.md` 表1 的 `VLA + PPO(基线)` / `VLA + GRPO(基线)` 行（目前是「5traj 未跑」）。
注意：这是 5traj 基座、task cycling 4/iter 协议，与 RLT 行可比。

### C. commit
全部验证后，用户同意即可 commit（之前说"先不 commit"，现 GRPO 已验证）。
分支 `rl-report`。

## 清理
- smoke 残留目录可删：`results/rlt_training/vla_{ppo,grpo}_qwen_alltasks_0531_*`（除 _2052 PPO 真run、_2112 GRPO 真run 外）
- `_2059` GRPO 是崩溃残run，可删
