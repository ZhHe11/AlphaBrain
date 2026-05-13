# `run_rl_scripts/` — VLA Online RL launchers (LIBERO)

End-to-end pipeline:

```
[VLA finetune]  →  [Phase-1 encoder pretrain]  →  [Phase-2 TD3 RL]  →  [Eval]
   pi05 only       run_rlt_pretrain.sh           run_rlt_rl.sh        run_eval_*.sh
```

---

## Two algorithm tracks: `RLT_a` (first release) and `RLT` (later, paper-faithful)

The repo ships two encoder/actor recipes side-by-side under
`AlphaBrain/training/reinforcement_learning/algos/`:

| Track | Encoder input | When/why we use it |
|:------|:--------------|:-------------------|
| **`RLT_a/`** *(first released)* | action-query slice `(B, M=chunk_len, H)` projected to a `D=256` bottleneck | the practical default — deliberately deviates from the RL Token paper so the recipe scales to multi-task and stays portable across VLM backbones |
| **`RLT/`** *(added later)* | full VLM token sequence `(B, L, H)`, `z_rl` kept at VLA `H` | a paper-faithful reference track, useful for research-side comparisons; closer to the original Eq. 1/2 construction |

### Why `RLT_a` deviates from the paper

Two concrete reasons:

1. **Language is in the loop — multi-task potential.**
   `RLT` (per the paper's footnote 1) drops language tokens from the
   encoder input on the assumption that each task has a fixed
   instruction. That's fine for a single task but loses the natural
   conditioning signal for multi-task training. `RLT_a` feeds the
   encoder the action-query slice, which is already downstream of the
   VLA's image-language attention — so language is implicitly baked
   into every hidden state and the same actor can cover many tasks.

2. **Encoder training is easier across VLMs.**
   Action queries are exposed in roughly the same way by every VLA
   (one `get_vla_action`-style call), so `RLT_a`'s encoder ports across
   backbones with only a small adapter. `RLT`'s encoder consumes the
   full VLM token sequence, which means each new VLA needs its own
   feature-extraction code that understands that VLA's token layout
   (cf. `algos/RLT/vla_features.py` for Qwen vs
   `vla_features_pi05.py` for Pi05 — two largely independent
   implementations).

`RLT` is the more recent addition: it's closer to the paper's exact
construction (full VLM tokens in, no extra projection, encoder-decoder
cross-attention) and exists for fair side-by-side comparisons. The two
tracks share the trainer, rollout, and eval infrastructure; only the
encoder/decoder class and the `--encoder_mode {action_token, rlt}` flag
differ. Full design notes for each are in their `algos/<track>/README.md`.

### Backbone × track support matrix

|                 | Qwen (`Qwenvl_OFT`) | Pi05 (`PaliGemmaPi05`) |
|:----------------|:--------------------|:-----------------------|
| **`RLT`**       | ✓                   | ✓                      |
| **`RLT_a`**     | ✓                   | roadmapped — not wired yet |

`RLT_a` × Pi05 isn't currently supported: `RLT_a`'s encoder consumes the
VLA's action-query hidden states (which `Qwenvl_OFT` exposes via
`get_vla_action`), but `PaliGemmaPi05`'s flow-matching head outputs
actions directly without an "action-query slice" equivalent — that
adapter is on the roadmap.

---

## File inventory

```
.
├── README.md
├── run_pi05_finetune.sh       # VLA finetune (Pi05; pick variant via env)
├── run_rlt_pretrain.sh        # Phase-1: RLT encoder pretrain
├── run_rlt_rl.sh              # Phase-2: TD3 RL on the RLT track (Qwen or Pi05)
│
├── run_eval_rlt.sh            # offline eval, RLT policy (single iter or all iters)
├── run_eval_rlt_a.sh          # offline eval, RLT_a policy (Qwen only)
│
├── example_results/           # reference plots / summaries
└── example_scripts/           # legacy launchers + pi05_eval.yaml (VLA-only eval configs)
```

(End-to-end `RLT_a` training launcher isn't in this dir; see
`AlphaBrain/training/reinforcement_learning/algos/RLT_a/README.md`.)

---

## Quick start

### Prerequisites

- LIBERO installed in a separate conda env. Set `LIBERO_PYTHON` and `LIBERO_HOME` (or put them in `.env`) so launchers can spawn env worker subprocesses.
- For Pi05: local PaliGemma tokenizer dir at `$PALIGEMMA_TOKENIZER_PATH` (default `/datasets/peligemma`). Otherwise tokenizer init falls through to HF hub fetch and then to `sentencepiece` (often absent in containers).
- Disk: pretrain + RL output lands under `results/rlt_training/<run_name>_<timestamp>/`.

### 1 — Finetune the VLA (Pi05 only; Qwen ckpts assumed pre-existing)

```bash
VARIANT=1traj bash scripts/run_rl_scripts/run_pi05_finetune.sh   # 1 traj/task
VARIANT=5traj bash scripts/run_rl_scripts/run_pi05_finetune.sh   # 5 traj/task
VARIANT=task0 bash scripts/run_rl_scripts/run_pi05_finetune.sh   # task 0 only
```

Each maps to a mode block in `configs/finetune_config.yaml`. Output: `results/training/Pi05-goal-<VARIANT>-openpi/checkpoints/steps_<N>/`.

### 2 — Phase-1: pretrain the RLT encoder

```bash
bash scripts/run_rl_scripts/run_rlt_pretrain.sh [GPU_ID]
```

Override `CKPT_PATH` / `OUTPUT_DIR` via env if you want a non-default VLA. Output: `results/rlt_training/<tag>/pretrain/checkpoints/pretrain_best/encoder.pt`.

### 3 — Phase-2: RL fine-tune (RLT track)

```bash
# Qwen track (default)
bash scripts/run_rl_scripts/run_rlt_rl.sh [GPU_ID]

# Pi05 track
BACKBONE=pi05 VARIANT=1traj TASK_ID=0 bash scripts/run_rl_scripts/run_rlt_rl.sh 0
BACKBONE=pi05 VARIANT=5traj TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_rl.sh 0
```

`ENCODER_PATH` is auto-discovered (latest matching pretrain dir). Override `CKPT_PATH` / `ENCODER_PATH` via env if needed.

For the `RLT_a` track, see `AlphaBrain/training/reinforcement_learning/algos/RLT_a/README.md`.

### 4 — Eval

```bash
# VLA-only (no RL): policy server + LIBERO client. The yaml has one mode
# block per finetune variant; edit `checkpoint:` to point at the desired
# steps_X dir before running.
bash scripts/run_base_vla/eval.sh pi05_goal_5traj_eval \
    scripts/run_rl_scripts/example_scripts/pi05_eval.yaml

# RLT offline eval: defaults to all iter ckpts under RUN_DIR, parallel
# across GPUS. Pass ITER=00300 to eval one ckpt only.
RUN_DIR=results/rlt_training/<run>/rl_offpolicy \
VLA_CKPT=results/training/Pi05-goal-5traj-openpi/checkpoints/steps_30000 \
GPUS="0 1 2" TASK_IDS=0 N_EPS=50 \
    bash scripts/run_rl_scripts/run_eval_rlt.sh

# RLT_a offline eval (Qwen only): 10 tasks split across 3 GPUs
bash scripts/run_rl_scripts/run_eval_rlt_a.sh <RUN_DIR>
```

---

## Output layout

```
results/
├── training/
│   ├── Pi05-goal-{1traj,5traj,task0}-openpi/checkpoints/steps_<N>/   # Pi05 VLA finetune
│   └── QwenOFT-5traj-libero_goal/final_model/                        # Qwen VLA
└── rlt_training/
    ├── <pretrain_tag>/pretrain/checkpoints/pretrain_best/encoder.pt   # Phase-1
    └── <rl_tag>_<timestamp>/rl_offpolicy/
        ├── checkpoints/rl_offpolicy_iter_<N>/                         # Phase-2 ckpts
        ├── train.log
        └── eval_iter_<N>_rlt/summary.json                             # offline eval
```

---

## Tips & gotchas

- **`num_envs` is GPU-memory-bounded** (~0.5 GB activation/env). H100 80 GB → ≈48–64; A100 40 GB → ≈24–32.
- **Host RAM also matters**: each LIBERO env subprocess is ~600 MB MuJoCo + a few GB of Python overhead. Multiple RL trainings in the same cgroup can OOM-kill workers — check `cat /sys/fs/cgroup/memory.max` before scaling out.
- **Pi05 tokenizer**: always export `PALIGEMMA_TOKENIZER_PATH` (or rely on the launcher default `/datasets/peligemma`). Without it the first rollout VLA call dies with `ModuleNotFoundError: sentencepiece`.
- **Step-lock (`--use_steplock`) is faster** than free-run rollout because it batches all envs through one VLA forward.
- **Async eval during training** uses socket-IPC env workers (matches rollout); the older pipe-IPC `LiberoEnv` deadlocks on some container setups.
- **Encoder must be pretrained on the same VLA you fine-tune on.** Mixing a 1-traj-VLA encoder with 5-traj-VLA RL silently degrades.
- **`[Warning]: datasets path ... does not exist!`** flooding the log: each env reset checks LIBERO's `datasets/` dir, which RL doesn't need. Silence by creating an empty dir:
  ```bash
  DATASETS_DIR=$("$LIBERO_PYTHON" -c "import libero.libero, os; print(os.path.realpath(os.path.join(os.path.dirname(libero.libero.__file__), '..', 'datasets')))")
  mkdir -p "$DATASETS_DIR"
  ```

---

## ⚠️ Disclaimer

A few honest notes on what this release is — and what it isn't:

- **Why we open-source `RLT_a` first.** One of the reasons we open-source `RLT_a` first is that we believe the core idea — compressing VLA hidden states through an information bottleneck and then editing a reference policy with residual actions — is novel and worth sharing with the community at an early stage, *especially in the multi-task, multi-VLM-backbone setting* that the two deviations above are designed for. This does **not** mean we consider GRPO / PPO any less important; on the contrary, we view them as core algorithms for VLA online RL and will progressively release our implementations of them.
- **On reproducing the RL Token paper in simulation.** Faithfully reproducing every detail of the original RL Token paper inside a simulator is hard — in particular the carefully curated tuning datasets and the timely human-in-the-loop interventions described in the paper are difficult to replicate one-for-one in a purely automated sim setup. The `RLT_a` recipe is therefore a best-effort simulation adaptation, not a line-by-line reproduction. The newer `RLT` track is closer to the paper, but still differs in pretrain data, base VLA, and the absence of human-in-the-loop. See each algo's `README.md` for the concrete deltas.
- **What we believe matters most.** Collecting high-quality positive trajectories is one of the most critical open problems in this area — good positives do far more than clever loss tricks. Going forward we plan to:
  1. broaden the online-RL algorithm coverage (GRPO, PPO, and others);
  2. improve tooling for positive-sample collection, filtering, and curation on sim and real-world data;
  3. release stronger, better-documented baselines and more reproducible recipes.

This sub-module is a living research snapshot; APIs, configs, and numbers may change between releases. Issues and PRs are welcome.
