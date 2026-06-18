#!/usr/bin/env python3
"""Per-iter multi-seed eval campaign for the convergence-figure std bands.

For each of the 6 RLT routes, evaluates seeds {43,44} at checkpoints
{100,200,300} with the SAME offline 50-ep all-task protocol as the seed-42 main
curve, so the figure can draw a REAL per-iteration std band (mean+/-std over
seeds at each of those 3 iterations) instead of a constant endpoint approximation.

Self-driving: runs <=2 evals at a time on free GPUs (default 6,7), skips outputs
that already exist (resumable), writes results/eval_band_seeds/<route>_s<seed>_iter<it>.json.

Run ON box2:  nohup python scripts/run_rl_scripts/eval_band_campaign.py > logs/eval_band_campaign.log 2>&1 &
"""
import glob, json, os, subprocess, time

ROOT = "/share/zhanghe/AlphaBrain-zh"
VLA = f"{ROOT}/results/training/QwenOFT-5traj-libero_goal/final_model"
OUT = f"{ROOT}/results/eval_band_seeds"
GPUS = [int(x) for x in os.environ.get("BAND_GPUS", "6,7").split(",")]
ITERS = [50, 100, 200, 300]
os.makedirs(OUT, exist_ok=True)

# recipe per route: (eval script, bottleneck, heads, prop_dim, residual, extra)
REC = {
    "rlt_ppo":   ("eval_libero_rlt.py", 2048, 8, 0, True,  ["--max_len", "4096"]),
    "rlt_grpo":  ("eval_libero_rlt.py", 2048, 8, 0, True,  ["--max_len", "4096"]),
    "rlt_td3":   ("eval_libero_rlt.py", 2048, 8, 8, False, ["--max_len", "4096"]),
    "rlta_ppo":  ("eval_libero.py",      256, 4, 0, True,  []),
    "rlta_grpo": ("eval_libero.py",      256, 4, 0, True,  []),
    "rlta_td3":  ("eval_libero.py",      256, 4, 8, False, []),
}
# route -> seed -> run-dir glob (newest match used)
RUNS = {
    "rlt_ppo":   {43: "rlt_ppo_qwen_alltasks_s43v3_*",   44: "rlt_ppo_qwen_alltasks_s44v2_*"},
    "rlt_grpo":  {43: "rlt_grpo_qwen_alltasks_s43v3_*",  44: "rlt_grpo_qwen_alltasks_s44v2_*"},
    "rlta_ppo":  {43: "rlt_a_ppo_qwen_alltasks_s43v2_*", 44: "rlt_a_ppo_qwen_alltasks_s44v2_*"},
    "rlta_grpo": {43: "rlt_a_grpo_qwen_alltasks_s43v3_*",44: "rlt_a_grpo_qwen_alltasks_s44v3_*"},
    "rlt_td3":   {43: "rlt_td3_qwen_alltasks_s43v4_*",   44: "rlt_td3_qwen_alltasks_s44v3_*"},
    "rlta_td3":  {43: "rlt_a_td3_qwen_alltasks_s43v5_*"},  # s44 has no 100/200/300 ckpts
}


def find_ckpt(pat, it):
    for d in sorted(glob.glob(f"{ROOT}/results/rlt_training/{pat}"), reverse=True):
        sub = next((s for s in os.listdir(d) if s.startswith("rl_")), None)
        if not sub:
            continue
        c = glob.glob(f"{d}/{sub}/checkpoints/*iter_{it:05d}")
        if c:
            return c[0]
    return None


def build_jobs():
    jobs = []
    for route, seeds in RUNS.items():
        for seed, pat in seeds.items():
            for it in ITERS:
                out = f"{OUT}/{route}_s{seed}_iter{it}.json"
                if os.path.exists(out):
                    continue
                ck = find_ckpt(pat, it)
                if ck:
                    jobs.append((route, seed, it, ck, out))
                else:
                    print(f"[skip] {route} s{seed} iter{it}: no ckpt", flush=True)
    return jobs


def launch(job, gpu):
    route, seed, it, ck, out = job
    script, bn, hd, prop, res, extra = REC[route]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl",
               TOKENIZERS_PARALLELISM="false", PYTHONPATH=ROOT)
    cmd = ["python", f"{ROOT}/AlphaBrain/training/reinforcement_learning/eval/{script}",
           "--vla_ckpt", VLA, "--action_token_ckpt", ck, "--suite", "libero_goal",
           "--task_ids", "0,1,2,3,4,5,6,7,8,9", "--n_eps_per_task", "50", "--gpu", "0",
           "--num_workers", "4", "--seed", "42", "--bottleneck_dim", str(bn),
           "--encoder_layers", "2", "--encoder_heads", str(hd), *extra,
           "--actor_hidden_dim", "512", "--ref_dropout", "0.5", "--fixed_std", "0.1",
           "--prop_dim", str(prop), *(["--residual"] if res else []),
           "--results_json", out]
    log = open(f"{ROOT}/logs/band_{route}_s{seed}_iter{it}.log", "w")
    print(f"[launch] gpu{gpu} {route} s{seed} iter{it}", flush=True)
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def gpu_free(gpu):
    """True if the GPU has little memory used (not running someone else's job)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode()
        for line in out.strip().splitlines():
            idx, used = [x.strip() for x in line.split(",")]
            if int(idx) == gpu:
                return int(used) < 2000   # <2GB => free
    except Exception:
        pass
    return False


def main():
    jobs = build_jobs()
    print(f"[campaign] {len(jobs)} evals to run on GPUs {GPUS}", flush=True)
    running = {}  # gpu -> proc
    while jobs or running:
        for gpu in GPUS:
            if gpu not in running and jobs and gpu_free(gpu):
                running[gpu] = launch(jobs.pop(0), gpu)
        time.sleep(20)
        for gpu, proc in list(running.items()):
            if proc.poll() is not None:
                print(f"[done] gpu{gpu} rc={proc.returncode}", flush=True)
                del running[gpu]
    print("[campaign] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
