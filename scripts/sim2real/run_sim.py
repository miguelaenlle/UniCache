#!/usr/bin/env python3
"""Simulate the sim2real subsets: capacity sweep and final predictions.

Every run is derived from an H1 LRU eval command (same engine flags as H1),
changing only the task config (the subset), the GPU block count, CPU blocks
(0), the eviction policy, and the output directory. Timing is H1's legacy
model. "Agent" injects the workload's best Table 2 agent policy exactly as
H1 did (UNICACHE_POLICY_CANDIDATE). OPT uses a future-use map from one NONE
profiling run per workload.

    .venv-sim312/bin/python scripts/sim2real/run_sim.py --out result/sim2real-transfer/sweep \\
        --workloads codeparrot mashqa alfred sharegpt \\
        --blocks 256 512 1024 2048 4096 8192 16384 32768 \\
        --policies LRU UniCache LeCaR LFU FIFO OPT Agent --jobs 3

Writes <out>/runs.csv (one row per workload x blocks x policy).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402

from common import ROOT, SUBSETS, agent_policy, replay_command  # noqa: E402

def run(workload, blocks, policy, out, use_map=None, mode="sim", rep=0, block_size=8):
    run_dir = out / workload / str(blocks) / (policy if rep == 0 else f"{policy}/rep-{rep}")
    done = sorted(run_dir.glob("*/summary.json"))
    if not done:
        run_dir.mkdir(parents=True, exist_ok=True)
        step_log = run_dir / "steps.csv" if mode == "gpu" else None
        # The subset's task config, with its data path resolved on this machine.
        tasks = yaml.safe_load((SUBSETS / workload / "eval.yaml").read_text())
        for task in tasks["tasks"]:
            task["data_path"] = str(SUBSETS / workload / "eval.jsonl")
        (run_dir / "tasks_config.yaml").write_text(yaml.safe_dump(tasks, sort_keys=False))
        cmd = replay_command(sys.executable, workload, blocks, policy, run_dir, mode,
                             use_map, step_log, block_size, run_dir / "tasks_config.yaml")
        env = dict(os.environ, PYTHONHASHSEED="0", VLLM_USE_V1="0", HF_HUB_OFFLINE="1",
                   VLLM_NO_USAGE_STATS="1", TOKENIZERS_PARALLELISM="false",
                   MPLBACKEND="Agg", COLLECT_TRACE="1" if policy == "NONE" else "0",
                   UNICACHE_COMPLETIONS=str(run_dir / "completed.json"))
        if policy == "Agent":
            env["UNICACHE_POLICY_CANDIDATE"] = str(agent_policy(workload))
        for var, sub in [("MPLCONFIGDIR", "matplotlib"), ("VLLM_CACHE_ROOT", "vllm"),
                         ("VLLM_CONFIG_ROOT", "vllm-config")]:
            (run_dir / ".cache" / sub).mkdir(parents=True, exist_ok=True)
            env[var] = str(run_dir / ".cache" / sub)
        (run_dir / "command.json").write_text(json.dumps(cmd, indent=2))
        (run_dir / "env.json").write_text(json.dumps(
            {k: env[k] for k in ["UNICACHE_POLICY_CANDIDATE", "COLLECT_TRACE"] if k in env}, indent=2))
        with (run_dir / "run.log").open("w") as log:
            status = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if status:
            print(f"FAILED {workload} {blocks} {policy}: {run_dir / 'run.log'}", flush=True)
            return {"workload": workload, "blocks": blocks, "policy": policy, "status": status}
        done = sorted(run_dir.glob("*/summary.json"))
    s = json.loads(done[-1].read_text())
    if policy == "NONE":  # profiling run for OPT's future map; no cache
        return {"workload": workload, "blocks": blocks, "policy": policy, "status": 0,
                "result_dir": str(done[-1].parent.relative_to(out))}
    lat = json.loads((done[-1].parent / "latency_summary.json").read_text())
    row = {"workload": workload, "blocks": blocks, "policy": policy, "rep": rep, "status": 0,
           "hit_rate": s["hit_rate_summary"]["global_hit_rate"],
           "hit_tokens": s["hit_rate_summary"]["global_hit_tokens_num"],
           "prompt_tokens": s["hit_rate_summary"]["global_tokens"],
           "replay_s": s["global_time"], "wall_s": s["simulation_time"],
           "ttft_mean_s": (lat.get("ttft_s") or {}).get("mean"),
           "ttft_p90_s": (lat.get("ttft_s") or {}).get("p90"),
           "itl_mean_s": (lat.get("itl_s") or {}).get("mean"),
           **{k: v for k, v in s.get("cache_stats", {}).items()},
           "result_dir": str(done[-1].parent.relative_to(out))}
    print(f"{workload:<10} {blocks:>6} {policy:<9} hit {row['hit_rate']:.4f} "
          f"evict {row.get('evictions')} remiss {row.get('evicted_then_missed')} "
          f"preempt {row.get('preemptions')} replay {row['replay_s']:.0f}s", flush=True)
    return row


def write_rows(rows, out):
    # Merge with earlier invocations on the same --out (newest row wins).
    key = lambda r: (r["workload"], str(r["blocks"]), r["policy"], str(r.get("rep", 0)))
    old = {key(r): r for r in csv.DictReader(open(out / "runs.csv"))} if (out / "runs.csv").exists() else {}
    old.update({key(r): r for r in rows})
    rows = list(old.values())
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in
                    ["workload", "blocks", "policy", "rep", "status", "hit_rate"], k))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["workload"], int(r["blocks"]), r["policy"],
                                               int(r.get("rep") or 0))))
    print(f"Wrote {out / 'runs.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--workloads", nargs="+", required=True)
    p.add_argument("--blocks", nargs="+", type=int, required=True)
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--mode", choices=["sim", "gpu"], default="sim",
                   help="gpu: same engine on CUDA in real time, one run at a time")
    p.add_argument("--block-size", type=int, default=8, help="tokens per KV block")
    p.add_argument("--repeat", type=int, default=1,
                   help="gpu only: run each policy this many times (rep-<k> dirs)")
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "gpu":
        if "OPT" in args.policies or args.jobs != 1:
            p.error("GPU mode: no OPT (its future map is stale under real timing), --jobs 1")
        rows = [run(w, b, pol, out, mode="gpu", rep=k, block_size=args.block_size)
                for w in args.workloads for b in args.blocks
                for pol in args.policies for k in range(args.repeat)]
        write_rows(rows, out)
        return

    rows = []
    with ThreadPoolExecutor(args.jobs) as pool:
        maps = {}
        if "OPT" in args.policies:
            # Legacy timing schedules prefills identically for every policy and
            # capacity, so one NONE profile per workload gives OPT's future map.
            profiles = {w: pool.submit(run, w, max(args.blocks), "NONE", out, block_size=args.block_size)
                        for w in args.workloads}
        futures = [pool.submit(run, w, b, pol, out, block_size=args.block_size)
                   for w in args.workloads for b in args.blocks
                   for pol in args.policies if pol != "OPT"]
        if "OPT" in args.policies:
            for w, f in profiles.items():
                r = f.result()
                maps[w] = out / r["result_dir"] / "use_map.json"
                if maps[w].stat().st_size < 100:
                    raise SystemExit(f"{w}: empty OPT use map {maps[w]}")
                futures += [pool.submit(run, w, b, "OPT", out, maps[w], block_size=args.block_size)
                            for b in args.blocks]
        rows = [f.result() for f in futures]
        rows += [f.result() for f in profiles.values()] if "OPT" in args.policies else []

    write_rows(rows, out)


if __name__ == "__main__":
    main()
