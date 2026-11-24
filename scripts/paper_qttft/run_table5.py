#!/usr/bin/env python3
"""Reproduce UniCache paper Table 5 (average QTTFT) in the simulator.

Paper setup (Sec. 6.4): one A100, Llama-3.2-3B, KV budget 0.9A, three
~1-hour mixed traces, eight policies; measured on real GPUs. Here:

* Traces: TaskData/Hybrid/test1, test2, test3, the repository's mixed
  configs whose multi-turn request shares (~80%, ~50%, ~20%) match the paper's
  Trace #1/#2/#3. ToolBench (one of four single-turn tasks) is not available
  locally and is dropped.
* Budget: 0.9 x 63 GiB (paper Table 4: A100 KV cache for the 3B model) in
  8-token blocks of the 3B model's KV size.
* Timing: ``--timing aic`` prices steps with aiconfigurator's A100 tables
  (uncalibrated unless --calibration); ``--timing legacy`` uses the repo's
  default power-law formula, i.e. the paper simulator's own timing model.
* QTTFT = arrival -> first output token, including queueing (the paper's
  definition) = ``ttft_s`` in latency_summary.json.

    .venv-sim312/bin/python scripts/paper_qttft/run_table5.py \\
      --out result/paper-qttft/aic-a100 --timing aic --jobs 9
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRACES = {"Trace#1": "test1", "Trace#2": "test2", "Trace#3": "test3"}
# Paper name -> engine policy name.
POLICIES = {"LRU": "LRU", "LFU": "LFU", "FIFO": "FIFO", "LeCaR": "LeCaR",
            "ARC": "ARC", "WA": "WA", "OPT (Offline)": "OPT",
            "UniCache": "SimpleHYBRID"}
PAPER_QTTFT_S = {  # Table 5
    "Trace#1": [4.94, 8.47, 6.87, 7.42, 4.54, 5.14, 3.84, 3.87],
    "Trace#2": [0.28, 0.53, 0.36, 0.47, 0.26, 0.29, 0.21, 0.24],
    "Trace#3": [0.13, 0.37, 0.25, 0.35, 0.12, 0.27, 0.08, 0.10],
}


def kv_bytes_per_block(model_dir: Path, block_size: int) -> int:
    c = json.loads((model_dir / "config.json").read_text())
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * head_dim * 2 * block_size


def trace_config(name: str, out: Path, arrival_scale: float) -> Path:
    """Copy a Hybrid config with absolute paths, dropping missing datasets."""
    cfg = yaml.safe_load((ROOT / f"TaskData/Hybrid/{name}/tasks_config.yaml").read_text())
    kept = []
    for task in cfg["tasks"]:
        path = ROOT / task["data_path"]
        if not path.is_file():
            # Configs point at an 8B-tokenized copy we don't have; the same
            # sessions one level up use the shared Llama-3 tokenizer.
            path = path.parent.parent / path.name
        if not path.is_file():
            print(f"  {name}: dropping {task['name']} (missing {task['data_path']})")
            continue
        task["data_path"] = str(path)
        task["lambda_rate"] = task["lambda_rate"] / arrival_scale
        kept.append(task)
    if not kept:
        raise SystemExit(f"{name}: no datasets available")
    cfg["tasks"] = kept
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tasks_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def simulate(policy, task_config, run_dir: Path, args, use_map=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "run_simulation.py", "--model", args.model,
           "--device", "cpu", "--max-num-seqs", "32", "--max-model-len", "131072",
           "--block-size", str(args.block_size), "--simulator-enable-virtual-time",
           "--simulator-num-gpu-blocks", str(args.blocks),
           "--simulator-num-cpu-blocks", "100000", "--eviction-policy", policy,
           "--simulator-output-dir", str(run_dir),
           "--simulator-task-config", str(task_config), "--log-level", "WARNING",
           "--simulator-timing", args.timing]
    if args.timing == "aic":
        cmd += ["--simulator-aic-system", args.system]
        if args.calibration:
            cmd += ["--simulator-timing-calibration", str(Path(args.calibration).resolve())]
    if policy != "NONE":
        cmd.append("--enable-prefix-caching")
    if use_map is not None:
        cmd += ["--opt-block-use-map-path", str(use_map)]
    env = dict(os.environ, PYTHONHASHSEED="0", VLLM_USE_V1="0", HF_HUB_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", TOKENIZERS_PARALLELISM="false",
               MPLBACKEND="Agg", COLLECT_TRACE="1" if policy == "NONE" else "0")
    for var, sub in [("MPLCONFIGDIR", "matplotlib"), ("VLLM_CACHE_ROOT", "vllm"),
                     ("VLLM_CONFIG_ROOT", "vllm-config")]:
        (run_dir / ".cache" / sub).mkdir(parents=True, exist_ok=True)
        env[var] = str(run_dir / ".cache" / sub)
    (run_dir / "command.json").write_text(json.dumps(cmd, indent=2))
    with (run_dir / "run.log").open("w") as log:
        status = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if status:
        raise RuntimeError(f"exit {status}: {run_dir / 'run.log'}")
    return sorted(run_dir.glob("*/summary.json"))[-1].parent


def spearman(a, b) -> float:
    """Rank correlation; ties get their average rank."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = (sum((x - ma)**2 for x in ra) * sum((y - mb)**2 for y in rb)) ** 0.5
    return cov / var if var else float("nan")


def latest_result(run_dir: Path) -> Path:
    return sorted(run_dir.glob("*/summary.json"))[-1].parent


def run_trace(label, name, args, pool):
    out = Path(args.out) / label
    if not args.table_only:
        task_config = trace_config(name, out, args.arrival_scale)
        # OPT needs the future-use map from a no-cache profile of the trace.
        profile = pool.submit(simulate, "NONE", task_config, out / "NONE", args)
        futures = [pool.submit(simulate, e, task_config, out / e, args)
                   for e in POLICIES.values() if e != "OPT"]
        use_map = profile.result() / "use_map.json"
        futures.append(pool.submit(simulate, "OPT", task_config, out / "OPT",
                                   args, use_map))
        for f in futures:
            f.result()
    rows = []
    for (paper_name, engine), paper_value in zip(POLICIES.items(),
                                                PAPER_QTTFT_S[label]):
        run_dir = out / engine
        if engine == "OPT" and (out / "OPT-refined").is_dir():
            run_dir = out / "OPT-refined"  # see refine_opt.py
        result = latest_result(run_dir)
        lat = json.loads((result / "latency_summary.json").read_text())
        hits = json.loads((result / "summary.json").read_text())["hit_rate_summary"]
        rows.append({"trace": label, "policy": paper_name,
                     "paper_qttft_s": paper_value,
                     "sim_qttft_s": lat["ttft_s"]["mean"],
                     "sim_qttft_p90_s": lat["ttft_s"]["p90"],
                     "sim_hit_rate": hits["global_hit_rate"],
                     "requests": lat["completed_requests"],
                     "run": str(run_dir.relative_to(Path(args.out)))})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--timing", choices=["aic", "legacy"], default="aic")
    p.add_argument("--system", default="a100_sxm")
    p.add_argument("--calibration", default=None)
    p.add_argument("--model", default=str(ROOT / "models/Llama-3.2-3B-Instruct"))
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--kv-gib", type=float, default=63.0, help="A (paper Table 4)")
    p.add_argument("--budget", type=float, default=0.9, help="fraction of A")
    p.add_argument("--arrival-scale", type=float, default=1.0)
    p.add_argument("--traces", nargs="+", default=list(TRACES))
    p.add_argument("--jobs", type=int, default=9)
    p.add_argument("--table-only", action="store_true",
                   help="rebuild table5.md/csv from existing run directories")
    args = p.parse_args()
    args.blocks = int(args.budget * args.kv_gib * 2**30
                      // kv_bytes_per_block(Path(args.model), args.block_size))
    print(f"KV budget {args.budget} x {args.kv_gib} GiB = {args.blocks} blocks "
          f"of {args.block_size} tokens; timing {args.timing}")

    with ThreadPoolExecutor(args.jobs) as pool:
        rows = [r for label in args.traces
                for r in run_trace(label, TRACES[label], args, pool)]

    out = Path(args.out)
    with open(out / "table5.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label in args.traces:
        group = [r for r in rows if r["trace"] == label]
        lru = next(r for r in group if r["policy"] == "LRU")
        for r in group:
            lines.append(f"| {label} | {r['policy']} | {r['paper_qttft_s']:.2f} | "
                         f"{r['sim_qttft_s']:.3f} | {r['sim_qttft_s'] / lru['sim_qttft_s']:.2f} | "
                         f"{r['paper_qttft_s'] / lru['paper_qttft_s']:.2f} | {r['sim_hit_rate']:.4f} |")
    lines += ["", "| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |",
              "|---|---:|---:|"]
    for label in args.traces:
        group = [r for r in rows if r["trace"] == label]
        sim = [r["sim_qttft_s"] for r in group]
        paper = [r["paper_qttft_s"] for r in group]
        lru_s, lru_p = sim[0], paper[0]  # LRU is first in POLICIES
        if max(sim) - min(sim) < 1e-9:
            lines.append(f"| {label} | n/a (all policies equal) | n/a |")
            continue
        same = sum((s < lru_s) == (q < lru_p) for s, q in zip(sim[1:], paper[1:]))
        lines.append(f"| {label} | {spearman(sim, paper):.2f} | {same}/{len(sim) - 1} |")
    (out / "table5.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
