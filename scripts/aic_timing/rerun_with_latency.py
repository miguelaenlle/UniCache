#!/usr/bin/env python3
"""Re-run recorded simulator replays with a chosen step-timing model.

Every replay in the H1 study saved its exact command (``command.json``). This
script re-executes those commands from *this* checkout, changing only the
output directory and the timing flags, then tabulates hit ratio next to
TTFT/ITL/E2E. With ``--timing legacy`` it should reproduce the recorded hit
ratios exactly, which is the regression check.

Example (CodeParrot, 8 GiB, three baselines, eval split, A100 tables):

    .venv-sim312/bin/python scripts/aic_timing/rerun_with_latency.py \\
      --commands '../UniCache-agent-budget-v1/result/agent-cache-budget-study/2026-11-01-v1-verified/baseline-replays/32768/codeparrot/{LRU,UniCache,OPT}/eval/command.json' \\
      --out result/aic-timing/codeparrot-32768 --timing aic --system a100_sxm
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Absolute paths recorded before the project moved (see top-level CLAUDE.md).
OLD_PREFIX = "/Users/miguel/Documents/ChatGPT/Agentic GPU Environment/"
NEW_PREFIX = str(ROOT.parent) + "/"


def expand(patterns):
    paths = []
    for pattern in patterns:
        # Support shell-style {a,b} lists without relying on the shell.
        if "{" in pattern and "}" in pattern:
            head, rest = pattern.split("{", 1)
            options, tail = rest.split("}", 1)
            paths += expand([head + o + tail for o in options.split(",")])
        else:
            paths += sorted(glob.glob(pattern))
    return paths


def remap(value: str) -> str:
    return value.replace(OLD_PREFIX, NEW_PREFIX)


def option(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def set_option(cmd, name, value):
    if name in cmd:
        cmd[cmd.index(name) + 1] = value
    else:
        cmd += [name, value]


def label(command_path: Path) -> str:
    """.../<budget>/<workload>/<policy>/<split>/command.json -> a label."""
    parts = command_path.parent.parts
    return "/".join(parts[-4:])


def prepare(command_path: Path, out_root: Path, args):
    cmd = [remap(c) for c in json.loads(command_path.read_text())]
    cmd[0] = sys.executable  # this checkout's environment
    run_dir = out_root / label(command_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Rewrite the task config so its data paths point into this folder.
    task_path = Path(option(cmd, "--simulator-task-config"))
    tasks = yaml.safe_load(task_path.read_text())
    for task in tasks["tasks"]:
        task["data_path"] = remap(task["data_path"])
        if args.arrival_scale != 1.0:
            # Load knob: compress (<1) or stretch (>1) first-turn arrival
            # times. Follow-up turns keep their recorded think times.
            if "frozen_arrivals" in task:
                task["frozen_arrivals"] = {
                    k: v * args.arrival_scale
                    for k, v in task["frozen_arrivals"].items()}
            else:
                task["lambda_rate"] = task["lambda_rate"] / args.arrival_scale
    local_tasks = run_dir / "tasks_config.yaml"
    local_tasks.write_text(yaml.safe_dump(tasks, sort_keys=False))
    set_option(cmd, "--simulator-task-config", str(local_tasks))
    set_option(cmd, "--simulator-output-dir", str(run_dir))

    if args.model:
        set_option(cmd, "--model", args.model)
    set_option(cmd, "--simulator-timing", args.timing)
    if args.timing == "aic":
        set_option(cmd, "--simulator-aic-system", args.system)
        if args.calibration:
            set_option(cmd, "--simulator-timing-calibration",
                       str(Path(args.calibration).resolve()))
    if args.step_log:
        set_option(cmd, "--simulator-step-log", str(run_dir / "steps.csv"))
    cmd += args.extra
    (run_dir / "command.json").write_text(json.dumps(cmd, indent=2))

    env = dict(os.environ, PYTHONHASHSEED="0", VLLM_USE_V1="0",
               HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               TOKENIZERS_PARALLELISM="false", MPLBACKEND="Agg",
               UNICACHE_COMPLETIONS=str(run_dir / "completed.json"))
    if args.candidate:
        # An agent-written v1 policy.py, injected exactly as the policy
        # agent's evaluator does (vllm/core/evictor.py:make_evictor).
        env["UNICACHE_POLICY_CANDIDATE"] = str(Path(args.candidate).resolve())
    for var, sub in [("MPLCONFIGDIR", "matplotlib"),
                     ("VLLM_CACHE_ROOT", "vllm"),
                     ("VLLM_CONFIG_ROOT", "vllm-config")]:
        (run_dir / ".cache" / sub).mkdir(parents=True, exist_ok=True)
        env[var] = str(run_dir / ".cache" / sub)
    return cmd, env, run_dir


def run(command_path: Path, out_root: Path, args):
    cmd, env, run_dir = prepare(command_path, out_root, args)
    with (run_dir / "run.log").open("w") as log:
        status = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    row = {"run": label(command_path), "status": status}
    summaries = sorted(run_dir.glob("*/summary.json"))
    if status == 0 and summaries:
        out = summaries[-1].parent
        hits = json.loads(summaries[-1].read_text())["hit_rate_summary"]
        lat = json.loads((out / "latency_summary.json").read_text())
        recorded = sorted(command_path.parent.glob("*/summary.json"))
        row["hit_rate"] = hits["global_hit_rate"]
        if recorded:
            row["recorded_hit_rate"] = json.loads(
                recorded[-1].read_text())["hit_rate_summary"]["global_hit_rate"]
        for metric in ["ttft_s", "itl_s", "e2e_s"]:
            for stat in ["mean", "p50", "p90", "p99"]:
                row[f"{metric}_{stat}"] = (lat[metric] or {}).get(stat)
    print(f"{row['run']}: exit {status}", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commands", nargs="+", required=True,
                   help="command.json paths or globs ({a,b} lists allowed)")
    p.add_argument("--out", required=True)
    p.add_argument("--timing", choices=["legacy", "aic"], default="aic")
    p.add_argument("--system", default="h100_sxm")
    p.add_argument("--calibration", default=None)
    p.add_argument("--model", default=None,
                   help="Override --model (e.g. a Llama-3.1-8B directory)")
    p.add_argument("--step-log", action="store_true")
    p.add_argument("--candidate", default=None,
                   help="Agent policy.py to evaluate (v1 interface)")
    p.add_argument("--arrival-scale", type=float, default=1.0,
                   help="Multiply first-turn arrival times (0.1 = 10x load)")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="After '--': extra run_simulation.py flags")
    args = p.parse_args()
    args.extra = [a for a in args.extra if a != "--"]

    commands = [Path(c).resolve() for c in expand(args.commands)]
    if not commands:
        sys.exit("No command.json matched")
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda c: run(c, out_root, args), commands))

    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k != "run", k != "status", k))
    with (out_root / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_root / 'results.csv'}")
    for r in rows:
        if "hit_rate" in r:
            print(f"  {r['run']:<40} hit {r['hit_rate']:.4f} "
                  f"(recorded {r.get('recorded_hit_rate', float('nan')):.4f}) "
                  f"TTFT mean {r['ttft_s_mean']*1e3:8.2f} ms "
                  f"p90 {r['ttft_s_p90']*1e3:8.2f} ms  "
                  f"ITL mean {(r['itl_s_mean'] or 0)*1e3:6.2f} ms")


if __name__ == "__main__":
    main()
