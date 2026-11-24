#!/usr/bin/env python3
"""Make offline OPT self-consistent under cache-aware timing.

OPT's future-use map is keyed by prefill iteration number and comes from a
no-cache (NONE) profiling run. With legacy timing every policy schedules
prefills in the same order, so that map is exact. With cache-aware timing
(``--simulator-timing aic``), cache hits change step times, which changes
batching and when follow-up turns arrive. The NONE order then no longer
matches the OPT run, and "OPT" can fall below LRU.

Fix: fixed-point iteration. Re-run OPT with COLLECT_TRACE=1 so it writes a
future-use map from its *own* prefill order, feed that map to the next run,
and stop when the hit ratio changes by less than ``--tol``. At the fixed
point the map is exactly the run's own future (Belady's definition).

    .venv-sim312/bin/python scripts/paper_qttft/refine_opt.py \\
      --opt-dir 'result/paper-qttft/aic-a100/Trace#1/OPT' --iterations 4
The last iteration is copied to <opt-dir>/../OPT-refined.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summary_dir(run_dir: Path) -> Path:
    return sorted(run_dir.glob("*/summary.json"))[-1].parent


def hit_and_qttft(result: Path):
    hits = json.loads((result / "summary.json").read_text())["hit_rate_summary"]
    lat = json.loads((result / "latency_summary.json").read_text())
    return hits["global_hit_rate"], lat["ttft_s"]["mean"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opt-dir", required=True, help="an OPT run with command.json")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--tol", type=float, default=0.001, help="hit-ratio change to stop")
    args = p.parse_args()

    opt_dir = Path(args.opt_dir).resolve()
    base_cmd = json.loads((opt_dir / "command.json").read_text())
    use_map = Path(base_cmd[base_cmd.index("--opt-block-use-map-path") + 1])
    first = summary_dir(opt_dir)
    # Resuming from an earlier OPT-iterN run: it already wrote its own map.
    start = int(opt_dir.name.split("iter")[-1]) if "-iter" in opt_dir.name else 0
    resumed = (first / "use_map.json").is_file()
    if resumed:
        use_map = first / "use_map.json"
    history = [dict(zip(["hit_rate", "qttft_s"], hit_and_qttft(first)),
                    iteration=start, use_map=str(use_map))]
    print(f"iter {start} ({opt_dir.name}): hit {history[0]['hit_rate']:.4f} "
          f"QTTFT {history[0]['qttft_s']:.3f}s")
    last_run = opt_dir
    for k in range(start + 1, start + args.iterations + 1):
        run_dir = opt_dir.parent / f"OPT-iter{k}"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir()
        cmd = list(base_cmd)
        cmd[cmd.index("--opt-block-use-map-path") + 1] = str(use_map)
        cmd[cmd.index("--simulator-output-dir") + 1] = str(run_dir)
        env = dict(os.environ, PYTHONHASHSEED="0", VLLM_USE_V1="0",
                   HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                   TOKENIZERS_PARALLELISM="false", MPLBACKEND="Agg",
                   COLLECT_TRACE="1")
        for var, sub in [("MPLCONFIGDIR", "matplotlib"), ("VLLM_CACHE_ROOT", "vllm"),
                         ("VLLM_CONFIG_ROOT", "vllm-config")]:
            (run_dir / ".cache" / sub).mkdir(parents=True, exist_ok=True)
            env[var] = str(run_dir / ".cache" / sub)
        (run_dir / "command.json").write_text(json.dumps(cmd, indent=2))
        with (run_dir / "run.log").open("w") as log:
            if subprocess.call(cmd, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT):
                raise SystemExit(f"OPT iteration {k} failed: {run_dir / 'run.log'}")
        result = summary_dir(run_dir)
        hit, qttft = hit_and_qttft(result)
        history.append({"iteration": k, "hit_rate": hit, "qttft_s": qttft,
                        "use_map": str(use_map)})
        print(f"iter {k}: hit {hit:.4f} QTTFT {qttft:.3f}s", flush=True)
        use_map = result / "use_map.json"  # this run's own future
        last_run = run_dir
        # Iteration 1 reuses the NONE map (it only adds trace collection), so
        # it always equals iteration 0; compare from iteration 2 on.
        if (resumed or k >= 2) and abs(hit - history[-2]["hit_rate"]) < args.tol:
            break

    refined = opt_dir.parent / "OPT-refined"
    shutil.rmtree(refined, ignore_errors=True)
    shutil.copytree(last_run, refined, ignore=shutil.ignore_patterns(".cache"))
    log_path = opt_dir.parent / "opt_refinement.json"
    earlier = json.loads(log_path.read_text()) if resumed and log_path.is_file() else []
    log_path.write_text(json.dumps(earlier + history[1:] if earlier else history,
                                   indent=2))
    print(f"Copied {last_run.name} to {refined}")


if __name__ == "__main__":
    main()
