"""Small launcher for the upstream simulator; never changes eviction logic."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
DEFAULT_MODEL = "models/Llama-3.1-8B-Instruct"
ONLINE = ["LRU", "LFU", "FIFO", "LeCaR", "ARC", "WA", "SimpleHYBRID"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_config(path):
    import yaml
    with open(path) as f:
        config = yaml.safe_load(f)
    if not config or not config.get("tasks"):
        raise ValueError(f"No tasks in {path}")
    missing = [t["data_path"] for t in config["tasks"]
               if not Path(t["data_path"]).is_file()]
    if missing:
        raise ValueError("Missing prepared datasets:\n  " + "\n  ".join(missing)
                         + "\nSee LOCAL_SETUP.md for data preparation.")
    return config


def download(model):
    from huggingface_hub import snapshot_download
    name = Path(model).name
    repo = f"unsloth/{name}"
    print(f"Downloading configuration/tokenizer only: {repo}", flush=True)
    snapshot_download(repo, local_dir=model, allow_patterns=[
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "generation_config.json",
    ])


def run(args):
    config = inspect_config(args.config)
    model = Path(args.model)
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        if not (model / file).is_file():
            raise ValueError(f"Missing {model / file}; run .venv/bin/python scripts/local_sim.py download --model {model}")
    policies = args.policies or (["SimpleHYBRID"] if args.command == "demo" else ONLINE)
    unknown = set(policies) - set(ONLINE + ["OPT", "OPT-Online", "SimpleHYBRID-OPT"])
    if unknown:
        raise ValueError(f"Unknown policies: {sorted(unknown)}")
    if any(b <= 0 for b in args.blocks):
        raise ValueError("Block counts must be positive")
    out = Path(args.output) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True)
    manifest = {
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arguments": vars(args), "task_config": config,
        "sha256": {p: digest(p) for p in {args.config, *[t["data_path"] for t in config["tasks"]],
                                          *map(str, model.glob("*.json"))}},
        "environment": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "VLLM_USE_V1",
            "SimpleHYBRID_Disable_Single_Time_Decay", "SimpleHYBRID_Disable_HIT_EFFICIENCY"]},
        "note": "Simulator results; not GPU latency measurements or verified exact paper traces.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "packages.txt").write_text(subprocess.check_output(
        ["uv", "pip", "freeze", "--python", sys.executable], text=True))
    rows = []

    def execute(policy, blocks, profile=False, use_map=None):
        folder = out / f"{blocks}-{policy}"
        folder.mkdir()
        cmd = [sys.executable, "run_simulation.py", "--model", str(model),
               "--max-num-seqs", str(args.max_num_seqs), "--max-model-len", "131072",
               "--block-size", "16", "--simulator-enable-virtual-time", "--device", "cpu",
               "--simulator-num-gpu-blocks", str(blocks), "--simulator-num-cpu-blocks", "0",
               "--eviction-policy", policy, "--simulator-output-dir", str(folder),
               "--simulator-task-config", args.config, "--log-level", "WARNING",
               "--simulator-ttft", args.ttft, "--simulator-tpot", str(args.tpot)]
        if not profile:
            cmd.append("--enable-prefix-caching")
        if use_map:
            cmd.extend(["--opt-block-use-map-path", str(use_map)])
        env = dict(os.environ, COLLECT_TRACE="1" if profile else "0", HF_HUB_OFFLINE="1")
        (folder / "command.json").write_text(json.dumps(cmd, indent=2))
        print(f"Running {policy}, {blocks} blocks; log: {folder / 'run.log'}", flush=True)
        with open(folder / "run.log", "w") as log:
            completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            tail = (folder / "run.log").read_text().splitlines()[-35:]
            raise RuntimeError("\n".join(tail) + f"\nRun failed: {folder / 'run.log'}")
        results = list(folder.glob("*/summary.json"))
        if len(results) != 1:
            raise RuntimeError(f"Expected one summary under {folder}, got {len(results)}")
        return results[0]

    for blocks in args.blocks:
        use_map = None
        if "OPT" in policies:
            profile = execute("NONE", blocks, profile=True)
            use_map = profile.parent / "use_map.json"
            if not use_map.is_file():
                raise RuntimeError(f"Oracle profile missing: {use_map}")
        for policy in policies:
            summary_file = execute(policy, blocks, use_map=use_map if policy == "OPT" else None)
            summary = json.loads(summary_file.read_text())
            hit = summary["hit_rate_summary"]
            row = dict(policy=policy, blocks=blocks, block_size=16,
                       requests=hit["total_requests"], hit_tokens=hit["global_hit_tokens_num"],
                       prompt_tokens=hit["global_tokens"], hit_ratio=hit["global_hit_rate"],
                       runtime_seconds=summary["simulation_time"], summary=str(summary_file))
            rows.append(row)
            with open(out / "comparison.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  {policy}: {row['hit_ratio']:.4%} token-weighted hit rate", flush=True)
    print(f"Results: {out / 'comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="Download model metadata only, no weights")
    d.add_argument("--model", default=DEFAULT_MODEL)
    doctor = sub.add_parser("doctor", help="Check a prepared task configuration")
    doctor.add_argument("--config", default="demo/tasks_config.yaml")
    for command in ["demo", "compare"]:
        p = sub.add_parser(command)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--config", default="demo/tasks_config.yaml")
        p.add_argument("--policies", nargs="+")
        p.add_argument("--blocks", nargs="+", type=int, default=[10000])
        p.add_argument("--max-num-seqs", type=int, default=32)
        p.add_argument("--ttft", default="3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)")
        p.add_argument("--tpot", type=float, default=0.01)
        p.add_argument("--output", default="result/local")
    args = parser.parse_args()
    if args.command == "download":
        download(args.model)
    elif args.command == "doctor":
        config = inspect_config(args.config)
        print(f"Python {sys.version.split()[0]}; {len(config['tasks'])} task files present.")
        for task in config["tasks"]:
            with open(task["data_path"]) as f:
                count = sum(1 for _ in f)
            print(f"  {task['name']}: {count} available sessions; limit {task.get('max_record', -1)}")
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
