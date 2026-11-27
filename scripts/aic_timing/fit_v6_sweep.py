#!/usr/bin/env python3
"""Fit the GH200 timing calibration (v6) on the synthetic sweep only (no trace data).

  prefill / decode: measured_ms = scale * aiconfigurator_ms + overhead_ms, least squares over every
                    GPU step of the sweep's built-in-LRU runs (scripts/aic_timing/fit_calibration.py
                    does the per-step fit, with the negative-slope guard).
  policy_call_ms:   (total GPU step time of the sandboxed-LRU eviction run - total of the built-in
                    LRU run, same workload and decisions) - worker CPU, divided by the number of
                    policy IPC calls (counted in the paired simulator run).

    .venv-sim312/bin/python scripts/aic_timing/fit_v6_sweep.py --gpu result/sim2real-transfer/gpu-sweep \\
      --sim result/sim2real-transfer/sweep-sim-test --out calibrations/gh200-llama-3.2-1b-v6.json
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def steps(path):
    return list(csv.DictReader(open(path)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", required=True, help="GPU sweep results dir (run_sim --out)")
    p.add_argument("--sim", required=True, help="simulator run of sweep-evict with LRU and Agent (step logs)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lru_steps = [f for f in glob.glob(f"{a.gpu}/sweep-*/*/LRU/steps.csv")]
    if not lru_steps:
        sys.exit("no GPU sweep step logs")
    tmp = Path(a.out).with_suffix(".base.json")
    subprocess.run([sys.executable, str(ROOT / "scripts/aic_timing/fit_calibration.py"),
                    "--model", str(ROOT / "models/Llama-3.2-1B-Instruct"), "--system", "h100_sxm",
                    "--out", str(tmp), "--steps", *lru_steps], check=True)
    cal = json.loads(tmp.read_text())
    # Policy IPC cost from the eviction pair.
    g_lru = steps(glob.glob(f"{a.gpu}/sweep-evict/512/LRU/steps.csv")[0])
    g_agent = steps(glob.glob(f"{a.gpu}/sweep-evict/512/Agent/steps.csv")[0])
    s_agent = steps(glob.glob(f"{a.sim}/sweep-evict/512/Agent/steps.csv")[0])
    extra_ms = sum(float(r["measured_ms"]) for r in g_agent) - sum(float(r["measured_ms"]) for r in g_lru)
    calls = sum(float(r["policy_calls"] or 0) for r in s_agent)
    worker_ms = sum(float(r["policy_cpu_ms"] or 0) for r in s_agent)
    call_ms = max(0.0, (extra_ms - worker_ms) / calls) if calls else 0.0
    cal.update({"cpu_scale": {"prefill": 0.0, "decode": 0.0}, "policy_call_ms": call_ms,
                "policy_cpu_scale": 1.0})
    cal["fit"]["source"] = "synthetic sweep only (scripts/sim2real/make_sweep.py); traces held out"
    cal["fit"]["policy"] = {"gpu_extra_ms_total": extra_ms, "ipc_calls": calls,
                            "worker_cpu_ms_total": worker_ms, "policy_call_ms": call_ms,
                            "gpu_steps_lru": len(g_lru), "gpu_steps_agent": len(g_agent)}
    Path(a.out).write_text(json.dumps(cal, indent=2))
    tmp.unlink()
    print(f"policy: GPU extra {extra_ms:.0f} ms over {calls:.0f} IPC calls (worker CPU {worker_ms:.0f} ms)"
          f" -> policy_call_ms {call_ms:.4f}")
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
