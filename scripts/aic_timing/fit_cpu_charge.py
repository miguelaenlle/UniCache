#!/usr/bin/env python3
"""Extend a timing calibration with a CPU-time term fitted from paired runs.

On a GPU, the engine's CPU work per step (scheduler, block manager, eviction-policy calls) is on
the critical path; aiconfigurator's kernel tables omit it, and so does the base calibration's
constant overhead when policies differ. For each (GPU run, simulator run) pair of the same
trace and policy, we fit, per phase, by least squares over all GPU steps:

    measured_ms = scale * aic_ms + overhead_ms + cpu_scale * mean_sim_cpu_ms(run)

where mean_sim_cpu_ms(run) is the simulator's mean measured engine.step() CPU time for that phase
(step-log column cpu_ms, written since 2026-11-26). cpu_scale converts the simulator machine's
CPU time into the GPU host's. The simulator then charges cpu_scale * cpu_ms on its virtual clock
for every step (vllm/simulator/simulator.py), so heavier policies cost time as on the GPU.

    .venv-sim312/bin/python scripts/aic_timing/fit_cpu_charge.py \\
      --pair <gpu steps.csv>=<sim steps.csv> ... \\
      --model models/Llama-3.2-1B-Instruct --system h100_sxm --out calibrations/<name>.json
A phase whose cpu_scale fit is negative or whose CPU time does not vary across runs keeps
cpu_scale 0 and the plain two-parameter fit.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vllm.simulator.step_timing import AicStepTiming, fpm_from_counts  # noqa: E402

COUNTS = ["num_prefill", "prefill_new_tokens", "prefill_cached_tokens", "num_decode",
          "decode_kv_tokens"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs="+", required=True, help="GPU_steps.csv=SIM_steps.csv")
    p.add_argument("--extra", nargs="*", default=[],
                   help="GPU_steps.csv=CPU_MS: steps with no paired simulator run (e.g. the "
                        "profile_steps_grid.py grid), given a fixed per-step CPU time")
    p.add_argument("--model", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-cpu", nargs="*", default=[],
                   help="phases fitted without the CPU term (e.g. decode: its simulator CPU time grows "
                        "with batch size and double-counts the batch-size slope)")
    a = p.parse_args()
    timing = AicStepTiming(a.model, a.system)
    data = {}  # phase -> list of (aic_ms, run_cpu_ms, measured_ms, run)
    run_cpu = {}
    for k, pair in enumerate(a.pair):
        gpu, sim = pair.split("=")
        sim_rows = list(csv.DictReader(open(sim)))
        for row in csv.DictReader(open(gpu)):
            ph = row["phase"]
            if ph not in run_cpu.setdefault(k, {}):
                xs = [float(r["cpu_ms"]) for r in sim_rows if r["phase"] == ph and r.get("cpu_ms")]
                run_cpu[k][ph] = float(np.mean(xs)) if xs else 0.0
            counts = {c: int(float(row[c])) for c in COUNTS}
            data.setdefault(ph, []).append(
                (timing.raw_ms_fpm(fpm_from_counts(**counts)), run_cpu[k][ph],
                 float(row["measured_ms"]), k))
    for extra in a.extra:
        gpu, cpu_ms = extra.rsplit("=", 1)
        k = len(a.pair)
        a.pair.append(f"{gpu}=fixed:{cpu_ms}")
        for row in csv.DictReader(open(gpu)):
            counts = {c: int(float(row[c])) for c in COUNTS}
            data.setdefault(row["phase"], []).append(
                (timing.raw_ms_fpm(fpm_from_counts(**counts)), float(cpu_ms),
                 float(row["measured_ms"]), k))
    cal = {"scale": {}, "overhead_ms": {}, "cpu_scale": {},
           "fit": {"model": a.model, "system": a.system, "pairs": a.pair, "phases": {}}}
    for ph, rows in data.items():
        x = np.array(rows)
        aic, cpu, y = x[:, 0], x[:, 1], x[:, 2]
        A3 = np.column_stack([aic, np.ones_like(aic), cpu])
        coef3 = np.linalg.lstsq(A3, y, rcond=None)[0]
        if ph not in a.no_cpu and np.ptp(cpu) > 1e-6 and coef3[2] > 0 and coef3[0] >= 0:
            scale, overhead, cpu_scale = coef3
        else:
            scale, overhead = np.polyfit(aic, y, 1) if np.ptp(aic) > 0 else (0.0, y.mean())
            if scale < 0:
                scale, overhead = 0.0, float(y.mean())
            cpu_scale = 0.0
        pred = scale * aic + overhead + cpu_scale * cpu
        per_run = {}
        for k in sorted(set(x[:, 3].astype(int))):
            m = x[:, 3] == k
            per_run[a.pair[k].split("=")[0]] = {
                "gpu_mean_ms": float(y[m].mean()), "pred_mean_ms": float(pred[m].mean()),
                "sim_cpu_ms": float(cpu[m][0])}
        cal["scale"][ph], cal["overhead_ms"][ph], cal["cpu_scale"][ph] = (
            float(scale), float(overhead), float(cpu_scale))
        cal["fit"]["phases"][ph] = {
            "rows": len(rows),
            "median_ape": float(np.median(np.abs(pred - y) / y)),
            "per_run_mean_abs_err_ms": float(np.mean([abs(v["gpu_mean_ms"] - v["pred_mean_ms"])
                                                      for v in per_run.values()])),
            "per_run": per_run}
        print(f"{ph:<8} scale {scale:.3f} overhead {overhead:.3f} ms cpu_scale {cpu_scale:.3f}"
              f"  step median error {cal['fit']['phases'][ph]['median_ape']:.1%}"
              f"  per-run mean |err| {cal['fit']['phases'][ph]['per_run_mean_abs_err_ms']:.2f} ms")
    Path(a.out).write_text(json.dumps(cal, indent=2))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
