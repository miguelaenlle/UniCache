#!/usr/bin/env python3
"""Fit the per-phase calibration of the aiconfigurator step-timing model.

Input: step CSVs with a measured GPU time per engine step. Two sources write
the same columns:
  * a GPU replay with ``--simulator-step-log`` (UniCache GPU mode), or
  * ``profile_steps_grid.py`` (controlled synthetic batches).

For each phase (prefill / decode / mixed) we fit, by least squares,

    measured_ms = scale * aiconfigurator_ms + overhead_ms

on 80% of the rows and report the error on the held-out 20%. The output JSON
is what ``--simulator-timing-calibration`` reads:

    {"scale": {"prefill": 1.12, "decode": 1.31},
     "overhead_ms": {"prefill": 0.8, "decode": 2.4},
     "fit": {...error before/after, row counts...}}

Example:
    .venv-sim312/bin/python scripts/aic_timing/fit_calibration.py \\
      --steps result/gpu/a100/*/steps.csv \\
      --model models/Llama-3.2-1B-Instruct --system h100_sxm \\
      --out calibrations/a100-llama-3.2-1b.json
"""
import argparse
import csv
import glob
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vllm.simulator.step_timing import (AicStepTiming,  # noqa: E402
                                        fpm_from_counts, phase_of)

COUNT_FIELDS = ["num_prefill", "prefill_new_tokens", "prefill_cached_tokens",
                "num_decode", "decode_kv_tokens"]


def load_rows(patterns, min_ms):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for r in csv.DictReader(f):
                    if not r.get("measured_ms"):
                        continue
                    counts = {k: int(float(r[k])) for k in COUNT_FIELDS}
                    ms = float(r["measured_ms"])
                    if ms < min_ms:
                        continue
                    rows.append({**counts, "measured_ms": ms, "file": path,
                                 "phase": phase_of(counts["num_prefill"],
                                                   counts["num_decode"])})
    return rows


def is_holdout(row, i):
    """Deterministic ~20% hold-out, stable across runs."""
    return zlib.crc32(f"{row['file']}:{i}".encode()) % 5 == 0


def errors(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    ape = np.abs(pred - true) / np.maximum(true, 1e-9)
    return {"mape": float(ape.mean()), "median_ape": float(np.median(ape)),
            "p90_ape": float(np.percentile(ape, 90)),
            "mean_bias_ms": float((pred - true).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", required=True, help="step CSV globs")
    p.add_argument("--model", required=True)
    p.add_argument("--system", required=True,
                   help="aiconfigurator tables to correct (e.g. h100_sxm)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-ms", type=float, default=0.0,
                   help="drop rows faster than this (timer noise)")
    args = p.parse_args()

    rows = load_rows(args.steps, args.min_ms)
    if not rows:
        sys.exit("No rows with measured_ms found")
    timing = AicStepTiming(args.model, args.system)
    for r in rows:
        r["aic_ms"] = timing.raw_ms_fpm(
            fpm_from_counts(**{k: r[k] for k in COUNT_FIELDS}))

    calibration = {"scale": {}, "overhead_ms": {}, "fit": {
        "model": args.model, "system": args.system,
        "aic_backend_version": timing.config["backend_version"],
        "steps": args.steps, "phases": {}}}
    for phase in ["prefill", "decode", "mixed"]:
        data = [(i, r) for i, r in enumerate(rows) if r["phase"] == phase]
        if not data:
            continue
        train = [r for i, r in data if not is_holdout(r, i)]
        test = [r for i, r in data if is_holdout(r, i)] or train
        x = np.array([r["aic_ms"] for r in train])
        y = np.array([r["measured_ms"] for r in train])
        if len(train) >= 2 and np.ptp(x) > 0:
            scale, overhead = np.polyfit(x, y, 1)
        else:  # not enough spread: keep the tables, fit a constant offset
            scale, overhead = 1.0, float((y - x).mean())
        calibration["scale"][phase] = float(scale)
        calibration["overhead_ms"][phase] = float(overhead)
        tx = np.array([r["aic_ms"] for r in test])
        ty = np.array([r["measured_ms"] for r in test])
        calibration["fit"]["phases"][phase] = {
            "train_rows": len(train), "test_rows": len(test),
            "test_error_uncalibrated": errors(tx, ty),
            "test_error_calibrated": errors(scale * tx + overhead, ty),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(calibration, indent=2))
    print(f"Wrote {args.out}")
    for phase, fit in calibration["fit"]["phases"].items():
        before = fit["test_error_uncalibrated"]["median_ape"]
        after = fit["test_error_calibrated"]["median_ape"]
        print(f"  {phase:<8} scale {calibration['scale'][phase]:.3f}  "
              f"overhead {calibration['overhead_ms'][phase]:.3f} ms  "
              f"held-out median error {before:.1%} -> {after:.1%} "
              f"({fit['train_rows']} train / {fit['test_rows']} test rows)")


if __name__ == "__main__":
    main()
