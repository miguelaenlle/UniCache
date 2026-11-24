#!/usr/bin/env python3
"""Compare simulated and measured latency for the same replay.

Both runs write ``latency_requests.csv`` (request ids are deterministic per
session and turn, so they line up). Reports, per metric, the measured and
simulated mean/p50/p90 and the per-request error, plus the step-level error
when both directories also contain ``steps.csv``.

    .venv-sim312/bin/python scripts/aic_timing/compare_latency.py \\
      --sim result/aic-timing/.../<run_dir> --real result/gpu/a100/.../<run_dir>
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ["ttft_s", "tpot_s", "e2e_s"]


def find(directory: Path, name: str) -> Path:
    hits = sorted(directory.rglob(name))
    if not hits:
        raise SystemExit(f"No {name} under {directory}")
    return hits[-1]


def read(path: Path):
    with open(path) as f:
        return {r["request_id"]: r for r in csv.DictReader(f)}


def stats(xs):
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--out", default=None, help="optional JSON report path")
    args = p.parse_args()

    sim = read(find(Path(args.sim), "latency_requests.csv"))
    real = read(find(Path(args.real), "latency_requests.csv"))
    common = sorted(set(sim) & set(real))
    report = {"requests": {"sim": len(sim), "real": len(real),
                           "matched": len(common)}, "metrics": {}}
    print(f"Matched {len(common)} requests (sim {len(sim)}, real {len(real)})")
    print(f"{'metric':<8}{'real mean':>12}{'sim mean':>12}{'real p90':>12}"
          f"{'sim p90':>12}{'median |err|':>14}")
    for m in METRICS:
        r = np.array([float(real[k][m]) for k in common])
        s = np.array([float(sim[k][m]) for k in common])
        keep = r > 0
        rel = np.abs(s[keep] - r[keep]) / r[keep]
        report["metrics"][m] = {"real": stats(r), "sim": stats(s),
                                "median_rel_error": float(np.median(rel)),
                                "p90_rel_error": float(np.percentile(rel, 90))}
        print(f"{m:<8}{r.mean()*1e3:>10.1f}ms{s.mean()*1e3:>10.1f}ms"
              f"{np.percentile(r, 90)*1e3:>10.1f}ms{np.percentile(s, 90)*1e3:>10.1f}ms"
              f"{np.median(rel):>14.1%}")

    real_steps = sorted(Path(args.real).rglob("steps.csv"))
    if real_steps:
        with open(real_steps[-1]) as f:
            rows = [r for r in csv.DictReader(f) if r.get("measured_ms")]
        if rows and rows[0].get("model_ms"):
            for phase in ["prefill", "decode"]:
                ph = [r for r in rows if r["phase"] == phase and r["model_ms"]]
                if ph:
                    model = np.array([float(r["model_ms"]) for r in ph])
                    meas = np.array([float(r["measured_ms"]) for r in ph])
                    rel = np.abs(model - meas) / meas
                    report.setdefault("steps", {})[phase] = {
                        "rows": len(ph), "median_rel_error": float(np.median(rel))}
                    print(f"step {phase}: median |err| {np.median(rel):.1%} "
                          f"over {len(ph)} steps")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
