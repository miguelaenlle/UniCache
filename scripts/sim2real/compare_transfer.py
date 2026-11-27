"""Sim vs GPU transfer table for the sim2real test (docs/sim2real-transfer-methodology.md).

Reads the pre-registered simulator hit ratios and the GPU runs.csv files, then for each trace
reports GPU hit ratios, policy-minus-LRU deltas on both sides, and whether each decision
transfers: same sign and |delta_sim - delta_gpu| <= max(0.5 pp, 2 x noise floor), where the
noise floor is the spread (max - min) of the GPU LRU repeats.

    python scripts/sim2real/compare_transfer.py --gpu result/sim2real-transfer/gpu \
        [--prereg result/sim2real-transfer/preregistration-b16/preregistration.json]
Incomplete traces are shown with the runs that exist.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

POLICIES = ["UniCache", "LeCaR", "Agent"]


def load_prereg(path):
    d = json.loads(Path(path).read_text())
    return {w: (ast.literal_eval(v) if isinstance(v, str) else v) for w, v in d.items()}


def load_gpu(gpu_dir, workload):
    f = Path(gpu_dir) / workload / "runs.csv"
    rows = list(csv.DictReader(f.open())) if f.exists() else []
    out = {}
    for r in rows:
        if r.get("status") == "0" and r.get("hit_rate"):
            out.setdefault(r["policy"], []).append(100 * float(r["hit_rate"]))
    return out


def fmt(x, sign=False):
    return "—" if x is None else (f"{x:+.2f}" if sign else f"{x:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", required=True, help="directory with <workload>/runs.csv")
    p.add_argument("--prereg", default="result/sim2real-transfer/preregistration-b16/preregistration.json")
    a = p.parse_args()
    pre = load_prereg(a.prereg)
    for w, spec in pre.items():
        if not spec.get("blocks"):
            continue
        sim = spec["sim_hit_pct"]
        gpu = load_gpu(a.gpu, w)
        lru = gpu.get("LRU", [])
        lru_mean = sum(lru) / len(lru) if lru else None
        noise = (max(lru) - min(lru)) if len(lru) > 1 else None
        tol = max(0.5, 2 * noise) if noise is not None else 0.5
        print(f"\n### {w} ({spec['blocks']} blocks)\n")
        print(f"GPU LRU runs: {', '.join(f'{x:.2f}' for x in lru) or 'none'}"
              f"; noise floor {fmt(noise)} pp; tolerance {tol:.2f} pp\n")
        print("| Policy | Sim hit % | GPU hit % | Sim − LRU (pp) | GPU − LRU (pp) | Sim vs GPU diff (pp) | Transfers? |")
        print("|---|---:|---:|---:|---:|---:|---|")
        print(f"| LRU | {fmt(sim['LRU'])} | {fmt(lru_mean)} | | | | |")
        for pol in POLICIES:
            g = gpu.get(pol)
            g = sum(g) / len(g) if g else None
            ds = sim[pol] - sim["LRU"]
            dg = (g - lru_mean) if (g is not None and lru_mean is not None) else None
            if dg is None:
                verdict, diff = "pending", None
            else:
                diff = abs(ds - dg)
                same = (ds > 0) == (dg > 0) or (abs(ds) < 1e-9 and abs(dg) <= tol)
                verdict = "yes" if (same and diff <= tol) else "no"
            print(f"| {pol} | {fmt(sim[pol])} | {fmt(g)} | {fmt(ds, True)} | {fmt(dg, True)} | {fmt(diff)} | {verdict} |")
        best = max(("UniCache", "LeCaR"), key=lambda k: sim[k])
        ga, gb = gpu.get("Agent"), gpu.get(best)
        ds = sim["Agent"] - sim[best]
        dg = (sum(ga) / len(ga) - sum(gb) / len(gb)) if (ga and gb) else None
        print(f"\nAgent − best heuristic ({best}): sim {fmt(ds, True)} pp, GPU {fmt(dg, True)} pp")


if __name__ == "__main__":
    main()
