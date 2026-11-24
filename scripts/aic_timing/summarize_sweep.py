#!/usr/bin/env python3
"""Tabulate and plot a load sweep made with rerun_with_latency.py.

Expects ``<sweep>/<config>/scale-<s>/[agent/]results.csv``. Writes
``<sweep>/sweep.csv`` (one row per config x load x policy) and
``<sweep>/ttft_vs_load.png`` (mean TTFT, absolute and relative to LRU, vs
offered load; one column per config, one line per policy).

    .venv-sim312/bin/python scripts/aic_timing/summarize_sweep.py \\
      result/aic-timing/codeparrot-32768-load-sweep
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-4 of the validated reference palette, fixed order.
POLICIES = [("LRU", "#2a78d6", "o"), ("UniCache", "#eb6834", "s"),
            ("Agent", "#1baf7a", "^"), ("OPT", "#eda100", "D")]


def policy_of(run_label: str) -> str:
    if "submission-" in run_label:
        return "Agent"
    return run_label.split("/")[-2]  # <budget>/<workload>/<policy>/<split>


def load(sweep: Path):
    rows = []
    for results in sorted(sweep.glob("*/scale-*/**/results.csv")):
        rel = results.relative_to(sweep).parts
        config, scale = rel[0], float(rel[1].removeprefix("scale-"))
        with open(results) as f:
            for r in csv.DictReader(f):
                if r.get("status") != "0" or not r.get("ttft_s_mean"):
                    continue
                rows.append({
                    "config": config, "arrival_scale": scale,
                    "load_x": 1.0 / scale, "policy": policy_of(r["run"]),
                    "hit_rate": float(r["hit_rate"]),
                    "ttft_mean_ms": float(r["ttft_s_mean"]) * 1e3,
                    "ttft_p90_ms": float(r["ttft_s_p90"]) * 1e3,
                    "itl_mean_ms": float(r["itl_s_mean"]) * 1e3,
                    "e2e_mean_s": float(r["e2e_s_mean"]),
                })
    return rows


def main():
    sweep = Path(sys.argv[1])
    rows = load(sweep)
    if not rows:
        sys.exit(f"No completed results under {sweep}")
    rows.sort(key=lambda r: (r["config"], r["load_x"],
                             [p for p, _, _ in POLICIES].index(r["policy"])))
    with open(sweep / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    configs = sorted({r["config"] for r in rows})
    fig, axes = plt.subplots(2, len(configs), figsize=(5.2 * len(configs), 7),
                             squeeze=False)
    for col, config in enumerate(configs):
        lru = {r["load_x"]: r["ttft_mean_ms"] for r in rows
               if r["config"] == config and r["policy"] == "LRU"}
        for row_i in range(2):
            ax = axes[row_i][col]
            for name, color, marker in POLICIES:
                pts = sorted((r["load_x"], r["ttft_mean_ms"]) for r in rows
                             if r["config"] == config and r["policy"] == name)
                if row_i == 1:  # relative to LRU at the same load
                    pts = [(x, y / lru[x]) for x, y in pts if x in lru]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, marker=marker, markersize=8,
                        linewidth=2, label=name)
            ax.set_xscale("log")
            if row_i == 0:
                ax.set_yscale("log")
                ax.set_title(f"{config}: mean TTFT", fontsize=11, loc="left")
                ax.set_ylabel("ms")
            else:
                ax.set_title(f"{config}: mean TTFT relative to LRU",
                             fontsize=11, loc="left")
                ax.set_ylabel("x LRU (lower is better)")
            ax.set_xlabel("Offered load (x recorded arrival rate)")
            ax.grid(True, which="major", color="#e4e3dd", linewidth=0.8)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            if col == 0 and row_i == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(sweep / "ttft_vs_load.png", dpi=150)

    print(f"Wrote {sweep / 'sweep.csv'} and {sweep / 'ttft_vs_load.png'}")
    print(f"{'config':<14}{'load':>6}  {'policy':<9}{'hit':>7}"
          f"{'TTFT mean':>12}{'TTFT p90':>12}{'ITL':>9}")
    for r in rows:
        print(f"{r['config']:<14}{r['load_x']:>5.0f}x  {r['policy']:<9}"
              f"{r['hit_rate']:>7.3f}{r['ttft_mean_ms']:>10.1f}ms"
              f"{r['ttft_p90_ms']:>10.1f}ms{r['itl_mean_ms']:>7.2f}ms")


if __name__ == "__main__":
    main()
