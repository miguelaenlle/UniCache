"""Step 3 report: simulator vs GH200 for the TTFT-objective agent run (CodeParrot-600, 25x, 4,096 blocks).

Reads the GPU replays (result/sim2real-transfer/gpu-step3/*/runs.csv), the agent's submissions
(tune and post-hoc eval simulator scores) and the simulator baselines, then writes
result/sim2real-transfer/step3/{sim-runs.csv,gpu-runs.csv,summary.json,*.png}.

Simulator baseline values (built-in policies, OPT) come from the v8-calibrated runs reported in
docs/step3-results.md sections 1 and 2a (agent run baseline-results.json and step3/sim-opt).
"""
import csv
import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPU = os.path.join(ROOT, "result/sim2real-transfer/gpu-step3")
OUT = os.path.join(ROOT, "result/sim2real-transfer/step3")
AGENT = os.path.join(ROOT, "result/policy-agent/2026-11-26/151145-668720-codeparrot")

# GPU run directory -> policy label. c1 = submission 4 (selected), c2 = submission 0, c3 = submission 1.
DIR_LABEL = {"A-c1": "Agent s4", "B-c1": "Agent s4", "A-c2": "Agent s0", "B-c3": "Agent s1"}

# (hit %, TTFT mean ms, TTFT p90 ms, ITL mean ms) from docs/step3-results.md.
SIM = {
    "tune": {
        "LRU": (14.50, 45.00, 130.58, 5.00), "LFU": (15.62, 46.62, 131.54, 5.01),
        "FIFO": (13.24, 45.77, 128.84, 5.01), "LeCaR": (15.30, 46.13, 136.74, 5.01),
        "UniCache": (15.61, 47.40, 140.05, 5.01), "OPT": (28.69, 36.57, 95.59, 4.92),
    },
    "eval": {
        "LRU": (15.44, 81.23, 313.35, 5.14), "LFU": (17.88, 79.24, 314.22, 5.10),
        "FIFO": (15.70, 80.89, 313.35, 5.13), "LeCaR": (17.82, 79.21, 317.47, 5.09),
        "UniCache": (18.43, 79.67, 319.27, 5.10), "OPT": (31.18, 74.16, 301.59, 5.00),
    },
}
ORDER = ["LRU", "LFU", "FIFO", "LeCaR", "UniCache", "Agent s4", "Agent s0", "Agent s1"]


def agent_sim():
    subs = [json.loads(l) for l in open(os.path.join(AGENT, "submissions.jsonl"))]
    for s in subs:
        for split in ("tune", "eval"):
            sc = s[split]
            SIM[split][f"Agent s{s['submission']}"] = (
                sc["token_weighted_hit_ratio"] * 100, sc["ttft_mean_s"] * 1e3,
                sc["ttft_p90_s"] * 1e3, sc["itl_mean_s"] * 1e3)
    return subs


def gpu_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(GPU, "*/runs.csv"))):
        d = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f)):
            rows.append({
                "split": "eval", "policy": DIR_LABEL.get(d, r["policy"]), "half": d[0],
                "rep": r["rep"], "node": "gh152", "hit_pct": float(r["hit_rate"]) * 100,
                "ttft_mean_ms": float(r["ttft_mean_s"]) * 1e3, "ttft_p90_ms": float(r["ttft_p90_s"]) * 1e3,
                "itl_mean_ms": float(r["itl_mean_s"]) * 1e3, "evictions": int(r["evictions"]),
                "replay_s": float(r["replay_s"]),
            })
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = agent_sim()
    runs = gpu_runs()
    with open(os.path.join(OUT, "gpu-runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    with open(os.path.join(OUT, "sim-runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "policy", "hit_pct", "ttft_mean_ms", "ttft_p90_ms", "itl_mean_ms"])
        for split in SIM:
            for p, v in SIM[split].items():
                w.writerow([split, p, *[round(x, 3) for x in v]])

    g = {}
    for p in ORDER:
        rs = [r for r in runs if r["policy"] == p]
        g[p] = {k: st.mean(r[k] for r in rs) for k in ("hit_pct", "ttft_mean_ms", "ttft_p90_ms", "itl_mean_ms")}
        g[p]["n"] = len(rs)
        g[p]["ttft_runs"] = [round(r["ttft_mean_ms"], 2) for r in rs]
        g[p]["itl_runs"] = [round(r["itl_mean_ms"], 3) for r in rs]
    sim_t = [SIM["eval"][p][1] for p in ORDER]
    gpu_t = [g[p]["ttft_mean_ms"] for p in ORDER]
    rho_all = spearmanr(sim_t, gpu_t)[0]
    builtin = ORDER[:5]
    rho_builtin = spearmanr([SIM["eval"][p][1] for p in builtin], [g[p]["ttft_mean_ms"] for p in builtin])[0]
    summary = {"gpu": g, "spearman_all": rho_all, "spearman_builtin": rho_builtin}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

    # Charts.
    x = range(len(ORDER))
    for metric, idx, fname, label in (("ttft_mean_ms", 1, "ttft-by-policy.png", "TTFT mean (ms)"),
                                      ("itl_mean_ms", 3, "itl-by-policy.png", "ITL mean (ms)")):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - 0.2 for i in x], [SIM["eval"][p][idx] for p in ORDER], 0.4, label="simulator (v8)", color="#7a9cc6")
        ax.bar([i + 0.2 for i in x], [g[p][metric] for p in ORDER], 0.4, label="GH200", color="#d08c60")
        for i, p in enumerate(ORDER):
            vals = g[p]["ttft_runs" if idx == 1 else "itl_runs"]
            ax.scatter([i + 0.2] * len(vals), vals, color="k", s=10, zorder=3)
        ax.axhline(SIM["eval"]["OPT"][idx], ls="--", color="gray", label="OPT (simulator)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(ORDER, rotation=20)
        ax.set_ylabel(label)
        ax.set_title(f"{label}, eval slice (dots = individual GPU runs)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, idx, metric, label in ((axs[0], 1, "ttft_mean_ms", "TTFT mean (ms)"), (axs[1], 3, "itl_mean_ms", "ITL mean (ms)")):
        s = [SIM["eval"][p][idx] for p in ORDER]
        r = [g[p][metric] for p in ORDER]
        ax.scatter(s, r)
        for p, a, b in zip(ORDER, s, r):
            ax.annotate(p, (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lo, hi = min(s + r) * 0.95, max(s + r) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"simulator {label}")
        ax.set_ylabel(f"GH200 {label}")
    fig.suptitle("Simulator vs GH200, eval slice")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-vs-gpu-scatter.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for p in ORDER:
        ax.scatter(g[p]["hit_pct"], g[p]["ttft_mean_ms"], color="#d08c60" if p.startswith("Agent") else "#4c72b0")
        ax.annotate(p, (g[p]["hit_pct"], g[p]["ttft_mean_ms"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("hit ratio (%)")
    ax.set_ylabel("TTFT mean (ms)")
    ax.set_title("GH200, eval: hit ratio vs TTFT (orange = sandboxed agent policies)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hit-vs-ttft-gpu.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [s["submission"] for s in subs]
    ax.plot(xs, [s["tune"]["ttft_mean_s"] * 1e3 for s in subs], "o-", label="agent submission (tune, sim)")
    ax.axhline(SIM["tune"]["LRU"][1], ls=":", color="k", label="LRU built-in (tune, sim)")
    ax.set_xlabel("submission")
    ax.set_ylabel("tune TTFT mean (ms)")
    ax.set_title("Agent selection curve (tune only; the agent sees this)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "agent-selection-curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
