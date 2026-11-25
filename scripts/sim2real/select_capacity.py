#!/usr/bin/env python3
"""Pick each subset's cache capacity C* from the sim sweep and pre-register.

Rules (docs/sim2real-transfer-methodology.md, "Ensuring real KV-cache
contention"), using heuristics and OPT only, never the agent:
  1. OPT - LRU >= 5 pp
  2. LRU evicts, and some evicted content is requested again (evicted_then_missed > 0)
  3. best - worst heuristic (LRU, UniCache, LeCaR, LFU, FIFO) >= 2 pp
  4. no thrashing: C >= 2 x the largest request's blocks, LRU preemptions <= 1% of requests
  5. replay <= 600 s
Among qualifying capacities, take the largest OPT - LRU.

Writes <sweep>/capacity_selection.csv and <out>/preregistration.{json,md}
with C* and the simulator's predicted hit ratio and delta vs LRU for every
Tier-1 policy at C*.

    .venv-sim312/bin/python scripts/sim2real/select_capacity.py \\
        result/sim2real-transfer/sweep result/sim2real-transfer/preregistration
"""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUBSETS, WORKLOADS  # noqa: E402

HEUR = ["LRU", "UniCache", "LeCaR", "LFU", "FIFO"]
TIER1 = ["LRU", "UniCache", "LeCaR", "Agent"]


def largest_request_blocks(workload):
    biggest = 0
    with open(SUBSETS / workload / "eval.jsonl") as f:
        for line in f:
            (_, s), = json.loads(line).items()
            for t in range(1, s["num_turns"] + 1):
                turn = s[f"turn_{t}"]
                biggest = max(biggest, len(turn["input_ids"]) + len(turn.get("response_ids", [])))
    return math.ceil(biggest / 8)


def main():
    sweep, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [r for r in csv.DictReader(open(sweep / "runs.csv")) if r["status"] == "0"
            and r["policy"] != "NONE"]
    by = {(r["workload"], int(r["blocks"]), r["policy"]): r for r in rows}
    subsets = {s["workload"]: s for s in json.loads((SUBSETS / "subsets.json").read_text())}
    table, prereg = [], {}
    for w in sorted({r["workload"] for r in rows}):
        need = 2 * largest_request_blocks(w)
        cands = []
        for b in sorted({k[1] for k in by if k[0] == w}):
            get = lambda p: by.get((w, b, p))
            if not all(get(p) for p in HEUR + ["OPT"]):
                continue
            h = {p: 100 * float(get(p)["hit_rate"]) for p in HEUR + ["OPT"]}
            lru = get("LRU")
            row = dict(workload=w, blocks=b, opt_minus_lru_pp=round(h["OPT"] - h["LRU"], 2),
                       heuristic_spread_pp=round(max(h[p] for p in HEUR) - min(h[p] for p in HEUR), 2),
                       lru_evictions=int(lru["evictions"]),
                       lru_evicted_then_missed=int(lru["evicted_then_missed"]),
                       lru_preemptions=int(lru["preemptions"]), replay_s=round(float(lru["replay_s"])),
                       min_blocks=need)
            row["ok_headroom"] = row["opt_minus_lru_pp"] >= 5
            row["ok_evictions"] = row["lru_evictions"] > 0 and row["lru_evicted_then_missed"] > 0
            row["ok_spread"] = row["heuristic_spread_pp"] >= 2
            row["ok_no_thrash"] = (b >= need and row["lru_preemptions"]
                                   <= 0.01 * subsets[w]["requests"])
            row["ok_replay"] = row["replay_s"] <= 600
            row["qualifies"] = all(row[k] for k in row if k.startswith("ok_"))
            table.append(row)
            if row["qualifies"]:
                cands.append(row)
        if not cands:
            prereg[w] = {"blocks": None, "reason": "no capacity qualifies; enlarge the subset"}
            continue
        best = max(cands, key=lambda r: (r["opt_minus_lru_pp"], -r["blocks"]))
        b = best["blocks"]
        hits = {p: 100 * float(by[(w, b, p)]["hit_rate"]) for p in TIER1 + ["OPT", "LFU", "FIFO"]
                if (w, b, p) in by}
        best_heur = max(["LRU", "UniCache", "LeCaR"], key=lambda p: hits[p])
        prereg[w] = {
            "blocks": b, "gib": round(b * 8 * 32768 / 2**30, 3),  # 32 KiB/token for 1B
            "subset": subsets[w], "selection": best,
            "sim_hit_pct": {p: round(v, 3) for p, v in hits.items()},
            "best_heuristic_tier1": best_heur,
            "predicted_delta_vs_lru_pp": {p: round(hits[p] - hits["LRU"], 3)
                                          for p in ["UniCache", "LeCaR", "Agent"] if p in hits},
            "predicted_agent_minus_best_pp": (round(hits["Agent"] - hits[best_heur], 3)
                                              if "Agent" in hits else None),
        }

    with open(sweep / "capacity_selection.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preregistration.json").write_text(json.dumps(prereg, indent=2))

    lines = ["# Sim2real pre-registration (simulator predictions before any GPU run)", "",
             "Simulator: H1's UniCache simulator (legacy timing), eval-split subsets, "
             "0 CPU blocks. Hit ratios in %, deltas in pp.", "",
             "| Workload | Sessions | Requests | C* (blocks) | OPT−LRU | LRU | UniCache | LeCaR | Agent | OPT | Agent−LRU | Agent−Best | Best (tier 1) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for w, p in prereg.items():
        if p["blocks"] is None:
            lines.append(f"| {w} | | | none | | | | | | | | | {p['reason']} |")
            continue
        h, d = p["sim_hit_pct"], p["predicted_delta_vs_lru_pp"]
        lines.append(f"| {w} | {p['subset']['sessions']} | {p['subset']['requests']} | {p['blocks']} | "
                     f"{p['selection']['opt_minus_lru_pp']:.2f} | {h['LRU']:.2f} | {h['UniCache']:.2f} | "
                     f"{h['LeCaR']:.2f} | {h.get('Agent', float('nan')):.2f} | {h['OPT']:.2f} | "
                     f"{d.get('Agent', float('nan')):+.2f} | "
                     f"{(p['predicted_agent_minus_best_pp'] or float('nan')):+.2f} | {p['best_heuristic_tier1']} |")
    (out / "preregistration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
