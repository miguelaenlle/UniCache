"""Build the A2 (paper Table 5) markdown tables from result/paper-repro/a2/<cfg>/<tag>/table5-T<n>.csv.

Usage: python scripts/paper_repro/a2_fill.py > result/paper-repro/a2/a2-tables.md
Headroom vs X = (QTTFT_X - QTTFT_OPT) / QTTFT_X. Best online = lowest-QTTFT non-OPT, non-NONE policy.
"""
import csv
import json
from pathlib import Path

A2 = Path(__file__).resolve().parents[2] / "result/paper-repro/a2"
CFGS = {"gh200-1b": "GH200-1B", "a100-3b": "A100-3B"}
TAGS = {"08kstar": "0.8 k*", "kstar": "k*"}
LOADS = {("gh200-1b", "08kstar"): (1.6, 2.4, 3.2), ("gh200-1b", "kstar"): (2, 3, 4),
         ("a100-3b", "08kstar"): (0.8, 1.6, 1.6), ("a100-3b", "kstar"): (1, 2, 2)}
ORDER = ["LRU", "LFU", "FIFO", "LeCaR", "ARC", "WA", "OPT (Offline)", "UniCache", "NONE (no caching)"]


def load(cfg, tag, n):
    p = A2 / cfg / tag / f"table5-T{n}.csv"
    if not p.exists():
        return None
    rows = {r["policy"]: r for r in csv.DictReader(open(p))}
    refined = (A2 / cfg / tag / f"Trace#{n}" / "OPT-refined").is_dir()
    return rows, refined


def opt_delta(cfg, tag, n):
    """Hit-ratio change in OPT's last refinement step (pp); small = converged fixed point."""
    p = A2 / cfg / tag / f"Trace#{n}" / "opt_refinement.json"
    if not p.exists():
        return ""
    h = [e["hit_rate"] for e in json.loads(p.read_text())]
    return f"{(h[-1] - h[-2]) * 100:+.2f}" if len(h) >= 2 else ""


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def stats(rows):
    paper = [p for p in ORDER[:-1]]
    sim = [float(rows[p]["sim_qttft_s"]) for p in paper]
    pap = [float(rows[p]["paper_qttft_s"]) for p in paper]
    same = sum((s < sim[0]) == (q < pap[0]) for s, q in zip(sim[1:], pap[1:]))
    online = {p: float(rows[p]["sim_qttft_s"]) for p in paper if p != "OPT (Offline)"}
    best = min(online, key=online.get)
    opt = float(rows["OPT (Offline)"]["sim_qttft_s"])
    return (spearman(sim, pap), same, (sim[0] - opt) / sim[0], (online[best] - opt) / online[best], best)


def main():
    out = []
    for tag, tname in TAGS.items():
        lg, la = LOADS[("gh200-1b", tag)], LOADS[("a100-3b", tag)]
        out.append(f"##### At {tname}: loads GH200-1B T#1–3 ×{lg[0]}/×{lg[1]}/×{lg[2]}; "
                   f"A100-3B ×{la[0]}/×{la[1]}/×{la[2]}\n")
        out.append("| Trace | Policy | Paper QTTFT (s) | Paper ×LRU | Ours, GH200-1B (s) | Ours ×LRU, GH200-1B "
                   "| Ours, A100-3B (s) | Ours ×LRU, A100-3B | Hit ratio GH200-1B | Hit ratio A100-3B |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n in (1, 2, 3):
            data = {c: load(c, tag, n) for c in CFGS}
            for pol in ORDER:
                cells = []
                ref = next((d[0][pol] for d in data.values() if d and pol in d[0]), None)
                pq = ref["paper_qttft_s"] if ref else ""
                plru = next((d[0]["LRU"]["paper_qttft_s"] for d in data.values() if d), "")
                cells.append(f"{float(pq):.2f}" if pq else "—")
                cells.append(f"{float(pq) / float(plru):.2f}" if pq and plru else "—")
                hits = []
                for c in CFGS:
                    d = data[c]
                    if not d or pol not in d[0]:
                        cells += ["", ""]; hits.append(""); continue
                    r = d[0][pol]
                    s, l = float(r["sim_qttft_s"]), float(d[0]["LRU"]["sim_qttft_s"])
                    cells += [f"{s:.3f}", f"{s / l:.2f}"]
                    hits.append(f"{float(r['sim_hit_rate']):.3f}")
                conv = all(opt_delta(c, tag, n) in ("+0.00", "-0.00") for c in CFGS)
                name = ("OPT" if conv else "OPT (not converged)") if pol.startswith("OPT") else pol
                out.append(f"| T#{n} | {name} | " + " | ".join(cells + hits) + " |")
        out.append("")
        out.append("| Trace | ρ vs paper / same side of LRU: GH200-1B | A100-3B | Headroom vs LRU: paper | "
                   "GH200-1B | A100-3B | Headroom vs best online: paper | GH200-1B | A100-3B | "
                   "OPT last-step Δhit (pp): GH200-1B | A100-3B |")
        out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        paper_h = {1: ("22 %", "1 %"), 2: ("25 %", "12 %"), 3: ("38 %", "20 %")}
        for n in (1, 2, 3):
            st = {c: (stats(load(c, tag, n)[0]) if load(c, tag, n) else None) for c in CFGS}
            f = lambda c, i: (f"{st[c][i] * 100:.0f} %" if st[c] else "")
            rho = lambda c: (f"{st[c][0]:.2f} / {st[c][1]}/7" if st[c] else "")
            best = lambda c: (f"{st[c][3] * 100:.0f} % ({st[c][4]})" if st[c] else "")
            out.append(f"| T#{n} | {rho('gh200-1b')} | {rho('a100-3b')} | {paper_h[n][0]} | {f('gh200-1b', 2)} | "
                       f"{f('a100-3b', 2)} | {paper_h[n][1]} | {best('gh200-1b')} | {best('a100-3b')} | "
                       f"{opt_delta('gh200-1b', tag, n)} | {opt_delta('a100-3b', tag, n)} |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
