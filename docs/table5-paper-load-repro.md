# UniCache Table 5 at the paper's load (×1)

Status (2026-11-27): done. A100-3B, ×1, one run per policy (OPT refined to convergence). GH200-1B runs exist but are left out here.
Earlier runs at other loads (0.8 k* and k*) are in `docs/qttft-headroom-and-agents.md`, Experiment A2.

## Question

When the three UniCache Table 5 traces are replayed as the repository defines them, with no load
scaling, does our simulator give the same QTTFT ordering of eviction policies as the paper? Is the spread
between policies large enough to optimise QTTFT with an agent?

## Experimental design

| Item | Choice |
|---|---|
| Traces | `TaskData/Hybrid/test1–3`, configs unchanged, load ×1. ToolBench dropped, because its source needs UCSD access |
| Policies | LRU, LFU, FIFO, LeCaR, ARC, WA, UniCache (`SimpleHYBRID`), OPT (Bélády, refined to a fixed point with `refine_opt.py`), plus NONE (no caching) as the floor |
| Model and timing | **A100-3B**: Llama-3.2-3B, aiconfigurator `a100_sxm` (uncalibrated), 8-token blocks, 66,355 blocks (0.9 × 63 GiB, the paper's setup). GH200-1B runs (our calibration) are left out of this writeup for now |
| Repeats | One run per policy × trace. OPT takes several refinement passes until its hit ratio changes by less than 0.1 pp |
| Metric | Mean QTTFT (arrival to first token, including queueing). ×LRU = QTTFT / LRU's QTTFT. Rank 1 = fastest |
| Reproduced if | Per trace: Spearman ρ ≥ 0.8 against the paper and ≥ 6/7 policies on the same side of LRU |
| Headroom | (QTTFT_best online − QTTFT_OPT) / QTTFT_best online |
| Code | `scripts/paper_repro/a2_grid.sh <cfg> Trace#N 1 x1`, tables by `scripts/paper_repro/a2_fill.py` |
| Outputs | `result/paper-repro/a2/a100-3b/x1/table5-T<n>.{csv,md}` |
| Cost | Laptop CPU only, ~1.5 h, no GPU |

### The traces

Each trace is a composite. Every task arrives as its own Poisson stream at 1 session/s. Follow-up turns
in multi-turn tasks arrive when the previous turn finishes, plus a lognormal think time. All values below
are measured at ×1.

| Trace | Single-turn tasks (1,800 sessions each, over ~30 min) | Multi-turn tasks | Requests | Multi-turn share, ours (paper) | Prompt tokens | Arrival span |
|---|---|---|---:|---:|---:|---:|
| T#1 | APPS, CodeParrot, MASH-QA (ToolBench dropped) | ALFRED: 500 sessions, ~40 turns each, ~6 s median think time. ShareGPT: 500 sessions, ~3 turns each, ~63 s median think time | 27,049 | 80 % (~80 %) | 52.5 M | 43 min |
| T#2 | same | ALFRED: 100 sessions, ~42 turns each. ShareGPT: 500 sessions | 11,188 | 52 % (~50 %) | 15.0 M | 40 min |
| T#3 | same | ShareGPT: 500 sessions | 6,965 | 22 % (~20 %) | 4.5 M | 48 min |

The traces differ only in how much ALFRED they carry: 500, 100 or 0 sessions.

All multi-turn sessions start within the first ~8 min, so the start of each trace is its busiest part.

### Known deviations from the paper

- **ToolBench is missing from all three traces.** It is cache-friendly: every session shares a long system
  prompt. Without it the traces carry less load and less prefix sharing.
- **The paper measured on real A100s;** these are simulator numbers. The A100-3B timing is uncalibrated.
- At ×1 in earlier LRU-only runs, T#2 and T#3 barely queue (LRU ≈ 0.018 s against the paper's 0.28 /
  0.13 s), so the policy spread there may be small.

## Results

### QTTFT per policy at ×1

**T#1** (paper LRU = 4.94 s)

| Policy | Paper QTTFT (s) | Paper ×LRU (rank) | A100-3B QTTFT (s) | A100-3B ×LRU (rank) |
|---|---:|---:|---:|---:|
| OPT | 3.84 | 0.78 (1) | 1.48 | 0.14 (1) |
| UniCache | 3.87 | 0.78 (2) | 9.18 | 0.86 (3) |
| ARC | 4.54 | 0.92 (3) | 10.68 | 1.00 (5) |
| LRU | 4.94 | 1.00 (4) | 10.67 | 1.00 (4) |
| WA | 5.14 | 1.04 (5) | 12.31 | 1.15 (8) |
| FIFO | 6.87 | 1.39 (6) | 11.06 | 1.04 (6) |
| LeCaR | 7.42 | 1.50 (7) | 6.81 | 0.64 (2) |
| LFU | 8.47 | 1.71 (8) | 12.08 | 1.13 (7) |
| NONE (no caching) | — | — | 17.75 | 1.66 |
| Spearman ρ vs paper / same side of LRU | | | | 0.52 / 5 of 7 |

**T#2** (paper LRU = 0.28 s)

| Policy | Paper QTTFT (s) | Paper ×LRU (rank) | A100-3B QTTFT (s) | A100-3B ×LRU (rank) |
|---|---:|---:|---:|---:|
| OPT | 0.21 | 0.75 (1) | 0.014 | 0.81 (1) |
| UniCache | 0.24 | 0.86 (2) | 0.016 | 0.91 (2) |
| ARC | 0.26 | 0.93 (3) | 0.017 | 1.00 (3) |
| LRU | 0.28 | 1.00 (4) | 0.018 | 1.00 (4) |
| WA | 0.29 | 1.04 (5) | 0.019 | 1.09 (5) |
| FIFO | 0.36 | 1.29 (6) | 0.021 | 1.18 (6) |
| LeCaR | 0.47 | 1.68 (7) | 0.024 | 1.38 (7) |
| LFU | 0.53 | 1.89 (8) | 0.060 | 3.44 (8) |
| NONE (no caching) | — | — | 0.867 | 49.40 |
| Spearman ρ vs paper / same side of LRU | | | | 1.00 / 7 of 7 |

**T#3** (paper LRU = 0.13 s)

| Policy | Paper QTTFT (s) | Paper ×LRU (rank) | A100-3B QTTFT (s) | A100-3B ×LRU (rank) |
|---|---:|---:|---:|---:|
| OPT | 0.08 | 0.62 (1) | 0.014 | 0.78 (1) |
| UniCache | 0.10 | 0.77 (2) | 0.016 | 0.87 (2) |
| ARC | 0.12 | 0.92 (3) | 0.018 | 1.00 (3) |
| LRU | 0.13 | 1.00 (4) | 0.018 | 1.00 (4) |
| WA | 0.27 | 2.08 (6) | 0.018 | 1.01 (5) |
| FIFO | 0.25 | 1.92 (5) | 0.018 | 1.02 (7) |
| LeCaR | 0.35 | 2.69 (7) | 0.018 | 1.01 (6) |
| LFU | 0.37 | 2.85 (8) | 0.019 | 1.05 (8) |
| NONE (no caching) | — | — | 0.025 | 1.37 |
| Spearman ρ vs paper / same side of LRU | | | | 0.93 / 7 of 7 |

### Summary

| Trace | ρ / same side of LRU: A100-3B | Headroom vs best online: paper | A100-3B | OPT converged? |
|---|---|---:|---:|---|
| T#1 | 0.52 / 5 of 7 | 1 % | 78 % (best online: LeCaR) | yes (8 passes, last +0.09 pp) |
| T#2 | 1.00 / 7 of 7 | 12 % | 10 % (best online: UniCache) | yes (3 passes, last +0.00 pp) |
| T#3 | 0.93 / 7 of 7 | 20 % | 11 % (best online: UniCache) | yes (3 passes, last +0.00 pp) |

### Takeaways

- **T#2 and T#3 reproduce the paper's ordering at ×1:** ρ 1.00 and 0.93, and 7/7 policies on the same
  side of LRU. They do not reproduce its absolute QTTFT. We are 10–15× lower (14–60 ms against
  0.08–0.53 s) because at ×1 our traces barely queue.
- **T#1 does not reproduce:** ρ 0.52, 5/7. At ×1 it is past the queueing knee in our simulator: LRU is
  10.7 s, 2.2× the paper. LeCaR comes 2nd here (7th in the paper), and OPT drops QTTFT to 0.14× LRU
  (paper 0.78×). The likely causes are the missing ToolBench and the queueing cliff; we have not separated them.
- **Headroom over the best online policy:** T#2 10 % and T#3 11 % (paper 12 % and 20 %). In absolute terms
  that is only about 2 ms per request. T#1 is 78 % (LeCaR 6.8 s → OPT 1.5 s).
- **T#3's six middle policies all sit at 0.018 s** (within 1 ms of each other), so their ranks are not meaningful.
- **For agent QTTFT optimisation:** T#1 at ×1 has seconds of spread. T#2 and T#3 have a reliable ordering
  but only milliseconds of spread at ×1; at their knee (see `docs/qttft-headroom-and-agents.md`) the gap
  is hundreds of ms.
