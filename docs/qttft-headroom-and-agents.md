# QTTFT headroom and agent-written eviction policies

Status (2026-11-27): A1 and A2 run (results below). OPT reaches 23–31 % lower QTTFT than the best online policy on T#2/T#3 (A100-3B); T#1 is provisional. B not started. Two experiments: A finds where eviction policy matters
for QTTFT; B asks whether an agent can beat the best heuristic there. B uses only pairs that qualify in A.

Shared setup:

| Item | Choice |
|---|---|
| Modeled hardware and model | 1 × A100 SXM 80 GB, Llama-3.2-3B-Instruct (the UniCache paper's setup) |
| Simulator timing | aiconfigurator `a100_sxm` tables; recalibrated with the sweep-only method once A100 runs are available |
| Real GPU | A100 (borrowed access) |
| Block size | 8 tokens (as in H1 and the paper) |
| QTTFT | arrival to first token, including queueing (`ttft_s` in `latency_requests.csv`) |
| Splits | 50/50 by session start (H1 splits; same rule for the paper mixes). Selection on tune; eval scored once |
| Policies | LRU, LFU, FIFO, LeCaR, ARC, WA, UniCache; OPT (refined with `refine_opt.py`) as the bound |

---

## Experiment A: is there headroom? (best-effort reproduction of two paper results)

### Question

Does eviction policy produce a real spread in (1) KV hit ratio and (2) QTTFT in our setup, as in the
UniCache paper's Fig. 1 / Table 1 and Table 5? If it does, the gap between the best heuristic and OPT
is the headroom an agent could exploit.

### Experimental design

Kept deliberately simple: one knob per result is set from LRU alone; every other policy's number is a
prediction. Where we deviate from the paper, the row says so.

| Item | A1: hit ratio (paper Fig. 1, Table 1) | A2: QTTFT (paper Table 5) |
|---|---|---|
| Paper setup | Llama-3.2-1B on A100; MASH-QA, ShareGPT, Mixed (ShareGPT + MASH-QA); LRU, LFU, FIFO, WA | A100; Trace #1–#3 (mixed multi-/single-turn); 8 policies |
| Our workloads | `mashqa`, `sharegpt` session data; Mixed = both tasks in one replay (new config) | `TaskData/Hybrid/test1–3` (ToolBench missing; mapping to Trace #1–#3 is our inference) |
| Model | Llama-3.2-1B-Instruct | Llama-3.2-1B (our GH200 calibration) and Llama-3.2-3B (paper-like A100 pricing) |
| Timing | Not needed for hit ratio (legacy timing) | (i) GH200 v8 calibration, our hardware; (ii) aiconfigurator `a100_sxm`, uncalibrated |
| Block size | 8 tokens | 8 tokens (A100 pricing); 16 (GH200 v8) |
| Knob set from LRU | KV budget: the paper does not give it, so pick the one budget where LRU on MASH-QA alone is closest to the paper's ≈ 0.63, then use it for all rows | Load: per trace, find LRU's knee k* (smallest arrival scale with LRU QTTFT > 3× its lowest-load value), run at 0.8 k* (pre-registered) and also at k*, because the knee is a cliff (see A2). Session starts and think times both scaled by k |
| Policies | Paper's 4 + OPT (refined) for headroom | Paper's 8 (OPT refined) + no caching (NONE) as the floor |
| Headroom | OPT − best of the 4 heuristics (pp) | (QTTFT_best online − QTTFT_OPT) / QTTFT_best online |
| Reproduced if | Same ordering of the 4 policies per row; each within ±0.05 of the paper | Per trace: Spearman ρ ≥ 0.8 vs paper and ≥ 6/7 policies on the same side of LRU (ratios, not absolute seconds) |
| Real-GPU check (optional) | none needed (hit ratio transfers within 0.2 pp in our CodeParrot sim2real) | one 15-min GH200 job: NONE ×2, LRU ×2 on a short T#1-like slice (checks v8's eviction cost) |
| Cost | ~12 sim runs + 3-point budget sweep, < 1 h laptop | knee sweeps + 3 traces × 9 policies × 2 timings, ~2–3 h laptop |

Paper hardware is an A100; ours is a GH200 (simulator calibrated on it). Fig. 1 values for single tasks
are read off the bar chart (≈ ±0.01); Table 1 and Table 5 values are exact.

### Results

#### A1. Hit ratio (paper Fig. 1 and Table 1)

KV budget: **125,000 blocks × 8 tokens (1.0M tokens)**, chosen so LRU on MASH-QA alone = 0.6305 (paper ≈ 0.63).
Sweep, LRU on MASH-QA: 50k 0.319, 100k 0.546, 125k 0.631, 140k 0.677, 200k 0.756, 400k 0.756.
Legacy (hit-ratio) timing, Llama-3.2-1B tokenizer. Runs: `result/paper-repro/a1/`, OPT in `result/paper-repro/a1-opt/`.

| Workload | Policy | Paper hit ratio | Our hit ratio | Δ (ours − paper) |
|---|---|---:|---:|---:|
| MASH-QA | LRU | ≈ 0.63 | 0.6305 | +0.00 |
| MASH-QA | LFU | ≈ 0.69 | 0.6997 | +0.01 |
| MASH-QA | FIFO | ≈ 0.58 | 0.5996 | +0.02 |
| MASH-QA | WA | ≈ 0.63 | 0.6305 | +0.00 |
| MASH-QA | OPT | — | 0.7562 | — |
| ShareGPT | LRU | ≈ 0.80 | 0.8284 | +0.03 |
| ShareGPT | LFU | ≈ 0.60 | 0.8121 | +0.21 |
| ShareGPT | FIFO | ≈ 0.79 | 0.8280 | +0.04 |
| ShareGPT | WA | ≈ 0.80 | 0.8284 | +0.03 |
| ShareGPT | OPT | — | 0.8288 | — |
| Mixed, total | LRU | 0.5709 | 0.6287 | +0.06 |
| Mixed, total | LFU | 0.3732 | 0.4805 | +0.11 |
| Mixed, total | FIFO | 0.5469 | 0.6073 | +0.06 |
| Mixed, total | WA | 0.5827 | 0.6592 | +0.08 |
| Mixed, total | OPT | — | 0.7774 | — |
| Mixed, MASH-QA part | LRU | 0.4247 | 0.5508 | +0.13 |
| Mixed, MASH-QA part | LFU | 0.3627 | 0.4294 | +0.07 |
| Mixed, MASH-QA part | FIFO | 0.4092 | 0.5266 | +0.12 |
| Mixed, MASH-QA part | WA | 0.5083 | 0.6031 | +0.09 |
| Mixed, MASH-QA part | OPT | — | 0.7562 | — |
| Mixed, ShareGPT part | LRU | 0.7457 | 0.8179 | +0.07 |
| Mixed, ShareGPT part | LFU | 0.3858 | 0.6045 | +0.22 |
| Mixed, ShareGPT part | FIFO | 0.7113 | 0.8032 | +0.09 |
| Mixed, ShareGPT part | WA | 0.6717 | 0.7954 | +0.12 |
| Mixed, ShareGPT part | OPT | — | 0.8288 | — |

| Workload | Best heuristic (ours) | Headroom: OPT − best heuristic (pp) | Ordering matches paper? |
|---|---|---:|---|
| MASH-QA | LFU | 5.7 | Yes |
| ShareGPT | LRU = WA | 0.0 | Yes (LFU gap far smaller: 0.02 vs 0.20) |
| Mixed, total | WA | 11.8 | Yes |
| Mixed, MASH-QA part | WA | 15.3 | Yes |
| Mixed, ShareGPT part | LRU | 1.1 | Yes |

Reading:
- **Ordering reproduced in all 5 rows.** Every policy falls in the same rank order as the paper.
- **Magnitudes:** single-task MASH-QA within 0.02 (by construction for LRU). Mixed rows are 0.06–0.13 higher
  than the paper. LFU on ShareGPT is the one large miss: +0.21 alone and +0.22 in the mix (LFU still ranks last).
- **Headroom is largest on the mixed workload:** 11.8 pp over the best heuristic (WA) in total, 15.3 pp on
  its MASH-QA part. ShareGPT alone fits in this cache (OPT − LRU = 0.04 pp), so it has no headroom here.
- OPT reaches the compulsory-miss ceiling everywhere (MASH-QA 0.756 = LRU with 200k+ blocks), so
  OPT here is "every reusable prefix hit", the loosest possible bound.

#### A2. QTTFT (paper Table 5)

LRU knee sweep (LRU QTTFT in seconds; `result/paper-repro/a2/<cfg>/knee/`):

| Timing | Trace | ×0.1 | ×0.25 | ×0.5 | ×1 | ×2 | ×3 | ×4 | ×6 | k* | 0.8 k* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GH200-1B (v8) | T#1 | | | | 0.025 | 5.98 | 11.27 | 14.55 | 21.42 | 2 | 1.6 |
| GH200-1B (v8) | T#2 | | | | 0.017 | 0.019 | 0.717 | 4.51 | 22.96 | 3 | 2.4 |
| GH200-1B (v8) | T#3 | | | | 0.015 | 0.016 | 0.027 | 0.534 | 21.13 | 4 | 3.2 |
| A100-3B (aic) | T#1 | 0.018 | 0.022 | 0.027 | 10.67 | 27.17 | | | | 1 | 0.8 |
| A100-3B (aic) | T#2 | 0.014 | 0.015 | 0.016 | 0.018 | 3.45 | | | | 2 | 1.6 |
| A100-3B (aic) | T#3 | 0.015 | 0.016 | 0.017 | 0.018 | 0.745 | | | | 2 | 1.6 |

- Below the knee QTTFT is ~15–30 ms (no queueing); one load step later it is seconds. There is no
  gentle middle, so 0.8 k* may sit in the no-queueing regime where policies barely differ, and k* sits
  in the queueing regime the paper's T#1 (4.94 s) suggests. We report both.
- The paper's ×1 for T#1 already queues on A100-3B (10.7 s vs paper 4.94 s) but not on GH200-1B
  (0.025 s): 3B on an A100 is ~5× slower per token than 1B on a GH200.

Main grid loads (0.8 k* / k*): GH200-1B T#1 ×1.6/×2, T#2 ×2.4/×3, T#3 ×3.2/×4; A100-3B T#1 ×0.8/×1,
T#2 ×1.6/×2, T#3 ×1.6/×2. The table below is at 0.8 k*; the k* table follows it. Generated by `scripts/paper_repro/a2_fill.py`
from `result/paper-repro/a2/<cfg>/<08kstar|kstar>/table5-T<n>.csv`. OPT is refined to a fixed point
(`refine_opt.py`); "OPT last-step Δhit" is its hit-ratio change in the final iteration (0.00 = converged).

**Findings (2026-11-27)**

- **Headroom reproduces on T#2 and T#3 under A100-3B pricing.** Both loads pass the bar (ρ ≥ 0.8,
  ≥ 6/7 on the same side of LRU; ρ 0.83–0.98). OPT cuts QTTFT 33–41 % vs LRU (paper 25 % / 38 %) and
  23–31 % vs the best online policy, UniCache in every case (paper 12 % / 20 %). That gap is the
  opportunity for an agent.
- **GH200-1B (our calibration) shows less headroom.** OPT vs best online is 10–13 % on T#2/T#3. T#2
  passes at both loads (ρ 0.98–1.00, 7/7); T#3 passes only at 0.8 k*.
- **Caveat: on GH200-1B, no caching (NONE) beats LRU on T#1 and T#3**, and beats OPT on T#3
  (0.030 s vs 0.036 s at 0.8 k*). v8 charges 0.057 ms per evicted block, and with a 1B model a miss is
  cheap, so eviction cost outweighs the prefill it saves. This is the advisor's warning; it is
  unvalidated on the GPU. On A100-3B, NONE is 1.7–28× worse than LRU, as expected.
- **T#1 does not reproduce, and its OPT has not converged.** After 5 iterations its hit ratio is still
  climbing 1.5–5 pp per step, so T#1 OPT rows and headroom are provisional. T#1 also sits on the
  queueing cliff at 0.8 k*: on A100-3B LRU is 0.44 s and UniCache 0.06 s, a 7× gap from a 6 pp
  hit-ratio difference.
- OPT here is Bélády on hit ratio, not QTTFT-optimal, so "headroom" indicates the opportunity; it is
  not a strict bound in either direction.

##### At 0.8 k*: loads GH200-1B T#1–3 ×1.6/×2.4/×3.2; A100-3B ×0.8/×1.6/×1.6

| Trace | Policy | Paper QTTFT (s) | Paper ×LRU | Ours, GH200-1B (s) | Ours ×LRU, GH200-1B | Ours, A100-3B (s) | Ours ×LRU, A100-3B | Hit ratio GH200-1B | Hit ratio A100-3B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T#1 | LRU | 4.94 | 1.00 | 2.608 | 1.00 | 0.441 | 1.00 | 0.614 | 0.760 |
| T#1 | LFU | 8.47 | 1.71 | 3.286 | 1.26 | 7.387 | 16.75 | 0.304 | 0.302 |
| T#1 | FIFO | 6.87 | 1.39 | 2.288 | 0.88 | 2.404 | 5.45 | 0.618 | 0.646 |
| T#1 | LeCaR | 7.42 | 1.50 | 2.060 | 0.79 | 1.525 | 3.46 | 0.659 | 0.693 |
| T#1 | ARC | 4.54 | 0.92 | 2.596 | 1.00 | 0.551 | 1.25 | 0.615 | 0.754 |
| T#1 | WA | 5.14 | 1.04 | 3.320 | 1.27 | 6.335 | 14.36 | 0.498 | 0.513 |
| T#1 | OPT (not converged) | 3.84 | 0.78 | 1.454 | 0.56 | 0.052 | 0.12 | 0.622 | 0.708 |
| T#1 | UniCache | 3.87 | 0.78 | 1.796 | 0.69 | 0.060 | 0.14 | 0.732 | 0.821 |
| T#1 | NONE (no caching) | — | — | 1.875 | 0.72 | 12.208 | 27.68 | 0.000 | 0.000 |
| T#2 | LRU | 0.28 | 1.00 | 0.072 | 1.00 | 0.418 | 1.00 | 0.777 | 0.778 |
| T#2 | LFU | 0.53 | 1.89 | 0.199 | 2.75 | 3.307 | 7.91 | 0.512 | 0.494 |
| T#2 | FIFO | 0.36 | 1.29 | 0.088 | 1.22 | 0.785 | 1.88 | 0.727 | 0.719 |
| T#2 | LeCaR | 0.47 | 1.68 | 0.090 | 1.25 | 0.954 | 2.28 | 0.684 | 0.678 |
| T#2 | ARC | 0.26 | 0.93 | 0.070 | 0.97 | 0.420 | 1.00 | 0.777 | 0.778 |
| T#2 | WA | 0.29 | 1.04 | 0.083 | 1.14 | 0.634 | 1.52 | 0.750 | 0.747 |
| T#2 | OPT | 0.21 | 0.75 | 0.058 | 0.81 | 0.247 | 0.59 | 0.865 | 0.868 |
| T#2 | UniCache | 0.24 | 0.86 | 0.065 | 0.90 | 0.358 | 0.86 | 0.819 | 0.821 |
| T#2 | NONE (no caching) | — | — | 0.143 | 1.98 | 7.743 | 18.53 | 0.000 | 0.000 |
| T#3 | LRU | 0.13 | 1.00 | 0.044 | 1.00 | 0.080 | 1.00 | 0.414 | 0.418 |
| T#3 | LFU | 0.37 | 2.85 | 0.048 | 1.09 | 0.097 | 1.22 | 0.352 | 0.357 |
| T#3 | FIFO | 0.25 | 1.92 | 0.046 | 1.03 | 0.084 | 1.06 | 0.390 | 0.396 |
| T#3 | LeCaR | 0.35 | 2.69 | 0.046 | 1.03 | 0.081 | 1.02 | 0.404 | 0.409 |
| T#3 | ARC | 0.12 | 0.92 | 0.046 | 1.03 | 0.080 | 1.00 | 0.415 | 0.419 |
| T#3 | WA | 0.27 | 2.08 | 0.046 | 1.03 | 0.081 | 1.02 | 0.409 | 0.412 |
| T#3 | OPT | 0.08 | 0.62 | 0.036 | 0.82 | 0.050 | 0.62 | 0.661 | 0.668 |
| T#3 | UniCache | 0.10 | 0.77 | 0.041 | 0.93 | 0.068 | 0.85 | 0.556 | 0.569 |
| T#3 | NONE (no caching) | — | — | 0.030 | 0.69 | 0.192 | 2.41 | 0.000 | 0.000 |

| Trace | ρ vs paper / same side of LRU: GH200-1B | A100-3B | Headroom vs LRU: paper | GH200-1B | A100-3B | Headroom vs best online: paper | GH200-1B | A100-3B | OPT last-step Δhit (pp): GH200-1B | A100-3B |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T#1 | 0.55 / 5/7 | 0.88 / 6/7 | 22 % | 44 % | 88 % | 1 % | 19 % (UniCache) | 12 % (UniCache) | +1.49 | +3.55 |
| T#2 | 1.00 / 7/7 | 0.98 / 6/7 | 25 % | 19 % | 41 % | 12 % | 10 % (UniCache) | 31 % (UniCache) | +0.00 | +0.00 |
| T#3 | 0.95 / 6/7 | 0.90 / 6/7 | 38 % | 18 % | 38 % | 20 % | 11 % (UniCache) | 27 % (UniCache) | +0.00 | +0.00 |

##### At k*: loads GH200-1B T#1–3 ×2/×3/×4; A100-3B ×1/×2/×2

| Trace | Policy | Paper QTTFT (s) | Paper ×LRU | Ours, GH200-1B (s) | Ours ×LRU, GH200-1B | Ours, A100-3B (s) | Ours ×LRU, A100-3B | Hit ratio GH200-1B | Hit ratio A100-3B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T#1 | LRU | 4.94 | 1.00 | 5.981 | 1.00 | 10.672 | 1.00 | 0.465 | 0.510 |
| T#1 | LFU | 8.47 | 1.71 | 5.820 | 0.97 | 12.078 | 1.13 | 0.288 | 0.289 |
| T#1 | FIFO | 6.87 | 1.39 | 6.049 | 1.01 | 11.062 | 1.04 | 0.440 | 0.478 |
| T#1 | LeCaR | 7.42 | 1.50 | 4.891 | 0.82 | 6.811 | 0.64 | 0.588 | 0.627 |
| T#1 | ARC | 4.54 | 0.92 | 6.010 | 1.00 | 10.678 | 1.00 | 0.464 | 0.509 |
| T#1 | WA | 5.14 | 1.04 | 6.388 | 1.07 | 12.306 | 1.15 | 0.404 | 0.441 |
| T#1 | OPT (not converged) | 3.84 | 0.78 | 4.058 | 0.68 | 3.788 | 0.35 | 0.490 | 0.655 |
| T#1 | UniCache | 3.87 | 0.78 | 5.644 | 0.94 | 9.177 | 0.86 | 0.509 | 0.552 |
| T#1 | NONE (no caching) | — | — | 4.053 | 0.68 | 17.755 | 1.66 | 0.000 | 0.000 |
| T#2 | LRU | 0.28 | 1.00 | 0.717 | 1.00 | 3.451 | 1.00 | 0.774 | 0.766 |
| T#2 | LFU | 0.53 | 1.89 | 1.310 | 1.83 | 8.051 | 2.33 | 0.502 | 0.495 |
| T#2 | FIFO | 0.36 | 1.29 | 0.895 | 1.25 | 5.798 | 1.68 | 0.710 | 0.681 |
| T#2 | LeCaR | 0.47 | 1.68 | 0.936 | 1.30 | 6.076 | 1.76 | 0.667 | 0.643 |
| T#2 | ARC | 0.26 | 0.93 | 0.705 | 0.98 | 3.457 | 1.00 | 0.774 | 0.766 |
| T#2 | WA | 0.29 | 1.04 | 0.902 | 1.26 | 8.538 | 2.47 | 0.728 | 0.601 |
| T#2 | OPT | 0.21 | 0.75 | 0.602 | 0.84 | 2.155 | 0.62 | 0.863 | 0.864 |
| T#2 | UniCache | 0.24 | 0.86 | 0.688 | 0.96 | 3.079 | 0.89 | 0.812 | 0.804 |
| T#2 | NONE (no caching) | — | — | 1.055 | 1.47 | 15.579 | 4.51 | 0.000 | 0.000 |
| T#3 | LRU | 0.13 | 1.00 | 0.534 | 1.00 | 0.745 | 1.00 | 0.410 | 0.415 |
| T#3 | LFU | 0.37 | 2.85 | 0.545 | 1.02 | 0.845 | 1.13 | 0.351 | 0.359 |
| T#3 | FIFO | 0.25 | 1.92 | 0.546 | 1.02 | 0.781 | 1.05 | 0.387 | 0.392 |
| T#3 | LeCaR | 0.35 | 2.69 | 0.522 | 0.98 | 0.750 | 1.01 | 0.399 | 0.406 |
| T#3 | ARC | 0.12 | 0.92 | 0.528 | 0.99 | 0.750 | 1.01 | 0.410 | 0.416 |
| T#3 | WA | 0.27 | 2.08 | 0.523 | 0.98 | 0.759 | 1.02 | 0.402 | 0.409 |
| T#3 | OPT | 0.08 | 0.62 | 0.417 | 0.78 | 0.498 | 0.67 | 0.662 | 0.668 |
| T#3 | UniCache | 0.10 | 0.77 | 0.477 | 0.89 | 0.646 | 0.87 | 0.552 | 0.563 |
| T#3 | NONE (no caching) | — | — | 0.311 | 0.58 | 1.729 | 2.32 | 0.000 | 0.000 |

| Trace | ρ vs paper / same side of LRU: GH200-1B | A100-3B | Headroom vs LRU: paper | GH200-1B | A100-3B | Headroom vs best online: paper | GH200-1B | A100-3B | OPT last-step Δhit (pp): GH200-1B | A100-3B |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T#1 | 0.26 / 4/7 | 0.52 / 5/7 | 22 % | 32 % | 65 % | 1 % | 17 % (LeCaR) | 44 % (LeCaR) | +3.32 | +5.33 |
| T#2 | 0.98 / 7/7 | 0.83 / 6/7 | 25 % | 16 % | 38 % | 12 % | 12 % (UniCache) | 30 % (UniCache) | +0.00 | +0.00 |
| T#3 | 0.55 / 5/7 | 0.88 / 6/7 | 38 % | 22 % | 33 % | 20 % | 13 % (UniCache) | 23 % (UniCache) | +0.00 | +0.00 |


Optional GH200 check (NONE ×2, LRU ×2): sim NONE/LRU ___, GPU ___, noise ___.

---

## Experiment B: agent policy development

### Question

On the pairs that qualify in A, can a coding agent write an eviction policy whose eval QTTFT is lower
than the best existing heuristic's, in the simulator and on a real A100?

### Experimental design

| Item | Choice |
|---|---|
| Pairs | The qualifying rows of A (at most 3, highest headroom first); T#2 ×m if it qualifies, since it has a paper reference |
| Agent | `claude:opus`, ≤ 30 submissions, start policy LRU |
| Objective | Mean tune QTTFT (simulator) |
| Policy execution | Loaded in-process for scoring (no per-call IPC); the whole replay runs offline under the sandbox. Removes the +45 % IPC overhead seen in step 3 |
| Information guards | Agent sees tune scores only; eval traces, eval scores and OPT never enter its workspace |
| Selection | Lowest tune QTTFT among submissions; eval scored once |
| Baselines | LRU and the best heuristic on tune (from A) |
| Paper reference | Paper's UniCache QTTFT relative to LRU (best online policy in the paper) |
| Real-GPU check | A100: LRU ×3, best heuristic ×2, agent ×2 on eval |
| Success | Agent below the best heuristic on eval by more than the noise floor, in the simulator and on the A100 |
| Cost | Per pair: agent run ~1–2 h on the laptop; A100 ~30–45 min |

### Results

QTTFT in ms; changes relative (negative = lower QTTFT).

| Pair (trace, KV blocks, load) | Paper: UniCache ×LRU | Best heuristic (tune) | LRU, tune | Best heuristic, tune | Agent, tune | LRU, eval | Best heuristic, eval | Agent, eval | Agent vs LRU, eval | Agent vs best heuristic, eval | Agent vs best heuristic, A100 eval | Beyond noise? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T#2, 66,355, ×m | 0.86 | | | | | | | | | | | |
| ___ | — | | | | | | | | | | | |
| ___ | — | | | | | | | | | | | |
