# QTTFT headroom and agent-written eviction policies

Status: design only; no runs yet (2026-11-27). Two experiments: A finds where eviction policy matters
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
| Knob set from LRU | KV budget: the paper does not give it, so pick the one budget where LRU on MASH-QA alone is closest to the paper's ≈ 0.63, then use it for all rows | Load: per trace, find LRU's knee k* (smallest arrival scale with LRU QTTFT > 3× its ×1 value), run at 0.8 k*. Session starts and think times both scaled by k |
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

Loads: T#1 ×___, T#2 ×___, T#3 ×___ (0.8 k*, GH200 v8); T#1 ×___, T#2 ×___, T#3 ×___ (0.8 k*, A100-3B).

| Trace | Policy | Paper QTTFT (s) | Paper ×LRU | Ours, GH200-1B (s) | Ours ×LRU, GH200-1B | Ours, A100-3B (s) | Ours ×LRU, A100-3B | Our hit ratio (GH200-1B) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T#1 | LRU | 4.94 | 1.00 | | | | | |
| T#1 | LFU | 8.47 | 1.71 | | | | | |
| T#1 | FIFO | 6.87 | 1.39 | | | | | |
| T#1 | LeCaR | 7.42 | 1.50 | | | | | |
| T#1 | ARC | 4.54 | 0.92 | | | | | |
| T#1 | WA | 5.14 | 1.04 | | | | | |
| T#1 | OPT | 3.84 | 0.78 | | | | | |
| T#1 | UniCache | 3.87 | 0.78 | | | | | |
| T#1 | NONE (no caching) | — | — | | | | | 0 |
| T#2 | LRU | 0.28 | 1.00 | | | | | |
| T#2 | LFU | 0.53 | 1.89 | | | | | |
| T#2 | FIFO | 0.36 | 1.29 | | | | | |
| T#2 | LeCaR | 0.47 | 1.68 | | | | | |
| T#2 | ARC | 0.26 | 0.93 | | | | | |
| T#2 | WA | 0.29 | 1.04 | | | | | |
| T#2 | OPT | 0.21 | 0.75 | | | | | |
| T#2 | UniCache | 0.24 | 0.86 | | | | | |
| T#2 | NONE (no caching) | — | — | | | | | 0 |
| T#3 | LRU | 0.13 | 1.00 | | | | | |
| T#3 | LFU | 0.37 | 2.85 | | | | | |
| T#3 | FIFO | 0.25 | 1.92 | | | | | |
| T#3 | LeCaR | 0.35 | 2.69 | | | | | |
| T#3 | ARC | 0.12 | 0.92 | | | | | |
| T#3 | WA | 0.27 | 2.08 | | | | | |
| T#3 | OPT | 0.08 | 0.62 | | | | | |
| T#3 | UniCache | 0.10 | 0.77 | | | | | |
| T#3 | NONE (no caching) | — | — | | | | | 0 |

Headroom = how much of the QTTFT perfect eviction (OPT) would remove: vs LRU, and vs the best online policy.

| Trace | ρ vs paper, same side of LRU: GH200-1B | ρ vs paper, same side of LRU: A100-3B | Headroom vs LRU: paper | GH200-1B | A100-3B | Headroom vs best online: paper | GH200-1B | A100-3B |
|---|---|---|---:|---:|---:|---:|---:|---:|
| T#1 | | | 22 % | | | 1 % | | |
| T#2 | | | 25 % | | | 12 % | | |
| T#3 | | | 38 % | | | 20 % | | |

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
