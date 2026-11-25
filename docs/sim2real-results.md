# Sim-to-real results: do simulated eviction-policy results hold on a GPU?

Status, 2026-11-25: **simulator side done; GPU side not run yet.** GPU
columns show "—" until the GH200 runs finish. The simulator numbers below were
committed before any GPU run (`result/sim2real-transfer/preregistration-b16/`).
**First GPU job: ALFRED only**; CodeParrot and ShareGPT follow if it works.
**GPU route:** DeltaAI's GH200 queue currently estimates a start about a day
away, so the first run may instead go to a teammate's **A100** (x86). There,
the fork's 8-token-block path works, so that run uses H1's 8-token blocks and
the 8-token pre-registration (`preregistration/`: ALFRED at 1,024 blocks, LRU
33.54, UniCache 33.54, LeCaR 35.45, Agent 36.15). A GH200 run uses the 16-token
numbers below. The results table will say which run each GPU number comes
from.
Full methodology: [sim2real-transfer-methodology.md](sim2real-transfer-methodology.md).

## 1. Question

When the UniCache simulator says one eviction policy has a higher hit ratio
than another, does a real GPU agree, and by about the same amount?

## 2. Experimental design

**Traces.** Three of the H1 workloads, each cut from its **eval split**: the
first N sessions by arrival time, with the recorded arrival and think times
kept. The subsets are short because a GPU replays in real time: a 1-hour
trace costs 1 GPU-hour.

| Trace | Type | Subset | Requests | Prompt tokens | Replay length |
|---|---|---|---:|---:|---:|
| AgentBank ALFRED | multi-turn agentic | first 60 sessions, ≤ 12 turns each | 713 | 461k | 5.4 min |
| CodeParrot | single-turn code | first 600 sessions | 600 | 311k | 9.6 min |
| ShareGPT | multi-turn chat | first 100 sessions, ≤ 2 turns each | 159 | 53k | 9.4 min |

**Arrival times (our design choice).** The datasets have no timestamps, so
we generate the arrivals:
* *First turns:* sessions are shuffled with a fixed seed and arrive as a
  **Poisson process at 1 session per second**. The times are drawn once and
  frozen into each split's yaml (`frozen_arrivals`, written by
  `policy_agent/runner.py`), so every run uses identical start times.
* *Follow-up turns* (multi-turn traces only): each arrives when the previous
  turn **finishes**, plus a **lognormal think time** seeded per session. The
  mean is ≈ 11 s for ALFRED and ≈ 100 s for ShareGPT.

The subsets keep these times unchanged, and the GPU releases requests on the
same schedule in real time. "Real" here means real execution on a real GPU
under our arrival schedule.

Not used: MASH-QA, because every policy is within ~1 pp of every other at
every size we could afford, so there is no decision to test. AgentBank APPS
is also out: OPT − LRU < 0.5 pp in H1.

**Cache size.** For each trace, chosen in the simulator as the size with the
most OPT − LRU headroom, subject to:
* OPT − LRU ≥ 5 pp;
* the cache evicts, and evicted content is requested again;
* the heuristics differ by ≥ 2 pp;
* no preemptions.

The engine enforces the block budget, so the GPU cache is exactly this size.
A block is **16 tokens** (512 KiB for Llama-3.2-1B) on both sides. H1 used 8,
but this fork runs 8-token blocks on CUDA only through xformers, which is
x86-only and does not run on the GH200's Arm host. The chosen capacities in
tokens are the same as at 8-token blocks.

| Trace | Cache size | OPT − LRU (sim) | Evictions (sim, LRU) | Evicted, then requested again | Preemptions |
|---|---:|---:|---:|---:|---:|
| ALFRED | 512 blocks (256 MiB, 8k tokens) | 26.3 pp | 19,158 | 15,304 | 0 |
| CodeParrot | 4,096 blocks (2 GiB, 64k tokens) | 15.1 pp | 18,611 | 2,943 | 0 |
| ShareGPT | 1,024 blocks (512 MiB, 16k tokens) | 30.9 pp | 4,790 | 1,152 | 0 |

**Policies.**
* LRU, run 3 times on the GPU (the repeats measure GPU noise);
* UniCache;
* LeCaR;
* Agent: the trace's best agent policy from report Table 2, the one with the
  largest eval gain over LRU. These are CodeParrot 8 GiB, ALFRED 8 GiB and
  ShareGPT 4 GiB, run here at the subset's cache size.

OPT runs only in the simulator, as the upper bound.

**Simulator vs real.**

| | Simulator | Real |
|---|---|---|
| Engine | UniCache vLLM V0 fork, CPU, virtual clock | the same fork on CUDA, wall clock |
| Timing | H1's formula: prefill power law, 10 ms per decode step | real GPU execution |
| Hardware | laptop | 1× NVIDIA GH200 (96 GB HBM3), NCSA DeltaAI |
| Model | Llama-3.2-1B-Instruct, outputs forced to the trace's recorded tokens (dummy weights on GPU) | same |
| Settings | block size 16, ≤ 32 sequences, chunked prefill off, prefix caching on, 0 CPU swap blocks | same |

**Metric.** Token-weighted hit ratio = cached prompt tokens ÷ all prompt
tokens. A policy's result **transfers** if its gap to LRU has the same sign on
sim and GPU, and the two gaps differ by ≤ max(0.5 pp, 2 × GPU LRU noise).

## 3. Results

### Hit ratio (%), simulator vs GPU

| Trace | Policy | Simulator | GPU | GPU − Sim |
|---|---|---:|---:|---:|
| ALFRED | LRU | 33.45 | — | — |
|  | UniCache | 33.45 | — | — |
|  | LeCaR | 35.34 | — | — |
|  | Agent | 35.25 | — | — |
|  | OPT (sim only) | 59.72 | n/a | n/a |
| CodeParrot | LRU | 16.05 | — | — |
|  | UniCache | 18.89 | — | — |
|  | LeCaR | 18.19 | — | — |
|  | Agent | 18.54 | — | — |
|  | OPT (sim only) | 31.18 | n/a | n/a |
| ShareGPT | LRU | 11.66 | — | — |
|  | UniCache | 11.66 | — | — |
|  | LeCaR | 16.89 | — | — |
|  | Agent | 14.31 | — | — |
|  | OPT (sim only) | 42.55 | n/a | n/a |

### Gap vs LRU (pp): does the decision transfer?

| Trace | Comparison | Simulator | GPU | Transfers? |
|---|---|---:|---:|---|
| ALFRED | UniCache − LRU | 0.00 | — | — |
|  | LeCaR − LRU | +1.89 | — | — |
|  | Agent − LRU | +1.80 | — | — |
|  | Agent − best heuristic (LeCaR) | −0.09 | — | — |
| CodeParrot | UniCache − LRU | +2.83 | — | — |
|  | LeCaR − LRU | +2.13 | — | — |
|  | Agent − LRU | +2.48 | — | — |
|  | Agent − best heuristic (UniCache) | −0.35 | — | — |
| ShareGPT | UniCache − LRU | 0.00 | — | — |
|  | LeCaR − LRU | +5.24 | — | — |
|  | Agent − LRU | +2.65 | — | — |
|  | Agent − best heuristic (LeCaR) | −2.59 | — | — |

GPU noise (spread of the 3 LRU runs): ALFRED —, CodeParrot —, ShareGPT —.

### Summary

*To be written once the GPU runs finish:* which gaps transfer, whether the
single-turn trace (CodeParrot) transfers better than the multi-turn ones
(ALFRED, ShareGPT) as predicted, and what that means for H1.
