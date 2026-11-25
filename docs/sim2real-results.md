# Sim-to-real results: do simulated eviction-policy results hold on a GPU?

Status, 2026-11-25: **simulator side done; GPU side not run yet.** GPU
columns show "—" until the GH200 runs finish. The simulator numbers below were
committed before any GPU run (`result/sim2real-transfer/preregistration/`).
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
A block is 8 tokens, or 256 KiB for Llama-3.2-1B.

| Trace | Cache size | OPT − LRU (sim) | Evictions (sim, LRU) | Evicted, then requested again | Preemptions |
|---|---:|---:|---:|---:|---:|
| ALFRED | 1,024 blocks (256 MiB, 8k tokens) | 26.5 pp | 38,631 | 30,898 | 0 |
| CodeParrot | 8,192 blocks (2 GiB, 66k tokens) | 15.3 pp | 37,500 | 5,942 | 0 |
| ShareGPT | 2,048 blocks (512 MiB, 16k tokens) | 30.9 pp | 9,559 | 2,293 | 0 |

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
| Settings | block size 8, ≤ 32 sequences, chunked prefill off, prefix caching on, 0 CPU swap blocks | same |

**Metric.** Token-weighted hit ratio = cached prompt tokens ÷ all prompt
tokens. A policy's result **transfers** if its gap to LRU has the same sign on
sim and GPU, and the two gaps differ by ≤ max(0.5 pp, 2 × GPU LRU noise).

## 3. Results

### Hit ratio (%), simulator vs GPU

| Trace | Policy | Simulator | GPU | GPU − Sim |
|---|---|---:|---:|---:|
| ALFRED | LRU | 33.54 | — | — |
| | UniCache | 33.54 | — | — |
| | LeCaR | 35.45 | — | — |
| | Agent | 36.15 | — | — |
| | OPT (sim only) | 60.02 | n/a | n/a |
| CodeParrot | LRU | 16.14 | — | — |
| | UniCache | 18.93 | — | — |
| | LeCaR | 17.76 | — | — |
| | Agent | 18.55 | — | — |
| | OPT (sim only) | 31.42 | n/a | n/a |
| ShareGPT | LRU | 13.09 | — | — |
| | UniCache | 13.09 | — | — |
| | LeCaR | 17.93 | — | — |
| | Agent | 16.23 | — | — |
| | OPT (sim only) | 43.97 | n/a | n/a |

### Gap vs LRU (pp): does the decision transfer?

| Trace | Comparison | Simulator | GPU | Transfers? |
|---|---|---:|---:|---|
| ALFRED | UniCache − LRU | 0.00 | — | — |
| | LeCaR − LRU | +1.91 | — | — |
| | Agent − LRU | +2.61 | — | — |
| | Agent − best heuristic (LeCaR) | +0.70 | — | — |
| CodeParrot | UniCache − LRU | +2.79 | — | — |
| | LeCaR − LRU | +1.62 | — | — |
| | Agent − LRU | +2.41 | — | — |
| | Agent − best heuristic (UniCache) | −0.38 | — | — |
| ShareGPT | UniCache − LRU | 0.00 | — | — |
| | LeCaR − LRU | +4.84 | — | — |
| | Agent − LRU | +3.13 | — | — |
| | Agent − best heuristic (LeCaR) | −1.71 | — | — |

GPU noise (spread of the 3 LRU runs): ALFRED —, CodeParrot —, ShareGPT —.

### Summary

*To be written once the GPU runs finish:* which gaps transfer, whether the
single-turn trace (CodeParrot) transfers better than the multi-turn ones
(ALFRED, ShareGPT) as predicted, and what that means for H1.
