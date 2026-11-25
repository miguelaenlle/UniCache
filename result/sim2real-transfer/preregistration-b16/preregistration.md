# Sim2real pre-registration (simulator predictions before any GPU run)

Simulator: H1's UniCache simulator (legacy timing), eval-split subsets, 16-token blocks, 0 CPU blocks. Hit ratios in %, deltas in pp.

| Workload | Sessions | Requests | C* (blocks) | OPT−LRU | LRU | UniCache | LeCaR | Agent | OPT | Agent−LRU | Agent−Best | Best (tier 1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| alfred | 60 | 713 | 512 | 26.26 | 33.45 | 33.45 | 35.34 | 35.25 | 59.72 | +1.80 | -0.09 | LeCaR |
| codeparrot@600 | 600 | 600 | 4096 | 15.13 | 16.05 | 18.89 | 18.19 | 18.54 | 31.18 | +2.48 | -0.35 | UniCache |
| mashqa@600 | | | none | | | | | | | | | no capacity qualifies; enlarge the subset |
| sharegpt@100x2 | 100 | 159 | 1024 | 30.90 | 11.66 | 11.66 | 16.89 | 14.31 | 42.55 | +2.65 | -2.59 | LeCaR |

**This supersedes `../preregistration/` (8-token blocks).** Both the simulator
and the GPU now use 16-token blocks, because this fork runs 8-token blocks on
CUDA only through xformers, which is x86-only. This was committed before any
GPU run (2026-11-25). Capacities come from `sweep-b16/capacity_selection.csv`,
chosen by the policy-neutral rules in `scripts/sim2real/select_capacity.py`
without using the agent column. They are the same token capacities as the
8-token pre-registration: 8k, 64k and 16k tokens.

Predicted deltas (pp). A delta transfers if its sign matches and
|Δ_sim − Δ_GPU| ≤ max(0.5 pp, 2 × the LRU noise floor):

| Trace (16-token blocks) | UniCache − LRU | LeCaR − LRU | Agent − LRU | Agent − Best |
|---|---:|---:|---:|---:|
| ALFRED, 60 × ≤ 12 turns, 512 blocks | 0.00 | +1.89 | +1.80 | −0.09 (vs LeCaR) |
| CodeParrot, 600 sessions, 4,096 blocks | +2.83 | +2.13 | +2.48 | −0.35 (vs UniCache) |
| ShareGPT, 100 × ≤ 2 turns, 1,024 blocks | 0.00 | +5.24 | +2.65 | −2.59 (vs LeCaR) |

Hypotheses: CodeParrot (single-turn) transfers; ALFRED and ShareGPT
(multi-turn, with follow-ups that depend on completion times) may not. The
first GPU job runs ALFRED only. MASH-QA is still excluded (heuristic spread
< 1 pp at every capacity).
