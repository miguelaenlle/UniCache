# Sim2real pre-registration (simulator predictions before any GPU run)

Simulator: H1's UniCache simulator (legacy timing), eval-split subsets, 0 CPU blocks. Hit ratios in %, deltas in pp.

| Workload | Sessions | Requests | C* (blocks) | OPT−LRU | LRU | UniCache | LeCaR | Agent | OPT | Agent−LRU | Agent−Best | Best (tier 1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| alfred | 60 | 713 | 1024 | 26.47 | 33.54 | 33.54 | 35.45 | 36.15 | 60.02 | +2.61 | +0.70 | LeCaR |
| codeparrot@600 | 600 | 600 | 8192 | 15.28 | 16.14 | 18.93 | 17.76 | 18.55 | 31.42 | +2.41 | -0.38 | UniCache |
| sharegpt@100x2 | 100 | 159 | 2048 | 30.88 | 13.09 | 13.09 | 17.93 | 16.23 | 43.97 | +3.13 | -1.71 | LeCaR |

Committed before any GPU run (2026-11-25). Capacities were chosen from the sim
sweeps (`sweep/capacity_selection.csv`, `sweep-variants/capacity_selection.csv`)
by the policy-neutral rules in `scripts/sim2real/select_capacity.py`; the agent
column was not used. All Tier-1 deltas below are what the GPU will be checked
against, with the rule: transfers if the sign matches and
|Δ_sim − Δ_GPU| ≤ max(0.5 pp, 2 × LRU noise floor).

Predicted deltas vs LRU (pp):

| Workload | UniCache − LRU | LeCaR − LRU | Agent − LRU | Agent − Best |
|---|---:|---:|---:|---:|
| ALFRED (60 × ≤12 turns, 1,024 blocks) | 0.00 | +1.91 | +2.61 | +0.70 (vs LeCaR) |
| CodeParrot (600 sessions, 8,192 blocks) | +2.79 | +1.62 | +2.41 | −0.38 (vs UniCache) |
| ShareGPT (100 × ≤2 turns, 2,048 blocks) | 0.00 | +4.84 | +3.13 | −1.71 (vs LeCaR) |

Hypotheses: CodeParrot (single-turn) transfers; ALFRED and ShareGPT
(multi-turn, completion-dependent follow-ups) may not.

Excluded: MASH-QA. No capacity from 512 to 8,192 blocks gives a heuristic
spread ≥ 2 pp even at 900 sessions (15 min); all policies are within ~1 pp,
so there is no decision to test. AgentBank APPS was excluded by design
(OPT − LRU < 0.5 pp in H1).
