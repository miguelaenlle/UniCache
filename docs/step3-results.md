# Step 3 results: agent-optimized eviction policy, simulator vs GH200

Status: complete (2026-11-26 18:40). Simulator: agent run 15:11–15:20, OPT 15:30. GPU: two 15-minute jobs on gh152, 18:14–18:25.

## Setup

| Item | Value |
|---|---|
| Trace | CodeParrot (H1 split, `TaskData/Codeparrot/agent-splits/codeparrot/v1-5000`) |
| Tune slice | first 600 sessions of the H1 **tune** split, by simulated start |
| Eval slice | first 600 sessions of the H1 **eval** split (= `codeparrot@600`); disjoint from tune |
| Load | first-turn arrivals × 0.04 (25× the recorded rate) |
| Cache | 4,096 blocks × 16 tokens (64k tokens) |
| Model | Llama-3.2-1B-Instruct (dummy weights on GPU, outputs forced to recorded tokens) |
| Simulator timing | `calibrations/gh200-llama-3.2-1b-v8.json` (fit on the synthetic sweep only) |
| GPU | 1 × NVIDIA GH200 (DeltaAI), all GPU replays in one job on one node |
| Agent | `claude:opus`, ≤ 30 submissions, objective = mean TTFT on tune, starting policy LRU |
| Agent run dir | `result/policy-agent/2026-11-26/151145-668720-codeparrot` (5 submissions; stopped early, see notes) |
| GPU job / node | half A: 3683573, half B: 3683941; both on gh152 (18:14–18:19, 18:20–18:25) |

**Split discipline.** The agent sees only tune scores (simulator). The submission is selected by
tune TTFT alone. Eval scores, eval traces and OPT are never shown to the agent. Eval is scored once,
after selection, in the simulator and on the GPU. Before/after and simulator-vs-GPU claims use eval
only.

## 1. Tune slice (simulator; what the agent optimized)

| Policy | Hit ratio (%) | TTFT mean (ms) | TTFT p90 (ms) | ITL mean (ms) |
|---|---:|---:|---:|---:|
| LRU (starting policy, built-in) | 14.50 | 45.00 | 130.58 | 5.00 |
| LRU via sandbox (agent's starting controller) | 14.50 | 92.32 | 278.90 | 5.30 |
| LFU | 15.62 | 46.62 | 131.54 | 5.01 |
| FIFO | 13.24 | 45.77 | 128.84 | 5.01 |
| LeCaR | 15.30 | 46.13 | 136.74 | 5.01 |
| UniCache | 15.61 | 47.40 | 140.05 | 5.01 |
| Best heuristic on tune: LRU | 14.50 | 45.00 | 130.58 | 5.00 |
| **Agent (selected)**, submission 4 | 14.50 | 91.72 | 268.61 | 5.29 |
| Agent runner-up 1, submission 0 | 14.50 | 92.32 | 278.90 | 5.30 |
| Agent runner-up 2, submission 1 | 14.50 | 92.71 | 275.05 | 5.30 |
| OPT (upper bound; future map not refined) | 28.69 | 36.57 | 95.59 | 4.92 |

## 2. Eval slice (held out)

### 2a. Simulator

| Policy | Hit ratio (%) | TTFT mean (ms) | TTFT p90 (ms) | ITL mean (ms) |
|---|---:|---:|---:|---:|
| LRU (starting policy, built-in) | 15.44 | 81.23 | 313.35 | 5.14 |
| LRU via sandbox (agent's starting controller) | 15.44 | 116.35 | 502.78 | 5.51 |
| LFU | 17.88 | 79.24 | 314.22 | 5.10 |
| FIFO | 15.70 | 80.89 | 313.35 | 5.13 |
| LeCaR | 17.82 | 79.21 | 317.47 | 5.09 |
| UniCache | 18.43 | 79.67 | 319.27 | 5.10 |
| Best heuristic on tune: LRU | 15.44 | 81.23 | 313.35 | 5.14 |
| **Agent (selected)** (submission 4) | 15.44 | 118.64 | 514.50 | 5.51 |
| Agent runner-up 1 (submission 0) | 15.44 | 116.35 | 502.78 | 5.51 |
| Agent runner-up 2 (submission 1) | 15.44 | 118.62 | 517.28 | 5.51 |
| OPT (upper bound; future map not refined) | 31.18 | 74.16 | 301.59 | 5.00 |

### 2b. GH200 (one node)

| Policy | Runs | Hit ratio (%) | TTFT mean (ms) | TTFT p90 (ms) | ITL mean (ms) |
|---|---:|---:|---:|---:|---:|
| LRU (starting policy, built-in) | 3 | 15.44 | 91.05 (94.72, 80.86, 97.56) | 394.17 | 5.37 |
| LFU | 1 | 17.71 | 92.85 | 412.57 | 5.42 |
| FIFO | 1 | 15.87 | 121.13 | 497.28 | 5.70 |
| LeCaR | 1 | 17.65 | 93.70 | 402.45 | 5.43 |
| UniCache | 2 | 18.11 | 98.53 (91.99, 105.07) | 438.75 | 5.55 |
| **Agent (selected, submission 4)** | 2 | 15.44 | **132.17** (135.90, 128.45) | 529.32 | 5.73 |
| Agent runner-up 1 (submission 0) | 1 | 15.44 | 159.28 | 575.86 | 5.94 |
| Agent runner-up 2 (submission 1) | 1 | 15.44 | 178.73 | 588.98 | 5.97 |

Noise floor (spread of the 3 LRU runs): hit ratio 0.00 pp, TTFT 16.70 ms (80.86–97.56, ±9 % of the
mean), ITL 0.21 ms. Hit ratios on the GPU match the simulator within 0.4 pp for every policy (eviction
counts differ by < 0.2 %); the agent policies make exactly LRU's decisions (48,048 hit tokens).

## 3. Before vs after (eval, GH200)

The best heuristic on tune (by TTFT) is LRU, so both rows compare with the same baseline.

| Comparison | TTFT change | ITL change | Hit ratio change (pp) | Beyond noise? |
|---|---:|---:|---:|---|
| Agent vs LRU (built-in) | **+45.2 %** (132.17 vs 91.05 ms) | +6.7 % (5.73 vs 5.37 ms) | 0.00 | Yes, worse: the agent's fastest run (128.45) is slower than LRU's slowest (97.56) |
| Agent vs best heuristic on tune (LRU) | +45.2 % | +6.7 % | 0.00 | Yes, worse (same as above) |

Equal-overhead comparison (eval, simulator; the sandboxed LRU is not replayed on the GPU):

| Comparison | TTFT change | ITL change | Hit ratio change (pp) |
|---|---:|---:|---:|
| Agent vs LRU via sandbox | +2.0% (118.64 vs 116.35 ms) | 0.0% (5.51 vs 5.51 ms) | 0.00 |

Reading: the agent did not improve the policy. Its selected submission makes the same eviction
decisions as LRU; it is slower only because it runs as a sandboxed Python process that the engine
calls over IPC. Submission 0 on the GPU (the unchanged starting policy, i.e. LRU through the sandbox)
is effectively the GPU's "LRU via sandbox" row: 159.28 ms, +75 % over built-in LRU.

## 4. Does the simulator's verdict hold on the GH200? (eval)

Error = (sim − GPU) / GPU. Rank 1 = lowest TTFT.

| Policy | Sim TTFT (ms) | GPU TTFT (ms) | TTFT error | Sim ITL (ms) | GPU ITL (ms) | ITL error | Rank by TTFT (sim) | Rank by TTFT (GPU) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LRU | 81.23 | 91.05 | −10.8 % | 5.14 | 5.37 | −4.4 % | 5 | 1 |
| LFU | 79.24 | 92.85 | −14.7 % | 5.10 | 5.42 | −5.9 % | 2 | 2 |
| FIFO | 80.89 | 121.13 | −33.2 % | 5.13 | 5.70 | −10.0 % | 4 | 5 |
| LeCaR | 79.21 | 93.70 | −15.5 % | 5.09 | 5.43 | −6.3 % | 1 | 3 |
| UniCache | 79.67 | 98.53 | −19.1 % | 5.10 | 5.55 | −8.1 % | 3 | 4 |
| Agent (selected) | 118.64 | 132.17 | −10.2 % | 5.51 | 5.73 | −3.8 % | 8 | 6 |
| Agent runner-up 1 | 116.35 | 159.28 | −27.0 % | 5.51 | 5.94 | −7.2 % | 6 | 7 |
| Agent runner-up 2 | 118.62 | 178.73 | −33.6 % | 5.51 | 5.97 | −7.7 % | 7 | 8 |

Rank agreement (Spearman ρ over policies): 0.67 over all 8; −0.10 over the 5 built-in heuristics alone.
Agent − LRU TTFT gap: simulator +46.1 %, GPU +45.2 %.

Reading:
- **The verdict that matters holds.** The simulator predicts the agent's policy is ~46 % slower than
  built-in LRU; the GPU measures 45 %. Both say "no improvement, slower". Hit ratios agree within 0.4 pp.
- **The simulator is optimistic on TTFT by 10–34 %** (median −17 %), more than the v8 held-out median
  (−13 %). ITL is optimistic by 4–10 %.
- **The order among the five built-in heuristics is not resolved by either side.** In the simulator they
  span 2.0 ms (2.5 %); on the GPU four of them span 7.5 ms, inside the 16.7 ms LRU noise floor (ρ = −0.10
  is noise). FIFO's 121 ms is a single run and likely a high draw; its hit ratio (15.87 %) is LRU's.
- **GPU noise comes from one burst.** Per-quartile TTFT (by arrival) is ~22 ms in the first quartile of
  every run; almost all of the mean comes from the second quartile, a burst where the queue builds up
  (257–438 ms). Small step-time differences compound there, so TTFT means at this load vary ±10–20 %
  run to run, and the sandboxed runs vary more (identical decisions: 128–179 ms).

## 5. Charts

| Chart | File |
|---|---|
| TTFT by policy, simulator vs GPU, eval (bars with run spread; OPT dashed) | `result/sim2real-transfer/step3/ttft-by-policy.png` |
| ITL by policy, simulator vs GPU, eval | `result/sim2real-transfer/step3/itl-by-policy.png` |
| Simulator vs GPU scatter, TTFT and ITL, eval | `result/sim2real-transfer/step3/sim-vs-gpu-scatter.png` |
| Hit ratio vs TTFT per policy, GPU, eval | `result/sim2real-transfer/step3/hit-vs-ttft-gpu.png` |
| Agent tune TTFT by submission (selection curve) | `result/sim2real-transfer/step3/agent-selection-curve.png` |

Regenerate tables and charts with `.venv-sim312/bin/python scripts/sim2real/step3_report.py`
(also writes `summary.json`).

## 6. Files

| What | Path |
|---|---|
| Simulator runs, tune and eval (one row per policy) | `result/sim2real-transfer/step3/sim-runs.csv` |
| GPU runs, eval (one row per replay) | `result/sim2real-transfer/step3/gpu-runs.csv` |
| Raw GPU replays (per-request latencies, step logs) | `result/sim2real-transfer/gpu-step3/{A,B,A-c1,A-c2,B-c1,B-c3}/` |
| Agent submissions (tune scores; eval scored after selection, never shown to the agent) | `result/policy-agent/2026-11-26/151145-668720-codeparrot/submissions.jsonl` |
| Selected policy and rationale | `<agent run dir>/policy.py`, `internal/submission-NNNN/rationale.md` |

Columns of `gpu-runs.csv`: `split, policy, half, rep, node, hit_pct, ttft_mean_ms, ttft_p90_ms,
itl_mean_ms, evictions, replay_s`. `sim-runs.csv` has `split, policy, hit_pct, ttft_mean_ms,
ttft_p90_ms, itl_mean_ms`.

## 7. Notes

- Deviations from this plan: OPT's future map was not refined (`refine_opt.py`); under `aic` timing its hit ratio is a slight underestimate of the true bound. The agent stopped after 5 of 30 submissions (stop reason: eviction order barely changes hits on this trace, policy compute raises TTFT). The 30-minute GPU job was split into two 15-minute halves, each with its own LRU run; both halves landed on the same node, so they are pooled.
- Eval-leak check: the agent's sandbox (`<agent run dir>/sandbox/`) contains no eval score; eval values exist only in the host-side `submissions.jsonl`.
- Caveats: one trace, one load (25×), one cache size, one node; 1–3 GPU runs per policy. At this load a 1B model's TTFT is dominated by queueing in one burst, so policy differences smaller than ~15 % are not measurable with this many runs. Hit-ratio gains of 2–3 pp (UniCache, LFU, LeCaR vs LRU) do not show up as TTFT gains on the GPU.
- What would make step 3 informative: (1) run the agent's policy in-process (no IPC sandbox) on the GPU so its TTFT reflects decisions, not overhead; (2) use a trace and cache size with more hit-ratio headroom (OPT − LRU here is 15.7 pp, but heuristics only reach +3 pp); (3) 3+ runs per policy, or a lighter load where TTFT is less burst-dominated.
