# GPU step timing with aiconfigurator (TTFT / ITL in the simulator)

Branch `experiment/aic-timing`. Prototype written 2026-11-23, not yet validated
on a GPU.

## What it does

The simulator already decides **what** runs in every engine step: vLLM v0's
scheduler and prefix cache pick the requests and record which prompt tokens hit
the cache, and the eviction policy decides what the cache holds. This prototype
adds **how long** each step takes on a chosen GPU, and records per-request
latency.

```
scheduler + prefix cache + eviction policy   (unchanged UniCache / vLLM v0)
        │  this step: prefill  new tokens, cached tokens per request
        │             decode   context length per request
        ▼
aiconfigurator (NVIDIA): measured vLLM kernel tables for the chosen GPU
        │  raw_ms
        ▼
step_ms = scale[phase] * raw_ms + overhead_ms[phase]      (calibration JSON)
        │
        ▼
virtual clock advances → per-request timestamps → TTFT, ITL, TPOT, E2E
```

Two properties matter for the research:

* **The KV cache is identical**: same engine, same policies. With
  `--simulator-timing legacy` (the default), runs reproduce recorded hit ratios
  exactly (checked: CodeParrot 8 GiB eval, LRU / UniCache / OPT).
* **Cache hits now save time.** The legacy formula charged prefill by *full*
  prompt length, so every policy got identical TTFT. Here a prefill step is
  charged only for its uncached tokens, using the same `computed_block_nums`
  that the hit-ratio statistic counts.

## Files

| File | Role |
|---|---|
| `vllm/simulator/step_timing.py` | `StepBatch` (token counts of one step), `AicStepTiming` (aiconfigurator lookup), `Calibration` (scale + overhead per phase) |
| `vllm/simulator/latency_metrics.py` | `LatencyRecorder` (TTFT/ITL/TPOT/E2E), `StepLog` (one CSV row per step) |
| `vllm/simulator/simulator.py` | Wires both into the event loop; legacy formula moved unchanged into `_advance_legacy_timing` |
| `vllm/engine/llm_engine.py` | One line: keeps `last_scheduled_metadata` for the step |
| `run_simulation.py` | New flags (below) |
| `scripts/aic_timing/` | Re-run recorded replays, GPU profiling, calibration fit, sim-vs-GPU comparison, sweep summary, tests |

## Flags (`run_simulation.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--simulator-timing {legacy,aic}` | `legacy` | Step latency model |
| `--simulator-aic-system` | `h100_sxm` | aiconfigurator GPU: `a100_sxm`, `h100_sxm`, `h200_sxm`, `b200_sxm`, ... |
| `--simulator-aic-model` | `--model` | Model whose shapes are priced (HF id or local dir) |
| `--simulator-aic-backend-version` | newest shipped | vLLM data version (`a100_sxm`: 0.14.0, `h100_sxm`: 0.24.0) |
| `--simulator-timing-calibration` | none | Calibration JSON |
| `--simulator-step-log PATH` | off | One CSV row per step: batch shape, `model_ms`, and in GPU mode `measured_ms` |

Every run, in both timing modes, now also writes `latency_requests.csv` and
`latency_summary.json` next to `summary.json`.

Metric definitions (seconds): **TTFT** = first token − arrival, including
queueing. **ITL** = gap between consecutive tokens of a request (pooled).
**TPOT** = a request's mean ITL. **E2E** = finish − arrival. Arrival is when
the trace releases the request (a follow-up turn: previous completion + think
time).

Setup: `uv pip install --python .venv-sim312/bin/python -r requirements/simulator-aic.txt`.

## Re-running the report's experiments with latency

`scripts/aic_timing/rerun_with_latency.py` re-executes any recorded replay
(`command.json` from the H1 study or an agent run), changing only the output
directory and the timing flags:

```bash
B=../UniCache-agent-budget-v1/result/agent-cache-budget-study/2026-11-01-v1-verified/baseline-replays
.venv-sim312/bin/python scripts/aic_timing/rerun_with_latency.py \
  --commands "$B/32768/codeparrot/{LRU,UniCache,OPT}/eval/command.json" \
  --out result/aic-timing/codeparrot-32768 --timing aic --system h100_sxm \
  --arrival-scale 0.03          # optional: 33x the recorded arrival rate

# The agent's best policy, same trace, evaluated the same way:
A=../UniCache-agent-budget-v1/result/policy-agent/2026-11-01/163922-145041-codeparrot
.venv-sim312/bin/python scripts/aic_timing/rerun_with_latency.py \
  --commands "$A/internal/submission-0020/eval/command.json" --candidate $A/policy.py \
  --out result/aic-timing/codeparrot-32768/agent --timing aic --system h100_sxm
```

`--arrival-scale` multiplies first-turn arrival times. At the recorded rate the
traces barely load one GPU, so TTFT is almost pure prefill time and there is no
queueing. Label compressed runs as a separate scenario.
`summarize_sweep.py` turns a sweep directory into `sweep.csv` and
`ttft_vs_load.png`.

## Tomorrow: calibrating on real GPUs

Goal: fit `scale` and `overhead_ms` per phase so that simulated step times match
measured ones, then check request-level TTFT/ITL against a real replay.

1. **Collect measured steps.** Do either or both:
   * Controlled grid, any vLLM install with CUDA (untested script):
     `VLLM_USE_V1=0 python scripts/aic_timing/profile_steps_grid.py --model <model> --out result/gpu/<gpu>/grid_steps.csv`
     (add `--quick` first).
   * Real workload: run a sim2real replay in UniCache GPU mode (no
     `--simulator-enable-virtual-time`) with
     `--simulator-step-log <dir>/steps.csv --simulator-timing aic --simulator-aic-system <tables>`.
     That gives `measured_ms` per step from `engine.step()` wall time, and
     `model_ms` for the same step.
2. **Fit:**
   `.venv-sim312/bin/python scripts/aic_timing/fit_calibration.py --steps 'result/gpu/<gpu>/**/steps.csv' --model <model> --system <tables> --out calibrations/<gpu>-<model>.json`.
   It prints held-out median error before and after calibration, per phase.
3. **Validate end to end:** re-run the same trace in simulation with
   `--simulator-timing-calibration calibrations/<gpu>-<model>.json`, then
   `scripts/aic_timing/compare_latency.py --sim <sim run dir> --real <GPU run dir>`.
4. **What to try if errors stay high:** the calibration is intentionally just
   two numbers per phase. aiconfigurator's own tuner
   (`RustForwardPassPerfModel.tune_with_fpms`, bucketed correction factors) is a
   drop-in alternative: `StepBatch.to_fpm(wall_time_s=...)` already produces its
   input format.

### GPU / model coverage notes

* **A100 + Llama-3.2-1B:** the `a100_sxm` vLLM tables lack attention data for
  head size 64, so `aic` refuses this pair with a clear error. Use
  `--simulator-aic-system h100_sxm` and let the A100 calibration's `scale`
  absorb the hardware gap (e.g. decode scale ≈ 2 means "A100 decode is about 2x
  the H100 tables"). Llama-3.1-8B (head size 128) has A100 tables.
* **GH200 (DeltaAI):** no aiconfigurator table. Start from `h100_sxm` (same GPU
  die; GH200 has 96 GB HBM3 and higher bandwidth) and calibrate.
* aiconfigurator tables are kernel times from newer vLLM releases (0.14/0.24),
  not UniCache's v0 fork, and they exclude Python/scheduler overhead. That is
  what `overhead_ms` is for, and it is large for a 1B model.

## First results (uncalibrated tables, 2026-11-24)

CodeParrot, 8 GiB budget (32,768 blocks), eval split, exact recorded commands
re-run with `--simulator-timing aic` and `--arrival-scale` from 1 down to 0.01
(1x to 100x the recorded arrival rate). "Agent" is the H1 agent's selected
policy (submission 20). Output: `result/aic-timing/codeparrot-32768-load-sweep/`
(`sweep.csv`, `ttft_vs_load.png`).

Mean TTFT, Llama-3.1-8B shapes on `a100_sxm` tables:

| Load | LRU | UniCache | Agent | OPT |
|---:|---:|---:|---:|---:|
| 1x | 31 ms | 28 ms (0.88x) | 28 ms (0.90x) | 25 ms (0.79x) |
| 3x | 33 ms | 29 ms (0.88x) | 30 ms (0.90x) | 26 ms (0.78x) |
| 10x | 11.6 s | 8.5 s (0.73x) | 9.1 s (0.78x) | 6.5 s (0.56x) |
| 33x | 95 s | 92 s (0.97x) | 93 s (0.97x) | 90 s (0.95x) |

Hit ratios: LRU 0.415, Agent 0.525–0.530, UniCache 0.548–0.555, OPT 0.682.

Reading:
* Hit-ratio gains now show up as latency gains. Under legacy timing every
  policy had identical TTFT (26.29 ms), because prefill was charged by full
  prompt length.
* The benefit peaks near saturation (10x here). Below it, TTFT is mostly
  prefill time and differences are 10–20%. Far past it, queueing swamps
  everything. This matches the paper's Table 5 pattern, where the most loaded
  trace has the largest absolute gaps.
* Llama-3.2-1B on `h100_sxm` saturates only near 100x, and the differences
  are smaller (0.92–0.94x LRU for UniCache and Agent). A 1B model on an
  H100-class GPU has little prefill to save.
* These use uncalibrated tables; tomorrow's GPU calibration changes the
  absolute numbers, and possibly where saturation sits.

## Known limitations

* The calibration is linear per phase. Batches mixing prefill and decode don't
  occur (chunked prefill is off), so `mixed` is unused.
* Preemption-by-recompute is priced as a normal prefill of the recomputed
  tokens. That's correct in spirit, but untested.
* Changing the timing changes when follow-up turns arrive and how recency
  orders blocks, so hit ratios shift slightly (−0.2 to −0.3 pp observed on
  CodeParrot 8 GiB for LRU and UniCache; OPT unchanged). Report hit ratios from
  the same timing mode as the latency numbers.
* The policy-agent loop itself still uses legacy timing. Its feedback is hit
  ratio, so this doesn't matter yet. Passing the flags through is a small
  change in `policy_agent/runner.py` once latency becomes a reward.
