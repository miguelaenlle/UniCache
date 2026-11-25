# GH200 calibration and sim2real transfer plan (DeltaAI)

Status: plan, 2026-11-25. Revised after an advisor review (a Fable agent read
the plan against the code). Nothing has run on DeltaAI yet.

## Goals

1. **Calibrate** the simulator's aiconfigurator step timing to a GH200, so
   that its TTFT/ITL can be trusted for Llama-3.2-1B. aiconfigurator has no
   GH200 table. The GH200 has the same H100 die, with more memory bandwidth and a
   Grace (aarch64) host, so we fit a per-phase correction on top of `h100_sxm`:
   `step_ms = scale[phase] · aic_ms + overhead_ms[phase]`.
2. **Test sim2real transfer of H1:** do the hit ratios the simulator gave the
   H1 policies (report Table 2) hold on a real GPU? Do the agent's gain over LRU
   and its gap to the best heuristic carry over? And does simulator timing
   (untuned vs calibrated) change the answer? Any outcome is a result: either
   the simulator's timing matters for policy search or it doesn't.

Llama-3.1-8B is dropped for now. It is in neither H1 nor the transfer test, and
dropping it saves P2 time, 16 GB of disk and the grid's memory risk. Add it
later for the "larger models make hits matter more for latency" question.

## Budget

| Resource | Plan |
|---|---|
| GH200 hours | **≈ 5–6 GPU-hours total**: smoke (≤ 15 min), calibration grid + subsets (≤ 1 h), transfer replays (≈ 4.5 h with truncation; ≈ 9 h without). 1 GPU per job on the batch partition `ghx4` (charge 1.0; interactive is 2.0). Never a whole node (charged as 4 GPUs). Everything else runs on the laptop, at 0 GPU-hours. |
| DeltaAI disk | ≈ 10 GB, < 1,000 files, under `/work/hdd/bcjw/maenlle2/kvcal` (confirm on DeltaAI; `/work/nvme/bcjw` is over its soft quota and home is at 93/100 GB). Software lives in one container image. No weights download: GPU mode runs `load_format="dummy"`, so only config and tokenizer are needed. Run caches go to `$TMPDIR`; each job tars its outputs into one file. |
| Laptop | Only CSV/JSON results come back (≈ 50 MB). Simulations run ≤ 3 at a time (an ALFRED replay holds ≈ 3 GB). |

## Facts that shape the plan (verified in the code)

* **GPU mode cannot run the H1 commands as-is.** `run_simulation.py` hardcodes
  `swap_space=0`, so the worker reports 0 CPU blocks, and `llm_engine.py`
  asserts CPU blocks ≥ `--simulator-num-cpu-blocks`. That is 50,000 in every
  H1 `command.json`. H2 only worked because its subsets use 0 CPU blocks.
* **Weights are random** (`load_format="dummy"`), and outputs are forced to the
  trace's recorded tokens. That is fine for timing and hit ratio.
* **`measured_ms` only times `engine.step()`.** Tokenizing each prompt
  (`add_request` runs between steps on raw text; ALFRED prompts average 2.4k
  tokens), the step log, the latency recorder and next-turn scheduling all
  happen outside the timer, but take real time on the GPU. So step error
  can pass while TTFT fails.
* **The GPU replays in real time.** Arrivals are released by wall clock, with
  a sleep when idle, and follow-ups arrive at finish time plus a lognormal think time; this
  matches the simulator's logic. A full eval split is 38–47 min of
  recorded arrivals, ≈ 0.75 GPU-hours per run.
* **`--arrival-scale` only scales first-turn arrivals, not think times.**
  Compressing time would mis-scale ALFRED and change the concurrency that is
  under test, so we **truncate** instead (below).
* **CodeParrot is not timing-insensitive.** Its eval split holds a request with
  53,410 output tokens, about 6,700 blocks (20% of a 32,768-block cache),
  pinned for 53k decode steps. How long it stays pinned scales directly with
  real decode step time.
* The source of report Table 2 is
  `UniCache-agent-budget-v1/result/agent-cache-budget-study/manual-20261101-162735/`
  (`agent-runs/<blocks>/<trace>.json` names each agent run and its selected
  submission). The `2026-11-01-v1-verified` directory lacks some Table 2 runs,
  for example ALFRED at 8 GiB.

## The transfer test design

For each chosen configuration, three policies run on the **eval split at
the Table 2 capacity**: LRU, the tune-selected best heuristic, and the agent's
selected submission. Each runs once on the GPU, and the result is compared with
three predictors:

| Predictor | Timing | Cost |
|---|---|---|
| S0: H1 simulator | legacy formula (the agents' training signal) | laptop rerun on the same truncated prefix |
| S1: aiconfigurator, uncalibrated | `h100_sxm` tables | laptop |
| S2: aiconfigurator, calibrated | `h100_sxm` + GH200 calibration, fit leaving this workload out | laptop |

**Metrics.**
* **Primary: do the decisions transfer?** Compare Agent − LRU and
  Agent − Best, predicted vs GPU: the error in pp, and whether the sign
  matches. On MASH-QA only Agent − Best (−2.22 pp) is pre-registered. Its
  Agent − LRU (−0.16 pp) is below any plausible noise.
* Per-policy |H_pred − H_GPU| (pp); TTFT mean/p90 and ITL per policy (S1, S2
  vs GPU); and whether the agent's hit gain becomes a TTFT gain on the GPU.
* Python-policy CPU time: per-step `measured_ms` of the agent vs LRU at
  matched batches. It sits inside `step()`, so on the GPU it delays completions
  and follow-up arrivals. That makes it a hit-ratio confound for multi-turn
  workloads, not only a latency cost, and the simulator does not model it.
* GPU noise: one LRU repeat per chosen config, the noise floor for that
  config. A difference within it is reported as unresolved.

**How to read the result.**

| S0 transfers? | S2 transfers? | Conclusion |
|---|---|---|
| yes | yes | Hit-ratio search is robust to timing; the calibrated simulator is needed only for latency claims |
| no | yes | A tuned simulator is necessary; the H1 numbers need re-scoring |
| no | no | A mismatch beyond timing (control plane, preemption, policy CPU time); diagnose from step logs |
| yes | no | Calibration hurts; check the fit |

The "no / yes" row is only reachable if hit ratios are sensitive to timing in
the first place. In the CodeParrot load sweep, going from legacy to aic timing
moved hit ratio by only −0.2 to −0.3 pp. So L0 measures that sensitivity
**before** any GPU time is spent and before the configurations are chosen.

## Steps

### L0. Laptop, before any GPU job (0 GPU-hours)

1. **Timing-sensitivity sweep.** For all nine Table 2 configs × {LRU, Best,
   Agent}, re-simulate the eval split under S0, under S1, and under synthetic
   calibrations (scale 0.5 and 2, overhead +5 ms). Report how far hit ratios
   and the Agent − LRU / Agent − Best deltas move. That is 108 runs, about
   1–2 h at 3 parallel.
   * If the deltas move by < 0.3 pp everywhere, that alone is the
     timing-robustness result. The GPU test then asks only whether the
     control plane transfers.
   * Choose the GPU configs from this sweep: the two most timing-sensitive
     ones plus one insensitive control. The default, if nothing is sensitive:
     CodeParrot 8 GiB, ALFRED 8 GiB, MASH-QA 12.207 GiB. Write the chosen
     configs and the predicted direction of each delta into this doc before
     T2 is submitted.
2. **CPU swap blocks.** Re-run S0 for the chosen configs at 0 CPU blocks.
   * If hit ratios don't change, use 0 everywhere: GPU, S0, S1, S2. The swap
     confound and the assert both go away.
   * If they do change, add a `--swap-space` flag to `run_simulation.py` and
     request enough `--mem` for ≈ 12 GiB of pinned host memory.
3. **Truncation.** Add `--max-first-arrival T` to `rerun_with_latency.py`. It
   keeps only sessions whose first turn arrives before T (all their turns
   stay), so the load and the concurrency stay at H1's operating point.
   Check in S0 that the deltas on the first 50% match the full-split deltas
   (within 0.3 pp). If they do, T2 replays only that prefix. Otherwise it
   replays full splits (≈ 2× the cost).
4. **Measurement fixes.**
   * Step log: add `loop_ms`, the wall time of the whole engine-loop
     iteration, including `add_request` tokenization, and `raw_ms`, the
     uncalibrated aic price of the exact FPM the simulator used (with
     variance).
   * `fit_calibration.py`: fit on `raw_ms`, weighted by 1/y (relative error),
     with the target selectable (`measured_ms` or `loop_ms`). Replace the
     random-row holdout with a workload holdout: consecutive decode steps are
     near-duplicates.
   * `profile_steps_grid.py`: block size 8, `max_num_seqs` 32, batch ≤ 32,
     dummy weights. Use it for residual diagnostics only, never mixed into a
     fit: it times a whole `generate` call, not one step.
5. **Policy manifest.** For each chosen config, list the agent `policy.py`
   (run dir, selected submission, sha256) and the baseline commands. Read each
   policy. On DeltaAI there is no Seatbelt, so policies run with networking
   off inside the container (see P0).

### P0. DeltaAI setup on the login node (0 GPU-hours)

1. Log in to a fixed node (`gh-login01`) in tmux. Confirm the account
   (`accounts`), partitions, quotas, and that `/work/hdd/bcjw` is mounted.
2. **Build the fork first; this is the main risk.** The fork pins torch 2.6.0,
   and on aarch64 it must compile its CUDA kernels (xformers is x86-only in
   `requirements/cuda.txt`). In order:
   * (a1) If the fork's C/CUDA sources match its upstream base, install that
     version's prebuilt aarch64 wheel (or use `VLLM_USE_PRECOMPILED=1`) and
     overlay the fork's Python. Check that an aarch64 wheel exists and that
     torch 2.6.0 has aarch64 CUDA wheels.
   * (a2) Otherwise compile inside an arm64 PyTorch container, for sm_90 only.
     It needs nvcc, not a GPU. Check NCSA policy on heavy login-node builds; a
     CPU-only job is the alternative.
   * (b) Upstream vLLM (V1) supports the calibration grid only. **It cannot
     run the transfer test**: there are no forced outputs and no policy hooks.
     So T is gated on (a1) or (a2).
3. Check that networking can be switched off for policy runs: `apptainer exec
   --net --network none`, or `unshare -rn`.
4. Copy the repo, the tokenizer/config, the subset inputs, the eval inputs of
   the chosen configs (CodeParrot 33 MB, MASH-QA 56 MB, ALFRED `v1-500`), and
   the policy manifest. Write one `sbatch` script per job, each with hard
   `--time` limits and one GPU.

### P1. Smoke job (≤ 15 min, 1 GPU)

`nvidia-smi`; the container sees the GPU; a 20-session demo replay in GPU mode
with the new step-log columns; one agent policy loads via
`UNICACHE_POLICY_CANDIDATE` with networking off; GPU mode accepts 32,768 (and
50,000) GPU blocks with the chosen CPU-block setting; two grid points.
Copy the outputs back and check the columns and that times are sane (a 1B
decode step should take a few ms). Stop here if anything is off.

### P2. Calibration job (≤ 1 h, 1 GPU)

The five H2 subsets with LRU only (for continuity with H2, and ShareGPT's
decode-timing question), then the restricted grid. Most of the calibration
data comes from T2, because those runs are in the right regime (block size
8, 16k–50k blocks, full sessions).

### T2. Transfer job (≈ 4.5 h with truncation, 1 GPU)

For each of the 3 chosen configs, run {LRU, Best, Agent, LRU repeat}
sequentially on the truncated eval prefix, each with the step log. That is 12
runs of ≈ 22 min. Each run outputs ≈ 1 MB. The job doesn't depend on P2, so
both are submitted as soon as P1 passes.

### P3. Fit (laptop)

LRU runs only. A policy's CPU cost would otherwise land in `overhead_ms`;
UniCache and agent overhead are reported separately as per-step deltas.
**Leave one workload out**: the S2 calibration used for workload W is fit on
the T2 LRU runs of the other workloads plus the P2 subsets. So S2 is never
tested on data it was fit on. Fit to `loop_ms` as well as `measured_ms`, and
keep the one that predicts TTFT better on held-out runs. If the residuals show
structure (by batch size or context), try aiconfigurator's `tune_with_fpms` on
the same rows. This step can be handed to an agent (Fable), ADRS-style, that
sees only the fit set.

Targets on held-out runs:

| Check | Target |
|---|---|
| Step time, median abs relative error (prefill, decode separately) | ≤ 10% |
| Request TTFT, median abs relative error | ≤ 20% |
| Policy order by mean TTFT (LRU, UniCache, agent; not OPT, whose map goes stale under real timing) | same as GPU, where the gap exceeds the noise floor |

### T3. Transfer analysis (laptop)

Run S0, S1 and S2 on the same truncated prefixes, ≤ 3 at a time. Write
`transfer.md`: the metrics table above, the reading table, and the L0
sensitivity numbers next to it.

### P5. Use it

Re-run the CodeParrot load sweep and the Table 5 reproduction with the
calibration, and update `docs/aic-timing.md`. If the reading is "no / yes",
re-score H1 under S2.

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| The fork won't build on aarch64 | Grid only on upstream vLLM; T is not possible on DeltaAI. Ask a teammate for A100 time for T instead (A100 tables exist) |
| Prefix deltas differ from full-split deltas | Full splits for 2 configs (≈ 6 GPU-hours) instead of 3 truncated |
| Swap blocks matter (L0.2) | Expose `swap_space`, request `--mem`; T1 checks it |
| `engine.step()` wall time misses async GPU work | Synchronize at the end of the timed region in GPU mode; check in P1 that decode steps match the grid |
| Small-model steps are noise-dominated | Fit on medians per batch shape; drop steps < 0.5 ms |
| Queue wait on `ghx4` | Submit P2 and T2 together right after P1 |
| Running several replays on one GPU to save hours | Not done: contention invalidates timing |

## Decisions needed

1. The DeltaAI account name (Delta's is `bcjw-delta-gpu`; DeltaAI allocations
   are separate), and whether `/work/hdd/bcjw` is the right place there.
2. Whether building on the login node is OK, or it should be a CPU-only job.
3. The budget: ≈ 5–6 GPU-hours as planned (3 truncated configs), or ≈ 20 for
   all nine Table 2 configs.
