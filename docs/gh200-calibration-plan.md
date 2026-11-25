# Calibrating the UniCache + aiconfigurator simulator on GH200 (DeltaAI)

Status: plan, 2026-11-25. Nothing has run on DeltaAI yet.

## Goal

Make the simulator's per-step latency match a GH200, so that its TTFT/ITL can
be trusted, and show that on held-out replays. Concretely:

* Fit `calibrations/gh200-<model>.json` (scale + overhead per phase) on top of
  aiconfigurator's `h100_sxm` tables. aiconfigurator has no GH200 table; the GH200
  is the same H100 die, with 96 GB HBM3, more memory bandwidth and a Grace
  (aarch64) host CPU. The two calibration numbers per phase absorb exactly
  those differences.
* Pass these targets on replays **not used for fitting**:

| Check | Target |
|---|---|
| Step time, median abs error (prefill, decode separately) | ≤ 10% |
| Request TTFT, median abs error | ≤ 20% |
| Policy order by mean TTFT (LRU, UniCache, OPT) | same as GPU |
| Hit ratio, sim vs GPU (as in H2) | within 1.5 pp; ShareGPT should improve |

## Budget (the binding constraints)

| Resource | Plan |
|---|---|
| GH200 hours | **≈ 3 GPU-hours total**: one smoke job (≤ 15 min) + one measurement job (≤ 2 h), 1 GPU each, batch partition `ghx4` (charge 1.0; `ghx4-interactive` is 2.0). Never request a whole node (charged as 4 GPUs). Fitting and all simulation happen on the laptop. |
| DeltaAI disk | ≈ 30 GB, < 1,000 files, under `/work/hdd/bcjw/maenlle2/kvcal` (to be confirmed on DeltaAI; `/work/nvme/bcjw` is over its soft quota, home is 93/100 GB). Software in one container image (one file), not a pip venv (tens of thousands of files; the group is ≈ 400k files from its limit). Hugging Face cache in the same directory. |
| Laptop disk/memory | Only CSV/JSON results come back (≈ 10–50 MB). Validation simulations run ≤ 3 at a time. |

## What gets measured, and why

Two kinds of data, both written as the same step-log CSV columns
(`num_prefill, prefill_new_tokens, prefill_cached_tokens, num_decode,
decode_kv_tokens, measured_ms`):

1. **Workload replays in UniCache GPU mode** (`--disable-simulation --device cuda
   --simulator-step-log ... --simulator-timing aic --simulator-aic-system h100_sxm`).
   This is the same vLLM v0 engine the simulator runs, with real kernels and
   outputs forced to the trace's recorded tokens. So every step is the same
   batch the simulator would price, and `measured_ms` (wall time of
   `engine.step()`) includes the engine's own CPU overhead. Each run also writes
   `latency_requests.csv` for request-level validation. **This is the
   primary data.**
2. **A controlled grid** (`scripts/aic_timing/profile_steps_grid.py`): pure
   prefill batches over new-token count × cached-prefix length × batch size,
   and pure decode batches over batch size × context length. Fills corners
   that workloads rarely hit, such as large decode batches and long cached
   prefixes. It can run on upstream vLLM if the fork can't be built.

Workloads: the five H2 sim2real subsets (`scripts/sim2real_subset.py`, ≈ 100
requests each, 512 blocks, so they finish in minutes), with LRU and UniCache.
**Fit on CodeParrot, MASH-QA and ALFRED; hold out ShareGPT and APPS** for
validation.

Models: **Llama-3.2-1B** (continuity with H1/H2) and **Llama-3.1-8B** (where
prefill savings are large enough to matter for latency). Both have
`h100_sxm` tables.

## Steps

### P0. Setup on the DeltaAI login node (0 GPU-hours)

1. Log in to a fixed node (`ssh maenlle2@gh-login01.delta.ncsa.illinois.edu`)
   in tmux. Confirm the account (`accounts`), partitions (`sinfo`), quotas
   (`quota`) and that `/work/hdd/bcjw` is mounted.
2. `mkdir -p /work/hdd/bcjw/maenlle2/kvcal/{models,sif,results,hf}`; set
   `HF_HOME` there.
3. Download weights from ungated mirrors (`unsloth/Llama-3.2-1B-Instruct`,
   `unsloth/Llama-3.1-8B-Instruct`, safetensors only): ≈ 19 GB, ≈ 20 files.
4. **Build feasibility of the UniCache vLLM fork on aarch64. This is the main
   risk.** The fork is vLLM v0, pinned to torch 2.6. In order of preference:
   * (a) An arm64 PyTorch container with a matching torch, and the fork built
     inside it into an overlay or image (`VLLM_TARGET_DEVICE=cuda`, sm_90 only
     to cut compile time). Compiling needs nvcc, not a GPU, so it can run on
     the login node or a CPU-only job. Check NCSA's policy on heavy login-node
     builds first.
   * (b) If (a) fails: skip GPU mode. Measure only the grid on an upstream
     arm64 vLLM container, and validate request latency with a small client
     that replays the subsets against upstream vLLM. Weaker: the V1 engine
     schedules differently, so the request-level check becomes approximate.
5. Copy the repo (`experiment/aic-timing`) and the five subset inputs
   (`result/sim2real-inputs/`, small) to DeltaAI. Write one `sbatch` script per
   stage, each with hard `--time` limits and `--gpus-per-node=1`.

### P1. Smoke job (≤ 15 min, 1 GPU, batch partition)

`nvidia-smi`; container imports torch/vLLM and sees the GPU; `profile_steps_grid.py
--quick` (1B); one 20-session demo replay in GPU mode with `--simulator-step-log`.
Copy the CSVs back and check that columns parse and times are sane (1B decode
step a few ms). Stop here if anything is off; nothing else is queued.

### P2. Measurement job (≤ 2 h, 1 GPU, batch partition)

One job, run sequentially:
1. GPU-mode replays, 1B: 5 subsets × {LRU, UniCache} (~10 short runs).
2. GPU-mode replays, 8B: CodeParrot + MASH-QA × {LRU, UniCache}.
3. Grid for 1B and 8B (`--repeats 3`).

Every run writes to `results/<stage>/<model>/<workload>/<policy>/`. The job
`tar`s the CSV/JSON outputs into one file at the end (one inode).

### P3. Fit (laptop, 0 GPU-hours)

`scripts/aic_timing/fit_calibration.py --steps <fit-set>/**/steps.csv
--model <model> --system h100_sxm` → per-phase scale/overhead with a held-out
error report. Then look at the residuals by batch size, prefix length and
context length:
* If the residuals have no structure: done; the linear calibration stays.
* If they do (for example decode error growing with batch size): try
  aiconfigurator's bucketed `tune_with_fpms` correction on the same rows
  (`StepBatch.to_fpm(wall_time_s=...)` already produces its input) and keep
  whichever does better on held-out steps. This is the step that can be handed
  to an agent (Fable), ADRS-style: it sees the fit set and the objective,
  never the held-out replays.

### P4. Validate (laptop)

Re-simulate the held-out replays (ShareGPT, APPS; LRU and UniCache) with the
calibration file, then run `scripts/aic_timing/compare_latency.py --sim ...
--real ...` for TTFT/TPOT/E2E per request and step-level error. Also compare
hit ratios with the GPU runs: ShareGPT's H2 gap came from decode timing, so
this is a direct test of whether calibration fixes it.

### P5. Use it

Re-run the CodeParrot load sweep and the Table 5 reproduction with
`--simulator-aic-system h100_sxm --simulator-timing-calibration
calibrations/gh200-<model>.json` (laptop, ≤ 3 jobs), and update
`docs/aic-timing.md`.

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| Fork won't build on aarch64 / torch 2.6 | P0 option (b): upstream container, grid + client replays |
| Wall time of `engine.step()` misses async GPU work | Synchronize (`torch.cuda.synchronize()`) inside the timed region in GPU mode; check in P1 that decode steps match the grid |
| Small-model steps are noise-dominated | Drop steps < 0.5 ms (`--min-ms`), use medians from the grid |
| Queue wait on `ghx4` | Submit P2 immediately after P1 passes; nothing interactive |
| Calibration overfits the fit workloads | Held-out ShareGPT/APPS decide acceptance; report both |

## Decisions needed

1. DeltaAI account name (Delta's is `bcjw-delta-gpu`; DeltaAI allocations are
   separate) and whether `/work/hdd/bcjw` is the right place on DeltaAI.
2. Is building the fork on the login node acceptable, or should it be a
   CPU-only job?
3. Models: 1B + 8B as above, or 1B only (halves P2).
