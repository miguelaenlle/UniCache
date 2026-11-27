# Sim2real GPU run log (DeltaAI GH200)

Living log for the first GPU runs of the sim2real transfer test
([methodology](sim2real-transfer-methodology.md), [results](sim2real-results.md)).
Times are US Central. "Wait" = Slurm submit → start.

## Method (short)

1. **Build the container without a GPU.** The fork's CUDA code equals upstream vLLM
   0.8.5, and there is no aarch64 binary for 0.8.5, so it must be compiled. A trimmed
   image (`scripts/sim2real/delta/unicache-lite.def`) compiles only `vllm._C` without the
   Hopper-only quantization kernels, plus FlashAttention 2. It builds on the DeltaAI
   login node (16 CPUs, 16 GiB soft memory cap, 4 compile jobs): no queue, no GPU charge.
   FA2 instead of FA3 does not change hit ratios; use the full image for timing work.
2. **Every GPU job is small and short.** 1 GH200, 8 CPUs, 32 GB, interactive partition,
   ≤ 60 min. Priority ignores job size; what helps is fitting into gaps. In our group's
   last 7 days (641 jobs), 1-GPU interactive jobs of ≤ 60 min started in a median of
   0 min (p75 3–4 min); `ghx4` batch jobs waited 1.5–10 days.
3. **One interactive job at a time** (QOS limit), so jobs run back to back:
   15-min smoke test, then each trace split into ≤ 60-min jobs
   (`run_trace.sbatch <workload> <blocks> <runs>`). A run counts as done only when its
   `summary.json` exists, so a killed job resumes at the unfinished run.
4. **Record** for every step: submit, start, end, wait, run time, outcome.

## Expected vs actual (tonight, Nov 25–26)

Estimates are mine, not measured. Updated as steps finish. Check live with
`bash $K/bundle/status.sh` on the login node.

| Step | Expected | Actual |
|---|---|---|
| Build attempt 1: deps (apt, torch, pip) | ~10 min | 20:36 → 20:42 (6 min) |
| Build attempt 1: compile `_C` (37 objects) + FA2 (63), 4 jobs | 25–45 min | 20:42 → 21:13 (31 min), peak 15 GiB; both libraries built |
| Build attempt 1: image test | seconds | **failed** 21:13: my `%test` imported `vllm._C`, which needs the GPU driver (`libcuda.so.1`), absent on login nodes; Apptainer discarded the image. Test now only checks the files exist. |
| Build attempt 2 (same recipe, fixed test) | ~37 min + 5 min pack → ~21:55 | **OK** 21:13 → 21:52 (39 min: deps 5, compile 32, pack 2); image 6.7 GB |
| Smoke test attempt 1 (job 3680373) | wait 0–30 min + ~10 min run | waited 2 min 48 s; **failed** after 14 s: no Triton for aarch64, and vLLM's Triton placeholder had no `__spec__`, so torch's CUDA init raised `triton.__spec__ is None` |
| Fix: placeholder gets a `ModuleSpec`; patched image layered on top (`unicache-lite-patch.def`) | ~5 min | 21:56 → 21:59 (image build 1.5 min) → `unicache-lite-2.sif` |
| Smoke test attempt 2 (job 3680400) | wait 0–30 min + ~10 min run | waited 8 min 32 s; ran 56 s. GPU import OK (`NVIDIA GH200 120GB`), but every replay failed: `No module named 'pandas'`. vLLM's requirements lack the simulator's own deps. |
| Fix: found all missing deps with a static import scan inside the image (pandas, matplotlib, ijson); added them in the patch layer; checked replay imports on the login node | ~5 min | 22:10 → 22:13 → `unicache-lite-3.sif` |
| Smoke test attempt 3 (job 3680448) | wait 0–30 min + ~10 min run | waited 10 min 48 s; ran 71 s. Replays failed in vLLM's tokenizer wrapper: the image had transformers 5.24 (vLLM's requirements are lower bounds only); vLLM 0.8.5 needs 4.x. |
| Fix: pinned the 51 packages the image shares with `requirements/simulator-local.lock` to the lock's versions (`unicache-lite-pin.def`), then on the login node: tokenizer load, `pip check`, and a full CPU simulator replay inside the image (LRU 8.27 %, Agent 7.51 % via the `unshare` sandbox) | ~5 min | 22:26 → 22:29 → `unicache-lite-4.sif` |
| Smoke test attempt 4 (job 3680515) | wait 0–30 min + ~8 min run → ~22:40–23:10 | submitted 22:29 |
| Traces: ALFRED (1 job), CodeParrot (2), ShareGPT (2), ≤ 60 min each | ~3 h of runs + 5 waits → ~01:00–03:00 | |

### Overnight run (started 23:49)

Driver `scripts/sim2real/delta/overnight.sh` on the login node submits one job at a time
(1 GPU, 8 CPUs, 32 GB). Live log: `$K/logs/overnight.log`.

| Job | Contents | Limit | Expected run | Status |
|---|---|---:|---:|---|
| 1 `session.sbatch` | smoke test, then ALFRED (6 runs) only if it passes; on failure holds the GPU ≤ 20 min for a fix | 60 min | ~50 min | submitted 23:49 (job 3680737); still pending at 05:35 (5 h 46 min); at 04:52: priority rose to 1,320 from age, 55th of 62; 25 one-GPU interactive jobs started since 03:30); at 03:26: now 56th of 63 pending; about 30 interactive jobs started since 01:30, 1-GPU 45–60 min ones after 1.8–3.6 h), beyond the usual night p90 of 74 min for a 60-min job); at 00:55: 598/628 GPUs busy, our priority 1,226 ranks 80th of 83 pending interactive jobs, 10 interactive jobs started since 00:00 |
| 2 | CodeParrot: LRU, UniCache, LeCaR | 50 min | ~35 min | |
| 3 | CodeParrot: Agent, LRU ×2 | 50 min | ~35 min | |
| 4 | ShareGPT: LRU, UniCache, LeCaR | 50 min | ~35 min | |
| 5 | ShareGPT: Agent, LRU ×2 | 50 min | ~35 min | |

Expected finish ~05:00–08:00 if waits stay typical. A part that times out is resubmitted
(runs resume); a part whose runs fail stops the pipeline. GPU budget: ~4 GPU-h of runs
(~8 h charged at the interactive 2×).

Smoke attempt 4 (job 3680515) was cancelled at 23:49 after 80 min pending, replaced by job 1.

**06:20 — switched to one replay per job.** Job 1 (60 min) never started in 6 h 31 min.
Jobs that started between 02:00 and 06:19 (1 GPU, interactive), by limit:
≤ 15 min: 5 jobs, waits 0–4 min; 16–30 min: 12 jobs, half ≤ 8 min, the rest 43–477 min;
31–60 min: 23 jobs, more than half waited > 80 min. At our priority only short jobs fit the
gaps. New driver `overnight_short.sh`: smoke (15 min), then each of the 18 replays as its own
job (ALFRED 15 min; CodeParrot and ShareGPT 20 min), one at a time. Same GPU time, plus about
1 min of startup per job. Job 1 (3680737) cancelled; smoke submitted as job 3681975, started 06:38:53 on gh008 (wait 18 min 35 s).

**Smoke attempt 5 (job 3681975): timed out at 15 min, no replay ran.** vLLM creates a 1-process
`torch.distributed` group over TCP on the node's public IP (141.142.248.92); that connection
timed out after 10 min per replay. Fix: `VLLM_HOST_IP=127.0.0.1` (passed as
`APPTAINERENV_VLLM_HOST_IP` in `smoke.sbatch`, `run_trace.sbatch`, `first_trace.sbatch`); the same
setup on the login node inside the image connects in 0.01 s. Driver restarted 06:55; smoke
attempt 6 is job 3682007.

**Smoke attempt 6 (job 3682007): waited 34 min 10 s, ran ~1.5 min.** The engine now starts on the
GPU (process group in seconds, FlashAttention backend loads), then vLLM's sanity check refuses
`max_model_len` 131,072 > KV capacity 8,192 tokens (512 blocks × 16). The CPU simulator never
runs that check, and lowering `--max-model-len` would also lower the default
`max_num_batched_tokens` (131,072 → 8,192) on the GPU side only, changing scheduling versus the
pre-registered sim runs. Fix: opt-in bypass of that one check in `vllm/worker/worker.py`
(`UNICACHE_ALLOW_SMALL_KV_CACHE=1`, set in the job scripts; default still raises), built as layer
`unicache-lite-5.sif` (1.5 min). Driver restarted 07:36; smoke attempt 7 is job 3682108.

**Smoke attempt 7 (job 3682108): waited 53 min 45 s, started 08:30:28 on gh007. First GPU replay
completed:** CodeParrot-120, LRU, 512 blocks → hit ratio **8.27 %** (5,088 / 61,550 prompt
tokens), identical to the CPU simulator in the same image; TTFT mean 15.4 ms (p90 18.3), ITL mean
2.26 ms; replay 110 s of real time, ~2 min per run including engine start and CUDA-graph capture.

**Smoke test PASSED** (job ran 6 min 50 s):

| Run (CodeParrot-120, 512 blocks) | GPU hit ratio | CPU sim, same image |
|---|---:|---:|
| LRU | 8.266 % | 8.27 % |
| LRU, repeat | 8.266 % (identical) | — |
| Agent (via `unshare` sandbox) | 7.33 % | 7.51 % |

LRU is deterministic on the GPU and matches the simulator exactly; the agent policy differs by
0.18 pp (it uses timing, which differs between virtual and real time). Trace replays started
08:37 (ALFRED LRU, job 3682188).

### Trace replays

| Trace | Run | Job | Wait | Ran | GPU hit ratio | Sim (pre-registered) |
|---|---|---|---:|---:|---:|---:|
| ALFRED (512 blocks) | LRU | 3682188 | 39 min 16 s | 6 min 15 s | **34.91 %** (evict 18,777, re-missed 14,923, replay 321 s) | 33.45 % (evict 19,158, re-missed 15,304) |
| ALFRED | UniCache | 3682294 | 38 s | 5 min 47 s | **34.73 %** (evict 18,829, re-missed 14,975) | 33.45 % |
| ALFRED | LeCaR | 3682312 | 55 s | 6 min 3 s | **37.47 %** (evict 18,207, re-missed 14,353) | 35.34 % |
| ALFRED | Agent | 3682323 | 29 min 3 s | 6 min 5 s | **37.58 %** (evict 18,723, re-missed 14,869) | 35.25 % |
| ALFRED | LRU repeat 2 | 3682423 | ~13 min | ~6 min | **34.95 %** (first run 34.91: spread 0.04 pp) | |
| — | grid (step timing, batches 1–32) | 3682481 | submitted 10:31 | | dropped (see Step 1) | |
| CodeParrot (4,096 blocks) | LRU | 3682598 | 34 min 25 s | 10 min 30 s | **15.88 %** (evict 18,680, re-missed 2,975, replay 576 s) | 16.05 % |
| CodeParrot | UniCache | 3682707 → 3683222 | cancelled for the sweep; re-run 13:53 → 14:30 (37 min) | ~10.5 min | **18.93 %** (evict 18,094, re-missed 2,384) | 18.89 % |
| CodeParrot | LeCaR | 3683324 | 2 min 9 s | 10 min 19 s | **17.89 %** (evict 18,359, re-missed 2,654) | 18.19 % |
| CodeParrot | Agent | 3683352 → 3683969 | cancelled for step 3; re-run 18:25 → 21:21 (2 h 56 min, gh032) | 10 min 23 s | **18.58 %** (evict 18,161, re-missed 2,451) | 18.54 % |

ALFRED verdicts with the noise floor (0.04 pp, so tolerance 0.5 pp), GPU − LRU vs sim − LRU:
UniCache −0.18 vs +0.00 (legacy) → transfers; LeCaR +2.56 vs +1.89 → **does not** (0.67 pp);
Agent +2.66 vs +1.80 → **does not** (0.86 pp). Signs all agree. With the calibrated `aic`
timing the sim deltas are +0.00 / +2.37 / +3.14, all within 0.5 pp (in-sample; see Step 1).

SSH dropped 09:42–10:03 (the driver kept submitting). Provisional ALFRED transfer, before the
LRU repeats set the noise floor (tolerance 0.5 pp until then):
UniCache − LRU: sim +0.00, GPU −0.18 → transfers; LeCaR − LRU: sim +1.89, GPU +2.56 → same sign,
0.67 pp apart (over 0.5 pp; may fall within 2 × noise); Agent − LRU: sim +1.80, GPU +2.66 →
same sign, 0.86 pp apart; Agent − LeCaR: sim −0.09, GPU +0.10 (sign flips, both near zero).
Generated with
`scripts/sim2real/compare_transfer.py --gpu result/sim2real-transfer/gpu`.

Earlier check, 23:34: smoke attempt 4 (job 3680515) still pending after 65 min (priority 1,233; 77 jobs pending in the interactive partition). Much slower than the 3–11 min waits earlier tonight. GPU time used so far: 15.4 min (0.26 GPU-h; ~0.5 h charged at 2×), 85 % of it the OOM build.

## Timeline

| # | Step | Job | Request | Submitted | Wait | Ran | Outcome |
|---|---|---|---|---|---:|---:|---|
| 1 | Full build, login node, 16 compile jobs | — | login node | Nov 25, start not logged | — | until 16:29 | stalled at the 16 GiB memory throttle; stopped |
| 2 | Full build, `ghx4` batch | 3679126 | 72 CPU, 110 GB, 2 h | 16:30 | never started (26 min) | — | cancelled: `ghx4` estimate was days |
| 3 | Full build, interactive | 3679233 | 72 CPU, 110 GB, 2 h → 50 min | 16:57 | never started (44 min) | — | cancelled to free the slot |
| 4 | GPU probe (`nvidia-smi`) | 3679445 | 8 CPU, 32 GB, 15 min | 17:41 | 28 min 43 s | 4 s | OK, GH200 visible (gh152) |
| 5 | Full build, interactive | 3679545 | 72 CPU, 110 GB, 50 min | 18:10 | 12 min 17 s | 12 min 58 s | OOM-killed: 24 compile jobs in 110 GB (machete kernels), at step 89/315 |
| 6 | Full build, interactive | 3679701 | 72 CPU, 220 GB, 80 min | 18:37 | never started (1 h 58 min) | — | cancelled: moved the build to the login node |
| 7 | Trimmed build, login node, 4 compile jobs | — | login node | 20:36 | 0 | 37 min | compiled OK; image discarded by a test that needs a GPU driver |
| 8 | Trimmed build, attempt 2 | — | login node | 21:13 | 0 | 39 min | OK, `sif/unicache-lite.sif` (6.7 GB) |
| 9 | Smoke test | 3680373 | 8 CPU, 32 GB, 15 min | 21:52 | 2 min 48 s | 14 s | failed: `triton.__spec__ is None` at torch CUDA init (fixed in `vllm/triton_utils/importing.py`) |
| 10 | Patched image layer | — | login node | 21:58 | 0 | 1.5 min | OK, `sif/unicache-lite-2.sif` |
| 11 | Smoke test | 3680400 | 8 CPU, 32 GB, 15 min | 22:00 | 8 min 32 s | 56 s | GPU import OK; replays failed: no `pandas` in the image |
| 12 | Patch layer + simulator deps | — | login node | 22:10 | 0 | 1.5 min | OK, `sif/unicache-lite-3.sif` |
| 13 | Smoke test | 3680448 | 8 CPU, 32 GB, 15 min | 22:13 | 10 min 48 s | 71 s | replays failed: transformers 5.x in the image |
| 14 | Pin layer + CPU checks on login node | — | login node | 22:26 | 0 | 2 min + 1 min | OK, `sif/unicache-lite-4.sif`; CPU replay works in the image |
| 15 | Smoke test | 3680515 | 8 CPU, 32 GB, 15 min | 22:29 | | | |

## Step 1: timing calibration (started 2026-11-26 10:20)

**v0 fit** (`calibrations/gh200-llama-3.2-1b-v0.json`) from the GPU step logs of ALFRED (4 runs)
and the CodeParrot-120 smoke test (3 runs), on aiconfigurator's `h100_sxm` tables
(`scripts/aic_timing/fit_calibration.py`):

| Phase | Fit: measured ≈ scale × table + overhead | Held-out median error, before → after | Data covers |
|---|---|---|---|
| Prefill | 0.664 × table + 9.45 ms | 79 % → 14 % | ≤ 2 prompts per step |
| Decode | constant 2.23 ms (scale clamped to 0) | 34 % → 1.4 % | ≤ 4 sequences per step |

The first decode fit had a negative scale (−0.57): at this load, decode steps are a flat
~2.2–2.6 ms of per-step overhead. `fit_calibration.py` now clamps negative scales to 0 (constant
step time) and records the data range each fit covers. These runs are too lightly loaded for
TTFT-vs-load work, so a controlled grid job (`grid.sbatch`: prefill and decode batches of 1–32)
was added to the GPU queue ahead of the remaining replays (driver v3, 10:22).

### First check: calibrated timing vs GPU (ALFRED, in-sample traces)

The v0 calibration was fitted on these same runs' step times (not their hit ratios), so this is
in-sample at the trace level; CodeParrot-600 and ShareGPT are the out-of-sample check.

Hit ratio (%), ALFRED 512 blocks:

| Policy | Sim, legacy timing | Sim, calibrated `aic` | GPU |
|---|---:|---:|---:|
| LRU | 33.45 | 34.80 | 34.91 |
| UniCache | 33.45 | 34.80 | 34.73 |
| LeCaR | 35.34 | 37.17 | 37.47 |
| Agent | 35.25 | 37.94 | 37.58 |

CodeParrot-120: LRU 8.27 in all three; Agent 7.51 (legacy) → 7.33 (aic) = 7.33 (GPU).
Policy − LRU with calibrated timing: UniCache +0.00 (GPU −0.18), LeCaR +2.37 (GPU +2.56),
Agent +3.14 (GPU +2.66): all within 0.5 pp. Most of the legacy-timing gap was the timing model.

Latency, per-request match (`scripts/aic_timing/compare_latency.py`, 713 requests):

| ALFRED | GPU TTFT mean | Sim legacy | Sim aic-cal | TPOT GPU / aic-cal | median TTFT error, legacy → aic-cal |
|---|---:|---:|---:|---|---|
| LRU | 14.3 ms | 33.9 ms | 11.6 ms | 2.4 / 2.4 ms | 119 % → 16 % |
| LeCaR | 15.2 ms | 33.9 ms | 11.6 ms | 2.4 / 2.4 ms | 106 % → 22 % |
| Agent | 16.4 ms | 33.9 ms | 11.5 ms | 2.4 / 2.4 ms | 92 % → 28 % |

**Finding: on the GPU, TTFT rises with hit ratio, because eviction-policy compute is on the
critical path.** GPU prefill steps: LRU 10.1 ms mean (424 new tokens/step), UniCache 11.5,
LeCaR 10.9 (407), Agent 11.9 (407; the agent's policy runs in a sandboxed process, one IPC round
trip per decision). Saving ~4 % of prefill tokens on a 1B model is worth less than the policy's
CPU time. The simulator charges no time for policy decisions, so a TTFT objective in simulation
would reward hit ratio while the GPU penalizes heavier policies. This has to be modeled before
step 3.

### v1 calibration: charge policy CPU time (provisional)

The simulator now records each engine step's own CPU time (`cpu_ms` in `steps.csv`) and, under
`aic`, can charge `cpu_scale × cpu_ms` on its virtual clock (opt-in via the calibration's
`cpu_scale`; absent = unchanged). `scripts/aic_timing/fit_cpu_charge.py` fits it from paired
GPU/sim runs of the same trace and policy (6 pairs: ALFRED ×4, CodeParrot-120 ×2):

| Phase | v1 fit | Step median error | Per-run mean error |
|---|---|---:|---:|
| Prefill | 0.686 × table + 8.43 ms + **2.08 × sim CPU ms** | 13.6 % | 0.38 ms |
| Decode | 2.23 ms constant (no policy effect) | 1.4 % | 0.02 ms |

Sim CPU per prefill step on the laptop: LRU/UniCache 0.27–0.28 ms, LeCaR 0.39, Agent 0.85–0.88
(sandbox IPC). The GPU pays about 2× that. Caveats: single GPU runs on different nodes vary by
about ±1 ms per prefill step (UniCache vs LRU), so `cpu_scale` is noisy; and charging measured
CPU time makes simulated timing depend on the simulator machine's load, so runs that use it
should go one at a time. `calibrations/gh200-llama-3.2-1b-v1.json`.

### Load sweep (simulator, v1 calibration), CodeParrot-600, 4,096 blocks

Arrival scale multiplies first-turn arrival times (0.03 = 33× the recorded rate). Hit % / mean TTFT:

| Load | LRU | UniCache | Agent |
|---:|---|---|---|
| 1× | 15.9 / 11.3 ms | 18.9 / 11.3 ms | 18.6 / 12.6 ms |
| 3.3× | 16.1 / 11.8 ms | 18.9 / 11.8 ms | 18.6 / 12.8 ms |
| 10× | 16.1 / 12.3 ms | 18.9 / 12.4 ms | 18.6 / 13.7 ms |
| 20× | 15.9 / 13.0 ms | 18.8 / 12.9 ms | 18.5 / 14.1 ms |
| 33× | 15.4 / 21.8 ms (p90 51) | 18.4 / 23.3 ms (p90 53) | 18.2 / 30.7 ms (p90 97) |

A 1B model on a GH200 has so little prefill to save that UniCache's +3 pp hit ratio is worth
about 0 ms, and the agent's sandboxed policy costs more CPU than it saves. Queueing starts near
33×. TTFT baselines on the GPU therefore use 20× (no queueing) and 33× (queueing).

### GPU jobs for the timing calibration

| Job | Contents | Status |
|---|---|---|
| 3682481 grid | step-time grid | failed in 21 s: `max_num_batched_tokens` 32,768 < `max_model_len` 131,072; fixed (budget 131,072, as in the replays) |
| 3682519 grid | step-time grid | failed in 18 s: `policy_agent` not importable (script ran from the bundle dir); fixed with `PYTHONPATH` |
| 3682591 grid | step-time grid | failed in 21 s: this fork's engine needs simulator conversation keys as request ids, so plain `LLM.generate` cannot run. **Grid dropped**: the loaded replays (20×, 33×) already cover decode batches 1–32 |
| next | `ttft_baseline.sbatch`: CodeParrot-600 at 20× and 33×, LRU ×3 + UniCache ×2 each, one node, 30 min | queued after the grid by driver v4 |

Driver v4 (10:42) runs the grid, the TTFT baselines, then the remaining sim2real replays. The
CodeParrot LRU replay (3682513) was cancelled while running to put the TTFT job first; it is
re-run later.

### TTFT baselines on the GPU (job 3682522, gh152, 10:44–10:52)

CodeParrot-600, 4,096 blocks, all 10 replays on one node:

| Load | Policy | GPU hit % | GPU TTFT mean (p90) | GPU ITL |
|---|---|---:|---:|---:|
| 20× | LRU ×3 | 15.74 ± 0.01 | 22.5 ms (28.0), runs 22.0–23.0 | 4.37 ms |
| 20× | UniCache ×2 | 18.49 | 26.5 ms (30.5) | 4.53 ms |
| 33× | LRU ×3 | 15.34 | 1,815 ms (3,270), runs 1,612–1,955 | 6.18 ms |
| 33× | UniCache ×2 | 17.77 | 1,934 ms (3,560) | 6.26 ms |

GPU decode steps grow with batch: 2.28 ms (1–4 sequences) → 3.88 ms (25–32). Prefill is almost
flat: 10.8 ms + 0.6 ms per 1,000 new tokens (single-prompt steps), so a cache hit saves very little
on a 1B model. UniCache is slower than LRU on the GPU (+4 ms at 20×) although its laptop CPU time
per step equals LRU's; not explained yet.

### Calibration versions vs the GPU baselines (CodeParrot-600; 33× held out of every fit)

| Version | What changed | 20× TTFT err | 20× ITL err | 33× TTFT err | 33× ITL err |
|---|---|---:|---:|---:|---:|
| v1 | light-load data only, decode constant 2.23 ms | −42 % | −35 % | −99 % (no saturation) | −45 % |
| v2 | + 20× runs; CPU term in both phases | +60 % | +16 % | +77 % | +18 % |
| v3 | decode without CPU term | −35 % | −15 % | −14 % | −2 % |
| **v4** | decode fit on the 20× runs only | **−32 %** | **−4 %** | **−23 %** | **−3 %** |

(LRU shown; UniCache is similar except its GPU penalty.) v4 hit ratios match the GPU within 0.06 pp
at both loads. Remaining gap: about 7 ms of per-request TTFT outside engine steps at 20×.
`calibrations/gh200-llama-3.2-1b-v4.json`: prefill 0.632 × table + 8.73 ms + 1.86 × sim CPU;
decode 6.61 × table − 6.85 ms.

**Load for step 3: 25× (arrival scale 0.04).** Simulator v4: LRU 72.2 ms TTFT, UniCache 68.7
(−5 %), LeCaR 73.4, H1 agent policy 103.0 (+43 %, sandbox CPU cost).

## Step 3: end-to-end agent run (started 11:04, relaunched 11:12)

`./policy-agent run --trace codeparrot --start-policy LRU --compare LRU,UniCache,LeCaR
--max-submissions 30 --slice-sessions 600 --gpu-blocks 4096 --block-size 16 --objective ttft
--timing aic --timing-calibration calibrations/gh200-llama-3.2-1b-v4.json --arrival-scale 0.04`
(log `result/sim2real-transfer/agent-ttft-run.log`). The eval slice is exactly `codeparrot@600`,
the GPU subset; the tune slice is 600 disjoint sessions.

First launch stopped at the starting-controller parity check: it compared whole score dicts,
which now include TTFT. Fixed to compare the deterministic cache fields only. Baselines (25×):

| Policy | Tune hit % / TTFT | Eval hit % / TTFT |
|---|---|---|
| LRU (built-in) | 14.50 / 63.9 ms | 15.44 / 67.9 ms |
| LRU via the agent's sandboxed controller | 14.50 / 93.5 ms | 15.44 / 92.5 ms |
| UniCache | 15.62 / 65.5 ms | 18.20 / 68.4 ms |
| LeCaR | 15.27 / 63.2 ms | 17.70 / 71.9 ms |

Same decisions, but the sandbox IPC costs ~30 ms (~45 %) of TTFT, so agent policies start behind
the built-in ones. Step 3 is reported against both: built-in baselines (deployment) and the
sandboxed LRU (decision quality at equal overhead).

**Objective noise.** The unchanged starting policy scored tune TTFT 93.5 ms in the first launch and
104.1 ms in the second; submission 2 scored 143.6 ms tune but 87.0 ms eval. The CPU term charges
measured wall time on the laptop, which also runs the coding agent, so the simulated TTFT objective
carries roughly ±10–40 % noise. Large effects survive (submission 1's O(n) scan: 3.1 s), small
ones do not. Mitigation: GPU validation replays the top 3 submissions by tune TTFT (not only the
selected one), plus LRU ×3 and UniCache ×2, on one node at 25×. A deterministic CPU cost (per
policy call) instead of measured wall time is the fix for a later version.

**Outcome (11:44): the agent stopped itself after 5 submissions; selected = submission 0 (the
unchanged starting LRU).** Its stop reason, paraphrased: every eviction order it tried
(decode-first, soft frequency, LRU-equivalent heap) gave the same 44,816 tune hit tokens, hard SLRU
lowered it, and TTFT differences between runs with identical decisions were wall-clock noise
(104–144 ms), so further attempts would only sample noise. It did not find UniCache's tune gain
(15.72 % vs LRU 14.50 %). Step 3 as run therefore has no agent improvement to validate on the GPU.
Contributing causes: (1) the objective's ±40 % noise from measured CPU time; (2) on a 1B model at
25× load, +1–4 pp of hit ratio is worth about 5 % of TTFT in the simulator, below that noise;
(3) the agent's sandbox overhead (+45 % TTFT) dwarfs any decision-quality gain against built-in
policies. Related result: the H1 agent policy, optimized for hit ratio, is +43 % worse on TTFT.

### Step 3 rerun with v8 timing (agent run 15:11–15:20; results in `docs/step3-results.md`)

| Job | Request | Submitted | Wait | Ran | Outcome |
|---|---|---|---:|---:|---|
| 3683420, all 12 GPU replays | 8 CPU, 32 GB, 30 min, interactive | 15:21 | never started (43 min) | — | cancelled; split into two 15-min halves |
| 3683573, half A (LRU, agent c1, LFU, UniCache, agent c2, LRU) | 8 CPU, 32 GB, 15 min, interactive | 16:05 | 2 h 9 min | 4 min 50 s (gh152) | OK, 6 replays × 24 s |
| 3683941, half B (FIFO, agent c1, LeCaR, agent c3, UniCache, LRU) | same | 18:20 | 40 s | 4 min 39 s (gh152) | OK, 6 replays × 24 s |

Result: agent +45 % TTFT vs built-in LRU on the GH200 (simulator +46 %); hit ratios within 0.4 pp; LRU run-to-run TTFT spread 16.7 ms. Full tables in `docs/step3-results.md`. The driver then resumed the sim2real queue (CodeParrot Agent, job 3683969).


## Simulator tuning summary (final: v5, 2026-11-26 ~12:20)

**Before** = H1's untuned simulator (legacy timing: prefill `3.9e-05 · batch^0.997 · max_len^1.014` s,
decode 10 ms/step). **After** = `--simulator-timing aic --simulator-aic-system h100_sxm
--simulator-timing-calibration calibrations/gh200-llama-3.2-1b-v5.json`. GPU = GH200 measurements.

| Metric | GPU | Before | After (v5) |
|---|---:|---:|---:|
| ALFRED hit %, LRU (light load) | 34.91 | 33.45 | 34.81 |
| ALFRED hit %, H1 agent policy | 37.58 | 35.25 | 36.30 |
| CodeParrot-600 hit %, LRU, 20× | 15.74 | 15.04 | 15.75 |
| CodeParrot-600 hit %, UniCache, 20× | 18.49 | 17.54 | 18.53 |
| CodeParrot-600 hit %, LRU, 33× | 15.34 | 15.30 | 15.30 |
| ALFRED TTFT mean, LRU (light) | 14.3 ms | 33.9 ms | 11.4 ms |
| ALFRED ITL mean, LRU (light) | 2.37 ms | 11.40 ms | 2.85 ms |
| CodeParrot-600 TTFT mean, LRU, 20× | 22.5 ms | 8,364 ms | 14.8 ms |
| CodeParrot-600 ITL mean, LRU, 20× | 4.37 ms | 13.84 ms | 4.15 ms |
| CodeParrot-600 TTFT mean, LRU, 33× (held out) | 1,815 ms | 13,898 ms | 1,405 ms |
| CodeParrot-600 ITL mean, LRU, 33× (held out) | 6.18 ms | 13.89 ms | 5.99 ms |
| Agent − LRU TTFT, ALFRED (policy cost) | +2.1 ms | 0 (all policies equal) | +1.5 ms |
| Saturation (TTFT blow-up) | between 20× and 33× | below 20× | between 20× and 33× |
| Same policy, two runs, TTFT | — | identical | LRU identical; sandboxed agent ±1.4 % |

Numbers changed (`calibrations/gh200-llama-3.2-1b-v5.json`, read by `Calibration` in
`vllm/simulator/step_timing.py`, applied in `vllm/simulator/simulator.py` when the virtual clock
advances by each engine step's duration):

| Parameter | Before | After | What it does |
|---|---|---|---|
| timing model | legacy formula | aiconfigurator `h100_sxm` kernel tables, Llama-3.2-1B shapes | base GPU time per step from the step's token counts |
| `scale.prefill` | (1.0) | 0.632 | prefill step = 0.632 × table + overhead (GH200 prefill faster than the table) |
| `overhead_ms.prefill` | (0) | 9.25 ms | fixed per-prefill-step cost (scheduler, Python, launches); ~95 % of a 1B prefill step |
| `scale.decode` | (1.0) | 6.61 | decode step = 6.61 × table − 6.85 ms: ~2.3 ms at 1–4 sequences, ~3.9 ms at 25–32 |
| `overhead_ms.decode` | (0) | −6.85 ms | (with the scale) the fitted line through the loaded runs' decode steps |
| `policy_call_ms` | 0 | 0.05 ms | charge per IPC round trip to a sandboxed agent policy (~25 per prefill step) |
| `policy_cpu_scale` | 0 | 1.0 | charge the policy worker's own process CPU time as is |
| `cpu_scale` | 0 | 0 | measured-wall-time CPU charge; tried in v1–v4, removed (±40 % noise) |

Fit data: GPU step logs of ALFRED ×4 and CodeParrot-120 ×3 (light load), CodeParrot-600 at 20× ×5;
33× held out. Remaining error: TTFT 20–35 % low (light load and 20×; not in step times, which
match), ITL +20 % at light load (decode line is fitted to loaded batches), agent policy hit ratio
1.3 pp low on ALFRED.

## Calibration v6: fit on a synthetic sweep only (started 12:06)

Rationale: fitting on controlled synthetic workloads and validating on the real traces separates
the effects that traces confound (batch size, context, hits, load all move together) and keeps
every trace out of the fit, so the trace comparison is a true held-out test.
`scripts/sim2real/make_sweep.py` writes 20 configs (prompts of single-token words, so lengths and
shared prefixes are exact): load sweep (512/128 tokens, 2–40 req/s), isolated prefills
(128–8,192 tokens; 2,048 with 50 % and 90 % shared prefix), decode bursts (batch 1/8/16/32 ×
context 256/2,048), and an eviction config (512 blocks) with built-in LRU and the same LRU via the
agent sandbox (policy IPC cost). Local simulator check: decode batch 16 reached, 90 % share gives
1,840 cached tokens, sandboxed LRU makes identical decisions with 2,728 IPC calls.
GPU job `sweep.sbatch` (21 replays, one node), job 3682822, queued 12:06 ahead of the remaining
sim2real replays (driver v5; the pending CodeParrot UniCache replay was cancelled and re-runs later).
It waited 65 min without starting (40 min limit, cut to 30 at 12:49) while 597/628 GPUs were busy
and ~20 jobs of 25–31 min from higher-priority accounts started within a minute. 13:11: split into
three ≤ 15-min jobs (`sweep.sbatch prefill|decode|load`, driver v6); prefill + policy pair first
(job 3683040). The parts may land on different nodes; each part's internal comparisons stay on one.

### v6 result (14:20): fit on the sweep only, validated on held-out traces

Sweep parts ran 13:35–13:53 (waits 23 min, 24 s, 25 s; 17 GPU-min). v6
(`calibrations/gh200-llama-3.2-1b-v6.json`): prefill 1.416 × table + 4.75 ms, decode
3.10 × table − 2.15 ms, 0.032 ms per policy IPC call; held-out sweep-step error 5.7 % / 8.0 %.
Full tables: `result/sim2real-transfer/timing-before-after.md`. Summary vs the GPU:

| Metric | Untuned | v5 (fit on traces) | v6 (fit on sweep only) |
|---|---:|---:|---:|
| Hit ratio, mean abs diff (9 conditions) | 0.65 pp | 0.17 pp | 0.34 pp |
| ITL, light load (5) | +350 % | +20 % | +1 to +5 % |
| ITL, 20× and 33× (4) | +120 to +220 % | −3 to −8 % | −32 to −37 % |
| TTFT, light load | +58 to +137 % | −20 to −25 % | −39 to −41 % |
| TTFT, 33× | +620 % | −23 to −28 % | −97 % (no saturation) |

Why v6 misses under load: per-step costs the isolated sweep configs did not trigger. On sweep
steps, v6 is within 1–3 % except (1) the eviction config (512 blocks): prefill steps 17 % slower
than predicted (~1.6 ms per prefill step, ~0.09 ms per eviction, block-manager CPU), and (2) 40 req/s
(queue deeper than the 32 running slots): decode steps 19 % slower. The traces have both
(small caches with ~18,700 evictions per CodeParrot run; queueing at 20×/33×). Trace steps under
v6: prefill −16 to −26 %, decode −18 to −26 %; under v5: about ±1–8 %. The two-parameter-per-phase
form cannot represent eviction or queue-depth costs, so v5 absorbed them implicitly by fitting on
the traces.

### v7 and v8: still fit on the sweep only (15:00–15:45) — **v8 is the final calibration**

What the sweep showed (no trace data used):
- H100 tables cannot follow GH200 decode across batch sizes (their decode estimate is nearly flat
  for a 1B model), so v7 replaces `scale × table + overhead` with a **direct step model** fitted on
  sweep steps: prefill = 7.25 + 0.93 × prompts + 1.16 × new/1k + 0.49 × (new/1k)² ms (fit on steps with
  ≤ 4 prompts, the regime real traces use; the burst configs' 8–32-prompt prefill batches skewed an
  unrestricted fit); decode = 2.19 + 0.035 × sequences + 0.020 × context/1k ms.
- **Eviction cost** 0.057 ms per evicted block (sweep eviction config, prefill excess over the base
  model divided by the simulator's 15.65 evictions per prefill step).
- Queue depth was not the cause of the 40 req/s decode slowdown (simulated decode steps rarely have
  a queue); decode time grows smoothly with batch (no CUDA-graph steps), matching bursts at equal
  batch and context.
- **Per-request arrival cost** (v8): on the GPU path the engine tokenizes each prompt in its own
  thread when the request arrives. Isolated-prefill sweep configs: GPU TTFT − prefill step =
  2.9 ms (128 tokens), 3.6 (512), 5.5–6.0 (2,048 at 0/50/90 % cached), 21.3 (8,192); depends on total
  prompt length, not cache hits. Fit 2.84 + 1.14 × n/1k + 0.136 × (n/1k)² ms (residuals ≤ 0.25 ms),
  charged on the virtual clock when the request is released.

Code: `Calibration.linear`, `evict_ms`, `request_ms` (`vllm/simulator/step_timing.py`), charged in
`vllm/simulator/simulator.py`; step logs gain `evictions`, `num_waiting`.

Held-out validation (9 trace conditions; full tables in `result/sim2real-transfer/timing-before-after.md`):

| Metric | Untuned | v5 (fit on traces) | v6 | v7 | **v8 (sweep only)** |
|---|---:|---:|---:|---:|---:|
| Hit ratio, mean abs diff | 0.65 pp | 0.17 | 0.34 | 0.21 | **0.12** |
| ITL, median abs error | 348 % | 19 % | 5 % | 1 % (−13 to −17 % loaded) | **3 %** (1–7 % everywhere) |
| TTFT, median abs error | 137 % | 25 % | 54 % | ~43 % | **13 %** |
| TTFT light load | +58 to +137 % | −20 to −25 % | −39 to −41 % | −25 to −30 % | **−8 to +1 %** |
| TTFT 20× | ×300+ | −34 to −44 % | −54 to −61 % | −43 to −53 % | **−14 to −28 %** |
| TTFT 33× | +620 % | −23 to −28 % | −97 % | −66 to −69 % | **−13 to −19 %** |

Outlier: CodeParrot-600 at 1× (v8 −40 %): the GPU TTFT there (23.4 ms, one replay, gh117) is
higher than the same trace at 20× (22.5 ms); recheck when that configuration reruns.
`calibrations/gh200-llama-3.2-1b-v8.json` is the calibration for step 3.

## Step 2: TTFT objective in the agent loop (code done 2026-11-26 ~10:45)

`policy_agent/runner.py` gains opt-in options; with none of them set, behaviour is unchanged
(existing unit tests pass):

| Option | Effect |
|---|---|
| `--objective {hit_ratio,ttft}` | prompt goal, feedback text and best-tune selection (ttft: lowest mean TTFT) |
| `--timing {legacy,aic}`, `--aic-system`, `--timing-calibration` | simulator step-time model; `ttft` requires `aic` |
| `--block-size` | KV block size (sim2real GPU runs use 16) |
| `--arrival-scale` | multiplies first-turn arrivals in run-local copies of the split configs (shared splits untouched) |

Every score now also records TTFT, ITL and E2E (mean, p90) from `latency_summary.json`. Checked on a
20-session CodeParrot slice (512 blocks, 16-token blocks, arrival scale 0.1, calibrated `aic`):
LRU tune TTFT 12.42 ms, hit ratio 69.9 %, the command carries the timing flags, and the shared
split files are byte-identical afterwards. `scripts/sim2real/common.py` takes extra replay flags
from `SIM2REAL_EXTRA_FLAGS` for the timing checks.

## Next (agreed 2026-11-26)

These runs (untuned legacy timing, H1 Table 2 agent policies from
`agent-cache-budget-study/manual-20261101-162735`) are the baseline. Then:
1. **Test TTFT/ITL estimation** in the simulator against these GPU runs, and tune the timing
   tables as needed. The aiconfigurator path on this branch (`--simulator-timing aic`,
   `docs/aic-timing.md`) has no GH200 table yet. Every GPU run here already writes per-step
   timings (`steps.csv`) and per-request TTFT/ITL, which is the calibration data.
2. **Switch the agent's objective** from KV hit ratio to TTFT.
3. **End-to-end loop:** the agent develops a policy on a trace slice, its policy is replayed on
   the GH200, and the TTFT result is quantified.
4. (open)

## Notes

- The SSH control connection dropped once (19:07) and needed a new password + Duo
  login. Background scripts on the login node (`setsid nohup`) survived it.
- Requests above ~113 GB of memory raise the charge (`billing` 2000 → 3967) and,
  judging from step 6, make a job much harder to place.

## GPU spending audit (2026-11-26 ~19:00, from `sacct`, all jobs since Nov 25)

Every job used 1 GPU on `ghx4-interactive`, billed at 2,000 per GPU-hour (2× the `ghx4` batch rate).
Billing is the maximum over the requested resources, so 8 CPUs / 32 GB (or even 72 CPUs / 110 GB) costs
the same as the GPU alone.

| Category | Jobs | Elapsed (s) | GPU-hours | Charged (2×) |
|---|---:|---:|---:|---:|
| Results: trace replays, TTFT baselines, calibration sweep, step 3 | 15 | 6,082 | 1.69 | 3.38 |
| Probe and working smoke tests | 3 | 489 | 0.14 | 0.27 |
| Debug smoke tests that found real bugs (pandas, transformers) | 2 | 127 | 0.04 | 0.07 |
| **Waste**: build on a GPU node, OOM-killed (3679545) | 1 | 778 | 0.22 | 0.43 |
| **Waste**: smoke test hung until its time limit (3681975, distributed init before the `VLLM_HOST_IP` fix) | 1 | 928 | 0.26 | 0.52 |
| **Waste**: quick failures (Triton, grid ×3, accidental cancel) | 5 | 77 | 0.02 | 0.04 |
| **Total** | 27 | 8,481 | **2.36** | **4.71** |

Waste: 0.50 GPU-hours (0.99 charged), 21 % of the total. Project balance at audit time: 37,982 of
50,000 hours (group-wide). Remaining queue (CodeParrot Agent, LRU ×2, LRU ×3, ShareGPT ×6): about
1.5 GPU-hours (3 charged). Exposure: the driver resubmits a timed-out part up to 3 times, so a hang
could burn up to 1 hour on one part.

20:37: job 3683969 (CodeParrot Agent) had pended 2 h 11 min at a 20-min limit. Lowered it to 15 min with
`scontrol update` (earlier CodeParrot 1× jobs ran 10.3–10.5 min), since ≤ 15-min jobs backfill sooner.

21:32: CodeParrot Agent on the GPU 18.58 % (sim 18.54 %). Agent − LRU gap: GPU +2.70 pp, simulator +2.49 pp (difference 0.21 pp, within the 0.5 pp tolerance; same sign): transfers. LRU ×2 (job 3685740) submitted, limit lowered to 15 min.

22:38: CodeParrot LRU rep 2 on the GPU (job 3685740, gh152, waited 55 min, ran 10 min 23 s): 15.89 % (rep 1: 15.88 %); LRU noise so far 0.01 pp. LRU ×3 (job 3685977) submitted, limit lowered to 15 min.

22:50: CodeParrot LRU rep 3 (job 3685977, gh151, waited 1.5 min, ran 10 min 26 s): 15.89 %. LRU ×3 on the GPU: 15.877 / 15.887 / 15.887 % (spread 0.01 pp; reps 2 and 3 on gh152 and gh151). CodeParrot sim2real complete. ShareGPT LRU (job 3686013) submitted at 20 min; kept at 20 min because ShareGPT's simulated makespan is 565 s and its decode timing was the H2 outlier, so the GPU replay may run longer.

23:00: ShareGPT LRU on the GPU (job 3686013, waited 21 s, ran 10 min 9 s, replay 564 s): 11.63 % (sim 11.66 %). UniCache (job 3686040) submitted, limit lowered to 15 min (replay fits with 4 min to spare).

23:12: ShareGPT UniCache on the GPU (job 3686040, gh151, waited 37 s, ran 10 min 9 s): 11.63 %, identical to LRU (sim gap 0.00 pp, GPU 0.00 pp). LeCaR (job 3686067) submitted, limit lowered to 15 min.

23:23: ShareGPT LeCaR on the GPU (job 3686067, gh151, ran 10 min 9 s): 18.96 % (sim 16.89 %). LeCaR − LRU: GPU +7.33 pp, sim +5.24 pp: same sign, but 2.09 pp apart, outside the 0.5 pp tolerance (does not transfer in magnitude). Note the pre-registered sim numbers used untuned timing; ShareGPT's timing mismatch (H2) is the likely cause. Agent (job 3686096) submitted, limit lowered to 15 min.

23:47: ShareGPT Agent on the GPU (job 3686096, gh151, waited 13 min, ran 10 min 11 s): 11.66 % (sim 14.31 %). Agent − LRU: GPU +0.03 pp, sim +2.65 pp: the simulator's predicted gain does not appear on the GPU (does not transfer). LRU ×2 (job 3686164) submitted, limit lowered to 15 min.

00:02: ShareGPT LRU rep 2 (job 3686164, gh151, ran 10 min 11 s): 11.63 % (rep 1: 11.63 %). LRU ×3 (job 3686216), last sim2real run, submitted, limit lowered to 15 min.
