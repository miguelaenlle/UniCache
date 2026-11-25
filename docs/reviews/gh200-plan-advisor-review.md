# Advisor review: `docs/gh200-calibration-plan.md`

Reviewer: Claude (Opus 5.5), advisor pass, 2026-11-25. This was a read-only review: I edited no plan or code and ran nothing on a GPU or DeltaAI.
Laptop-only probes ran in the session scratchpad: a copied `systems/` root and
perturbed parquet copies, priced through the installed `aiconfigurator_core`
0.12.0.

Legend: **[V]** verified against code or data in this checkout. **[V-probe]**
verified by running aiconfigurator on the laptop. **[U]** unverified (needs
DeltaAI, the web, or a GPU).

## TL;DR: the five changes that matter most

1. **Agent policies see a frozen clock in GPU mode** (`now = 0`, event
   `time = 0`), while vLLM core passes them wall-clock `last_accessed`. At
   least 4 of the 14 selected Table 2 agent policies read `now`. Fix this before
   T2, or the "Agent" arm on the GPU will be a different policy. [V]
2. **The policy sandbox is macOS-only.** `/usr/bin/sandbox-exec` is hardcoded in
   both bridges, so no agent policy can load on DeltaAI. This needs a code
   change in L0, not only a P1 check. [V]
3. **Truncation is invalid for ALFRED, and for CodeParrot it removes the very
   thing that makes it timing-sensitive.** ALFRED's 250 sessions all arrive in
   the first 243 s and overlap, so truncating halves concurrency. CodeParrot's
   53,410-token request arrives at 2,272 s of 2,482 s, so a 50% prefix drops
   it. [V]
4. **Replace real-time replay with a "measured clock" GPU mode** that runs the
   GPU, advances a virtual clock by the measured loop time, and skips idle
   gaps. CodeParrot is about 75% idle at 1 req/s. This is the largest GPU-hour
   saving available. It makes truncation unnecessary, allows full splits for
   more configs, and fixes item 1 by giving everything a single clock. [V for
   the idle fraction under the dev calibration; savings estimate U]
5. **Bring back the controlled step grid run through the fork, and fit S2 on
   it, not on T2.** The aiconfigurator tables price isolated upstream-0.24
   kernels. At the operating point (batch-1 to batch-5 decode), the step is
   dominated by V0/Grace CPU overhead, which the tables do not contain. So
   "native table ⇒ scale ≈ 1" is not a valid test, and the native table route
   (P2a) should not gate anything in goal 2. [V-probe for the kernel share; the
   overhead size is U]

---

## Prioritized recommendations

### Critical: blocks T2 or silently invalidates it

**C1. Use one clock for core and policies in GPU mode.**
*What:* `policy_agent/bridge.py:101` (`'now': GLOBAL_TIMER.get_current_time()`),
`policy_agent/v2_bridge.py:147,178` and `policy_agent/events.py:53` read
`GLOBAL_TIMER` unconditionally. `GLOBAL_TIMER` only advances inside
virtual-time branches (`simulator.py:184,307,338,381`). The core uses
`time.time()` when virtual time is off: `scheduler.py:1111-1114`, `1545-1548`,
`evictor.py:365-368`, `1215-1218`, `llm_engine.py:860-862`.

*Effect on the GPU:* every policy gets `now = 0` and `event['time'] = 0`, while
block `last_accessed` is around 1.7e9 (epoch seconds). Of the selected
submissions behind Table 2:
- CodeParrot 4 GiB (sel 22) computes `age = max(0, now - recency)`, which is
  always 0.
- CodeParrot 24 GiB (sel 9) and ShareGPT 8 GiB (sel 20) key on `now`.
- ALFRED 8 GiB (sel 14), a default T2 config, sets `self.clock = now`. That
  value is the fallback score whenever `last_accessed` is missing, so any such
  block scores about 0 against about 1.7e9 and is evicted first.
- CodeParrot 8 GiB (sel 14) uses `max(clock, now or 0)`, which is harmless.
- MASH-QA 12.207 GiB (sel 7) does not use time.

*Fix:* C1 is subsumed by H1 (measured clock). If you keep real time instead,
make the bridges call the same conditional as the core. Add a unit test that
checks event `time`, `now` and block `last_accessed` share one clock in both
modes.
*Matters:* high. Without the fix, Agent − LRU on the GPU measures a broken
policy.

**C2. Add a Linux sandbox backend for policy workers.**
*What:* `bridge.py:37-38` and `v2_bridge.py:33-34` spawn
`['/usr/bin/sandbox-exec', ...]`, which exists only on macOS. Add a backend
selector: `bwrap --unshare-net --ro-bind ...`, `unshare -rn`, or run the whole
engine under `apptainer exec --net --network none --containall`. Keep the
"policies are offline" guard. Test on the login node in P0 (0 GPU-hours), and
again in the P1 smoke run.
*Matters:* blocking for the Agent arm. The plan's P0.3 checks whether
networking can be disabled, but it does not list this code change.
[V for the hardcoded path; U for whether DeltaAI allows unprivileged user
namespaces or apptainer `--network none`]

**C3. Drop truncation for ALFRED. For CodeParrot, either don't truncate or stop
calling it timing-sensitive.**
Data [V]:

| Eval split | Sessions | Turns per session | First-arrival span | Think time (mean) | S0 sim duration |
|---|---:|---:|---:|---:|---:|
| ALFRED v1-500 | 250 | 40.1 (max 161) | 0–243 s | 11.1 s | 2,401 s |
| CodeParrot v1-5000 | 2,500 | 1 | 0–2,482 s | n/a | 2,810 s |
| MASH-QA v1-5000 | 2,500 | 1 | 0–2,482 s | n/a | 2,482 s |
| ShareGPT v1-1000 | 500 | multi | 0–513 s | 101.6 s | — |
| AgentBank APPS | 2,204 | — | 0–2,122 s | 11.1 s | — |

- **ALFRED:** every session is live at the same time. Keeping only sessions
  whose first turn arrives before T halves the number of concurrent sessions,
  and so roughly halves the working set competing for 32,768 blocks. That moves
  the operating point the plan says truncation preserves. It also saves little
  GPU time, because the run length is set by session length (about 40 turns ×
  (11 s think + service)), not by the arrival span. Replay ALFRED in full.
- **CodeParrot:** the 53,410-output request (`Code-2219de62-…-24`) arrives at
  2,271.9 s, which is in the last 9% of the split. The plan's "Facts" section
  cites it as the reason CodeParrot is timing-sensitive, and a 50% prefix
  removes it. Even on the full split it only affects requests that arrive in
  the last ~210 s.
- The fact "a full eval split is 38–47 min of recorded arrivals" holds only for
  MASH-QA, CodeParrot and APPS. ALFRED has 4 min of arrivals and ShareGPT 8.5
  min.
- The L0.3 check "first-50% deltas within 0.3 pp of full-split deltas" is
  necessary but not sufficient: it compares S0 with S0. Under H1 (measured
  clock) truncation is not needed at all.

*Matters:* high for ALFRED, which is the config where the transfer test has
power (see H3).

### High: large validity or GPU-hour gains

**H1. Use a measured-clock GPU replay instead of real time.**
*What:* add a GPU mode where the engine runs on CUDA, the virtual clock
advances by the measured wall time of each loop iteration (`loop_ms`,
including `add_request` and tokenization), and the existing idle path jumps
the clock to the next event (`simulator.py:332-338`) instead of sleeping
(`simulator.py:340-346`). Point core and policy time reads at that clock (this
also fixes C1). Remove the guard at `vllm/simulator/config.py:66-69` for this
mode only.

*Why:* on the dev CodeParrot LRU run
(`result/aic-dev/codeparrot-32768-aic-h100/.../LRU/eval/steps.csv`), modelled
busy time is 649 s of 2,482 s. 99% of that is decode, with decode batch median
1 and p99 3. A real-time replay therefore leaves the GPU idle about 75% of the
time. Because arrivals are frozen (`frozen_arrivals` in every eval yaml) and
think times come from per-session seeded RNGs (`simulator.py:202-210`),
skipping idle time does not change the schedule semantics. The only
difference from real time is host work not captured by `loop_ms`.

*Validate once:* run ALFRED 8 GiB LRU in both real time and measured clock,
and require hit ratio within the noise floor and TTFT within about 5%. Then use
measured clock everywhere.

*Matters:* full splits for 3 configs at about 1–1.5 GPU-hours instead of 4.5 h
truncated, or all 8 agent configs at about 3–4 h. This estimate depends on real
GH200 step times. [U]

**H2. Bring back the controlled step grid through the fork, and fit S2 on it.**
*Why:*
- The aiconfigurator tables are isolated kernel timings from upstream vLLM 0.24
  (`kernel_source` = `vllm_flash_attn_fa3` and `torch.nn.functional.linear`).
  The replays run the fork: V0, torch 2.6, CUDA graphs, async output
  processing, a Python sampler over a 128,256-entry vocabulary, and block tables
  at block size 8, all on a Grace CPU.
- Raw `h100_sxm` puts a batch-1 decode step for Llama-3.2-1B at **1.44 ms**
  [V-probe]. About 1.1 ms of that is GEMM: GEMMs with n = 3072 (≈0.12 ms),
  n = 2048 (o_proj plus down, ≈0.35 ms), n = 16384 (≈0.46 ms), and lm_head
  (≈0.19 ms).
- A V0 step at batch 1 is plausibly several ms [U]. The difference is
  overhead, and it grows with `num_seqs` and block-table length. A per-phase
  constant `overhead_ms` cannot fit that, so `scale` will soak up the batch
  dependence. Then "(iii) needs a scale ≈ 1" fails for reasons unrelated to
  whether the native table is right.

*What:*
- Grid: decode with batch {1, 2, 4, 8, 16, 32} × context {256, 1k, 4k, 16k};
  prefill with new tokens {16 … 32k} × cached tokens {0, 2k, 8k}; 3 repeats,
  medians. That is about 10–15 min in the P2b slot.
- Fit `overhead = a_phase + b·num_seqs` (optionally `+ c·Σblocks`) on top of
  `scale·raw_ms`.
- All T2 runs then become pure held-out validation, so you don't need the
  leave-one-workload-out construction for the step model.
- `profile_steps_grid.py` already exists. The plan drops it because "the
  collected tables replace it", but they measure different things.

*Matters:* high for goal 1's credibility and for S2 in goal 2.

**H3. Make the L0 sweep target the mechanisms that actually differ, and
pre-commit to where power is expected.**
- *The S0 mechanism the sweep must cover:* the legacy prefill cost is
  `3.9e-5 · batch^0.997 · max_len^1.014` seconds on the **full** prompt
  (`simulator.py:354-366`). Hits do not shorten S0 service. On the GPU, and in
  S1/S2, they do, so a better policy lowers its own load and changes when
  follow-ups arrive. Include a variant that charges prefill by all tokens
  versus by uncached tokens (with the same aic decode), and decode step
  variants of {2, 6, 10} ms. Legacy decode is 10 ms (`--simulator-tpot .01`).
- *Expected result:* MASH-QA and CodeParrot evals are single-turn with frozen
  arrivals (every session has `num_turns` = 1) [V]. Timing can reach their hit
  ratio only through the blocks pinned by running requests, and the decode
  batch is ≤ 5. So expect almost no sensitivity there. ALFRED (11 s think) and
  ShareGPT (100 s think) are the only places it can matter. Write this
  prediction down before L0 runs.
- *Time-denominated policies:*
  - UniCache is `SimpleHYBRID` (`policy_agent/runner.py:24`), whose priority
    is `1 − lognorm_cdf(int(elapsed_seconds))` (`evictor.py:1223-1231`).
  - LeCaR discounts by `discount_rate**(Δt seconds)` (`evictor.py:996-1000`).
  - The ALFRED 8 GiB agent expresses its constants in seconds
    (`DECODE_PENALTY = 60`, `DEPTH_WEIGHT = 18` per 1000 tokens).

  Their behaviour depends on the time scale, and on the GPU ALFRED runs maybe
  3–5× faster than the 2,401 s S0 run [U]. This is the concrete mechanism
  behind "timing matters". Report it as a mechanism, not only as a delta.
- *Power:* only one ALFRED config has an agent (8 GiB; at 4 GiB the agent cell
  is "—"). Add **ALFRED 4 GiB LRU vs LeCaR** (eval 30.50% vs 32.18%,
  +1.67 pp) as a second timing-sensitive config for the Best − LRU delta. It
  is cheap and multi-turn.
- *Count:* "nine configs × 3 policies = 108 runs" should be 26 runs per variant
  (ALFRED 4 GiB has no agent), or 27 if you add OPT as a sanity check.

*Matters:* high. It decides whether T2 can answer "does timing matter" at all.

**H4. Treat L0.2 (CPU swap blocks) as a sanity run, not a decision branch.**
V0 preempts by RECOMPUTE whenever a group has one running sequence
(`scheduler.py:1831-1836`), and CPU blocks are only reached through swap
(`cpu_gpu_block_allocator.py` is the only CPU-device user) [V]. Every trace is
n = 1, so 50,000 CPU blocks are never used. Set 0 CPU blocks in GPU mode, S0, S1
and S2. Run one S0 replay to confirm the hit ratios are bit-identical, and drop
the `--swap-space` and pinned `--mem` fallback.
*Matters:* medium. It removes a risk row and a code path.

### Medium

**M1. Fixes to the `systems_path` hook** (Q1 details are below).
`literal_backend_version()` (`step_timing.py:155-162`) passes
`systems_path=None`, so it fails for `gh200` (seen in the probe: "failed to get
system='gh200'"). Thread the path through, or require
`--simulator-aic-backend-version` whenever `--simulator-aic-systems-path` is
set. `fit_calibration.py` and `rerun_with_latency.py` need the flag too.

**M2. Make sure the native collection includes lm_head and the large-m shapes**
(details under Q1). Collect n = 128256 at k = 2048 for m ∈ {1 … 256}. Collect
m beyond 32,768: with chunked prefill off and `max_model_len = 131072`, ALFRED
prefill batches can exceed the grid. Check how often from S1 step logs in L0
[U].

**M3. Calibration targets and holdout.**
- *Leakage:* P3's LOWO fits on "T2 LRU runs of other workloads **plus the P2
  subsets**". The P2 subsets include the held-out workload's own H2 subset.
  Exclude it, or adopt H2 and the issue goes away.
- *Mixed phase:* the "mixed" phase cannot occur, because
  `enable_chunked_prefill=False` (`run_simulation.py:219`), and the CodeParrot
  step log has only prefill and decode steps. Drop it from fits and targets.
- *Step-error target:* "≤ 10% median step error" is dominated by about 430k
  near-identical batch-1 decode rows per CodeParrot run. Report it per phase ×
  batch bucket, and add a time-weighted error.
- *TTFT:* add an ALFRED-specific TTFT target, since that is where prefill
  matters.

**M4. Make the decision rule quantitative and per config.**
"Transfers" is undefined. Proposal: the sign matches, and
|Δ_pred − Δ_GPU| ≤ max(0.5 pp, 2 × the |LRU − LRU-repeat| difference). Write
down beforehand that for configs where L0 shows no sensitivity, S0 ≡ S2 by
construction, so only the yes/yes and no/no rows are reachable there. Also,
"no/yes ⇒ re-score H1 under S2" re-scores policies that were *searched* under
S0. It cannot tell you whether a search under S2 would pick different
policies. Name that as a separate follow-up (rerun the agent loop under S2), not
P5.

**M5. Place the noise repeats where nondeterminism lives.**
Single-turn frozen-arrival traces should be nearly deterministic on the GPU.
Check that with one short repeat in P1, for example a 300 s CodeParrot prefix
run twice. Spend the T2 repeats on ALFRED: one LRU repeat and one Agent repeat.
That gives two noise estimates in the config that carries the power.

**M6. Job structure.**
Use one sbatch per config (≤ 1–1.5 h with H1) instead of one 4.5 h job. Write
a completion marker after each run, let a failed run not stop the next, and
copy each run's tarball to `/work` as soon as the run finishes. If a job hits
its time limit, everything still in `$TMPDIR` is lost.

### Low

- **L1.** Add the vLLM V0 settings in force on the GPU to every run's metadata:
  `use_async_output_proc` (on by default on CUDA, so `step()` returns the
  previous step's outputs), CUDA-graph capture sizes (`max_seq_len_to_capture`
  defaults to `max_model_len = 131072`, `config.py:836-837`), the FA version
  (FA3 on sm_90 if built, `vllm/attention/utils/fa_utils.py:21`), and
  `max_num_batched_tokens`.
- **L2.** The MASH-QA eval yaml's task `name` is `Codeparrot`. This is cosmetic,
  but it shows up in logs and can confuse the T3 tables.
- **L3.** Goal 1 mentions 8 data-parallel workers, but nothing in the plan
  tests DP. The per-GPU table is correct for DP. Hit ratio under DP = 8,
  however, depends on the router (session affinity vs round-robin), which the
  simulator does not model. Mark DP = 8 out of scope, or add a router to the
  simulator later.

---

## Q1. The native `gh200` table route

**What the installed package expects** [V]:
- *Layout:* `systems/<name>.yaml` plus `systems/data/<name>/<op>/<backend>/<version>/`.
  For dense Llama on vLLM, only `gemm/vllm/<v>/gemm_perf.parquet` and
  `attention/vllm/<v>/{context,generation}_attention_perf.parquet` are needed.
  A root holding only those two directories gave estimates **identical** to the
  packaged `h100_sxm` for five batch shapes [V-probe]. Elementwise ops, norms
  and memory moves are analytic, as the plan says.
- *`collection_meta.yaml`* is a minimal file: `schema_version: 1`,
  `provenance`, `runtime: {framework: vllm, version: X}`, and
  `tables: {gemm_perf: {status: complete}}`, or the two attention tables.
- *Parquet schema:*
  - gemm: `framework, version, device, op_name, kernel_source, gemm_dtype, m, n, k, latency`.
    The `h100_sxm` table is a **generic grid**, not model-specific: 22 values
    each of n and k from 32 to 65,536, and about 75 values of m from 1 to
    32,768, for 35.7k bf16 rows.
  - attention: `framework, version, device, op_name, kernel_source, batch_size, isl, num_heads, num_key_value_heads, head_dim, beam_width, attn_dtype, kv_cache_dtype, step, window_size, latency`.
    For 32/8/64 bf16: context covers isl ≤ 16,384 and batch ≤ 256 (119 rows);
    generation covers step ≤ 131,071 and batch ≤ 2,048 (164 rows).
- *`query_versions.yaml`* is read only from the given roots, the first one that
  has the file (`perf_database.py:169-179`). A custom root without it turns the
  version gate off, so any collected version directory resolves. `gh200` has no
  override in the packaged file, so the packaged default would be vLLM 0.24.0.
  Ship a 3-line `query_versions.yaml` in your root naming the collected version,
  so the gate is explicit.
- *`perf_data_reuse_manifest.yaml`* only drives shared-layer inheritance from
  sibling directories. A self-contained root did not need it [V-probe]. The
  Rust resolver internals were not read [U].

**Is the `systems_path` hook enough?** Almost:
- The Rust model takes `systems_path` as **one literal path**. Passing
  `"<root>,default"` failed with an I/O error on `"<root>,default/gh200.yaml"`
  [V-probe]. The gh200 root must be self-contained. Copy
  `attention_lane_defaults.yaml` into it too, or it falls back to the packaged
  copy with a warning.
- `literal_backend_version()` must receive the path (see M1).
- Parsed tables are cached per process. Editing the parquet files in-process
  had no effect, but a fresh process picked the edits up [V-probe]. That is
  fine for the simulator, but `fit_calibration.py` or an agent-driven fit
  loop must start a new process per table version.

**How lm_head (n = 128256) is priced** [V-probe]: by extrapolation from the
largest measured n. Multiplying only the n ∈ {51200, 65536} bf16 rows by 10
moved a batch-1 decode from 1.44 to 3.13 ms. Multiplying the n = 16384, 3072 or
2048 rows by 10 moved it by the amounts their layers predict. So lm_head costs
about **0.19 ms**, roughly 13% of a raw batch-1 decode step and about the same
per prefill step, because only the last tokens reach lm_head. That matches a
weight-read bound: 128,256 × 2,048 × 2 B = 525 MB, divided by
(3.35 TB/s × 0.8), is 0.196 ms. For `h100_sxm` the extrapolation is
physically sensible. For `gh200`, if the collection is "model-centric" and
contains no n ≥ 51,200 rows, lm_head will be extrapolated from whatever largest
n exists, for example 16,384, which is less reliable. Collect n = 128,256
directly (M2).

**Is the collector job realistic (≤ 2 h, no clock locking)?**
- *Size:* the slice you need is small. Five GEMM shapes × about 75 values of m
  is under 400 GEMMs; with the generic grid restricted to k ∈ {2048, 8192},
  n ≤ 128k, it is about 2–3k. Add a few hundred attention points. At 3–5
  repeats that is minutes, not hours. The 2 h cap is generous *if* the
  collector can be restricted to those ops and shapes.
- *Unverified:* the collector's CLI (`--model-path`, `--plan-only`, `--smoke`,
  `--gpu gh200`) is not in the pip package, so I could not check it [U]. If the
  collector cannot slice its sweeps, a full bf16 GEMM sweep (35k shapes) could
  take several hours. Run `--plan-only` on the login node first, as the plan
  says.
- *Alternative that removes the second container:* once the fork builds,
  write a roughly 60-line collector inside the fork's environment that times
  `torch.nn.functional.linear` and the fork's `vllm_flash_attn` FA3 calls with
  CUDA events and writes the schema above. It then measures *the kernels the
  replay actually runs* (FA3 from the fork's vintage, not from 0.24).
- *Clocks:* not locking clocks is acceptable for a 1B model, where the step is
  overhead-dominated. Record `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE` before
  and after, and take medians. The GH200 module power limit is shared between
  the CPU and the GPU [U], so CPU-heavy V0 steps can lower GPU clocks. That is
  one more reason the replay's own grid (H2) is the better calibration source.

**Bottom line for Q1:** the route is technically sound and cheap once a
container works. Its marginal value for goal 2 is small, though: it refines
about 1.4 ms of kernel time per step out of a step that is mostly CPU. Run it
after T2, or inside the same job as the H2 grid, and never as a gate.

## Q2. The transfer-test design

Confounds and gaps, beyond C1–C3 and H3:
- **Hit-to-speed feedback in S0.** S0 charges full prompts (H3), so S0 differs
  from the GPU in a *policy-dependent* way: a 10 pp better ALFRED policy gets
  faster prefills on the GPU but not in S0. That is a real effect of "timing",
  but it should be isolated by the all-tokens vs uncached-tokens L0 variant, or
  T3 cannot say *which* part of timing mattered.
- **Policy CPU time on Grace vs the laptop.** The v1 bridge flushes events to
  a subprocess worker every 2,048 events and on every eviction choice
  (`bridge.py:95-103`). Its per-step cost inside `step()` on Grace (Neoverse V2)
  is unknown [U]. The plan's per-step `measured_ms` delta between Agent and LRU
  is the right measurement. Also record the number of eviction calls, to
  normalize.
- **Async output processing** shifts finish detection by one step on the GPU.
  It is negligible for hit ratio, but it matters for ITL accounting (L1).
- **Pre-registration:** good in spirit. Add the quantitative rule (M4), and the
  prediction that MASH-QA and CodeParrot will be timing-insensitive (H3).
- **Reading table:** it is fine as a summary, but read it *per config*. With
  the default three configs only ALFRED can land off the diagonal, so the
  global conclusion "hit-ratio search is robust to timing" would rest on a
  single config. Adding ALFRED 4 GiB (H3) helps.
- **Truncation:** not valid as designed (C3). With H1 it is unnecessary.

## Q3. GPU-hour efficiency

| Item | Plan | Recommended | Why |
|---|---:|---:|---|
| Fork build (if no CPU-only partition) | not budgeted | 1–2 h | DeltaAI appears to be GH200-only [U], so a build job is charged as a GPU job |
| Smoke P1 | ≤ 0.25 | ≤ 0.25 | Add the sandbox test and a 300 s determinism repeat |
| Step grid via the fork (H2) | dropped | 0.25 | Workload-independent calibration; makes all of T2 held-out |
| Native tables P2a | ≤ 2 | ≤ 0.5, after T2 | Small slice; low value for goal 2 |
| P2b H2 subsets | 0.5 | 0.1 (ShareGPT only) or cut | 512-block regime, far from T2 |
| T2, real time, truncated | ≈ 4.5 | — | Invalid for ALFRED (C3) |
| T2, measured clock, full splits, 4 configs (+ALFRED 4 GiB) | — | ≈ 1.5–2.5 [U] | Mostly idle skipped (H1) |
| Measured-clock validation run | — | ≈ 0.3 | One real-time ALFRED LRU run |
| **Total** | ≈ 7–8 | **≈ 4–6** | and with stronger evidence |

Also:
- Never run two replays on one GPU (the plan already says this).
- Keep one GPU per job.
- Bind CPUs to the Grace that is NUMA-local to the job's GPU. Record
  `nvidia-smi topo -m`, `numactl -s`, the hostname and the GPU UUID per run.
  Cross-socket placement would add CPU-overhead noise to the calibration [U].

## Q4. Calibration methodology (P3) and the held-out design

- *Fit target:* fitting on `raw_ms` re-priced from the step log loses the
  variance terms. `fit_calibration.py` prices `fpm_from_counts(**counts)` with
  `var = 0`, but the simulator passes variance at run time
  (`step_timing.py:64-75`). The plan's `raw_ms` column fixes this; keep it.
- *Weighting:* 1/y (relative-error) weighting is right. Even with it, the batch
  dependence of overhead needs its own term (H2), or "scale" stops meaning
  "kernel-time correction".
- *LOWO:* the design is sound, apart from the leakage (M3). With only three
  workloads of very different shape, though (batch-1 decode, prefill-heavy
  multi-turn, single-turn QA), each held-out workload is out of distribution in
  its phase mix. A workload-independent grid fit (H2), validated on every T2
  run, is both cleaner and a stronger claim.
- *Choosing `loop_ms` vs `measured_ms` by held-out TTFT* is good. Under H1 the
  clock *is* `loop_ms`, so the GPU ground truth and S2 use the same quantity.
- *ADRS-style agent fit (Fable):* fine, as long as it sees only the fit set and
  each table version runs in a fresh process (Q1 caching).
- *OPT:* leaving OPT out of the TTFT ordering check is correct. `refine_opt.py`
  applies only to S1 and S2.

## Q5. DeltaAI risks

- **Fork build:**
  - The fork's history is squashed: `git log` shows `bc51340 initial commit`,
    and `vllm/_version.py` gives `0.1.dev22+ge7b231e90`. So the upstream base
    commit that `VLLM_USE_PRECOMPILED=1` and the wheel-overlay route (a1) need
    is not recoverable from git [V].
  - Find it on the laptop instead, by diffing `csrc/` and `vllm/` against
    upstream v0.8.x tags. The torch 2.6.0 pin points to roughly 0.8.0–0.8.5
    [U].
  - Then check, from the laptop and at 0 GPU-hours, whether aarch64 wheels
    exist:
    `pip download torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126 --platform manylinux_2_28_aarch64 --only-binary=:all: --no-deps`,
    and the same for that vLLM version. My recollection is that upstream vLLM
    did not ship aarch64 wheels in the 0.8 era [U]. Plan for (a2).
- **The (a2) compile:**
  - FA3 for sm_90 is the slowest, most memory-hungry part of the build. Use
    `TORCH_CUDA_ARCH_LIST=9.0`, a moderate `MAX_JOBS`, ccache, and a node-local
    build directory, then keep only the wheel, so the build tree doesn't eat
    the file quota.
  - If build time is the bottleneck, building FA2 only
    (`VLLM_FLASH_ATTN_VERSION=2`) is acceptable: the grid fit (H2) absorbs the
    kernel difference.
  - Ask NCSA about heavy builds on login nodes. If they say no, budget the build
    as GPU-hours.
- **Disk:**
  - Two images (an upstream vLLM arm64 image for the collector and a PyTorch
    arm64 image for the build) are each likely 8–15 GB as `.sif` [U], so a
    10 GB budget is tight. Keep one image if you take the in-fork collector
    route, or delete the collector image after P2a.
  - Set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` to `/work/hdd/...` *before*
    the first pull. The default is under home, which is at 93/100 GB.
  - Also redirect `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`,
    `TORCHINDUCTOR_CACHE_DIR` and `HF_HOME` to `$TMPDIR`, and set
    `HF_HUB_OFFLINE=1`.
  - `models/Llama-3.2-1B-Instruct` is 17 MB with no weights, which is good
    [V]. The ALFRED eval jsonl is 242 MB, CodeParrot 17 MB and MASH-QA 30 MB
    [V].
- **Network off:** see C2. Test both `unshare -rn` (needs unprivileged user
  namespaces) and `apptainer --net --network none` on the login node [U]. Since
  weights are dummy and the tokenizer is local, running the whole engine
  offline is simplest and covers the policy subprocess too.
- **Grace CPU:** V0 with a 1B model is CPU-bound, and Grace's per-core speed
  differs from both the laptop and x86 hosts [U]. That is exactly what the
  overhead term captures, so do not reuse an A100 or H100 overhead fit. Bind
  NUMA as described in Q3.
- **Hardware identity:** the 96 GB HBM3 part is 4.0 TB/s and the 144 GB HBM3e
  part is 4.9 TB/s. Confirm which one DeltaAI has with `nvidia-smi -q` in P1 [U].
  `sm_version: 90` and the 989 TFLOPS bf16 figure are the same as for H100 SXM;
  the draft yaml is consistent with that.
- **Partition and charging details** (`ghx4`, charge factors, the
  `/work/hdd/bcjw` path, the account name): I could not check any of these
  without ssh [U].

## What I verified, and how

- `h200_sxm.yaml` differs from `h100_sxm.yaml` only in `data_dir`, `mem_bw`,
  `mem_capacity` and `nccl_version`, using `diff`. The plan's quoted H100 GEMM
  and attention values match the parquet files exactly.
- The systems root and lm_head findings come from a copied root under the
  session scratchpad, plus parquet perturbations, each priced in a fresh
  process through `RustForwardPassPerfModel.from_native`.
- The trace facts (turns, arrivals, think times, the long request) come from
  the eval `.yaml` and `.jsonl` files under `UniCache-agent-budget-v1/TaskData/*/agent-splits/`.
- The time usage of the selected agent policies comes from
  `manual-20261101-162735/agent-runs/*/*.json`, which give the run and the
  selected submission, and from grepping `internal/submission-NNNN/policy.py`.
- The code facts are cited inline with file:line.
