# GH200 experiments: questions, designs, results

Status: plan, 2026-11-25. Nothing has run on DeltaAI yet. The Result sections
hold the simulator numbers we already have and blank GPU columns to fill in.

Related documents:
* [gh200-calibration-plan.md](gh200-calibration-plan.md) is the operations
  plan: jobs, budget, DeltaAI setup.
* [reviews/gh200-plan-advisor-review.md](reviews/gh200-plan-advisor-review.md)
  is the advisor review. The evidence for the design choices below is there.
* The H1 numbers come from report Table 2, sourced from
  `UniCache-agent-budget-v1/result/agent-cache-budget-study/manual-20261101-162735/`.

| # | Question | GPU-hours | Status |
|---|---|---:|---|
| E1 | Do eviction-policy hit ratios and policy rankings transfer from the simulator to a GH200? (a) pure UniCache, (b) UniCache + GPU step timing | ≈ 2 | planned |
| E2 | Can the simulator predict the TTFT (and ITL) that results from an eviction-policy choice? | shares E1's runs, + 0.25 | planned |
| E3 | *(proposed)* Does a native GH200 kernel table predict step time better than H100 tables plus a fitted correction? | ≈ 0.5 | proposed |
| E4 | *(proposed, conditional on E1)* Does the timing model change which policy the agent finds? | ≈ 0.3 | proposed |

E1 and E2 answer the two questions in the request. Item 3 was left open, so E3
and E4 are proposals; delete them if they are out of scope.

---

## Common setup

**Hardware and serving.** One GH200 (96 GB HBM3, NCSA DeltaAI, `ghx4`
partition, one GPU per job). The engine is the UniCache fork of vLLM (V0) in GPU
mode, with dummy weights and outputs forced to the trace's recorded tokens.
Data-parallel serving with 8 workers is out of scope, because its hit ratio
depends on a router that we do not model.

**Model and cache.** Llama-3.2-1B-Instruct, block size 8, `max_num_seqs` 32,
`max_model_len` 131,072, and chunked prefill off. These match the H1 commands.
CPU swap blocks are set to 0 everywhere, because the traces are single-sequence
and vLLM V0 preempts by recompute, so the H1 setting of 50,000 CPU blocks is
never used.

**Metric.** Token-weighted hit ratio H = Σ hit tokens / Σ prompt tokens,
reported in percent, with deltas in pp. Only the eval split is run on the GPU.

**Predictors.** These are simulator configurations. All of them run on the
laptop at 0 GPU-hours.

| Name | Control plane | Step timing | Role |
|---|---|---|---|
| S0 | UniCache simulator | legacy formula: prefill `3.9e-5·batch^0.997·len^1.014` s on the **full** prompt; decode 10 ms | What the H1 agents were scored on ("pure UniCache") |
| S1 | UniCache simulator | aiconfigurator, `h100_sxm` tables, uncalibrated; prefill charged by uncached tokens | GPU step timing without any GH200 data |
| S2 | UniCache simulator | aiconfigurator, `h100_sxm` (or `gh200`, see E3) + GH200 calibration fit on the step grid | GPU step timing calibrated to GH200 |
| G | UniCache fork on GH200 | real | Ground truth |

The request's "UniCache + aisimulate" corresponds to S1 and S2. The timing layer
in this repo is **aiconfigurator** (`vllm/simulator/step_timing.py`). AISimulate
is NVIDIA's successor to it, and the earlier Dynamo work in
`reference/dynamo-kv-research/` used it. Swapping AISimulate in would be a
laptop-only variant of S1/S2, done if its table format matches.

**GPU replay mode: measured clock.** The GPU executes every step. The replay
clock advances by the measured wall time of each engine-loop iteration, which
includes tokenization in `add_request`, and it jumps over idle gaps instead of
sleeping through them. Arrivals are frozen, and think times come from a seeded
random generator per session, so this follows the same schedule as a real-time
replay. The only difference is host work outside the loop. At 1 request/s,
CodeParrot leaves the GPU about 75% idle in real time, so this mode is what
makes full eval splits affordable and lets us drop truncation. One real-time run
(ALFRED 8 GiB, LRU) validates the mode.

**Prerequisites before any GPU job.** These are code changes made on the laptop:
1. One clock for vLLM core and policies. Today the policy bridges read the
   virtual clock even in GPU mode, so agent policies see `now = 0` and event
   `time = 0`.
2. A Linux sandbox backend for policy workers. The bridges hardcode macOS
   `/usr/bin/sandbox-exec`.
3. The measured-clock GPU mode.
4. Step-log columns `loop_ms`, `measured_ms` and `raw_ms` (uncalibrated aic
   price, with variance), plus `--simulator-aic-systems-path`.

**Candidate configurations.** These are the Table 2 eval numbers from S0. The
final set is chosen after the L0 sweep in E1.

| Config | Turns per session | Best heuristic (tune-selected) | LRU | Best | Agent | Agent − LRU | Agent − Best | Best − LRU | Why it is in the set |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| CodeParrot 8 GiB | 1 | UniCache | 41.59 | 55.45 | 52.99 | +11.40 | −2.46 | +13.86 | Largest Agent − LRU; expected to be timing-insensitive (control) |
| ALFRED 8 GiB | 40 (multi) | LeCaR | 54.00 | 64.79 | 64.08 | +10.08 | −0.71 | +10.79 | Multi-turn, so expected to be timing-sensitive |
| MASH-QA 12.207 GiB | 1 | UniCache | 30.54 | 32.60 | 30.38 | −0.16 | −2.22 | +2.06 | Agent loses to Best; only Agent − Best is tested |
| ALFRED 4 GiB | 40 (multi) | LeCaR | 30.50 | 32.18 | — | — | — | +1.67 | Second timing-sensitive config (no agent run exists) |

---

## E1. Do KV-cache eviction results hold on real hardware?

### Question
Do the hit ratios the simulator gave the H1 policies, and the decisions built on
them, hold on a GH200? The decisions are whether the agent beats LRU, whether it
trails the best heuristic, and whether the best heuristic beats LRU.

* **E1a, pure UniCache:** does S0, the simulator the agents were trained and
  scored on, predict G?
* **E1b, UniCache + GPU step timing:** do S1 and S2 predict G better than S0? Is
  the remaining error due to timing, or to something else (control plane, policy
  CPU time)?

### Why timing can matter at all
Hit ratio depends on timing only through *when* requests arrive and how long
blocks stay pinned. Those can change in three ways:
1. **Service time of follow-up turns.** In multi-turn traces, turn k+1 arrives
   at the finish time of turn k plus think time, so a faster server reorders
   which blocks are recent.
2. **Hits feeding back into speed.** S0 charges prefill on the full prompt, so
   hits never make a request faster. On the GPU they do, so a better policy
   lowers its own load.
3. **Policies measured in seconds.** UniCache (`SimpleHYBRID`) scores blocks by a
   lognormal reuse probability over elapsed seconds. LeCaR discounts by Δt in
   seconds. The ALFRED 8 GiB agent policy has constants in seconds (60 s decode
   penalty). If the clock runs 3–5× faster, these policies behave differently.

MASH-QA and CodeParrot are single-turn with frozen arrivals, and their decode
batch is at most 5, so all three channels are nearly closed for them. ALFRED,
with 11 s mean think time, and ShareGPT, with 100 s, are where timing can show
up.

### Design
**L0, timing-sensitivity sweep (laptop, 0 GPU-hours, before any GPU job).**
* Scope: all Table 2 configs × {LRU, Best, Agent} on the eval split. That is 26
  runs per timing variant, because ALFRED 4 GiB has no agent.
* Timing variants:
  * S0;
  * S1;
  * S1 with prefill charged on all tokens (isolates channel 2);
  * aic decode step forced to 2, 6 and 10 ms;
  * S1 with scale 0.5 and 2.
* Output: how far each hit ratio and each delta moves across variants.
* Use: the GPU configs are picked from this sweep: the two most sensitive configs
  plus one insensitive control. The default is the four-config table above. The
  chosen set and the predicted sign of each delta are written into this doc
  before any GPU run.

**GPU runs (T2).**
* Each chosen config runs LRU, Best and Agent on the **full** eval split in
  measured-clock mode, with the step log on.
* ALFRED 8 GiB also gets an LRU repeat and an Agent repeat, to measure the noise
  floor where it matters.
* One short repeat in the smoke job checks that single-turn traces are
  deterministic.
* About 13 runs in all, one sbatch job per config.

**Simulator side.** S0, S1 and S2 rerun the same commands on the laptop, at most
3 in parallel.

**Metrics.**
* Primary: for each config and each delta (Agent − LRU, Agent − Best,
  Best − LRU), predicted minus GPU in pp, and whether the sign matches.
* Secondary: per-policy |H_pred − H_G| in pp.
* Policy CPU time: per-step `measured_ms` of Agent vs LRU at matched batch
  shapes, and the number of eviction calls.

**Decision rule (pre-registered).** A delta *transfers* if its sign matches and
|Δ_pred − Δ_G| ≤ max(0.5 pp, 2 × |LRU − LRU-repeat|). Each predictor is judged
per config:

| S0 transfers | S2 transfers | Reading for that config |
|---|---|---|
| yes | yes | Hit-ratio search is robust to timing here; the calibrated simulator is needed only for latency claims |
| no | yes | Timing matters; H1 numbers for this config need re-scoring under S2, which motivates E4 |
| no | no | A mismatch beyond timing (control plane, policy CPU time); diagnose from step logs |
| yes | no | The calibration hurts; check the fit |

For configs where L0 shows no sensitivity, S0 and S2 give the same hit ratios
by construction, so only the yes/yes and no/no rows are reachable there. That
is stated now so it isn't read as a finding later.

**Predictions (pre-registered).**
* CodeParrot 8 GiB and MASH-QA 12.207 GiB: S0 ≈ S1 ≈ S2 within 0.3 pp. Any
  sim-to-GPU error comes from the control plane, not timing.
* ALFRED: L0 deltas move by more than 0.3 pp, and S2 predicts G better than S0.
* Sign of each delta on the GPU equals its sign in S0, for every delta with
  |Δ| > 1 pp.

### Result
Status: not run.

What we already have (H2, A100, 512 GPU blocks, LRU and 11 heuristics). The GPU
values were transcribed from a chart, not taken from raw logs:

| Workload | Mean \|sim − GPU\| (pp) | Max (pp) | Note |
|---|---:|---:|---|
| ALFRED | 0.05 | 0.05 | 97% hit ratio at 512 blocks, so no cache pressure; uninformative |
| AgentBank APPS | 0.22 | 1.01 | |
| MASH-QA | 3.23 | 25.47 | |
| CodeParrot | 3.79 | 29.56 | |
| ShareGPT | 50.75 | 55.75 | Decode-timing mismatch |

E1 will be the first transfer test at the H1 operating point (16k–50k blocks),
with our own GPU measurements, and with agent policies.

Table to fill in (one block per chosen config):

| Config | Delta | S0 | S1 | S2 | G | S0 transfers? | S2 transfers? |
|---|---|---:|---:|---:|---:|---|---|
| CodeParrot 8 GiB | Agent − LRU | +11.40 | | | | | |
| | Agent − Best | −2.46 | | | | | |
| ALFRED 8 GiB | Agent − LRU | +10.08 | | | | | |
| | Agent − Best | −0.71 | | | | | |
| MASH-QA 12.207 GiB | Agent − Best | −2.22 | | | | | |
| ALFRED 4 GiB | Best − LRU | +1.67 | | | | | |

L0 sensitivity (max movement of each delta across timing variants, in pp):
*to fill in*.

---

## E2. Can we accurately simulate the TTFT that results from eviction-policy selection?

### Question
For a given eviction policy and workload, does the calibrated simulator predict
TTFT (mean and p90) and ITL on the GH200? Does it predict the **TTFT difference
between policies**, in particular whether the agent's hit-ratio gain becomes a
TTFT gain?

### Design
There are three levels, and each is validated on data it was not fit on.

1. **Step time.** A controlled step grid runs through the *fork* on the GPU
   (about 15 min, in its own job):
   * decode at batch {1, 2, 4, 8, 16, 32} × context {256, 1k, 4k, 16k};
   * prefill at new tokens {16 … 32k} × cached tokens {0, 2k, 8k};
   * 3 repeats, medians.

   S2's calibration is fit on this grid only, as
   `step_ms = scale[phase]·raw_ms + a[phase] + b[phase]·num_seqs`, weighted by
   relative error. The fit uses `loop_ms` or `measured_ms`, whichever predicts
   held-out TTFT better. The grid is workload-independent, so every E1 GPU run
   becomes held-out validation and no leave-one-workload-out fit is needed.
2. **Request.** For each E1 GPU run, compare TTFT and ITL per request between
   S1, S2 and G, on the same requests.
3. **Policy.** For each config, compare the predicted vs measured mean-TTFT
   order of LRU, Best and Agent, and the predicted vs measured
   TTFT(Agent) − TTFT(LRU). OPT is excluded, because its future map goes stale
   under real timing.

S0 also reports TTFT. Including it shows how wrong the agents' training
simulator was on latency.

**Targets (on held-out runs).**

| Check | Target |
|---|---|
| Step time, median abs relative error, per phase × batch bucket, plus time-weighted | ≤ 10% |
| Request TTFT, median abs relative error | ≤ 20% overall; reported separately for ALFRED (prefill-heavy) |
| TTFT order of policies | Same as G wherever the gap exceeds the noise floor |
| Δ TTFT (Agent − LRU) | Same sign as G, error ≤ 30% of \|Δ_G\| |

**Predictions (pre-registered).**
* Raw S1 underestimates GH200 step time several-fold. Its batch-1 decode is
  1.44 ms, and a V0 step on Grace is expected to be several ms, dominated by
  CPU overhead.
* S2 meets the decode target. Prefill (ALFRED) is the hardest case.
* For CodeParrot, an 11 pp hit gain gives only a small TTFT gain, because about
  99% of busy time is decode. For ALFRED the TTFT gain is large, because
  prompts are about 2.4k tokens and outputs about 19.
* S0's TTFT is far from G. Legacy prefill costs about 90 ms for a 2k-token
  prompt, and it ignores hits.

### Result
Status: not run. No measured GPU step data exists yet. The H2 sim2real work
made no latency claim, and the existing latency results
(`result/aic-timing/`, `result/paper-qttft/`) use uncalibrated tables.

Tables to fill in: (a) step error by phase × batch bucket for S1 and S2; (b)
per-config mean and p90 TTFT and mean ITL for S0, S1, S2 and G; (c) policy
TTFT order and Δ TTFT predicted vs G.

---

## E3 (proposed). Native GH200 table vs H100 tables plus a correction

### Question
Does measuring GH200 kernels, through aiconfigurator's collector or a small
in-fork GEMM and attention timer, predict step time and TTFT better than the
`h100_sxm` tables with a fitted correction?

### Design
Ablation on the same held-out E1 runs:
* (i) `h100_sxm` raw;
* (ii) `h100_sxm` + grid fit;
* (iii) native `gh200` + grid fit.

Collect only Llama-3.2-1B's shapes: the five GEMMs including `lm_head` at
n = 128,256 (today it is extrapolated from n ≤ 65,536), and attention for
32/8/64 heads in bf16. Load them through `--simulator-aic-systems-path` from a
self-contained root.

### Prediction
Small gain for decode. At batch 1 the kernels are about 1.4 ms of a step that is
mostly CPU overhead. The gain may be measurable for prefill (ALFRED). If (iii)
does not beat (ii) on held-out TTFT, report "kernel tables are not the
bottleneck for small models; overhead is".

### Result
Status: not run. A laptop probe found that a self-contained root with only
gemm + attention tables reproduces `h100_sxm` estimates exactly, so the loading
path works.

---

## E4 (proposed, only if E1 says timing matters). Does the timing model change what the agent finds?

### Question
If timing changes the outcome for ALFRED, would an agent searching under S2
instead of S0 select a different policy that does better on the GPU?

### Design
* Rerun the H1 agent loop for ALFRED 8 GiB (30 submissions, Claude Opus) on the
  laptop under S2.
* Score its selected policy and the original S0-selected policy on the GPU eval
  split: 2 runs.
* Compare G(S2-selected) − G(S0-selected) against the ALFRED noise floor from
  E1.

### Result
Status: not run. It is gated on E1's reading being "no / yes" for ALFRED.

---

## Order, dependencies and budget

| Step | Where | GPU-hours | Depends on |
|---|---|---:|---|
| Prerequisites 1–4 (clock, Linux sandbox, measured clock, step-log columns) | laptop | 0 | — |
| L0 sensitivity sweep; choose configs; write predictions here | laptop | 0 | prerequisites |
| Fork build on DeltaAI (if no CPU-only partition) | DeltaAI | 1–2 | — |
| Smoke run: GPU visible, policy loads offline, determinism repeat, step-log sanity | GH200 | ≤ 0.25 | build |
| Step grid (E2 fit data) | GH200 | ≈ 0.25 | smoke |
| E1/E2 replays: about 13 full-split runs + 1 real-time validation run | GH200 | ≈ 2–2.5 | smoke, L0 |
| E3 table collection | GH200 | ≤ 0.5 | smoke |
| E4 | laptop + GH200 | ≈ 0.3 | E1 result |
| Analysis: S0/S1/S2 reruns, fit, write-up | laptop | 0 | GPU runs |
| **Total** | | **≈ 4–6** | |

The per-run GPU times assume GH200 step times we have not measured yet.
Confirm them in the smoke job before submitting the E1 runs.
