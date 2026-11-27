# Advisor review: why eviction policy moves QTTFT in the UniCache paper but not for us

Reviewer: Claude (Opus 5.5), advisor pass, 2026-11-26/27. Read-only for the project: no code,
plan or doc was edited except this file, nothing was committed, and no GPU job was submitted.
One small simulator pilot ran on the laptop (outputs in `result/advisor-qttft/`, driver scripts in
the session scratchpad, reproduced in Appendix B).

Legend: **[V]** verified against code or data in this checkout. **[P]** measured in the pilot
for this review. **[U]** unverified or inferred.

## Summary

1. **The paper's large ratios are bad policies, not good ones.** Against LRU, the paper's best online policy is 0.77–0.86× and OPT 0.62–0.78×. The headline "up to 3.63×" compares with LFU. [V]
2. **On a 1B model on a GH200 a miss is nearly free.** A full hit saves 6% of a 520-token request under v8, against 64% for A100-3B. Cache-sensitive work is 1–2% of GPU busy time, against 6–13%. [V]
3. **In v8 the CPU cost per eviction (0.057 ms per block) is 3× the GPU compute it replaces**, so caching itself barely pays: no caching is 0.95–1.19× LRU on the paper's Trace #2 mix. [P]
4. **Pilot (Trace #2 mix, v8, 3× load, below the knee):** OPT 0.86–0.90×, UniCache 0.96×, LFU 1.25–1.34× LRU. An agent's reachable upside is ≈ 10–14%, close to GPU noise; the downside is real. [P]
5. **Step 3 used the least sensitive trace we have:** 600 single-turn CodeParrot requests in 23 s. Heuristics move uncached tokens by 3.5%, against +11% to +119% for LFU in the mixes. [V]
6. **Both step 3 and the Hybrid traces are scored on a cliff or a burst.** Mean TTFT there tracks queue transients, not hits (at 4.5×, no caching = 0.53× LRU). [V]/[P]
7. **Trace differences:** our mixes lack ToolBench (6–21% of requests) and resample ALFRED and ShareGPT. The paper does not state which model or budget gave its QTTFT table. [V]
8. **Experiment** (§3): a simulator window screen (NONE / LRU / LFU / UniCache / OPT × 3 traces × 3 sub-knee loads; about 2 h on the laptop), plus at most one 15-min GH200 job (0.25 GPU-h) to check v8's eviction charge.

---

## 1. What differs, ranked by how much it compresses policy effects

The policy effect on mean QTTFT is roughly **(queueing amplification) × (share of service time
that depends on hits) × (how much the policy changes the number of misses)**. The table ranks each
difference by how many times smaller it makes that product for us. It gives two columns, because
the paper's spread is mostly downside (bad policies) while an agent can at best reach the upside
(OPT).

| Rank | Cause | Ours (step 3 / pilot) | Paper or paper-like | Size of effect | Evidence |
|---:|---|---|---|---|---|
| 1 | **Cost of a miss relative to everything else** (model × GPU × software) | 1B on GH200. GPU prefill step = 10.6 ms + 1.8 ms per 1k new tokens (5,577 single-prompt steps, step 3 GPU logs). Decode ≈ 2.2–3.9 ms/step. Cache-sensitive token work = **1.8%** of busy time (step 3), **1.0%** (Trace #2, 3×). | A100 (80 GB), 1B/3B/8B. Priced as uncalibrated aic A100-3B: marginal ≈ 26 ms per 1k tokens, decode ≈ 6–8 ms/step. Same schedules re-priced: **12.6%** (step 3), **5.6%** (Trace #2, 3×). | **≈ 6–7×** on both upside and downside. Isolated request, full hit vs full miss: v8 1.06× (CodeParrot 520 tok), 1.09× (700 tok), 1.38× (ALFRED 2.4k); A100-3B 2.8×, 3.7×, 11×. | [V] logs + `step_timing.py`; re-pricing script in Appendix B |
| 2 | **Per-eviction CPU cost comparable to the compute saved** (part of rank 1, but a separate mechanism) | v8 `evict_ms` = 0.057 ms per evicted 16-token block, against 0.019 ms of GPU compute for those tokens. Trace #2 at 3×: LRU's eviction charge is 18.1 s against a 33.6 s prefill saving vs no caching. LFU's is 34.7 s against 22.6 s, so LFU pays more than it saves. Steady window: NONE = 0.95× LRU while single-turn tasks arrive, 1.19× in the multi-turn tail; OPT = 0.86–0.90×. | Unknown. Real vLLM V0 pays this CPU too (the same Python block manager), but on A100-3B a miss costs ~0.42 ms of GPU compute per 16 tokens, so the CPU share is ~7× smaller. | Caps the upside. Under v8, going from 74% to 0% hits is worth −5% to +19%, and OPT's extra 11 pp is worth 10–14%. | [P] `result/advisor-qttft/t2-v8-x3/`. The GPU runlog finds "TTFT rises with hit ratio, because eviction-policy compute is on the critical path" (LRU prefill step 10.1 ms, UniCache 11.5 ms) [V] |
| 3 | **Trace: homogeneous single-turn vs heterogeneous mix with long multi-turn contexts** | Step 3 = CodeParrot only, 100% single-turn, 519-token prompts. Uncached tokens vs LRU: heuristics −3.5% at best; OPT −18.6%. | Mixes in which ALFRED holds 70–91% of prompt tokens (T#1, T#2). Uncached tokens vs LRU (our sim of those traces): LFU **+45% / +119% / +11%**, best online −21 to −26%, OPT −32 to −44%. | **Downside: 10–40×.** A "bad" policy needs a mix that fools frequency- or FIFO-based ordering. **Upside: ≈ 2×** (OPT −19% vs −32 to −44%). | [V] `result/paper-qttft/aic-a100/table5.csv`, `docs/step3-results.md` |
| 4 | **Load regime** | Step 3 at 25× sits on a cliff: GPU 22.5 ms (20×), 91 ms (25×), 1,815 ms (33×). Eval mean TTFT by arrival quartile: 22 / **310** / 33 / 25 ms. One burst sets the mean. Pilot Trace #2 under v8: 16.8 ms (1×), 25.8 ms (3×), 11.3 s (6×). | Paper LRU QTTFT 4.94 / 0.28 / 0.13 s against unloaded TTFT of tens of ms, which implies sustained queueing. Our aic-A100 replay at the paper's rates gave 10.7 s / 18 ms / 18 ms, so we cannot recover the paper's load (likely 7–15× more queueing on T#2/T#3). | Amplifies any service-time change and its noise equally. It explains why the paper's ratios are large given a real effect, not why ours is zero. | [V] runlog TTFT baselines; [P] |
| 5 | **Policy CPU cost that the simulator does not see** | The sim charges every built-in policy the same 0.057 ms per eviction and 0 policy CPU (`cpu_scale` = 0). The GPU sees real per-policy costs: UniCache +4 ms TTFT at 20×, FIFO 121 ms (one run), agent sandbox +45%. | The paper measured on GPUs, so its LFU/LeCaR/FIFO/WA slowdowns include their Python CPU cost at 0.5–2.3M-token caches. In our aic-A100 replays FIFO and WA took 20–30% more laptop wall time than LRU. | Unknown size. It can only make the paper's "bad policy" ratios *larger* than hit ratio alone explains, and make our sim's TTFT ordering wrong. | [V] `step_timing.py`, `summary.json` `simulation_time`; magnitude [U] |
| 6 | **Duration and scale of the trace** | 600 requests, 23 s, 0.31M prompt tokens. Tune and eval slices land in different regimes (LRU 45 vs 81 ms). | 7k–27k requests over 40–48 min, with session reuse over minutes. | Adds variance, not bias. It makes the agent's tune signal unrelated to eval. | [V] |
| 7 | **Cache size** | 4,096 × 16 = 65,536 tokens. OPT−LRU = 15.7 pp (eval, OPT unrefined). | 0.5A–0.9A, i.e. ≈ 0.53M tokens (3B, our convention) to ≈ 2.3M tokens (1B). Our replays of the mixes at 0.53M tokens: OPT−LRU = 16 / 9 / 25 pp. | **Not a cause:** the headroom is similar. The level differs (15% vs 40–80% hits), but the paper's QTTFT spread tracks *differences* in misses. | [V] |
| 8 | **Missing ToolBench, resampled ALFRED and ShareGPT** | See §2. | — | Small for T#1/T#2 (ToolBench is 6–14% of requests). Larger for T#3 (21%): it is the task UniCache's structural-reuse queue targets. | [V]; ToolBench prompt sizes [U] |

**Bottom line for "why".** For the *upside* an agent can reach, cost structure (ranks 1 and 2,
≈ 6–7× and then some) dominates and the trace is second (≈ 2×). For the *downside* (the paper's
headline ratios), the trace dominates (10–40×). Load (rank 4) amplifies whatever effect exists,
and in step 3 it also amplified the noise past it.

## 2. Are we using different traces? Yes, in four ways

**What the paper says.** The extended abstract on disk
(`~/Downloads/unicache-sigmetrics26-extended_abstract.pdf`) says "three heterogeneous traces with
different mixes of multi-turn and single-turn requests". It names no datasets, mix ratios, arrival
rates or trace lengths. Its QTTFT table is **Table 1**, and it does not say which model or KV budget
the table used. The experiments ran on "an NVIDIA A100 (80GB)" with Llama 1B, 3B and 8B. Our docs
and `run_table5.py` cite "Table 5", "Sec. 6.4", "Table 4: 63 GiB" and "Llama-3.2-3B, 0.9A". Those
come from a longer version that is not in this checkout. [V that it is absent from the abstract.
The longer version is U.] The authors' released scripts (`TaskData/Hybrid/test1/run.sh`,
`run_new.sh`) run **1B** on A100, with 50k–100k blocks of 8 tokens or 10k–80k blocks of 16. [V]
**Action: find the full paper, or ask the authors which model and budget produced Table 1.** If
it was 1B, rank 1 above shrinks, because their per-token cost would be about 3× lower than A100-3B.

**Which traces we use.** Our "Trace #1–#3" are the repo's `TaskData/Hybrid/test1–3` configs. The
mapping to the paper's traces is our inference from multi-turn request shares; `LOCAL_SETUP.md`
itself says the repo "does not clearly identify an exact mapping". [V]

| Component (task) | Paper config (`tasks_config.yaml`) | Ours, T#1 / T#2 / T#3 (requests, from the 2026-11-24 replays) | Prompt / output mean (tokens) | Difference |
|---|---|---|---|---|
| AgentBank APPS (single-turn) | 1,800 sessions, Poisson 1/s, all three traces | 1,800 / 1,800 / 1,800 | 392 / 406 | Same file and sampling, but regenerated with unseeded randomness. Not byte-identical. |
| CodeParrot (single-turn, structural) | 1,800, 1/s, all three | 1,800 / 1,800 / 1,800 | 506 / 188 | Same as above |
| MASH-QA (single-turn) | 1,800, 1/s, all three | 1,800 / 1,800 / 1,800 | 994 / 89 | Same as above |
| **ToolBench (single-turn, templated)** | 1,800, 1/s, all three | **missing: 0 / 0 / 0** | — | Raw Preble G1 workload not downloaded. This is 6% / 14% / 21% of each trace's requests and the main "structural reuse" task. |
| AgentBank ALFRED (multi-turn, agentic) | `alfred_scale_5000`, first **500** (T#1) / **100** (T#2) / none (T#3) sessions, think time lognormal(1.81, 1.092) (median 6 s) | 20,084 / 4,223 / 0 requests (≈ 40 turns per session) | **2,389 / 19** | We re-tokenized the first 600 sessions of our own `alfred_scale_5000.json`. `Alfred_scale_up.py` prepends a UUID to each session's first message **by design** ("Avoid cross session prefix share"), so the lack of cross-session reuse matches the authors'. The unseeded shuffle makes *which* ALFRED episodes appear differ. |
| ShareGPT (multi-turn, chat) | first 500 sessions, think time lognormal(4.15, 0.971) (median 63 s), all three | 1,565 / 1,565 / 1,565 | 697 / 282 | Deduplicated file regenerated locally. Same caveat. |
| **Whole trace** | — | 27,049 / 11,188 / 6,965 requests over 2,567 / 2,416 / 2,880 s | T#1: ALFRED is **91%** of prompt tokens; T#2: 70%; T#3: 0% | Multi-turn request share 80% / 52% / 22% without ToolBench, which matches the paper's ≈ 80 / 50 / 20. It would be 75% / 45% / 18% with ToolBench. |

**Step 3 is a different trace altogether.** It is the first 600 sessions of the H1 CodeParrot
eval split (`TaskData/sim2real/codeparrot@600`). All 600 are **single-turn** (no turn > 1), with
arrivals compressed 25× into **23 s**. Mean prompt is 519 tokens (p90 730), mean output 179, and
hits come from shared code templates (about 80 tokens per request under LRU). It contains no
session reuse, which is where the paper's policies separate. [V]

**Other differences in the setup.** Block size 16 (GPU, xformers constraint) vs 8 in the authors'
Hybrid scripts. `--simulator-num-cpu-blocks` 0 vs 100,000 (swap space only, likely irrelevant).
`max-num-seqs` is 32 in both. Tokenizer: the configs point at 8B-tokenized copies, and we use the
shared Llama-3 tokenizer files, which are equivalent. [V]

## 3. Proposed experiment

### Question it answers

*In our setting (1B, GH200, v8), is there any (trace, cache, load) cell where eviction policy can
change mean TTFT by more than the GPU noise, and which cell is it?* The pilot already suggests
"barely, and only on the downside". The experiment makes that a decision.

### Experiment A: sensitivity-window screen (simulator only)

For each cell, run two bounds and two online probes. The bounds give the *window* any eviction
policy can occupy:

* **NONE** (prefix caching off): the most misses any policy can have. With `evict_ms` it is not a
  strict worst case, since LFU beat it in the pilot, so the LFU probe also covers the downside.
* **OPT, refined once** (`refine_opt.py --iterations 2`): the upside bound on hit ratio. It is not
  a strict TTFT bound, but it is the right yardstick.
* **LRU** (baseline), **LFU** (paper's worst), **UniCache** (`SimpleHYBRID`, the best online in the
  paper and the pilot).

| Item | Setting |
|---|---|
| Model / timing | Llama-3.2-1B-Instruct, `--simulator-timing aic --simulator-aic-system h100_sxm --simulator-timing-calibration calibrations/gh200-llama-3.2-1b-v8.json` |
| Block size / seqs | 16 tokens, `--max-num-seqs 32`, `--simulator-num-cpu-blocks 0` (as in step 3) |
| Traces × cache | **(a)** Hybrid T#1 (ALFRED-heavy, prefill-dominated: 19-token outputs), 33,177 blocks = 530,832 tokens. **(b)** Hybrid T#2 (paper-like mix), 33,177 blocks. **(c)** CodeParrot `codeparrot@600` eval (step 3 control), 4,096 blocks = 65,536 tokens. Add (a′) T#1 at 8,192 blocks (131k tokens) only if (a) has OPT−LRU < 10 pp. |
| Load | Scale **both** session starts and think times by k (multiply `lambda_rate` by k, subtract ln k from `mu`) so the whole trace is time-compressed. First, an LRU-only sweep k ∈ {1, 2, 3, 4, 5, 6} per trace finds the knee k\* (the smallest k with LRU mean TTFT > 3× its k = 1 value). Then run the policies at **k = 0.5 k\*, 0.8 k\*, 0.95 k\***. |
| Policies | NONE, LRU, LFU, SimpleHYBRID, OPT (+1 refinement) |
| Sensitivity arm | At 0.8 k\* only: LRU, NONE and OPT with a copy of v8 that sets `evict_ms` = 0. It shows how much the answer depends on the least-validated parameter. |
| Control arm | Same cells priced with uncalibrated `a100_sxm` tables and `models/Llama-3.2-3B-Instruct` at its own 0.8 k\*. This swaps only the cost model. |
| Measurement window | The Hybrid configs start every task at t = 0 with the same rate, so the single-turn tasks (1,800 sessions each) finish at 1,800/k s while multi-turn tails run on: a cold-cache opening burst, a mixed phase, then a multi-turn tail. **Exclude the first 200 s of virtual time** (cache fill and burst; the pilot shows a [200/k, 1,600/k) window still contains the burst at 4.5×). Report the mixed phase and the tail separately, plus the whole-run mean for comparison. Best: replace the Poisson-from-zero starts with `frozen_arrivals` spread evenly over a common span, so the mix is stationary. |
| Outputs | Per cell: mean, p50 and p90 TTFT ratio vs LRU in the window; hit ratio; evictions; busy share. Write under `result/advisor-qttft/` or a new `result/qttft-window/`. |

**Predictions.**

| Hypothesis | What A shows |
|---|---|
| H-cost (rank 1–2 dominate) | v8: on every trace, OPT/LRU ≥ 0.85, best online ≥ 0.95 and NONE/LRU ≤ 1.25 below the knee; only LFU stands out. A100-3B control: T#1/T#2 show OPT ≤ 0.80 and LFU ≥ 1.5 even at 0.5 k\*. *The pilot's Trace #2 at 3× is borderline: OPT 0.86–0.90, UniCache 0.96, NONE 0.95–1.19, LFU 1.25–1.34.* |
| H-trace (rank 3 dominates) | T#1/T#2 show a wide window under **both** cost models, CodeParrot a narrow one under both. |
| H-load (rank 4 dominates) | The window is narrow at 0.5 k\* and wide at 0.95 k\* under both cost models and on all traces. |
| H-evict (the `evict_ms` term decides) | With `evict_ms` = 0, OPT/LRU drops by ≥ 10 points and NONE/LRU rises clearly above 1.1. |

**Decision rule** (fixed before running):

1. A cell is **usable for TTFT work** if, at a load ≤ 0.8 k\* (below the cliff, where GPU noise was
   ±2% for CodeParrot at 20×), **OPT/LRU ≤ 0.85** *and* at least one online heuristic is ≥ 10% away
   from LRU. Use the usable cell with the lowest load for agents, and give the agent the same load
   and slice on tune and eval.
2. If **no v8 cell is usable** but A100-3B cells are, eviction cannot matter for mean TTFT on 1B /
   GH200. Stop optimizing TTFT through eviction in this setting. Options: (i) a larger model (8B
   fits on a GH200; first check `aic h100_sxm` 8B in the sim), (ii) longer prefill-heavy prompts
   (ALFRED-like, 19-token outputs), or (iii) keep hit ratio as the agent's objective and report TTFT
   only at the cell where it moves.
3. If the answer flips with `evict_ms` = 0 (H-evict), the GPU check in Experiment B decides it.
   Do not trust either sim arm before then.

**Cost.** A T#2 replay takes about 2.5–4 min of laptop wall time, a T#1 replay about 5 min (3 GB
RAM), a CodeParrot-600 replay under a minute.
* Knee sweeps: 3 traces × 6 k × LRU = 18 runs, ≈ 25 min at 3 parallel.
* Main grid: 3 traces × 3 loads × 5 policies (+1 OPT refinement) = 54 runs, ≈ 60–75 min at 3 parallel.
* Sensitivity and control arms: ≈ 25 runs, ≈ 30 min.
* **Total about 2 h of laptop time, no GPU.** Trace #2 at 3× is already done (Appendix A).

### Experiment B: one GPU job, only to settle the `evict_ms` question (conditional)

Run this only if Experiment A's decision depends on `evict_ms` (case 3), or before you claim "policy
can't matter" in a writeup.

* Slice: T#1-like ALFRED-heavy mix, time-compressed (k from A) and cut to ≤ 2.5 min of replay.
  ShareGPT's 63 s median think time makes T#2 too long to replay four times in 15 minutes.
* Replays: **NONE ×2, LRU ×2** (×2 gives the noise floor). Add LFU ×1 if the slice is under 2 min.
* Pre-register the simulator's NONE/LRU for this exact slice first. The pilot's multi-turn tail
  gives ≈ 1.2. If the GPU ratio is within ±0.1 of the sim's, v8's eviction and compute costs are right
  and Experiment A's v8 column stands. If it is much higher (≥ sim + 0.2), v8 over-charges evictions
  or under-prices prefill, and the sim *understates* policy effects. Refit `evict_ms` from these
  steps and rerun A's v8 cells. If it is ≤ 1.05, caching does not pay on this GPU at all, and no
  eviction policy can help TTFT here.
* **Cost: 1 interactive job, 15 min, 0.25 GPU-h (0.5 charged).** This is the only GPU spending
  proposed.

## 4. Methodology concerns

1. **Report ratios against LRU, not against the worst policy.** The paper's "1.10×–3.63×" and our
   report's "3.87 s to 8.47 s" (mid-semester report, Introduction) compare UniCache with LFU.
   Against LRU the paper shows UniCache 0.78 / 0.86 / 0.77 and OPT 0.78 / 0.75 / 0.62. Our
   expectation should have been "≤ 25–40% upside", not "2–3×".
2. **One load point on a cliff is not a measurement.** Step 3's 25× is between 20× (22.5 ms) and
   33× (1.8 s). There, ±9–17% noise comes from one burst (quartile 2 = 310 ms, others 22–33 ms).
   Report TTFT-vs-load curves, or work at ≤ 0.8 of the knee, and report p50 next to the mean. The
   mean at the knee mostly measures how one burst resolves.
3. **Tune and eval are in different regimes.** In step 3, LRU TTFT is 45 ms on tune and 81 ms on
   eval at the same scale factor: different burst structure. The agent optimized a signal that
   does not predict eval. Match load by measured utilization or knee position, not by a shared
   scale factor.
4. **The Hybrid traces are not stationary, and their mean TTFT is set by the opening burst.** Every task starts at t = 0 with a 1/s Poisson rate, so the four single-turn tasks deliver 7,200 requests in the first 1,800/k s, into a cold cache. Pilot, Trace #2 at 4.5×: requests arriving in the first 200 s average 694 ms (NONE), 1,358 ms (LRU) and 1,425 ms (LFU). After that window: 60, 60 and 121 ms (p50 22, 19, 30 ms). The whole-run mean (NONE 0.54× LRU) comes entirely from how a cold cache absorbs the burst. A plausible cause is v8's eviction charge, since NONE never evicts, but LRU's total prefill path is the cheaper one, so the mechanism is unverified. The paper's QTTFT is presumably a whole-run mean of the same construction, so part of its spread may be this transient too [U]. Score a warm, steady window (see Experiment A).
5. **`--arrival-scale` only compresses first-turn arrivals.** Follow-up turns keep their think
   times (`mu`, `sigma`), so "25×" means something different for single-turn and multi-turn traces.
   It also changes how many sessions overlap, and with that the hit ratio: the pilot's LRU hit ratio
   falls from 78.2% at 1× to 74.0% at 3× on the same trace. Scale think times too (shift `mu` by
   −ln k) when you want "the same trace, faster".
6. **The simulator cannot rank built-in policies by TTFT.** Every built-in policy gets the same
   0.057 ms per eviction and no decision cost. The GPU shows real per-policy CPU differences
   (UniCache +4 ms at 20×, unexplained; LRU 10.1 vs UniCache 11.5 ms per prefill step on ALFRED).
   On 1B / GH200 those differences are as large as the compute effects. Either fit a per-policy
   cost (time `evict()` in-process per policy on the laptop and scale it, as was done for the
   sandbox) or label simulated TTFT orderings among heuristics as untested.
7. **`evict_ms` comes from one synthetic config and now decides the answer.** It was fitted as
   0.9 ms of prefill excess divided by 15.65 evictions per step on a 512-block config. In the pilot
   LRU's eviction charge is about half its compute saving over no caching (18.1 s vs 33.6 s), so a
   2× error in `evict_ms` moves NONE/LRU by roughly ±10 points. Experiment B is the cheap check.
8. **OPT is a hit-ratio oracle, not a TTFT oracle, and step 3's was not refined.** Under cache-aware
   timing the no-cache future map is stale (T#1's refined OPT went 0.395 → 0.669 in hit ratio, and
   its QTTFT of 0.14× LRU never converged). Use at least one refinement, and do not call OPT's TTFT
   a bound. Under v8, a hit-maximizing order is not the TTFT-minimizing order when evictions cost
   CPU.
9. **The agent's objective rewards cheapness, not decisions.** The sandbox adds +45% TTFT, 10–100×
   the decision effect available in step 3. The agent "learned" to copy LRU. Until policies run
   in-process (or the score subtracts the sandboxed-LRU overhead, as the +2.0% equal-overhead row
   in step 3 does), a TTFT-objective agent run measures IPC cost. Score against sandboxed LRU, or
   charge a fixed per-call cost that is identical for all submissions.
10. **The best-heuristic selector under noise is a coin flip.** With a 2.5% spread among
   heuristics and ±9% noise, "best heuristic on tune by TTFT" is random. Select by hit ratio on
   tune (deterministic in the sim) and report TTFT, or require a ≥ 2σ gap.
11. **The paper's QTTFT table has unknown provenance.** The model, budget and load are not in the
    abstract we have (see §2), and the paper measured on GPUs while its simulator's own timing gives
    identical QTTFT for every policy (our `legacy` result). Do not tune our setup to match its
    absolute numbers. Match its *mechanism* instead (miss cost share × load), which Experiment A's
    control arm does.
12. **Small numbers to fix in our docs.** `aic-timing.md` says the T#3 reproduction "barely
    queues", which is right, but it calls the paper's T#2/T#3 load "10× more" from TTFT alone. TTFT
    also scales with per-request cost (A100 vs model), so load and cost are confounded in that
    ratio. Trace lengths are 40–48 min (2,416–2,880 s), not "~1 hour".

---

## Appendix A. Pilot: paper Trace #2 mix under our v8 timing [P]

Setup: `Hybrid/test2` mix as replayed on 2026-11-24 (no ToolBench), Llama-3.2-1B, v8 calibration,
block 16, **33,177 blocks = 530,832 tokens** (the repro's token budget), `max-num-seqs` 32, CPU
blocks 0. Load = session-start rate × m (think times unchanged, as in our current tooling). OPT is
unrefined (no-cache map). Outputs: `result/advisor-qttft/t2-v8-x{1,3,4.5,6}/`.

Whole-run and steady-window TTFT. The window is arrivals in [200/m, 1,600/m) s, the period when every task is arriving.

| Load m | Policy | Hit ratio | Whole-run mean TTFT (ms) | vs LRU | Window mean (ms) | vs LRU | Window p50 (ms) | Window p90 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1× | LRU | 78.2% | 16.8 | 1.00 | 17.3 | 1.00 | 15.8 | 23.6 |
| 3× | NONE (no caching) | 0.0% | 26.0 | 1.01 | 26.8 | 0.95 | 17.5 | 35.3 |
| 3× | LRU | 74.0% | 25.8 | 1.00 | 28.2 | 1.00 | 17.4 | 31.0 |
| 3× | LFU | 43.2% | 32.0 | 1.24 | 35.2 | 1.25 | 20.0 | 47.6 |
| 3× | UniCache | 78.2% | 24.8 | 0.96 | 26.9 | 0.96 | 17.0 | 28.6 |
| 3× | OPT (unrefined) | 84.9% | 22.8 | 0.88 | 24.3 | **0.86** | 16.2 | 25.7 |
| 4.5× | NONE (no caching) | 0.0% | 367.9 | 0.54 | 514.8 | 0.53 | 31.2 | 2,026 |
| 4.5× | LRU | 73.3% | 681.2 | 1.00 | 978.2 | 1.00 | 65.2 | 2,955 |
| 4.5× | LFU | 43.2% | 744.1 | 1.09 | 1,069.7 | 1.09 | 299.4 | 3,198 |
| 4.5× | UniCache | 78.8% | 567.5 | 0.83 | 810.3 | 0.83 | 38.3 | 2,647 |
| 4.5× | OPT (unrefined, **map stale**) | 43.3% | 499.6 | 0.73 | 706.2 | 0.72 | 87.3 | 2,443 |
| 6× | LRU | 74.5% | 11,284 | 1.00 | 18,492 | 1.00 | 21,411 | 22,524 |

The 6× policy runs were stopped: LRU is already saturated there. Same 3× cell, scoring only arrivals ≥ 200 s (mostly
the multi-turn tail after the single-turn tasks end): NONE 23.8 ms (**1.19×**), LRU 20.0, UniCache 19.3 (0.965×),
OPT 18.0 (**0.90×**), LFU 26.8 (1.34×).

Reading:
* **Below the knee (3×) the window an agent could exploit is 10–14% (OPT) and ≈ 4% (UniCache).** That is the size
  of GPU noise at the knee (±9–17%) and about 2× the noise at light load (±2%). The downside (LFU +25–34%) is
  clearly measurable.
* **Whether caching itself pays depends on the phase.** While the single-turn tasks arrive, no caching ties LRU (0.95×).
  In the multi-turn tail it is 1.19×. Single-turn reuse is worth less than the eviction CPU it costs. Session reuse
  (ALFRED, ShareGPT) is worth something.
* **At the knee (4.5×) TTFT no longer tracks hit ratio.** A stale-map "OPT" with 43% hits is 0.72× LRU, and no caching
  is 0.53×. This regime measures queue dynamics in the opening burst, not eviction quality. Do not pick configs from it.

Busy-time breakdown of the LRU schedules, re-priced without feedback (same steps, two cost models):

| Schedule | v8: prefill share of busy | v8: cache-sensitive token work | A100-3B: prefill share | A100-3B: cache-sensitive token work |
|---|---:|---:|---:|---:|
| Step 3 eval, GPU LRU log | 24.0% | **1.8%** | 19.7% | **12.6%** |
| Trace #2, 1× | 6.0% | **0.3%** | 3.3% | **1.8%** |
| Trace #2, 3× | 13.9% | **1.0%** | 9.3% | **5.6%** |

Eviction vs compute at Trace #2, 3× (simulated seconds over the run):

| Policy | Hit ratio | Evictions | Eviction charge | Prefill model time | New prompt tokens |
|---|---:|---:|---:|---:|---:|
| NONE | 0% | 0 | 0.0 s | 122.5 s | 15.04 M |
| LRU | 74.0% | 314,768 | 18.1 s | 88.9 s | 3.91 M |
| LFU | 43.2% | 603,927 | 34.7 s | 99.9 s | 8.54 M |

## Appendix B. How the pilot numbers were made

Scripts are in the session scratchpad (not in the repo):
* `pilot.py <cell> <mult> <policies…>`: copies `result/paper-qttft/aic-a100/Trace#2/tasks_config.yaml`,
  multiplies every `lambda_rate` by `mult`, and runs `run_simulation.py` with the flags above plus
  `--simulator-step-log`. It runs NONE first when OPT is requested (for the use map).
* `summ.py <cells…>`: reads `latency_summary.json`, `summary.json` and `steps.csv`.
* `reprice.py <steps.csv>`: re-prices every step with v8's `Calibration.linear_ms` and with
  `AicStepTiming(Llama-3.2-3B, a100_sxm)`. Per-sequence token counts are split evenly within the
  step and rounded to 64. "Cache-sensitive token work" = step time − step time if every prompt
  token but one were cached.
* Isolated-request numbers: v8 `request_charge(n)` + `linear_ms` for one prompt with n new tokens
  vs 16 new and n − 16 cached; A100-3B `step_ms` for the same two batches.
