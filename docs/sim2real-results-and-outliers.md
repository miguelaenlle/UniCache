# Sim-to-real results and outliers

Date: November 3, 2026.

## Scope

We replayed the supplied subsets using the native UniCache CPU simulator and
compared their token-weighted prefix-cache hit ratios with the supplied A100
chart. Reported GPU setup: Llama-3.2-1B-Instruct, one A100 worker. The archived
configuration specifies 512 blocks × 16 tokens, maximum context 8,192, maximum
sequences 32, tensor parallel degree 1, and no CPU cache.

Five workloads × 12 policies plus five NONE runs for offline-OPT profiling
produced 65 successful replays. All 6,305 request completions passed exact prompt
and output-token equality checks. These are repeated executions of 485 distinct
requests across policies, not 6,305 distinct requests.

GPU references are manually transcribed chart values rounded to 0.01 percentage
points (pp). We excluded seven ALFRED values explicitly labeled estimates. Raw
GPU logs, effective engine settings, and timestamp mode are unavailable.

## Results

Errors below compare the 11 non-OPT policy labels per workload, except ALFRED,
which has only five measured policy labels. MAE is an unweighted mean across
those policy labels. Some labels share behavior; these are not independent trials.

| Workload | Sessions / requests | Compared policies | Mean absolute error (pp) | Maximum absolute error (pp) | Within 1 pp |
|---|---:|---:|---:|---:|---:|
| ShareGPT | 20 / 82 | 11 | 51.161 | 55.749 | 0/11 |
| ALFRED | 2 / 103 | 5 | 0.047 | 0.047 | 5/5 |
| CodeParrot | 100 / 100 | 11 | 1.450 | 6.334 | 8/11 |
| AgentBank APPS | 100 / 100 | 11 | 0.148 | 0.323 | 11/11 |
| MASH-QA | 100 / 100 | 11 | 1.208 | 3.815 | 6/11 |

UniCache's improvement over LRU transfers more closely for several workloads:

| Workload | GPU UniCache − LRU (pp) | Simulator UniCache − LRU (pp) | Simulator − GPU improvement (pp) |
|---|---:|---:|---:|
| ShareGPT | −0.100 | 0.000 | +0.100 |
| CodeParrot | +19.200 | +18.505 | −0.695 |
| AgentBank APPS | +0.690 | +1.016 | +0.326 |
| MASH-QA | +3.920 | +3.481 | −0.439 |

ALFRED is omitted from this improvement table because its GPU UniCache value is
an estimate, not a completed measurement.

## Outliers and possible explanations

**ShareGPT: large systematic gap.** LRU is 38.590% in simulation versus 90.57%
in the GPU chart, a −51.980 pp difference. Every policy is substantially lower
in the primary simulator run. This suggests a shared workload/timing/execution
difference rather than an isolated eviction-policy issue, but does not prove its
cause.

We tested timing sensitivity while preserving all 82 requests, LRU, cache size,
arrival parameters, and the prefill formula:

| Simulated decode-step time | Simulator hit ratio (%) | GPU chart (%) |
|---|---:|---:|
| 10 ms, original setting | 38.590 | 90.57 |
| 3 ms, diagnostic | 92.007 | 90.57 |
| 1 ms, diagnostic | 93.200 | 90.57 |

ShareGPT follow-ups are scheduled after the preceding response completes, with
zero configured thinking delay. Timing therefore changes interleaving and cache
competition even when the initial arrival-rate setting is unchanged. The probes
demonstrate sensitivity; they are not measured A100 calibration and do not
replace the primary result. We have not established whether the GPU run used
wall-clock time, virtual time, or a different replay loop. A GPU device by itself
does not establish the timestamp source.

**Other online-policy outliers.** CodeParrot ARC differs by +5.188 pp and
OFFSET-HEAD by +6.334 pp. MASH-QA OFFSET-HEAD differs by −3.815 pp and LFU by
−2.771 pp. Execution order, tie-breaking, policy state, effective defaults, and
measurement definitions need checking before assigning a cause.

**The GPU OPT series is not a verified upper bound.** For CodeParrot it reports
45.55%, below UniCache's 64.75%; for MASH-QA it reports 27.25%, below several
online policies. Our separately profiled native simulator OPT yields 75.110%
and 52.723%, respectively. Future-use maps, request/block identity, fallback
behavior, and the exact OPT implementation must be checked. We exclude OPT from
the online-policy error summary, while retaining it in the full comparison.

**ALFRED's agreement is limited evidence.** Only two sessions were used. All
simulator policies obtain 97.283%, while the five measured GPU labels report
97.33%. This subset does not demonstrate accurate behavior under strong eviction
competition. CodeParrot also has only 18 distinct full prompts across 100 requests.

## What we can conclude

APPS and measured ALFRED values agree closely; CodeParrot and MASH-QA broadly
preserve UniCache's benefit over LRU, with some policy-specific discrepancies.
ShareGPT does not validate under the original timing approximation. Exact CPU
token replay has been verified, but GPU execution order and completion coverage
have not. There are no latency measurements here, no agent-policy transfer test,
and no repeated GPU runs supporting uncertainty estimates.

Next, request the resolved GPU configuration and timestamp/completion logs using
[the parameter checklist](sim2real-parameters-and-gpu-agent-request.md). Establish
clock mode and arrival semantics before changing timing coefficients.

## Reproduce and inspect

From the repository root:

```bash
bash scripts/run-sim2real-subset.sh \
  --output result/sim2real-subset/REPRODUCE-NEW --jobs 4
.venv-sim312/bin/python scripts/probe_sim2real_timing.py \
  --study result/sim2real-subset/REPRODUCE-NEW
python3 scripts/summarize_sim2real_subset.py \
  --study result/sim2real-subset/REPRODUCE-NEW
```

The completed study is in
`result/sim2real-subset/2026-11-03-friend-512-verified/`: `comparison.csv`,
`comparison.png`, `comparison.pdf`, `summary.md`, and `manifest.json`. It retains
the reference chart, inputs, commands, code snapshots, hashes, and per-run logs.
See [the reproduction guide](sim2real-subset.md) for archive preparation.
