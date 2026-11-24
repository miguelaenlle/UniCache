# Replaying the supplied A100 subsets

This study compares the native CPU simulator with the user's GPU chart for
Llama-3.2-1B-Instruct on one A100 worker. It evaluates token-weighted prefix
cache hit ratio. The chart contains no TTFT/ITL measurements or raw GPU logs.

## Inputs and settings

The supplied archive is `unicache-tested-traces.tar.gz`, SHA-256:
`94149333d8335466b92839f25beebc522241c2186bac23cf012087c3a4191a23`.
Its 38 listed file checksums are verified before replay. The workload subsets are:

| Workload | Sessions | Requests |
|---|---:|---:|
| ShareGPT | 20 | 82 |
| ALFRED | 2 | 103 |
| CodeParrot | 100 | 100 |
| AgentBank APPS | 100 | 100 |
| MASH-QA | 100 | 100 |

Use **512 blocks × 16 tokens**, 8,192 maximum context length, 32 maximum
sequences, no CPU cache, and tensor parallel degree 1. This differs from the
earlier agent study's 8-token blocks. Original session records, arrival parameters,
random seeds, and starter TTFT/TPOT functions are retained. Only local paths are
adapted for the YAML loader. Local model/tokenizer metadata are checked against
the bundle; this simulation does not load model weights or execute on an A100.

Each finished simulator request must reproduce its archived prompt and output
token arrays exactly. Missing or duplicate requests, inconsistent hit counters,
and modified recorded artifacts fail validation. These checks validate CPU replay;
they cannot establish what completed on the GPU without its raw logs.

## Run

From the repository root, using the existing `.venv-sim312` environment and
`models/Llama-3.2-1B-Instruct` metadata directory:

```bash
# Already extracted on the current machine. To restore from the original archive:
mkdir -p result/sim2real-inputs/unicache-tested-traces-94149333d833
tar -xzf ~/Downloads/unicache-tested-traces.tar.gz \
  -C result/sim2real-inputs/unicache-tested-traces-94149333d833

# Use a fresh output directory for a new run.
bash scripts/run-sim2real-subset.sh \
  --output result/sim2real-subset/REPRODUCE-NEW --jobs 4

# Optional timing diagnostic, separate from the primary experiment.
.venv-sim312/bin/python scripts/probe_sim2real_timing.py \
  --study result/sim2real-subset/REPRODUCE-NEW

python3 scripts/summarize_sim2real_subset.py \
  --study result/sim2real-subset/REPRODUCE-NEW
```

The matrix runs 12 policies per workload plus one NONE profile per workload for
offline OPT: 65 replays total. The UniCache label selects native `SimpleHYBRID`.
All evaluations use native eviction implementations, without a candidate policy.
There is no agent search or new GPU execution. Four workloads run concurrently;
policies within a workload run sequentially. Existing successful runs can resume
if the recorded source files have not changed.

To regenerate tables and plots without rerunning simulations:

```bash
bash scripts/run-sim2real-subset.sh --stage compare \
  --output result/sim2real-subset/REPRODUCE-NEW
python3 scripts/summarize_sim2real_subset.py \
  --study result/sim2real-subset/REPRODUCE-NEW
```

## Outputs and interpretation

`summary.md` contains workload error statistics and UniCache-versus-LRU effects.
`comparison.csv`, `comparison.png`, and `comparison.pdf` contain all policies.
`manifest.json` and `code-snapshot/` record input/source hashes and dependency
versions. Each replay retains its command, log, exact completion ledger, and
native simulator outputs.

GPU values are manually transcribed from the supplied chart, rounded to 0.01
percentage points. Seven ALFRED bars explicitly labeled estimates are excluded
from transfer statistics. Offline OPT is also excluded from the online-policy
summary: the chart's OPT series falls below several online policies and its
future-use metadata have not been verified. Its values remain visible in the
full table.

The completed study is
`result/sim2real-subset/2026-11-03-friend-512-verified`. All 6,305 request
completions passed exact token checks. Online-policy mean absolute errors were
51.161 pp (ShareGPT), 0.047 pp (ALFRED), 1.450 pp (CodeParrot), 0.148 pp (APPS),
and 1.208 pp (MASH-QA).

ShareGPT is timing sensitive: changing only simulated TPOT from 10 ms to 3 ms
changed LRU hit ratio from 38.590% to 92.007%; the chart reports 90.57%.
These probes are sensitivity checks, not measured A100 calibration. Later turns
arrive after request completion, so timing changes interleaving and cache
competition. Exact input matching alone does not establish timing fidelity or
uniform sim-to-real agreement. GPU scheduling/completion logs would help isolate
the remaining differences.
