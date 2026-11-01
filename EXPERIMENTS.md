# Hit-ratio matrix

The matrix uses a separate Python 3.12 environment. This avoids Python 3.10's
process-dependent `hash(None)`, which makes the released OPT future-use maps
incompatible between profile and replay processes. No block-hashing or eviction
implementation changes are needed; `PYTHONHASHSEED=0` remains required.

Create the environment once (requires internet for dependencies):

```bash
bash setup-simulator312.sh
.venv-sim312/bin/python scripts/check_opt_hashes.py
```

This preserves `.venv` (Python 3.10). The setup uses the existing dependency
pins and installs the local source without native kernels. Prepared datasets
and model metadata must already exist locally. Each new matrix invocation
regenerates OPT profiles; do not reuse maps from Python 3.10. Previously
reported OPT rows from the Python 3.10 environment are invalid.

Run from the UniCache repository root; no environment activation is needed:

```bash
bash scripts/run-hit-ratio-matrix.sh
```

This runs the seven main paper baselines plus UniCache on five accessible
individual workloads. ToolBench is omitted at the user's request because
the authors' source requires UCSD access. No substitute dataset is used.

The runner does not modify eviction policies or report estimated QTTFT.

## Parallel execution

Run all five available workloads and all eight policies with up to two
simulator processes at a time:

```bash
bash scripts/run-hit-ratio-matrix.sh --jobs 2
```

`--jobs N` bounds simultaneous replays across both policies and workloads.
It defaults to 1 for serial execution. The launcher defaults to
`.venv-sim312/bin/python`; `SIMULATOR_PYTHON` can override the interpreter,
but the runner rejects Python below 3.12 when OPT is selected. Each replay has an independent process,
log, output directory, and writable plotting/vLLM caches. Start with 2 on a
Mac; larger values require more RAM because every process loads its own data
and Python environment. Parallel execution changes wall-clock runtime; the
virtual-time configuration, random seeds, cache budgets, and policies stay the same.

For offline OPT, each workload's uncached (`NONE`) trace-collection pass
must succeed before that workload's OPT replay starts. Other policies can run
while trace collection is in progress. A failed profile marks its OPT cell
`ERR` without stopping independent runs. Only the coordinator writes the
aggregate tables, which update as runs finish.

Ctrl-C or SIGTERM stops active replay process groups and preserves completed
cells. Unfinished cells remain blank; there is no automatic resume option.
To validate the selected inputs without launching replays:

```bash
bash scripts/run-hit-ratio-matrix.sh --jobs 2 --dry-run
```

Orchestration checks (synthetic summaries and a small cancellation subprocess;
no actual simulation):

```bash
.venv-sim312/bin/python scripts/test_hit_ratio_matrix.py
```

## Output

Each invocation prints a fresh directory under `result/hit-ratio-matrix/`:

- `hit_ratio_matrix.csv`: policy rows, workload columns, numeric ratios from 0 to 1.
- `hit_ratio_matrix.md`: the same table formatted as percentages.
- `results.json`: detailed hit counts, request counts, runtimes, and summary paths.
- `manifest.json`: model/dataset hashes, configurations, and experiment parameters.
- `<workload>/<policy>/run.log`: logs; nested directories contain upstream outputs.
- `<workload>/NONE/`: trace-collection pass for the offline OPT future-use map.

Rows: LRU, LFU, FIFO, LeCaR, ARC, WA, OPT, UniCache. UniCache uses the
repository policy name `SimpleHYBRID`. Cells use `global_hit_rate`, total
hit tokens divided by total prompt tokens, not an unweighted request average.

Tables update after each policy. An unrun cell is blank; a failed cell is
`ERR`. Failures never become zero hit ratios. The runner finishes the other
policies and exits unsuccessfully if any run fails.

## Workloads and common configuration

| Column | Prepared available sessions | Upstream standalone session limit |
|---|---:|---:|
| ShareGPT | 57,519 | 1,000 |
| MASH-QA | 21,718 | 5,000 |
| AgentBank-APPS | 4,408 | 4,408 |
| ALFRED | 623 | 500 |
| CodeParrot | 20,562 | 5,000 |

The standalone YAMLs are read unchanged, including task IDs, arrival rates,
inter-turn distributions, and seeds. ALFRED uses the original 623-session
dataset, not the separate 5,000-session expansion used in some hybrid configs.
Some upstream standalone task names are misleading (MASH-QA is named
`Codeparrot`); their original names are retained.

All columns use the same Llama-3.2-1B-Instruct tokenizer/configuration,
8-token blocks, 50,000 simulated GPU blocks (400,000 tokens), 32 maximum
sequences, and 131,072 maximum model length. This common capacity differs
from some authors' individual scripts; it is intentional for a common-budget
comparison. Timing defaults to the upstream simulator expression and 0.01
seconds per decode token. The existing simulator-only CPU block-size exception
allows 8-token blocks on this Mac without actual CPU inference.

To inspect without running:

```bash
bash scripts/run-hit-ratio-matrix.sh --dry-run
```

To run selected columns or another common capacity:

```bash
bash scripts/run-hit-ratio-matrix.sh --workloads ShareGPT MASH-QA --gpu-blocks 100000
```

`--max-records N` caps each existing session limit for a smaller experiment.
`--ttft 'expression'` and `--tpot SECONDS` override timing explicitly.

Offline OPT follows the authors' uncached (`NONE`) profile and prefill-order
future-use-map workflow with `PYTHONHASHSEED=0`. It is the released OPT
implementation, not a separately verified oracle for every policy-dependent
schedule. These individual workload experiments do not reconstruct the paper's
final mixed Traces #1–3 or establish exact Figure 1 settings.

## Data preparation provenance

ShareGPT and MASH-QA provenance is recorded beside those datasets.
`TaskData/baseline-dataset-provenance.json` records the newly prepared
AgentBank/APPS, ALFRED, and CodeParrot datasets and verifies that the authors'
preparation/tokenization code matches the upstream commit.

AgentBank was obtained by running the original `prepare_agentbank_data.py`.
CodeParrot was obtained by running the original `prepare_data.py`, including
its original `load_dataset` call. Both used the existing Python 3.10 environment.
All prepared session files use the unchanged `prepare_session_data.py` with
the Llama-3.2-1B-Instruct tokenizer. Original random sampling and UUID behavior
were retained; generated files are frozen by SHA-256 for policy comparisons,
but are not claimed to match the authors' exact sampled files.

## 1B cache-budget sweep

`bash scripts/run-cache-budget-sweep.sh --jobs 8` runs the eight main policies on
all five prepared workloads, on both frozen tune and eval splits, at 0.5A/0.7A/0.9A.
The model, timing formula, original standalone source limits, seeds and scheduler
settings match the preceding agent batch. No agent/API calls occur.

Memory convention: idealized A100 with 80 GiB total memory; A is total minus the
FP16 weight storage calculated from the local tied-embedding model configuration
(1,235,814,400 parameters). This does not reserve or measure runtime activation and
workspace memory. FP16 KV cache: 262,144 bytes per 8-token block, TP=1. Thus A is
77.6981163 GiB, and the three capacities are 159,125 / 222,776 / 286,426 blocks.
This is an explicit modeled normalization, not calibrated physical GPU reproduction.

Each policy replay validates actual request completion and prompt/response token
counts. OPT uses a separate no-cache profile for every budget/trace/split. Outputs:
`manifest.json` (memory conversion and input hashes), `hit_ratios.csv`, `headroom.csv`,
`report.md`, and `completion.json` (all cases required; failures never become zero).
Best online policy is chosen from tune scores and then reported on eval, preventing
selection on eval. OPT gaps are measured reference gaps, not guaranteed achievable
online gains. A small test can use `--traces sharegpt --sessions 4`.

### Headroom discovery protocol (1B only)

The physical-memory-normalized reference is run unchanged with:

```bash
bash scripts/run-cache-budget-sweep.sh --jobs 8
```

A smaller experiment preserves full tune workloads but reduces policy count and
omits eval. Its explicit budgets are NOT called 0.5A/0.7A/0.9A:

```bash
bash scripts/run-cache-budget-sweep.sh --jobs 4 \
  --blocks 16384 32768 50000 98304 \
  --policies LRU SimpleHYBRID OPT --splits tune
```

These are approximately 0.0515A / 0.1030A / 0.1571A / 0.3089A under the documented
80 GiB post-weight convention. Keeping complete tune workloads avoids creating a
misleadingly small working set by truncating sessions.

Selection is predeclared: per trace, maximize the demonstrated UniCache-minus-LRU
tune gain, break ties by larger OPT-minus-LRU tune gap and then larger budget. If no
positive online gain exists, use the largest oracle gap but label it oracle-only
evidence. Exclude failed/incomplete replays; inspect warnings/runtime before final
confirmation. Confirm selected budgets with all eight main policies and both splits.
Changing cache capacity does not change model timing, block size, request selection,
or seeds. No coding agents or API calls are part of this experiment.

Prerequisites: existing Python 3.12 simulator setup, local 1B model metadata/tokenizer,
and the five prepared datasets (see POLICY_AGENT.md and dataset installation/provenance
files). The run manifest pins full source-data checksums and each frozen split's
configuration/data hashes. Source data and weights are not bundled into git; model
weights are not needed. Exact generated split inputs are deterministic from the
pinned source files, source limits and seeds.

After the pilot has completed, the deterministic selection and confirmation are:

```bash
.venv-sim312/bin/python scripts/select_cache_budgets.py \
  --pilot result/cache-budget-study/2026-11-01/pilot \
  --output result/cache-budget-study/2026-11-01/selection.json
bash result/cache-budget-study/2026-11-01/selection.sh
```

The selector refuses incomplete pilots or pilots including eval. It saves all
candidate scores, selected budgets, input hash and exact confirmation commands.
Its generated shell script runs the five confirmations sequentially, four replays
at a time, each with the full main-policy set and both frozen splits. Use a fresh
output root to repeat the complete study without overwriting prior results.
Each actual run also has `reproduce.json` and a source snapshot; no manual spreadsheet
editing or undocumented budget selection is required.

Prepared-input portability: the study archives the exact source session prefixes and
model metadata once under `result/cache-budget-study/2026-11-01/inputs/`, with split
IDs/configs/hashes. Raw preparation scripts included randomness, so use the archive
rather than expecting fresh raw preprocessing to reproduce those sessions exactly.
The archive is about 628 MiB; unused source rows and duplicate split contents are omitted.
Transfer the study directory to another checkout, install the Python 3.12 environment
using `bash setup-simulator312.sh`, then verify/restore missing inputs:

```bash
.venv-sim312/bin/python scripts/restore_budget_study_inputs.py \
  --study result/cache-budget-study/2026-11-01 --restore-missing
```

The helper verifies hashes and refuses to overwrite differing existing files. Because
unused source rows are omitted, the restored full-file source checksum differs from
the original full dataset, but the consumed session prefix and regenerated frozen
split bytes match. Every command's outputs use a fresh directory; wall-clock timings
may vary by machine, whereas request/timing seeds and simulated timing are fixed.

The headroom-focused study also adds two explicit exploratory counterpoints at
32,768 blocks, chosen using tune curves only and recorded in decisions.json:

```bash
bash scripts/run-cache-budget-sweep.sh --jobs 4 --blocks 32768 \
  --traces mashqa codeparrot \
  --output result/cache-budget-study/2026-11-01/confirm-counterpoints
```

These retain the primary selections and measure the tradeoff between demonstrated
online gains and larger oracle gaps. Generate all tables/graphs only after primary
and counterpoint confirmations finish:

```bash
.venv-sim312/bin/python scripts/report_cache_budget_study.py \
  --study result/cache-budget-study/2026-11-01
```
