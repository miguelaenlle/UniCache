# Hit-ratio matrix

Run from the UniCache repository root; no environment activation is needed:

```bash
bash scripts/run-hit-ratio-matrix.sh
```

This runs the seven main paper baselines plus UniCache on five accessible
individual workloads. ToolBench is omitted at the user's request because
the authors' source requires UCSD access. No substitute dataset is used.

The runner does not modify eviction policies or report estimated QTTFT.

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
