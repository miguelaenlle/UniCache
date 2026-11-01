# Policy agent v1

Requires macOS, the `.venv-sim312` simulator environment, prepared ShareGPT data,
and the Claude Code CLI. No weights or GPU are used. The coding model uses the
Claude service; simulator replays remain offline.

The isolated CLI needs `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in your shell.
For subscription authentication, run `claude setup-token` yourself and export the
result as `CLAUDE_CODE_OAUTH_TOKEN`. Your regular Claude home directory is not mounted.
Do not put credentials in repository files.

```bash
./policy-agent run \
  --trace sharegpt \
  --serving_model llama-3.2-1b-instruct \
  --coding_model claude:opus \
  --start-policy UniCache \
  --compare LRU,LFU,FIFO,LeCaR,ARC,WA,OPT,UniCache \
  --max-submissions 30
```

Add `--prepare-only` to inspect the sandbox without simulations or API calls.
For a small integration trial, add `--sessions 4 --compare UniCache --max-submissions 1`.

Default source subset: the authors' standalone ShareGPT limit of 1,000 sessions.
The first 1,000 source sessions are assigned the original generator's seeded shuffled
arrival order, then split 500/500 by simulated session start. This is not genuine
collection chronology. Entire sessions stay together. Prepared data, frozen first
arrivals, split IDs and checksums live in `TaskData/SharedGPT-multi-turn/agent-splits/`.
Later-turn gaps retain the upstream per-session seeded, completion-relative behavior.

Results are printed at startup under `result/policy-agent/YYYY-MM-DD/<run-id>/`:

- `hit-ratio.png`: tune/eval progression and baseline lines.
- `comparison.csv`: baselines versus the selected policy, both split hit ratios.
- `policy.py`: selected submitted policy, chosen only by tune score.
- `submissions.jsonl`, `run.json`, `selection.json`: scores and reproducibility records.
- `claude.log`: full agent stream only when `--debug` is supplied. By default actions stream to the terminal without being saved. Claude session persistence and diagnostic file logging are disabled. Private simulator replay logs remain in `internal/`.

Claude sees only its prompt/interface/context, candidate, rationale, and tune history.
Comparison-policy names/scores and eval results stay outside its context (except the
chosen starting policy). The default cap is 30 accepted attempts; failed attempts
count. `python submit.py --done --reason "done"` finishes early.

Claude can run these commands without spending submission attempts:

```bash
python smoke_test.py
python run_local.py tests/my_test.py
python run_local.py tests/my_test.py argument1
```

The built-in smoke test checks the current policy's interface, updates/removals and
legal eviction decisions using synthetic events. It reports tracebacks for failures.
It does not measure trace hit ratio or replace submission validation.

Claude may author custom Python scripts anywhere inside its workspace, then execute
them through `run_local.py`. The trusted evaluator launches each script in an offline
Seatbelt process with standard-library Python and access to workspace files (including
`policy.py`), excluding the coding session's home/credentials. Scripts can write only
to `local-output/`. Limits: 30 seconds wall time, 10 seconds CPU, 1 MiB captured output,
10 MiB per output file; no strict total disk/memory cap. Processes cannot fork.
Stdout and tracebacks are returned directly, not kept as per-test logs. Local-test spool
requests/responses are deleted after the client reads them. Files deliberately created
by custom tests under `local-output/` remain until the user/agent removes them.

Add `--debug` to the outer run command to retain the full agent event stream. This does
not change saved submission rationales/scores/policy snapshots or simulator replay logs.
Existing runs retain their previous logging and command permissions; restart to use this.

The policy is an **eviction controller** around the chosen built-in policy, preserving
its allocator tracking hooks. `Policy.on_event(event)` maintains state;
`Policy.evict(now)` returns an eligible block ID or `None` to retain the starting
policy's decision. The unchanged controller must match baseline scores before Claude
starts. V1 does not copy the full UniCache internals into the editable file.

Validation checks completed request identities/response lengths, per-request prompt
and hit-token bounds, aggregate totals, and eviction eligibility. The trusted allocator
continues enforcing its normal block/hash/reference-count invariants. These are replay
checks, not a separate token-by-token semantic verification of model outputs.

Claude runs directly with `--restricted --safe-mode`, an empty home, no inherited
project instructions/plugins/MCP servers, and no prior conversation. File tools are
confined by Claude to its working directory. Only Read/Edit/Write and the allowlisted submission, smoke-test, and local-test
Bash commands are enabled; other commands are denied without
prompting. Web tools are unavailable. Claude still needs its API connection.

This is a practical tool-level boundary, not OS isolation of the coding agent. It does
not prevent every possible escape or tampering with local feedback; authoritative
scores and budgets remain in the external evaluator. Do not treat it as hostile-code
hosting. Claude startup stops after 90 seconds if no output is received.

Submitted policies still execute separately under macOS Seatbelt, with no network,
user-data reads or filesystem writes, plus callback/CPU deadlines. System/runtime
files and directory names remain readable. A strict policy memory cap is not yet
implemented. Policy execution refuses to run without Seatbelt.

Local isolation checks (run from a normal Terminal, not another Seatbelt sandbox):

```bash
.venv-sim312/bin/python -m unittest policy_agent.test_sandbox policy_agent.test_local_runner -v
```

## All workloads and all comparisons

```bash
bash scripts/run-policy-agent-all-traces.sh
```

This sequential batch starts from LRU, uses Claude Opus, allows at most 30 candidate
attempts per workload (early completion remains allowed), and evaluates every
available comparison policy before optimization. No raw agent logs are retained.

| Trace ID | Source sessions | Tune / eval |
|---|---:|---:|
| sharegpt | 1,000 | 500 / 500 |
| mashqa | 5,000 | 2,500 / 2,500 |
| agentbank-apps | 4,408 | 2,204 / 2,204 |
| alfred | 500 | 250 / 250 |
| codeparrot | 5,000 | 2,500 / 2,500 |

These preserve the authors' standalone source limits. ToolBench remains omitted
because the source requires unavailable UCSD access. Each workload has its own
prepared split directory and independent agent session.

`--compare all` includes LRU, LFU, FIFO, LeCaR, ARC, WA, OPT, UniCache,
OPT-Online, SimpleHYBRID-OPT, OFFSET-HEAD, OFFSET-TAIL, and NONE. The last is a
prefix-caching-disabled control; the online/offset variants are experimental
comparisons, not additional main-paper baselines.

`--require-baselines OPT` prevents starting the coding agent unless offline OPT
has valid results for both splits. OPT's profiling data and future-use map live
under the host-only `internal/baselines/OPT/` directory. The agent receives no
baseline scores, baseline implementations, future-use maps, or eval feedback.
Its starting LRU policy and own submission-0 tune score are necessarily visible.
The coding boundary is Claude's restricted tools (a practical guardrail, not a
hardened OS boundary); submitted policies and custom tests retain Seatbelt isolation.

For a single workload:

```bash
./policy-agent run --trace mashqa \
  --serving_model llama-3.2-1b-instruct --coding_model claude:opus \
  --start-policy LRU --compare all --require-baselines OPT --max-submissions 30
```

Per-workload output directories end with the trace ID. The driver runs workloads
one at a time and stops on a failed run instead of silently skipping it.
A small real simulator preflight (no API calls) is available as:

```bash
.venv-sim312/bin/python scripts/check-agent-traces.py
```

## Reproducible summary tables

Create a summary from explicit completed run directories, one per workload:

```bash
bash scripts/summarize-policy-agent.sh --runs \
  result/policy-agent/<date>/<sharegpt-run> \
  result/policy-agent/<date>/<mashqa-run> \
  result/policy-agent/<date>/<apps-run> \
  result/policy-agent/<date>/<alfred-run> \
  result/policy-agent/<date>/<codeparrot-run>
```

The command never chooses the latest run implicitly. It rejects duplicate workloads,
mixed models/cache budgets/starting policies, failed runs, inconsistent scores, or a
final policy that does not match the selected tune-winning snapshot.

Outputs are `eval.csv`, `tune.csv` (policy rows, workload columns; ratios 0–1),
`summary.md` (percentage tables), and `runs.csv` (attempt counts, selected submission,
source directories and stop reasons). Each Agent cell represents a different policy
trained for that workload. Missing/failed values are not replaced with zero.

The summary saves small input snapshots and their SHA256 hashes in `manifest.json`.
Rebuild using those snapshots, even if original run folders subsequently change:

```bash
bash scripts/summarize-policy-agent.sh \
  --manifest result/policy-agent-summary/lru-opus-all/manifest.json
```

A fresh timestamped output folder is created. Snapshot tampering fails verification.
The manifest also records the summarizer's code hash. No experiments or API calls are
performed by the summary command. The completed five-workload batch's initial report
is `result/policy-agent-summary/lru-opus-all/summary.md`.
