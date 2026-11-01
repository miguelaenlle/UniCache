# Agent/cache-budget experiment (v1)

This experiment deliberately freezes the previous agent interface. The expanded v2
interface is a separate experiment. No hidden v2 metadata is added here.

Defaults: five available workloads, 16,384 / 32,768 / 50,000 / 98,304 GPU blocks,
Llama-3.2-1B-Instruct, block size 8, LRU starting policy, Claude Opus, 30 submissions
maximum per workload/budget. The agent may stop early. All 13 built-in controls are
compared, including host-only offline OPT. ToolBench remains inaccessible and excluded.

Restore the exact prepared data first (from the preceding archived budget study):

```bash
python3 scripts/restore_budget_study_inputs.py \
  --study /absolute/path/to/cache-budget-study/2026-11-01 --restore-missing
```

Use the existing Python 3.12 simulator environment. All splits are whole-session
50/50 by simulated session start, using the previous deterministic seed/generator.
Each tune/eval replay starts cold. The model timing approximation, scheduler limits,
source session limits and trace data remain fixed across cache budgets.

```bash
OUT=result/agent-cache-budget-study/my-run
bash scripts/run-agent-cache-budget-sweep.sh --stage baselines --jobs 8 --output "$OUT"
claude auth login
bash scripts/run-agent-cache-budget-sweep.sh --stage agents --jobs 2 \
  --normal-cli-auth --output "$OUT"
bash scripts/run-agent-cache-budget-sweep.sh --stage report --output "$OUT"
```

The normal login option lets the Claude CLI retrieve its own existing login through
normal HOME and public OS account variables. No credential is extracted or copied.
`--safe-mode` disables CLAUDE.md, memories/customizations and plugins;
`--restricted` confines file tools to the sandbox directory; Bash remains restricted
to submit/local-test clients. Coding-agent isolation is a tool-permission boundary,
not an OS filesystem guarantee. Submitted policies and custom tests remain offline
and isolated by Seatbelt. Alternatively export a normal `ANTHROPIC_API_KEY` or
`CLAUDE_CODE_OAUTH_TOKEN` and omit `--normal-cli-auth` to retain the original clean HOME.

The baseline command must run outside a parent sandbox that blocks OpenMP shared
memory. It can run concurrently with the agent command by adding
`--wait-for-baselines` to the latter. The agent command waits for all comparisons of
each workload/budget before starting its corresponding coding run.

Baselines run once and are privately checksummed/reused. Every submitted candidate
still runs through the normal tune/eval evaluator. Baseline scores, eval feedback,
trace files and the OPT oracle are never placed in the coding workspace.

Output files:

- `manifest.json`: frozen runtime revision/digest, budgets, traces and policy list.
- `prepared/`: frozen split paths, hashes and expected completion ledger.
- `baselines/`: private validated aggregate scores and replay-artifact checksums.
- `agent-logs/`: whitelisted host progress only (submission/replay/status/heartbeats);
  coding-model text, thinking and tool calls are filtered before persistence.
- `agent-runs/`: explicit selected run paths and submission counts.
- `scores.csv`, `report.md`: tune/eval policy-by-workload tables for every budget.
- `headroom.csv`: OPT−LRU, Agent−LRU and their ratio for each case/split.
- `hit-ratio-by-budget.png`: eval LRU/Agent/OPT curves.
- `*-completion.json`: stage completion and failures; missing cells mean pending.

The reported OPT gap is a privileged reference gap, not guaranteed attainable by an
online algorithm. A matching or better score on one seed is exploratory evidence.
Selection uses tune only; eval remains hidden from Claude. Existing eval data has
already been explored, so this is not a fresh blind holdout or a confidence interval.

Resume the same command with the same output folder; completed valid baselines and
agent runs are reused. The experiment rejects changed runtime/input identities. If a
replay fails partway, inspect its log and move that incomplete replay directory to a
diagnostic folder before retrying; no failed score is silently treated as zero.

Each active coding case prints a heartbeat every 30 seconds and has a two-hour
wall-clock deadline. On deadline, its own process group is terminated and the case
is reported as failed, not as a completed optimization result.

Submission commands run synchronously: the CLI receives
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` and both `BASH_DEFAULT_TIMEOUT_MS` and
`BASH_MAX_TIMEOUT_MS` set to `1800000`. These are supported
[Claude Code environment settings](https://code.claude.com/docs/en/env-vars).
This prevents a long replay being backgrounded while the coding model exits before
seeing its feedback. A zero CLI exit code is insufficient: stop time must follow the
last submission's feedback/history update, otherwise the optimization is rejected
as `agent_exited_while_submission_pending` and must be retried. Historical runs must
pass the same check before reuse.

For this study, five existing 50,000-block coding runs can be reused after strict
verification and fresh selected-policy replays; only the other 15 coding runs are
new. Before the agent stage, run:

```bash
.venv-sim312/bin/python scripts/import_historical_budget_agents.py \
  --study "$OUT" \
  --summary-manifest /absolute/path/to/policy-agent-summary/lru-opus-all/manifest.json
```

The importer verifies source/input/model/config checksums, normalized frozen-arrival
configs, unchanged simulator/controller code, evaluator/prompt/interface ASTs, Python
version and tune-only policy selection. It then requires exact tune/eval score
reproduction by fresh candidate replays before recording historical reuse. A mismatch
requires a fresh coding run for that case. Historical dates/attempts/source snapshots
and fresh verification scores remain explicit in `agent-runs` and `agent-origins.json`.

Reproduce a final selected policy without any Claude/API call:

```bash
.venv-sim312/bin/python scripts/replay_budget_policy.py --study "$OUT" \
  --blocks 32768 --trace alfred --output result/replay-alfred-32768
```

This checks the selected source digest and requires exact recorded tune/eval scores,
including complete request/token accounting. Run outside a parent sandbox that blocks
OpenMP. Replaying frozen source is deterministic under the pinned environment; a new
Opus coding run can produce different policies and stop at a different attempt.
