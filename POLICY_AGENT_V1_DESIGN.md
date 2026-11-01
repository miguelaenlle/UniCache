# UniCache policy agent: v1 design

Status: design reference. A local v1 is implemented; see POLICY_AGENT.md for the runnable command, implemented controller interface, checks, and current limitations. Some stronger isolation/validation requirements below remain design targets. The coding agent now uses Claude restricted mode and an isolated home, not an OS sandbox; submitted policy execution retains Seatbelt isolation.

## Objective

Build a policy optimization loop around the existing UniCache simulator. The agent starts from an existing eviction policy and optimizes overall token-weighted cache hit ratio on a fixed tune split. A separate eval split measures generalization.

V1 reports aggregate cache-hit statistics. It does not estimate TTFT or QTTFT.

## Command

Run from the repository:

```bash
cd "/Users/miguel/Documents/ChatGPT/Agentic GPU Environment/UniCache"

./policy-agent run \
  --trace sharegpt \
  --serving_model llama-3.2-1b-instruct \
  --coding_model claude:opus \
  --start-policy UniCache \
  --compare LRU,LFU,FIFO,LeCaR,ARC,WA,OPT,UniCache \
  --max-submissions 30
```

The command:

1. Resolves `sharegpt` to its registered trace manifest and fixed tune/eval splits.
2. Loads trace data tokenized for the selected serving model.
3. Initializes the agent's `policy.py` from the selected starting policy.
4. Evaluates the starting policy and comparison policies on both splits.
5. Runs the agent for up to 30 submissions.
6. Selects the valid submission with the highest tune score and writes the final artifacts.

`--serving_model` selects the simulated serving model. `--coding_model` selects the model implementing the policy. V1 accepts only `claude:<model-id-or-alias>` and invokes the installed Claude Code CLI directly; it rejects other coding providers. `claude:opus` passes `opus` to Claude's `--model` option. Record the requested identifier, CLI version, and resolved model identifier when exposed by the CLI in `run.json`. `UniCache` maps to the repository's `SimpleHYBRID` implementation.

Block size, cache capacity, arrival timing, and simulator settings come from the trace manifest and remain fixed across submissions and baselines.

## Trace registration and split storage

Each workload has a manifest beside its existing prepared data:

```text
TaskData/SharedGPT-multi-turn/
  trace.yaml
  splits/
    tune-session-ids.json
    eval-session-ids.json
  ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
```

The trace registry maps the CLI name `sharegpt` directly to this manifest. An illustrative manifest is:

```yaml
id: sharegpt
version: 1

models:
  llama-3.2-1b-instruct:
    data: ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
    tokenizer: ../../models/Llama-3.2-1B-Instruct
    # The implemented manifest also records source and tokenizer checksums.

splits:
  tune: splits/tune-session-ids.json
  eval: splits/eval-session-ids.json

arrival:
  lambda_rate: 1
  mu: 4.15
  sigma: 0.971
  seed: 42

simulator:
  block_size: 8
  gpu_blocks: 50000
  cpu_blocks: 50000
  max_num_seqs: 32
  max_model_len: 131072
```

Paths resolve relative to the manifest. The runner verifies checksums, selects the listed sessions, and materializes simulator inputs outside the agent sandbox. The complete implemented preset must also freeze virtual-time timing parameters and other settings that can affect scheduling and cache behavior.

Split preparation follows these rules:

- Partition whole sessions, keeping every conversation's turns together.
- Split 50/50 chronologically by session start: the earliest `floor(N / 2)` sessions form tune; the remaining sessions form eval. Do not shuffle. Break equal start-time ties by original source row index. Store the ordering basis, boundary, and selected session IDs.
- For traces without genuine timestamps, construct one complete, seeded session-arrival timeline using the existing workload generator, freeze it, and split by those generated session starts. Label this explicitly as simulated chronology, not original collection chronology. Source-file order alone is not evidence of real chronology.
- Preserve full sessions even if their turns cross the split boundary. This is chronology by session start, not a strict wall-clock cutoff with no overlapping turns. Record that convention in the manifest.
- Check that the two splits are disjoint and that all selected IDs exist in the source data.
- Generate and persist the full arrival timeline before splitting, then materialize each split without resampling arrivals. Rebase each split to its own time origin while preserving internal time gaps; reuse exactly those arrivals for every policy evaluation.
- Begin each evaluation with an empty cache and fresh policy state.
- Process the complete selected split without silently applying an additional record limit.

These are our experiment splits, not a claim that the authors used the same partition. The manifest must record any source subset or record limit used before splitting.

Selecting another serving model requires registered trace data tokenized for that model. Reject unsupported trace/model combinations instead of silently reusing another model's token IDs.

## Agent sandbox

The agent sees only:

```text
sandbox/
  prompt.md
  policy_api.md
  policy.py
  rationale.md
  submit.py
  smoke_test.py
  run_local.py
  local_client.py
  tests/
  local-output/
  run_context.json
  history.jsonl
  feedback/
```

`prompt.md` contains the concise task prompt shown below. `policy.py` and `rationale.md` are the files Claude fills out. `submit.py` is the read-only submission command.

`history.jsonl` is an evaluator-maintained log of previous submissions in this optimization run, starting with submission 0. Each completed attempt appends its submission number, policy hash, rationale, status, aggregate tune metrics, and remaining budget. Failures append a sanitized error. Claude can read the complete history to compare its previous strategies. A resumed run restores the same log; unrelated experiment histories are not supplied automatically.

The log and per-submission feedback files are read-only to Claude and contain no eval results or comparison-policy scores. The evaluator also preserves an authoritative copy outside the sandbox. Raw simulator logs remain outside the sandbox.

The agent can read the interface and starting implementation, modify `policy.py`, submit a candidate with a rationale, and read aggregate tune feedback.

Claude's restricted file tools stay within the working directory. Its permitted Bash commands are submission, `python smoke_test.py`, and `python run_local.py <script.py> [args...]`; other commands are denied without prompting. Safe mode, a clean HOME, and no resume option prevent inherited personal instructions or conversation history. Web tools are disabled. Claude still connects to its API. This is a practical local guardrail, not protection against every deliberate escape.

The evaluator keeps authoritative scores and budgets outside that directory and never trusts local feedback. Treat prompt, client, history, and feedback as read-only by convention; only policy/rationale and scratch files should be edited. Submitted policy execution retains filesystem and network isolation through Seatbelt.

## Claude launch and exact agent context

The runner starts Claude Code directly in the generated sandbox directory, with a clean HOME. Its file tools are confined by restricted mode; only the submission command is allowlisted for Bash. Submitted policy code runs separately under OS isolation. Conceptually, the direct command is:

```bash
claude --print \
  --model opus \
  --restricted --safe-mode --no-session-persistence \
  --debug-file /dev/null \
  --permission-mode dontAsk --permission-prompts none \
  --setting-sources "" \
  --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
  --tools "Read,Edit,Write,Bash" \
  --allowedTools "Read,Edit,Write,Bash(python submit.py *),Bash(python smoke_test.py),Bash(python run_local.py *)" \
  --append-system-prompt "$(cat /sandbox/prompt.md)" \
  "Read policy_api.md, run_context.json, and policy.py. Follow prompt.md."
```

The runner constructs subprocess arguments directly rather than interpolating user-provided text into a shell command. The example shows the intended CLI invocation; Claude tool permissions are the coding boundary in this local v1; they are not an OS security boundary. The policy worker retains OS isolation. Web tools are unavailable; only the API connection is required. Silent CLI startup times out after 90 seconds. Disable inherited plugins, MCP servers, hooks, project instructions, and personal memory in the isolated CLI configuration. Claude can also author custom Python test scripts inside the workspace and run them through the local launcher. This command launches Claude Code, not a separate homegrown LLM tool loop.

The exact task text rendered into `prompt.md` is:

```text
Improve policy.py to increase cache hit ratio on the tune split.
Read policy_api.md and run_context.json. Use past and current cache information only.
Check your policy with: python smoke_test.py
You may write custom Python tests and run: python run_local.py tests/my_test.py
Local tests are offline, use no traces, and do not consume submission attempts.
Explain each change briefly in rationale.md, then run:
python submit.py --rationale-file rationale.md
Read the feedback and history.jsonl, then iterate. You get {max_submissions} attempts;
accepted failures count. Eval results are hidden. Your best tune submission is kept.
Finish early: python submit.py --done --reason "<why you are stopping>"
```

`{max_submissions}` is replaced with the actual integer, not left as a placeholder. No hidden per-submission coaching prompt is added: the same Claude session reads each generated feedback file and decides its next change.

The supplied information is explicitly limited to:

| File | Information supplied to Claude |
|---|---|
| `prompt.md` | The exact task prompt above, with the submission limit filled in. |
| `policy_api.md` | Method signatures, event order, available metadata, legal return values, allowed dependencies, and resource limits. No policy strategies or comparison implementations. |
| `policy.py` | Working starting policy and necessary policy helpers. |
| `run_context.json` | Trace name/version, serving model, starting policy, fixed cache/scheduler/timing settings, 50/50 split convention, aggregate tune session/request counts, and maximum candidate submissions. No host paths, eval statistics, or future-request information. |
| `history.jsonl` | Log of previous submissions in this run: rationale, policy hash, status, aggregate tune metrics, and remaining budget. |
| `feedback/submission-NNNN.json` | Candidate status, aggregate tune statistics, policy hash, attempts used/remaining, and sanitized errors. No comparison-policy scores or eval results. |

The evaluator supplies any required lightweight API module as a read-only dependency. It does not expose the simulator package or evaluator implementation through that module. The exact rendered prompts and context are archived outside the sandbox so the run can be audited.

## Local tests and custom Python scripts

Claude may write custom Python scripts in the workspace. It runs tests using these
allowlisted commands:

```bash
# Built-in synthetic check of the current policy.
python smoke_test.py

# Custom test written by Claude; optional script arguments follow the path.
python run_local.py tests/my_test.py
python run_local.py tests/my_test.py argument1
```

Both commands request execution from the external evaluator. The evaluator launches
a separate offline Seatbelt process; it never imports agent scripts into itself.
Scripts may read workspace files and import `policy.py`, but cannot read host trace
data or the coding-session home/credentials. Standard-library Python is available.
Only `local-output/` is writable. No network or child processes are allowed.

Limits: 30 seconds wall time, 10 seconds CPU, 1 MiB captured output, 10 MiB per output
file. A strict total disk or memory cap is not implemented. Output and tracebacks
return to Claude. Local tests do not access tune/eval traces, affect scores, or consume
candidate attempts. Requests/responses are temporary and removed after receipt; no
per-test logs are retained. Explicit files created in `local-output/` remain.

The built-in smoke test checks method signatures, add/update/remove events and
repeated eligible eviction decisions. Returning `None` is valid delegation to the
starting policy. These checks catch basic bugs; they do not replace trusted replay
validation or provide a hit-ratio estimate.

## Agent logging

Stream Claude's explanations and tool calls to the terminal, but do not save the raw
agent event stream by default. Disable Claude conversation persistence and redirect
its diagnostic log to `/dev/null`. `--debug` on `policy-agent run` opts into saving the
full emitted stream in `claude.log`.

Always preserve submitted policies, rationales, high-level tune/eval results, validation
errors, and reproducibility metadata. Existing simulator replay logs/intermediates
remain in `internal/`; this change does not introduce a global disk quota or prune them.

## Submission calls and budget

The agent edits `policy.py`, writes a short `rationale.md`, then runs:

```bash
# Submit the current policy; wait for aggregate tune feedback.
python submit.py --rationale-file rationale.md

# After another edit, submit again using the same command.
python submit.py --rationale-file rationale.md

# Stop early when further changes seem unhelpful.
python submit.py --done --reason "No further improvement"
```

Default: **30 candidate attempts**, set by `--max-submissions`. Submission 0 and comparison evaluations are free. Every accepted candidate counts, even if evaluation fails. Missing files rejected before acceptance do not count. Each call returns the feedback path and remaining attempts; evaluation is serial. The evaluator enforces the cap and rejects submissions after completion.

The run ends at the cap, on `--done`, or when Claude exits normally. Crashes and interruptions are marked incomplete. The best valid tune-scoring submission wins, including submission 0; unsubmitted edits are never selected.

## Policy integration

`policy_api.md` is a short interface reference: it tells Claude which methods it must implement, what cache events and metadata it receives, and what it must return. This lets it write a compatible `policy.py` without accessing the simulator source. It supplies no suggested strategy or baseline implementations. The starting `policy.py` is also a working example of that contract.

Preserve UniCache's scheduler, admission behavior, and cache allocator. The editable policy controls eviction behavior through a small adapter exposing:

- Notifications when a block becomes eligible for eviction.
- Access and reuse updates.
- Removal notifications.
- Selection of an eligible block to evict.
- Current simulated time and explicitly permitted online block metadata.

The starting implementation is copied into `policy.py`, including necessary policy-specific helpers. The adapter contract must document each event, its ordering, available fields, and valid return values.

UniCache's policy behavior spans both its evictor and block-tracking hooks. Replacing only `make_evictor()` is insufficient. The adapter must preserve those hooks so the initial exported policy reproduces the built-in implementation.

Submitted code executes in a restricted policy worker with a narrow event interface. It receives online cache events, never future requests. The evaluator validates eviction decisions and enforces time and memory limits. Invalid decisions, crashes, and timeouts fail the submission instead of falling back silently to another policy.

Offline `OPT` runs as a trusted comparison baseline with future knowledge. Submitted policies do not receive that privilege.

## Evaluation gate: correctness and solution validity

`submit.py` is only the client. It uploads an immutable candidate snapshot and rationale to the trusted evaluator, which runs the checks below outside the agent's control. A successful process exit alone is insufficient to pass. The evaluator derives scores from trusted simulator events, never from candidate-reported counters or files.

### 1. Correctness checks

Before optimization starts, validate the trace and evaluation setup:

- Verify source, tokenizer, split, and arrival-timeline checksums against the manifest.
- Verify that selected session IDs exist, occur once in the partition, and are disjoint across tune/eval. Verify the recorded chronological 50/50 boundary and whole-session convention.
- Validate request identities, session/turn order, token-array schema and lengths, supported context limits, and finite, nonnegative arrival times. Fail clearly on unsupported inputs instead of silently dropping or truncating them.
- Freeze the expected request IDs, prompt-token counts, and trace-prescribed response lengths for each split before running any candidate.
- Verify that the unchanged exported starting policy reproduces the built-in implementation's cache-hit counts and ratios. This checks the policy adapter and harness rather than the candidate's quality.

For every accepted candidate, check:

- **Interface:** the module loads, implements the documented callbacks, and returns values with the required types.
- **Cache invariants:** each eviction targets a currently eligible block; active blocks cannot be evicted. Allocator ownership, reference counts, capacity, and block/hash mappings remain consistent after policy decisions.
- **Prefix correctness:** a reported reusable prefix corresponds to the requested prefix in the trusted allocator, including block alignment and computed-block eligibility. A policy cannot obtain a hit by returning unrelated cached content. Use trusted token-prefix identity checks in the validation path, not policy assertions.
- **Complete trace replay:** each expected request completes exactly once; turn dependencies and arrival eligibility are respected. Prompt-token totals and prescribed response lengths match the frozen trace. Policies may change scheduling and cache reuse, but cannot skip, shorten, duplicate, or replace requests to improve the score.
- **Metric consistency:** per-request hit tokens are between zero and that request's prompt-token count. Trusted totals agree with the completed request ledger, and the aggregate ratio equals hit tokens divided by prompt tokens. Empty or incomplete replay does not produce a valid score.

Replay validates the simulator's trace-driven token/accounting behavior, not the semantic quality of generated text: this simulator replays prescribed outputs rather than verifying model-generated answers.

### 2. Solution validity

A correct replay must also be a valid solution to the intended online eviction problem:

- The snapshot contains the submitted `policy.py` and rationale and uses only the documented policy interface and allowed dependencies.
- The policy can act only on online cache events and permitted metadata. It receives no future requests, split labels, baseline implementations, eval scores, or trace files. Enforce this through the sandbox and event schema; a source-code scan alone cannot establish it.
- The candidate cannot modify the scheduler, allocator, admission rules, trace, scoring code, submission budget, or simulator configuration.
- The policy worker has no host filesystem access, network access, persistent state between independent evaluations, or channel to the coding agent during replay.
- Enforce fixed memory and execution-time limits. Crashes, timeouts, prohibited operations, and invalid return values fail the submission; never substitute a fallback policy silently.

Passing these checks does not require beating the starting policy. A valid policy with a lower hit ratio still receives its score. This gate establishes compliance with the interface and evaluation rules, not a proof of optimality or correctness on every possible trace.

### Outcome and errors

Only a candidate that passes both gates and completes tune replay receives an `ok` tune score. Otherwise it receives `correctness_failed` or `validity_failed`, a short sanitized reason, and no score. Accepted attempts consume the budget even when they fail. Evaluator/infrastructure errors are recorded separately and retried without charging an additional candidate attempt.

Use the same checks independently for eval replay. Eval scores and failures remain outside the agent's feedback. An eval failure is recorded with a blank eval score and an error in the user-facing artifacts; it does not trigger selection of a different candidate using eval information. Select the final candidate using valid tune results only, and clearly flag an incomplete eval.

## Submission interface and feedback

Inside the sandbox:

```bash
python submit.py --rationale-file rationale.md
```

The agent writes the implementation to `policy.py` and a short explanation to `rationale.md`, then invokes the command above through its Bash tool. `rationale.md` must state the change, the expected cache-reuse benefit, and the tradeoff or uncertainty; a concise engineering explanation is sufficient. No separate registration or edits to simulator files are required.

The client snapshots `policy.py` and the required rationale, computes a content hash, and asks the external evaluator to assign a submission number. The evaluator preserves an immutable copy before running it.

The returned feedback file is:

```text
sandbox/feedback/submission-0003.json
```

Illustrative successful feedback:

```json
{
  "submission": 3,
  "status": "ok",
  "attempts_used": 3,
  "attempts_remaining": 27,
  "tune": {
    "token_weighted_hit_ratio": 0.8041,
    "hit_tokens": 1608200,
    "prompt_tokens": 2000000,
    "requests": 2500
  }
}
```

The primary metric is:

```text
token-weighted hit ratio = total cache-hit prompt tokens / total requested prompt tokens
```

This differs from averaging per-request hit ratios. V1 returns only aggregate statistics, not per-request records. Failed submissions receive a short error description and no score.

Eval results stay hidden from the agent. The evaluator can evaluate every submission on eval for the user's graph, but neither eval scores nor eval-dependent selection decisions feed back into the optimization loop.

## High-level cache performance metrics

Each successful tune evaluation writes the following metrics to its feedback file and `history.jsonl`:

| Metric | Meaning |
|---|---|
| `token_weighted_hit_ratio` | Total cache-hit prompt tokens divided by total prompt tokens; the optimization objective. |
| `hit_tokens` | Total prompt tokens served from cached prefixes. |
| `prompt_tokens` | Total requested prompt tokens, including reused prefixes. |
| `requests` | Number of evaluated requests. |

These are aggregate metrics for the whole tune split. No per-request records, baseline scores, eval scores, or estimated latency are supplied to Claude. Cache utilization and occupancy are not part of v1 feedback because our current simulator results do not export them.

The user-facing `comparison.csv` reports the same hit-ratio definition on tune and eval for the baselines and selected policy. `submissions.jsonl` outside the sandbox retains both split scores for every candidate; `history.jsonl` inside it exposes tune only.

## Optimization and selection

1. Evaluate the built-in starting policy and comparison baselines on both frozen splits.
2. Give the agent the starting implementation, interface, fixed experiment context, and submission 0 tune feedback. Keep comparison-policy identities, implementations, and scores outside its context, except for the explicitly chosen starting policy.
3. Let the agent edit `policy.py` and submit it with a rationale.
4. Snapshot the submission and evaluate it with fresh simulator and policy state.
5. Return aggregate tune feedback to the sandbox; retain eval results outside it.
6. Repeat until the submission limit or explicit agent completion, as specified above.
7. Select the valid submitted policy with the highest tune hit ratio, breaking ties by earliest submission number.

Submission 0 is the unchanged starting policy and is eligible for selection. If no candidate improves on it, export that policy. Eval scores never determine the selected final policy.

Comparison policies are evaluated once per split under the same model, trace, cache budget, and simulator settings. Their scores can then be reused throughout that run.

## Result directory

Use the user's local date and a unique run identifier:

```text
result/policy-agent/YYYY-MM-DD/sharegpt-<run-id>/
  hit-ratio.png
  comparison.csv
  policy.py
  submissions.jsonl
  run.json
  internal/
    submissions/
```

Visible artifacts:

- `hit-ratio.png`: two panels, tune and eval. The horizontal axis is submission number and the vertical axis is token-weighted hit ratio. Baselines are horizontal reference lines. Submission 0 is the starting policy. Mark the selected final submission; leave failed submissions unscored.
- `comparison.csv`: one row per comparison policy plus the selected agent-developed policy, with tune and eval token-weighted hit ratios.
- `policy.py`: the selected final submitted policy, copied exactly from its immutable snapshot.
- `submissions.jsonl`: submission numbers, rationales, policy hashes, aggregate scores, and errors.
- `run.json`: resolved trace and split checksums, serving model/tokenizer identifiers, complete simulator configuration, seed settings, code versions, baseline results, and selected submission number.

Example `comparison.csv` (illustrative values, ratios on a 0–1 scale):

```csv
policy,kind,submission,tune_hit_ratio,eval_hit_ratio
LRU,baseline,,0.7904,0.7810
UniCache,baseline,,0.7980,0.7890
Agent,selected,7,0.8120,0.7950
```

Include every requested comparison policy. The `Agent` row identifies the selected submission, which may be submission 0 if no candidate improved on it. Selection uses tune only. Per-submission tune/eval scores remain in `submissions.jsonl`; comparison failures have blank scores and an error recorded in `run.json`.

Comparison names, scores, the CSV, and the graph are visible to the user only. They are not mounted into the sandbox or returned in submission feedback. The selected starting policy is necessarily visible to the agent; this setup measures improvements from that starting point, not discovery without any prior policy example.

The internal directory preserves immutable candidate snapshots and evaluator records for reproducibility. It is inaccessible to the agent. The user-facing result remains centered on the graph and final policy.

## V1 acceptance checks

- The unchanged exported starting policy reproduces the built-in policy's hit-token count and hit ratio on each frozen split.
- Repeated evaluation of the same candidate reproduces its aggregate score under fixed inputs and seeds.
- Tune and eval session sets are disjoint, and every policy processes the same requests within each split.
- The agent cannot read eval data, eval scores, evaluator files, or host files through either its coding environment or submitted policy worker.
- Invalid policy output fails clearly and cannot corrupt subsequent evaluations.
- The exported final policy's hash matches the selected submission snapshot.

The first acceptance check establishes that improvements come from policy changes rather than a different evaluation path.

## Implemented multi-workload batch

`bash scripts/run-policy-agent-all-traces.sh` runs the five prepared individual
workloads sequentially, starting each from LRU with Claude Opus and a 30-candidate
cap. It uses `--compare all --require-baselines OPT`. See POLICY_AGENT.md for source
limits and the exact comparison list. ToolBench is excluded because its data is
inaccessible. Every trace uses an independent chronological 50/50 session split,
agent context, and policy state. Baseline outputs stay outside the agent context;
OPT must pass both splits before Claude starts. Experimental policies and a NONE
control are included and labelled by their repository names.
