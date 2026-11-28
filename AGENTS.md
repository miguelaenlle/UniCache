# Agent context: coding agents for prefix-cache policy optimization

Shared context for every team member's coding agent (Claude Code, Codex, …).
Read this first, then the doc linked for the area you work on. Last updated
2026-11-26. When a fact here disagrees with the code or a newer doc, trust the
code and fix this file.

CS 498 research project: Miguel Aenlle, Albert Bogdan, Josef Hiller, Danh
Nguyen, Steven Piotrowski.

## The project in one paragraph

A coding agent edits `policy.py`, a prefix-cache (KV-cache block) eviction
policy. It submits the policy to the UniCache simulator, reads the hit ratio on
the **tune** split, and iterates for up to 30 submissions. Its best tune
submission is scored once on a held-out **eval** split. We compare it with LRU
and with the **best-heuristic selector** (the best of LRU, LFU, FIFO, LeCaR and
UniCache on tune), with Bélády OPT as the upper bound.

* **H1, agents beat heuristics:** +4.25 pp over LRU on average, but −1.46 pp
  behind the best-heuristic selector, on the 9 configurations with > 5 pp of
  OPT − LRU tune headroom. Partly supported.
* **H2, sim hit ratios track real GPU hit ratios:** yes for 4 of 5 workloads
  (512 blocks, A100). ShareGPT is badly off because decode timing is mismatched.
* **H3, multi-agent or RL-trained agents beat a single agent at equal budget:**
  not started.

## Setup

| Item | Value |
|---|---|
| Simulator | UniCache: the vLLM V0 control plane on CPU with a virtual clock, no weights. This repo. |
| Serving model | Llama-3.2-1B-Instruct |
| Block size | 8 tokens in H1. The sim2real GPU runs use 16, because 8-token blocks on CUDA need xformers, which does not run on GH200's Arm host. |
| KV capacities (H1) | 4 / 8 / 12.207 / 24 GiB = 16,384 / 32,768 / 50,000 / 98,304 blocks |
| Workloads | ShareGPT, MASH-QA, CodeParrot, AgentBank APPS, AgentBank ALFRED |
| Splits | Each trace is split 50/50 by session start into tune and eval (`TaskData/*/agent-splits/`, gitignored) |
| Arrivals | The datasets have no timestamps. First turns are a seeded Poisson process at 1 session/s, frozen into the split yaml. Follow-up turns arrive when the previous turn finishes, plus a seeded lognormal think time. |
| Metric | Token-weighted hit ratio H = Σ hit prompt tokens / Σ prompt tokens. Headroom = H_OPT − H_LRU. |

## Repositories and branches

* **Main repo (this one):** <https://github.com/miguelaenlle/UniCache>, a
  **public** fork of <https://github.com/xsyslab/UniCache>. We have never pushed
  to the upstream repo. Never put credentials, tokens or private data in commits.
* **Earlier approach (reference only):** <https://github.com/albogdan7/dynamo>,
  an NVIDIA Dynamo fork. `research/kv-eviction/` on `main` holds Albert's
  leaf-first LRU tiebreak. Branch `research/continuum-mori` holds the Continuum
  and MORI retention and offload experiments.

| Branch | Builds on | What it holds |
|---|---|---|
| `main` | upstream `main` | Local simulator setup, hit-ratio matrix, 1B cache-budget sweep and headroom reports, the policy agent (`./policy-agent`, `policy_agent/`) with the v1 and v2 policy interfaces and the offline sandbox. |
| `experiment/agent-budget-v1` | `main` | The H1 agent × cache-budget sweep behind report Table 2 and Figure 1, plus the analysis scripts (`scripts/build_agent_master_table.py`, `plot_cache_results.py`, …). See `AGENT_CACHE_BUDGET_EXPERIMENT.md` on that branch. |
| `experiment/sim2real-subset` | `main` | H2: subset replay, timing probe, sim-vs-GPU reports (`docs/sim2real-subset.md`). |
| `experiment/aic-timing` | `sim2real-subset` | **Active branch.** GPU step timing from NVIDIA aiconfigurator, TTFT/ITL metrics, the UniCache-paper Table 5 reproduction, and the sim2real transfer test on GH200. Some commits may still be local and unpushed. Ask Miguel before you build on something you can't find on GitHub. |

## Who owns what (post-midterm plan)

| # | Work item | Owner | Status |
|---|---|---|---|
| 1 | Reproduce results in a simulator that yields TTFT/ITL, so we report full serving metrics | Miguel | Prototype on `experiment/aic-timing` (`docs/aic-timing.md`). The timing tables are not calibrated yet. |
| 2 | Tune (calibrate) the simulator | Miguel | Planned for GH200 on NCSA DeltaAI (`docs/gh200-calibration-plan.md`) |
| 3 | Sim2real transfer testing | Miguel | Simulator predictions pre-registered. GPU runs pending, ALFRED first (`docs/sim2real-results.md`). |
| 4 | Online policy development: the agent re-adapts the policy as the workload shifts, every few hours | Teammates | Can start now on the simulator. TTFT/ITL from (1) can be added later. |
| 5 | RSI framing: recursive improvement of the serving system, or of the agent itself | Teammates: experiential learning (the agent keeps and reuses optimization experience from the online runs in item 4). The rest of the framing is unassigned. | Not started |
| 6 | Switch the agent's objective from KV hit ratio to TTFT, then the end-to-end loop: the agent develops a policy on a trace slice, its policy is replayed on the GH200, and the TTFT result is quantified | Miguel | After (1)–(3) |

Items 1–3 need GPU access and touch the simulator core (`vllm/simulator/`,
`vllm/core/`). If you are working on items 4 or 5, don't change those files
without coordinating with Miguel. Build on the interfaces below.

## Where things are

This file lives on `main`, but the latency and sim2real work is only on
`experiment/aic-timing`. That covers `vllm/simulator/step_timing.py`,
`latency_metrics.py`, `cache_stats.py`, `engine_time()`, `scripts/aic_timing/`,
`scripts/paper_qttft/`, `scripts/sim2real/`, `docs/aic-timing.md`, and the
`docs/sim2real-*` and `docs/gh200-*` docs. Check out that branch to use them.

| Path | What |
|---|---|
| `run_simulation.py`, `./sim` | Simulator entry point. `./sim` runs a 20-session ShareGPT smoke test. |
| `vllm/simulator/` | Simulator event loop, `step_timing.py` (aiconfigurator timing), `latency_metrics.py` (TTFT/ITL/TPOT/E2E), `cache_stats.py` |
| `vllm/core/block/prefix_caching_block.py` | Prefix cache, where eviction happens |
| `policy_agent/` | Agent loop (`runner.py`), policy bridges and workers, sandbox. `policy_agent/v2_api.md` is the v2 policy contract (events, `evict(context)`). |
| `./policy-agent` | CLI for agent runs (`POLICY_AGENT.md`) |
| `OPT/` | Offline Bélády OPT oracle |
| `scripts/` | Matrix and sweep runners, `aic_timing/`, `paper_qttft/`, `sim2real/` |
| `TaskData/*/agent-splits/` | Prepared traces and tune/eval splits (gitignored, large; get them from Miguel or regenerate with `prepare_session_data.py`) |
| `result/` | Experiment outputs (`hit-ratio-matrix/`, `cache-budget-study/`, `aic-timing/`, `paper-qttft/`, `sim2real-transfer/`, …) |

Docs to read by topic:
* Whole project: the mid-semester report (ask Miguel for the PDF).
* Running baselines: `EXPERIMENTS.md`, `LOCAL_SETUP.md`.
* Agent loop: `POLICY_AGENT.md`, `POLICY_AGENT_V1_DESIGN.md`, `POLICY_INTERFACE_V2_PLAN.md`, `policy_agent/v2_api.md`.
* Latency: `docs/aic-timing.md`.
* Sim2real: `docs/sim2real-transfer-methodology.md`, `docs/sim2real-results.md`, `docs/gh200-calibration-plan.md`, `docs/gh200-experiments.md`, `docs/sim2real-delta-runbook.md`.

## Running

Python 3.12 virtualenv per checkout. It is an editable install, so never copy
a venv between checkouts.

```bash
bash setup-simulator312.sh                     # creates .venv-sim312
./sim                                          # 20-session ShareGPT smoke run
bash scripts/run-hit-ratio-matrix.sh --jobs 2  # baseline matrix
./policy-agent run --trace sharegpt --serving_model llama-3.2-1b-instruct \
  --coding_model claude:opus --start-policy UniCache --max-submissions 30
```

Add `--prepare-only` to `./policy-agent run` to inspect the agent sandbox
without simulations or API calls. Add `--policy-interface v2` for the editable
v2 interface.

Latency mode: `run_simulation.py --simulator-timing aic --simulator-aic-system h100_sxm`
(the default timing is `legacy`, which reproduces recorded hit ratios exactly).
Every run writes `latency_requests.csv`. Under `aic` timing, refine offline
OPT with `scripts/paper_qttft/refine_opt.py`, because its no-cache future map
goes stale once cache hits change step times.

Practical limits:
* A long multi-turn replay (ALFRED-heavy) holds about 3 GB of RAM. Run about 3
  in parallel on a 48 GB machine.
* Submitted policies run offline under Seatbelt on macOS or Bubblewrap with
  libseccomp on Linux. Linux requires unprivileged user and network namespaces.
* Baseline replays must run outside any parent sandbox that blocks OpenMP
  shared memory.
* The coding agent needs `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in the
  shell, or `--normal-cli-auth`. Never write credentials to repo files.

## Rules (reward-hacking guards and reporting)

* **Keep eval scores, eval traces and the OPT oracle out of the agent's
  workspace.** Agents may use only past and current cache information. For
  online adaptation (item 4) this means the adapting agent sees only the trace
  up to "now", never the future window it will be scored on.
* Report hit ratios in **pp**, with **tune and eval reported separately**, and
  compare with **both** LRU and the best-heuristic selector, not LRU alone.
* Pick the agent's policy by tune score only. Score eval once.
* Pre-register predictions (commit them) before running GPU experiments.
* Time-based policies must read the engine's clock (`engine_time()` in
  `vllm/simulator/timer_utils.py`): virtual seconds in sim, wall time on GPU.
  The `now` field in `sequence_step` events already does this.
* Keep this project separate from SysPilot / DynoSim, which are unrelated.

## Notes for item 4 (online adaptation)

Things that already exist and are worth reusing:
* `policy_agent/runner.py` drives the agent loop: workspace, submissions,
  tune scoring, selection by tune score. An online loop can call it on a
  sliding window (e.g. tune on the last k hours, deploy on the next hour).
* The v2 interface (`policy_agent/v2_api.md`) gives the policy an event stream
  with timestamps, and its state persists for the length of a replay. Online
  history survives physical cache resets.
* To replay a frozen policy on a new window, use `scripts/replay_budget_policy.py`
  on `experiment/agent-budget-v1`.
* The traces have no real timestamps, so "workload shift every few hours" has
  to be built, e.g. by concatenating windows from different workloads or
  changing the arrival rate or session mix. Write down how you built it.
