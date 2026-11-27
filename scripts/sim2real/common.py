"""Shared definitions for the sim-to-real transfer test.

See docs/sim2real-transfer-methodology.md. Everything about a workload's
subset and its H1 sources lives in WORKLOADS, so the sim and GPU sides read
the same definition.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # UniCache/
H1 = ROOT.parent / "UniCache-agent-budget-v1"
STUDY = H1 / "result/agent-cache-budget-study/manual-20261101-162735"
AGENT_RUNS = H1 / "result/policy-agent/2026-11-01"
SUBSETS = ROOT / "TaskData/sim2real"                 # gitignored (jsonl)
RESULTS = ROOT / "result/sim2real-transfer"

# Tier-1 policies (engine names). "Agent" is the workload's best Table 2 agent.
HEURISTICS = ["LRU", "UniCache", "LeCaR"]
ENGINE_NAME = {"LRU": "LRU", "UniCache": "SimpleHYBRID", "LeCaR": "LeCaR",
               "LFU": "LFU", "FIFO": "FIFO", "OPT": "OPT", "NONE": "NONE"}

WORKLOADS = {
    "codeparrot": dict(
        split=H1 / "TaskData/Codeparrot/agent-splits/codeparrot/v1-5000/eval",
        sessions=300, max_turns=None,
        # Best Table 2 agent: CodeParrot 8 GiB (32,768 blocks), +11.40 pp eval.
        agent_blocks=32768),
    "mashqa": dict(
        split=H1 / "TaskData/mashqa_data/agent-splits/mashqa/v1-5000/eval",
        sessions=300, max_turns=None,
        agent_blocks=98304),   # MASH-QA 24 GiB, +1.22 pp
    "alfred": dict(
        split=H1 / "TaskData/AgentBankData/agent-splits/alfred/v1-500/eval",
        sessions=60, max_turns=12,
        agent_blocks=32768),   # ALFRED 8 GiB, +10.08 pp
    "sharegpt": dict(
        split=H1 / "TaskData/SharedGPT-multi-turn/agent-splits/sharegpt/v1-1000/eval",
        sessions=150, max_turns=4,
        agent_blocks=16384),   # ShareGPT 4 GiB, +0.04 pp
}


def spec(name: str) -> dict:
    """Subset spec. A variant "<workload>@<sessions>[x<max_turns>]" overrides sizes,
    e.g. "codeparrot@600" or "sharegpt@100x2"; it shares the base workload's agent."""
    base, _, size = name.partition("@")
    s = dict(WORKLOADS[base], base=base)
    if size:
        n, _, k = size.partition("x")
        s["sessions"] = int(n)
        if k:
            s["max_turns"] = int(k)
    return s


def baseline_command(workload: str) -> Path:
    """An H1 LRU eval command for this workload; every run is derived from it."""
    return STUDY / f"baseline-replays/16384/{workload}/LRU/eval/command.json"


def agent_policy(workload: str) -> Path:
    """The selected submission's policy.py for the workload's best agent.

    On machines without the H1 results (DeltaAI), set SIM2REAL_AGENT_DIR to a
    directory holding <base workload>.py copies (see make_bundle.py)."""
    import json
    import os
    if os.environ.get("SIM2REAL_AGENT_DIR"):
        # Known workloads map variants to their base ("codeparrot@600" -> codeparrot.py);
        # synthetic sweep workloads (not in WORKLOADS) use their own name.
        base = workload.partition("@")[0]
        return Path(os.environ["SIM2REAL_AGENT_DIR"]) / f"{base}.py"
    s = spec(workload)
    rec = json.loads((STUDY / f"agent-runs/{s['agent_blocks']}/{s['base']}.json").read_text())
    run = AGENT_RUNS / Path(rec["run"]).name
    return run / "internal" / f"submission-{int(rec['selected']):04d}" / "policy.py"


MODEL = ROOT / "models/Llama-3.2-1B-Instruct"  # config + tokenizer only (dummy weights)
# H1's engine flags (identical in every H1 baseline command.json).
# Block size is a parameter: H1 used 8, but on CUDA this fork runs 8-token blocks
# only through xformers (x86-only), so the sim2real runs use 16 on both sides.
H1_FLAGS = ["--max-num-seqs", "32", "--max-model-len", "131072",
            "--simulator-ttft", "3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)",
            "--simulator-tpot", ".01", "--log-level", "WARNING"]


def replay_command(python, workload, blocks, policy, run_dir, mode="sim",
                   use_map=None, step_log=None, block_size=8, task_config=None):
    """One replay. mode='sim': H1's simulator (virtual time, legacy timing).
    mode='gpu': the same engine on CUDA in real time (no virtual clock)."""
    cmd = [str(python), "-m", "policy_agent.replay", "--model", str(MODEL),
           "--device", "cpu" if mode == "sim" else "cuda", *H1_FLAGS,
           "--block-size", str(block_size),
           "--simulator-num-gpu-blocks", str(blocks), "--simulator-num-cpu-blocks", "0",
           "--eviction-policy", ENGINE_NAME["LRU" if policy == "Agent" else policy],
           "--simulator-output-dir", str(run_dir),
           "--simulator-task-config", str(task_config or SUBSETS / workload / "eval.yaml")]
    if mode == "sim":
        cmd.append("--simulator-enable-virtual-time")
    else:
        cmd.append("--disable-simulation")
    if policy != "NONE":
        cmd.append("--enable-prefix-caching")
    if use_map is not None:
        cmd += ["--opt-block-use-map-path", str(use_map)]
    if step_log is not None:
        cmd += ["--simulator-step-log", str(step_log)]
    # Optional extra replay flags, e.g. the timing model for step 1 of the TTFT work:
    #   SIM2REAL_EXTRA_FLAGS="--simulator-timing aic --simulator-aic-system h100_sxm
    #     --simulator-timing-calibration calibrations/gh200-llama-3.2-1b-v0.json"
    # They are recorded in each run's command.json.
    import os
    import shlex
    cmd += shlex.split(os.environ.get("SIM2REAL_EXTRA_FLAGS", ""))
    return cmd
