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


def baseline_command(workload: str) -> Path:
    """An H1 LRU eval command for this workload; every run is derived from it."""
    return STUDY / f"baseline-replays/16384/{workload}/LRU/eval/command.json"


def agent_policy(workload: str) -> Path:
    """The selected submission's policy.py for the workload's best agent."""
    import json
    rec = json.loads((STUDY / f"agent-runs/{WORKLOADS[workload]['agent_blocks']}/{workload}.json").read_text())
    run = AGENT_RUNS / Path(rec["run"]).name
    return run / "internal" / f"submission-{int(rec['selected']):04d}" / "policy.py"
