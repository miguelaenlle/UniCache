#!/usr/bin/env python3
"""Cut the sim2real subsets from the H1 eval splits.

For each workload: the first N sessions by recorded first-turn arrival, each
cut to its first K turns (multi-turn only). Arrival times are kept exactly
(frozen_arrivals), and think times are seeded per session id, so they are
unchanged too. Writes TaskData/sim2real/<workload>/eval.{jsonl,yaml} and
prints sizes.

    .venv-sim312/bin/python scripts/sim2real/build_subsets.py [workload ...]
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUBSETS, WORKLOADS  # noqa: E402


def build(name: str) -> dict:
    spec = WORKLOADS[name]
    cfg = yaml.safe_load(spec["split"].with_suffix(".yaml").read_text())
    assert len(cfg["tasks"]) == 1
    task = cfg["tasks"][0]
    arrivals = task["frozen_arrivals"]
    keep = sorted(arrivals, key=lambda s: (arrivals[s], s))[: spec["sessions"]]
    keep_set = set(keep)

    out = SUBSETS / name
    out.mkdir(parents=True, exist_ok=True)
    turns = requests = prompt_tokens = 0
    with open(spec["split"].with_suffix(".jsonl")) as src, \
            open(out / "eval.jsonl", "w") as dst:
        for line in src:
            record = json.loads(line)
            (sid, session), = record.items()
            if sid not in keep_set:
                continue
            k = spec["max_turns"]
            if k is not None and session["num_turns"] > k:
                for t in range(k + 1, session["num_turns"] + 1):
                    session.pop(f"turn_{t}")
                session["num_turns"] = k
            requests += session["num_turns"]
            prompt_tokens += sum(len(session[f"turn_{t}"]["input_ids"])
                                 for t in range(1, session["num_turns"] + 1))
            dst.write(json.dumps({sid: session}) + "\n")
            turns += session["num_turns"]
    task = dict(task, data_path=str(out / "eval.jsonl"), max_record=len(keep),
                frozen_arrivals={s: arrivals[s] for s in keep})
    (out / "eval.yaml").write_text(yaml.safe_dump(dict(cfg, tasks=[task]), sort_keys=False))
    span = max(arrivals[s] for s in keep)
    return dict(workload=name, sessions=len(keep), requests=requests,
                prompt_tokens=prompt_tokens, arrival_span_s=round(span, 1))


def main():
    names = sys.argv[1:] or list(WORKLOADS)
    rows = [build(n) for n in names]
    for r in rows:
        print(r)
    (SUBSETS / "subsets.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
