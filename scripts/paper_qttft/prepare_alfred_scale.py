#!/usr/bin/env python3
"""Tokenize the first N sessions of AgentBank_alfred_scale_5000.json.

The paper's mixed traces (TaskData/Hybrid/test1-3) read
AgentBank_alfred_scale_5000_session_data.jsonl, of which only the first 500
sessions are used; the original preparation was interrupted. This prepares
just those, with the same function and the Llama-3 tokenizer (shared by the
1B/3B/8B models). The scale-up step inserted random UUIDs, so contents match
the paper's statistically, not byte for byte.
"""
import argparse
import json
import sys
from pathlib import Path

import ijson
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vllm.simulator.data_parser import prepare_session_data  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--sessions", type=int, default=600)
p.add_argument("--tokenizer", default=str(ROOT / "models/Llama-3.2-1B-Instruct"))
args = p.parse_args()

src = ROOT / "TaskData/AgentBankData/AgentBank_alfred_scale_5000.json"
dst = ROOT / "TaskData/AgentBankData/AgentBank_alfred_scale_5000_session_data.jsonl"
tok = AutoTokenizer.from_pretrained(args.tokenizer)
with open(src) as f_in, open(dst, "w") as f_out:
    for i, entry in enumerate(ijson.items(f_in, "item")):
        if i >= args.sessions:
            break
        r = prepare_session_data(entry, tok)
        f_out.write(json.dumps({r["id"]: r}, ensure_ascii=False) + "\n")
print(f"Wrote {i} sessions to {dst}")
