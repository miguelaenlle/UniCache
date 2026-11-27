#!/usr/bin/env python3
"""Synthetic calibration workloads for the timing model (fit on these only; traces = validation).

Prompts are random sequences of words that are each exactly one Llama-3 token, so prompt lengths
and shared prefixes are exact. Each config becomes TaskData/sim2real/<name>/eval.{jsonl,yaml} in
the same format as the trace subsets, so run_sim.py replays it unchanged (sim or GPU).

Groups (single-turn requests):
  A  sweep-rps<r>       load sweep: 150 requests, 512-token prompts, 128-token outputs, Poisson r/s
  B  sweep-pre<n>-s<p>  prefill: 30 requests, one per second, n-token prompts, p% shared prefix,
                        8-token outputs (isolated prefills; shared prefix is cached after the first)
  C  sweep-dec<b>-c<c>  decode: 3 bursts of b simultaneous requests (t = 0, 15, 30 s), c-token
                        prompts, 256-token outputs (decode steps at batch b, context ~c)
  D  sweep-evict        policy cost: 150 requests at 10/s, 512-token prompts sharing a 256-token
                        prefix within groups of 10, 64-token outputs; replayed with a small cache
                        so evictions happen, with built-in LRU and with LRU via the agent sandbox

    .venv-sim312/bin/python scripts/sim2real/make_sweep.py
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ROOT / "TaskData" / "sim2real"
SEED = 1234


def single_token_words(tok):
    words = []
    for t, i in tok.get_vocab().items():
        if t.startswith("Ġ") and re.fullmatch(r"[a-z]{3,8}", t[1:]):
            if tok.encode(" " + t[1:], add_special_tokens=False) == [i]:
                words.append(t[1:])
    return sorted(set(words))


def write_config(name, requests, arrivals, tok):
    """requests: list of (prompt_words, response_words); arrivals: seconds, same length."""
    out = SUBSETS / name
    out.mkdir(parents=True, exist_ok=True)
    frozen = {}
    tokens = 0
    with open(out / "eval.jsonl", "w") as f:
        for k, ((prompt, response), t) in enumerate(zip(requests, arrivals)):
            sid = f"{name}-{k:04d}"
            text = "".join(" " + w for w in prompt)
            resp = "".join(" " + w for w in response)
            ids = tok.encode(text)
            rids = tok.encode(resp, add_special_tokens=False)
            assert len(ids) == len(prompt) + 1 and len(rids) == len(response), name
            tokens += len(ids)
            f.write(json.dumps({sid: {"id": sid, "num_turns": 1, "turn_1": {
                "input_text": text, "input_ids": ids, "response_text": resp,
                "response_ids": rids}}}) + "\n")
            frozen[sid] = float(t)
    task = {"name": name, "data_path": str(out / "eval.jsonl"), "max_record": len(requests),
            "lambda_rate": 1, "mu": 1.81, "sigma": 1.092, "seed": SEED, "task_id": 1,
            "max_trace_duration": -1, "frozen_arrivals": frozen}
    (out / "eval.yaml").write_text(yaml.safe_dump({"tasks": [task]}, sort_keys=False))
    return {"name": name, "requests": len(requests), "prompt_tokens": tokens,
            "span_s": round(max(arrivals), 1)}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Llama-3.2-1B-Instruct"))
    words = single_token_words(tok)
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    rand = lambda n: [rng.choice(words) for _ in range(n)]
    made = []
    # A: load sweep.
    for r in [2, 5, 10, 20, 40]:
        arr = np.cumsum(nrng.exponential(1 / r, 150)) - 0
        arr -= arr[0]
        made.append(write_config(f"sweep-rps{r}", [(rand(512), rand(128)) for _ in range(150)], arr, tok))
    # B: isolated prefills, with and without a shared (cached) prefix.
    for n, share in [(128, 0), (512, 0), (2048, 0), (8192, 0), (2048, 50), (2048, 90)]:
        prefix = rand(n * share // 100)
        reqs = [(prefix + rand(n - len(prefix)), rand(8)) for _ in range(30)]
        made.append(write_config(f"sweep-pre{n}-s{share}", reqs, [float(k) for k in range(30)], tok))
    # C: decode batches (bursts of simultaneous requests).
    for b in [1, 8, 16, 32]:
        for c in [256, 2048]:
            reqs, arr = [], []
            for burst in range(3):
                for _ in range(b):
                    reqs.append((rand(c), rand(256)))
                    arr.append(15.0 * burst)
            made.append(write_config(f"sweep-dec{b}-c{c}", reqs, arr, tok))
    # D: evictions under a small cache (policy cost).
    arr = np.cumsum(nrng.exponential(1 / 10, 150)); arr -= arr[0]
    groups = [rand(256) for _ in range(15)]
    reqs = [(groups[k // 10] + rand(256), rand(64)) for k in range(150)]
    made.append(write_config("sweep-evict", reqs, arr, tok))
    (SUBSETS / "sweep.json").write_text(json.dumps(made, indent=2))
    for m in made:
        print(m)


if __name__ == "__main__":
    sys.exit(main())
