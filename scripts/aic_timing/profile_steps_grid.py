#!/usr/bin/env python3
"""Measure real prefill/decode step times on a GPU over a controlled grid.

UNTESTED until run on a GPU (written 2026-11-23 without one). Output columns
match ``--simulator-step-log`` so ``fit_calibration.py`` reads either.

Each measurement is one ``LLM.generate`` call, so it includes the engine's
own per-step CPU overhead -- which is exactly what the calibration's
``overhead_ms`` term should absorb.

  prefill row  B prompts = shared cached prefix (P tokens, warmed first)
               + a fresh suffix (N tokens); max_tokens=1  -> one prefill step
  decode row   B fresh prompts of C tokens; time(max_tokens=K+1) minus
               time(max_tokens=1), divided by K           -> one decode step

Run with chunked prefill off so batches are pure prefill or pure decode, as
in the simulator (vLLM v0 engine). Example on the A100:

    VLLM_USE_V1=0 python scripts/aic_timing/profile_steps_grid.py \\
      --model models/Llama-3.2-1B-Instruct --out result/gpu/a100/grid_steps.csv

Quick check first: add --quick (a handful of points, about a minute).
"""
import argparse
import csv
import random
import statistics
import time
from pathlib import Path

FIELDS = ["step", "t_s", "phase", "num_prefill", "prefill_new_tokens",
          "prefill_cached_tokens", "num_decode", "decode_kv_tokens",
          "model_ms", "measured_ms"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-num-batched-tokens", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:  # older vLLM accepts plain dicts
        TokensPrompt = dict

    llm = LLM(model=args.model, enable_prefix_caching=True,
              enable_chunked_prefill=False, block_size=args.block_size,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_num_seqs=256, gpu_memory_utilization=args.gpu_memory_utilization,
              seed=0)
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)

    def tokens(n):
        return [rng.randrange(1000, vocab - 1000) for _ in range(n)]

    def generate(prompts, max_tokens):
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True,
                                temperature=0.0)
        start = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=t) for t in prompts],
                     params, use_tqdm=False)
        return (time.perf_counter() - start) * 1000.0

    if args.quick:
        prefill_grid = [(1, 256, 0), (4, 256, 1024), (8, 1024, 0)]
        decode_grid = [(1, 512), (16, 512), (64, 2048)]
    else:
        prefill_grid = [(b, n, pre) for b in [1, 2, 4, 8, 16, 32]
                        for n in [16, 64, 256, 1024, 4096]
                        for pre in [0, 512, 2048, 8192]
                        if b * (n + pre) <= args.max_num_batched_tokens]
        decode_grid = [(b, c) for b in [1, 4, 16, 32, 64, 128, 256]
                       for c in [128, 512, 2048, 8192]]

    generate([tokens(64)], 4)  # warm-up (CUDA graphs, allocator)
    rows = []

    for b, n, pre in prefill_grid:
        samples = []
        for _ in range(args.repeats):
            prefix = tokens(pre)  # fresh prefix per repeat
            if pre:
                generate([prefix + tokens(args.block_size)], 1)  # warm it
            samples.append(generate([prefix + tokens(n) for _ in range(b)], 1))
        # A fully aligned prefix is cached; vLLM recomputes partial blocks.
        cached = (pre // args.block_size) * args.block_size
        rows.append({"phase": "prefill", "num_prefill": b,
                     "prefill_new_tokens": b * (n + pre - cached),
                     "prefill_cached_tokens": b * cached,
                     "num_decode": 0, "decode_kv_tokens": 0,
                     "measured_ms": statistics.median(samples)})
        print(f"prefill b={b} new={n} prefix={pre}: {rows[-1]['measured_ms']:.2f} ms",
              flush=True)

    k = args.decode_steps
    for b, c in decode_grid:
        samples = []
        for _ in range(args.repeats):
            one = generate([tokens(c) for _ in range(b)], 1)
            many = generate([tokens(c) for _ in range(b)], k + 1)
            samples.append((many - one) / k)
        rows.append({"phase": "decode", "num_prefill": 0,
                     "prefill_new_tokens": 0, "prefill_cached_tokens": 0,
                     "num_decode": b, "decode_kv_tokens": b * (c + k // 2),
                     "measured_ms": statistics.median(samples)})
        print(f"decode b={b} ctx={c}: {rows[-1]['measured_ms']:.2f} ms", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"step": i, "t_s": "", "model_ms": "", **r})
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
