# Sim2real run on one A100: instructions for the teammate

This replays one short trace (AgentBank ALFRED: 60 sessions, about 5 min of
requests) on a real A100 with Miguel's UniCache vLLM fork. It checks whether
the UniCache simulator's eviction-policy hit ratios hold on real hardware.

* About **1 hour of one A100**: 6 replays of ~7.5 min plus a 6-min smoke test.
* No model weights are needed (the engine uses dummy weights and replays
  recorded outputs). Nothing is downloaded and nothing leaves your machine.

## Requirements

* x86 Linux, one NVIDIA A100, CUDA driver ≥ 12.4.
* The Python environment you used for the earlier sim2real runs: the UniCache
  fork built with torch 2.6 and xformers. If you no longer have it, build one
  from `unicache/` (see `unicache/LOCAL_SETUP.md`), which takes about 1 h.
* `unshare` working for your user (`unshare --user --map-root-user --net true`).
  If it doesn't, install bubblewrap and set `UNICACHE_POLICY_SANDBOX=bwrap`.
  This runs the agent-written eviction policies with no network access. They
  are pure Python (heapq/math) and were reviewed.

## Steps

```bash
tar -xzf sim2real-a100-bundle.tar.gz && cd sim2real-a100-bundle
# Reuse your compiled fork so nothing is recompiled:
bash overlay_compiled.sh /path/to/your/built/UniCache
# Run on one GPU (pick a free one):
CUDA_VISIBLE_DEVICES=0 PY=/path/to/that/env/bin/python bash run_a100.sh
```

The script runs these in order, stopping at the first failure:
1. imports and the sandbox check;
2. a smoke test: a 2-min CodeParrot subset run twice;
3. ALFRED with LRU, UniCache, LeCaR and the agent policy;
4. two more LRU runs, to measure noise.

When it prints `DONE`, send back `results-a100.tar.gz` (a few MB).
If anything fails, send `results-a100.tar.gz` anyway, along with the terminal
output.
