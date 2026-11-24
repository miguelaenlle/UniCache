# Parameters to confirm for the 512-block GPU runs

Fill in the GPU column with actual settings, including defaults. Note differences
by trace or policy. Use short values or `unknown`.

| Parameter | Our simulator | GPU run: fill in |
|---|---|---|
| Model / tokenizer revision | Llama-3.2-1B-Instruct; metadata matches archive | |
| Input subset; sessions/requests | Archived subset; ShareGPT 20/82, ALFRED 2/103, others 100/100 | |
| Input order / prompt changes | Seeded session shuffle / no added truncation or prefixes | |
| Device / GPU variant / workers | CPU | A100, 1 worker (reported; confirm) |
| Clock mode / disable_simulation | Virtual / False | |
| Weights / output generation | Dummy / forced archived tokens | |
| Prefill time (seconds) | `3.9e-05 * batch_size**0.997 * max_len**1.014` | |
| Decode time (seconds/step) | 0.01 | |
| Arrival rate / distribution | 1 session/second / Poisson | |
| Arrival seed / task ID | 42 / 0 | |
| Follow-up trigger | Previous response completion | |
| Thinking delay | ALFRED: lognormal mu=1.81, sigma=1.092; others: 0 seconds | |
| Client concurrency / pacing | All due arrivals released at engine-step boundaries | |
| Duration cap / timeout / retries | No follow-up cutoff / 1,800-second wall timeout / none | |
| Requested / actual GPU blocks; tokens/block | 512 / 512 simulated; 16 | |
| CPU blocks / offload / swap | 0 / 0 GB / 0 GB | |
| Prefix caching / LMCache / cache reset | On / off / fresh per run | |
| Maximum context / sequences / batched tokens | 8,192 / 32 / engine default | |
| Chunked prefill / scheduler / preemption | Off / native default / recompute observed | |
| TP / PP / DP | TP=1; others not configured | |
| Model dtype / KV dtype / quantization | Automatic defaults / automatic defaults / none explicit | |
| GPU memory utilization | 0.8 configured | |
| Attention backend / graphs / async / speculative decode | Platform and engine defaults | |
| Temperature / output limit / stop rules | 0 / archived response length / remaining defaults | |
| Engine version / commit / local patches | V0 / base e7b231e / snapshots saved | |
| Policy implementation / overrides / tie-breaker | Native; UniCache=SimpleHYBRID; defaults | |
| WA fit interval / policy seed / state reset | 1200 / native default / fresh per run | |
| OPT profile / fallback | Separate NONE profile per trace / native behavior | |
| Hit-ratio definition | Sum cached prompt tokens / sum prompt tokens | |
| Completed / failed / skipped; warmup / repeats | All / 0 / 0; none / 1 primary run | |
| Estimated chart cells | Seven ALFRED values excluded from comparison | |
| Python / torch / CUDA / RNG versions | 3.12.14 / 2.6.0 / none / NumPy 2.2.6 | |
| Environment / other overrides | PYTHONHASHSEED=0; VLLM_USE_V1=0; remaining settings in manifest | |

Priority: ShareGPT clock mode, follow-up trigger, arrival rate/order, decode timing,
actual block allocation, and hit-ratio definition.

Our reference: `result/sim2real-subset/2026-11-03-friend-512-verified/manifest.json`.
