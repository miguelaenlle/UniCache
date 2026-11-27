# UniCache paper Table 5 (QTTFT) reproduction: before vs after

Status: plan. "Before" is the 2026-11-24 simulator run (`result/paper-qttft/aic-a100/table5.md`).
"After" cells are empty until the runs land.

| | Before | After |
|---|---|---|
| Timing | `aic`, uncalibrated `a100_sxm` tables | GH200 calibration for Llama-3.2-3B, fit on the synthetic sweep only (v8 method) |
| Load | paper arrival rates | arrivals scaled to match the paper's utilization: ×___ (T#1), ×___ (T#2), ×___ (T#3) |
| GPU check | none | T#2 only, 10-min slice, LRU / LFU / LeCaR / UniCache, 1 × GH200 |

Shared: Llama-3.2-3B, 66,355-block KV budget (0.9 × A100 KV), traces `TaskData/Hybrid/test1–3`
(no ToolBench), OPT refined with `refine_opt.py`.

## QTTFT relative to LRU (same trace)

| Trace | Policy | Paper | Before (sim) | After (sim) | After (GH200) |
|---|---|---:|---:|---:|---:|
| T#1 | LRU | 1.00 | 1.00 | 1.00 | — |
| T#1 | LFU | 1.71 | 1.13 | | — |
| T#1 | FIFO | 1.39 | 1.04 | | — |
| T#1 | LeCaR | 1.50 | 0.64 | | — |
| T#1 | ARC | 0.92 | 1.00 | | — |
| T#1 | WA | 1.04 | 1.15 | | — |
| T#1 | OPT | 0.78 | 0.14 (not converged) | | — |
| T#1 | UniCache | 0.78 | 0.86 | | — |
| T#2 | LRU | 1.00 | 1.00 | 1.00 | 1.00 |
| T#2 | LFU | 1.89 | 3.44 | | |
| T#2 | FIFO | 1.29 | 1.18 | | — |
| T#2 | LeCaR | 1.68 | 1.38 | | |
| T#2 | ARC | 0.93 | 1.00 | | — |
| T#2 | WA | 1.04 | 1.09 | | — |
| T#2 | OPT | 0.75 | 0.81 | | — |
| T#2 | UniCache | 0.86 | 0.91 | | |
| T#3 | LRU | 1.00 | 1.00 | 1.00 | — |
| T#3 | LFU | 2.85 | 1.05 | | — |
| T#3 | FIFO | 1.92 | 1.02 | | — |
| T#3 | LeCaR | 2.69 | 1.01 | | — |
| T#3 | ARC | 0.92 | 1.00 | | — |
| T#3 | WA | 2.08 | 1.01 | | — |
| T#3 | OPT | 0.62 | 0.78 | | — |
| T#3 | UniCache | 0.77 | 0.87 | | — |

## Agreement with the paper

| Trace | LRU QTTFT, paper (s) | Before (s) | After, sim (s) | Spearman ρ before | ρ after (sim) | Same side of LRU, before | After (sim) |
|---|---:|---:|---:|---:|---:|---:|---:|
| T#1 | 4.94 | 10.67 | | 0.52 | | 5/7 | |
| T#2 | 0.28 | 0.018 | | 1.00 | | 7/7 | |
| T#3 | 0.13 | 0.018 | | 0.93 | | 7/7 | |

GH200 check (T#2, 4 policies): ρ vs paper ___, ρ vs simulator ___, LRU noise floor ___ ms.
