| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#3 | LRU | 0.13 | 0.018 | 1.00 | 1.00 | 0.4248 |
| Trace#3 | LFU | 0.37 | 0.019 | 1.05 | 2.85 | 0.3628 |
| Trace#3 | FIFO | 0.25 | 0.018 | 1.02 | 1.92 | 0.3988 |
| Trace#3 | LeCaR | 0.35 | 0.018 | 1.01 | 2.69 | 0.4157 |
| Trace#3 | ARC | 0.12 | 0.018 | 1.00 | 0.92 | 0.4252 |
| Trace#3 | WA | 0.27 | 0.018 | 1.01 | 2.08 | 0.4160 |
| Trace#3 | OPT (Offline) | 0.08 | 0.014 | 0.78 | 0.62 | 0.6793 |
| Trace#3 | UniCache | 0.10 | 0.016 | 0.87 | 0.77 | 0.5757 |
| Trace#3 | NONE (no caching) | — | 0.025 | 1.37 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#3 | 0.93 | 7/7 |
