| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#3 | LRU | 0.13 | 0.044 | 1.00 | 1.00 | 0.4140 |
| Trace#3 | LFU | 0.37 | 0.048 | 1.09 | 2.85 | 0.3523 |
| Trace#3 | FIFO | 0.25 | 0.046 | 1.03 | 1.92 | 0.3900 |
| Trace#3 | LeCaR | 0.35 | 0.046 | 1.03 | 2.69 | 0.4037 |
| Trace#3 | ARC | 0.12 | 0.046 | 1.03 | 0.92 | 0.4147 |
| Trace#3 | WA | 0.27 | 0.046 | 1.03 | 2.08 | 0.4085 |
| Trace#3 | OPT (Offline) | 0.08 | 0.036 | 0.82 | 0.62 | 0.6615 |
| Trace#3 | UniCache | 0.10 | 0.041 | 0.93 | 0.77 | 0.5557 |
| Trace#3 | NONE (no caching) | — | 0.030 | 0.69 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#3 | 0.95 | 6/7 |
