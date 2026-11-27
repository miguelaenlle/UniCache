| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#1 | LRU | 4.94 | 2.608 | 1.00 | 1.00 | 0.6139 |
| Trace#1 | LFU | 8.47 | 3.286 | 1.26 | 1.71 | 0.3036 |
| Trace#1 | FIFO | 6.87 | 2.288 | 0.88 | 1.39 | 0.6178 |
| Trace#1 | LeCaR | 7.42 | 2.060 | 0.79 | 1.50 | 0.6588 |
| Trace#1 | ARC | 4.54 | 2.596 | 1.00 | 0.92 | 0.6153 |
| Trace#1 | WA | 5.14 | 3.320 | 1.27 | 1.04 | 0.4980 |
| Trace#1 | OPT (Offline) | 3.84 | 1.454 | 0.56 | 0.78 | 0.6220 |
| Trace#1 | UniCache | 3.87 | 1.796 | 0.69 | 0.78 | 0.7320 |
| Trace#1 | NONE (no caching) | — | 1.875 | 0.72 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#1 | 0.55 | 5/7 |
