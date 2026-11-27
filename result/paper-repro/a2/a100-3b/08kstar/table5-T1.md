| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#1 | LRU | 4.94 | 0.441 | 1.00 | 1.00 | 0.7596 |
| Trace#1 | LFU | 8.47 | 7.387 | 16.75 | 1.71 | 0.3018 |
| Trace#1 | FIFO | 6.87 | 2.404 | 5.45 | 1.39 | 0.6458 |
| Trace#1 | LeCaR | 7.42 | 1.525 | 3.46 | 1.50 | 0.6927 |
| Trace#1 | ARC | 4.54 | 0.551 | 1.25 | 0.92 | 0.7536 |
| Trace#1 | WA | 5.14 | 6.335 | 14.36 | 1.04 | 0.5132 |
| Trace#1 | OPT (Offline) | 3.84 | 0.052 | 0.12 | 0.78 | 0.7081 |
| Trace#1 | UniCache | 3.87 | 0.060 | 0.14 | 0.78 | 0.8208 |
| Trace#1 | NONE (no caching) | — | 12.208 | 27.68 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#1 | 0.88 | 6/7 |
