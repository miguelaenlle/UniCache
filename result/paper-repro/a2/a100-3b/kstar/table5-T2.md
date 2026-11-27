| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#2 | LRU | 0.28 | 3.451 | 1.00 | 1.00 | 0.7660 |
| Trace#2 | LFU | 0.53 | 8.051 | 2.33 | 1.89 | 0.4953 |
| Trace#2 | FIFO | 0.36 | 5.798 | 1.68 | 1.29 | 0.6807 |
| Trace#2 | LeCaR | 0.47 | 6.076 | 1.76 | 1.68 | 0.6433 |
| Trace#2 | ARC | 0.26 | 3.457 | 1.00 | 0.93 | 0.7661 |
| Trace#2 | WA | 0.29 | 8.538 | 2.47 | 1.04 | 0.6014 |
| Trace#2 | OPT (Offline) | 0.21 | 2.155 | 0.62 | 0.75 | 0.8640 |
| Trace#2 | UniCache | 0.24 | 3.079 | 0.89 | 0.86 | 0.8043 |
| Trace#2 | NONE (no caching) | — | 15.579 | 4.51 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#2 | 0.83 | 6/7 |
