| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#2 | LRU | 0.28 | 0.717 | 1.00 | 1.00 | 0.7737 |
| Trace#2 | LFU | 0.53 | 1.310 | 1.83 | 1.89 | 0.5025 |
| Trace#2 | FIFO | 0.36 | 0.895 | 1.25 | 1.29 | 0.7097 |
| Trace#2 | LeCaR | 0.47 | 0.936 | 1.30 | 1.68 | 0.6670 |
| Trace#2 | ARC | 0.26 | 0.705 | 0.98 | 0.93 | 0.7738 |
| Trace#2 | WA | 0.29 | 0.902 | 1.26 | 1.04 | 0.7276 |
| Trace#2 | OPT (Offline) | 0.21 | 0.602 | 0.84 | 0.75 | 0.8627 |
| Trace#2 | UniCache | 0.24 | 0.688 | 0.96 | 0.86 | 0.8125 |
| Trace#2 | NONE (no caching) | — | 1.055 | 1.47 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#2 | 0.98 | 7/7 |
