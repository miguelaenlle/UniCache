| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#2 | LRU | 0.28 | 0.018 | 1.00 | 1.00 | 0.7815 |
| Trace#2 | LFU | 0.53 | 0.060 | 3.44 | 1.89 | 0.5207 |
| Trace#2 | FIFO | 0.36 | 0.021 | 1.18 | 1.29 | 0.7340 |
| Trace#2 | LeCaR | 0.47 | 0.024 | 1.38 | 1.68 | 0.6903 |
| Trace#2 | ARC | 0.26 | 0.017 | 1.00 | 0.93 | 0.7817 |
| Trace#2 | WA | 0.29 | 0.019 | 1.09 | 1.04 | 0.7578 |
| Trace#2 | OPT (Offline) | 0.21 | 0.014 | 0.81 | 0.75 | 0.8689 |
| Trace#2 | UniCache | 0.24 | 0.016 | 0.91 | 0.86 | 0.8275 |
| Trace#2 | NONE (no caching) | — | 0.867 | 49.40 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#2 | 1.00 | 7/7 |
