| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#2 | LRU | 0.28 | 0.418 | 1.00 | 1.00 | 0.7781 |
| Trace#2 | LFU | 0.53 | 3.307 | 7.91 | 1.89 | 0.4939 |
| Trace#2 | FIFO | 0.36 | 0.785 | 1.88 | 1.29 | 0.7192 |
| Trace#2 | LeCaR | 0.47 | 0.954 | 2.28 | 1.68 | 0.6779 |
| Trace#2 | ARC | 0.26 | 0.420 | 1.00 | 0.93 | 0.7779 |
| Trace#2 | WA | 0.29 | 0.634 | 1.52 | 1.04 | 0.7470 |
| Trace#2 | OPT (Offline) | 0.21 | 0.247 | 0.59 | 0.75 | 0.8681 |
| Trace#2 | UniCache | 0.24 | 0.358 | 0.86 | 0.86 | 0.8213 |
| Trace#2 | NONE (no caching) | — | 7.743 | 18.53 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#2 | 0.98 | 6/7 |
