| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#2 | LRU | 0.28 | 0.072 | 1.00 | 1.00 | 0.7766 |
| Trace#2 | LFU | 0.53 | 0.199 | 2.75 | 1.89 | 0.5120 |
| Trace#2 | FIFO | 0.36 | 0.088 | 1.22 | 1.29 | 0.7272 |
| Trace#2 | LeCaR | 0.47 | 0.090 | 1.25 | 1.68 | 0.6840 |
| Trace#2 | ARC | 0.26 | 0.070 | 0.97 | 0.93 | 0.7766 |
| Trace#2 | WA | 0.29 | 0.083 | 1.14 | 1.04 | 0.7499 |
| Trace#2 | OPT (Offline) | 0.21 | 0.058 | 0.81 | 0.75 | 0.8655 |
| Trace#2 | UniCache | 0.24 | 0.065 | 0.90 | 0.86 | 0.8191 |
| Trace#2 | NONE (no caching) | — | 0.143 | 1.98 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#2 | 1.00 | 7/7 |
