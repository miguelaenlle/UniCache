| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#3 | LRU | 0.13 | 0.080 | 1.00 | 1.00 | 0.4185 |
| Trace#3 | LFU | 0.37 | 0.097 | 1.22 | 2.85 | 0.3574 |
| Trace#3 | FIFO | 0.25 | 0.084 | 1.06 | 1.92 | 0.3955 |
| Trace#3 | LeCaR | 0.35 | 0.081 | 1.02 | 2.69 | 0.4091 |
| Trace#3 | ARC | 0.12 | 0.080 | 1.00 | 0.92 | 0.4193 |
| Trace#3 | WA | 0.27 | 0.081 | 1.02 | 2.08 | 0.4119 |
| Trace#3 | OPT (Offline) | 0.08 | 0.050 | 0.62 | 0.62 | 0.6675 |
| Trace#3 | UniCache | 0.10 | 0.068 | 0.85 | 0.77 | 0.5688 |
| Trace#3 | NONE (no caching) | — | 0.192 | 2.41 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#3 | 0.90 | 6/7 |
