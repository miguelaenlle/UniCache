| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#3 | LRU | 0.13 | 0.534 | 1.00 | 1.00 | 0.4095 |
| Trace#3 | LFU | 0.37 | 0.545 | 1.02 | 2.85 | 0.3515 |
| Trace#3 | FIFO | 0.25 | 0.546 | 1.02 | 1.92 | 0.3867 |
| Trace#3 | LeCaR | 0.35 | 0.522 | 0.98 | 2.69 | 0.3994 |
| Trace#3 | ARC | 0.12 | 0.528 | 0.99 | 0.92 | 0.4098 |
| Trace#3 | WA | 0.27 | 0.523 | 0.98 | 2.08 | 0.4024 |
| Trace#3 | OPT (Offline) | 0.08 | 0.417 | 0.78 | 0.62 | 0.6618 |
| Trace#3 | UniCache | 0.10 | 0.477 | 0.89 | 0.77 | 0.5525 |
| Trace#3 | NONE (no caching) | — | 0.311 | 0.58 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#3 | 0.55 | 5/7 |
