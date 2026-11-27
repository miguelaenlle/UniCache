| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#3 | LRU | 0.13 | 0.745 | 1.00 | 1.00 | 0.4154 |
| Trace#3 | LFU | 0.37 | 0.845 | 1.13 | 2.85 | 0.3590 |
| Trace#3 | FIFO | 0.25 | 0.781 | 1.05 | 1.92 | 0.3921 |
| Trace#3 | LeCaR | 0.35 | 0.750 | 1.01 | 2.69 | 0.4062 |
| Trace#3 | ARC | 0.12 | 0.750 | 1.01 | 0.92 | 0.4155 |
| Trace#3 | WA | 0.27 | 0.759 | 1.02 | 2.08 | 0.4085 |
| Trace#3 | OPT (Offline) | 0.08 | 0.498 | 0.67 | 0.62 | 0.6677 |
| Trace#3 | UniCache | 0.10 | 0.646 | 0.87 | 0.77 | 0.5632 |
| Trace#3 | NONE (no caching) | — | 1.729 | 2.32 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#3 | 0.88 | 6/7 |
