| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#1 | LRU | 4.94 | 10.672 | 1.00 | 1.00 | 0.5100 |
| Trace#1 | LFU | 8.47 | 12.078 | 1.13 | 1.71 | 0.2885 |
| Trace#1 | FIFO | 6.87 | 11.062 | 1.04 | 1.39 | 0.4775 |
| Trace#1 | LeCaR | 7.42 | 6.811 | 0.64 | 1.50 | 0.6268 |
| Trace#1 | ARC | 4.54 | 10.678 | 1.00 | 0.92 | 0.5091 |
| Trace#1 | WA | 5.14 | 12.306 | 1.15 | 1.04 | 0.4409 |
| Trace#1 | OPT (Offline) | 3.84 | 1.485 | 0.14 | 0.78 | 0.6688 |
| Trace#1 | UniCache | 3.87 | 9.177 | 0.86 | 0.78 | 0.5520 |
| Trace#1 | NONE (no caching) | — | 17.755 | 1.66 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#1 | 0.52 | 5/7 |
