| Trace | Policy | Paper QTTFT (s) | Sim QTTFT (s) | Sim / LRU | Paper / LRU | Sim hit ratio |
|---|---|---:|---:|---:|---:|---:|
| Trace#1 | LRU | 4.94 | 5.981 | 1.00 | 1.00 | 0.4649 |
| Trace#1 | LFU | 8.47 | 5.820 | 0.97 | 1.71 | 0.2885 |
| Trace#1 | FIFO | 6.87 | 6.049 | 1.01 | 1.39 | 0.4397 |
| Trace#1 | LeCaR | 7.42 | 4.891 | 0.82 | 1.50 | 0.5883 |
| Trace#1 | ARC | 4.54 | 6.010 | 1.00 | 0.92 | 0.4640 |
| Trace#1 | WA | 5.14 | 6.388 | 1.07 | 1.04 | 0.4036 |
| Trace#1 | OPT (Offline) | 3.84 | 4.058 | 0.68 | 0.78 | 0.4897 |
| Trace#1 | UniCache | 3.87 | 5.644 | 0.94 | 0.78 | 0.5090 |
| Trace#1 | NONE (no caching) | — | 4.053 | 0.68 | — | 0.0000 |

| Trace | Spearman rho (sim vs paper QTTFT) | Same side of LRU |
|---|---:|---:|
| Trace#1 | 0.26 | 4/7 |
