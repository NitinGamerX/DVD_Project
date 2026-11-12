# Improved transistor models vs original (original test set, 1,500 samples per case)

Relative error, pooled over Id, Ig, Is, Ib (median / p95 / mean), and 'vs largest' = error relative to the device's largest terminal current (p95).

| Case | Variant | Median | p95 | Mean | vs largest p95 | Id mean | Ig mean | Is mean |
|---|---|---|---|---|---|---|---|---|
| nmos_on | original (7,000 samples, 1 network) | 0.458 % | 2.75 % | 0.90 % | 1.65 % | 0.80 % | 1.12 % | 0.78 % |
| nmos_on | more data (27,000 samples, 1 network) | 0.401 % | 2.26 % | 0.72 % | 1.26 % | 0.64 % | 0.78 % | 0.58 % |
| nmos_on | more data + average of 5 networks | 0.265 % | 1.76 % | 0.55 % | 1.09 % | 0.49 % | 0.60 % | 0.45 % |
| nmos_off | original (7,000 samples, 1 network) | 0.442 % | 20.02 % | 32.67 % | 0.72 % | 2.27 % | 0.37 % | 2.74 % |
| nmos_off | more data (27,000 samples, 1 network) | 0.442 % | 11.94 % | 4.30 % | 0.70 % | 2.47 % | 0.35 % | 2.12 % |
| nmos_off | more data + average of 5 networks | 0.250 % | 7.25 % | 2.56 % | 0.43 % | 1.06 % | 0.20 % | 1.51 % |
| pmos_on | original (7,000 samples, 1 network) | 0.364 % | 2.18 % | 0.72 % | 1.11 % | 0.57 % | 0.91 % | 0.53 % |
| pmos_on | more data (27,000 samples, 1 network) | 0.330 % | 1.79 % | 0.57 % | 0.90 % | 0.44 % | 0.55 % | 0.43 % |
| pmos_on | more data + average of 5 networks | 0.208 % | 1.51 % | 0.44 % | 0.67 % | 0.33 % | 0.39 % | 0.32 % |
| pmos_off | original (7,000 samples, 1 network) | 0.336 % | 4.90 % | 2.40 % | 0.77 % | 3.95 % | 0.41 % | 2.64 % |
| pmos_off | more data (27,000 samples, 1 network) | 0.328 % | 3.29 % | 2.40 % | 0.70 % | 2.04 % | 0.39 % | 4.91 % |
| pmos_off | more data + average of 5 networks | 0.173 % | 2.54 % | 1.38 % | 0.45 % | 1.29 % | 0.20 % | 2.55 % |
