# Transistor models v1 / v2 / v3: uniform and rail-focused validation and test sets

Pooled relative error over Id, Ig, Is, Ib. 'vs largest p95' = 95th percentile of the error relative to the device's largest terminal current (what matters when currents are summed into gate leakage).

| Case | Set | Variant | Median | p95 | Mean | vs largest p95 | Ig p95 |
|---|---|---|---|---|---|---|---|
| nmos_on | uniform test | v1 (10k uniform, 1 network) | 0.458 % | 2.75 % | 0.90 % | 1.65 % | 4.02 % |
| nmos_on | uniform test | v2 (27k uniform, 5 networks) | 0.265 % | 1.76 % | 0.55 % | 1.09 % | 2.51 % |
| nmos_on | uniform test | v3, 1 network | 0.522 % | 2.47 % | 0.80 % | 1.55 % | 2.75 % |
| nmos_on | uniform test | v3 (34k incl. rail, 5 networks) | 0.302 % | 1.65 % | 0.51 % | 1.08 % | 1.81 % |
| nmos_on | rail test | v1 (10k uniform, 1 network) | 0.620 % | 10.34 % | 882.78 % | 4.89 % | 58.91 % |
| nmos_on | rail test | v2 (27k uniform, 5 networks) | 0.356 % | 5.38 % | 578.10 % | 2.44 % | 35.77 % |
| nmos_on | rail test | v3, 1 network | 0.651 % | 3.74 % | 3.22 % | 2.44 % | 4.04 % |
| nmos_on | rail test | v3 (34k incl. rail, 5 networks) | 0.392 % | 2.49 % | 1.78 % | 1.52 % | 2.94 % |
| nmos_off | uniform test | v1 (10k uniform, 1 network) | 0.442 % | 20.02 % | 32.67 % | 0.72 % | 0.97 % |
| nmos_off | uniform test | v2 (27k uniform, 5 networks) | 0.250 % | 7.25 % | 2.56 % | 0.43 % | 0.56 % |
| nmos_off | uniform test | v3, 1 network | 0.393 % | 10.88 % | 13.44 % | 0.61 % | 0.74 % |
| nmos_off | uniform test | v3 (34k incl. rail, 5 networks) | 0.255 % | 6.51 % | 2.30 % | 0.45 % | 0.51 % |
| nmos_off | rail test | v1 (10k uniform, 1 network) | 0.902 % | 79.61 % | 2997.99 % | 3.43 % | 4.24 % |
| nmos_off | rail test | v2 (27k uniform, 5 networks) | 0.529 % | 30.25 % | 141.00 % | 2.06 % | 3.07 % |
| nmos_off | rail test | v3, 1 network | 0.651 % | 17.41 % | 5.67 % | 1.39 % | 1.42 % |
| nmos_off | rail test | v3 (34k incl. rail, 5 networks) | 0.427 % | 9.98 % | 2.89 % | 1.03 % | 1.01 % |
| pmos_on | uniform test | v1 (10k uniform, 1 network) | 0.364 % | 2.18 % | 0.72 % | 1.11 % | 2.55 % |
| pmos_on | uniform test | v2 (27k uniform, 5 networks) | 0.208 % | 1.51 % | 0.44 % | 0.67 % | 1.33 % |
| pmos_on | uniform test | v3, 1 network | 0.337 % | 1.65 % | 0.53 % | 0.93 % | 1.68 % |
| pmos_on | uniform test | v3 (34k incl. rail, 5 networks) | 0.211 % | 1.23 % | 0.37 % | 0.68 % | 1.29 % |
| pmos_on | rail test | v1 (10k uniform, 1 network) | 0.506 % | 7.91 % | 173.23 % | 2.96 % | 21.24 % |
| pmos_on | rail test | v2 (27k uniform, 5 networks) | 0.294 % | 5.76 % | 108.34 % | 1.91 % | 9.49 % |
| pmos_on | rail test | v3, 1 network | 0.480 % | 2.94 % | 1.28 % | 1.61 % | 3.15 % |
| pmos_on | rail test | v3 (34k incl. rail, 5 networks) | 0.276 % | 2.11 % | 1.03 % | 1.22 % | 2.32 % |
| pmos_off | uniform test | v1 (10k uniform, 1 network) | 0.336 % | 4.90 % | 2.40 % | 0.77 % | 0.98 % |
| pmos_off | uniform test | v2 (27k uniform, 5 networks) | 0.173 % | 2.54 % | 1.38 % | 0.45 % | 0.57 % |
| pmos_off | uniform test | v3, 1 network | 0.366 % | 3.67 % | 2.56 % | 0.76 % | 1.08 % |
| pmos_off | uniform test | v3 (34k incl. rail, 5 networks) | 0.205 % | 2.56 % | 1.50 % | 0.45 % | 0.54 % |
| pmos_off | rail test | v1 (10k uniform, 1 network) | 0.687 % | 16.96 % | 29.83 % | 3.33 % | 5.23 % |
| pmos_off | rail test | v2 (27k uniform, 5 networks) | 0.370 % | 10.98 % | 18.20 % | 1.65 % | 2.97 % |
| pmos_off | rail test | v3, 1 network | 0.497 % | 5.05 % | 3.27 % | 1.14 % | 1.72 % |
| pmos_off | rail test | v3 (34k incl. rail, 5 networks) | 0.307 % | 3.72 % | 1.39 % | 0.86 % | 0.96 % |
