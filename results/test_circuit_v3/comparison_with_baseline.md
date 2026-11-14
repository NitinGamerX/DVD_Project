# 3-bit CLA: this run vs earlier runs

| Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err |
|---|---|---|---|---|---|---|---|
| total from every source at VDD | results/test_circuit | 0.97 % | 4.36 % | 1.41 % | 98.59 % | -1.19 % | 8.91 % |
| total from every source at VDD | results/test_circuit_v2 | 0.73 % | 2.81 % | 1.01 % | 98.99 % | -0.86 % | 5.45 % |
| total from every source at VDD | results/test_circuit_v3 (this run, models_v3) | 0.82 % | 2.46 % | 0.96 % | 99.04 % | -0.86 % | 4.38 % |
| VDD supply only | results/test_circuit | 1.02 % | 4.48 % | 1.47 % | 98.53 % | -1.24 % | 8.98 % |
| VDD supply only | results/test_circuit_v2 | 0.79 % | 2.91 % | 1.06 % | 98.94 % | -0.92 % | 5.53 % |
| VDD supply only | results/test_circuit_v3 (this run, models_v3) | 0.87 % | 2.55 % | 1.01 % | 98.99 % | -0.90 % | 4.47 % |
