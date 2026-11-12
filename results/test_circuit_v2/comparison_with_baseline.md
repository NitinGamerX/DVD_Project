# 3-bit CLA: this run (models_v2) vs baseline (results/test_circuit)

| Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err |
|---|---|---|---|---|---|---|---|
| total from every source at VDD | baseline (results/test_circuit) | 0.97 % | 4.36 % | 1.41 % | 98.59 % | -1.19 % | 8.91 % |
| total from every source at VDD | this run (models_v2) | 0.73 % | 2.81 % | 1.01 % | 98.99 % | -0.86 % | 5.45 % |
| VDD supply only | baseline (results/test_circuit) | 1.02 % | 4.48 % | 1.47 % | 98.53 % | -1.24 % | 8.98 % |
| VDD supply only | this run (models_v2) | 0.79 % | 2.91 % | 1.06 % | 98.94 % | -0.92 % | 5.53 % |
