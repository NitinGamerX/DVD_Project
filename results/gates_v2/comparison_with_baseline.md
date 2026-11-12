# This run (models_v2) vs baseline (results/gates)

| Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err |
|---|---|---|---|---|---|---|---|
| total from every source at VDD | baseline (results/gates) | 0.71 % | 3.47 % | 1.09 % | 98.91 % | -0.59 % | 25.7 % |
| total from every source at VDD | this run (models_v2) | 0.42 % | 1.63 % | 0.58 % | 99.42 % | -0.35 % | 5.0 % |
| VDD supply only | baseline (results/gates) | 0.94 % | 4.60 % | 1.44 % | 98.56 % | -0.59 % | 25.7 % |
| VDD supply only | this run (models_v2) | 0.57 % | 2.04 % | 0.76 % | 99.24 % | -0.41 % | 7.2 % |
