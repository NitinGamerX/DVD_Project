# Gate leakage: this run vs earlier runs (same 3600 operating points)

| Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err | Points > 2 % |
|---|---|---|---|---|---|---|---|---|
| total from every source at VDD | results/gates | 0.71 % | 3.47 % | 1.09 % | 98.91 % | -0.59 % | 25.7 % | 521 |
| total from every source at VDD | results/gates_v2 | 0.42 % | 1.63 % | 0.58 % | 99.42 % | -0.35 % | 5.0 % | 89 |
| total from every source at VDD | results/gates_v3 (this run, models_v3) | 0.40 % | 1.46 % | 0.52 % | 99.48 % | -0.23 % | 3.6 % | 48 |
| VDD supply only | results/gates | 0.94 % | 4.60 % | 1.44 % | 98.56 % | -0.59 % | 25.7 % | 790 |
| VDD supply only | results/gates_v2 | 0.57 % | 2.04 % | 0.76 % | 99.24 % | -0.41 % | 7.2 % | 198 |
| VDD supply only | results/gates_v3 (this run, models_v3) | 0.48 % | 1.54 % | 0.59 % | 99.41 % | -0.28 % | 6.2 % | 68 |
