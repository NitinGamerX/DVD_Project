# Error analysis across model versions

Mean accuracy = 100 % − mean relative error. Same 3,600 gate points and 25,600 adder points for every run.

## Accuracy progression

| Level | Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err |
|---|---|---|---|---|---|---|---|---|
| gates | total from every source at VDD | v1: 10k uniform, 1 network | 0.71 % | 3.47 % | 1.09 % | 98.91 % | -0.59 % | 25.7 % |
| gates | VDD supply only | v1: 10k uniform, 1 network | 0.94 % | 4.60 % | 1.44 % | 98.56 % | -0.59 % | 25.7 % |
| 3-bit CLA | total from every source at VDD | v1: 10k uniform, 1 network | 0.97 % | 4.36 % | 1.41 % | 98.59 % | -1.19 % | 8.9 % |
| 3-bit CLA | VDD supply only | v1: 10k uniform, 1 network | 1.02 % | 4.48 % | 1.47 % | 98.53 % | -1.24 % | 9.0 % |
| gates | total from every source at VDD | v2: 27k uniform, 5 networks | 0.42 % | 1.63 % | 0.58 % | 99.42 % | -0.35 % | 5.0 % |
| gates | VDD supply only | v2: 27k uniform, 5 networks | 0.57 % | 2.04 % | 0.76 % | 99.24 % | -0.41 % | 7.2 % |
| 3-bit CLA | total from every source at VDD | v2: 27k uniform, 5 networks | 0.73 % | 2.81 % | 1.01 % | 98.99 % | -0.86 % | 5.5 % |
| 3-bit CLA | VDD supply only | v2: 27k uniform, 5 networks | 0.79 % | 2.91 % | 1.06 % | 98.94 % | -0.92 % | 5.5 % |
| gates | total from every source at VDD | v3: + rail data & features, 5 networks | 0.40 % | 1.46 % | 0.52 % | 99.48 % | -0.23 % | 3.6 % |
| gates | VDD supply only | v3: + rail data & features, 5 networks | 0.48 % | 1.54 % | 0.59 % | 99.41 % | -0.28 % | 6.2 % |
| 3-bit CLA | total from every source at VDD | v3: + rail data & features, 5 networks | 0.82 % | 2.46 % | 0.96 % | 99.04 % | -0.86 % | 4.4 % |
| 3-bit CLA | VDD supply only | v3: + rail data & features, 5 networks | 0.87 % | 2.55 % | 1.01 % | 98.99 % | -0.90 % | 4.5 % |

## Error by corner condition (total leakage)

Median |error| in the lower / middle / upper third of each corner variable, and the rank correlation (Spearman rho) of |error| with that variable.

### gates, v1: 10k uniform, 1 network

| Variable | Lower third | Middle third | Upper third | rho with abs(error) |
|---|---|---|---|---|
| vdd | 0.68 % | 0.56 % | 0.92 % | +0.13 |
| temp | 0.57 % | 0.61 % | 1.17 % | +0.27 |
| vth0_n | 0.76 % | 0.68 % | 0.69 % | -0.04 |
| toxe_n | 0.62 % | 0.67 % | 0.90 % | +0.13 |
| ndep_n | 0.88 % | 0.66 % | 0.61 % | -0.13 |
| u0_n | 0.73 % | 0.72 % | 0.69 % | -0.03 |
| xj_n | 0.73 % | 0.62 % | 0.80 % | +0.05 |
| vth0_p | 0.59 % | 0.70 % | 0.90 % | +0.14 |
| toxe_p | 0.63 % | 0.70 % | 0.85 % | +0.12 |
| ndep_p | 0.91 % | 0.62 % | 0.62 % | -0.16 |
| u0_p | 0.70 % | 0.76 % | 0.67 % | -0.02 |
| xj_p | 0.83 % | 0.61 % | 0.70 % | -0.06 |

### 3-bit CLA, v1: 10k uniform, 1 network

| Variable | Lower third | Middle third | Upper third | rho with abs(error) |
|---|---|---|---|---|
| vdd | 1.03 % | 0.63 % | 1.18 % | +0.15 |
| temp | 0.49 % | 0.76 % | 2.31 % | +0.63 |
| vth0_n | 1.18 % | 0.94 % | 0.78 % | -0.14 |
| toxe_n | 0.97 % | 0.77 % | 1.25 % | +0.14 |
| ndep_n | 1.17 % | 0.83 % | 0.85 % | -0.14 |
| u0_n | 1.15 % | 0.83 % | 0.94 % | -0.05 |
| xj_n | 0.96 % | 0.77 % | 1.24 % | +0.11 |
| vth0_p | 0.71 % | 1.00 % | 1.37 % | +0.20 |
| toxe_p | 0.89 % | 0.72 % | 1.54 % | +0.15 |
| ndep_p | 1.39 % | 0.79 % | 0.73 % | -0.21 |
| u0_p | 0.80 % | 1.05 % | 1.00 % | +0.04 |
| xj_p | 1.17 % | 0.72 % | 0.96 % | -0.08 |

### gates, v2: 27k uniform, 5 networks

| Variable | Lower third | Middle third | Upper third | rho with abs(error) |
|---|---|---|---|---|
| vdd | 0.33 % | 0.39 % | 0.58 % | +0.19 |
| temp | 0.32 % | 0.41 % | 0.63 % | +0.25 |
| vth0_n | 0.45 % | 0.41 % | 0.40 % | -0.05 |
| toxe_n | 0.39 % | 0.39 % | 0.51 % | +0.09 |
| ndep_n | 0.44 % | 0.41 % | 0.40 % | -0.04 |
| u0_n | 0.45 % | 0.39 % | 0.44 % | +0.00 |
| xj_n | 0.43 % | 0.41 % | 0.42 % | +0.01 |
| vth0_p | 0.36 % | 0.40 % | 0.58 % | +0.16 |
| toxe_p | 0.35 % | 0.39 % | 0.61 % | +0.19 |
| ndep_p | 0.43 % | 0.40 % | 0.43 % | +0.01 |
| u0_p | 0.44 % | 0.45 % | 0.39 % | -0.01 |
| xj_p | 0.47 % | 0.37 % | 0.44 % | -0.02 |

### 3-bit CLA, v2: 27k uniform, 5 networks

| Variable | Lower third | Middle third | Upper third | rho with abs(error) |
|---|---|---|---|---|
| vdd | 0.61 % | 0.61 % | 0.97 % | +0.15 |
| temp | 0.29 % | 0.69 % | 1.74 % | +0.73 |
| vth0_n | 0.92 % | 0.87 % | 0.50 % | -0.20 |
| toxe_n | 0.68 % | 0.63 % | 0.99 % | +0.10 |
| ndep_n | 0.81 % | 0.67 % | 0.75 % | -0.01 |
| u0_n | 0.87 % | 0.53 % | 0.83 % | -0.04 |
| xj_n | 0.77 % | 0.62 % | 0.78 % | +0.05 |
| vth0_p | 0.46 % | 0.65 % | 1.27 % | +0.24 |
| toxe_p | 0.49 % | 0.59 % | 1.18 % | +0.25 |
| ndep_p | 0.78 % | 0.70 % | 0.72 % | +0.02 |
| u0_p | 0.73 % | 0.86 % | 0.66 % | -0.07 |
| xj_p | 1.09 % | 0.53 % | 0.79 % | -0.08 |

### gates, v3: + rail data & features, 5 networks

| Variable | Lower third | Middle third | Upper third | rho with abs(error) |
|---|---|---|---|---|
| vdd | 0.45 % | 0.36 % | 0.40 % | -0.03 |
| temp | 0.36 % | 0.39 % | 0.45 % | +0.13 |
| vth0_n | 0.43 % | 0.37 % | 0.40 % | -0.01 |
| toxe_n | 0.41 % | 0.40 % | 0.39 % | -0.01 |
| ndep_n | 0.40 % | 0.38 % | 0.43 % | +0.01 |
| u0_n | 0.41 % | 0.40 % | 0.40 % | -0.03 |
| xj_n | 0.38 % | 0.42 % | 0.40 % | +0.03 |
| vth0_p | 0.39 % | 0.40 % | 0.41 % | +0.03 |
| toxe_p | 0.43 % | 0.40 % | 0.36 % | -0.08 |
| ndep_p | 0.44 % | 0.36 % | 0.39 % | -0.06 |
| u0_p | 0.39 % | 0.43 % | 0.39 % | +0.01 |
| xj_p | 0.40 % | 0.41 % | 0.38 % | -0.01 |

### 3-bit CLA, v3: + rail data & features, 5 networks

| Variable | Lower third | Middle third | Upper third | rho with abs(error) |
|---|---|---|---|---|
| vdd | 0.96 % | 0.70 % | 0.78 % | -0.09 |
| temp | 0.39 % | 0.80 % | 1.53 % | +0.73 |
| vth0_n | 0.95 % | 0.90 % | 0.65 % | -0.18 |
| toxe_n | 0.78 % | 0.87 % | 0.83 % | +0.05 |
| ndep_n | 0.82 % | 0.76 % | 0.92 % | +0.01 |
| u0_n | 0.89 % | 0.72 % | 0.87 % | -0.04 |
| xj_n | 0.82 % | 0.80 % | 0.86 % | +0.03 |
| vth0_p | 0.60 % | 0.87 % | 1.01 % | +0.23 |
| toxe_p | 0.74 % | 0.80 % | 0.88 % | +0.05 |
| ndep_p | 0.91 % | 0.77 % | 0.79 % | -0.07 |
| u0_p | 0.83 % | 0.88 % | 0.74 % | +0.00 |
| xj_p | 0.99 % | 0.67 % | 0.77 % | -0.05 |

## Worst gate points of v3: + rail data & features, 5 networks, traced to transistors

Each row: one device terminal's share of the gate leakage error, ML vs exact single-transistor SPICE at the same terminal voltages (only the largest contributions).

- **AND2 10, corner 123: -3.62 %** (T = 110 °C, VDD = 1.044 V, NMOS vth0 -9.1 %, NMOS toxe +4.9 %)
  - mnb (NMOS OFF) drain: -1.25 % (ML/exact − 1 = -1.9 %)
  - mna (NMOS ON) gate: -0.15 % (ML/exact − 1 = -2.7 %)
  - mpb (PMOS ON) drain: +0.11 % (ML/exact − 1 = -0.2 %)
- **NOR2 11, corner 145: -3.13 %** (T = 97 °C, VDD = 1.090 V, NMOS vth0 -6.5 %, NMOS toxe +2.5 %)
  - mnb (NMOS ON) gate: -1.52 % (ML/exact − 1 = -4.0 %)
  - mna (NMOS ON) gate: -1.52 % (ML/exact − 1 = -4.0 %)
  - mpb (PMOS OFF) gate: -0.07 % (ML/exact − 1 = -0.6 %)
- **AND2 10, corner 145: -3.09 %** (T = 97 °C, VDD = 1.090 V, NMOS vth0 -6.5 %, NMOS toxe +2.5 %)
  - mnb (NMOS OFF) drain: -0.76 % (ML/exact − 1 = -1.3 %)
  - mpb (PMOS ON) source: -0.75 % (ML/exact − 1 = -1.1 %)
  - mpb (PMOS ON) drain: +0.73 % (ML/exact − 1 = -1.1 %)
- **AND2 10, corner 186: -3.08 %** (T = 121 °C, VDD = 0.959 V, NMOS vth0 -9.1 %, NMOS toxe +0.5 %)
  - mnb (NMOS OFF) drain: +0.14 % (ML/exact − 1 = +0.3 %)
  - mpi (PMOS OFF) source: -0.07 % (ML/exact − 1 = -0.2 %)
  - mpb (PMOS ON) source: -0.05 % (ML/exact − 1 = -0.1 %)
- **OR2 00, corner 109: -3.07 %** (T = 82 °C, VDD = 1.075 V, NMOS vth0 +9.7 %, NMOS toxe +3.9 %)
  - mpb (PMOS ON) body: -1.67 % (ML/exact − 1 = -14.0 %)
  - mpa (PMOS ON) body: -1.09 % (ML/exact − 1 = -8.8 %)
  - mpb (PMOS ON) source: +0.40 % (ML/exact − 1 = +1.2 %)
