# Decision log

Choices the project document (docs/DVD_ML_PROJECT.pdf) does not fix.
Each default is mirrored in `config.yaml → pending_ta`. When a TA answers,
update the status here and the value there in the same commit.

| # | Question | Current default | Status |
|---|----------|-----------------|--------|
| D1 | Supply voltage meaning of the assigned "1 V" | Nominal VDD = 1.0 V, sampled 0.9–1.1 V (§2 table lists it as "Nominal VDD") | Settled by the document |
| D2 | Leakage definition for gates and the CLA | Total current from all sources at VDD level (supply + inputs held high) = static power / VDD. I(VDD) alone is also recorded. | **Ask TA** |
| D3 | Transistor widths inside gates and the CLA | INV 44/88 nm, NAND2 88/88 nm, NOR2 44/176 nm (W_n/W_p), all within 1–8 × Lmin | **Ask TA** |
| D4 | Source/drain geometry | AS = AD = W·Ldiff, PS = PD = 2(W + Ldiff), Ldiff = 44 nm (value defined in the example netlist), identical in training and gate/CLA netlists | Team requirement to include areas; formula **ask TA** |
| D5 | Internal node voltages for gate ML prediction | SPICE-recorded voltages (§3 step 4) drive the reported results; ML-solved nodes reported alongside (§5 "from both") | **Ask TA** |
| D6 | 3-bit CLA structure and carry-in | Built only from INV/NAND2/NOR2/AND2/OR2; carry-in present → 7 inputs, 128 combinations | **Ask TA** |
| D7 | Sign convention of currents in the CSVs | ngspice source branch current I(Vx), as in the example netlist and the slides' sample database | Default |
| D8 | Simulator | ngspice 42 (Ubuntu 24.04 package), BSIM4 level 54 from the model file | Default |
| D9 | Which oxide parameters vary | Only `toxe`, as section 4 lists; `toxp`, `toxm`, `toxref`, `dtox` stay at their model-file values. ngspice already ignores `dtox` for this file (warning on every run, also with the unmodified file). | Follows the document |
| D10 | Current used for the transistor `is` column when source = drain voltage | Whatever ngspice reports; no special handling (Vd = Vs has probability zero under continuous sampling) | Default |

## Findings from Phase 1

- Batch (alterparam) and literal netlists agree to about 1e-14 relative on Id/Is, so every sampled parameter is applied.
- Sensitivity at nominal bias, −10 % → +10 % of each parameter: vth0 changes OFF current by 70–77 %, toxe changes gate current by about 90 %, xj changes currents by at most about 1 %.
- PMOS OFF body current is below the simulator's resolution (about 1e-16 A) in most samples; see `data/fet/README.md`.
