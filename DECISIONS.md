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
| D11 | Kirchhoff pass criterion | \|ΣI\| ≤ 1e-6·Σ\|I\| + 1e-13 A: reltol of the netlist, plus an absolute floor 10× above the worst measured rounding residual and ~100× below the smallest dataset current | Default; reported with full residual statistics |
| D12 | What "OFF current" means in the trend plots | Current at the physical drain, per µm of effective width (W − 2·wint), from samples with the source side within 5 % of its rail and \|Vds\| ≥ 0.5 VDD; sweeps at \|Vds\| = VDD | Default |
| D13 | Transistor error metric | Relative error, scored only where SPICE resolves the value (≥ 10 × measured noise floor); pooled over Id, Ig, Is, Ib and reported per current; R² on log10\|I\| | Default |
| D14 | Model selection rule | Lowest validation p95 (pooled), ties by median; test set used once, for the chosen model | Default |
| D15 | Learning targets | Channel conductance, gate current, body current (sign-safe); Id and Is rebuilt so the four currents sum to zero exactly | Default |
| D16 | Data for the final model | Training set only (70 %), so the validation numbers stay an honest selection score | Default |
| D17 | Training hardware | CPU only, one fit per core in parallel; scikit-learn has no GPU support and these models are too small to gain from one | Default |

## Findings from Phase 1

- Batch (alterparam) and literal netlists agree to about 1e-14 relative on Id/Is, so every sampled parameter is applied.
- Sensitivity at nominal bias, −10 % → +10 % of each parameter: vth0 changes OFF current by 70–77 %, toxe changes gate current by about 90 %, xj changes currents by at most about 1 %.
- PMOS OFF body current is below the simulator's resolution (about 1e-16 A) in most samples; see `data/fet/README.md`.

## Findings from Phase 2

- All 40,000 samples pass the Kirchhoff check; the worst residual is 1.2e-14 A, a rounding floor rather than a convergence error.
- OFF current at the drain is partly gate tunnelling: 78 % (NMOS) and 42 % (PMOS) at −40 °C, so the total rises only ×15 and ×9 to 125 °C while the channel part rises ×64 and ×15.
- Currents scale with effective width W − 2·wint, not drawn width (exactly so for gate current); worth using as a model input in Phase 3.

## Findings from Phase 3 (development runs; final numbers come from `make train`)

- In a one-setting-per-type run on all four cases, the MLP beat polynomial ridge, random forest and gradient boosting by about 10× in median validation error.
- One network per target beats one shared network (gate-current error roughly halved, NMOS OFF).
- Training and validation errors are nearly equal: the networks underfit rather than overfit. A larger network (96×3), alpha from 1e-7 to 1e-4 and optimiser restarts did not help; scaling the targets by 100 made it worse (reverted).
- Id/Is tail errors come from samples where Id is a small difference of large parts (near its sign change); away from those, Id error is ~0.2 % median, ~1.7 % p95 (NMOS OFF).
- The NMOS OFF body current changes sign around 1e-17 A and dominates that case's pooled p95; relative to each device's largest current the error is under 1 %.
