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
| D5 | Internal node voltages for gate ML prediction | SPICE-recorded voltages (§3 step 4) drive the reported results; internal nodes also solved by Kirchhoff with the ML models and reported alongside (§5 "from both") | Implemented in Phase 5 |
| D6 | 3-bit CLA structure and carry-in | Built only from INV/NAND2/NOR2/AND2/OR2; carry-in present → 7 inputs, 128 combinations | **Ask TA** |
| D7 | Sign convention of currents in the CSVs | ngspice source branch current I(Vx), as in the example netlist and the slides' sample database | Default |
| D8 | Simulator and environment | ngspice 45.2, Python 3.14.4 on the project laptop (WSL2), recorded in `results/env/environment.txt`; development used ngspice 42 / Python 3.12, and the Phase 1 sensitivity table matched to 4 decimals | Settled |
| D9 | Which oxide parameters vary | Only `toxe`, as section 4 lists; `toxp`, `toxm`, `toxref`, `dtox` stay at their model-file values. ngspice already ignores `dtox` for this file (warning on every run, also with the unmodified file). | Follows the document |
| D10 | Current used for the transistor `is` column when source = drain voltage | Whatever ngspice reports; no special handling (Vd = Vs has probability zero under continuous sampling) | Default |
| D11 | Kirchhoff pass criterion | \|ΣI\| ≤ 1e-6·Σ\|I\| + 1e-13 A: reltol of the netlist, plus an absolute floor 10× above the worst measured rounding residual and ~100× below the smallest dataset current | Default; reported with full residual statistics |
| D12 | What "OFF current" means in the trend plots | Current at the physical drain, per µm of effective width (W − 2·wint), from samples with the source side within 5 % of its rail and \|Vds\| ≥ 0.5 VDD; sweeps at \|Vds\| = VDD | Default |
| D13 | Transistor error metric | Relative error, scored only where SPICE resolves the value (≥ 10 × measured noise floor); pooled over Id, Ig, Is, Ib and reported per current; R² on log10\|I\| | Default |
| D14 | Model selection rule | Lowest validation p95 (pooled), ties by median; test set used once, for the chosen model | Default |
| D15 | Learning targets | Channel conductance, gate current, body current (sign-safe); Id and Is rebuilt so the four currents sum to zero exactly | Default |
| D16 | Data for the final model | Training set only (70 %), so the validation numbers stay an honest selection score | Default |
| D17 | Training hardware | CPU only, one fit per core in parallel; scikit-learn has no GPU support and these models are too small to gain from one | Default |
| D18 | Order of series transistors | NAND2 pull-down: B on top (output side), A at the bottom; NOR2 pull-up: A at the top (VDD side), B at the bottom; as in the lecture slides' stack examples | Default |
| D19 | Bodies in gates and the test circuit | NMOS bodies to ground, PMOS bodies to VDD | Standard static CMOS |
| D20 | Gate corners | 200 corners, seed 5050: VDD, temperature, then NMOS and PMOS process draws; saved in `data/gates/corners.csv` and reused for the test circuit | Follows the document |
| D21 | Gate leakage from transistor currents | Kirchhoff over the node set grown from the sources through ON transistors (exact; carried by the OFF devices). Checked: exact single-transistor SPICE currents reproduce gate SPICE to 6e-8 for INV/NAND2/NOR2 | Default |
| D22 | Gate voltage of transistors driven by internal nodes | Read as its logic rail (models know only Vg = 0 or VDD). Leaves a floor of 0.08–0.67 % median error in AND2/OR2 (nand_out / nor_out sit up to ~7 mV off the rail) | Inherent in the four-case method |
| D23 | Improvement step (Phase 3b) | 20,000 extra samples per case (seeds 2101–2104, `data/fet_extra/`) added to the original 7,000-sample training split; validation and test unchanged; 5 networks per case with the originally chosen configuration, averaged → `models_v2/`. The original 10,000-row datasets, models and results are never modified (checked by file fingerprints). Whether the extra samples count toward deliverable 1 is a TA question | Proposed |

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

## Findings from Phase 4

- All 3600 gate operating points converge with every output and internal logic node within 0.71 % of VDD of its rail.
- Stack effect visible: NAND2 at AB = 00 holds x at about 0.1 V; NOR2 at AB = 11 holds y at about 0.9 VDD.
- The two leakage definitions (D2) differ by up to 14× (NOR2 at AB = 11, median), because inputs held high drive gate tunnelling current into NMOS gates. The TA's answer to D2 decides which one the Phase 5 and 6 results lead with.

## Findings from Phase 5 (development models; final numbers come from `make gates-ml`)

- Gate leakage, total definition: median error 0.78 %, p95 3.72 %, mean 1.22 % (mean accuracy 98.78 %). VDD-only definition: 0.99 % / 4.86 % / 1.55 %.
- Nearly all of the error comes from the transistor models, mainly the OFF devices' channel currents. In a gate those devices sit with one terminal at each rail, a corner of the uniformly sampled training range where the models are about twice as inaccurate (test set: 0.8–1.0 % vs 0.4–0.45 % mean conductance error) and slightly underestimate, giving a ~−1 % bias.
- Rare large failures where several inputs are near their range limits at once: corner 57 (VDD 1.099 V, NMOS vth0 +8 %, 79 °C) gives +103 % for NAND2 00, because the NMOS OFF model overpredicts the top stack device (no training sample within that region).
- ML-solved internal nodes agree with SPICE within 3 mV except stack nodes balanced between two OFF devices (NAND2/AND2 x at 00, NOR2/OR2 y at 11): up to 95 mV in one corner.
