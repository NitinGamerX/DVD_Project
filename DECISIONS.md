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
