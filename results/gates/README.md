# Gate results (Part B)

| File | Produced by | Content |
|------|-------------|---------|
| `spice_summary.csv`, `.md`, `spice_leakage.png` | `04_gate_spice.py` | SPICE leakage per case over the 200 corners |
| `gate_results.csv`, `.md` | `05_gate_ml.py` | **deliverable 6**: per case SPICE median, ML median, bias, median / p95 / mean error (primary leakage definition) |
| `gate_results_total.csv`, `gate_results_vdd.csv` | `05_gate_ml.py` | the same for each leakage definition |
| `gate_parity.png` | `05_gate_ml.py` | **deliverable 7**: ML vs SPICE leakage, all 3600 operating points, log axes |
| `gate_errors.png` | `05_gate_ml.py` | error distribution per case, with the composition floor |
| `gate_predictions.csv` | `05_gate_ml.py` | every operating point: SPICE and ML leakage (both definitions), oracle and ML-solved-node leakage, SPICE output voltage, internal node voltages from SPICE and from the ML models |

## How the ML leakage is computed (`src/dvd/compose.py`)

1. **Terminal voltages.** Each transistor's drain and source take the SPICE node
   voltages (section 3 step 4); its body is at 0 (NMOS) or VDD (PMOS); its gate is at the
   logic rail of its node. The transistor's type and gate level select one of the four models.
2. **Currents.** The model gives the four terminal currents (Kirchhoff exact per transistor).
3. **Leakage.** The current delivered by a set of sources equals the net current leaving any
   node set that contains only those sources. That set is grown from VDD (plus the inputs
   at 1 for the total) through every ON transistor; the leakage is the current into all
   transistor terminals inside it. This is exact by Kirchhoff, so it relies on the OFF
   transistors and gate oxides at the boundary, the actual leakage paths.

## Columns in the results table

- **Bias**: mean signed error, (ML − SPICE) / SPICE.
- **Floor (exact currents)**: the same composition with exact single-transistor SPICE
  currents instead of the models. Zero up to rounding for INV, NAND2 and NOR2. Non-zero
  for AND2 / OR2 because the inner inverter's gate (nand_out / nor_out) sits a few mV off
  its rail, which the four-case transistor models cannot represent.
- **ML-solved nodes**: leakage when the internal nodes (x, y, nand_out, nor_out) are
  solved from Kirchhoff with the ML models instead of taken from SPICE (output still
  from SPICE); the internal-node table compares those voltages with SPICE.
