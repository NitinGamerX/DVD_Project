# Test circuit results (Part C)

| File | Produced by | Content |
|------|-------------|---------|
| `spice_summary.md`, `spice_per_vector.csv`, `spice_leakage.png` | `06_cla_spice.py` | SPICE leakage of the full adder per input combination |
| `cla_results.md`, `cla_summary.csv`, `cla_results_per_input.csv` | `07_cla_ml.py` | **deliverable 8**: hierarchical ML vs SPICE, both leakage definitions, overall and per input combination |
| `cla_parity.png` | `07_cla_ml.py` | **deliverable 8**: ML vs SPICE leakage, all 25,600 operating points, log axes |
| `cla_errors.png` | `07_cla_ml.py` | median error per input combination, with the composition floor |
| `cla_predictions.csv` | `07_cla_ml.py` | every operating point: SPICE, ML, naive and floor estimates |

This folder uses the original 10,000-sample models (`models/`). The same files made with
the improved models (`models_v2/`) are in `results/test_circuit_v2/`, together with
`comparison_with_baseline.md`.

## How the circuit leakage is estimated (`scripts/07_cla_ml.py`)

For each input combination a logic simulation gives every cell's input bits; each cell's
leakage comes from its Part B gate model (transistor models + recorded gate node voltages),
assuming every cell input sits exactly at 0 or VDD and ignoring loading (as the document says).

- **Total leakage** = sum of the cells' total leakages.
- **Supply only** = that sum minus the current drawn from the adder's primary inputs held at 1.
- **Naive supply-only sum** (reported to show the point): adding the cells' supply-only leakages
  misses the gate current that internal nets at 1 feed into the cells they drive, so it comes
  out consistently low.
- **Composition floor**: the same sums with exact gate SPICE values instead of ML. Its error is
  the cost of treating the adder as independent gates (loading, nets slightly off the rails).
