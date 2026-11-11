# Model selection (deliverables 4 and 5)

Produced by `scripts/03_train_models.py` (`make train`). Settings: `config.yaml → part_a`.

| File | Content |
|------|---------|
| `model_comparison.csv`, `.png` | best configuration of each of the 4 model types per case, validation set (deliverable 4) |
| `all_configs.csv` | every grid point tried, with its validation metrics |
| `chosen_models.csv` | chosen model per case, test-set metrics (deliverable 5) |
| `pred_vs_spice_<case>.png` | predicted vs SPICE for Id, Ig, Is, Ib on the test set (deliverable 5) |
| `learning_curve.csv`, `.png` | validation error of the chosen configuration vs training-set size |
| `tables.md` | the two tables in readable form |
| `runs/` | cache, one JSON per fit; delete it to force a full re-run |

## Method

- **Split:** 70 / 15 / 15 train / validation / test per case, fixed seed (`data/fet/splits.json`).
- **Model types:** polynomial ridge regression, random forest, gradient boosting, MLP
  (tanh, L-BFGS, one network per target), each over the grid in `config.yaml`.
- **Targets:** channel conductance G = (Jd − Js) / 2(Vd − Vs), gate current and body
  current (sign-safe asinh). Id and Is are rebuilt from them, so the four predicted
  currents always sum to zero. See `src/dvd/fetmodel.py`.
- **Error:** relative error |pred − SPICE| / |SPICE| in %, pooled over Id, Ig, Is, Ib
  and also reported per current. A SPICE value is scored only where the simulator
  resolves it (≥ 10 × its measured noise floor); this excludes only most PMOS OFF body
  currents. R² is on log10|I|.
- **Selection:** lowest validation p95 (pooled), ties broken by median. The test set
  is used once, for the chosen model only.
- **Also reported:** error relative to each device's largest terminal current
  (`scaled_*` columns), the measure that matters when currents are summed into a
  gate's leakage.

## Known limits

- Where Id or Is is a small difference of large parts (near its sign change), its
  relative error is large even when every part is accurate; this sets the Id/Is p95.
- The NMOS OFF body current changes sign around 1e-17 A and dominates that case's
  pooled p95, though it is about a million times smaller than the drain current.
