# Data check (deliverable 3)

Produced by `scripts/02_data_check.py` (`make data-check`) from `data/fet/*.csv`.
Settings are in `config.yaml → data_check`.

| File | Content |
|------|---------|
| `kirchhoff.csv`, `kirchhoff.png` | Kirchhoff check per case: residual statistics and residual vs current magnitude |
| `off_current_vs_temperature.png` | trend 1: OFF current vs temperature (NMOS OFF, PMOS OFF) |
| `off_current_vs_vth0.png` | trend 2: OFF current vs vth0 (NMOS OFF, PMOS OFF) |
| `gate_current_vs_toxe.png` | trend 3: gate current vs toxe (all four cases) |
| `trend_summary.csv` | rank correlations and sweep metrics behind the three trends |
| `numerical_resolution.csv` | solver noise floor per current (written by Phase 1) |
| `process_sensitivity.csv` | ±10 % sensitivity of each process parameter (written by Phase 1) |

## Kirchhoff check

Criterion: |Id + Ig + Is + Ib| ≤ 1e-6 · (|Id| + |Ig| + |Is| + |Ib|) + 1e-13 A.
The relative part is the netlist `reltol`. The absolute part is set by double-precision
rounding in the solver, not by convergence: residuals are flat with current magnitude
(see the figure). They sit near 1e-16 A when a terminal is held at VDD and near
1e-23 A for NMOS OFF, where gate, body and source sources are at or near 0 V. The netlist
`abstol` (1e-18 A) is below what double precision can resolve at a 1 V node, which is
why a purely relative test would fail about 5 % of PMOS OFF samples.
Result: 40,000 / 40,000 samples pass; worst residual 1.2e-14 A.

## Trends

Each figure overlays three things: the dataset (scatter, coloured by the main
confounding variable), its binned median, and a SPICE sweep of the single parameter
with everything else nominal. Currents are per µm of effective width W − 2·wint
(wint = 5 nm in the model file); gate current scales exactly with this width.

OFF current is the current at the physical drain, taken from samples at classic OFF
bias: source side within 5 % of its rail and |Vds| ≥ 0.5 VDD. The sweeps split the
drain current into a channel (subthreshold) part and a gate-tunnelling part.

- **OFF current vs temperature.** Rises with temperature in data and sweep.
  Sweep from −40 to 125 °C: total ×15 (NMOS) and ×9 (PMOS), channel part ×64 and ×15.
  At −40 °C gate tunnelling is 78 % (NMOS) and 42 % (PMOS) of the OFF drain current.
  It does not depend on temperature, which flattens the total.
- **OFF current vs vth0.** Falls with |vth0|. The channel part falls one decade per
  81–82 mV of |vth0|, consistent with a subthreshold ideality factor of about 1.37.
  The dataset correlation is modest (−0.37 NMOS, −0.29 PMOS) because temperature
  dominates the scatter; with temperature held fixed it is −0.44 and −0.40.
- **Gate current vs toxe.** Falls by about 0.8 decades per Å of toxe in every case.
  Dataset correlation −0.44 to −0.61, and −0.69 to −0.89 with oxide voltage held
  fixed. The sweep is at the largest oxide voltage, so it bounds the cloud from above.

The data and the sweeps give the physically expected direction for every trend.
The dataset sits below the sweep in the OFF-current plots because most samples have
less than VDD across drain and gate, and the source a little off the rail.
