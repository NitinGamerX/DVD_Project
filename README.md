# DVD Project — ML-Driven PVT Variation Aware Leakage Estimation

Team assignment: **22nm MGK** model file · **nominal VDD = 1.0 V** · test circuit **3-bit CLA**.
Requirements: [`docs/DVD_ML_PROJECT.pdf`](docs/DVD_ML_PROJECT.pdf), followed strictly.
Every number the code uses lives in [`config.yaml`](config.yaml); open questions are in [`DECISIONS.md`](DECISIONS.md).

## Setup (WSL2 Ubuntu 24.04)

```bash
./setup.sh                # ngspice, Python venv with pinned packages, git init, checks
./setup.sh --with-latex   # same, plus LaTeX for the report
code .                    # open in VS Code (WSL)
```

`make check` re-runs the environment check at any time.

## Layout

| Path | Contents |
|------|----------|
| `model/22nm_MGK.pm` | assigned model file, never edited (sha256 in `model/SHA256SUMS`) |
| `model/derived/` | generated copy with vth0, toxe, ndep, u0, xj as SPICE parameters (Phase 1) |
| `docs/` | project document and annotated lecture slides |
| `reference/22n_NMOS_ON.net` | instructor's example netlist, as received |
| `config.yaml` | all parameters, ranges, seeds and rules from the document |
| `src/dvd/` | shared library: config, model-file parser, sampling, ngspice runner, netlist builder, single-FET simulation |
| `scripts/NN_*.py` | one script per step, run in numeric order |
| `netlists/` | example netlists: `single_fet/`, `gates/`, `test_circuit/` (deliverable 2) |
| `data/` | datasets: `fet/` (four CSVs, deliverable 1), `gates/`, `test_circuit/` |
| `models/` | four saved transistor models (deliverable 5) |
| `results/` | tables and plots per phase; `env/` holds simulator and package versions |
| `report/` | LaTeX report (deliverable 10) |

## Run order

```
make check        00  environment + config check
make fet-data     01  NMOS/PMOS ON/OFF datasets, 10,000 samples each
make data-check   02  Kirchhoff check, trend plots
make train        03  4 model types x 4 cases, pick best on validation, score on test
make gates-spice  04  18 gate/input cases x 200 corners (evaluation only)
make gates-ml     05  gate leakage from transistor models vs SPICE
make cla-spice    06  3-bit CLA, all input combinations x same 200 corners
make cla-ml       07  CLA leakage from gate models vs SPICE
make analysis     08  error analysis
make report           build report/main.pdf
make all              everything, in order, fixed seeds
```

## Rules enforced by the code

- Every netlist contains `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`.
- Single-transistor netlists give every terminal its own voltage source.
- Every transistor has source/drain areas and perimeters (AS, AD, PS, PD).
- Gate/CLA SPICE results are never used for training.
- All random draws use the seeds in `config.yaml`.
