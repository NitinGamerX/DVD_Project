#!/usr/bin/env python3
"""Step 01 (Part A, step 1): single-transistor datasets.

For each case (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF):
  * draw N samples of every section 4 parameter (VDD first), fixed seed per case
  * simulate each sample in ngspice with the assigned model file
  * write data/fet/<case>.csv with exactly the 16 deliverable columns

Then, as checks on the data just produced:
  * re-simulate a random subset through an independent literal-netlist path and
    require agreement (catches any parameter that silently failed to apply)
  * measure how much each process parameter moves the currents (no parameter
    may be inert)
  * write one stand-alone example netlist per case (deliverable 2) and confirm
    it reproduces its dataset row
  * write data/fet/manifest.json (seeds, versions, checksums, ranges)

Usage: python scripts/01_generate_fet_data.py [--n N] [--cases c1,c2] [--jobs J] [--verify K]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import config, fetsim, sampling, spice  # noqa: E402

cfg = config.load()
CASES = list(cfg["cases"])
COLUMNS = cfg["part_a"]["csv_columns"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n", type=int, default=cfg["part_a"]["samples_per_case"], help="samples per case")
ap.add_argument("--cases", default=",".join(CASES))
ap.add_argument("--jobs", type=int, default=None, help="parallel ngspice processes (default: all cores)")
ap.add_argument("--verify", type=int, default=25, help="samples per case re-checked via literal netlists")
args = ap.parse_args()
cases = [c.strip() for c in args.cases.split(",")]
unknown = set(cases) - set(CASES)
if unknown:
    sys.exit(f"unknown case(s): {sorted(unknown)}; valid: {CASES}")

data_dir = config.path("data", "fet")
data_dir.mkdir(parents=True, exist_ok=True)
failures: list[str] = []


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# Two simulations of the same sample agree if they match to reltol (1e-6) or differ by
# less than the simulator's own numerical floor. That floor is measured in step 2b below:
# double-precision rounding inside the solver leaves up to ~1e-14 A of noise on a
# current whose terminal sits at VDD (gate current at Vg=VDD, PMOS body current).
REL_TOL, ABS_TOL = 1e-6, 1e-14


def agree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) <= REL_TOL * np.abs(b) + ABS_TOL


# ---- 1. datasets ---------------------------------------------------------------
param_model = fetsim.write_param_model()
print(f"parametrised model card: {param_model.relative_to(config.ROOT)}")
print(f"ngspice {spice.version()}, {args.jobs or fetsim.n_jobs()} parallel jobs, {args.n} samples per case\n")

datasets: dict[str, pd.DataFrame] = {}
timing: dict[str, float] = {}
for case in cases:
    seed = cfg["part_a"]["seeds"]["sampling"][case]
    t0 = time.time()
    inputs = sampling.draw_fet_inputs(case, args.n, seed)
    currents = fetsim.simulate(case, inputs, jobs=args.jobs)
    df = inputs.copy()
    df[fetsim.CURRENTS] = currents
    df = df[COLUMNS]
    assert list(df.columns) == COLUMNS and len(df) == args.n and np.isfinite(df.values).all()
    out = data_dir / f"{case}.csv"
    df.to_csv(out, index=False)          # default float format round-trips exactly
    timing[case] = time.time() - t0
    datasets[case] = df
    print(f"  {case:9s} {len(df):6d} samples  seed {seed}  {timing[case]:6.1f} s  -> {out.relative_to(config.ROOT)}")

# ---- 1b. every input inside its section 4 range ---------------------------------------
print("\nRange check against section 4:")
a, sw = cfg["assignment"], cfg["sweep"]
nom = sampling.process_nominals()
eps = 1e-12
for case, df in datasets.items():
    t = cfg["cases"][case]["type"]
    lo, hi = a["vdd_nominal"] * (1 + np.array(sw["vdd_rel"]))
    checks = {
        "VDD in nominal +/-10 %": df.vdd.between(lo - eps, hi + eps).all(),
        "0 <= Vd, Vs <= VDD": ((df.vd >= 0) & (df.vd <= df.vdd) & (df.vs >= 0) & (df.vs <= df.vdd)).all(),
        "W in [1, 8] x Lmin": df.w.between(a["lmin"] * sw["w_mult"][0] * (1 - eps),
                                           a["lmin"] * sw["w_mult"][1] * (1 + eps)).all(),
        "T in [-40, 125] C": df.temp.between(*sw["temp_c"]).all(),
        "Vg, Vb as the case table": (np.allclose(df.vg, df.vdd if str(cfg["cases"][case]["vg"]).upper() == "VDD" else 0, rtol=0, atol=0)
                                     and np.allclose(df.vb, df.vdd if str(cfg["cases"][case]["vb"]).upper() == "VDD" else 0, rtol=0, atol=0)),
        "process params in nominal +/-10 %": all(
            ((df[p] / nom[t][p] - 1).between(sw["process_rel"][0] - eps, sw["process_rel"][1] + eps)).all()
            for p in sw["process_params"]),
    }
    bad = [k for k, v in checks.items() if not v]
    print(f"  {case:9s} " + ("all inputs in range" if not bad else "OUT OF RANGE: " + ", ".join(bad)))
    if bad:
        failures.append(f"{case}: inputs out of range ({', '.join(bad)})")

# ---- 2. independent verification ---------------------------------------------------
print(f"\nVerification: {args.verify} random samples per case re-simulated as literal netlists")
lit_diff: dict[str, np.ndarray] = {}
for case, df in datasets.items():
    if args.verify <= 0:
        break
    rng = np.random.default_rng(cfg["part_a"]["seeds"]["sampling"][case] + 10_000)
    rows = rng.choice(len(df), size=min(args.verify, len(df)), replace=False)
    ref = np.array([fetsim.simulate_literal(case, df.iloc[i]) for i in rows])
    got = df.iloc[rows][fetsim.CURRENTS].to_numpy()
    ok = agree(got, ref)
    lit_diff[case] = np.abs(got - ref).max(axis=0)
    rel_idis = np.max(np.abs(got[:, [0, 2]] - ref[:, [0, 2]]) / np.abs(ref[:, [0, 2]]))
    print(f"  {case:9s} {ok.all(axis=1).sum():3d}/{len(rows)} agree   Id/Is worst relative difference {rel_idis:.1e}"
          "   max |difference| id/ig/is/ib: " + "/".join(f"{x:.0e}" for x in lit_diff[case]))
    if not ok.all():
        failures.append(f"{case}: batch and literal simulations disagree")

# ---- 2b. numerical resolution of each current -------------------------------------
# Re-simulate the first samples with T and W nudged by 1e-13 (relative): physically
# nothing changes, so any change in the output is solver rounding noise.
print("\nNumerical resolution: |change| when inputs move by 1e-13 relative (max over samples, A)")
noise_rows = []
for case, df in datasets.items():
    base = df[sampling.INPUT_COLUMNS].iloc[:400].reset_index(drop=True)
    nudged = base.copy()
    nudged["temp"] *= 1 + 1e-13
    nudged["w"] *= 1 + 1e-13
    d = np.abs(fetsim.simulate(case, base, jobs=args.jobs) - fetsim.simulate(case, nudged, jobs=args.jobs))
    mag = df[fetsim.CURRENTS].abs()
    row = {"case": case}
    for k, c in enumerate(fetsim.CURRENTS):
        row[f"{c}_noise_A"] = float(max(d[:, k].max(), lit_diff.get(case, np.zeros(4))[k]))
        row[f"{c}_frac_below_1e-15A"] = float((mag[c] < 1e-15).mean())
        row[f"{c}_exact_zeros"] = int((df[c] == 0).sum())
    noise_rows.append(row)
    print(f"  {case:9s} " + "  ".join(f"{c} {row[f'{c}_noise_A']:.0e}" for c in fetsim.CURRENTS)
          + f"   | ib < 1e-15 A in {100 * row['ib_frac_below_1e-15A']:.0f}% of samples"
          + (f", {row['ib_exact_zeros']} exactly zero" if row["ib_exact_zeros"] else ""))
noise_file = config.path("results", "data_check", "numerical_resolution.csv")
noise_file.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(noise_rows).to_csv(noise_file, index=False)

# ---- 3. process-parameter sensitivity ------------------------------------------------
print("\nProcess sensitivity at VDD=nominal, 27 C, W=2*Lmin (NMOS Vd=VDD Vs=0, PMOS Vs=VDD Vd=0):")
print("  change of |current| from param = -10 % to +10 % of nominal")
a = cfg["assignment"]
names = cfg["sweep"]["process_params"]
nom = sampling.process_nominals()
rows_out = []
for case in cases:
    spec = cfg["cases"][case]
    t = spec["type"]
    vdd = a["vdd_nominal"]
    base = dict(vdd=vdd, temp=27.0, w=2 * a["lmin"],
                vd=vdd if t == "n" else 0.0, vs=0.0 if t == "n" else vdd,
                vg=vdd if str(spec["vg"]).upper() == "VDD" else 0.0,
                vb=vdd if str(spec["vb"]).upper() == "VDD" else 0.0, **nom[t])
    grid = []
    for p in names:
        for f in (0.9, 1.1):
            r = dict(base)
            r[p] = nom[t][p] * f
            grid.append(r)
    cur = np.abs(fetsim.simulate(case, pd.DataFrame(grid)[sampling.INPUT_COLUMNS], jobs=1))
    line = []
    for k, p in enumerate(names):
        lo, hi = cur[2 * k], cur[2 * k + 1]
        ch = 100 * (hi - lo) / lo
        rows_out.append(dict(case=case, param=p, d_id_pct=ch[0], d_ig_pct=ch[1], d_ib_pct=ch[3]))
        line.append(f"{p} Id {ch[0]:+7.1f}% Ig {ch[1]:+7.1f}%")
        if np.all(np.abs(ch[:3]) < 1e-6):
            failures.append(f"{case}: parameter {p} has no effect on the currents")
    print(f"  {case:9s} " + " | ".join(line))
sens_file = config.path("results", "data_check", "process_sensitivity.csv")
sens_file.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows_out).to_csv(sens_file, index=False, float_format="%.4f")

# ---- 4. example netlists (deliverable 2) ---------------------------------------------
print("\nExample netlists (sample 0 of each dataset):")
nl_dir = config.path("netlists", "single_fet")
nl_dir.mkdir(parents=True, exist_ok=True)
for case, df in datasets.items():
    deck = fetsim.example_deck(case, df.iloc[0], 0)
    path = nl_dir / f"{case}.cir"
    path.write_text(deck)
    v = spice.printed(spice.run_deck(deck, nl_dir))
    got = np.array([v[f"i(v{x})"] for x in "dgsb"])
    ok = agree(got, df.iloc[0][fetsim.CURRENTS].to_numpy(dtype=float)).all()
    print(f"  {path.relative_to(config.ROOT)}  reproduces its CSV row: {'yes' if ok else 'NO'}")
    if not ok:
        failures.append(f"{case}: example netlist does not reproduce its dataset row")

# ---- 5. manifest ----------------------------------------------------------------
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                            capture_output=True, text=True).stdout.strip() or None
except OSError:
    commit = None
manifest_path = data_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"cases": {}}
manifest.update({
    "generator": "scripts/01_generate_fet_data.py",
    "git_commit": commit,
    "ngspice": spice.version(),
    "model_file_sha256": sha256(config.model_file()),
    "param_model_sha256": sha256(param_model),
    "current_convention": "ngspice source branch current I(Vx): positive = out of the device terminal",
    "units": {"vdd": "V", "temp": "degC", "w": "m", "vd": "V", "vs": "V", "vg": "V", "vb": "V",
              "vth0": "V", "toxe": "m", "ndep": "cm^-3", "u0": "m^2/(V s)",
              "xj": "m", "id": "A", "ig": "A", "is": "A", "ib": "A"},
})
for case, df in datasets.items():
    kcl = df[fetsim.CURRENTS].sum(axis=1).abs()
    manifest["cases"][case] = {
        "file": f"data/fet/{case}.csv",
        "sha256": sha256(data_dir / f"{case}.csv"),
        "samples": len(df),
        "seed": cfg["part_a"]["seeds"]["sampling"][case],
        "seconds": round(timing[case], 1),
        "max_abs_kcl_residual_A": float(kcl.max()),
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in COLUMNS},
    }
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"\nwrote {manifest_path.relative_to(config.ROOT)}, {sens_file.relative_to(config.ROOT)}, "
      f"{noise_file.relative_to(config.ROOT)}")

# ---- summary ------------------------------------------------------------------
print("\nDataset summary (|current| in A: 1st percentile / median / 99th percentile)")
for case, df in datasets.items():
    parts = [f"{c} {df[c].abs().quantile(.01):.1e}/{df[c].abs().median():.1e}/{df[c].abs().quantile(.99):.1e}"
             for c in fetsim.CURRENTS]
    print(f"  {case:9s} " + "  ".join(parts))

if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nPhase 1 complete: all checks passed.")
