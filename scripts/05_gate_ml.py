#!/usr/bin/env python3
"""Step 05 (Part B, steps 4-5; deliverables 6 and 7): gate leakage from the transistor models.

For every gate, input combination and corner of Phase 4:
  * terminal voltages of each transistor from the SPICE-recorded node voltages
    (inputs, output, internal nodes; section 3 step 4), gate at its logic rail
  * terminal currents from the four transistor models (ON/OFF by gate level and type)
  * gate leakage by Kirchhoff (src/dvd/compose.py), both definitions:
      total = current from every source at VDD (supply + inputs at 1)
      vdd   = current from the VDD supply alone
Also computed, to explain the errors:
  * oracle: the same composition with exact single-transistor SPICE currents instead of
    the models; its error is the composition floor (zero up to rounding for INV, NAND2,
    NOR2; small for AND2/OR2, whose inner inverter gate sits a few mV off the rail)
  * ML-solved internal nodes: x, y, nand_out, nor_out found from Kirchhoff with the
    models (output still from SPICE); leakage recomputed with them (section 5: internal
    node voltages "from both")

Outputs: results/gates/gate_predictions.csv (every operating point),
gate_results.csv/.md (deliverable 6), gate_parity.png (deliverable 7), gate_errors.png.

Options: --models-dir (default models) and --out-dir (default results/gates) run the same
evaluation with other models into another folder, e.g. the improved models:
  python scripts/05_gate_ml.py --models-dir models_v2 --out-dir results/gates_v2 --baseline-dir results/gates
--baseline-dir adds a side-by-side comparison with an earlier run (read only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import compose, config, fetsim, gates  # noqa: E402
from dvd.plotting import plt, save  # noqa: E402

cfg = config.load()
PRIMARY = "total" if cfg["pending_ta"]["leakage_definition"].startswith("total") else "vdd"
SECOND = "vdd" if PRIMARY == "total" else "total"
DEF_LABEL = {"total": "total from every source at VDD", "vdd": "VDD supply only"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--jobs", type=int, default=None, help="parallel gate cases (default: all cores)")
ap.add_argument("--cases", default=None, help="development: only these gates, e.g. inv,and2")
ap.add_argument("--models-dir", default="models", help="folder with the four transistor models")
ap.add_argument("--out-dir", default="results/gates", help="folder for this run's outputs")
ap.add_argument("--baseline-dir", action="append", default=[], help="earlier run(s) to compare with (read only; repeatable)")
args = ap.parse_args()
OUT = config.path(*Path(args.out_dir).parts)
if any(OUT.resolve() == config.path(*Path(b).parts).resolve() for b in args.baseline_dir):
    sys.exit("--out-dir must differ from every --baseline-dir (baselines are never overwritten)")
if args.models_dir != "models" and OUT.resolve() == config.path("results", "gates").resolve():
    sys.exit("other models must write to another --out-dir, not results/gates")
OUT.mkdir(parents=True, exist_ok=True)
from threadpoolctl import threadpool_limits  # noqa: E402
threadpool_limits(1)   # thousands of tiny predictions: BLAS threads only add overhead
failures: list[str] = []

spice = pd.read_csv(config.path("data", "gates", "spice_gates.csv"), dtype={"inputs": str})
corners = pd.read_csv(config.path("data", "gates", "corners.csv")).set_index("corner")
models = compose.load_models(args.models_dir)
print(f"models ({args.models_dir}): " + ", ".join(
    f"{c} {m.kind} {tuple(m.hp['hidden'])}" + (f" x{m.hp['ensemble_members']}" if "ensemble_members" in m.hp else "")
    for c, m in models.items()))
print(f"primary leakage definition: {DEF_LABEL[PRIMARY]} (config pending_ta.leakage_definition)\n")

def run_case(gate: str, vec: tuple[int, ...]) -> tuple[list[dict], str]:
    """Everything for one gate / input combination over all corners (runs in a worker)."""
    import warnings
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore")
    threadpool_limits(1)
    mdl = compose.load_models(args.models_dir)
    p_ml = lambda case, df: mdl[case].predict(df)                       # noqa: E731
    p_sp = lambda case, df: fetsim.simulate(case, df, jobs=1)           # noqa: E731
    fets, pins, nodes = gates.standalone(gate)
    internal = nodes[1:]
    inp = "".join(map(str, vec))
    s = spice[(spice.gate == gate) & (spice.inputs == inp)].sort_values("corner").reset_index(drop=True)
    c = corners.loc[s.corner].reset_index()
    V = {p: s.vdd.to_numpy() * b for p, b in zip(pins, vec)}
    V.update({n: s[f"v_{n}"].to_numpy() for n in nodes})
    high = [p for p, b in zip(pins, vec) if b]
    C = compose.Circuit(fets, c, V, high)
    J_ml, J_or = C.currents(p_ml), C.currents(p_sp)
    res = dict(ml_total=C.leak_total(J_ml), ml_vdd=C.leak_vdd(J_ml),
               oracle_total=C.leak_total(J_or), oracle_vdd=C.leak_vdd(J_or))
    pt_ml, pt_or = C.power_terms(J_ml), C.power_terms(J_or)
    res.update(pbal_total=pt_ml["total"], pbal_oracle_total=pt_or["total"],
               share_channel=pt_or["channel"] / pt_or["total"], share_gate=pt_or["gate"] / pt_or["total"],
               share_body=pt_or["body"] / pt_or["total"])
    V_full = C.solve_all_nodes(p_ml, internal)                          # fully SPICE-free gate
    C3 = compose.Circuit(fets, c, {**V, **V_full}, high)
    C3.levels = C.levels
    J3 = C3.currents(p_ml)
    res.update(fullml_total=C3.leak_total(J3), fullml_vdd=C3.leak_vdd(J3))
    V_ml = C.solve_nodes(p_ml, internal) if internal else {}
    if internal:
        C2 = compose.Circuit(fets, c, {**V, **V_ml}, high)
        C2.levels = C.levels
        J2 = C2.currents(p_ml)
        res.update(mlnodes_total=C2.leak_total(J2), mlnodes_vdd=C2.leak_vdd(J2))
    else:
        res.update(mlnodes_total=res["ml_total"], mlnodes_vdd=res["ml_vdd"])
    out_rows = []
    for i in range(len(s)):
        r = dict(gate=gate, inputs=inp, corner=int(s.corner[i]), vdd=s.vdd[i], temp=s.temp[i],
                 spice_total=s.leak_total[i], spice_vdd=s.leak_vdd[i],
                 **{k: float(v[i]) for k, v in res.items()}, v_out_spice=s.v_out[i],
                 v_out_fullml=float(V_full["out"][i]))
        for n in ("x", "y", "nand_out", "nor_out"):
            r[f"v_{n}_spice"] = s[f"v_{n}"][i]
            r[f"v_{n}_ml"] = float(V_ml[n][i]) if n in V_ml else np.nan
        out_rows.append(r)
    e = np.abs(res[f"ml_{PRIMARY}"] / s[f"leak_{PRIMARY}"].to_numpy() - 1) * 100
    return out_rows, f"  {gate:6s} {inp:2s}  ML median error {np.median(e):5.2f} %  p95 {np.percentile(e, 95):5.2f} %"


from joblib import Parallel, delayed  # noqa: E402
todo = [(g, v) for g in gates.GATE_ORDER if not args.cases or g in args.cases.split(",") for v in gates.vectors(g)]
t_all = time.time()
n_par = min(len(todo), args.jobs or os.cpu_count() or 1)
print(f"{len(todo)} gate/input cases on {n_par} parallel job(s)")
rows = []
for case_rows, line in Parallel(n_jobs=n_par)(delayed(run_case)(g, v) for g, v in todo):
    rows += case_rows
    print(line, flush=True)
pred = pd.DataFrame(rows)
pred.to_csv(OUT / "gate_predictions.csv", index=False)
print(f"\n{len(pred)} operating points in {time.time() - t_all:.0f} s -> {args.out_dir}/gate_predictions.csv")


# ---- bookkeeping check: exact transistor currents must reproduce SPICE where gates sit on the rails
for d in ("total", "vdd"):
    rel = (pred[f"oracle_{d}"] / pred[f"spice_{d}"] - 1).abs()
    worst = rel[pred.gate.isin(["inv", "nand2", "nor2"])].max()
    if worst > 1e-6:
        failures.append(f"composition with exact transistor currents is off by {worst:.1e} ({d}) for INV/NAND2/NOR2")
print("composition check (exact transistor currents, INV/NAND2/NOR2): worst relative difference "
      f"{max((pred[f'oracle_{d}'] / pred[f'spice_{d}'] - 1)[pred.gate.isin(['inv', 'nand2', 'nor2'])].abs().max() for d in ('total', 'vdd')):.1e}")
pb = (pred.pbal_oracle_total / pred.spice_total - 1).abs()[pred.gate.isin(["inv", "nand2", "nor2"])].max()
print(f"independent check (power balance with exact transistor currents, INV/NAND2/NOR2): worst difference {pb:.1e}")
if pb > 1e-6:
    failures.append(f"power-balance check with exact currents is off by {pb:.1e}")
if not np.isfinite(pred.filter(like="ml_").to_numpy()).all() or (pred[f"ml_{PRIMARY}"] <= 0).any():
    failures.append("non-finite or non-positive ML leakage")


# ---- results tables (deliverable 6) ---------------------------------------------------------
def stats(d: str) -> pd.DataFrame:
    out = []
    for (gate, inp), s in pred.groupby(["gate", "inputs"], sort=False):
        sp, ml = s[f"spice_{d}"], s[f"ml_{d}"]
        signed = 100 * (ml / sp - 1)
        err = signed.abs()
        r = dict(gate=gate, inputs=inp, corners=len(s),
                 spice_median_nA=1e9 * sp.median(), ml_median_nA=1e9 * ml.median(),
                 bias_pct=signed.mean(), median_err_pct=err.median(), p95_err_pct=np.percentile(err, 95),
                 mean_err_pct=err.mean(), max_err_pct=err.max(),
                 composition_floor_median_pct=100 * (s[f"oracle_{d}"] / sp - 1).abs().median(),
                 mlnodes_median_err_pct=100 * (s[f"mlnodes_{d}"] / sp - 1).abs().median())
        for n in ("x", "y", "nand_out", "nor_out"):
            if s[f"v_{n}_ml"].notna().any():
                r[f"{n}_ml_vs_spice_max_mV"] = 1e3 * (s[f"v_{n}_ml"] - s[f"v_{n}_spice"]).abs().max()
        out.append(r)
    return pd.DataFrame(out)


tabs = {d: stats(d) for d in (PRIMARY, SECOND)}
for d, t in tabs.items():
    t.to_csv(OUT / f"gate_results_{d}.csv", index=False)
tabs[PRIMARY].to_csv(OUT / "gate_results.csv", index=False)


def overall(d):
    err = 100 * (pred[f"ml_{d}"] / pred[f"spice_{d}"] - 1).abs()
    return err.median(), np.percentile(err, 95), err.mean()


md = []
for d in (PRIMARY, SECOND):
    t = tabs[d]
    med, p95, mean = overall(d)
    md += [f"# Gate results: leakage = {DEF_LABEL[d]}" + (" (primary)" if d == PRIMARY else ""), "",
           f"All {len(pred)} operating points: median error {med:.2f} %, p95 {p95:.2f} %, "
           f"mean {mean:.2f} % (mean accuracy {100 - mean:.2f} %).", "",
           "| Gate | AB | SPICE median (nA) | ML median (nA) | Bias | Median err | p95 err | Mean err | "
           "Floor (exact currents) | ML-solved nodes: median err |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        md.append(f"| {r.gate.upper()} | {r.inputs} | {r.spice_median_nA:.2f} | {r.ml_median_nA:.2f} | "
                  f"{r.bias_pct:+.2f} % | {r.median_err_pct:.2f} % | {r.p95_err_pct:.2f} % | {r.mean_err_pct:.2f} % | "
                  f"{r.composition_floor_median_pct:.3f} % | {r.mlnodes_median_err_pct:.2f} % |")
    md.append("")
node_cols = [c for c in tabs[PRIMARY].columns if c.endswith("_max_mV")]
md += ["# Internal node voltages: ML-solved vs SPICE (worst over 200 corners, mV)", "",
       "| Gate | AB | " + " | ".join(c.replace("_ml_vs_spice_max_mV", "") for c in node_cols) + " |",
       "|---|---|" + "---|" * len(node_cols)]
for _, r in tabs[PRIMARY].iterrows():
    if any(pd.notna(r.get(c)) for c in node_cols):
        md.append(f"| {r.gate.upper()} | {r.inputs} | " + " | ".join(
            f"{r[c]:.2f}" if pd.notna(r[c]) else "–" for c in node_cols) + " |")
md += ["", "# Leakage mechanisms (exact SPICE transistor currents, power balance; median over corners)", "",
       "Share of each gate's total leakage carried by the channel (sub-threshold), gate tunnelling and the "
       "body / junctions.", "", "| Gate | AB | Channel | Gate tunnelling | Body / junction |", "|---|---|---|---|---|"]
for (gate, inp), s_ in pred.groupby(["gate", "inputs"], sort=False):
    md.append(f"| {gate.upper()} | {inp} | {100 * s_.share_channel.median():.1f} % | {100 * s_.share_gate.median():.1f} % | "
              f"{100 * s_.share_body.median():.3f} % |")
md += ["", "# Estimator variants (total leakage): median / p95 / mean error vs SPICE", "",
       "| Gate | AB | Spec method (SPICE nodes, node-set sum) | Power-balance sum | ML-solved internal nodes | "
       "Fully SPICE-free (internal + output nodes from ML) |", "|---|---|---|---|---|---|"]
def _st(a, b):
    e = 100 * (a / b - 1).abs()
    return f"{e.median():.2f} / {np.percentile(e, 95):.2f} / {e.mean():.2f} %"
for (gate, inp), s_ in list(pred.groupby(["gate", "inputs"], sort=False)) + [(("ALL", ""), pred)]:
    md.append(f"| {gate.upper()} | {inp} | {_st(s_.ml_total, s_.spice_total)} | {_st(s_.pbal_total, s_.spice_total)} | "
              f"{_st(s_.mlnodes_total, s_.spice_total)} | {_st(s_.fullml_total, s_.spice_total)} |")
(OUT / "gate_results.md").write_text("\n".join(md) + "\n")

# ---- parity plot (deliverable 7) -------------------------------------------------------------
COL = {"inv": "#4c72b0", "nand2": "#dd8452", "nor2": "#55a868", "and2": "#c44e52", "or2": "#8172b3"}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), layout="constrained")
for ax, d in zip(axes, (PRIMARY, SECOND)):
    for gate in gates.GATE_ORDER:
        s = pred[pred.gate == gate]
        ax.scatter(s[f"spice_{d}"], s[f"ml_{d}"], s=5, alpha=0.6, color=COL[gate], label=gate.upper(), rasterized=True)
    lo = min(pred[f"spice_{d}"].min(), pred[f"ml_{d}"].min()) / 1.5
    hi = max(pred[f"spice_{d}"].max(), pred[f"ml_{d}"].max()) * 1.5
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("SPICE leakage (A)")
    ax.set_ylabel("ML leakage (A)")
    med, p95, mean = overall(d)
    ax.set_title(f"{DEF_LABEL[d]}{' (primary)' if d == PRIMARY else ''}\n"
                 f"median {med:.2f} %, p95 {p95:.2f} %, mean {mean:.2f} %", fontsize=9.5)
    ax.legend(loc="upper left", markerscale=3)
fig.suptitle(f"Gate leakage, ML vs SPICE: 18 gate/input cases x {len(corners)} corners")
save(fig, OUT / "gate_parity")

# ---- error distribution per case -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.4), layout="constrained")
labels, data, floors = [], [], []
for (gate, inp), s in pred.groupby(["gate", "inputs"], sort=False):
    labels.append(f"{gate.upper()}\n{inp}")
    data.append(100 * (s[f"ml_{PRIMARY}"] / s[f"spice_{PRIMARY}"] - 1))
    floors.append(100 * (s[f"oracle_{PRIMARY}"] / s[f"spice_{PRIMARY}"] - 1).abs().median())
pos = np.arange(len(data))
ax.boxplot(data, positions=pos, widths=0.6, flierprops=dict(ms=2, alpha=0.4))
ax.plot(pos, floors, "rD", ms=4, label="composition floor (exact transistor currents), median |error|")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(pos, labels, fontsize=8)
ax.set_ylabel("ML − SPICE (% of SPICE)")
ax.set_title(f"Gate leakage error per case, {DEF_LABEL[PRIMARY]}")
ax.legend(loc="upper left")
save(fig, OUT / "gate_errors")

print()
for d in (PRIMARY, SECOND):
    med, p95, mean = overall(d)
    print(f"{DEF_LABEL[d]:32s}: median {med:.2f} %, p95 {p95:.2f} %, mean {mean:.2f} % "
          f"-> mean accuracy {100 - mean:.2f} %")
if args.baseline_dir:
    key = ["gate", "inputs", "corner"]
    runs = [(b, pd.read_csv(config.path(*Path(b).parts) / "gate_predictions.csv", dtype={"inputs": str}))
            for b in args.baseline_dir] + [(f"{args.out_dir} (this run, {args.models_dir})", pred)]
    lines = ["# Gate leakage: this run vs earlier runs (same 3600 operating points)", "",
             "| Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err | Points > 2 % |",
             "|---|---|---|---|---|---|---|---|---|"]
    print("\nComparison with earlier runs (same 3600 operating points):")
    for d in (PRIMARY, SECOND):
        for name, dfr in runs:
            j = dfr.sort_values(key)
            sgn = 100 * (j[f"ml_{d}"] / j[f"spice_{d}"] - 1)
            e = sgn.abs()
            lines.append(f"| {DEF_LABEL[d]} | {name} | {e.median():.2f} % | {np.percentile(e, 95):.2f} % | {e.mean():.2f} % | "
                         f"{100 - e.mean():.2f} % | {sgn.mean():+.2f} % | {e.max():.1f} % | {(e > 2).sum()} |")
            print(f"  {DEF_LABEL[d]:32s} {name:46s} median {e.median():.2f} %, p95 {np.percentile(e, 95):.2f} %, "
                  f"mean {e.mean():.2f} % (accuracy {100 - e.mean():.2f} %), max {e.max():.1f} %")
    (OUT / "comparison_with_baseline.md").write_text("\n".join(lines) + "\n")
print(f"\nwrote {args.out_dir}/gate_predictions.csv, gate_results.csv/.md (+ _total/_vdd), gate_parity.png, gate_errors.png")
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nPhase 5 complete.")
