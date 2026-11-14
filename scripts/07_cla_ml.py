#!/usr/bin/env python3
"""Step 07 (Part C, steps 7-8; deliverable 8): hierarchical ML leakage of the 3-bit CLA.

Gate models (Part B). For every gate type, input pattern and corner, the gate's leakage
is predicted from the transistor models exactly as in Phase 5 (recorded gate node
voltages, src/dvd/compose.py), together with the current each input pin draws.
Circuit (Part C). For each adder input combination a logic simulation gives every
cell's input bits; the matching gate predictions are summed over the 39 cells:
  total leakage  = sum of the cells' total leakages. A cell input driven by an internal
                   net draws its gate current through the driving cell from VDD, and the
                   cell's own total already counts it; inputs driven by the adder's
                   primary inputs draw from those sources. Both are part of the total.
  supply only    = that sum minus the current drawn from primary inputs held at 1 (the
                   only part not delivered by VDD). The plain sum of the cells'
                   supply-only leakages is also reported ('naive'): it misses the gate
                   current that internal nets at 1 deliver to the cells they drive.
Like the document says, this treats every cell input as sitting exactly at 0 or VDD and
ignores loading between cells.
Composition floor: the same sums with the exact gate SPICE leakages and pin currents
(Phase 4) instead of the ML predictions, i.e. the error of the hierarchy itself.

Usage: python scripts/07_cla_ml.py [--models-dir models] [--gate-dir results/gates]
         [--out-dir results/test_circuit] [--baseline-dir DIR] [--spice-dir data/test_circuit]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import cla, compose, config, gates, protect  # noqa: E402
from dvd.plotting import plt, save  # noqa: E402

cfg = config.load()
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--models-dir", default="models")
ap.add_argument("--gate-dir", default="results/gates", help="Phase 5 results made with the same models (checked against)")
ap.add_argument("--out-dir", default="results/test_circuit")
ap.add_argument("--baseline-dir", action="append", default=[], help="earlier Part C ML run(s) to compare with (read only; repeatable)")
ap.add_argument("--spice-dir", default="data/test_circuit")
args = ap.parse_args()
PRIMARY = "total" if cfg["pending_ta"]["leakage_definition"].startswith("total") else "vdd"
DEFS = [PRIMARY, "vdd" if PRIMARY == "total" else "total"]
LABEL = {"total": "total from every source at VDD", "vdd": "VDD supply only"}

PROTECTED = ["model", "data", "models", "models_v2", "results/data_check", "results/model_selection",
             "results/model_selection_v2", "results/gates", "results/gates_v2", "netlists"]
PROTECTED += [b for b in args.baseline_dir if b not in PROTECTED]
protect.refuse_overlap([args.out_dir], PROTECTED)
if any(Path(args.out_dir).resolve() == Path(b).resolve() for b in args.baseline_dir):
    sys.exit("--out-dir must differ from every --baseline-dir")
before = protect.fingerprint(PROTECTED)
OUT = config.path(*Path(args.out_dir).parts)
OUT.mkdir(parents=True, exist_ok=True)
failures: list[str] = []

from threadpoolctl import threadpool_limits  # noqa: E402
threadpool_limits(1)

corners = pd.read_csv(config.path("data", "gates", "corners.csv"))
sg = pd.read_csv(config.path("data", "gates", "spice_gates.csv"), dtype={"inputs": str})
spice_cla = pd.read_csv(config.path(*Path(args.spice_dir).parts, "spice_cla3.csv"), dtype={"inputs": str})
models = compose.load_models(args.models_dir)
predict = lambda case, df: models[case].predict(df)  # noqa: E731
print(f"models: {args.models_dir}; gate node voltages: data/gates/spice_gates.csv; CLA SPICE: {args.spice_dir}")

# ---- 1. gate models over all corners ---------------------------------------------------------
t0 = time.time()
lib: dict[tuple[str, str], dict[str, np.ndarray]] = {}
for gate in gates.GATE_ORDER:
    fets, pins, nodes = gates.standalone(gate)
    for vec in gates.vectors(gate):
        inp = "".join(map(str, vec))
        s = sg[(sg.gate == gate) & (sg.inputs == inp)].sort_values("corner").reset_index(drop=True)
        assert list(s.corner) == list(corners.corner)
        V = {p: s.vdd.to_numpy() * b for p, b in zip(pins, vec)}
        V.update({n: s[f"v_{n}"].to_numpy() for n in nodes})
        C = compose.Circuit(fets, corners, V, [p for p, b in zip(pins, vec) if b])
        J = C.currents(predict)
        entry = dict(ml_total=C.leak_total(J), ml_vdd=C.leak_vdd(J),
                     sp_total=s.leak_total.to_numpy(), sp_vdd=s.leak_vdd.to_numpy())
        for p in pins:     # current drawn through each input pin (into the transistor gates on it)
            entry[f"ml_pin_{p}"] = sum(J[:, k, 1] for k, f in enumerate(fets) if f.g == p)
            entry[f"sp_pin_{p}"] = -s[f"i_{p}"].to_numpy()
        lib[(gate, inp)] = entry
print(f"gate models: {len(lib)} gate/input cases x {len(corners)} corners in {time.time() - t0:.0f} s")

# consistency with Phase 5 (same models, same node voltages)
gp = config.path(*Path(args.gate_dir).parts, "gate_predictions.csv")
if gp.exists():
    p5 = pd.read_csv(gp, dtype={"inputs": str})
    worst = 0.0
    for (gate, inp), e in lib.items():
        ref = p5[(p5.gate == gate) & (p5.inputs == inp)].sort_values("corner").ml_total.to_numpy()
        worst = max(worst, float(np.max(np.abs(e["ml_total"] / ref - 1))))
    print(f"consistency with {args.gate_dir}/gate_predictions.csv (same models): worst difference {worst:.1e}")
    if worst > 1e-9:
        failures.append(f"gate leakage differs from {args.gate_dir} by {worst:.1e}: models and gate results do not match")
else:
    print(f"(no {args.gate_dir}/gate_predictions.csv to check against)")

# ---- 2. the adder: sum the cells --------------------------------------------------------------
rows = []
n = len(corners)
for vec in cla.vectors():
    levels, states = cla.evaluate(vec)
    acc = {k: np.zeros(n) for k in ("ml_total", "ml_inputs", "ml_naive_vdd", "sp_total", "sp_inputs")}
    for inst, bits in states:
        e = lib[(inst.kind, "".join(map(str, bits)))]
        acc["ml_total"] += e["ml_total"]
        acc["sp_total"] += e["sp_total"]
        acc["ml_naive_vdd"] += e["ml_vdd"]
        for pin, net, bit in (("a", inst.a, bits[0]), ("b", inst.b, bits[1])):
            if net in cla.INPUTS and bit:          # this pin is driven by a primary input at 1
                acc["ml_inputs"] += e[f"ml_pin_{pin}"]
                acc["sp_inputs"] += e[f"sp_pin_{pin}"]
    inp = "".join(map(str, vec))
    ones = sum(levels[c.y] for c in cla.instances())
    for k in range(n):
        rows.append(dict(inputs=inp, corner=int(corners.corner[k]), internal_ones=ones,
                         ml_total=acc["ml_total"][k], ml_vdd=acc["ml_total"][k] - acc["ml_inputs"][k],
                         ml_naive_vdd=acc["ml_naive_vdd"][k],
                         floor_total=acc["sp_total"][k], floor_vdd=acc["sp_total"][k] - acc["sp_inputs"][k]))
ml = pd.DataFrame(rows)
pred = spice_cla.merge(ml, on=["inputs", "corner"], how="inner")
if len(pred) != len(spice_cla):
    failures.append(f"only {len(pred)} of {len(spice_cla)} SPICE operating points matched")
pred = pred.rename(columns={"leak_total": "spice_total", "leak_vdd": "spice_vdd"})
keep = ["inputs", "A", "B", "cin", "corner", "vdd", "temp", "spice_total", "spice_vdd", "ml_total", "ml_vdd",
        "ml_naive_vdd", "floor_total", "floor_vdd", "max_net_offset_mV", "internal_ones"]
pred[keep].to_csv(OUT / "cla_predictions.csv", index=False)


# ---- 3. metrics, tables (deliverable 8) ------------------------------------------------------------
def err(a, b):
    return 100 * (a / b - 1)


def stats(e: pd.Series) -> dict:
    a = e.abs()
    return dict(median=a.median(), p95=np.percentile(a, 95), mean=a.mean(), max=a.max(), bias=e.mean())


summary = []
for d in DEFS:
    for what, col in (("ML (hierarchical)", f"ml_{d}"), ("composition floor (gate SPICE summed)", f"floor_{d}")) + (
            (("naive: sum of cell supply-only leakages", "ml_naive_vdd"),) if d == "vdd" else ()):
        st = stats(err(pred[col], pred[f"spice_{d}"]))
        summary.append(dict(definition=LABEL[d], estimate=what, **{f"{k}_pct": v for k, v in st.items()}))
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "cla_summary.csv", index=False)

per_vec = []
for (inp, A, B, ci), s in pred.groupby(["inputs", "A", "B", "cin"], sort=False):
    r = dict(inputs=inp, A=A, B=B, cin=ci)
    for d in DEFS:
        e = err(s[f"ml_{d}"], s[f"spice_{d}"])
        r.update({f"{d}_spice_median_nA": 1e9 * s[f"spice_{d}"].median(), f"{d}_ml_median_nA": 1e9 * s[f"ml_{d}"].median(),
                  f"{d}_bias_pct": e.mean(), f"{d}_median_err_pct": e.abs().median(),
                  f"{d}_p95_err_pct": np.percentile(e.abs(), 95), f"{d}_mean_err_pct": e.abs().mean()})
    per_vec.append(r)
per_vec = pd.DataFrame(per_vec)
per_vec.to_csv(OUT / "cla_results_per_input.csv", index=False)

md = [f"# 3-bit CLA: hierarchical ML vs SPICE ({args.models_dir})", "",
      f"{len(pred)} operating points: {pred.inputs.nunique()} input combinations x {pred.corner.nunique()} corners. "
      "Error = (estimate − SPICE) / SPICE.", "",
      "| Leakage | Estimate | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in summary.iterrows():
    md.append(f"| {r.definition} | {r.estimate} | {r.median_pct:.2f} % | {r.p95_pct:.2f} % | {r.mean_pct:.2f} % | "
              f"{100 - r.mean_pct:.2f} % | {r.bias_pct:+.2f} % | {r.max_pct:.2f} % |")
for d in DEFS:
    md += ["", f"## Per input combination: {LABEL[d]}", "",
           "| A | B | cin | SPICE median (nA) | ML median (nA) | Bias | Median err | p95 err | Mean err |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in per_vec.iterrows():
        md.append(f"| {r.A} | {r.B} | {r.cin} | {r[f'{d}_spice_median_nA']:.1f} | {r[f'{d}_ml_median_nA']:.1f} | "
                  f"{r[f'{d}_bias_pct']:+.2f} % | {r[f'{d}_median_err_pct']:.2f} % | {r[f'{d}_p95_err_pct']:.2f} % | "
                  f"{r[f'{d}_mean_err_pct']:.2f} % |")
(OUT / "cla_results.md").write_text("\n".join(md) + "\n")

# ---- analyses (from the teammate's implementation) -------------------------------------------
from scipy.stats import spearmanr  # noqa: E402
pred["temp_tercile"] = pd.qcut(pred.temp, 3, labels=["low", "mid", "high"])
ana = ["", "## Error by temperature (total leakage)", "",
       "Composition = sum of exact gate SPICE leakages vs the full netlist; ML part = hierarchical ML vs that sum.", "",
       "| Temperature third | Range (°C) | ML vs SPICE: median err | ML vs SPICE: bias | Composition bias | ML part: median err |",
       "|---|---|---|---|---|---|"]
for t, s_ in pred.groupby("temp_tercile", observed=True):
    e = err(s_.ml_total, s_.spice_total)
    ana.append(f"| {t} | {s_.temp.min():.0f} – {s_.temp.max():.0f} | {e.abs().median():.2f} % | {e.mean():+.2f} % | "
               f"{err(s_.floor_total, s_.spice_total).mean():+.2f} % | {err(s_.ml_total, s_.floor_total).abs().median():.2f} % |")
pv = pred.groupby("inputs").agg(ones=("internal_ones", "first"),
                                 bias=("ml_total", lambda x: 0)).reset_index()
pv["bias"] = [err(g.ml_total, g.spice_total).mean() for _, g in pred.groupby("inputs")]
pv["floor_bias"] = [err(g.floor_total, g.spice_total).mean() for _, g in pred.groupby("inputs")]
rho, rho_f = spearmanr(pv.ones, pv.bias).statistic, spearmanr(pv.ones, pv.floor_bias).statistic
ana += ["", f"Bias vs number of internal nets at logic 1 (per input combination): Spearman rho = {rho:+.2f} "
            f"(composition floor alone: {rho_f:+.2f}). A negative value for the floor means the adder leaks more than its "
            f"isolated gates when more internal nets sit at 1: those nets drive cell inputs with slightly degraded '1' "
            f"levels, which the gate models (inputs exactly at the rails) do not see.",
        f"ML part alone (hierarchical ML vs exact gate SPICE summed): median {err(pred.ml_total, pred.floor_total).abs().median():.2f} %, "
        f"p95 {np.percentile(err(pred.ml_total, pred.floor_total).abs(), 95):.2f} %."]
md_text = (OUT / "cla_results.md").read_text().split("\n## Per input combination")
(OUT / "cla_results.md").write_text(md_text[0].rstrip() + "\n" + "\n".join(ana) + "\n\n## Per input combination"
                                    + "\n## Per input combination".join(md_text[1:]))
print("\n".join(a for a in ana if a.startswith("|") or a.startswith("Bias") or a.startswith("ML part")))

# parity plot (deliverable 8) and error per input combination
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), layout="constrained")
for ax, d in zip(axes, DEFS):
    ax.scatter(pred[f"spice_{d}"], pred[f"ml_{d}"], s=3, alpha=0.35, color="#4c72b0", rasterized=True, label="ML (hierarchical)")
    lo = min(pred[f"spice_{d}"].min(), pred[f"ml_{d}"].min()) / 1.3
    hi = max(pred[f"spice_{d}"].max(), pred[f"ml_{d}"].max()) * 1.3
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("SPICE leakage, full CLA netlist (A)")
    ax.set_ylabel("ML leakage, sum of gate models (A)")
    st = summary[(summary.definition == LABEL[d]) & (summary.estimate == "ML (hierarchical)")].iloc[0]
    ax.set_title(f"{LABEL[d]}\nmedian {st.median_pct:.2f} %, p95 {st.p95_pct:.2f} %, mean {st.mean_pct:.2f} %", fontsize=9.5)
    ax.legend(loc="upper left", markerscale=3)
fig.suptitle(f"3-bit CLA leakage, hierarchical ML ({args.models_dir}) vs SPICE: "
             f"{pred.inputs.nunique()} input combinations x {pred.corner.nunique()} corners")
save(fig, OUT / "cla_parity")

fig, ax = plt.subplots(figsize=(11.5, 4.2), layout="constrained")
x = np.arange(len(per_vec))
for d, colr in zip(DEFS, ("#1f77b4", "#d62728")):
    e = pred.groupby("inputs", sort=False).apply(lambda s, d=d: err(s[f"ml_{d}"], s[f"spice_{d}"]).abs().median())
    ax.plot(x, e.to_numpy(), "-", color=colr, lw=1.2, label=f"ML, {LABEL[d]}: median |error| over corners")
    f = pred.groupby("inputs", sort=False).apply(lambda s, d=d: err(s[f"floor_{d}"], s[f"spice_{d}"]).abs().median())
    ax.plot(x, f.to_numpy(), ":", color=colr, lw=1.2, label=f"composition floor, {LABEL[d]}")
ax.set_xlabel("input combination index (bits a0 a1 a2 b0 b1 b2 cin)")
ax.set_ylabel("|error| (%)")
ax.set_title("3-bit CLA: median error per input combination")
ax.legend(fontsize=8)
save(fig, OUT / "cla_errors")

print(f"\n{len(pred)} operating points; error vs full-netlist SPICE:")
for _, r in summary.iterrows():
    print(f"  {r.definition:32s} {r.estimate:42s} median {r.median_pct:5.2f} %, p95 {r.p95_pct:5.2f} %, "
          f"mean {r.mean_pct:5.2f} % (accuracy {100 - r.mean_pct:.2f} %), bias {r.bias_pct:+.2f} %")

if args.baseline_dir:
    runs = [(b, pd.read_csv(config.path(*Path(b).parts, "cla_summary.csv"))) for b in args.baseline_dir] + \
           [(f"{args.out_dir} (this run, {args.models_dir})", summary)]
    lines = [f"# 3-bit CLA: this run vs earlier runs", "",
             "| Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err |", "|---|---|---|---|---|---|---|---|"]
    print("\nComparison with earlier runs:")
    for d in DEFS:
        for name, t in runs:
            r = t[(t.definition == LABEL[d]) & (t.estimate == "ML (hierarchical)")].iloc[0]
            lines.append(f"| {LABEL[d]} | {name} | {r.median_pct:.2f} % | {r.p95_pct:.2f} % | {r.mean_pct:.2f} % | "
                         f"{100 - r.mean_pct:.2f} % | {r.bias_pct:+.2f} % | {r.max_pct:.2f} % |")
            print(f"  {LABEL[d]:32s} {name:48s} median {r.median_pct:.2f} %, p95 {r.p95_pct:.2f} %, "
                  f"mean {r.mean_pct:.2f} % (accuracy {100 - r.mean_pct:.2f} %)")
    (OUT / "comparison_with_baseline.md").write_text("\n".join(lines) + "\n")

changed = protect.check(before, PROTECTED)
if changed:
    failures.append(f"protected files changed: {changed[:5]}")
print(f"\nprotection check: {'all ' + str(len(before)) + ' earlier files unchanged' if not changed else 'CHANGED'}")
print(f"wrote {args.out_dir}/cla_predictions.csv, cla_summary.csv, cla_results.md, cla_results_per_input.csv, "
      "cla_parity.png, cla_errors.png" + (", comparison_with_baseline.md" if args.baseline_dir else ""))
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nPart C ML complete.")
