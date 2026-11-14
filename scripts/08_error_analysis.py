#!/usr/bin/env python3
"""Step 08 (Phase 7 material): error analysis across model versions, for the report.

For each available run (v1 = original 10k models, v2 = + extra uniform data and averaged
networks, v3 = + rail-focused data and extra features):
  * accuracy progression at gate and adder level, both leakage definitions
  * error by corner condition: thirds of temperature, VDD and every process parameter, plus
    the rank correlation (Spearman) of |error| with each corner variable
  * the worst gate points of the latest run traced to individual transistors: the gate
    leakage error split into the contributions of each device's terminal currents
    (ML vs exact single-transistor SPICE at the same terminal voltages)
Reads only; writes results/analysis/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import compose, config, fetsim, gates, protect  # noqa: E402
from dvd.plotting import plt, save  # noqa: E402

RUNS = [("v1: 10k uniform, 1 network", "models", "results/gates", "results/test_circuit"),
        ("v2: 27k uniform, 5 networks", "models_v2", "results/gates_v2", "results/test_circuit_v2"),
        ("v3: + rail data & features, 5 networks", "models_v3", "results/gates_v3", "results/test_circuit_v3")]
OUT = config.path("results", "analysis")
PROTECTED = ["model", "data", "models", "models_v2", "models_v3", "netlists"] + [r[2] for r in RUNS] + [r[3] for r in RUNS]
protect.refuse_overlap(["results/analysis"], PROTECTED)
before = protect.fingerprint(PROTECTED)
OUT.mkdir(parents=True, exist_ok=True)
runs = [r for r in RUNS if config.path(*Path(r[2]).parts, "gate_predictions.csv").exists()]
print("runs found: " + ", ".join(r[0] for r in runs))
corners = pd.read_csv(config.path("data", "gates", "corners.csv"))
CORNER_VARS = [c for c in corners.columns if c != "corner"]
LABEL = {"total": "total from every source at VDD", "vdd": "VDD supply only"}


def err(a, b):
    return 100 * (a / b - 1)


def stats(e):
    a = e.abs()
    return a.median(), np.percentile(a, 95), a.mean(), e.mean(), a.max()


# ---- 1. progression --------------------------------------------------------------------------------
md = ["# Error analysis across model versions", "",
      "Mean accuracy = 100 % − mean relative error. Same 3,600 gate points and 25,600 adder points for every run.", "",
      "## Accuracy progression", "",
      "| Level | Leakage | Run | Median err | p95 err | Mean err | Mean accuracy | Bias | Max err |",
      "|---|---|---|---|---|---|---|---|---|"]
prog = []
data = {}
for name, mdir, gdir, cdir in runs:
    g = pd.read_csv(config.path(*Path(gdir).parts, "gate_predictions.csv"), dtype={"inputs": str})
    cp = config.path(*Path(cdir).parts, "cla_predictions.csv")
    c = pd.read_csv(cp, dtype={"inputs": str}) if cp.exists() else None
    data[name] = (g, c)
    for level, df in (("gates", g), ("3-bit CLA", c)):
        if df is None:
            continue
        for d in ("total", "vdd"):
            m, p95, mean, bias, mx = stats(err(df[f"ml_{d}"], df[f"spice_{d}"]))
            prog.append(dict(level=level, leakage=d, run=name, mean_accuracy=100 - mean))
            md.append(f"| {level} | {LABEL[d]} | {name} | {m:.2f} % | {p95:.2f} % | {mean:.2f} % | {100 - mean:.2f} % | "
                      f"{bias:+.2f} % | {mx:.1f} % |")

# ---- 2. corner conditions ----------------------------------------------------------------------------
md += ["", "## Error by corner condition (total leakage)", "",
       "Median |error| in the lower / middle / upper third of each corner variable, and the rank correlation "
       "(Spearman rho) of |error| with that variable.", ""]
for name, (g, c) in data.items():
    for level, df in (("gates", g), ("3-bit CLA", c)):
        if df is None:
            continue
        j = df.drop(columns=[k for k in ("vdd", "temp") if k in df.columns]).merge(corners, on="corner")
        e = err(j.ml_total, j.spice_total).abs()
        md += [f"### {level}, {name}", "", "| Variable | Lower third | Middle third | Upper third | rho with abs(error) |",
               "|---|---|---|---|---|"]
        for v in CORNER_VARS:
            q = pd.qcut(j[v], 3, labels=False)
            med = [e[q == k].median() for k in range(3)]
            md.append(f"| {v} | {med[0]:.2f} % | {med[1]:.2f} % | {med[2]:.2f} % | {spearmanr(j[v], e).statistic:+.2f} |")
        md.append("")

# ---- 3. worst gate points of the latest run, traced to transistors ------------------------------------------
name, mdir, gdir, _ = runs[-1]
g = data[name][0]
models = compose.load_models(mdir)
predict_ml = lambda case, df: models[case].predict(df)                       # noqa: E731
predict_sp = lambda case, df: fetsim.simulate(case, df, jobs=1)              # noqa: E731
sg = pd.read_csv(config.path("data", "gates", "spice_gates.csv"), dtype={"inputs": str})
cidx = corners.set_index("corner")
g = g.assign(e=err(g.ml_total, g.spice_total))
worst = g.reindex(g.e.abs().sort_values(ascending=False).index).head(5)
md += [f"## Worst gate points of {name}, traced to transistors", "",
       "Each row: one device terminal's share of the gate leakage error, ML vs exact single-transistor SPICE at the "
       "same terminal voltages (only the largest contributions).", ""]
TERM = ("drain", "gate", "source", "body")
for _, w in worst.iterrows():
    fets, pins, nodes = gates.standalone(w.gate)
    vec = tuple(int(ch) for ch in w.inputs)
    s = sg[(sg.gate == w.gate) & (sg.inputs == w.inputs) & (sg.corner == w.corner)].reset_index(drop=True)
    cc = cidx.loc[[int(w.corner)]].reset_index()
    V = {p: s.vdd.to_numpy() * b for p, b in zip(pins, vec)}
    V.update({nd: s[f"v_{nd}"].to_numpy() for nd in nodes})
    C = compose.Circuit(fets, cc, V, [p for p, b in zip(pins, vec) if b])
    H = C.high_set(["vdd"] + C.high_pins)
    Jm, Jo = C.currents(predict_ml), C.currents(predict_sp)
    contrib = []
    for k, f in enumerate(fets):
        state = "ON" if C.is_on(f)[0] else "OFF"
        for t, nd in enumerate((f.d, f.g, f.s, f.b)):
            if H[nd][0]:
                d = (Jm[0, k, t] - Jo[0, k, t]) / s.leak_total[0]
                contrib.append((abs(d), f"{f.name} ({f.type.upper()}MOS {state}) {TERM[t]}: {100 * d:+.2f} % "
                                        f"(ML/exact − 1 = {100 * (Jm[0, k, t] / Jo[0, k, t] - 1):+.1f} %)"))
    c0 = cidx.loc[int(w.corner)]
    md.append(f"- **{w.gate.upper()} {w.inputs}, corner {int(w.corner)}: {w.e:+.2f} %** (T = {c0.temp:.0f} °C, "
              f"VDD = {c0.vdd:.3f} V, NMOS vth0 {100 * (c0.vth0_n / 0.3692 - 1):+.1f} %, NMOS toxe "
              f"{100 * (c0.toxe_n / 6.5e-10 - 1):+.1f} %)")
    for _, txt in sorted(contrib, reverse=True)[:3]:
        md.append(f"  - {txt}")
(OUT / "error_analysis.md").write_text("\n".join(md) + "\n")

# ---- figure: progression -------------------------------------------------------------------------------
pg = pd.DataFrame(prog)
fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
labels = [f"{lv}\n{LABEL[d]}" for lv in ("gates", "3-bit CLA") for d in ("total", "vdd")]
x = np.arange(len(labels))
wbar = 0.8 / max(1, len(runs))
for k, (name, *_r) in enumerate(runs):
    vals = [pg[(pg.level == lv) & (pg.leakage == d) & (pg.run == name)].mean_accuracy.mean()
            for lv in ("gates", "3-bit CLA") for d in ("total", "vdd")]
    bars = ax.bar(x + (k - (len(runs) - 1) / 2) * wbar, vals, wbar, label=name)
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
ax.axhline(99, color="k", ls="--", lw=0.8, label="99 % target")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylim(97.5, 100)
ax.set_ylabel("mean accuracy (%)")
ax.set_title("Mean accuracy vs SPICE across model versions")
ax.legend(fontsize=7.5, loc="lower right")
save(fig, OUT / "accuracy_progression")

changed = protect.check(before, PROTECTED)
print(f"protection check: {'all ' + str(len(before)) + ' files unchanged' if not changed else 'CHANGED ' + str(changed[:3])}")
print("wrote results/analysis/error_analysis.md, accuracy_progression.png")
sys.exit(1 if changed else 0)
