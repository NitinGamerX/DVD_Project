#!/usr/bin/env python3
"""Step 04 (Part B, step 3): SPICE reference for the gates. Evaluation only, never training.

  1. draw the 200 corners (section 5: one draw of VDD, temperature and process
     parameters; all NMOS share one draw, all PMOS another) -> data/gates/corners.csv,
     reused unchanged by the test circuit in Part C
  2. simulate INV, NAND2, NOR2, AND2 and OR2 as complete transistor-level netlists for
     every input combination (inputs at exactly 0 or VDD) at every corner: 18 cases
     x 200 corners. AND2 and OR2 are simulated as whole gates.
  3. record, per operating point: every source current, the output voltage and the
     internal node voltages (x, y, nand_out, nor_out) -> data/gates/spice_gates.csv
  4. leakage, both definitions (open question D2 in DECISIONS.md):
       leak_total = current delivered by every source held at VDD (supply and inputs
                    at logic 1) = static power / VDD
       leak_vdd   = current delivered by the VDD supply alone
  5. checks: every output and internal logic node at the right rail; independent
     re-simulation of random points through literal netlists; one example netlist per
     gate (deliverable 2) that reproduces its rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import config, fetsim, gates, gatesim, sampling, spice  # noqa: E402
from dvd.plotting import plt, save  # noqa: E402

cfg = config.load()
pb = cfg["part_b"]
DATA = config.path("data", "gates")
OUT = config.path("results", "gates")
NL = config.path("netlists", "gates")
for d in (DATA, OUT, NL):
    d.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--jobs", type=int, default=None, help="parallel ngspice processes (default: all cores)")
ap.add_argument("--verify", type=int, default=6, help="random operating points per gate re-simulated literally")
args = ap.parse_args()
failures: list[str] = []
LOGIC_TOL = 0.05          # a logic node must be within 5 % of VDD of its rail


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- 1. corners ---------------------------------------------------------------------
fetsim.write_param_model()
corners = sampling.draw_corners(pb["n_corners"], pb["corner_seed"])
corners.to_csv(DATA / "corners.csv", index=False)
print(f"{len(corners)} corners (seed {pb['corner_seed']}) -> data/gates/corners.csv; ngspice {spice.version()}")

# ---- 2-4. simulate every gate ----------------------------------------------------------
rows = []
raw: dict[str, tuple] = {}
t_all = time.time()
for gate in gates.GATE_ORDER:
    fets, pins, nodes = gates.standalone(gate)
    vecs = gates.vectors(gate)
    t0 = time.time()
    R = gatesim.simulate(fets, pins, nodes, corners, vecs, title=gate, jobs=args.jobs)
    raw[gate] = (fets, pins, nodes, vecs, R)
    names = gatesim.record_names(pins, nodes)
    for ci, c in corners.iterrows():
        for vi, vec in enumerate(vecs):
            r = dict(zip(names, R[ci, vi]))
            delivered_vdd = -r["i(vdd)"]
            delivered_in = sum(-r[f"i(v{p})"] for p, bit in zip(pins, vec) if bit)
            row = dict(gate=gate, inputs="".join(map(str, vec)), corner=int(c.corner), vdd=c.vdd, temp=c.temp,
                       leak_total=delivered_vdd + delivered_in, leak_vdd=delivered_vdd,
                       i_vdd=r["i(vdd)"], i_a=r["i(va)"], i_b=r.get("i(vb)", np.nan))
            for n in ("out", "x", "y", "nand_out", "nor_out"):
                row[f"v_{n}"] = r.get(f"v({n})", np.nan)
            rows.append(row)
    print(f"  {gate:6s} {len(fets)} transistors, {len(vecs)} input cases x {len(corners)} corners  "
          f"{time.time() - t0:5.1f} s")
df = pd.DataFrame(rows)
df.to_csv(DATA / "spice_gates.csv", index=False)
print(f"  {len(df)} operating points in {time.time() - t_all:.1f} s -> data/gates/spice_gates.csv")

# ---- 5a. logic levels and sanity ---------------------------------------------------------
print("\nLogic-level check (output and nand_out / nor_out within 5 % of VDD of the right rail):")
for gate in gates.GATE_ORDER:
    sub = df[df.gate == gate]
    worst = 0.0
    for vec_str, s in sub.groupby("inputs"):
        vec = tuple(int(ch) for ch in vec_str)
        for node, lvl in gates.expected_levels(gate, vec).items():
            target = s.vdd * lvl
            off = (s[f"v_{node}"] - target).abs() / s.vdd
            worst = max(worst, off.max())
            if (off > LOGIC_TOL).any():
                failures.append(f"{gate} {vec_str}: {node} not at its logic level in {(off > LOGIC_TOL).sum()} corners")
    print(f"  {gate:6s} worst offset from rail: {100 * worst:.3f} % of VDD")
if (df.leak_total <= 0).any():
    failures.append(f"{(df.leak_total <= 0).sum()} operating points with non-positive total leakage")
for n in ("x", "y"):
    v = df[f"v_{n}"].dropna()
    vd = df.loc[v.index, "vdd"]
    if ((v < -0.05) | (v > vd + 0.05)).any():
        failures.append(f"internal node {n} outside [0, VDD]")

# ---- 5b. independent literal re-simulation --------------------------------------------------
print(f"\nVerification: {args.verify} random operating points per gate re-simulated as literal netlists")
rng = np.random.default_rng(pb["corner_seed"] + 1)
for gate, (fets, pins, nodes, vecs, R) in raw.items():
    worst_i, worst_v = 0.0, 0.0
    ok = True
    names = gatesim.record_names(pins, nodes)
    is_v = np.array([n.startswith("v(") for n in names])
    for _ in range(args.verify):
        ci, vi = int(rng.integers(len(corners))), int(rng.integers(len(vecs)))
        lit = gatesim.simulate_literal(fets, pins, nodes, corners.iloc[ci], vecs[vi], gate)
        got = R[ci, vi]
        d = np.abs(got - lit)
        ok &= bool(np.all(d[~is_v] <= 1e-6 * np.abs(lit[~is_v]) + 1e-14) and np.all(d[is_v] <= 1e-9 + 1e-6 * np.abs(lit[is_v])))
        worst_i = max(worst_i, float(np.max(d[~is_v] / np.maximum(np.abs(lit[~is_v]), 1e-14))))
        worst_v = max(worst_v, float(np.max(d[is_v])))
    print(f"  {gate:6s} {'agree' if ok else 'DISAGREE'}  (worst current difference {worst_i:.1e} relative, "
          f"worst voltage difference {worst_v:.1e} V)")
    if not ok:
        failures.append(f"{gate}: batch and literal simulations disagree")

# ---- 5c. example netlists (deliverable 2) --------------------------------------------------
print("\nExample netlists (corner 0, every input combination):")
rel_inc = os.path.relpath(config.path(*fetsim.PARAM_MODEL.parts), NL)
for gate, (fets, pins, nodes, vecs, R) in raw.items():
    title = (f"{gate.upper()} - static CMOS, transistor level. Inputs {', '.join(pins)}; output out"
             + (f"; internal {', '.join(nodes[1:])}" if len(nodes) > 1 else ""))
    deck = gatesim.example_deck(fets, pins, nodes, corners.iloc[0], vecs, title, rel_inc)
    path = NL / f"{gate}.cir"
    path.write_text(deck)
    out = spice.run_deck(deck, NL)
    names = gatesim.record_names(pins, nodes)
    blocks = out.split("@@inputs ")[1:]
    good = len(blocks) == len(vecs)
    for vi, blk in enumerate(blocks):
        vals = spice.printed(blk)
        got = np.array([vals[n] for n in names])
        good &= bool(np.all(np.abs(got - R[0, vi]) <= 1e-6 * np.abs(R[0, vi]) + 1e-14))
    print(f"  {path.relative_to(config.ROOT)}  reproduces its corner-0 rows: {'yes' if good else 'NO'}")
    if not good:
        failures.append(f"{gate}: example netlist does not reproduce its rows")

# ---- summary table + figure -----------------------------------------------------------------
summ = []
for (gate, inp), s in df.groupby(["gate", "inputs"], sort=False):
    vec = tuple(int(ch) for ch in inp)
    r = dict(gate=gate, inputs=inp, out=gates.expected_levels(gate, vec)["out"],
             leak_total_median_A=s.leak_total.median(), leak_total_min_A=s.leak_total.min(),
             leak_total_max_A=s.leak_total.max(), leak_vdd_median_A=s.leak_vdd.median(),
             total_over_vdd_median=(s.leak_total / s.leak_vdd).median(),
             out_rail_offset_max_mV=1e3 * (s.v_out - s.vdd * gates.expected_levels(gate, vec)["out"]).abs().max())
    for n in ("x", "y", "nand_out", "nor_out"):
        if s[f"v_{n}"].notna().any():
            r[f"v_{n}_median_frac_vdd"] = (s[f"v_{n}"] / s.vdd).median()
    summ.append(r)
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "spice_summary.csv", index=False)

md = ["# Gate SPICE reference (Part B, step 3)", "",
      f"{len(corners)} corners per case. Leakage in nA: total = from every source at VDD (supply + inputs at 1); "
      "VDD = supply only.", "",
      "| Gate | AB | Out | Total median | Total min – max | VDD-only median | Total / VDD-only | Internal nodes (median, × VDD) |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in summ.iterrows():
    internal = ", ".join(f"{n} {r[f'v_{n}_median_frac_vdd']:.3f}" for n in ("x", "y", "nand_out", "nor_out")
                         if f"v_{n}_median_frac_vdd" in r and pd.notna(r[f"v_{n}_median_frac_vdd"]))
    md.append(f"| {r.gate.upper()} | {r.inputs} | {r.out} | {1e9 * r.leak_total_median_A:.2f} | "
              f"{1e9 * r.leak_total_min_A:.2f} – {1e9 * r.leak_total_max_A:.1f} | {1e9 * r.leak_vdd_median_A:.2f} | "
              f"{r.total_over_vdd_median:.2f} | {internal or '–'} |")
(OUT / "spice_summary.md").write_text("\n".join(md) + "\n")

fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
labels, data = [], []
for (gate, inp), s in df.groupby(["gate", "inputs"], sort=False):
    labels.append(f"{gate.upper()}\n{inp}")
    data.append(s.leak_total.to_numpy())
pos = np.arange(len(data))
ax.boxplot(data, positions=pos, widths=0.6, showfliers=True, flierprops=dict(ms=2, alpha=0.4))
med_vdd = [df[(df.gate == g) & (df.inputs == i)].leak_vdd.median() for g, i in
           df.groupby(["gate", "inputs"], sort=False).size().index]
ax.plot(pos, med_vdd, "rD", ms=4, label="median, VDD supply only")
ax.set_yscale("log")
ax.set_xticks(pos, labels, fontsize=8)
ax.set_ylabel("leakage current (A)")
ax.set_title(f"Gate leakage from SPICE over {len(corners)} corners (box: total from every source at VDD)")
ax.legend()
save(fig, OUT / "spice_leakage")

# ---- manifest -----------------------------------------------------------------------------
manifest = dict(generator="scripts/04_gate_spice.py", ngspice=spice.version(),
                model_file_sha256=sha256(config.model_file()),
                param_model_sha256=sha256(config.path(*fetsim.PARAM_MODEL.parts)),
                corner_seed=pb["corner_seed"], n_corners=len(corners),
                corners_sha256=sha256(DATA / "corners.csv"), spice_gates_sha256=sha256(DATA / "spice_gates.csv"),
                gate_sizing_m=cfg["pending_ta"]["gate_sizing_m"], operating_points=len(df),
                current_convention="i_* columns: ngspice source branch current (negative = source delivers)",
                leakage_definitions={"leak_total": "current delivered by every source at VDD (supply + inputs at 1)",
                                     "leak_vdd": "current delivered by the VDD supply only"})
(DATA / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(f"\nwrote data/gates/corners.csv, spice_gates.csv, manifest.json; netlists/gates/*.cir; "
      f"results/gates/spice_summary.csv/.md, spice_leakage.png")
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nPhase 4 complete: all checks passed.")
