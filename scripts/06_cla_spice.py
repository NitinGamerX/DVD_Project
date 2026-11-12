#!/usr/bin/env python3
"""Step 06 (Part C, step 6): SPICE reference for the test circuit, a 3-bit CLA.

The whole adder (src/dvd/cla.py: 39 cells, 186 transistors) is simulated as ONE
transistor-level netlist, for every input combination (a2a1a0, b2b1b0, cin: 128) at the
same 200 corners as the gates (data/gates/corners.csv). Inputs at exactly 0 or VDD.
Recorded per operating point: every source current, every cell output net, and the
circuit leakage under both definitions:
    leak_total = current from every source at VDD (supply + inputs at 1)
    leak_vdd   = current from the VDD supply alone
Checks: adder outputs and every internal net at the right logic level; independent
literal re-simulation of random points; an example netlist that reproduces its rows.
Earlier results are fingerprinted before and checked after (never modified).

Usage: python scripts/06_cla_spice.py [--jobs N] [--corners N]
  --corners N: first N corners only, written under work/cla_dev/ (development test)
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
from dvd import cla, config, fetsim, gatesim, protect, spice  # noqa: E402
from dvd.plotting import plt, save  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--jobs", type=int, default=None)
ap.add_argument("--corners", type=int, default=None, help="development: first N corners, outputs in work/cla_dev/")
ap.add_argument("--chunk", type=int, default=4, help="corners per ngspice run")
ap.add_argument("--verify", type=int, default=3, help="random operating points re-simulated literally")
args = ap.parse_args()

dev = args.corners is not None
pref = Path("work", "cla_dev") if dev else Path()
DATA, NL, OUT = (pref / "data" / "test_circuit", pref / "netlists" / "test_circuit", pref / "results" / "test_circuit")
PROTECTED = ["model", "data/fet", "data/fet_extra", "data/gates", "models", "models_v2", "results/data_check",
             "results/model_selection", "results/model_selection_v2", "results/gates", "results/gates_v2",
             "netlists/single_fet", "netlists/gates"]
protect.refuse_overlap([str(DATA), str(NL), str(OUT)], PROTECTED)
before = protect.fingerprint(PROTECTED)
for d in (DATA, NL, OUT):
    config.path(*d.parts).mkdir(parents=True, exist_ok=True)
DATA, NL, OUT = (config.path(*d.parts) for d in (DATA, NL, OUT))
failures: list[str] = []
LOGIC_TOL = 0.05

corners = pd.read_csv(config.path("data", "gates", "corners.csv"))
if dev:
    corners = corners.iloc[:args.corners]
fets, pins, nodes, vecs = cla.transistors(), cla.INPUTS, cla.nets(), cla.vectors()
print(f"3-bit CLA: {len(cla.instances())} cells, {len(fets)} transistors; {len(vecs)} input combinations x "
      f"{len(corners)} corners = {len(vecs) * len(corners)} operating points; ngspice {spice.version()}"
      + ("  [DEVELOPMENT SUBSET -> work/cla_dev/]" if dev else ""))

# ---- simulate ---------------------------------------------------------------------------------
t0 = time.time()
R = gatesim.simulate(fets, pins, nodes, corners, vecs, title="cla3", chunk=args.chunk, jobs=args.jobs)
print(f"  simulated in {time.time() - t0:.0f} s")
names = gatesim.record_names(pins, nodes)
col = {n: k for k, n in enumerate(names)}

rows = []
worst_logic = 0.0
for vi, vec in enumerate(vecs):
    level, _ = cla.evaluate(vec)
    A, B, ci = cla.operands(vec)
    for k, c in corners.reset_index(drop=True).iterrows():
        r = R[k, vi]
        vdd = float(c.vdd)
        off = np.array([abs(r[col[f"v({n})"]] - vdd * level[n]) / vdd for n in nodes])
        worst_logic = max(worst_logic, off.max())
        row = dict(inputs="".join(map(str, vec)), A=A, B=B, cin=ci, corner=int(c.corner), vdd=vdd, temp=float(c.temp),
                   leak_total=-r[col["i(vdd)"]] - sum(r[col[f"i(v{p})"]] for p, b in zip(pins, vec) if b),
                   leak_vdd=-r[col["i(vdd)"]], i_vdd=r[col["i(vdd)"]],
                   **{f"i_{p}": r[col[f"i(v{p})"]] for p in pins},
                   **{f"v_{o}": r[col[f"v({o})"]] for o in cla.OUTPUTS},
                   max_net_offset_mV=1e3 * float((off * vdd).max()), worst_net=nodes[int(off.argmax())],
                   nets_off_logic=int((off > LOGIC_TOL).sum()))
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(DATA / "spice_cla3.csv", index=False)

# ---- checks -----------------------------------------------------------------------------------
print(f"\nLogic check: every one of the {len(nodes)} cell outputs (incl. s0-s2, c3) at its rail within 5 % of VDD")
print(f"  worst offset from rail: {100 * worst_logic:.3f} % of VDD; operating points with a net off its level: "
      f"{(df.nets_off_logic > 0).sum()}")
if (df.nets_off_logic > 0).any():
    failures.append(f"{(df.nets_off_logic > 0).sum()} operating points have nets off their logic level")
if (df.leak_total <= 0).any() or (df.leak_vdd <= 0).any():
    failures.append("non-positive leakage")

rng = np.random.default_rng(606)
worst_i = 0.0
ok = True
for _ in range(args.verify):
    k, vi = int(rng.integers(len(corners))), int(rng.integers(len(vecs)))
    lit = gatesim.simulate_literal(fets, pins, nodes, corners.iloc[k], vecs[vi], "cla3")
    d = np.abs(R[k, vi] - lit)
    isv = np.array([n.startswith("v(") for n in names])
    ok &= bool(np.all(d[~isv] <= 1e-6 * np.abs(lit[~isv]) + 1e-14) and np.all(d[isv] <= 1e-9 + 1e-6 * np.abs(lit[isv])))
    worst_i = max(worst_i, float(np.max(d[~isv] / np.maximum(np.abs(lit[~isv]), 1e-14))))
print(f"Literal re-simulation of {args.verify} random points: {'agree' if ok else 'DISAGREE'} "
      f"(worst current difference {worst_i:.1e} relative)")
if not ok:
    failures.append("batch and literal simulations disagree")

# ---- example netlist (deliverable 2) -------------------------------------------------------------
rel_inc = os.path.relpath(config.path(*fetsim.PARAM_MODEL.parts), NL)
title = (f"3-bit carry-lookahead adder, transistor level ({len(cla.instances())} cells, {len(fets)} transistors). "
         f"Inputs {' '.join(pins)}; outputs {' '.join(cla.OUTPUTS)}")
deck = gatesim.example_deck(fets, pins, nodes, corners.iloc[0], vecs, title, rel_inc)
(NL / "cla3.cir").write_text(deck)
out = spice.run_deck(deck, NL)
blocks = out.split("@@inputs ")[1:]
good = len(blocks) == len(vecs)
for vi, blk in enumerate(blocks):
    vals = spice.printed(blk)
    got = np.array([vals[n] for n in names])
    good &= bool(np.all(np.abs(got - R[0, vi]) <= 1e-6 * np.abs(R[0, vi]) + 1e-14))
print(f"Example netlist {(NL / 'cla3.cir').relative_to(config.ROOT)} (corner {int(corners.corner.iloc[0])}, all "
      f"{len(vecs)} input combinations) reproduces its rows: {'yes' if good else 'NO'}")
if not good:
    failures.append("example netlist does not reproduce its rows")

# ---- summary ------------------------------------------------------------------------------------
per_vec = df.groupby("inputs", sort=False).agg(A=("A", "first"), B=("B", "first"), cin=("cin", "first"),
                                               total_median_A=("leak_total", "median"), vdd_median_A=("leak_vdd", "median"))
per_vec.to_csv(OUT / "spice_per_vector.csv")
lines = ["# 3-bit CLA: SPICE reference (Part C, step 6)", "",
         f"{len(cla.instances())} cells, {len(fets)} transistors; {len(vecs)} input combinations x {len(corners)} corners.", "",
         "| Leakage | Median | Min | Max | Lowest-leakage input (median) | Highest-leakage input (median) |", "|---|---|---|---|---|---|"]
for d, lab in (("total", "total from every source at VDD"), ("vdd", "VDD supply only")):
    s = df[f"leak_{d}"]
    pv = per_vec[f"{d}_median_A"]
    lo, hi = per_vec.loc[pv.idxmin()], per_vec.loc[pv.idxmax()]
    lines.append(f"| {lab} | {1e6 * s.median():.3f} µA | {1e6 * s.min():.3f} µA | {1e6 * s.max():.3f} µA | "
                 f"A={int(lo.A)} B={int(lo.B)} cin={int(lo.cin)}: {1e6 * pv.min():.3f} µA | "
                 f"A={int(hi.A)} B={int(hi.B)} cin={int(hi.cin)}: {1e6 * pv.max():.3f} µA |")
lines += ["", f"Worst internal-net offset from its rail: {100 * worst_logic:.3f} % of VDD "
              f"({df.max_net_offset_mV.max():.2f} mV)."]
(OUT / "spice_summary.md").write_text("\n".join(lines) + "\n")

fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
x = np.arange(len(per_vec))
for d, lab, colr in (("total", "total from every source at VDD", "#1f77b4"), ("vdd", "VDD supply only", "#d62728")):
    q = df.groupby("inputs", sort=False)[f"leak_{d}"].quantile([0.05, 0.5, 0.95]).unstack()
    ax.fill_between(x, q[0.05], q[0.95], color=colr, alpha=0.18)
    ax.plot(x, q[0.5], "-", color=colr, lw=1.2, label=f"{lab}: median, band = 5-95 % of corners")
ax.set_yscale("log")
ax.set_xlabel("input combination index (bits a0 a1 a2 b0 b1 b2 cin, a0 most significant in the index)")
ax.set_ylabel("leakage current (A)")
ax.set_title(f"3-bit CLA leakage from SPICE, {len(vecs)} input combinations x {len(corners)} corners")
ax.legend()
save(fig, OUT / "spice_leakage")

h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
(DATA / "manifest.json").write_text(json.dumps(dict(
    generator="scripts/06_cla_spice.py", ngspice=spice.version(), cells=len(cla.instances()), transistors=len(fets),
    inputs=pins, outputs=cla.OUTPUTS, vectors=len(vecs), corners=len(corners),
    corners_sha256=h(config.path("data", "gates", "corners.csv")), spice_cla3_sha256=h(DATA / "spice_cla3.csv"),
    development_subset=dev), indent=1) + "\n")

changed = protect.check(before, PROTECTED)
if changed:
    failures.append(f"protected files changed: {changed[:5]}")
print(f"protection check: {'all ' + str(len(before)) + ' earlier files unchanged' if not changed else 'CHANGED: ' + str(changed[:5])}")
print(f"\nwrote {DATA.relative_to(config.ROOT)}/spice_cla3.csv, manifest.json; {NL.relative_to(config.ROOT)}/cla3.cir; "
      f"{OUT.relative_to(config.ROOT)}/spice_summary.md, spice_per_vector.csv, spice_leakage.png")
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nPart C SPICE reference complete.")
