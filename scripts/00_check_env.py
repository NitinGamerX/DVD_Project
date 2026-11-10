#!/usr/bin/env python3
"""Step 00: verify the toolchain and project configuration before any data is generated.

Checks
  1. Python version and pinned packages (requirements.txt)
  2. ngspice is installed; its version is recorded for the report's Setup section
  3. The assigned model file is byte-identical to the one we were given
  4. config.yaml matches the rules of DVD_ML_PROJECT.pdf
  5. Process-parameter nominals can be read from the model file (NMOS and PMOS)
  6. One single-transistor netlist per case is written to work/check/ (scratch)
     and simulated: required .options line, own source per terminal,
     source/drain areas present, Kirchhoff sum ~ 0, ON current >> OFF current

Writes results/env/environment.txt. Exits non-zero if anything fails.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import config, modelcard, netlist, spice  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
failures: list[str] = []
report: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{OK if cond else BAD}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(label)


def section(title: str) -> None:
    print(f"\n== {title}")


cfg = config.load()

# 1 ---------------------------------------------------------------------------
section("Python environment")
check(sys.version_info >= (3, 12), "Python >= 3.12", platform.python_version())
check(sys.prefix != sys.base_prefix, "running inside the project venv", sys.prefix)
report += [f"python: {platform.python_version()} ({sys.executable})"]
for line in config.path("requirements.txt").read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if "==" not in line:
        continue
    name, want = line.split("==")
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        have = "missing"
    check(have == want, f"{name}=={want}", f"installed {have}")
    report.append(f"{name}: {have}")

# 2 ---------------------------------------------------------------------------
section("Simulator")
try:
    ver = spice.version()
    major = int("".join(c for c in ver.split(".")[0] if c.isdigit()) or 0)
    check(True, "ngspice found", spice.ngspice_binary())
    check(major >= 42, "ngspice version >= 42 (version used in the project pilot)", ver)
    report.append(f"ngspice: {ver}")
except spice.SpiceError as e:
    check(False, "ngspice found", str(e))

# 3 ---------------------------------------------------------------------------
section("Assigned model file")
mf = config.model_file()
expected = dict(reversed(l.split()) for l in (mf.parent / "SHA256SUMS").read_text().splitlines() if l.strip())
digest = hashlib.sha256(mf.read_bytes()).hexdigest()
check(digest == expected.get(mf.name), f"{mf.name} unmodified (sha256)", digest[:16] + "...")
report.append(f"model file: {cfg['assignment']['model_file']} sha256={digest}")

# 4 ---------------------------------------------------------------------------
section("config.yaml against the project document")
a, s, pa, pb = cfg["assignment"], cfg["sweep"], cfg["part_a"], cfg["part_b"]
check(a["vdd_nominal"] == 1.0, "nominal VDD = 1.0 V (team assignment)")
check(a["test_circuit"] == "cla3", "test circuit = 3-bit CLA (team assignment)")
check(abs(a["lmin"] - 22e-9) < 1e-15, "L = 22 nm (feature size of the 22nm node)")
check(s["vdd_rel"] == [-0.1, 0.1], "VDD = nominal +/-10 %")
check(s["vd_frac"] == [0.0, 1.0] and s["vs_frac"] == [0.0, 1.0], "Vd, Vs uniform in [0, VDD]")
check(s["w_mult"] == [1.0, 8.0], "W uniform in [1, 8] x Lmin")
check(s["temp_c"] == [-40.0, 125.0], "temperature uniform in [-40, 125] C")
check(s["process_params"] == ["vth0", "toxe", "ndep", "u0", "xj"] and s["process_rel"] == [-0.1, 0.1],
      "vth0, toxe, ndep, u0, xj = nominal +/-10 %")
want_cases = {"nmos_on": ("VDD", "0"), "nmos_off": ("0", "0"), "pmos_on": ("0", "VDD"), "pmos_off": ("VDD", "VDD")}
check(all((str(cfg["cases"][c]["vg"]), str(cfg["cases"][c]["vb"])) == v for c, v in want_cases.items()),
      "gate/body voltages per case match the table in section 4")
check(cfg["spice"]["options"] == ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9",
      "mandatory .options line")
check(abs(sum(pa["split"]) - 1) < 1e-12 and pa["split"] in ([0.7, 0.15, 0.15], [0.8, 0.1, 0.1]),
      "train/val/test split is 70/15/15 or 80/10/10", str(pa["split"]))
check(pa["csv_columns"] == "vdd temp w vd vs vg vb vth0 toxe ndep u0 xj id ig is ib".split(),
      "dataset CSV columns exactly as listed in section 6")
check(pb["n_corners"] == 200, "200 corners for gates and test circuit")
n_gate_cases = sum(2 ** len(g["inputs"]) for g in pb["gates"].values())
check(n_gate_cases == 18, "18 gate/input cases", str(n_gate_cases))
sizing = cfg["pending_ta"]["gate_sizing_m"]
check(all(a["lmin"] - 1e-15 <= w <= 8 * a["lmin"] + 1e-15 for ws in sizing.values() for w in ws),
      "default gate widths inside 1..8 x Lmin")

# 5 ---------------------------------------------------------------------------
section("Process-parameter nominals read from the model file")
for t, key in (("NMOS", "nmos_model"), ("PMOS", "pmos_model")):
    try:
        nom = modelcard.nominals(mf, a[key], s["process_params"])
        print(f"  {t}: " + "  ".join(f"{k}={v:.6g}" for k, v in nom.items()))
        report.append(f"{t} nominals: " + ", ".join(f"{k}={v:.6g}" for k, v in nom.items()))
        check(True, f"{t} vth0/toxe/ndep/u0/xj found")
    except Exception as e:  # noqa: BLE001
        check(False, f"{t} nominals", str(e))

# 6 ---------------------------------------------------------------------------
section("Single-transistor smoke test (work/check/)")
outdir = config.path("work", "check")
outdir.mkdir(parents=True, exist_ok=True)
rel_model = Path(os.path.relpath(mf, outdir))
vdd, temp, w = a["vdd_nominal"], 27.0, 2 * a["lmin"]
currents: dict[str, dict[str, float]] = {}
print(f"  operating point: VDD={vdd} V, T={temp} C, W={w*1e9:.0f} nm; "
      "NMOS Vd=VDD Vs=0, PMOS Vs=VDD Vd=0")
print(f"  {'case':9s} {'I(Vd)':>12s} {'I(Vg)':>12s} {'I(Vs)':>12s} {'I(Vb)':>12s}  KCL residual")
for case, spec in cfg["cases"].items():
    vd, vs = (vdd, 0.0) if spec["type"] == "n" else (0.0, vdd)
    deck = netlist.single_fet_deck(case, vdd, w, vd, vs, temp, rel_model)
    (outdir / f"{case}.cir").write_text(deck)
    try:
        vals = spice.printed(spice.run_deck(deck, outdir))
        i = [vals[f"i(v{t})"] for t in "dgsb"]
        resid = abs(sum(i)) / max(map(abs, i))
        currents[case] = dict(zip("dgsb", i))
        print(f"  {case:9s} " + " ".join(f"{x:12.4e}" for x in i) + f"  {resid:.1e}")
        check(resid < 1e-6, f"{case}: Kirchhoff sum of four terminal currents ~ 0")
    except (spice.SpiceError, KeyError) as e:
        check(False, f"{case}: simulation", str(e)[:300])
    text = (outdir / f"{case}.cir").read_text()
    check(cfg["spice"]["options"] in text and all(f"V{t} {t} 0" in text for t in "dgsb")
          and " AS=" in text and " AD=" in text, f"{case}: netlist rules (options, 4 sources, AS/AD)")
for t in ("nmos", "pmos"):
    if f"{t}_on" in currents and f"{t}_off" in currents:
        ratio = abs(currents[f"{t}_on"]["d"]) / abs(currents[f"{t}_off"]["d"])
        check(ratio > 100, f"{t.upper()}: ON drain current >> OFF drain current", f"ratio {ratio:.0f}")

# -----------------------------------------------------------------------------
env_file = config.path("results", "env", "environment.txt")
env_file.parent.mkdir(parents=True, exist_ok=True)
report.insert(0, f"os: {platform.platform()}")
env_file.write_text("\n".join(report) + "\n")
print(f"\nwrote {env_file.relative_to(config.ROOT)}")
if failures:
    print(f"\n{len(failures)} check(s) failed: " + "; ".join(failures))
    sys.exit(1)
print("\nAll checks passed. Environment is ready.")
