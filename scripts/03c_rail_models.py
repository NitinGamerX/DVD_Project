#!/usr/bin/env python3
"""Step 03c (v3): v2 + rail-focused samples + extra physics features (ideas from the
teammate's implementation, tested in a prototype before adoption; DECISIONS.md D26).

Completely separate from earlier runs:
  * reads: data/fet (+ splits.json), data/fet_extra, models/, models_v2/,
    results/model_selection/chosen_models.csv
  * writes only: data/fet_rail/, models_v3/, results/model_selection_v3/, work/
  * fingerprints every earlier data/model/result folder before and checks after
Steps:
  1. 10,000 rail-focused samples per case (Vd, Vs each with p = 0.5 a log-uniform
     distance 1 uV - 0.3 V from a random rail; everything else as section 4), checked like
     Phase 1 (ranges, Kirchhoff, literal re-simulation), split 70/15/15 (fixed seed)
  2. training set = original uniform training split (7,000) + the 20,000 extra uniform
     samples (v2) + the rail training split (7,000)
  3. 5 networks per case (FetModelV3: conductance target, extra features, chosen
     configuration per case), different random starts, trained in parallel, averaged
  4. v1 / v2 / v3 compared on the original uniform validation and test sets AND on the
     rail-focused validation and test sets

Usage: python scripts/03c_rail_models.py [--jobs N] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import config, ensemble, fetmodel_v3, fetsim, metrics, modelcard, protect, sampling, spice  # noqa: E402

cfg = config.load()
pa, rl = cfg["part_a"], cfg["part_a_rail"]
CASES = list(cfg["cases"])
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--jobs", type=int, default=None)
ap.add_argument("--smoke", action="store_true", help="tiny pipeline test, everything under work/smoke_v3/")
args = ap.parse_args()
jobs = args.jobs or (os.cpu_count() or 1)
root = config.ROOT
pref = Path("work", "smoke_v3") if args.smoke else Path()
DATA_R, MODELS3, RES3 = (pref / rl["data_dir"], pref / rl["models_dir"], pref / rl["results_dir"])
MEMBERS = Path("work", "smoke_v3_members" if args.smoke else "rail_members")
N_RAIL = 1500 if args.smoke else rl["samples_per_case"]
N_MEMBERS = 2 if args.smoke else rl["ensemble_members"]

PROTECTED = ["model", "data/fet", "data/fet_extra", "data/gates", "data/test_circuit", "models", "models_v2",
             "results/data_check", "results/model_selection", "results/model_selection_v2", "results/gates",
             "results/gates_v2", "results/test_circuit", "results/test_circuit_v2", "netlists"]
protect.refuse_overlap([str(DATA_R), str(MODELS3), str(RES3), str(MEMBERS)], PROTECTED)
for need, why in (("data/fet/splits.json", "Phase 3"), ("results/model_selection/chosen_models.csv", "Phase 3"),
                  ("data/fet_extra/manifest.json", "the improvement step (make improve)"),
                  (str(fetsim.PARAM_MODEL), "Phase 1")):
    if not config.path(*Path(need).parts).exists():
        sys.exit(f"{need} is missing: it comes from {why}, which must have run first")
before = protect.fingerprint(PROTECTED)
print(f"protected: {len(before)} earlier files fingerprinted (checked again at the end)")
DATA_R, MODELS3, RES3, MEMBERS = (config.path(*d.parts) for d in (DATA_R, MODELS3, RES3, MEMBERS))
for d in (DATA_R, MODELS3, RES3, MEMBERS):
    d.mkdir(parents=True, exist_ok=True)
failures: list[str] = []
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731

# =============================================================================
# 1. rail-focused samples
# =============================================================================
man_path = DATA_R / "manifest.json"
man = json.loads(man_path.read_text()) if man_path.exists() else {"cases": {}}
print(f"\n1. Rail-focused samples: {N_RAIL} per case -> {DATA_R.relative_to(root)}/")
rail: dict[str, pd.DataFrame] = {}
for case in CASES:
    f = DATA_R / f"{case}.csv"
    rec = man["cases"].get(case)
    if f.exists() and rec and rec["samples"] == N_RAIL and rec["seed"] == rl["seeds"][case] and rec["sha256"] == sha(f):
        rail[case] = pd.read_csv(f)
        print(f"  {case:9s} reusing existing file (same size, seed and checksum)")
        continue
    t0 = time.time()
    inp = sampling.draw_fet_inputs_rail(case, N_RAIL, rl["seeds"][case], tuple(rl["distance_V"]), rl["p_near"])
    df = inp.copy()
    df[fetsim.CURRENTS] = fetsim.simulate(case, inp, jobs=jobs)
    df = df[pa["csv_columns"]]
    I = df[fetsim.CURRENTS].to_numpy()
    in_range = bool(((df.vd > 0) & (df.vd < df.vdd) & (df.vs > 0) & (df.vs < df.vdd)).all() and np.isfinite(I).all())
    kcl = np.abs(I.sum(1)) <= cfg["data_check"]["kirchhoff_reltol"] * np.abs(I).sum(1) + cfg["data_check"]["kirchhoff_abstol_A"]
    rows = np.random.default_rng(rl["seeds"][case] + 10_000).choice(len(df), size=min(10, len(df)), replace=False)
    lit = np.array([fetsim.simulate_literal(case, df.iloc[i]) for i in rows])
    agree = bool(np.all(np.abs(df.iloc[rows][fetsim.CURRENTS].to_numpy() - lit) <= 1e-6 * np.abs(lit) + 1e-14))
    if not (in_range and kcl.all() and agree):
        failures.append(f"{case}: rail data failed checks (range {in_range}, Kirchhoff {kcl.all()}, literal {agree})")
    df.to_csv(f, index=False)
    near = lambda v: np.minimum(v, df.vdd - v) < 2e-3  # noqa: E731
    man["cases"][case] = dict(file=str(f.relative_to(root)), samples=len(df), seed=rl["seeds"][case], sha256=sha(f),
                              both_terminals_within_2mV_of_rail=float((near(df.vd) & near(df.vs)).mean()))
    rail[case] = df
    print(f"  {case:9s} {len(df)} samples, seed {rl['seeds'][case]}, {time.time() - t0:.1f} s | in (0, VDD): {in_range}, "
          f"Kirchhoff {kcl.sum()}/{len(df)}, literal re-check {'agrees' if agree else 'DISAGREES'}, "
          f"both terminals within 2 mV of a rail: {100 * man['cases'][case]['both_terminals_within_2mV_of_rail']:.1f} %")
man.update(generator="scripts/03c_rail_models.py", ngspice=spice.version(),
           note="rail-focused training/evaluation samples (v3); section 4 ranges, Vd/Vs biased toward the rails")
man_path.write_text(json.dumps(man, indent=1) + "\n")
rsplit = {}
for case in CASES:
    idx = np.random.default_rng(rl["split_seed"]).permutation(len(rail[case]))
    a, b = int(rl["split"][0] * len(idx)), int((rl["split"][0] + rl["split"][1]) * len(idx))
    rsplit[case] = {"train": sorted(map(int, idx[:a])), "val": sorted(map(int, idx[a:b])), "test": sorted(map(int, idx[b:]))}
(DATA_R / "splits.json").write_text(json.dumps(rsplit) + "\n")

# =============================================================================
# 2-3. averaged v3 networks
# =============================================================================
splits = json.loads(config.path("data", "fet", "splits.json").read_text())
chosen = pd.read_csv(config.path("results", "model_selection", "chosen_models.csv")).set_index("case")
extra = {c: pd.read_csv(config.path("data", "fet_extra", f"{c}.csv")) for c in CASES}
_cards = modelcard.parse(config.model_file())
WINT = {"n": _cards[cfg["assignment"]["nmos_model"]]["wint"], "p": _cards[cfg["assignment"]["pmos_model"]]["wint"]}


def member_hp(case: str) -> dict:
    hp = json.loads(chosen.loc[case, "hp"])
    hp.update(channel=rl["channel_target"], extra_features=bool(rl["extra_features"]))
    if args.smoke:
        hp["max_iter"] = 200
    return hp


def training_frame(case: str) -> pd.DataFrame:
    orig = pd.read_csv(config.path("data", "fet", f"{case}.csv"))
    return pd.concat([orig.iloc[splits[case]["train"]], extra[case], rail[case].iloc[rsplit[case]["train"]]],
                     ignore_index=True)


def member_key(case: str, i: int) -> str:
    blob = json.dumps(dict(case=case, i=i, hp=member_hp(case), rail=man["cases"][case]["sha256"],
                           extra=sha(config.path("data", "fet_extra", f"{case}.csv")),
                           orig=sha(config.path("data", "fet", f"{case}.csv")),
                           code=[sha(config.path("src", "dvd", f)) for f in ("fetmodel.py", "fetmodel_v3.py")],
                           split=rl["split"], split_seed=rl["split_seed"]), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def fit_member(case: str, i: int, path: str):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    with threadpool_limits(1):
        t = cfg["cases"][case]["type"]
        m = fetmodel_v3.FetModelV3(case, "mlp", member_hp(case), wint=WINT[t],
                                   s_b=float(metrics.score_floors(case)[3]), seed=pa["seeds"]["model"] + i)
        t0 = time.time()
        m.fit(training_frame(case))
        joblib.dump(m, path, compress=3)
    return case, i, time.time() - t0, m.dropped


n_train = {c: len(splits[c]["train"]) + len(extra[c]) + len(rsplit[c]["train"]) for c in CASES}
tasks = [(c, i, str(MEMBERS / f"{c}_m{i}_{member_key(c, i)}.joblib")) for c in CASES for i in range(N_MEMBERS)]
todo = [t for t in tasks if not Path(t[2]).exists()]
c0 = CASES[0]
print(f"\n2. Training {N_MEMBERS} networks per case on {n_train[c0]:,} samples ({len(splits[c0]['train']):,} uniform + "
      f"{len(extra[c0]):,} extra uniform + {len(rsplit[c0]['train']):,} rail); {len(todo)} to fit on "
      f"{min(jobs, max(1, len(todo)))} parallel jobs, {len(tasks) - len(todo)} cached")
if todo:
    todo.sort(key=lambda t: -(2 if json.loads(chosen.loc[t[0], "hp"])["hidden"][0] > 64 else 1))
    t0 = time.time()
    for case, i, secs, dropped in Parallel(n_jobs=min(jobs, len(todo)))(delayed(fit_member)(*t) for t in todo):
        print(f"  {case:9s} network {i}: {secs:.0f} s (rows left out per target, where undefined: {dropped})", flush=True)
    print(f"  all fitted in {time.time() - t0:.0f} s")

v3 = {}
for case in CASES:
    members = [joblib.load(MEMBERS / f"{case}_m{i}_{member_key(case, i)}.joblib") for i in range(N_MEMBERS)]
    v3[case] = ensemble.FetEnsemble(members)
    joblib.dump(v3[case], MODELS3 / f"{case}.joblib", compress=3)

# =============================================================================
# 4. v1 / v2 / v3 on uniform and rail-focused validation and test sets
# =============================================================================
print("\n3. Comparison (pooled relative error over Id, Ig, Is, Ib: median / p95; 'vs largest' = p95 of error "
      "relative to the device's largest current)")
variants = {"v1 (10k uniform, 1 network)": "models", "v2 (27k uniform, 5 networks)": "models_v2"}
loaded = {name: {c: joblib.load(config.path(d, f"{c}.joblib")) for c in CASES}
          for name, d in variants.items() if config.path(d, f"{CASES[0]}.joblib").exists()}
loaded[f"v3, 1 network"] = {c: v3[c].members[0] for c in CASES}
loaded[f"v3 ({n_train[c0] // 1000}k incl. rail, {N_MEMBERS} networks)"] = v3
rows = []
for case in CASES:
    uni = pd.read_csv(config.path("data", "fet", f"{case}.csv"))
    floors = metrics.score_floors(case)
    sets = {"uniform val": uni.iloc[splits[case]["val"]], "uniform test": uni.iloc[splits[case]["test"]],
            "rail val": rail[case].iloc[rsplit[case]["val"]], "rail test": rail[case].iloc[rsplit[case]["test"]]}
    for sname, part in sets.items():
        T = part[metrics.CURRENTS].to_numpy()
        for vname, ms in loaded.items():
            rows.append(dict(case=case, set=sname, variant=vname, **metrics.evaluate(ms[case].predict(part), T, floors)))
    meta = dict(case=case, model="mlp (FetModelV3)", hyperparameters=member_hp(case), ensemble_members=N_MEMBERS,
                member_seeds=[pa["seeds"]["model"] + i for i in range(N_MEMBERS)], train_samples=n_train[case],
                train_parts={"uniform": len(splits[case]["train"]), "extra_uniform": len(extra[case]),
                             "rail": len(rsplit[case]["train"])},
                rail_data_sha256=man["cases"][case]["sha256"],
                test={s: {k: v for k, v in r.items() if k not in ("case", "set", "variant")}
                      for r in rows if r["case"] == case and r["variant"].startswith("v3 (") for s in [r["set"]]})
    (MODELS3 / f"{case}.json").write_text(json.dumps(meta, indent=1) + "\n")
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(RES3 / "comparison.csv", index=False)
md = ["# Transistor models v1 / v2 / v3: uniform and rail-focused validation and test sets", "",
      "Pooled relative error over Id, Ig, Is, Ib. 'vs largest p95' = 95th percentile of the error relative to "
      "the device's largest terminal current (what matters when currents are summed into gate leakage).", "",
      "| Case | Set | Variant | Median | p95 | Mean | vs largest p95 | Ig p95 |", "|---|---|---|---|---|---|---|---|"]
for _, r in cmp_df[cmp_df.set.str.endswith("test")].iterrows():
    md.append(f"| {r.case} | {r.set} | {r.variant} | {r.median_err_pct:.3f} % | {r.p95_err_pct:.2f} % | {r.mean_err_pct:.2f} % | "
              f"{r.scaled_p95_err_pct:.2f} % | {r.ig_p95_err_pct:.2f} % |")
(RES3 / "comparison.md").write_text("\n".join(md) + "\n")
for case in CASES:
    print(f"  {case}")
    for sname in ("uniform test", "rail test"):
        for _, r in cmp_df[(cmp_df.case == case) & (cmp_df.set == sname)].iterrows():
            print(f"    {sname:12s} {r.variant:32s} {r.median_err_pct:6.3f} % / {r.p95_err_pct:6.2f} %   "
                  f"vs largest p95 {r.scaled_p95_err_pct:5.2f} %   Ig p95 {r.ig_p95_err_pct:6.2f} %")

from dvd.plotting import CASE_LABEL, plt, save  # noqa: E402
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
names = list(dict.fromkeys(cmp_df.variant))
x = np.arange(len(CASES))
wbar = 0.8 / len(names)
for ax, sname in zip(axes, ("uniform test", "rail test")):
    for k, name in enumerate(names):
        vals = [cmp_df[(cmp_df.case == c) & (cmp_df.set == sname) & (cmp_df.variant == name)].scaled_p95_err_pct.iloc[0] for c in CASES]
        ax.bar(x + (k - (len(names) - 1) / 2) * wbar, vals, wbar, label=name)
    ax.set_xticks(x, [CASE_LABEL[c] for c in CASES])
    ax.set_ylabel("p95 error relative to the largest terminal current (%)")
    ax.set_title(sname)
axes[0].legend(fontsize=7.5)
fig.suptitle("Transistor models v1 / v2 / v3 (the rail set mimics the operating points inside gates)")
save(fig, RES3 / "comparison")

changed = protect.check(before, PROTECTED)
if changed:
    failures.append(f"PROTECTED FILES CHANGED: {changed[:5]}")
print(f"\nprotection check: {'all ' + str(len(before)) + ' earlier files byte-identical' if not changed else 'FAILED'}")
print(f"wrote {DATA_R.relative_to(root)}/, {MODELS3.relative_to(root)}/, {RES3.relative_to(root)}/comparison.csv/.md/.png")
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nv3 models complete. Next: make gates-ml-v3, make cla-ml-v3")
