#!/usr/bin/env python3
"""Step 03b: improved transistor models - more training data and averaged networks.

Kept completely separate from the original 10,000-sample run:
  * reads only: data/fet (datasets, splits.json), models/ (original models),
    results/model_selection/chosen_models.csv (chosen configuration per case)
  * writes only: data/fet_extra/, models_v2/, results/model_selection_v2/, work/
  * fingerprints every file in the protected folders before starting and checks them
    again at the end; any change is reported as a failure
Steps:
  1. extra samples per case with new seeds, same section 4 distributions, verified like
     Phase 1 (ranges, Kirchhoff, independent literal re-simulation)
  2. training set = original training split + all extra samples; validation and test
     are the original 1,500-sample sets, so old and new models are scored identically
  3. K networks per case with the originally chosen configuration, different random
     starts, trained in parallel; averaged in target space -> models_v2/<case>.joblib
  4. comparison on the original test set: original model / one network on the larger
     data (effect of data alone) / average of K networks (data + averaging)

Usage: python scripts/03b_improve_models.py [--jobs N] [--extra N] [--members K] [--smoke]
  --smoke: tiny settings, written under work/smoke_v2/ (pipeline test only)
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
from dvd import config, ensemble, fetmodel, fetsim, metrics, modelcard, sampling, spice  # noqa: E402

cfg = config.load()
pa, imp = cfg["part_a"], cfg["part_a_improved"]
CASES = list(cfg["cases"])

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--jobs", type=int, default=None, help="parallel jobs (default: all cores)")
ap.add_argument("--extra", type=int, default=imp["extra_samples_per_case"], help="extra samples per case")
ap.add_argument("--members", type=int, default=imp["ensemble_members"], help="networks averaged per case")
ap.add_argument("--smoke", action="store_true", help="tiny pipeline test under work/smoke_v2/")
args = ap.parse_args()
jobs = args.jobs or (os.cpu_count() or 1)

root = config.ROOT
base = root / "work" / "smoke_v2" if args.smoke else root
DATA_X = base / imp["data_dir"]
MODELS2 = base / imp["models_dir"]
RES2 = base / imp["results_dir"]
MEMBERS = root / "work" / ("smoke_members" if args.smoke else "improved_members")
if args.smoke:
    args.extra, args.members = min(args.extra, 1500), min(args.members, 2)

# ---- protection of the original run ----------------------------------------------------------
PROTECTED = ["data/fet", "data/gates", "models", "results/model_selection", "results/gates",
             "results/data_check", "netlists", "model"]
for out in (DATA_X, MODELS2, RES2, MEMBERS):
    for p in PROTECTED:
        prot = (root / p).resolve()
        o = out.resolve()
        if o == prot or prot in o.parents or o in prot.parents:
            sys.exit(f"refusing to run: output folder {out} overlaps protected folder {p}")


def fingerprint() -> dict[str, str]:
    fp = {}
    for p in PROTECTED:
        for f in sorted((root / p).rglob("*")):
            if f.is_file() and f.name != ".spiceinit":
                fp[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return fp


before = fingerprint()
print(f"protected: {len(before)} files in {', '.join(PROTECTED)} fingerprinted (checked again at the end)")
for d in (DATA_X, MODELS2, RES2, MEMBERS):
    d.mkdir(parents=True, exist_ok=True)
failures: list[str] = []


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# =============================================================================
# 1. extra samples (new seeds, new folder)
# =============================================================================
for need, phase in ((config.path(*fetsim.PARAM_MODEL.parts), "Phase 1 (make fet-data)"),
                    (config.path("data", "fet", "splits.json"), "Phase 3 (make train)"),
                    (config.path("results", "model_selection", "chosen_models.csv"), "Phase 3 (make train)")):
    if not need.exists():
        sys.exit(f"{need.relative_to(root)} is missing: it comes from {phase}, which must have run first")
seeds = imp["extra_seeds"]
if set(seeds.values()) & set(pa["seeds"]["sampling"].values()):
    sys.exit("extra seeds must differ from the original sampling seeds")
manifest_x = DATA_X / "manifest.json"
mx = json.loads(manifest_x.read_text()) if manifest_x.exists() else {"cases": {}}
print(f"\n1. Extra samples: {args.extra} per case -> {DATA_X.relative_to(root)}/")
extra: dict[str, pd.DataFrame] = {}
for case in CASES:
    f = DATA_X / f"{case}.csv"
    rec = mx["cases"].get(case)
    if f.exists() and rec and rec["samples"] == args.extra and rec["seed"] == seeds[case] and rec["sha256"] == sha(f):
        extra[case] = pd.read_csv(f)
        print(f"  {case:9s} reusing existing file (same size, seed and checksum)")
        continue
    t0 = time.time()
    inputs = sampling.draw_fet_inputs(case, args.extra, seeds[case])
    df = inputs.copy()
    df[fetsim.CURRENTS] = fetsim.simulate(case, inputs, jobs=jobs)
    df = df[pa["csv_columns"]]
    # checks as in Phase 1
    a, sw = cfg["assignment"], cfg["sweep"]
    nom = sampling.process_nominals()[cfg["cases"][case]["type"]]
    ok = (np.isfinite(df.values).all() and df.vdd.between(0.9 * a["vdd_nominal"] - 1e-12, 1.1 * a["vdd_nominal"] + 1e-12).all()
          and ((df.vd >= 0) & (df.vd <= df.vdd) & (df.vs >= 0) & (df.vs <= df.vdd)).all()
          and df.temp.between(*sw["temp_c"]).all()
          and all((df[p] / nom[p] - 1).between(-0.1 - 1e-12, 0.1 + 1e-12).all() for p in sw["process_params"]))
    I = df[fetsim.CURRENTS].to_numpy()
    kcl = np.abs(I.sum(1)) <= cfg["data_check"]["kirchhoff_reltol"] * np.abs(I).sum(1) + cfg["data_check"]["kirchhoff_abstol_A"]
    rng = np.random.default_rng(seeds[case] + 10_000)
    rows = rng.choice(len(df), size=10, replace=False)
    lit = np.array([fetsim.simulate_literal(case, df.iloc[i]) for i in rows])
    agree = np.all(np.abs(df.iloc[rows][fetsim.CURRENTS].to_numpy() - lit) <= 1e-6 * np.abs(lit) + 1e-14)
    if not (ok and kcl.all() and agree):
        failures.append(f"{case}: extra data failed its checks (ranges {ok}, Kirchhoff {kcl.all()}, literal {agree})")
    df.to_csv(f, index=False)
    mx["cases"][case] = dict(file=str(f.relative_to(root)), samples=len(df), seed=seeds[case], sha256=sha(f),
                             seconds=round(time.time() - t0, 1))
    extra[case] = df
    print(f"  {case:9s} {len(df)} samples, seed {seeds[case]}, {time.time() - t0:.1f} s | ranges ok: {ok}, "
          f"Kirchhoff {kcl.sum()}/{len(df)}, literal re-check {'agrees' if agree else 'DISAGREES'}")
mx.update(generator="scripts/03b_improve_models.py", ngspice=spice.version(),
          note="extra training samples; same distributions as data/fet, different seeds")
manifest_x.write_text(json.dumps(mx, indent=1) + "\n")

# =============================================================================
# 2-3. averaged networks on original training split + extra samples
# =============================================================================
splits = json.loads(config.path("data", "fet", "splits.json").read_text())
chosen = pd.read_csv(config.path("results", "model_selection", "chosen_models.csv")).set_index("case")
_cards = modelcard.parse(config.model_file())
WINT = {"n": _cards[cfg["assignment"]["nmos_model"]]["wint"], "p": _cards[cfg["assignment"]["pmos_model"]]["wint"]}


def training_frame(case: str) -> pd.DataFrame:
    orig = pd.read_csv(config.path("data", "fet", f"{case}.csv"))
    return pd.concat([orig.iloc[splits[case]["train"]], extra[case]], ignore_index=True)


def member_hp(case: str) -> dict:
    hp = json.loads(chosen.loc[case, "hp"])
    if args.smoke:
        hp["max_iter"] = 200
    return hp


def member_key(case: str, i: int) -> str:
    blob = json.dumps(dict(case=case, i=i, hp=member_hp(case), extra=mx["cases"][case]["sha256"],
                           orig=json.loads(config.path("data", "fet", "manifest.json").read_text())["cases"][case]["sha256"],
                           code=sha(config.path("src", "dvd", "fetmodel.py"))), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def fit_member(case: str, i: int, path: str) -> tuple[str, int, float]:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    with threadpool_limits(1):
        t = cfg["cases"][case]["type"]
        m = fetmodel.FetModel(case, str(chosen.loc[case, "model"]), member_hp(case), wint=WINT[t],
                              s_b=float(metrics.score_floors(case)[3]), seed=pa["seeds"]["model"] + i)
        t0 = time.time()
        m.fit(training_frame(case))
        joblib.dump(m, path, compress=3)
    return case, i, time.time() - t0


tasks = []
for case in CASES:
    for i in range(args.members):
        p = MEMBERS / f"{case}_m{i}_{member_key(case, i)}.joblib"
        if not p.exists():
            tasks.append((case, i, str(p)))
n_train = {c: len(splits[c]["train"]) + len(extra[c]) for c in CASES}
print(f"\n2. Training {args.members} networks per case on {n_train[CASES[0]]} samples "
      f"({len(splits[CASES[0]]['train'])} original + {args.extra} extra); "
      f"{len(tasks)} to fit on {min(jobs, max(1, len(tasks)))} parallel jobs, "
      f"{args.members * len(CASES) - len(tasks)} cached")
if tasks:
    cost = {c: (2 if json.loads(chosen.loc[c, "hp"])["hidden"][0] > 64 else 1) for c in CASES}
    tasks.sort(key=lambda t: -cost[t[0]])
    t0 = time.time()
    for case, i, secs in Parallel(n_jobs=min(jobs, len(tasks)))(delayed(fit_member)(*t) for t in tasks):
        print(f"  {case:9s} network {i}: {secs:.0f} s", flush=True)
    print(f"  all fitted in {time.time() - t0:.0f} s")

ensembles = {}
for case in CASES:
    members = [joblib.load(MEMBERS / f"{case}_m{i}_{member_key(case, i)}.joblib") for i in range(args.members)]
    ensembles[case] = ensemble.FetEnsemble(members)
    joblib.dump(ensembles[case], MODELS2 / f"{case}.joblib", compress=3)

# =============================================================================
# 4. comparison on the original validation and test sets
# =============================================================================
print("\n3. Original test set (1,500 samples per case): median / p95 / mean relative error, pooled over Id, Ig, Is, Ib")
originals = {c: joblib.load(config.path("models", f"{c}.joblib")) for c in CASES}
rows = []
for case in CASES:
    df = pd.read_csv(config.path("data", "fet", f"{case}.csv"))
    floors = metrics.score_floors(case)
    variants = {"original (7,000 samples, 1 network)": originals[case],
                f"more data ({n_train[case]:,} samples, 1 network)": ensembles[case].members[0],
                f"more data + average of {args.members} networks": ensembles[case]}
    for split in ("val", "test"):
        part = df.iloc[splits[case][split]]
        T = part[metrics.CURRENTS].to_numpy()
        for name, m in variants.items():
            r = metrics.evaluate(m.predict(part), T, floors)
            rows.append(dict(case=case, set=split, variant=name, **r))
    meta = dict(case=case, model=str(chosen.loc[case, "model"]), hyperparameters=member_hp(case),
                ensemble_members=args.members, member_seeds=[pa["seeds"]["model"] + i for i in range(args.members)],
                train_samples=n_train[case], train_original=len(splits[case]["train"]), train_extra=len(extra[case]),
                extra_data_sha256=mx["cases"][case]["sha256"],
                test={k: v for k, v in rows[-1].items() if k not in ("case", "set", "variant")})
    (MODELS2 / f"{case}.json").write_text(json.dumps(meta, indent=1) + "\n")
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(RES2 / "comparison.csv", index=False)

md = ["# Improved transistor models vs original (original test set, 1,500 samples per case)", "",
      "Relative error, pooled over Id, Ig, Is, Ib (median / p95 / mean), and 'vs largest' = error relative to "
      "the device's largest terminal current (p95).", "",
      "| Case | Variant | Median | p95 | Mean | vs largest p95 | Id mean | Ig mean | Is mean |",
      "|---|---|---|---|---|---|---|---|---|"]
for _, r in cmp_df[cmp_df.set == "test"].iterrows():
    md.append(f"| {r.case} | {r.variant} | {r.median_err_pct:.3f} % | {r.p95_err_pct:.2f} % | {r.mean_err_pct:.2f} % | "
              f"{r.scaled_p95_err_pct:.2f} % | {r.id_mean_err_pct:.2f} % | {r.ig_mean_err_pct:.2f} % | {r.is_mean_err_pct:.2f} % |")
(RES2 / "comparison.md").write_text("\n".join(md) + "\n")
for case in CASES:
    sub = cmp_df[(cmp_df.case == case) & (cmp_df.set == "test")]
    print(f"  {case}")
    for _, r in sub.iterrows():
        print(f"    {r.variant:44s} {r.median_err_pct:6.3f} % / {r.p95_err_pct:6.2f} % / {r.mean_err_pct:6.2f} %"
              f"   (vs largest current p95 {r.scaled_p95_err_pct:.2f} %)")

# figure: test error per variant and case
from dvd.plotting import CASE_LABEL, plt, save  # noqa: E402
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
names = list(dict.fromkeys(cmp_df.variant))
x = np.arange(len(CASES))
for ax, col, title in zip(axes, ("median_err_pct", "p95_err_pct"), ("median", "95th percentile")):
    for k, name in enumerate(names):
        vals = [cmp_df[(cmp_df.case == c) & (cmp_df.set == "test") & (cmp_df.variant == name)][col].iloc[0] for c in CASES]
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=name)
    ax.set_xticks(x, [CASE_LABEL[c] for c in CASES])
    ax.set_yscale("log")
    ax.set_ylabel(f"test relative error, {title} (%)")
axes[0].legend(fontsize=8)
fig.suptitle("Transistor models: original vs more data vs more data + averaged networks (original test set)")
save(fig, RES2 / "comparison")

# ---- protection check -----------------------------------------------------------------------
after = fingerprint()
changed = sorted(k for k in before if after.get(k) != before[k])
added = sorted(set(after) - set(before))
if changed or added:
    failures.append(f"PROTECTED FILES CHANGED: {changed[:5]} added: {added[:5]}")
    print(f"\nPROTECTION CHECK FAILED: {len(changed)} changed, {len(added)} added in protected folders")
else:
    print(f"\nprotection check: all {len(before)} protected files byte-identical to the start of the run")
print(f"wrote {DATA_X.relative_to(root)}/, {MODELS2.relative_to(root)}/, {RES2.relative_to(root)}/comparison.csv/.md/.png")
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nImprovement step complete. Next: python scripts/05_gate_ml.py --models-dir models_v2 "
      "--out-dir results/gates_v2 --baseline-dir results/gates")
