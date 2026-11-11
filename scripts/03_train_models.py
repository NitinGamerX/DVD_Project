#!/usr/bin/env python3
"""Step 03 (Part A, step 2; deliverables 4 and 5): transistor models.

For each case:
  1. split the dataset 70/15/15 (train/val/test) with a fixed seed
  2. fit every grid point of four model types (config.yaml -> part_a.model_grid) on the
     training set and score it on the validation set
  3. the best configuration of each type -> one row of the comparison table (deliverable 4)
  4. the best type -> the chosen model; refit (deterministic, checked), save to
     models/<case>.joblib, score once on the test set (deliverable 5)
  5. learning curve of the chosen configuration on growing training subsets

The test set is touched only in step 4. Every fit is cached in
results/model_selection/runs/, so an interrupted run resumes where it stopped and a
re-run with unchanged data and code refits nothing.

Runs on the CPU: one fit per core in parallel (scikit-learn has no GPU support, and at
this data size a GPU would not be faster).

Usage: python scripts/03_train_models.py [--jobs N] [--cases c1,c2] [--quick] [--no-learning-curve]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import config, fetmodel, metrics, modelcard  # noqa: E402

cfg = config.load()
pa = cfg["part_a"]
CASES = list(cfg["cases"])
OUT = config.path("results", "model_selection")
RUNS = OUT / "runs"
MODELS = config.path("models")
SEL = pa["selection_metric"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--jobs", type=int, default=None, help="parallel fits (default: all cores)")
ap.add_argument("--cases", default=",".join(CASES))
ap.add_argument("--quick", action="store_true", help="smallest grid point of each type only (pipeline test)")
ap.add_argument("--no-learning-curve", action="store_true")
args = ap.parse_args()
cases = [c.strip() for c in args.cases.split(",")]
jobs = args.jobs or (os.cpu_count() or 1)
for d in (OUT, RUNS, MODELS):
    d.mkdir(parents=True, exist_ok=True)


# ---- data, splits, cache keys -----------------------------------------------------
def load(case: str) -> pd.DataFrame:
    return pd.read_csv(config.path("data", "fet", f"{case}.csv"))


def make_splits() -> dict:
    """70/15/15 (or as configured) per case, fixed seed. Saved for inspection."""
    f_tr, f_va, f_te = pa["split"]
    out = {}
    for case in CASES:
        idx = np.arange(len(load(case)))
        tr, rest = train_test_split(idx, train_size=f_tr, random_state=pa["seeds"]["split"])
        va, te = train_test_split(rest, train_size=f_va / (f_va + f_te), random_state=pa["seeds"]["split"])
        out[case] = {"train": sorted(map(int, tr)), "val": sorted(map(int, va)), "test": sorted(map(int, te))}
    return out


SPLITS = make_splits()
(config.path("data", "fet", "splits.json")).write_text(json.dumps(SPLITS) + "\n")

_manifest = json.loads(config.path("data", "fet", "manifest.json").read_text())
_code = hashlib.sha256(b"".join(config.path("src", "dvd", f).read_bytes()
                                for f in ("fetmodel.py", "metrics.py"))).hexdigest()


def cache_key(case, kind, hp, n_train) -> str:
    blob = json.dumps(dict(case=case, kind=kind, hp=hp, n_train=n_train, seed=pa["seeds"]["model"],
                           data=_manifest["cases"][case]["sha256"], split=pa["split"],
                           split_seed=pa["seeds"]["split"], floor=pa["score_floor_factor"],
                           code=_code, sklearn=sklearn.__version__), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def grid(kind: str) -> list[dict]:
    g = pa["model_grid"][kind]
    keys = list(g)
    pts = [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]
    return pts[:1] if args.quick else pts


_cards = modelcard.parse(config.model_file())
WINT = {"n": _cards[cfg["assignment"]["nmos_model"]]["wint"], "p": _cards[cfg["assignment"]["pmos_model"]]["wint"]}


def new_model(case: str, kind: str, hp: dict) -> fetmodel.FetModel:
    t = cfg["cases"][case]["type"]
    return fetmodel.FetModel(case, kind, hp, wint=WINT[t], s_b=float(metrics.score_floors(case)[3]),
                             seed=pa["seeds"]["model"])


def train_index(case: str, n_train: int | None) -> np.ndarray:
    tr = np.array(SPLITS[case]["train"])
    if n_train is None or n_train >= len(tr):
        return tr
    rng = np.random.default_rng(pa["seeds"]["split"] + 1)      # nested subsets: same order for every size
    return tr[rng.permutation(len(tr))[:n_train]]


# ---- one fit (runs in a worker process) ---------------------------------------------
def fit_one(case: str, kind: str, hp: dict, n_train: int | None, save_to: str | None) -> dict:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # L-BFGS iteration cap: expected
    with threadpool_limits(1):                  # one thread per fit; parallelism comes from the fits
        df = load(case)
        tr = train_index(case, n_train)
        va = np.array(SPLITS[case]["val"])
        t0 = time.time()
        model = new_model(case, kind, hp).fit(df.iloc[tr])
        secs = time.time() - t0
        floors = metrics.score_floors(case)
        val = metrics.evaluate(model.predict(df.iloc[va]), df.iloc[va][metrics.CURRENTS].to_numpy(), floors)
        if save_to:
            joblib.dump(model, save_to, compress=3)
    return dict(case=case, kind=kind, hp=hp, n_train=int(len(tr)), fit_seconds=round(secs, 1), val=val)


def run_all(tasks: list[tuple]) -> list[dict]:
    """tasks: (case, kind, hp, n_train, save_to). Cached unless save_to is set."""
    todo, results = [], {}
    for t in tasks:
        key = cache_key(*t[:4])
        f = RUNS / f"{t[0]}__{t[1]}__{key}.json"
        if f.exists() and t[4] is None:
            results[key] = json.loads(f.read_text())
        else:
            todo.append((key, f, t))
    if todo:
        cost = {"mlp": 3, "random_forest": 2, "grad_boost": 1, "poly_ridge": 0}
        todo.sort(key=lambda x: -cost[x[2][1]])                 # long fits first
        print(f"  fitting {len(todo)} configuration(s) on {min(jobs, len(todo))} parallel job(s) "
              f"({len(tasks) - len(todo)} cached)", flush=True)
        t0 = time.time()
        outs = Parallel(n_jobs=min(jobs, len(todo)), verbose=0)(delayed(fit_one)(*t) for _, _, t in todo)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
        for (key, f, t), r in zip(todo, outs):
            r["key"] = key
            f.write_text(json.dumps(r, indent=1) + "\n")
            results[key] = r
    return [results[cache_key(*t[:4])] for t in tasks]


# =============================================================================
# 1-3. search and comparison table (validation set only)
# =============================================================================
print(f"Model search: {', '.join(fetmodel.MODEL_TYPES)} on {', '.join(cases)}; split {pa['split']}; "
      f"selection by validation {SEL}")
tasks = [(c, k, hp, None, None) for c in cases for k in fetmodel.MODEL_TYPES for hp in grid(k)]
records = run_all(tasks)

all_rows = [dict(case=r["case"], model=r["kind"], hp=json.dumps(r["hp"], sort_keys=True),
                 fit_seconds=r["fit_seconds"], **{f"val_{k}": v for k, v in r["val"].items()}) for r in records]
all_df = pd.DataFrame(all_rows)
all_df.to_csv(OUT / "all_configs.csv", index=False)

best = (all_df.sort_values(["case", "model", f"val_{SEL}", "val_median_err_pct"])
        .groupby(["case", "model"], sort=False).head(1))
cmp_cols = ["case", "model", "hp", "val_r2_log", "val_median_err_pct", "val_p95_err_pct", "val_mean_err_pct",
            "val_scaled_p95_err_pct"] + [f"val_{c}_{m}" for c in metrics.CURRENTS
                                         for m in ("median_err_pct", "p95_err_pct", "r2_log")] + ["fit_seconds"]
comparison = best[cmp_cols].copy()
comparison["case"] = pd.Categorical(comparison["case"], CASES)
comparison["model"] = pd.Categorical(comparison["model"], fetmodel.MODEL_TYPES)
comparison = comparison.sort_values(["case", "model"]).reset_index(drop=True)
comparison.to_csv(OUT / "model_comparison.csv", index=False)

chosen = (comparison.sort_values(["case", f"val_{SEL}", "val_median_err_pct"]).groupby("case", observed=True)
          .head(1).set_index("case"))

print("\nModel comparison on the validation set (deliverable 4): R2 of log10|I| / median / p95 relative error")
print(f"  {'case':9s} " + "".join(f"{fetmodel.MODEL_LABEL[k]:>30s}" for k in fetmodel.MODEL_TYPES))
for case in cases:
    cells = []
    for k in fetmodel.MODEL_TYPES:
        r = comparison[(comparison.case == case) & (comparison.model == k)].iloc[0]
        mark = "*" if chosen.loc[case, "model"] == k else " "
        cells.append(f"{r.val_r2_log:8.4f} {r.val_median_err_pct:7.2f}% {r.val_p95_err_pct:8.2f}%{mark}")
    print(f"  {case:9s} " + "".join(f"{c:>30s}" for c in cells))
print("  (* = chosen: lowest validation p95)")

# =============================================================================
# 4-5. chosen models (refit + save + test) and learning curve
# =============================================================================
print("\nChosen models: refit, save, learning curve")
final_tasks, lc_tasks = [], []
for case in cases:
    k = str(chosen.loc[case, "model"])
    hp = json.loads(chosen.loc[case, "hp"])
    final_tasks.append((case, k, hp, None, str(MODELS / f"{case}.joblib")))
    if not args.no_learning_curve:
        lc_tasks += [(case, k, hp, n, None) for n in pa["learning_curve_sizes"]]
outs = run_all(final_tasks + lc_tasks)
final_out, lc_out = outs[:len(final_tasks)], outs[len(final_tasks):]

failures = []
chosen_rows = []
for case, r in zip(cases, final_out):
    rec = chosen.loc[case]
    if abs(r["val"][SEL] - rec[f"val_{SEL}"]) > 1e-9 * max(1.0, abs(rec[f"val_{SEL}"])):
        failures.append(f"{case}: refit does not reproduce the searched model")
    model: fetmodel.FetModel = joblib.load(MODELS / f"{case}.joblib")
    df = load(case)
    te = np.array(SPLITS[case]["test"])
    floors = metrics.score_floors(case)
    test = metrics.evaluate(model.predict(df.iloc[te]), df.iloc[te][metrics.CURRENTS].to_numpy(), floors)
    meta = dict(case=case, model=r["kind"], label=fetmodel.MODEL_LABEL[r["kind"]], hyperparameters=r["hp"],
                features=fetmodel.FEATURES, targets=fetmodel.TARGETS, train_samples=r["n_train"],
                fit_seconds=r["fit_seconds"], validation=r["val"], test=test,
                score_floors_A=dict(zip(metrics.CURRENTS, map(float, floors))),
                dataset_sha256=_manifest["cases"][case]["sha256"], sklearn=sklearn.__version__,
                seed=pa["seeds"]["model"], split=pa["split"], split_seed=pa["seeds"]["split"])
    (MODELS / f"{case}.json").write_text(json.dumps(meta, indent=1) + "\n")
    chosen_rows.append(dict(case=case, model=r["kind"], hp=json.dumps(r["hp"], sort_keys=True),
                            **{f"test_{k}": v for k, v in test.items()}))
chosen_df = pd.DataFrame(chosen_rows)
chosen_df.to_csv(OUT / "chosen_models.csv", index=False)

print("\nChosen models on the test set (deliverable 5): relative error median / p95")
for _, r in chosen_df.iterrows():
    parts = "  ".join(f"{c} {r[f'test_{c}_median_err_pct']:.2f}/{r[f'test_{c}_p95_err_pct']:.1f}%"
                      for c in metrics.CURRENTS)
    print(f"  {r.case:9s} {fetmodel.MODEL_LABEL[r.model]:22s} all {r.test_median_err_pct:.3f}% / "
          f"{r.test_p95_err_pct:.2f}%  | {parts} | vs largest current p95 {r.test_scaled_p95_err_pct:.2f}%")

lc_df = pd.DataFrame([dict(case=r["case"], model=r["kind"], n_train=r["n_train"],
                           **{f"val_{k}": v for k, v in r["val"].items()}) for r in lc_out + final_out])
if not args.no_learning_curve:
    lc_df.sort_values(["case", "n_train"]).to_csv(OUT / "learning_curve.csv", index=False)

# =============================================================================
# markdown tables + plots
# =============================================================================
from dvd.plotting import CASE_LABEL, plt, save  # noqa: E402

md = ["# Model comparison (validation set, deliverable 4)", "",
      "R² of log10|I|, pooled median / 95th-percentile relative error over Id, Ig, Is, Ib.",
      "Best configuration of each type; **bold** = chosen (lowest validation p95).", "",
      "| Case | " + " | ".join(fetmodel.MODEL_LABEL[k] for k in fetmodel.MODEL_TYPES) + " |",
      "|---|" + "---|" * len(fetmodel.MODEL_TYPES)]
for case in cases:
    cells = []
    for k in fetmodel.MODEL_TYPES:
        r = comparison[(comparison.case == case) & (comparison.model == k)].iloc[0]
        txt = f"{r.val_r2_log:.4f} / {r.val_median_err_pct:.2f}% / {r.val_p95_err_pct:.1f}%"
        cells.append(f"**{txt}**" if chosen.loc[case, "model"] == k else txt)
    md.append(f"| {CASE_LABEL[case]} | " + " | ".join(cells) + " |")
md += ["", "# Chosen models (test set, deliverable 5)", "",
       "| Case | Model | Median | p95 | Id med/p95 | Ig med/p95 | Is med/p95 | Ib med/p95 |", "|---|---|---|---|---|---|---|---|"]
for _, r in chosen_df.iterrows():
    md.append(f"| {CASE_LABEL[r.case]} | {fetmodel.MODEL_LABEL[r.model]} | {r.test_median_err_pct:.3f}% | "
              f"{r.test_p95_err_pct:.2f}% | " + " | ".join(
                  f"{r[f'test_{c}_median_err_pct']:.2f}% / {r[f'test_{c}_p95_err_pct']:.1f}%" for c in metrics.CURRENTS) + " |")
(OUT / "tables.md").write_text("\n".join(md) + "\n")

# comparison bar chart
fig, axes = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.8), sharey=True, layout="constrained")
axes = np.atleast_1d(axes)
xk = np.arange(len(fetmodel.MODEL_TYPES))
for ax, case in zip(axes, cases):
    sub = comparison[comparison.case == case].set_index("model").loc[fetmodel.MODEL_TYPES]
    ax.bar(xk - 0.2, sub.val_median_err_pct, 0.4, label="median", color="#4c72b0")
    ax.bar(xk + 0.2, sub.val_p95_err_pct, 0.4, label="95th percentile", color="#dd8452")
    ax.set_yscale("log")
    ax.set_xticks(xk, ["poly\nridge", "random\nforest", "grad.\nboost", "MLP"])
    ax.set_title(CASE_LABEL[case])
axes[0].set_ylabel("validation relative error (%)")
axes[0].legend()
fig.suptitle("Model comparison: pooled relative error over Id, Ig, Is, Ib (validation set)")
save(fig, OUT / "model_comparison")

# predicted vs SPICE, test set, one figure per case
for case in cases:
    model = joblib.load(MODELS / f"{case}.joblib")
    df = load(case)
    te = np.array(SPLITS[case]["test"])
    P = model.predict(df.iloc[te])
    T = df.iloc[te][metrics.CURRENTS].to_numpy()
    floors = metrics.score_floors(case)
    E = metrics.rel_errors(P, T, floors)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8), layout="constrained")
    for k, (ax, c) in enumerate(zip(axes.flat, metrics.CURRENTS)):
        x, y = np.abs(T[:, k]), np.abs(P[:, k])
        scored = np.isfinite(E[:, k])
        wrong = scored & (np.sign(P[:, k]) != np.sign(T[:, k]))
        ok = scored & ~wrong
        lo = max(min(x[scored].min(), y[scored].min()), 1e-30) / 3
        hi = max(x.max(), y.max()) * 3
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.scatter(x[ok], y[ok], s=4, alpha=0.6, color="#1f77b4", label="test samples", rasterized=True)
        if wrong.any():
            ax.scatter(x[wrong], y[wrong], s=14, marker="x", color="#d62728", label=f"sign wrong ({wrong.sum()})")
        if (~scored).any():
            ax.scatter(np.maximum(x[~scored], lo), np.maximum(y[~scored], lo), s=4, color="0.6", alpha=0.5,
                       label=f"below SPICE resolution ({(~scored).sum()})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"|{c.upper()[0]}{c[1:]}| SPICE (A)")
        ax.set_ylabel(f"|{c.upper()[0]}{c[1:]}| predicted (A)")
        e = E[scored, k]
        ax.set_title(f"{c}: median {np.median(e):.2f}%, p95 {np.percentile(e, 95):.1f}%")
        ax.legend(loc="upper left", fontsize=7.5, markerscale=2)
    fig.suptitle(f"{CASE_LABEL[case]}: {fetmodel.MODEL_LABEL[model.kind]} vs SPICE, test set ({len(te)} samples)")
    save(fig, OUT / f"pred_vs_spice_{case}")

# learning curve
if not args.no_learning_curve:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for case in cases:
        sub = lc_df[lc_df.case == case].sort_values("n_train")
        line, = ax.plot(sub.n_train, sub.val_p95_err_pct, "-o", label=f"{CASE_LABEL[case]} p95")
        ax.plot(sub.n_train, sub.val_median_err_pct, "--s", color=line.get_color(), ms=4,
                label=f"{CASE_LABEL[case]} median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("validation relative error (%)")
    ax.set_title("Learning curve of the chosen model type")
    ax.legend(fontsize=7, ncol=2)
    save(fig, OUT / "learning_curve")

print(f"\nwrote models/<case>.joblib + .json, {OUT.relative_to(config.ROOT)}/: model_comparison.csv/.png, "
      "chosen_models.csv, all_configs.csv, tables.md, pred_vs_spice_<case>.png, learning_curve.csv/.png")
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nPhase 3 complete.")
