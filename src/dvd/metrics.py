"""Error metrics.

Relative error of a prediction p against SPICE value s:  e = |p - s| / |s|  (in %).

A SPICE value is only scored where it is resolved: |s| >= floor, with floor =
score_floor_factor x the simulator's measured noise for that case and current
(results/data_check/numerical_resolution.csv). Below that floor SPICE itself does not
know the value to better than ~10 %, so there is no ground truth to compare against.
This affects only the PMOS OFF body current (64 % of its samples); every other
current is resolved in every sample. Unscored entries are counted and reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CURRENTS = ["id", "ig", "is", "ib"]


def score_floors(case: str) -> np.ndarray:
    """Per-current floor (A) below which a SPICE value is not scored."""
    k = config.load()["part_a"]["score_floor_factor"]
    nz = pd.read_csv(config.path("results", "data_check", "numerical_resolution.csv")).set_index("case")
    return np.array([k * nz.loc[case, f"{c}_noise_A"] for c in CURRENTS])


def rel_errors(pred: np.ndarray, true: np.ndarray, floors: np.ndarray) -> np.ndarray:
    """Relative errors in %, NaN where the SPICE value is below its floor."""
    e = 100 * np.abs(pred - true) / np.abs(true).clip(min=1e-300)
    return np.where(np.abs(true) >= floors, e, np.nan)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def evaluate(pred: np.ndarray, true: np.ndarray, floors: np.ndarray) -> dict:
    """Pooled and per-current statistics for one model on one set of samples."""
    E = rel_errors(pred, true, floors)
    out: dict = {}
    pooled = E[np.isfinite(E)]
    out["median_err_pct"] = float(np.median(pooled))
    out["p95_err_pct"] = float(np.percentile(pooled, 95))
    out["mean_err_pct"] = float(np.mean(pooled))
    r2s = []
    for k, c in enumerate(CURRENTS):
        m = np.isfinite(E[:, k])
        e = E[m, k]
        lt, lp = np.log10(np.abs(true[m, k])), np.log10(np.abs(pred[m, k]).clip(min=1e-300))
        out[f"{c}_scored"] = int(m.sum())
        out[f"{c}_median_err_pct"] = float(np.median(e))
        out[f"{c}_p95_err_pct"] = float(np.percentile(e, 95))
        out[f"{c}_mean_err_pct"] = float(np.mean(e))
        out[f"{c}_sign_agree"] = float(np.mean(np.sign(pred[m, k]) == np.sign(true[m, k])))
        out[f"{c}_r2_log"] = r2(lt, lp)
        r2s.append(out[f"{c}_r2_log"])
    out["r2_log"] = float(np.mean(r2s))
    # error on each current relative to the device's largest terminal current:
    # what matters when currents are summed into a gate's leakage
    scale = np.abs(true).max(axis=1, keepdims=True)
    S = 100 * np.abs(pred - true) / scale
    out["scaled_median_err_pct"] = float(np.median(S))
    out["scaled_p95_err_pct"] = float(np.percentile(S, 95))
    out["unscored_entries"] = int((~np.isfinite(E)).sum())
    return out
