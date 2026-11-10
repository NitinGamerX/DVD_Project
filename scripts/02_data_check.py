#!/usr/bin/env python3
"""Step 02 (deliverable 3): data check on the four single-transistor datasets.

1. Kirchhoff check per case: the four terminal currents must sum to zero within
   reltol * sum|I| + abstol (config.yaml -> data_check). Full residual statistics
   are reported, not only the pass count.
2. Three trend plots, each from the dataset itself (scatter + binned median) with a
   controlled one-parameter SPICE sweep overlaid (all other inputs nominal):
     a. OFF current vs temperature      (NMOS OFF, PMOS OFF)
     b. OFF current vs vth0             (NMOS OFF, PMOS OFF)
     c. gate current vs toxe            (all four cases)
   OFF current = |current at the physical drain| (higher-potential diffusion for
   NMOS, lower for PMOS) from samples at classic OFF bias (source side within
   off_rail_frac * VDD of its rail, |Vd - Vs| >= min_vds_frac * VDD). The OFF-current
   sweeps also show the drain current split into its channel (subthreshold) part
   and its gate-tunnelling part.
   Currents are per micrometre of effective width W - 2*wint (wint from the model
   file) so that W does not hide the trend.
   Each trend must have the physically expected direction in both the dataset
   (Spearman rank correlation) and the sweep (monotonic).

Outputs (results/data_check/): kirchhoff.csv, kirchhoff.png, trend_summary.csv,
off_current_vs_temperature.png, off_current_vs_vth0.png, gate_current_vs_toxe.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvd import config, fetsim, modelcard, sampling  # noqa: E402
from dvd.plotting import CASE_COLOR, CASE_LABEL, plt, save  # noqa: E402

cfg = config.load()
dc = cfg["data_check"]
a = cfg["assignment"]
CASES = list(cfg["cases"])
OUT = config.path("results", "data_check")
OUT.mkdir(parents=True, exist_ok=True)
failures: list[str] = []

data = {c: pd.read_csv(config.path("data", "fet", f"{c}.csv")) for c in CASES}
for c, df in data.items():
    if list(df.columns) != cfg["part_a"]["csv_columns"]:
        sys.exit(f"{c}.csv does not have the deliverable columns; run `make fet-data` first")

# =============================================================================
# 1. Kirchhoff check
# =============================================================================
print("Kirchhoff check: |Id+Ig+Is+Ib| <= %.0e * sum|I| + %.0e A" % (dc["kirchhoff_reltol"], dc["kirchhoff_abstol_A"]))
rows = []
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.4), sharex=True, sharey=True, layout="constrained")
for ax, case in zip(axes.flat, CASES):
    I = data[case][fetsim.CURRENTS].to_numpy()
    resid = np.abs(I.sum(axis=1))
    mag = np.abs(I).sum(axis=1)
    tol = dc["kirchhoff_reltol"] * mag + dc["kirchhoff_abstol_A"]
    ok = resid <= tol
    rel = resid / mag
    worst = int(np.argmax(rel))
    rows.append(dict(case=case, samples=len(I), passed=int(ok.sum()),
                     pass_rate=float(ok.mean()),
                     pass_rate_reltol_only=float((resid <= dc["kirchhoff_reltol"] * mag).mean()),
                     median_abs_residual_A=float(np.median(resid)),
                     p99_abs_residual_A=float(np.quantile(resid, .99)),
                     max_abs_residual_A=float(resid.max()),
                     median_rel_residual=float(np.median(rel)),
                     p99_rel_residual=float(np.quantile(rel, .99)),
                     max_rel_residual=float(rel.max()),
                     exact_zero_residuals=int((resid == 0).sum()),
                     worst_rel_sample=worst))
    print(f"  {case:9s} {ok.sum():5d}/{len(I)} pass | |sum| median {np.median(resid):.1e} A, max {resid.max():.1e} A"
          f" | sum/sum|I| median {np.median(rel):.1e}, max {rel.max():.1e}")
    if not ok.all():
        failures.append(f"{case}: {int((~ok).sum())} samples fail the Kirchhoff check")
    shown = np.maximum(resid, 1e-26)
    ax.scatter(mag, shown, s=2, alpha=0.35, color=CASE_COLOR[case], rasterized=True, label="samples")
    xs = np.logspace(np.log10(mag.min()) - 0.3, np.log10(mag.max()) + 0.3, 200)
    ax.plot(xs, dc["kirchhoff_reltol"] * xs + dc["kirchhoff_abstol_A"], "k--", lw=1.1, label="tolerance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{CASE_LABEL[case]}: {ok.sum()}/{len(I)} within tolerance")
    ax.legend(loc="upper left", markerscale=4)
for ax in axes[1]:
    ax.set_xlabel("|Id| + |Ig| + |Is| + |Ib|  (A)")
for ax in axes[:, 0]:
    ax.set_ylabel("|Id + Ig + Is + Ib|  (A)")
fig.suptitle("Kirchhoff check: sum of the four terminal currents (exact zeros drawn at 1e-26 A)")
kfig = save(fig, OUT / "kirchhoff")
pd.DataFrame(rows).to_csv(OUT / "kirchhoff.csv", index=False)

# =============================================================================
# 2. Trends
# =============================================================================
nom = sampling.process_nominals()
vdd0 = a["vdd_nominal"]
w_sweep = dc["sweep_w_mult"] * a["lmin"]
npts = dc["sweep_points"]


_cards = modelcard.parse(config.model_file())
WINT = {"n": _cards[a["nmos_model"]]["wint"], "p": _cards[a["pmos_model"]]["wint"]}


def per_um(i, w, typ):
    """|current| per micrometre of effective width W - 2*wint."""
    return np.abs(i) / ((np.asarray(w) - 2 * WINT[typ]) * 1e6)


def off_bias(df: pd.DataFrame, typ: str) -> pd.Series:
    """Samples at classic OFF bias (see module docstring)."""
    lo, hi = np.minimum(df.vd, df.vs), np.maximum(df.vd, df.vs)
    rail = lo <= dc["off_rail_frac"] * df.vdd if typ == "n" else hi >= (1 - dc["off_rail_frac"]) * df.vdd
    return rail & ((hi - lo) >= dc["min_vds_frac"] * df.vdd)


def off_current(df: pd.DataFrame, typ: str) -> pd.Series:
    """|current at the physical drain|: NMOS higher-potential terminal, PMOS lower."""
    d_is_drain = df.vd >= df.vs if typ == "n" else df.vd <= df.vs
    return pd.Series(np.where(d_is_drain, df["id"].abs(), df["is"].abs()), index=df.index)


# Controlled-sweep bias per case (all process parameters nominal unless swept).
SWEEP_BIAS = {  # (vd, vs) in units of VDD
    "nmos_off": (1.0, 0.0),   # Vds = VDD: classic Ioff
    "pmos_off": (0.0, 1.0),   # Vsd = VDD
    "nmos_on": (0.0, 0.0),    # Vgs = Vgd = VDD: maximum gate-to-channel tunnelling
    "pmos_on": (1.0, 1.0),
}


def sweep(case: str, param: str, values: np.ndarray) -> np.ndarray:
    """Currents (n, 4) from a one-parameter SPICE sweep at nominal everything else."""
    spec = cfg["cases"][case]
    t = spec["type"]
    vd, vs = SWEEP_BIAS[case]
    base = dict(vdd=vdd0, temp=27.0, w=w_sweep, vd=vd * vdd0, vs=vs * vdd0,
                vg=vdd0 if str(spec["vg"]).upper() == "VDD" else 0.0,
                vb=vdd0 if str(spec["vb"]).upper() == "VDD" else 0.0, **nom[t])
    rows_ = []
    for v in values:
        r = dict(base)
        r[param] = v
        rows_.append(r)
    return fetsim.simulate(case, pd.DataFrame(rows_)[sampling.INPUT_COLUMNS], jobs=1)


def binned_median(x, y, nbins):
    edges = np.linspace(np.min(x), np.max(x), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    cx, cy = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 10:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            cy.append(10 ** np.median(np.log10(y[m])))
    return np.array(cx), np.array(cy)


def partial_spearman(x, y, z) -> float:
    """Rank correlation of x and y with z held fixed (partial Spearman)."""
    rxy, rxz, ryz = (spearmanr(u, v).statistic for u, v in ((x, y), (x, z), (y, z)))
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def monotonic(y, sign):
    d = np.diff(np.log10(y))
    return bool(np.all(d > 0)) if sign > 0 else bool(np.all(d < 0))


summary = []


def trend_panel(ax, x, y, c, clabel, xs, ys, case, xlabel, ylabel, expected_sign, metric_name, metric,
                parts=None, sweep_label="SPICE sweep, rest nominal"):
    sc = ax.scatter(x, y, c=c, s=4, alpha=0.6, cmap="viridis", rasterized=True)
    bx, by = binned_median(x, y, dc["trend_bins"])
    ax.plot(bx, by, "k-o", ms=3, lw=1.4, label="dataset: binned median")
    ax.plot(xs, ys, "--", color="#d62728", lw=2, label=sweep_label)
    for (lab, yy, style) in (parts or []):
        ax.plot(xs, yy, style, lw=1.5, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=7.5)
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label(clabel)
    rho = spearmanr(x, y).statistic
    prho = partial_spearman(x, y, np.asarray(c))
    ok = (np.sign(rho) == expected_sign) and (np.sign(prho) == expected_sign) and monotonic(ys, expected_sign)
    ax.set_title(f"{CASE_LABEL[case]}   Spearman rho = {rho:+.2f}  ({prho:+.2f} with {clabel.split(' (')[0]} held fixed)",
                 fontsize=9.5)
    summary.append(dict(trend=ylabel.split(" (")[0], x=xlabel.split(" (")[0], case=case, samples_plotted=len(x),
                        spearman_rho=float(rho), partial_rho=prho, held_fixed=clabel.split(" (")[0],
                        expected_sign="+" if expected_sign > 0 else "-",
                        sweep_monotonic=monotonic(ys, expected_sign),
                        sweep_metric=metric_name, sweep_metric_value=float(metric), passed=bool(ok)))
    if not ok:
        failures.append(f"{case}: {ylabel} vs {xlabel} trend has the wrong direction")
    return rho, prho


def parts_styled(cur: np.ndarray, typ: str) -> list[tuple[str, np.ndarray, str]]:
    """Drain current split for the OFF sweeps: channel part (|Is| at the grounded-side
    source, where Vgs = 0 and no tunnelling) and gate-tunnelling part (|Ig|)."""
    return [("  channel part |Is| (subthreshold)", per_um(cur[:, 2], w_sweep, typ), ":"),
            ("  gate-tunnelling part |Ig|", per_um(cur[:, 1], w_sweep, typ), "-.")]


# ---- a. OFF current vs temperature -------------------------------------------
print("\nTrend a: OFF current vs temperature")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
temps = np.linspace(*cfg["sweep"]["temp_c"], npts)
for ax, case in zip(axes, ["nmos_off", "pmos_off"]):
    t = cfg["cases"][case]["type"]
    df = data[case]
    m = off_bias(df, t)
    y = per_um(off_current(df[m], t), df.w[m], t).to_numpy()
    cur = sweep(case, "temp", temps)          # sweep bias: terminal d is the physical drain
    sw = per_um(cur[:, 0], w_sweep, t)
    ratio, ratio_ch = sw[-1] / sw[0], abs(cur[-1, 2] / cur[0, 2])
    rho, prho = trend_panel(ax, df.temp[m].to_numpy(), y, np.abs(df.vth0[m]), "|vth0| (V)", temps, sw, case,
                      "temperature (°C)", "OFF current (A/µm of W_eff)", +1, "I(125C)/I(-40C) total", ratio,
                      parts=parts_styled(cur, t), sweep_label="SPICE sweep: drain current (total)")
    summary[-1]["channel_only_metric"] = float(ratio_ch)
    print(f"  {case:9s} dataset rho {rho:+.2f} (partial {prho:+.2f}, {m.sum()} samples) | sweep -40 -> 125 °C: total x{ratio:.0f}, "
          f"channel part x{ratio_ch:.0f}; gate part is {abs(cur[0, 1] / cur[0, 0]):.0%} of the drain current at -40 °C")
fig.suptitle("OFF current vs temperature (dataset: source side within "
             f"{dc['off_rail_frac']:.0%} of its rail, |Vds| >= {dc['min_vds_frac']} VDD; "
             f"sweep: |Vds| = VDD, W = {w_sweep*1e9:.0f} nm)")
fa = save(fig, OUT / "off_current_vs_temperature")

# ---- b. OFF current vs vth0 ---------------------------------------------------
print("\nTrend b: OFF current vs vth0")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
rel = np.linspace(*cfg["sweep"]["process_rel"], npts)
for ax, case in zip(axes, ["nmos_off", "pmos_off"]):
    t = cfg["cases"][case]["type"]
    df = data[case]
    m = off_bias(df, t)
    y = per_um(off_current(df[m], t), df.w[m], t).to_numpy()
    vth = nom[t]["vth0"] * (1 + rel)
    cur = sweep(case, "vth0", vth)
    sw = per_um(cur[:, 0], w_sweep, t)
    dv = 1e3 * (np.abs(vth[-1]) - np.abs(vth[0]))
    mv_per_dec = dv / abs(np.log10(sw[-1] / sw[0]))
    mv_per_dec_ch = dv / abs(np.log10(cur[-1, 2] / cur[0, 2]))
    rho, prho = trend_panel(ax, np.abs(df.vth0[m]).to_numpy(), y, df.temp[m], "temperature (°C)", np.abs(vth), sw, case,
                      "|vth0| (V)", "OFF current (A/µm of W_eff)", -1, "mV of |vth0| per decade, total", mv_per_dec,
                      parts=parts_styled(cur, t), sweep_label="SPICE sweep: drain current (total)")
    summary[-1]["channel_only_metric"] = float(mv_per_dec_ch)
    print(f"  {case:9s} dataset rho {rho:+.2f} (partial {prho:+.2f}, {m.sum()} samples) | sweep: one decade per {mv_per_dec:.0f} mV "
          f"of |vth0| (total), {mv_per_dec_ch:.0f} mV (channel part)")
fig.suptitle("OFF current vs vth0 (dataset: source side within "
             f"{dc['off_rail_frac']:.0%} of its rail, |Vds| >= {dc['min_vds_frac']} VDD; "
             f"sweep: |Vds| = VDD, 27 °C, W = {w_sweep*1e9:.0f} nm)")
fb = save(fig, OUT / "off_current_vs_vth0")

# ---- c. gate current vs toxe ----------------------------------------------------
print("\nTrend c: gate current vs toxe")
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
bias_txt = {"nmos_on": "Vd = Vs = 0", "pmos_on": "Vd = Vs = VDD", "nmos_off": "Vd = VDD, Vs = 0", "pmos_off": "Vd = 0, Vs = VDD"}
for ax, case in zip(axes.flat, CASES):
    t = cfg["cases"][case]["type"]
    df = data[case]
    y = per_um(df.ig, df.w, t).to_numpy()
    vox = np.abs(df.vg - 0.5 * (df.vd + df.vs))
    tox = nom[t]["toxe"] * (1 + rel)
    sw = per_um(sweep(case, "toxe", tox)[:, 1], w_sweep, t)
    dec_per_A = abs(np.polyfit(tox * 1e10, np.log10(sw), 1)[0])
    rho, prho = trend_panel(ax, (df.toxe * 1e10).to_numpy(), y, vox, "|Vg - (Vd+Vs)/2| (V)", tox * 1e10, sw, case,
                      "toxe (Å)", "gate current (A/µm of W_eff)", -1, "decades per Å of toxe", dec_per_A)
    ax.set_title(ax.get_title() + f"\nsweep: {bias_txt[case]}", fontsize=9.5)
    print(f"  {case:9s} dataset rho {rho:+.2f} (partial {prho:+.2f}) | sweep: {dec_per_A:.2f} decades per Å")
fig.suptitle(f"Gate current vs toxe (all samples; sweep at the largest oxide voltage, 27 °C, "
             f"W = {w_sweep*1e9:.0f} nm, so it bounds the cloud from above)")
fc = save(fig, OUT / "gate_current_vs_toxe")

pd.DataFrame(summary).to_csv(OUT / "trend_summary.csv", index=False)
print("\nwrote " + ", ".join(str(p.relative_to(config.ROOT)) for p in
                            [OUT / "kirchhoff.csv", kfig, OUT / "trend_summary.csv", fa, fb, fc]))
if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nPhase 2 complete: all checks passed.")
