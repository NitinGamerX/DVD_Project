"""v3 transistor model: our structure plus the best ideas from the teammate's implementation.

Kept in its own module so that src/dvd/fetmodel.py (v1/v2 models, Phase 3 cache) stays
byte-identical. Differences from FetModel (src/dvd/fetmodel.py):

Channel target. Instead of log10(G/Weff) with G = T/(Vd - Vs), the model learns
    y1 = log10( T / (phi(Vds) * Weff) ),   phi(v) = sign(v) * Vt * (1 - exp(-|v| / Vt))
with Vt = kT/q. phi ~ Vds for |Vds| << Vt (so y1 ~ log G, smooth through Vds = 0) and
phi -> Vt once the current saturates (so y1 ~ log of the saturation current, flat in Vds).
OFF devices in gates run saturated at |Vds| ~ VDD, where log G instead keeps falling as
-log Vds and the network has to learn that slope.

Features. On top of the v1/v2 features: 1000/T, junction reverse-bias terms
1 - exp(-Vr/Vt) for both diffusions and their log-sum, tanh(Vds/Vt), and the log distance
of the gate from each diffusion, log(|Vg - V| + 0.01). The teammate found the last pair
cut the ON-state gate-current tail threefold and the junction terms halved the body tail.

Fitting. One network per target, each fitted only on rows where its target is defined:
in rail-focused data a few rows have |Vds| of microvolts or an unbiased junction, where the
sign assumptions behind the log targets do not hold (the currents there are ~0).
Kirchhoff is still exact by construction (same rebuild as FetModel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .fetmodel import FetModel

K_B_OVER_Q = 8.617333262e-5          # V/K


def _vt(df: pd.DataFrame) -> np.ndarray:
    return K_B_OVER_Q * (df.temp.to_numpy(float) + 273.15)


def phi(v: np.ndarray, vt: np.ndarray) -> np.ndarray:
    return np.sign(v) * vt * (-np.expm1(-np.abs(v) / vt))


class FetModelV3(FetModel):
    FEATURES = ["vdd", "temp_k", "log_weff", "vhi_b", "vlo_b", "vth0", "toxe_A", "log_ndep", "u0", "xj_nm",
                "inv_t", "j_hi", "j_lo", "log_j", "tanh_vds", "log_g_hi", "log_g_lo"]

    # hp["channel"]: "phi" (default) or "g" (v1/v2 conductance target)
    # hp["extra_features"]: True (default) or False (v1/v2 features only)
    def features(self, df: pd.DataFrame) -> np.ndarray:
        base = super().features(df)
        if not self.hp.get("extra_features", True):
            return base
        vt = _vt(df)
        vhi = np.maximum(df.vd, df.vs).to_numpy(float)
        vlo = np.minimum(df.vd, df.vs).to_numpy(float)
        vb, vg = df.vb.to_numpy(float), df.vg.to_numpy(float)
        sgn = 1.0 if self.case.startswith("n") else -1.0          # reverse bias: NMOS v - vb, PMOS vb - v
        j_hi = -np.expm1(-np.clip(sgn * (vhi - vb), 0, None) / vt)
        j_lo = -np.expm1(-np.clip(sgn * (vlo - vb), 0, None) / vt)
        extra = np.column_stack([1000.0 / (df.temp.to_numpy(float) + 273.15), j_hi, j_lo,
                                 np.log(j_hi + j_lo + 0.01), np.tanh((vhi - vlo) / vt),
                                 np.log(np.abs(vg - vhi) + 0.01), np.log(np.abs(vg - vlo) + 0.01)])
        return np.column_stack([base, extra])

    def _parts(self, df: pd.DataFrame):
        J = -df[["id", "ig", "is", "ib"]].to_numpy(float)
        weff = (df.w - 2 * self.wint).to_numpy()
        dv = (df.vd - df.vs).to_numpy(float)
        ph = phi(dv, _vt(df)) if self.hp.get("channel", "phi") == "phi" else dv
        return J, weff, ph

    def targets(self, df: pd.DataFrame) -> np.ndarray:
        """Targets; NaN where undefined (sign assumption broken, current ~0)."""
        J, weff, ph = self._parts(df)
        ratio = (J[:, 0] - J[:, 2]) / 2 / ph
        with np.errstate(invalid="ignore", divide="ignore"):
            y1 = np.where(ratio > 0, np.log10(ratio / weff), np.nan)
            g = self.gate_sign * J[:, 1]
            y2 = np.where(g > 0, np.log10(g / weff), np.nan)
        y3 = np.arcsinh(J[:, 3] / (2 * self.s_b))
        return np.column_stack([y1, y2, y3])

    def currents_from(self, Y: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        weff = (df.w - 2 * self.wint).to_numpy()
        dv = (df.vd - df.vs).to_numpy(float)
        T = 10 ** Y[:, 0] * weff * (phi(dv, _vt(df)) if self.hp.get("channel", "phi") == "phi" else dv)
        Jg = self.gate_sign * 10 ** Y[:, 1] * weff
        Jb = 2 * self.s_b * np.sinh(Y[:, 2])
        S = -(Jg + Jb) / 2
        return -np.column_stack([T + S, Jg, -T + S, Jb])

    def fit(self, df: pd.DataFrame) -> "FetModelV3":
        jg = -df["ig"].to_numpy()
        self.gate_sign = float(np.sign(np.median(jg)))
        Y = self.targets(df)
        X = self.features(df)
        ok = np.isfinite(Y)
        self.y_mean = np.array([Y[ok[:, k], k].mean() for k in range(3)])
        self.y_std = np.array([Y[ok[:, k], k].std() for k in range(3)])
        self.dropped = [int((~ok[:, k]).sum()) for k in range(3)]
        hp = self.hp
        self.estimator = []
        for k in range(3):
            net = MLPRegressor(hidden_layer_sizes=tuple(hp["hidden"]), activation="tanh", solver="lbfgs",
                               alpha=hp["alpha"], max_iter=hp["max_iter"], max_fun=4 * hp["max_iter"],
                               tol=1e-10, random_state=self.seed)
            est = make_pipeline(StandardScaler(), net)
            est.fit(X[ok[:, k]], (Y[ok[:, k], k] - self.y_mean[k]) / self.y_std[k])
            self.estimator.append(est)
        return self

    def predict_targets(self, df: pd.DataFrame) -> np.ndarray:
        X = self.features(df)
        Z = np.column_stack([e.predict(X) for e in self.estimator])
        return Z * self.y_std + self.y_mean
