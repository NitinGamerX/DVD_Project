"""ML model of one transistor case: inputs (12 dataset columns) -> Id, Ig, Is, Ib.

Targets. Raw terminal currents are awkward to learn: Id and Is change sign inside the
sampled range, the body current changes sign in the OFF cases, and magnitudes span
up to ten decades. Write J for the current flowing *into* each terminal (J = -I(Vx)).
Kirchhoff gives Jd + Jg + Js + Jb = 0, so Jd and Js split exactly into

    Jd = T + S,   Js = -T + S,   T = (Jd - Js) / 2,   S = -(Jg + Jb) / 2.

T is the channel current, and in every sample of all four datasets the conductance
G = T / (Vd - Vs) is positive. The model therefore learns three smooth quantities,

    y1 = log10(G / Weff)          channel conductance per effective width
    y2 = log10(|Jg| / Weff)       gate current per effective width (fixed sign per case)
    y3 = asinh(Jb / (2 s_b))      body current, sign-safe; ~ sign * ln(|Jb| / s_b)
                                  above the scale s_b, linear below it

and rebuilds all four currents from them. The rebuilt currents satisfy Kirchhoff
exactly, and Id/Is may change sign wherever the physics does.

Features. Drain and source are interchangeable (the model is symmetric under a swap;
verified to 1e-13), so the higher and lower of Vd, Vs are used, both referred to the
body. Gate and body voltages are fixed functions of VDD per case and add nothing.
Widths enter as log(Weff), Weff = W - 2*wint, because the currents scale with Weff.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = ["vdd", "temp_k", "log_weff", "vhi_b", "vlo_b", "vth0", "toxe_A", "log_ndep", "u0", "xj_nm"]
TARGETS = ["log_g", "log_ig", "asinh_ib"]
MODEL_TYPES = ["poly_ridge", "random_forest", "grad_boost", "mlp"]
MODEL_LABEL = {"poly_ridge": "Polynomial ridge", "random_forest": "Random forest",
               "grad_boost": "Gradient boosting", "mlp": "Neural network (MLP)"}


def make_estimator(kind: str, hp: dict, seed: int):
    """Unfitted estimator for one model family. Inputs are standardised for all."""
    if kind == "poly_ridge":
        return make_pipeline(StandardScaler(), PolynomialFeatures(hp["degree"]),
                             Ridge(alpha=hp["alpha"]))
    if kind == "random_forest":
        return make_pipeline(StandardScaler(), RandomForestRegressor(
            n_estimators=hp["n_estimators"], min_samples_leaf=hp["min_samples_leaf"],
            max_features=hp["max_features"], random_state=seed, n_jobs=1))
    if kind == "grad_boost":
        return make_pipeline(StandardScaler(), MultiOutputRegressor(HistGradientBoostingRegressor(
            learning_rate=hp["learning_rate"], max_iter=hp["max_iter"],
            max_leaf_nodes=hp["max_leaf_nodes"], l2_regularization=hp.get("l2", 0.0),
            early_stopping=False, random_state=seed)))
    if kind == "mlp":
        net = MLPRegressor(hidden_layer_sizes=tuple(hp["hidden"]), activation="tanh", solver="lbfgs",
                           alpha=hp["alpha"], max_iter=hp["max_iter"], max_fun=4 * hp["max_iter"],
                           tol=1e-10, random_state=seed)
        # per_target: one network per target instead of one shared network
        return make_pipeline(StandardScaler(), MultiOutputRegressor(net) if hp.get("per_target") else net)
    raise ValueError(f"unknown model type {kind}")


@dataclass
class FetModel:
    """One fitted model for one case. predict() returns I(Vd), I(Vg), I(Vs), I(Vb)."""
    case: str
    kind: str
    hp: dict
    wint: float                      # m, from the model file
    s_b: float                       # A, scale of the body-current transform
    seed: int = 0
    gate_sign: float = 0.0           # sign of Jg in this case (learned from data)
    y_mean: np.ndarray = field(default=None, repr=False)
    y_std: np.ndarray = field(default=None, repr=False)
    estimator: object = field(default=None, repr=False)

    # ---- transforms --------------------------------------------------------------
    def features(self, df: pd.DataFrame) -> np.ndarray:
        vhi = np.maximum(df.vd, df.vs) - df.vb
        vlo = np.minimum(df.vd, df.vs) - df.vb
        weff = df.w - 2 * self.wint
        return np.column_stack([df.vdd, df.temp + 273.15, np.log(weff * 1e9), vhi, vlo,
                                df.vth0, df.toxe * 1e10, np.log10(df.ndep), df.u0 * 100,
                                df.xj * 1e9]).astype(float)

    def targets(self, df: pd.DataFrame) -> np.ndarray:
        J = -df[["id", "ig", "is", "ib"]].to_numpy(float)          # into the device
        weff = (df.w - 2 * self.wint).to_numpy()
        G = (J[:, 0] - J[:, 2]) / 2 / (df.vd - df.vs).to_numpy()
        if np.any(G <= 0):
            raise ValueError(f"{self.case}: non-positive channel conductance in training data")
        return np.column_stack([np.log10(G / weff),
                                np.log10(self.gate_sign * J[:, 1] / weff),
                                np.arcsinh(J[:, 3] / (2 * self.s_b))])

    def currents_from(self, Y: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        weff = (df.w - 2 * self.wint).to_numpy()
        T = 10 ** Y[:, 0] * weff * (df.vd - df.vs).to_numpy()
        Jg = self.gate_sign * 10 ** Y[:, 1] * weff
        Jb = 2 * self.s_b * np.sinh(Y[:, 2])
        S = -(Jg + Jb) / 2
        J = np.column_stack([T + S, Jg, -T + S, Jb])
        return -J                                                     # back to I(Vx)

    # ---- fit / predict ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FetModel":
        jg = -df["ig"].to_numpy()
        self.gate_sign = float(np.sign(np.median(jg)))
        if not np.all(np.sign(jg) == self.gate_sign):
            raise ValueError(f"{self.case}: gate current changes sign; log target not valid")
        Y = self.targets(df)
        self.y_mean, self.y_std = Y.mean(axis=0), Y.std(axis=0)
        self.estimator = make_estimator(self.kind, self.hp, self.seed)
        self.estimator.fit(self.features(df), (Y - self.y_mean) / self.y_std)
        return self

    def predict_targets(self, df: pd.DataFrame) -> np.ndarray:
        Z = np.asarray(self.estimator.predict(self.features(df)), float).reshape(len(df), -1)
        return Z * self.y_std + self.y_mean

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Terminal currents (n, 4) in the dataset convention: I(Vd), I(Vg), I(Vs), I(Vb)."""
        return self.currents_from(self.predict_targets(df), df)
