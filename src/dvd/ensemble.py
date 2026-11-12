"""Average of several transistor models (improvement step, scripts/03b_improve_models.py).

Kept in its own module so that src/dvd/fetmodel.py, and with it the Phase 3 training
cache, stays exactly as in the original run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fetmodel import FetModel


class FetEnsemble:
    """Several FetModels of the same case and configuration, trained from different
    random starts; predictions are averaged in target space (log conductance, log gate
    current, asinh body current), i.e. a geometric mean of the members' currents.
    Behaves like a FetModel for prediction (predict, predict_targets, targets)."""

    def __init__(self, members: list[FetModel]):
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = members
        first = members[0]
        self.case, self.kind, self.hp = first.case, first.kind, dict(first.hp)
        self.hp["ensemble_members"] = len(members)

    def targets(self, df: pd.DataFrame) -> np.ndarray:
        return self.members[0].targets(df)

    def predict_targets(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_targets(df) for m in self.members], axis=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Terminal currents (n, 4) in the dataset convention: I(Vd), I(Vg), I(Vs), I(Vb)."""
        return self.members[0].currents_from(self.predict_targets(df), df)
