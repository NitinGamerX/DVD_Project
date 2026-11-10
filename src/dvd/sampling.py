"""Random sampling of transistor inputs exactly as specified in section 4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, modelcard, netlist

INPUT_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]


def process_nominals() -> dict[str, dict[str, float]]:
    """{'n': {...}, 'p': {...}} nominal process values read from the model file."""
    cfg = config.load()
    names = cfg["sweep"]["process_params"]
    mf = config.model_file()
    return {"n": modelcard.nominals(mf, cfg["assignment"]["nmos_model"], names),
            "p": modelcard.nominals(mf, cfg["assignment"]["pmos_model"], names)}


def draw_fet_inputs(case: str, n: int, seed: int) -> pd.DataFrame:
    """n independent samples for one case. Columns in deliverable order (inputs only).

    Order of draws (fixed for reproducibility): VDD first, then Vd and Vs
    (which depend on VDD), W, temperature, then the five process parameters.
    Gate and body voltages are set by the case table, not drawn.
    """
    cfg = config.load()
    a, s = cfg["assignment"], cfg["sweep"]
    spec = cfg["cases"][case]
    rng = np.random.default_rng(seed)

    vdd = a["vdd_nominal"] * (1.0 + rng.uniform(*s["vdd_rel"], n))
    vd = vdd * rng.uniform(*s["vd_frac"], n)
    vs = vdd * rng.uniform(*s["vs_frac"], n)
    w = a["lmin"] * rng.uniform(*s["w_mult"], n)
    temp = rng.uniform(*s["temp_c"], n)
    nom = process_nominals()[spec["type"]]
    proc = {p: nom[p] * (1.0 + rng.uniform(*s["process_rel"], n)) for p in s["process_params"]}

    vg = np.array([netlist.resolve_terminal(spec["vg"], v) for v in vdd])
    vb = np.array([netlist.resolve_terminal(spec["vb"], v) for v in vdd])
    df = pd.DataFrame(dict(vdd=vdd, temp=temp, w=w, vd=vd, vs=vs, vg=vg, vb=vb, **proc))
    return df[INPUT_COLUMNS]
