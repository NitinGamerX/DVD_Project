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


def corner_columns() -> list[str]:
    names = config.load()["sweep"]["process_params"]
    return ["vdd", "temp"] + [f"{p}_n" for p in names] + [f"{p}_p" for p in names]


def draw_corners(n: int, seed: int) -> pd.DataFrame:
    """n circuit corners (Part B step 3, also reused in Part C).

    One corner = one draw of VDD, temperature and process parameters, with the
    section 4 distributions. All NMOS of a circuit share the '_n' draw, all PMOS the
    '_p' draw. Order of draws: VDD, temperature, NMOS parameters, PMOS parameters.
    """
    cfg = config.load()
    a, s = cfg["assignment"], cfg["sweep"]
    rng = np.random.default_rng(seed)
    nom = process_nominals()
    out = {"vdd": a["vdd_nominal"] * (1.0 + rng.uniform(*s["vdd_rel"], n)),
           "temp": rng.uniform(*s["temp_c"], n)}
    for t in ("n", "p"):
        for p in s["process_params"]:
            out[f"{p}_{t}"] = nom[t][p] * (1.0 + rng.uniform(*s["process_rel"], n))
    df = pd.DataFrame(out)[corner_columns()]
    df.insert(0, "corner", np.arange(n))
    return df


def draw_fet_inputs_rail(case: str, n: int, seed: int, dist: tuple[float, float] = (1e-6, 0.3),
                         p_near: float = 0.5) -> pd.DataFrame:
    """Rail-focused samples (improvement step v3): identical to draw_fet_inputs except that
    Vd and Vs are each, with probability p_near, placed a log-uniform distance dist (V)
    from a randomly chosen rail (0 or VDD) instead of uniformly in [0, VDD]. Inside a static
    gate every transistor terminal sits within about 20 uV - 2 mV of a rail, a region that
    uniform sampling almost never visits."""
    cfg = config.load()
    a, s = cfg["assignment"], cfg["sweep"]
    spec = cfg["cases"][case]
    rng = np.random.default_rng(seed)
    vdd = a["vdd_nominal"] * (1.0 + rng.uniform(*s["vdd_rel"], n))

    def terminal():
        v = vdd * rng.uniform(0.0, 1.0, n)
        near = rng.uniform(size=n) < p_near
        d = 10 ** rng.uniform(np.log10(dist[0]), np.log10(dist[1]), n)
        at_vdd = rng.uniform(size=n) < 0.5
        return np.where(near, np.where(at_vdd, vdd - d, d), v)

    vd = terminal()
    vs = terminal()
    w = a["lmin"] * rng.uniform(*s["w_mult"], n)
    temp = rng.uniform(*s["temp_c"], n)
    nom = process_nominals()[spec["type"]]
    proc = {p: nom[p] * (1.0 + rng.uniform(*s["process_rel"], n)) for p in s["process_params"]}
    vg = np.array([netlist.resolve_terminal(spec["vg"], v) for v in vdd])
    vb = np.array([netlist.resolve_terminal(spec["vb"], v) for v in vdd])
    df = pd.DataFrame(dict(vdd=vdd, temp=temp, w=w, vd=vd, vs=vs, vg=vg, vb=vb, **proc))
    return df[INPUT_COLUMNS]
