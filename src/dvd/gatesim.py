"""SPICE simulation of a transistor-level circuit over corners x input vectors.

Used for the gates (Part B) and the test circuit (Part C). The circuit is a list of
dvd.gates.Fet. Every input pin gets its own voltage source held at exactly 0 or VDD,
the supply is source Vdd, and all bodies are tied as the cell library specifies.

Batch path: one ngspice process per chunk of corners; the parametrised model card
carries the corner's NMOS and PMOS process values; `alterparam` + `reset` + `op` per
(corner, vector). Literal path: one sample per process, every value written as a
number into a private copy of the model card (independent check of the batch path).

For each operating point we record the branch current of every source (ngspice
convention: negative = the source delivers current into the circuit) and the voltage
of every requested node.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import config, fetsim, modelcard, netlist, spice
from .gates import Fet

_TAG = re.compile(r"^@@ (\d+) (\d+)\s*$")
_VAL = re.compile(r"^(\S+) = (\S+)\s*$")


def _model(t: str) -> str:
    a = config.load()["assignment"]
    return a["nmos_model"] if t == "n" else a["pmos_model"]


def device_lines(fets: list[Fet]) -> list[str]:
    return [netlist.mos_line(f.name, f.d, f.g, f.s, f.b, _model(f.type), f.w) for f in fets]


def record_names(pins: list[str], nodes: list[str]) -> list[str]:
    """Column order of the recorded quantities."""
    return ["i(vdd)"] + [f"i(v{p})" for p in pins] + [f"v({n})" for n in nodes]


def _process_values(corner: pd.Series) -> dict[str, dict[str, float]]:
    names = config.load()["sweep"]["process_params"]
    return {t: {p: float(corner[f"{p}_{t}"]) for p in names} for t in ("n", "p")}


def batch_deck(fets, pins, nodes, corners: pd.DataFrame, vectors, include: str, title: str) -> str:
    cfg = config.load()
    names = cfg["sweep"]["process_params"]
    from .sampling import process_nominals
    nom = process_nominals()
    lines = [f"* {title}"]
    lines += [".param " + " ".join(f"{p}_{t}={repr(float(nom[t][p]))}" for p in names) for t in ("n", "p")]
    lines += [f".include {include}", cfg["spice"]["options"],
              ".param tc=27 vdd=1 " + " ".join(f"in_{p}=0" for p in pins), ".temp {tc}",
              "Vdd vdd 0 {vdd}"] + [f"V{p} {p} 0 {{in_{p}}}" for p in pins]
    lines += device_lines(fets)
    lines += [".control", "set numdgt=15"]
    prints = "print " + " ".join(record_names(pins, nodes))
    for _, c in corners.iterrows():
        vdd = float(c.vdd)
        lines += [f"alterparam vdd={repr(vdd)}", f"alterparam tc={repr(float(c.temp))}"]
        lines += [f"alterparam {p}_{t}={repr(float(c[f'{p}_{t}']))}" for t in ("n", "p") for p in names]
        for vi, vec in enumerate(vectors):
            lines += [f"alterparam in_{p}={repr(vdd * bit)}" for p, bit in zip(pins, vec)]
            lines += ["reset", "op", f"echo @@ {int(c.corner)} {vi}", prints, "destroy all"]
    lines += [".endc", ".end", ""]
    return "\n".join(lines)


def _parse(stdout: str, names: list[str]) -> dict[tuple[int, int], np.ndarray]:
    out, key, cur = {}, None, {}
    for line in stdout.splitlines():
        m = _TAG.match(line)
        if m:
            if key is not None:
                out[key] = cur
            key, cur = (int(m.group(1)), int(m.group(2))), {}
            continue
        m = _VAL.match(line)
        if m and key is not None:
            cur[m.group(1).lower()] = float(m.group(2))
    if key is not None:
        out[key] = cur
    return {k: np.array([v.get(n, np.nan) for n in names]) for k, v in out.items()}


def simulate(fets: list[Fet], pins: list[str], nodes: list[str], corners: pd.DataFrame,
             vectors: list[tuple[int, ...]], title: str = "circuit", chunk: int = 10,
             jobs: int | None = None) -> np.ndarray:
    """Recorded quantities, shape (n_corners, n_vectors, len(record_names))."""
    work = config.path("work", "circuits")
    work.mkdir(parents=True, exist_ok=True)
    include = os.path.relpath(config.path(*fetsim.PARAM_MODEL.parts), work)
    names = record_names(pins, nodes)
    corners = corners.reset_index(drop=True)
    pos = {int(c): i for i, c in enumerate(corners.corner)}
    result = np.full((len(corners), len(vectors), len(names)), np.nan)

    def run(start: int):
        sub = corners.iloc[start:start + chunk]
        out = spice.run_deck(batch_deck(fets, pins, nodes, sub, vectors, include, title), work)
        return _parse(out, names)

    with ThreadPoolExecutor(max_workers=jobs or fetsim.n_jobs()) as ex:
        for block in ex.map(run, range(0, len(corners), chunk)):
            for (ci, vi), vals in block.items():
                result[pos[ci], vi] = vals
    bad = np.argwhere(np.isnan(result).any(axis=2))
    if len(bad):
        raise spice.SpiceError(f"{title}: missing results for {len(bad)} operating points, e.g. "
                               f"corner {int(corners.corner[bad[0][0]])} vector {vectors[bad[0][1]]}")
    return result


def literal_deck(fets, pins, nodes, corner: pd.Series, vec, card: str, title: str) -> str:
    cfg = config.load()
    vdd = float(corner.vdd)
    lines = [f"* literal check: {title}", f".include {card}", cfg["spice"]["options"],
             f".temp {repr(float(corner.temp))}", f"Vdd vdd 0 {repr(vdd)}"]
    lines += [f"V{p} {p} 0 {repr(vdd * bit)}" for p, bit in zip(pins, vec)]
    lines += device_lines(fets)
    lines += [".control", "set numdgt=15", "op", "print " + " ".join(record_names(pins, nodes)),
              ".endc", ".end", ""]
    return "\n".join(lines)


def simulate_literal(fets, pins, nodes, corner: pd.Series, vec, title: str = "circuit") -> np.ndarray:
    """One operating point through the literal path (fresh model card, fresh process)."""
    work = config.path("work", "circuits_literal")
    work.mkdir(parents=True, exist_ok=True)
    card = work / f"card_{os.getpid()}_{int(corner.corner)}.pm"
    names = config.load()["sweep"]["process_params"]
    card.write_text(modelcard.render(config.model_file(), names, _process_values(corner)))
    try:
        out = spice.run_deck(literal_deck(fets, pins, nodes, corner, vec, card.name, title), work)
    finally:
        card.unlink(missing_ok=True)
    vals = spice.printed(out)
    return np.array([vals[n] for n in record_names(pins, nodes)])


def example_deck(fets, pins, nodes, corner: pd.Series, vectors, title: str, rel_include: str) -> str:
    """Readable stand-alone netlist: one corner, every input combination (deliverable 2)."""
    cfg = config.load()
    names = cfg["sweep"]["process_params"]
    vdd = float(corner.vdd)
    lines = [f"* {title}",
             f"* Corner {int(corner.corner)} of data/gates/corners.csv: VDD = {vdd!r} V, "
             f"T = {float(corner.temp)!r} C",
             "* NMOS and PMOS process parameters (all NMOS share one draw, all PMOS another):"]
    lines += [".param " + " ".join(f"{p}_{t}={repr(float(corner[f'{p}_{t}']))}" for p in names)
              for t in ("n", "p")]
    lines += [f".include {rel_include}", cfg["spice"]["options"], f".temp {repr(float(corner.temp))}",
              ".param " + " ".join(f"in_{p}=0" for p in pins),
              f"Vdd vdd 0 {vdd!r}"] + [f"V{p} {p} 0 {{in_{p}}}" for p in pins]
    lines += device_lines(fets)
    lines += [".control", "set numdgt=15"]
    for vec in vectors:
        lines += [f"alterparam in_{p}={repr(vdd * bit)}" for p, bit in zip(pins, vec)]
        lines += ["reset", "op", f"echo @@inputs {''.join(map(str, vec))}",
                  "print " + " ".join(record_names(pins, nodes))]
    lines += [".endc", ".end", ""]
    return "\n".join(lines)
