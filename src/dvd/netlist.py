"""Netlist builders. Every deck gets the mandatory .options line from config.yaml."""
from __future__ import annotations

from pathlib import Path

from . import config


def fmt(x: float) -> str:
    """Shortest text that round-trips the float exactly."""
    return repr(float(x))


def mos_line(name: str, d: str, g: str, s: str, b: str, model: str, w: float) -> str:
    """One transistor, including source/drain diffusion geometry (AS, AD, PS, PD)."""
    cfg = config.load()
    lmin, ldiff = cfg["assignment"]["lmin"], cfg["spice"]["ldiff"]
    area, perim = w * ldiff, 2.0 * (w + ldiff)
    return (f"{name} {d} {g} {s} {b} {model} W={fmt(w)} L={fmt(lmin)} "
            f"AS={fmt(area)} AD={fmt(area)} PS={fmt(perim)} PD={fmt(perim)}")


def mos_line_expr(name: str, d: str, g: str, s: str, b: str, model: str, wexpr: str = "w") -> str:
    """Same device line with W given as a SPICE parameter expression (for batch runs)."""
    cfg = config.load()
    lmin, ldiff = fmt(cfg["assignment"]["lmin"]), fmt(cfg["spice"]["ldiff"])
    return (f"{name} {d} {g} {s} {b} {model} W={{{wexpr}}} L={lmin} "
            f"AS={{{wexpr}*{ldiff}}} AD={{{wexpr}*{ldiff}}} "
            f"PS={{2*({wexpr}+{ldiff})}} PD={{2*({wexpr}+{ldiff})}}")


def resolve_terminal(expr, vdd: float) -> float:
    """Case table entries are 'VDD' or a number."""
    if isinstance(expr, (int, float)):
        return float(expr)
    e = str(expr).strip().upper()
    return vdd if e == "VDD" else float(e)


def single_fet_deck(case: str, vdd: float, w: float, vd: float, vs: float,
                    temp_c: float, model_path: Path) -> str:
    """Single-transistor deck for one of the four cases; every terminal has its own source."""
    cfg = config.load()
    spec = cfg["cases"][case]
    model = cfg["assignment"]["nmos_model" if spec["type"] == "n" else "pmos_model"]
    vg = resolve_terminal(spec["vg"], vdd)
    vb = resolve_terminal(spec["vb"], vdd)
    return "\n".join([
        f"* single {model.upper()} - case {case}",
        f"* VDD={fmt(vdd)} V  T={fmt(temp_c)} C  W={fmt(w)} m",
        f".include {model_path}",
        cfg["spice"]["options"],
        f".temp {fmt(temp_c)}",
        mos_line("M1", "d", "g", "s", "b", model, w),
        f"Vd d 0 {fmt(vd)}",
        f"Vg g 0 {fmt(vg)}",
        f"Vs s 0 {fmt(vs)}",
        f"Vb b 0 {fmt(vb)}",
        ".control",
        "set numdgt=12",
        "op",
        "print i(vd) i(vg) i(vs) i(vb)",
        ".endc",
        ".end",
        "",
    ])
