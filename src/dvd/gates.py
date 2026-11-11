"""Static CMOS cells as lists of transistors.

One description per cell drives everything: the SPICE netlists (Part B and the test
circuit in Part C) and, later, the ML leakage composition. So the transistor-level
circuit that SPICE simulates and the one the ML models are applied to cannot differ.

Conventions
  * NMOS bodies to ground, PMOS bodies to VDD (standard static CMOS).
  * Series stacks: input A is the transistor nearer the rail, input B the one nearer
    the output (NAND2: B on top of A in the pull-down; NOR2: A below VDD, B above the
    output), following the stacked examples in the lecture slides.
  * Internal node names follow the project document: x (between the series NMOS of a
    NAND2), y (between the series PMOS of a NOR2), nand_out / nor_out (inside AND2 /
    OR2). A suffix keeps names unique when cells are instantiated many times.
  * Widths come from config.yaml (pending_ta.gate_sizing_m), all within 1-8 x Lmin.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from . import config

VDD, GND = "vdd", "0"


@dataclass(frozen=True)
class Fet:
    name: str     # SPICE instance name (starts with m)
    type: str     # 'n' or 'p'
    d: str
    g: str
    s: str
    b: str
    w: float      # m


def _w(cell: str) -> tuple[float, float]:
    wn, wp = config.load()["pending_ta"]["gate_sizing_m"][cell]
    return float(wn), float(wp)


def inv(a: str, y: str, sfx: str = "") -> list[Fet]:
    wn, wp = _w("inv")
    return [Fet(f"mp{sfx}", "p", y, a, VDD, VDD, wp),
            Fet(f"mn{sfx}", "n", y, a, GND, GND, wn)]


def nand2(a: str, b: str, y: str, sfx: str = "") -> list[Fet]:
    wn, wp = _w("nand2")
    x = f"x{sfx}"
    return [Fet(f"mpa{sfx}", "p", y, a, VDD, VDD, wp),     # pull-up: parallel PMOS
            Fet(f"mpb{sfx}", "p", y, b, VDD, VDD, wp),
            Fet(f"mnb{sfx}", "n", y, b, x, GND, wn),       # pull-down: B on top (output side)
            Fet(f"mna{sfx}", "n", x, a, GND, GND, wn)]     #            A at the bottom (ground side)


def nor2(a: str, b: str, y: str, sfx: str = "") -> list[Fet]:
    wn, wp = _w("nor2")
    yy = f"y{sfx}"
    return [Fet(f"mpa{sfx}", "p", yy, a, VDD, VDD, wp),    # pull-up: A at the top (VDD side)
            Fet(f"mpb{sfx}", "p", y, b, yy, VDD, wp),      #          B at the bottom (output side)
            Fet(f"mna{sfx}", "n", y, a, GND, GND, wn),     # pull-down: parallel NMOS
            Fet(f"mnb{sfx}", "n", y, b, GND, GND, wn)]


def and2(a: str, b: str, y: str, sfx: str = "") -> list[Fet]:
    n = f"nand_out{sfx}"
    return nand2(a, b, n, sfx) + inv(n, y, f"i{sfx}")


def or2(a: str, b: str, y: str, sfx: str = "") -> list[Fet]:
    n = f"nor_out{sfx}"
    return nor2(a, b, n, sfx) + inv(n, y, f"i{sfx}")


# cell name -> (builder, input pins, logic function, internal nodes when built stand-alone)
CELLS = {
    "inv":   (lambda a, b, y, s="": inv(a, y, s), ["a"],      lambda a, b=0: 1 - a,        []),
    "nand2": (nand2,                               ["a", "b"], lambda a, b: 1 - (a & b),    ["x"]),
    "nor2":  (nor2,                                ["a", "b"], lambda a, b: 1 - (a | b),    ["y"]),
    "and2":  (and2,                                ["a", "b"], lambda a, b: a & b,          ["x", "nand_out"]),
    "or2":   (or2,                                 ["a", "b"], lambda a, b: a | b,          ["y", "nor_out"]),
}
GATE_ORDER = ["inv", "nand2", "nor2", "and2", "or2"]


def standalone(gate: str) -> tuple[list[Fet], list[str], list[str]]:
    """(transistors, input pins, nodes to record) for one gate on its own, output 'out'."""
    build, pins, _logic, internal = CELLS[gate]
    fets = build("a", "b", "out") if len(pins) == 2 else build("a", None, "out")
    return fets, pins, ["out"] + internal


def vectors(gate: str) -> list[tuple[int, ...]]:
    """Every input combination, in the order 0, 1 (INV) or 00, 01, 10, 11 (AB)."""
    return list(product((0, 1), repeat=len(CELLS[gate][1])))


def expected_levels(gate: str, vec: tuple[int, ...]) -> dict[str, int]:
    """Logic level of the output and of any internal logic node, for checking SPICE."""
    logic = CELLS[gate][2]
    out = logic(*vec)
    levels = {"out": out}
    if gate == "and2":
        levels["nand_out"] = 1 - out
    if gate == "or2":
        levels["nor_out"] = 1 - out
    return levels
