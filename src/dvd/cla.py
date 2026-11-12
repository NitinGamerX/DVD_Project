"""3-bit carry-lookahead adder (the assigned test circuit), built only from the Part B cells.

Inputs a0..a2, b0..b2, cin (bit 0 = least significant); outputs s0..s2 and c3 (carry out).

  generate      g_i = a_i AND b_i                        (AND2)
  propagate     p_i = a_i XOR b_i                        (XOR = 4 NAND2)
  lookahead     c1 = g0 + p0 cin
                c2 = g1 + p1 g0 + p1 p0 cin
                c3 = g2 + p2 g1 + p2 p1 g0 + p2 p1 p0 cin  (AND2 / OR2 terms)
  sum           s_i = p_i XOR c_i                        (c0 = cin)

Every carry is formed directly from g, p and cin (shared product terms, no ripple
through c1/c2). 39 cells: 24 NAND2 (6 XORs), 9 AND2, 6 OR2; 186 transistors.
The same cell builders (src/dvd/gates.py) give the SPICE netlist and the hierarchical
ML model, so both describe exactly the same circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from . import gates

INPUTS = ["a0", "a1", "a2", "b0", "b1", "b2", "cin"]
OUTPUTS = ["s0", "s1", "s2", "c3"]


@dataclass(frozen=True)
class Inst:
    kind: str      # nand2 / and2 / or2 (/ inv / nor2)
    name: str
    a: str         # input nets (cell pins a, b)
    b: str
    y: str         # output net


def instances() -> list[Inst]:
    cells: list[Inst] = []

    def xor(tag: str, a: str, b: str, y: str) -> None:
        n1, n2, n3 = f"{tag}_n1", f"{tag}_n2", f"{tag}_n3"
        cells.extend([Inst("nand2", f"{tag}a", a, b, n1), Inst("nand2", f"{tag}b", a, n1, n2),
                      Inst("nand2", f"{tag}c", b, n1, n3), Inst("nand2", f"{tag}d", n2, n3, y)])

    for i in range(3):
        cells.append(Inst("and2", f"g{i}", f"a{i}", f"b{i}", f"g{i}"))
        xor(f"xp{i}", f"a{i}", f"b{i}", f"p{i}")
    cells += [
        Inst("and2", "t1", "p0", "cin", "t1"),      # p0 cin
        Inst("or2", "c1", "g0", "t1", "c1"),        # c1
        Inst("and2", "t2", "p1", "g0", "t2"),       # p1 g0
        Inst("and2", "t3", "p1", "t1", "t3"),       # p1 p0 cin
        Inst("or2", "o2", "t2", "t3", "o2"),
        Inst("or2", "c2", "g1", "o2", "c2"),        # c2
        Inst("and2", "t4", "p2", "g1", "t4"),       # p2 g1
        Inst("and2", "t5", "p2", "t2", "t5"),       # p2 p1 g0
        Inst("and2", "t6", "p2", "t3", "t6"),       # p2 p1 p0 cin
        Inst("or2", "o3a", "g2", "t4", "o3a"),
        Inst("or2", "o3b", "t5", "t6", "o3b"),
        Inst("or2", "c3", "o3a", "o3b", "c3"),      # c3 (carry out)
    ]
    xor("xs0", "p0", "cin", "s0")
    xor("xs1", "p1", "c1", "s1")
    xor("xs2", "p2", "c2", "s2")
    return cells


def transistors() -> list[gates.Fet]:
    """Flat transistor list of the whole adder (unique device and internal-node names)."""
    fets: list[gates.Fet] = []
    for c in instances():
        build = gates.CELLS[c.kind][0]
        fets += build(c.a, c.b, c.y, f"_{c.name}")
    return fets


def nets() -> list[str]:
    """Every cell output net (including the four adder outputs)."""
    return [c.y for c in instances()]


def vectors() -> list[tuple[int, ...]]:
    """All 128 input combinations, bits in the order of INPUTS."""
    return list(product((0, 1), repeat=len(INPUTS)))


def evaluate(vec: tuple[int, ...]) -> tuple[dict[str, int], list[tuple[Inst, tuple[int, int]]]]:
    """Logic simulation: value of every net, and each cell's input bits."""
    v = dict(zip(INPUTS, vec))
    states = []
    for c in instances():
        bits = (v[c.a], v[c.b])
        v[c.y] = gates.CELLS[c.kind][2](*bits)
        states.append((c, bits))
    return v, states


def operands(vec: tuple[int, ...]) -> tuple[int, int, int]:
    v = dict(zip(INPUTS, vec))
    return (v["a0"] + 2 * v["a1"] + 4 * v["a2"], v["b0"] + 2 * v["b1"] + 4 * v["b2"], v["cin"])
