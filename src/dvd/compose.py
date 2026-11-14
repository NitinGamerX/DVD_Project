"""Circuit leakage from per-transistor currents (Part B step 4, Part C step 7).

Terminal voltages. Every transistor's drain and source voltages come from the circuit's
node voltages (SPICE-recorded, or solved by the ML models). Its body sits at 0 (NMOS) or
VDD (PMOS). Its gate is at a logic rail: the transistor models know only the four
section 4 cases (gate exactly at 0 or VDD), so a gate node is read as logic 1 if it is
above VDD/2. For gates driven by circuit inputs this is exact; for a gate driven by an
internal node (the inverter inside AND2 / OR2) the node sits a few mV off its rail and
that offset is not modelled. Each transistor is then ON or OFF by its gate level and
type, which selects one of the four models.

Leakage. Let H be a set of nodes whose only voltage sources are the sources whose
current we want (the VDD supply, plus the inputs held at VDD for the total leakage).
By Kirchhoff, the current those sources deliver equals the net current flowing out of H,
which is the sum over all transistors of the currents flowing into them at terminals
in H. Any such H gives the same exact answer. We grow H from the sources through
every ON transistor's channel, so the sum is carried by the OFF transistors and gate
oxides on its boundary (the actual leakage paths) rather than by small differences of
large ON-device currents. Ground can never join H in a static CMOS circuit (checked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .gates import GND, VDD, Fet

CASES = ("nmos_on", "nmos_off", "pmos_on", "pmos_off")


def load_models(models_dir: str = "models") -> dict:
    """The four transistor models (FetModel or FetEnsemble) from a models folder."""
    import joblib
    return {c: joblib.load(config.path(models_dir, f"{c}.joblib")) for c in CASES}


class Circuit:
    """One circuit, one input vector, n corners. V maps node -> voltage array (n,)."""

    def __init__(self, fets: list[Fet], corners: pd.DataFrame, V: dict[str, np.ndarray],
                 high_pins: list[str]):
        self.fets = fets
        self.corners = corners.reset_index(drop=True)
        self.n = len(corners)
        self.vdd = self.corners.vdd.to_numpy(float)
        self.V = {VDD: self.vdd, GND: np.zeros(self.n), **{k: np.asarray(v, float) for k, v in V.items()}}
        self.high_pins = list(high_pins)
        # logic level of every node, fixed once from the given voltages: it is the
        # circuit's logic state and must not change while internal nodes are solved
        self.levels = {k: v > 0.5 * self.vdd for k, v in self.V.items()}

    # ---- logic -------------------------------------------------------------------------
    def is_on(self, f: Fet) -> np.ndarray:
        g = self.levels[f.g]
        return g if f.type == "n" else ~g

    # ---- transistor inputs and currents ----------------------------------------------------
    def fet_inputs(self, f: Fet, V: dict | None = None) -> pd.DataFrame:
        V = V or self.V
        t = f.type
        names = config.load()["sweep"]["process_params"]
        gate_hi = self.levels[f.g]
        d = dict(vdd=self.vdd, temp=self.corners.temp.to_numpy(float), w=np.full(self.n, f.w),
                 vd=V[f.d], vs=V[f.s], vg=np.where(gate_hi, self.vdd, 0.0),
                 vb=self.vdd if t == "p" else np.zeros(self.n))
        for p in names:
            d[p] = self.corners[f"{p}_{t}"].to_numpy(float)
        return pd.DataFrame(d)

    def currents(self, predict, fets: list[Fet] | None = None, V: dict | None = None) -> np.ndarray:
        """Currents INTO each transistor terminal (d, g, s, b): shape (n, n_fets, 4).

        predict(case, inputs_df) -> I(Vx) array (n, 4) in the dataset convention."""
        fets = fets if fets is not None else self.fets
        V = V or self.V
        J = np.zeros((self.n, len(fets), 4))
        # one predict call per (case) over all transistors that fall in it
        blocks: dict[str, list] = {c: [] for c in CASES}
        for k, f in enumerate(fets):
            inp = self.fet_inputs(f, V)
            pre = "nmos" if f.type == "n" else "pmos"
            case = np.where(self.is_on(f), f"{pre}_on", f"{pre}_off")
            for c in np.unique(case):
                rows = np.flatnonzero(case == c)
                blocks[str(c)].append((k, rows, inp.iloc[rows]))
        for c, items in blocks.items():
            if not items:
                continue
            big = pd.concat([it[2] for it in items], ignore_index=True)
            I = predict(c, big)
            pos = 0
            for k, rows, part in items:
                J[rows, k, :] = -I[pos:pos + len(part)]
                pos += len(part)
        return J

    # ---- leakage ----------------------------------------------------------------------------
    def high_set(self, seeds: list[str]) -> dict[str, np.ndarray]:
        nodes = {n for f in self.fets for n in (f.d, f.g, f.s, f.b)} | {VDD, GND}
        H = {n: np.zeros(self.n, bool) for n in nodes}
        for s in seeds:
            H[s][:] = True
        changed = True
        while changed:
            changed = False
            for f in self.fets:
                on = self.is_on(f)
                for a, b in ((f.d, f.s), (f.s, f.d)):
                    if b == GND:
                        continue
                    add = on & H[a] & ~H[b]
                    if add.any():
                        H[b] |= add
                        changed = True
        if H[GND].any():
            raise RuntimeError("ground joined the high set: the circuit has a VDD-to-ground ON path")
        return H

    def leakage(self, J: np.ndarray, seeds: list[str]) -> np.ndarray:
        H = self.high_set(seeds)
        total = np.zeros(self.n)
        for k, f in enumerate(self.fets):
            for t, node in enumerate((f.d, f.g, f.s, f.b)):
                total += np.where(H[node], J[:, k, t], 0.0)
        return total

    def leak_total(self, J):
        return self.leakage(J, [VDD] + self.high_pins)

    def leak_vdd(self, J):
        return self.leakage(J, [VDD])

    # ---- internal nodes solved with the transistor models ---------------------------------
    def _net_current(self, predict, node: str, V: dict) -> np.ndarray:
        """Net current from `node` into every transistor terminal on it (Kirchhoff residual)."""
        fets = [f for f in self.fets if node in (f.d, f.g, f.s, f.b)]
        J = self.currents(predict, fets, V)
        net = np.zeros(self.n)
        for k, f in enumerate(fets):
            for t, nd in enumerate((f.d, f.g, f.s, f.b)):
                if nd == node:
                    net += J[:, k, t]
        return net

    def _bisect(self, fn, iters: int) -> np.ndarray:
        """Root of fn(v) on [-0.05, 1.05] x VDD per corner; fn is monotonic in v."""
        lo, hi = -0.05 * self.vdd, 1.05 * self.vdd
        inc = fn(hi) >= fn(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            up = (fn(mid) < 0) == inc
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        return 0.5 * (lo + hi)

    def solve_nodes(self, predict, unknown: list[str], iters: int = 30) -> dict[str, np.ndarray]:
        """Internal node voltages from Kirchhoff with the transistor models: the net current
        into all transistor terminals on each unknown node is zero (gate terminals
        included). Every other node keeps its given voltage. One unknown: bisection.
        Two unknowns (AND2 / OR2): nested bisection - for each trial value of the first,
        the second is solved exactly - which converges even when the two nodes are tied
        together by an ON transistor (one-node-at-a-time sweeps do not, in practice)."""
        V = dict(self.V)
        if len(unknown) == 1:
            u = unknown[0]
            V[u] = self._bisect(lambda v: self._net_current(predict, u, {**V, u: v}), iters)
            return {u: V[u]}
        if len(unknown) == 2:
            u1, u2 = unknown
            cache = {}

            def inner(v1):
                v2 = self._bisect(lambda v: self._net_current(predict, u2, {**V, u1: v1, u2: v}), iters)
                cache["v2"] = v2
                return v2

            def outer(v1):
                v2 = inner(v1)
                return self._net_current(predict, u1, {**V, u1: v1, u2: v2})

            v1 = self._bisect(outer, iters)
            return {u1: v1, u2: inner(v1)}
        raise ValueError("solve_nodes handles at most two unknown nodes")

    # ---- from the teammate's implementation ------------------------------------------------
    def power_terms(self, J: np.ndarray, V: dict | None = None) -> dict[str, np.ndarray]:
        """Static power balance divided by VDD, split by mechanism:
            I = (1/VDD) * sum_devices [ Jd (Vd - Vs) + Jg (Vg - Vs) + Jb (Vb - Vs) ]
        (J = current into each terminal; uses Jd + Jg + Js + Jb = 0 per device). With node
        voltages and currents that satisfy Kirchhoff at every node, and every source at
        exactly 0 or VDD, this equals the current delivered by the sources at VDD (the total
        leakage) exactly. The three parts are the channel, gate-tunnelling and body/junction
        shares of the leakage."""
        V = V or self.V
        parts = {"channel": np.zeros(self.n), "gate": np.zeros(self.n), "body": np.zeros(self.n)}
        for k, f in enumerate(self.fets):
            vs = V[f.s]
            parts["channel"] += J[:, k, 0] * (V[f.d] - vs)
            parts["gate"] += J[:, k, 1] * (V[f.g] - vs)
            parts["body"] += J[:, k, 3] * (V[f.b] - vs)
        parts = {k: v / self.vdd for k, v in parts.items()}
        parts["total"] = parts["channel"] + parts["gate"] + parts["body"]
        return parts

    def solve_all_nodes(self, predict, internal: list[str], out: str = "out", passes: int = 2) -> dict[str, np.ndarray]:
        """Fully SPICE-free gate: the output AND the internal nodes from Kirchhoff with the
        transistor models (starting from the output at its logic rail). One or two unknowns
        in total are solved jointly (nested bisection); with three (AND2 / OR2: x, nand_out /
        nor_out, out) the internal pair and the output are solved alternately, `passes` times;
        the output is only weakly coupled to them (through the inverter's gate current)."""
        unknown = list(internal) + [out]
        if len(unknown) <= 2:
            return self.solve_nodes(predict, unknown)
        saved = dict(self.V)
        try:
            self.V[out] = np.where(self.levels[out], self.vdd, 0.0)
            sol: dict[str, np.ndarray] = {}
            for _ in range(passes):
                sol.update(self.solve_nodes(predict, list(internal)))
                self.V.update(sol)
                sol.update(self.solve_nodes(predict, [out]))
                self.V.update(sol)
            return {u: sol[u] for u in unknown}
        finally:
            self.V = saved
