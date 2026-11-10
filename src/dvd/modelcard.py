"""Read and rewrite the assigned BSIM4 model file.

The original file is never modified. For process variation we render a copy in
which the varied parameters (vth0, toxe, ndep, u0, xj) are replaced either by
SPICE parameters ({vth0_n}, ...) or by literal numbers. Both renderings come from
the same substitution routine, so any card produced here differs from the
original only in those values.
"""
from __future__ import annotations

import re
from pathlib import Path

_NUM = r"[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?"
_MODEL_RE = re.compile(r"^\s*\.model\s+(\S+)\s+(nmos|pmos)\b", re.I | re.M)
_PARAM_RE = re.compile(rf"([A-Za-z_]\w*)\s*=\s*({_NUM})")
TYPE_OF = {"nmos": "n", "pmos": "p"}


def _cards(text: str) -> list[tuple[str, str, int, int]]:
    """(model name, 'n'/'p', start, end) for each .model card in the text."""
    heads = list(_MODEL_RE.finditer(text))
    if not heads:
        raise ValueError("no .model cards found")
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((m.group(1).lower(), TYPE_OF[m.group(2).lower()], m.start(), end))
    return out


def parse(model_path: Path) -> dict[str, dict[str, float]]:
    """Return {model_name: {param: value}} for every .model card in the file."""
    text = Path(model_path).read_text()
    cards: dict[str, dict[str, float]] = {}
    for name, _t, start, end in _cards(text):
        params: dict[str, float] = {}
        for line in text[start:end].splitlines():
            if line.lstrip().startswith("*"):
                continue
            for pname, val in _PARAM_RE.findall(line):
                params[pname.lower()] = float(val)
        cards[name] = params
    return cards


def nominals(model_path: Path, model_name: str, names: list[str]) -> dict[str, float]:
    card = parse(model_path)[model_name.lower()]
    missing = [n for n in names if n not in card]
    if missing:
        raise KeyError(f"{model_name}: parameters not in model file: {missing}")
    return {n: card[n] for n in names}


def render(model_path: Path, names: list[str],
           values: dict[str, dict[str, float]] | None = None) -> str:
    """Rewrite the varied parameters of every card.

    values=None  -> parametrised: `vth0 = {vth0_n}` in the NMOS card, `{vth0_p}` in PMOS.
    values given -> literal: values['n']['vth0'] etc. written as numbers.
    Every other character of the file is preserved.
    """
    text = Path(model_path).read_text()
    pieces, pos = [], 0
    for _name, t, start, end in _cards(text):
        card = text[start:end]
        for p in names:
            pat = re.compile(rf"(?<![\w])({re.escape(p)}\s*=\s*)({_NUM})", re.I)
            hits = pat.findall(card)
            if len(hits) != 1:
                raise ValueError(f"parameter {p} appears {len(hits)} times in the {t}-card")
            repl = "{%s_%s}" % (p, t) if values is None else repr(float(values[t][p]))
            card = pat.sub(lambda m, r=repl: m.group(1) + r, card, count=1)
        pieces += [text[pos:start], card]
        pos = end
    pieces.append(text[pos:])
    return "".join(pieces)
