"""Read parameter values straight from the assigned BSIM4 model file."""
from __future__ import annotations

import re
from pathlib import Path

_MODEL_RE = re.compile(r"^\s*\.model\s+(\S+)\s+(nmos|pmos)\b", re.I | re.M)
_PARAM_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)")


def parse(model_path: Path) -> dict[str, dict[str, float]]:
    """Return {model_name: {param: value}} for every .model card in the file."""
    text = Path(model_path).read_text()
    heads = list(_MODEL_RE.finditer(text))
    if not heads:
        raise ValueError(f"no .model cards found in {model_path}")
    cards: dict[str, dict[str, float]] = {}
    for i, m in enumerate(heads):
        body = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        params: dict[str, float] = {}
        for line in body.splitlines():
            line = line.split("*", 1)[0] if line.lstrip().startswith("*") else line
            for name, val in _PARAM_RE.findall(line):
                params[name.lower()] = float(val)
        cards[m.group(1).lower()] = params
    return cards


def nominals(model_path: Path, model_name: str, names: list[str]) -> dict[str, float]:
    card = parse(model_path)[model_name.lower()]
    missing = [n for n in names if n not in card]
    if missing:
        raise KeyError(f"{model_name}: parameters not in model file: {missing}")
    return {n: card[n] for n in names}
