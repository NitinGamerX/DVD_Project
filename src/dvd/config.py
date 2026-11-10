"""Load config.yaml and resolve repository paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def path(*parts: str) -> Path:
    """Absolute path inside the repository."""
    return ROOT.joinpath(*parts)


def model_file() -> Path:
    return path(load()["assignment"]["model_file"])
