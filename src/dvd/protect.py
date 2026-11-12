"""Guard earlier results: fingerprint folders before a step runs and check them after."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import config


def fingerprint(folders: list[str]) -> dict[str, str]:
    """SHA-256 of every file in the given folders (relative paths); skips ngspice's .spiceinit."""
    fp = {}
    for p in folders:
        root = config.path(*Path(p).parts)
        if root.exists():
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.name != ".spiceinit":
                    fp[str(f.relative_to(config.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return fp


def check(before: dict[str, str], folders: list[str]) -> list[str]:
    """Files that changed, appeared or disappeared since `before`."""
    after = fingerprint(folders)
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def refuse_overlap(outputs: list[str], protected: list[str]) -> None:
    """Exit if any output folder is, contains, or lies inside a protected folder."""
    import sys
    for o in outputs:
        op = config.path(*Path(o).parts).resolve()
        for p in protected:
            pp = config.path(*Path(p).parts).resolve()
            if op == pp or pp in op.parents or op in pp.parents:
                sys.exit(f"refusing to run: output folder {o} overlaps protected folder {p}")
