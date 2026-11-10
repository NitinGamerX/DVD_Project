"""Run ngspice in batch mode and read back printed values."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Any of these in the ngspice log means the result cannot be trusted.
_FATAL = re.compile(
    r"(^Error\b|\berror:|singular matrix|Timestep too small|"
    r"iteration limit reached|no convergence|gmin stepping failed|"
    r"source stepping failed|Could not find include file)",
    re.I | re.M,
)
_PRINT = re.compile(r"^\s*([\w#().+\-*/]+)\s*=\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*$", re.M)


class SpiceError(RuntimeError):
    pass


def ngspice_binary() -> str:
    exe = shutil.which("ngspice")
    if exe is None:
        raise SpiceError("ngspice not found on PATH (run ./setup.sh)")
    return exe


def version() -> str:
    out = subprocess.run([ngspice_binary(), "--version"], capture_output=True, text=True).stdout
    m = re.search(r"ngspice-(\S+)", out)
    return m.group(1) if m else out.strip()


def run_deck(deck: str, cwd: Path, timeout: float = 600.0) -> str:
    """Run one deck with `ngspice -b`; return stdout or raise SpiceError."""
    cwd = Path(cwd)
    with tempfile.NamedTemporaryFile("w", suffix=".cir", dir=cwd, delete=False) as f:
        f.write(deck)
        deck_path = Path(f.name)
    try:
        r = subprocess.run([ngspice_binary(), "-b", str(deck_path)], cwd=cwd,
                           capture_output=True, text=True, timeout=timeout)
    finally:
        deck_path.unlink(missing_ok=True)
    log = r.stdout + r.stderr
    bad = _FATAL.search(log)
    if r.returncode != 0 or bad:
        reason = bad.group(0) if bad else "exit code %d" % r.returncode
        raise SpiceError("ngspice failed (%s)\n%s" % (reason, log[-3000:]))
    return r.stdout


def printed(stdout: str) -> dict[str, float]:
    """Parse `print a b c` output lines of the form `name = value`."""
    return {k.lower(): float(v) for k, v in _PRINT.findall(stdout)}
