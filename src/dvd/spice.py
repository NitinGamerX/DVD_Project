"""Run ngspice in batch mode and read back printed values."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Any of these in the ngspice log means the result cannot be trusted.
_FATAL = re.compile(
    r"(^\s*Error\b|simulation\(s\) aborted|singular matrix|Timestep too small|"
    r"iteration limit reached|Could not find include file|undefined (?:parameter|symbol)|"
    r"unknown parameter|cannot recover)",
    re.I | re.M,
)
_PRINT = re.compile(r"^\s*([\w#().+\-*/]+)\s*=\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*$", re.M)


class SpiceError(RuntimeError):
    pass


# ngspice (Ubuntu build) runs BSIM4 with OpenMP and picks its own thread count (2 by
# default), ignoring OMP_NUM_THREADS. With many ngspice processes at once the paired
# threads spin while waiting for each other and waste most of the CPU (measured on the
# project laptop: 16 simultaneous runs took 2.4 s and 55 s of CPU instead of 0.014 s).
# Our circuits are far too small to gain from threads inside one process; parallelism
# comes from running many processes. So every run gets one OpenMP thread (.spiceinit in
# its working folder) and passive waiting (environment) as a backstop.
_ENV = dict(os.environ, OMP_WAIT_POLICY="passive", OMP_NUM_THREADS="1")
_SPICEINIT = "* written by src/dvd/spice.py: one OpenMP thread per ngspice process\nset num_threads=1\n"


def _ensure_spiceinit(cwd: Path) -> None:
    f = Path(cwd) / ".spiceinit"
    if not f.exists():
        f.write_text(_SPICEINIT)
    elif "num_threads" not in f.read_text():
        with f.open("a") as fh:
            fh.write("\nset num_threads=1\n")


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
    _ensure_spiceinit(cwd)
    with tempfile.NamedTemporaryFile("w", suffix=".cir", dir=cwd, delete=False) as f:
        f.write(deck)
        deck_path = Path(f.name)
    try:
        r = subprocess.run([ngspice_binary(), "-b", str(deck_path)], cwd=cwd, env=_ENV,
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
