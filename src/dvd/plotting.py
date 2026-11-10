"""Common figure style for all project plots (headless backend for WSL)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

CASE_LABEL = {"nmos_on": "NMOS ON", "nmos_off": "NMOS OFF", "pmos_on": "PMOS ON", "pmos_off": "PMOS OFF"}
CASE_COLOR = {"nmos_on": "#1f6fb4", "nmos_off": "#4ba3e3", "pmos_on": "#c0392b", "pmos_off": "#e8846f"}

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


def save(fig, path: Path) -> Path:
    """Save with the format and resolution from config.yaml; returns the written path."""
    rep = config.load()["reporting"]
    path = Path(path).with_suffix("." + rep["plot_format"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=rep["plot_dpi"])
    plt.close(fig)
    return path
