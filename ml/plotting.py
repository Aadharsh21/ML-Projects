"""Shared chart style for the notebooks and README figures.

Thin marks, hairline grids, direct labels where they help, and a categorical
palette checked for colour-vision deficiency (blue, orange and aqua first).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .data import REPO_ROOT

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

INK = "#13233a"
INK_2 = "#44536a"
INK_3 = "#5b687d"
RULE = "#dce2ea"
GRAY = "#c2cad6"
LIGHT = "#e9edf2"

IMAGES = REPO_ROOT / "docs" / "images"


def style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "grid.color": RULE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "text.parse_math": False,  # dollar signs in labels stay dollar signs
    })


def save(fig: plt.Figure, name: str) -> Path:
    """Save a figure for the README under docs/images."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / name
    fig.savefig(path)
    return path


def dollars(x, _=None) -> str:
    """Tick formatter: $1,200 and -$300."""
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def label_bars(ax, bars, fmt: str = "{:.3f}", inside: bool = False, color: str = INK) -> None:
    """Value at the tip of each horizontal bar."""
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.annotate(fmt.format(w), (w, y), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9.5, color=color)


def australia(ax, **kwargs) -> None:
    """Draw the outline of Australia (from web/public/data/rain/australia.json)."""
    geo = json.loads((REPO_ROOT / "web" / "public" / "data" / "rain" / "australia.json").read_text())
    for feature in geo["features"]:
        if feature["properties"]["name"] != "Australia":
            continue
        for polygon in feature["geometry"]["coordinates"]:
            ring = np.array(polygon[0])
            ax.fill(ring[:, 0], ring[:, 1], color=kwargs.get("color", "#eef1f5"),
                    ec=kwargs.get("ec", GRAY), lw=0.6, zorder=0)
    ax.set_aspect(1 / np.cos(np.deg2rad(-27)))
    ax.set_xlim(112, 155)
    ax.set_ylim(-44.5, -10)
    ax.axis("off")
