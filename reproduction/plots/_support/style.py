"""Shared semantic palette and rendering style for StackelbergPOMDP figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


DEEP_PALETTE = (
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
    "#CCB974",  # yellow
    "#64B5CD",  # cyan
)


METHOD_COLORS = {
    "MSPM": DEEP_PALETTE[0],
    "SPM": DEEP_PALETTE[1],
}


LINE_WIDTH = 1.8
LEGEND_LOCATION = "lower right"


def configure_paper_style() -> None:
    """Apply the paper-wide Seaborn-deep visual language without Seaborn."""
    mpl.rcParams.update(
        {
            "axes.axisbelow": True,
            "axes.edgecolor": "white",
            "axes.facecolor": "#EAEAF2",
            "axes.grid": True,
            "axes.grid.axis": "both",
            "axes.grid.which": "major",
            "axes.labelcolor": "#262626",
            "axes.linewidth": 0.8,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "axes.prop_cycle": mpl.cycler(color=DEEP_PALETTE),
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "figure.dpi": 120,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 12,
            "grid.alpha": 1.0,
            "grid.color": "white",
            "grid.linewidth": 0.8,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
            "legend.loc": LEGEND_LOCATION,
            "legend.title_fontsize": 11,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.04,
            "text.color": "#262626",
            "xtick.direction": "out",
            "xtick.color": "#262626",
            "xtick.labelsize": 11,
            "ytick.direction": "out",
            "ytick.color": "#262626",
            "ytick.labelsize": 11,
        }
    )


def despine(ax: plt.Axes) -> None:
    """Remove the nonessential top and right borders."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_paper_figure(
    fig: plt.Figure,
    output_prefix: Path,
    *,
    pdf_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save the vector PDF used by the manuscript."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        format="pdf",
        facecolor="white",
        metadata=pdf_metadata,
    )
    return pdf_path


# Preserve the scale and uncertainty bands of the released Atari panels.
ATARI_THREE_PANEL_FIGSIZE = (6.0 / 0.48, 3.5)
ATARI_LINE_WIDTH = 1.5
ATARI_MARKER_SIZE = 6.0
ATARI_STANDARD_ERROR_ALPHA = 0.25


def configure_atari_style() -> None:
    """Use the released Atari panels' Matplotlib/Seaborn defaults."""
    plt.rcdefaults()
    sns.set_theme()
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def configure_mspm_style() -> None:
    """Retain the released MSPM figure's exact rendering settings."""
    mpl.rcdefaults()
    mpl.rcParams.update({
        "axes.axisbelow": True, "axes.edgecolor": "white",
        "axes.facecolor": "#EAEAF2", "axes.grid": True,
        "axes.grid.axis": "both", "axes.grid.which": "major",
        "axes.labelcolor": "#262626", "axes.labelsize": 13,
        "axes.linewidth": 0.8, "axes.spines.right": False,
        "axes.spines.top": False, "figure.facecolor": "white",
        "figure.dpi": 100, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 12, "grid.color": "white", "grid.linewidth": 0.8,
        "legend.fontsize": 11, "legend.frameon": False,
        "legend.handlelength": 2.4, "lines.linewidth": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "savefig.pad_inches": 0.04,
        "text.color": "#262626", "xtick.color": "#262626",
        "xtick.labelsize": 11, "ytick.color": "#262626", "ytick.labelsize": 11,
    })
