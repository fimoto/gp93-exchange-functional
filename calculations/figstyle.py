"""Common figure style: plain line drawings in the manner of classic gnuplot plots.

Every figure is drawn at the width at which the manuscript includes it
(Physical Review two-column layout: one column 3.4 in, full page 7.0 in), so
that LaTeX does not rescale it and the printed text sizes are the ones set
here.  The style uses a Helvetica-like sans-serif font, inward ticks on all
four sides, frameless keys with the line sample to the right of the label,
and a small set of saturated colors that remain distinguishable in gray
through their line styles.

The preferred font is Liberation Sans, a TrueType font metrically
compatible with Arial and close to Helvetica, which embeds cleanly as a
Type 42 font in PDF.  If it is absent, Arial, Helvetica, Nimbus Sans, and
DejaVu Sans are tried in that order; the figures are then equivalent but not
byte-identical.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager

COLUMN_IN = 3.4
PAGE_IN = 7.0

FONT_SIZE = {"label": 8.5, "tick": 7.5, "legend": 7.5, "annotation": 7.5, "panel": 8.5}
LINE_WIDTH = {"axis": 0.6, "main": 1.5, "secondary": 1.1, "thin": 0.7}

# Saturated colors with distinct line styles (gnuplot-like).
RED = "#e8000b"
BLUE = "#0000e0"
GREEN = "#00a000"
MAGENTA = "#c000c0"
BLACK = "#000000"
GRAY = "#808080"

STYLE = {
    "GP93": {"color": RED, "ls": "-", "lw": LINE_WIDTH["main"]},
    "GP93R": {"color": BLUE, "ls": (0, (5, 2)), "lw": LINE_WIDTH["main"]},
    "PBE": {"color": GREEN, "ls": (0, (6, 2, 1.5, 2)), "lw": LINE_WIDTH["secondary"]},
    "bare": {"color": MAGENTA, "ls": (0, (1.2, 1.5)), "lw": LINE_WIDTH["secondary"]},
    "exact": {"color": BLACK, "ls": "-", "lw": LINE_WIDTH["thin"]},
}

_PREFERRED = ("Liberation Sans", "Arial", "Helvetica", "Nimbus Sans",
              "DejaVu Sans")


def _font_family() -> str:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for name in _PREFERRED:
        if name in available:
            return name
    return "DejaVu Sans"


def apply() -> str:
    """Set the global Matplotlib parameters and return the font family used."""

    family = _font_family()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family],
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family,
        "mathtext.cal": family,
        "mathtext.tt": family,
        "axes.unicode_minus": True,
        "font.size": FONT_SIZE["label"],
        "axes.labelsize": FONT_SIZE["label"],
        "axes.titlesize": FONT_SIZE["label"],
        "xtick.labelsize": FONT_SIZE["tick"],
        "ytick.labelsize": FONT_SIZE["tick"],
        "legend.fontsize": FONT_SIZE["legend"],
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.25,
        "axes.linewidth": LINE_WIDTH["axis"],
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": LINE_WIDTH["axis"],
        "ytick.major.width": LINE_WIDTH["axis"],
        "xtick.minor.width": LINE_WIDTH["axis"],
        "ytick.minor.width": LINE_WIDTH["axis"],
        "lines.solid_capstyle": "butt",
        "lines.dash_capstyle": "butt",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    })
    return family


def panel_label(ax, text: str, x: float = 0.035, y: float = 0.955) -> None:
    """Panel label such as (a), inside the upper-left corner of the axes."""

    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_SIZE["panel"])


def save(fig, path_without_suffix) -> tuple:
    """Save at the exact figure size as PDF (for the manuscript) and PNG (preview)."""

    pdf = path_without_suffix.with_suffix(".pdf")
    png = path_without_suffix.with_suffix(".png")
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300)
    return pdf, png
