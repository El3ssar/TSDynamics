"""Shared plotting utilities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotConfig:
    """Lightweight plot configuration."""

    figsize: Tuple[float, float] = (6.0, 4.0)
    dpi: int = 120
    tight_layout: bool = True
    facecolor: Optional[str] = None


def new_fig_ax(
    nrows: int = 1,
    ncols: int = 1,
    *,
    cfg: Optional[PlotConfig] = None,
    projection: Optional[str] = None,
):
    """Create a Matplotlib figure and axes with consistent defaults."""
    cfg = cfg or PlotConfig()
    fig_kw = {"figsize": cfg.figsize, "dpi": cfg.dpi}
    if cfg.facecolor is not None:
        fig_kw["facecolor"] = cfg.facecolor
    if projection and (nrows, ncols) == (1, 1):
        fig = plt.figure(**fig_kw)
        ax = fig.add_subplot(111, projection=projection)
        if cfg.facecolor is not None:
            ax.set_facecolor(cfg.facecolor)
        if cfg.tight_layout:
            fig.tight_layout()
        return fig, ax
    subplot_kw = {"projection": projection} if projection else None
    if cfg.facecolor is not None:
        subplot_kw = subplot_kw or {}
        subplot_kw["facecolor"] = cfg.facecolor
    fig, axes = plt.subplots(nrows, ncols, **fig_kw, subplot_kw=subplot_kw)
    if cfg.tight_layout:
        fig.tight_layout()
    return fig, axes
