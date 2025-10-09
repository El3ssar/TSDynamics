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


def new_fig_ax(
    nrows: int = 1,
    ncols: int = 1,
    *,
    cfg: Optional[PlotConfig] = None,
    projection: Optional[str] = None,
):
    """Create a Matplotlib figure and axes with consistent defaults."""
    cfg = cfg or PlotConfig()
    if projection and (nrows, ncols) == (1, 1):
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        ax = fig.add_subplot(111, projection=projection)
        if cfg.tight_layout:
            fig.tight_layout()
        return fig, ax
    fig, axes = plt.subplots(
        nrows, ncols, figsize=cfg.figsize, dpi=cfg.dpi,
        subplot_kw={"projection": projection} if projection else None
    )
    if cfg.tight_layout:
        fig.tight_layout()
    return fig, axes
