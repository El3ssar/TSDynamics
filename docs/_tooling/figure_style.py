"""Documentation illustration styles, kept separate from library defaults.

Analytical figures use opaque white paper and strong, portable sans-serif text.
Geometry-only system previews use a dark stage, matching their interactive view.
Every caller scopes these settings with ``matplotlib.rc_context``.
"""

from __future__ import annotations

PAPER = "#ffffff"
INK = "#273947"
MUTED = "#526675"
RULE = "#b8c6cf"
TEAL = "#11857A"
INDIGO = "#574FCF"
AMBER = "#B76D18"
ROSE = "#D64562"
STAGE = "#0B0F14"
BRIGHT_TEAL = "#47D7BF"
PALETTE = (TEAL, INDIGO, AMBER, ROSE, "#3F809A")

PAPER_RC = {
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titlepad": 12,
    "axes.labelpad": 7,
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": PAPER,
    "axes.grid": False,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.transparent": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "text.color": INK,
    "grid.color": "#e4ebef",
    "grid.linewidth": 0.6,
}


def paper_plot(plot, *, font_size=11):
    """Apply an explicit, local library theme to a documentation Plot."""
    return plot.theme(
        "minimal",
        background=PAPER,
        foreground=INK,
        font_family="sans-serif",
        font_size=font_size,
        title_size=font_size + 1.5,
        palette=PALETTE,
        line_width=1.3,
    )


def finish_figure(fig):
    """Polish manual Matplotlib axes without changing data, scales or limits."""
    from matplotlib.colors import to_rgba
    from matplotlib.text import Text

    # Replace old low-contrast generator ink; retain purposeful semantic colours.
    old_grey = to_rgba("#888888")
    for text in fig.findobj(Text):
        if to_rgba(text.get_color()) == old_grey:
            text.set_color(MUTED)
        if text.get_fontsize() < 9:
            text.set_fontsize(9)
    for ax in fig.axes:
        if ax.name != "3d":
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(RULE)
        ax.title.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
    fig.set_facecolor(PAPER)
    return fig


def save_figure(fig, path, **kwargs):
    """Export crisp, opaque analytical artwork with reproducible SVG metadata."""
    finish_figure(fig)
    kwargs.setdefault("dpi", 150)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.16)
    kwargs.setdefault("transparent", False)
    kwargs.setdefault("metadata", {"Date": None})
    fig.savefig(path, **kwargs)


def dark_portrait(fig, ax):
    """Give a geometry-only preview an uncluttered, opaque dark stage."""
    fig.set_facecolor(STAGE)
    ax.set_facecolor(STAGE)
    ax.set_axis_off()
    ax.margins(0.04)


class FigureCanvas:
    """A tiny local drawing workspace, without pyplot's global backend or figures.

    Manual docs illustrations need only figure/subplots/subplot/close. Building
    bare Agg figures avoids changing a notebook's selected interactive backend,
    and ``close('all')`` clears only this workspace, never the caller's windows.
    """

    def __init__(self):
        import matplotlib as mpl

        self.rcParams = mpl.rcParams
        self._figures = []

    def figure(self, **kwargs):
        """Create a locally owned figure with its own noninteractive canvas."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(**kwargs)
        FigureCanvasAgg(fig)
        self._figures.append(fig)
        return fig

    def subplots(self, nrows=1, ncols=1, *, figsize=None, layout=None, **kwargs):
        """Create an axes arrangement on a locally owned figure."""
        fig = self.figure(figsize=figsize, layout=layout)
        return fig, fig.subplots(nrows, ncols, **kwargs)

    def gcf(self):
        """Return the latest local figure, creating one when needed."""
        return self._figures[-1] if self._figures else self.figure()

    def subplot(self, *args, **kwargs):
        """Add a subplot to the current local figure."""
        return self.gcf().add_subplot(*args, **kwargs)

    def close(self, figure="all"):
        """Clear owned figures, or one explicit bare figure, without pyplot."""
        if isinstance(figure, str) and figure == "all":
            for fig in self._figures:
                self._clear(fig)
            self._figures.clear()
        else:
            self._clear(figure)
            self._figures = [fig for fig in self._figures if fig is not figure]

    @staticmethod
    def _clear(fig):
        # Matplotlib's clear resets a shared axis to (0, 1), which warns if a
        # sibling still has a logarithmic scale. These figures are being disposed
        # after export; reset their scales together before releasing the artists.
        for ax in fig.axes:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        fig.clear()
