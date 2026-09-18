"""The matplotlib 2-D reference renderer core (stream VIZ-MPL-CORE).

This module turns a backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into a
:class:`matplotlib.figure.Figure` using matplotlib's **object-oriented** API only
— :class:`matplotlib.figure.Figure` with the Agg canvas.  It imports **no**
``matplotlib.pyplot``: every figure is built explicitly so the renderer is
stateless, thread-friendly, and free of pyplot's implicit global figure manager.

The renderer is the *conformance oracle* for the visualization seam: its
:class:`~tsdynamics.viz.render.caps.RendererCapabilities` declare *all* kinds, so
dispatch falls back to it whenever a partial backend (plotly / json / three.js)
declines a spec.  Every 2-D :class:`~tsdynamics.viz.spec.PlotKind` must draw here
without error.

Theme and style application
---------------------------
The renderer follows the three-step contract from the design spec (§3):

1. **Resolve the theme** (``spec.theme or get_theme(None)``) and apply it
   figure-locally: background facecolor, default color cycle (from
   ``theme.palette``), font family + size via per-artist kwargs (no global
   ``rcParams`` mutation — the mutation is scoped to the figure), default
   ``line_width`` / ``marker_size``, and default grid on/off.
2. **Per-layer style**: ``normalize_style(layer.style, warn=False)`` (the
   dispatcher already warned), then translate canonical keys to mpl kwargs,
   OVERRIDING theme defaults.
3. **Honor enriched dataclass fields**: :class:`~tsdynamics.viz.spec.Axis` (grid,
   color, label_size, tick_size, tick_rotation, tickformat via
   ``FormatStrFormatter``), :class:`~tsdynamics.viz.spec.Legend` (font_size, ncol,
   frame), :class:`~tsdynamics.viz.spec.Colorbar` (label_size), and ``zorder``.

The pieces
----------
- :data:`KIND_PRESETS` — semantic kind → axis/aspect/colorbar defaults.  A
  preset sets the figure's aspect, whether a colorbar is wanted, and the default
  colormap / norm for a colored kind.  The per-spec :class:`Axis` / :class:`Colorbar`
  always win over a preset (the spec is the source of truth).
- :data:`MARK_DISPATCH` — layer mark → drawing function.  Each draws one
  :class:`~tsdynamics.viz.spec.Layer` onto an :class:`~matplotlib.axes.Axes`,
  reading the layer's channel data (``x`` / ``y`` / ``c`` / ``lo`` / ``hi`` /
  ``err`` / ``cat`` / ``size`` / ``u`` / ``v``) and neutral style keys.
- :func:`render` — the entry point: build a :class:`Figure`, normalise the
  spec's semantic kind, draw every layer, apply axes / colorbar / legend /
  annotations, and return the figure.

3-D marks (``LINE3D`` / ``SURFACE3D``) are drawn by :mod:`._threed`.  This core
raises a clear :class:`NotImplementedError` for a 3-D mark so the capability
declaration (``supports_3d=True``) routes 3-D here while the drawing is filled in
by that module.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from tsdynamics.errors import InvalidParameterError

from ..._visibility import listing_dir
from ...producers import autostyle_enabled, autostyle_line, autostyle_marker
from ...spec import Annotation, Axis, Colorbar, Layer, PlotKind, PlotSpec
from ...style import Theme, normalize_style
from .. import normalize_kind
from ..caps import RenderResult

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure


__all__ = ["KIND_PRESETS", "MARK_DISPATCH", "render"]

__dir__ = listing_dir(__all__)


# ---------------------------------------------------------------------------
# Kind presets
# ---------------------------------------------------------------------------


class _KindPreset:
    """Axis / aspect / colorbar defaults for one semantic :class:`PlotKind`.

    A preset captures the *presentation intent* of a semantic kind that is not
    already carried by the spec's typed :class:`~tsdynamics.viz.spec.Axis` /
    :class:`~tsdynamics.viz.spec.Colorbar` — chiefly the aspect ratio and the
    default colormap / norm for a colored image kind.  The spec's own axes and
    colorbar always override these defaults; a preset only fills a gap.

    Parameters
    ----------
    aspect : {"auto", "equal"}, optional
        Default aspect ratio when the spec leaves :attr:`PlotSpec.aspect` at its
        ``"auto"`` default.  Phase portraits / sections / images want
        ``"equal"``.
    cmap : str, optional
        Default colormap for the kind's color channel, used only when neither the
        spec's :class:`~tsdynamics.viz.spec.Colorbar` nor the layer style sets
        one.  ``None`` lets matplotlib pick.
    norm : {"linear", "log", "symlog"}, optional
        Default color norm, used only when the spec's colorbar leaves it unset.
    """

    __slots__ = ("aspect", "cmap", "norm")

    def __init__(
        self,
        *,
        aspect: str = "auto",
        cmap: str | None = None,
        norm: str | None = None,
    ) -> None:
        self.aspect = aspect
        self.cmap = cmap
        self.norm = norm


#: Semantic kind → presentation preset.  A kind absent from the table renders
#: with the neutral default preset (``"auto"`` aspect, backend-default colors),
#: so the renderer never trips over a kind it has no special-casing for.
KIND_PRESETS: dict[PlotKind, _KindPreset] = {
    # equal-aspect geometric kinds
    PlotKind.PHASE_PORTRAIT_2D: _KindPreset(aspect="equal"),
    PlotKind.PHASE_PORTRAIT_FIELD: _KindPreset(aspect="equal"),
    PlotKind.POINCARE_SECTION: _KindPreset(aspect="equal"),
    PlotKind.VECTOR_FIELD: _KindPreset(aspect="equal"),
    PlotKind.EIGENVALUE_PLANE: _KindPreset(aspect="equal"),
    PlotKind.FIXED_POINTS_OVERLAY: _KindPreset(aspect="equal"),
    # image kinds with a color channel
    PlotKind.RECURRENCE_PLOT: _KindPreset(aspect="equal", cmap="binary"),
    PlotKind.BASINS_IMAGE: _KindPreset(aspect="equal", cmap="tab20"),
    PlotKind.IMAGE: _KindPreset(cmap="viridis"),
    PlotKind.SPACETIME: _KindPreset(cmap="viridis"),
    # a 2-D spatial field is a viridis heatmap (its equal aspect rides on the
    # spec, set by the producer for the 2-D case only); a 1-D field is a plain
    # auto-aspect line, which ignores the cmap.  See stream VIZ-SPATIAL-FIELD.
    PlotKind.SPATIAL_FIELD: _KindPreset(cmap="viridis"),
}

_DEFAULT_PRESET = _KindPreset()


def _preset_for(kind: PlotKind) -> _KindPreset:
    """Return the preset for ``kind`` (the neutral default if none is declared)."""
    return KIND_PRESETS.get(kind, _DEFAULT_PRESET)


# ---------------------------------------------------------------------------
# Theme application helpers
# ---------------------------------------------------------------------------


#: Mapping from canonical linestyle names (from normalize_style) to mpl spellings.
_LINESTYLE_MPL: dict[str, str] = {
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dashdot": "dashdot",
}

#: Mapping from canonical marker names (from normalize_style) to mpl spellings.
_MARKER_MPL: dict[str, str] = {
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
    "cross": "+",
    "x": "x",
    "star": "*",
    "none": "None",
}


def _resolve_theme(spec: PlotSpec) -> Theme:
    """Return the effective theme for ``spec`` (its own theme or the global default)."""
    return spec.resolved_theme


def _apply_theme_to_figure(fig: Any, ax: Any, theme: Theme) -> None:
    """Apply figure-level theme settings: background, font (figure-local, not rcParams).

    This is a **figure-local** mutation: we set facecolor on the figure and axes
    directly, and store font properties on the figure-level text objects.  We do
    NOT mutate global ``rcParams`` — that would bleed into other figures created in
    the same process.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    theme : Theme
        The resolved theme to apply.
    """
    if theme.background is not None:
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

    if theme.foreground is not None:
        # Color the spines, tick labels, and axis labels
        for spine in ax.spines.values():
            spine.set_edgecolor(theme.foreground)
        ax.tick_params(colors=theme.foreground, which="both")
        ax.xaxis.label.set_color(theme.foreground)
        ax.yaxis.label.set_color(theme.foreground)
        if ax.get_title():
            ax.title.set_color(theme.foreground)


#: Colours tried, in order, for an unstyled curve drawn ON TOP OF a field layer.
#: White then black covers a dark and a light backdrop; the remaining two give a
#: second and third curve something distinguishable.  Deliberately not palette
#: colours: a palette is chosen to separate curves *from each other*, which says
#: nothing about separating them from an image underneath.
_ON_FIELD_CYCLE: tuple[str, ...] = ("#ffffff", "#000000", "#ff3b30", "#ffcc00")


def _apply_theme_color_cycle(ax: Any, theme: Theme, spec: PlotSpec | None = None) -> None:
    """Set the axes colour cycle, avoiding a collision with a field layer.

    Normally the cycle is the theme's palette.  But when the spec draws a curve or
    markers *over* a field (a basin image, a recurrence plot, a spacetime image),
    the palette is the wrong source: ``basins_image`` renders through ``tab20``,
    whose first swatch is ``#1f77b4`` — byte-identical to the default palette's
    first colour.  An unstyled trajectory over basin 0 was therefore drawn in
    exactly the basin's own colour and was invisible, which is what the flagship
    composition call produces if nothing intervenes.

    So when a field layer is present the cycle switches to :data:`_ON_FIELD_CYCLE`,
    chosen for contrast against an arbitrary image rather than against other
    curves.  An explicit per-layer ``color`` always wins over either cycle, so this
    only ever decides what an *unstyled* overlay looks like.
    """
    palette = list(theme.palette)
    if spec is not None and _draws_over_a_field(spec):
        palette = [*_ON_FIELD_CYCLE, *palette]
    ax.set_prop_cycle(color=palette)


#: Layer *marks* that paint a backdrop the rest of the figure is drawn on top of.
#: Note this is a mark test, not a semantic-kind test: ``Layer.kind`` holds the
#: mark (``image``), while ``PlotSpec.kind`` holds the semantic kind
#: (``basins_image``), and a hand-built or composed spec may carry an image layer
#: under any semantic kind at all.
#:
#: ``QUIVER`` is deliberately **not** here.  A quiver paints no backdrop — the
#: page shows through between the arrows — so switching to the contrast cycle
#: makes an unstyled overlay **white on white**: the host orbit of
#: ``phase_portrait_field`` was drawn in ``#ffffff`` and was invisible on every
#: light page, legend entry and all.  The cycle exists for an image's colormap,
#: which is a thing a quiver does not have.
_FIELD_MARKS: frozenset[PlotKind] = frozenset({PlotKind.IMAGE})


def _draws_over_a_field(spec: PlotSpec) -> bool:
    """Report whether ``spec`` has a field layer AND something drawn on top of it."""
    marks = {layer.kind for layer in spec.layers}
    return bool(marks & _FIELD_MARKS) and bool(marks - _FIELD_MARKS)


def _apply_theme_grid(ax: Any, spec: PlotSpec, theme: Theme) -> None:
    """Apply grid visibility from Axis.grid (per-axis) falling back to theme.grid.

    Also applies theme grid_color / grid_alpha when a grid is shown.
    """
    x_grid = spec.x.grid if spec.x.grid is not None else theme.grid
    y_grid = spec.y.grid if spec.y.grid is not None else theme.grid

    grid_kw: dict[str, Any] = {}
    if theme.grid_color is not None:
        grid_kw["color"] = theme.grid_color
    if theme.grid_alpha is not None:
        grid_kw["alpha"] = theme.grid_alpha

    # matplotlib warns (and force-enables the grid) if line properties are
    # supplied while the grid is being turned off, so only pass grid_kw when the
    # grid is actually shown.
    if x_grid or y_grid:
        ax.grid(True, **grid_kw)
    else:
        ax.grid(False)
    if x_grid != y_grid:
        # Per-axis grid when they differ
        if x_grid:
            ax.grid(True, axis="x", **grid_kw)
        else:
            ax.grid(False, axis="x")
        if y_grid:
            ax.grid(True, axis="y", **grid_kw)
        else:
            ax.grid(False, axis="y")


# ---------------------------------------------------------------------------
# Figure construction (the one place a Figure is built)
# ---------------------------------------------------------------------------

#: The matplotlib layout engine every figure this backend produces is built with.
#:
#: Without one, matplotlib places axes on a fixed fractional grid and simply lets
#: decorations overflow: a 2x2 composite collides row-2 titles into row-1 tick
#: labels and **clips the row-1 x-axis labels off the artifact entirely** — the
#: label is not merely cramped, it is absent from the saved PNG.  A single-panel
#: figure with a long y-label or a colorbar loses text the same way.
#: ``"constrained"`` solves the layout instead of assuming it, and (unlike
#: ``tight_layout``) works with shared axes, 3-D axes and colorbars.
_LAYOUT_ENGINE: Literal["constrained", "compressed", "tight"] | None = "constrained"

#: Extensions that ``render(path=...)`` routes to an animation's own writer
#: (ffmpeg / pillow) rather than to ``Figure.savefig``.
_MOVIE_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".gif", ".webm", ".mov", ".m4v", ".apng"})


def new_figure(
    figsize: tuple[float, float] | None = None,
    dpi: float | None = None,
    layout: Literal["constrained", "compressed", "tight"] | None = None,
) -> Figure:
    """Build the backend's :class:`~matplotlib.figure.Figure` (Agg, constrained).

    Every figure this backend returns — single panel, 3-D, composite, animated —
    is built here, so the Agg canvas attachment and the :data:`_LAYOUT_ENGINE`
    are applied exactly once and cannot be forgotten at a new construction site.
    Uses matplotlib's object-oriented API only (no ``pyplot``).

    Parameters
    ----------
    figsize, dpi : optional
        Already-resolved geometry (see :func:`figure_geometry`); ``None`` leaves
        matplotlib's own default.
    layout : {"constrained", "compressed", "tight"}, optional
        The layout engine; ``None`` uses :data:`_LAYOUT_ENGINE`.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as _Figure

    fig = _Figure(figsize=figsize, dpi=dpi, layout=layout if layout is not None else _LAYOUT_ENGINE)
    FigureCanvasAgg(fig)
    return fig


def figure_geometry(
    spec: PlotSpec,
    figsize: tuple[float, float] | None = None,
    *,
    theme: Theme | None = None,
) -> tuple[
    tuple[float, float] | None, float | None, Literal["constrained", "compressed", "tight"] | None
]:
    """Resolve ``(figsize, dpi, layout_engine)`` for ``spec``, honouring the theme.

    Precedence, most specific first:

    1. the explicit ``figsize=`` render keyword,
    2. ``spec.meta["figsize"]`` / ``spec.meta["dpi"]`` (what ``spec.size(...)``
       and ``save(size=..., dpi=...)`` write),
    3. the resolved :class:`~tsdynamics.viz.style.Theme`'s :attr:`~tsdynamics.viz.style.Theme.figsize`
       / :attr:`~tsdynamics.viz.style.Theme.dpi` / :attr:`~tsdynamics.viz.style.Theme.layout_engine`,
    4. matplotlib's own defaults.

    Step 3 is the point of this helper.  ``Theme`` grew those three fields
    precisely so a theme could carry its output geometry — the ``"publication"``
    theme declares ``figsize=(5.0, 3.5)`` at ``dpi=300`` — but nothing read them,
    so ``spec.theme("publication").render()`` still produced matplotlib's default
    6.4x4.8 in at 100 dpi.  A documented theme field that silently does nothing is
    the failure mode this layer exists to eliminate, so every figure the backend
    builds resolves its geometry here.
    """
    theme = theme if theme is not None else _resolve_theme(spec)
    meta = spec.meta if isinstance(spec.meta, dict) else {}

    if figsize is None:
        meta_figsize = meta.get("figsize")
        if meta_figsize is not None:
            w, h = meta_figsize
            if w is not None and h is not None:
                figsize = (float(w), float(h))
    if figsize is None and theme.figsize is not None:
        figsize = (float(theme.figsize[0]), float(theme.figsize[1]))

    dpi: float | None = None
    if "dpi" in meta and meta["dpi"] is not None:
        dpi = float(meta["dpi"])
    elif theme.dpi is not None:
        dpi = float(theme.dpi)

    return figsize, dpi, theme.layout_engine


# ---------------------------------------------------------------------------
# Style coercion helpers
# ---------------------------------------------------------------------------


def _canon_style(
    layer: Layer,
    theme: Theme,
    *,
    n: int | None = None,
    autostyle: bool = True,
) -> dict[str, Any]:
    """Return a dict of mpl-ready style kwargs from the layer's canonical style + theme defaults.

    Calls ``normalize_style(warn=False)`` on the layer's raw style dict (the
    dispatcher already emitted any consolidated warning), then maps canonical keys
    to matplotlib-specific spellings.  Theme defaults for ``line_width`` /
    ``marker_size`` are used only when the layer carries no override.

    Parameters
    ----------
    layer : Layer
    theme : Theme
        The resolved theme (provides default line_width / marker_size).
    n : int, optional
        Sample count of the curve being drawn.  When given, the theme's
        ``line_width`` / full opacity defaults are resolved through
        :func:`~tsdynamics.viz.producers.autostyle_line` so a dense trajectory
        gets a thinner, slightly translucent stroke instead of a solid blob.
        **Only the defaults** are affected — an explicit ``linewidth`` / ``alpha``
        on the layer always wins.
    autostyle : bool, optional
        Whether density-aware resolution applies (the
        ``spec.meta["autostyle"] = False`` escape hatch).  Default ``True``.

    Returns
    -------
    dict
        Matplotlib artist kwargs (``color``, ``lw``, ``linestyle``, ``marker``,
        ``ms``, ``alpha``, ``zorder``, …).
    """
    canon = normalize_style(layer.style, warn=False)
    kw: dict[str, Any] = {}
    auto_lw, auto_alpha = (
        autostyle_line(n, line_width=theme.line_width, enabled=autostyle)
        if n is not None
        else (theme.line_width, None)
    )

    if "color" in canon:
        kw["color"] = canon["color"]
    if "alpha" in canon:
        kw["alpha"] = float(canon["alpha"])
    elif auto_alpha is not None:
        kw["alpha"] = auto_alpha
    if "zorder" in canon:
        kw["zorder"] = int(canon["zorder"])
    # NOTE: ``fill`` / ``fillalpha`` are AREA/ENSEMBLE_FAN-only knobs and are
    # consumed directly by ``_draw_area`` (which reads them off ``canon``).  They
    # are deliberately NOT forwarded here: a ``Line2D`` (``ax.plot``) and a
    # ``PathCollection`` (``ax.scatter``) both reject a ``fill=`` kwarg, so
    # leaking it crashes a fully-styled LINE/SCATTER layer.

    # linewidth: canonical key → mpl "linewidth" (density-resolved default)
    if "linewidth" in canon:
        kw["linewidth"] = float(canon["linewidth"])
    elif auto_lw is not None:
        kw["linewidth"] = float(auto_lw)

    # linestyle: canonical → mpl spelling
    if "linestyle" in canon:
        kw["linestyle"] = _LINESTYLE_MPL.get(str(canon["linestyle"]), canon["linestyle"])

    # marker: canonical → mpl spelling
    if "marker" in canon:
        kw["marker"] = _MARKER_MPL.get(str(canon["marker"]), canon["marker"])
        # ``filled=False`` → a hollow marker on a Line2D (the scatter path spells
        # the same thing with facecolors/edgecolors; see ``_draw_scatter``).
        if canon.get("filled") is False:
            kw["markerfacecolor"] = "none"

    # markersize: canonical key → mpl "markersize"
    if "markersize" in canon:
        kw["markersize"] = float(canon["markersize"])
    elif theme.marker_size is not None:
        kw["markersize"] = float(theme.marker_size)

    return kw


def _line_kwargs(
    layer: Layer, theme: Theme, *, n: int | None = None, autostyle: bool = True
) -> dict[str, Any]:
    """Collect matplotlib line kwargs from a layer's style + theme defaults."""
    kw = _canon_style(layer, theme, n=n, autostyle=autostyle)
    if layer.label is not None:
        kw["label"] = layer.label
    return kw


def _resolve_cmap(spec: PlotSpec, layer: Layer, preset: _KindPreset) -> str | None:
    """Pick the colormap: layer style > spec colorbar > kind preset > backend default."""
    canon = normalize_style(layer.style, warn=False)
    cmap = canon.get("cmap")
    if cmap is not None:
        return str(cmap)
    if spec.colorbar is not None and spec.colorbar.cmap is not None:
        return spec.colorbar.cmap
    return preset.cmap


def _resolve_norm(spec: PlotSpec, preset: _KindPreset) -> str | None:
    """Pick the color norm: spec colorbar > kind preset > linear (``None``)."""
    if spec.colorbar is not None and spec.colorbar.norm is not None:
        return spec.colorbar.norm
    return preset.norm


def _make_norm(name: str | None, clim: tuple[float, float] | None) -> Any:
    """Build a matplotlib color norm for ``name`` (``None``/``"linear"`` → ``Normalize``)."""
    import matplotlib.colors as mcolors

    vmin, vmax = clim if clim is not None else (None, None)
    if name == "log":
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if name == "symlog":
        # A modest linear threshold keeps a symlog norm well-defined for the
        # default case; callers wanting a specific linthresh build their own.
        return mcolors.SymLogNorm(linthresh=1e-8, vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Mark drawing functions (one per layer mark)
# ---------------------------------------------------------------------------

#: A mark-drawing function takes ``(ax, layer, spec, preset, theme)`` and draws the
#: layer, returning a colour-mappable artist when it produced one (for the
#: colorbar) else ``None``.
_MarkDrawer = Callable[["Axes", Layer, PlotSpec, _KindPreset, Theme], "ScalarMappable | None"]


def _channel(layer: Layer, name: str) -> np.ndarray | None:
    """Return a layer's channel array as float, or ``None`` if absent."""
    arr = layer.data.get(name)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _draw_line(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw a ``LINE`` mark — a poly-line, optionally colour-by-``c``.

    A plain ``y`` (no ``x``) plots against the sample index.  When the layer
    carries a per-vertex ``"c"`` channel the segments are coloured by it through
    a :class:`~matplotlib.collections.LineCollection`, returned so the caller can
    attach a colorbar.
    """
    y = _channel(layer, "y")
    if y is None:
        return None
    x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    c = _channel(layer, "c")
    if c is not None and c.size == y.size and y.size >= 2:
        return _draw_colored_line(ax, x, y, c, spec, layer, preset, theme)
    kw = _line_kwargs(layer, theme, n=int(y.size), autostyle=autostyle_enabled(spec))
    ax.plot(x, y, **kw)
    return None


def _draw_colored_line(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    spec: PlotSpec,
    layer: Layer,
    preset: _KindPreset,
    theme: Theme,
) -> ScalarMappable:
    """Draw a line whose segments are coloured by ``c`` (a ``LineCollection``)."""
    from matplotlib.collections import LineCollection

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = _resolve_cmap(spec, layer, preset)
    norm = _make_norm(_resolve_norm(spec, preset), spec.clim)
    # LineCollection wants a sequence of (N_i, 2) segment arrays; pass the stacked
    # array as a list so the type checker sees the expected Sequence shape.
    lc = LineCollection(list(segments), cmap=cmap, norm=norm)
    lc.set_array(c[:-1])
    canon = normalize_style(layer.style, warn=False)
    auto_lw, auto_alpha = autostyle_line(
        int(y.size), line_width=theme.line_width, enabled=autostyle_enabled(spec)
    )
    lw = canon.get("linewidth") or auto_lw
    if lw is not None:
        lc.set_linewidth(float(lw))
    if "zorder" in canon:
        lc.set_zorder(int(canon["zorder"]))
    if "alpha" in canon:
        lc.set_alpha(float(canon["alpha"]))
    elif auto_alpha is not None:
        lc.set_alpha(auto_alpha)
    if layer.label is not None:
        lc.set_label(layer.label)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


#: A marker cloud denser than this fraction of the figure's pixels cannot be
#: read: the markers overlap, and what you see is the *envelope*, not the data.
_SATURATION_FRACTION = 0.25


def _warn_if_recurrence_cannot_fit(ax: Axes, spec: PlotSpec, n_points: int) -> None:
    """Say so when a recurrence plot has more points than the figure has pixels.

    Measured: a 1501x1501 matrix at a **verified 5% density** draws 112 060
    markers into ~137 000 device pixels and renders as a near-solid black block —
    destroying exactly the diagonal-line structure DET / L_max / ENTR measure.
    The recurrence plot *is* the deliverable in RQA work, so silently handing
    back a picture that misrepresents a matrix whose density the caller has
    already checked is a wrong answer, not a cosmetic one.

    Restricted to :data:`~tsdynamics.viz.spec.PlotKind.RECURRENCE_PLOT`
    deliberately: an orbit diagram is *meant* to saturate (its black bands are
    the chaotic bands), so the same warning there would be noise.
    """
    import warnings

    if normalize_kind(spec.kind) is not PlotKind.RECURRENCE_PLOT:
        return
    fig = ax.get_figure(root=True)
    if fig is None:  # pragma: no cover - an axes always has a figure here
        return
    width, height = fig.get_size_inches()
    pixels = float(width) * float(height) * float(fig.dpi) ** 2
    if n_points < _SATURATION_FRACTION * pixels:
        return
    from ..caps import VisualizationDegraded

    warnings.warn(
        f"this recurrence plot draws {n_points:,} markers into about {int(pixels):,} device "
        "pixels, so they overlap and the diagonal structure is lost. Draw it as an image "
        "(ts.plot(traj, 'recurrence')), raise the resolution (p.size(w, h) / dpi= at save), "
        "or shorten/decimate the series.",
        VisualizationDegraded,
        stacklevel=3,
    )


def _draw_scatter(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw a ``SCATTER`` / ``MARKERS`` mark, optionally colour-/size-mapped."""
    y = _channel(layer, "y")
    if y is None:
        return None
    x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    _warn_if_recurrence_cannot_fit(ax, spec, int(np.size(y)))
    c = _channel(layer, "c")
    size = _channel(layer, "size")
    canon = normalize_style(layer.style, warn=False)
    kw: dict[str, Any] = {}
    if "alpha" in canon:
        kw["alpha"] = float(canon["alpha"])
    if "marker" in canon:
        kw["marker"] = _MARKER_MPL.get(str(canon["marker"]), canon["marker"])
    if "zorder" in canon:
        kw["zorder"] = int(canon["zorder"])
    if layer.label is not None:
        kw["label"] = layer.label
    if size is not None:
        kw["s"] = size
    # ``markersize`` / ``theme.marker_size`` are a marker *diameter* (pt, the
    # Line2D convention), but mpl scatter's ``s`` is a marker *area* (pt²) — square
    # the diameter so a canonical markersize matches a Line2D marker of that size.
    elif "markersize" in canon:
        kw["s"] = float(canon["markersize"]) ** 2
    else:
        # Density-aware, exactly as the LINE path is: the theme's constant marker
        # is right for a hundred points and renders a hundred thousand as one
        # solid blob.  An explicit ``markersize``/``size`` never reaches here.
        auto = autostyle_marker(
            int(np.size(y)),
            marker_size=theme.marker_size,
            enabled=autostyle_enabled(spec),
        )
        if auto is not None:
            kw["s"] = float(auto) ** 2
    if c is not None:
        kw["c"] = c
        kw["cmap"] = _resolve_cmap(spec, layer, preset)
        kw["norm"] = _make_norm(_resolve_norm(spec, preset), spec.clim)
        return ax.scatter(x, y, **kw)
    # ``filled=False`` draws a hollow/open marker — the textbook unstable-fixed-point
    # convention.  matplotlib spells it "no facecolor, ink in the edge", so the
    # layer's colour has to move from ``color`` to ``edgecolors``: passing both
    # ``color`` and ``facecolors`` to ``scatter`` is a conflict it silently resolves
    # in favour of ``color``, which would fill the marker anyway.
    if canon.get("filled") is False:
        kw["facecolors"] = "none"
        kw["edgecolors"] = canon.get("color", theme.foreground or "C0")
        kw.setdefault("linewidths", 1.2)
    elif "color" in canon:
        kw["color"] = canon["color"]
    ax.scatter(x, y, **kw)
    return None


def _make_discrete_cmap_norm(
    img: np.ndarray, spec: PlotSpec, layer: Layer, preset: _KindPreset
) -> tuple[Any, Any]:
    """Build a ``ListedColormap`` + ``BoundaryNorm`` for integer-label images.

    Reads ``spec.meta["palette_index"]`` (``{attractor_id: swatch_index}``) and
    ``spec.meta["diverged_color"]`` for basin diagrams; falls back to evenly
    sampling the base colormap for generic discrete images.
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    unique_vals = np.unique(img.ravel().astype(int))
    n = len(unique_vals)

    meta: dict[str, Any] = dict(spec.meta) if spec.meta else {}
    palette_index: dict[int, int] = meta.get("palette_index", {})
    diverged_color: str | None = meta.get("diverged_color")

    base_name = _resolve_cmap(spec, layer, preset) or "tab20"
    base_cmap = mpl.colormaps[base_name]
    # The swatch-centre fraction is ``(2*swatch + 1) / (2 * size)``; ``size`` must be
    # the number of swatches in the *resolved* map, not a hardcoded 20.  For tab20
    # (size 20) this is unchanged — ``(2s+1)/40``; for any other qualitative map
    # (Set1 size 9, …) or a continuous map (viridis size 256) it samples that map's
    # own swatch grid evenly so each label lands on a distinct, intended colour.
    swatch_size = max(1, int(getattr(base_cmap, "N", 20)))

    colors: list[Any] = []
    for v in unique_vals:
        v_int = int(v)
        if diverged_color is not None and v_int == -1:
            colors.append(mcolors.to_rgba(diverged_color))
        elif v_int in palette_index:
            swatch = palette_index[v_int]
            colors.append(base_cmap((2 * (swatch % swatch_size) + 1) / (2 * swatch_size)))
        else:
            i = int(np.searchsorted(unique_vals, v))
            colors.append(base_cmap(i / max(n, 1)))

    listed_cmap = mcolors.ListedColormap(colors)
    boundaries = np.concatenate([[float(unique_vals[0]) - 0.5], unique_vals.astype(float) + 0.5])
    norm = mcolors.BoundaryNorm(boundaries, n)
    return listed_cmap, norm


def _draw_image(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw an ``IMAGE`` mark from a 2-D ``z`` (or ``c``) channel.

    The image data is the ``"z"`` channel (falling back to ``"c"``); ``"x"`` /
    ``"y"`` channels, when 1-D, set the pixel-edge extent.  When neither is
    present the spec's axis limits are used as the extent, so a grid-backed
    image (e.g. basin diagrams) is placed at the correct physical coordinates.
    Origin is lower-left so row 0 sits at the bottom.

    When the spec's colorbar carries ``discrete=True`` (e.g. basin diagrams),
    a :class:`~matplotlib.colors.ListedColormap` and
    :class:`~matplotlib.colors.BoundaryNorm` are built so each unique integer
    label maps to exactly one discrete colour swatch.
    """
    z = layer.data.get("z")
    if z is None:
        z = layer.data.get("c")
    if z is None:
        return None
    raw = np.asarray(z)
    extent = _image_extent(layer, spec)
    interp = layer.style.get("interpolation", "nearest")

    discrete = spec.colorbar is not None and getattr(spec.colorbar, "discrete", False)
    if discrete:
        cmap, norm = _make_discrete_cmap_norm(raw, spec, layer, preset)
        im = ax.imshow(
            raw.astype(float),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation=interp,
        )
        return im

    img = raw.astype(float)
    cmap = _resolve_cmap(spec, layer, preset)
    norm = _make_norm(_resolve_norm(spec, preset), spec.clim)
    im = ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation=interp,
    )
    return im


def _image_extent(
    layer: Layer, spec: PlotSpec | None = None
) -> tuple[float, float, float, float] | None:
    """Return an ``(x0, x1, y0, y1)`` imshow extent.

    Priority: layer ``x``/``y`` channel edges → spec axis limits → ``None``
    (matplotlib default pixel-index placement).

    The ``x``/``y`` channels are pixel-*centre* coordinates (matching the plotly
    heatmap path, which passes coordinate vectors with centre semantics).  An
    imshow ``extent`` specifies the outer pixel *edges*, so for evenly-spaced
    centres the extent is expanded by half a cell on each side
    (``x0 - dx/2 .. x1 + dx/2``).  This places the image at the same physical
    coordinates as plotly and registers overlaid scatter/annotations correctly.
    """
    x = layer.data.get("x")
    y = layer.data.get("y")
    if x is not None and y is not None:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if xa.ndim == 1 and ya.ndim == 1 and xa.size >= 2 and ya.size >= 2:
            dx = (float(xa[-1]) - float(xa[0])) / (xa.size - 1)
            dy = (float(ya[-1]) - float(ya[0])) / (ya.size - 1)
            return (
                float(xa[0]) - dx / 2.0,
                float(xa[-1]) + dx / 2.0,
                float(ya[0]) - dy / 2.0,
                float(ya[-1]) + dy / 2.0,
            )
    if spec is not None and spec.x is not None and spec.y is not None:
        x_lim = spec.x.limits
        y_lim = spec.y.limits
        if x_lim is not None and y_lim is not None:
            return (float(x_lim[0]), float(x_lim[1]), float(y_lim[0]), float(y_lim[1]))
    return None


def _draw_histogram(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw a ``HISTOGRAM`` mark.

    Two shapes are accepted: pre-binned (``x`` = bin centres / edges, ``y`` =
    counts → a step/bar histogram) or raw samples (only ``x`` → matplotlib bins
    them).
    """
    x = _channel(layer, "x")
    if x is None:
        return None
    y = _channel(layer, "y")
    canon = normalize_style(layer.style, warn=False)
    kw: dict[str, Any] = {}
    if "color" in canon:
        kw["color"] = canon["color"]
    if "alpha" in canon:
        kw["alpha"] = float(canon["alpha"])
    if layer.label is not None:
        kw["label"] = layer.label
    if y is not None:
        # Pre-binned: draw counts at the given centres as a bar histogram.
        width = float(np.median(np.diff(x))) if x.size >= 2 else 1.0
        ax.bar(x, y, width=width, align="center", **kw)
        return None
    bins = layer.style.get("bins", "auto")
    ax.hist(x, bins=bins, **kw)
    return None


def _draw_bar(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw a ``BAR`` mark — values ``y`` at positions ``x`` / ``cat``."""
    y = _channel(layer, "y")
    if y is None:
        return None
    x = _channel(layer, "cat")
    if x is None:
        x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    canon = normalize_style(layer.style, warn=False)
    kw: dict[str, Any] = {}
    if "color" in canon:
        kw["color"] = canon["color"]
    if "alpha" in canon:
        kw["alpha"] = float(canon["alpha"])
    if layer.label is not None:
        kw["label"] = layer.label
    ax.bar(x, y, **kw)
    return None


def _draw_area(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw an ``AREA`` mark — a shaded ``lo <= hi`` band over ``x``.

    Falls back to a band around ``y`` if only one edge is present, and to a plain
    filled area to zero if neither ``lo`` nor ``hi`` is given.
    """
    x = _channel(layer, "x")
    lo = _channel(layer, "lo")
    hi = _channel(layer, "hi")
    y = _channel(layer, "y")
    if x is None:
        ref = lo if lo is not None else (hi if hi is not None else y)
        if ref is None:
            return None
        x = np.arange(ref.size, dtype=float)
    if lo is None:
        lo = y if y is not None else np.zeros_like(x)
    if hi is None:
        hi = y if y is not None else np.zeros_like(x)
    canon = normalize_style(layer.style, warn=False)
    fill_alpha = canon.get("fillalpha", canon.get("alpha", 0.3))
    # ``fill`` (AREA-only) suppresses the shaded band when False; the central
    # line (when a distinct ``y`` is present) still draws.  Default True.
    show_fill = bool(canon.get("fill", True))
    kw: dict[str, Any] = {"alpha": float(fill_alpha)}
    if "color" in canon:
        kw["color"] = canon["color"]
    # A band and the line through it are ONE series, so exactly one of the two
    # artists carries the legend label.  Both did, and matplotlib legends every
    # labelled artist: measured, a 2-attractor continuation produced FOUR
    # entries reading "attractor 1 · attractor 1 · attractor 2 · attractor 2",
    # which says there are four things when there are two.  The LINE takes it
    # when there is one — its swatch is the solid colour, where the band's is
    # the same colour at alpha 0.3 and reads as a different series.
    centre = y if (y is not None and "lo" in layer.data) else None
    if layer.label is not None and centre is None:
        kw["label"] = layer.label
    if show_fill:
        ax.fill_between(x, lo, hi, **kw)
    if centre is not None:
        line_kw: dict[str, Any] = {k: v for k, v in kw.items() if k == "color"}
        if layer.label is not None:
            line_kw["label"] = layer.label
        ax.plot(x, centre, **line_kw)
    return None


def _draw_errorbar(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw an ``ERRORBAR`` mark — ``y`` vs ``x`` with symmetric ``err`` bars."""
    y = _channel(layer, "y")
    if y is None:
        return None
    x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    err = _channel(layer, "err")
    canon = normalize_style(layer.style, warn=False)
    raw_marker = canon.get("marker", "o")
    mpl_marker = (
        _MARKER_MPL.get(str(raw_marker), raw_marker) if isinstance(raw_marker, str) else "o"
    )
    kw: dict[str, Any] = {"fmt": mpl_marker}
    if "color" in canon:
        kw["color"] = canon["color"]
    if "alpha" in canon:
        kw["alpha"] = float(canon["alpha"])
    if layer.label is not None:
        kw["label"] = layer.label
    ax.errorbar(x, y, yerr=err, **kw)
    return None


def _draw_quiver(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Draw a ``QUIVER`` mark — arrows ``(u, v)`` at positions ``(x, y)``.

    A scalar ``"c"`` channel colours the arrows (returned for a colorbar).
    """
    x = _channel(layer, "x")
    y = _channel(layer, "y")
    u = _channel(layer, "u")
    v = _channel(layer, "v")
    if x is None or y is None or u is None or v is None:
        return None
    c = _channel(layer, "c")
    canon = normalize_style(layer.style, warn=False)
    if c is not None:
        cmap = _resolve_cmap(spec, layer, preset)
        norm = _make_norm(_resolve_norm(spec, preset), spec.clim)
        q = ax.quiver(x, y, u, v, c, cmap=cmap, norm=norm)
        return q
    kw: dict[str, Any] = {}
    if "color" in canon:
        kw["color"] = canon["color"]
    ax.quiver(x, y, u, v, **kw)
    return None


def _draw_3d_unsupported(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset, theme: Theme
) -> ScalarMappable | None:
    """Raise for a 3-D mark — drawn by :mod:`._threed`."""
    raise NotImplementedError(
        f"the matplotlib reference renderer draws 2-D marks only; {layer.kind.value!r} "
        "(3-D) is drawn by the _threed module."
    )


#: Layer mark → drawing function.  A renderer ignores channels a mark does not
#: consume (the closed channel vocabulary).  3-D marks raise a clear
#: NotImplementedError here until the VIZ-MPL-3D stream fills them in.
MARK_DISPATCH: dict[PlotKind, _MarkDrawer] = {
    PlotKind.LINE: _draw_line,
    PlotKind.SCATTER: _draw_scatter,
    PlotKind.MARKERS: _draw_scatter,
    PlotKind.IMAGE: _draw_image,
    PlotKind.HISTOGRAM: _draw_histogram,
    PlotKind.BAR: _draw_bar,
    PlotKind.AREA: _draw_area,
    PlotKind.ERRORBAR: _draw_errorbar,
    PlotKind.QUIVER: _draw_quiver,
    PlotKind.LINE3D: _draw_3d_unsupported,
    PlotKind.SURFACE3D: _draw_3d_unsupported,
}


# ---------------------------------------------------------------------------
# Axis / colorbar / legend / annotation application
# ---------------------------------------------------------------------------


def _apply_axis(ax: Axes, axis: Axis, which: Literal["x", "y"], theme: Theme) -> None:
    """Apply one :class:`~tsdynamics.viz.spec.Axis` — all fields including new ones.

    Handles the existing fields (label, scale, limits, ticks, categories) plus the
    new enriched fields: ``grid``, ``color``, ``label_size``, ``tick_size``,
    ``tick_rotation``, and ``tickformat`` (now honored via
    :class:`~matplotlib.ticker.FormatStrFormatter`).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : Axis
        The typed axis spec.
    which : {"x", "y"}
        Which axis to apply to.
    theme : Theme
        The resolved theme (provides foreground color, font sizes).
    """
    import matplotlib.ticker as mticker

    set_label = ax.set_xlabel if which == "x" else ax.set_ylabel
    set_scale = ax.set_xscale if which == "x" else ax.set_yscale
    set_lim = ax.set_xlim if which == "x" else ax.set_ylim
    set_ticks = ax.set_xticks if which == "x" else ax.set_yticks
    set_ticklabels = ax.set_xticklabels if which == "x" else ax.set_yticklabels
    get_axis_obj = ax.xaxis if which == "x" else ax.yaxis

    # Determine effective ink color: axis.color > theme.foreground > None
    ink = axis.color if axis.color is not None else theme.foreground

    # Label
    if axis.label:
        label_kw: dict[str, Any] = {}
        if ink is not None:
            label_kw["color"] = ink
        eff_label_size = axis.label_size if axis.label_size is not None else theme.font_size
        if eff_label_size is not None:
            label_kw["fontsize"] = float(eff_label_size)
        if theme.font_family is not None:
            label_kw["fontfamily"] = theme.font_family
        set_label(axis.label, **label_kw)

    # Scale
    if axis.scale in ("log", "symlog"):
        set_scale(axis.scale)
    elif axis.scale == "categorical" and axis.categories is not None:
        positions = np.arange(len(axis.categories), dtype=float)
        set_ticks(positions)
        set_ticklabels(list(axis.categories))

    # Limits
    if axis.limits is not None:
        set_lim(axis.limits[0], axis.limits[1])

    # Explicit ticks
    if axis.ticks is not None:
        set_ticks(list(axis.ticks))

    # Tick formatter (honor tickformat — previously IGNORED)
    if axis.tickformat is not None:
        fmt_str = axis.tickformat
        # Use StrMethodFormatter for Python str.format strings, else FormatStrFormatter
        if "{" in fmt_str:
            get_axis_obj.set_major_formatter(mticker.StrMethodFormatter(fmt_str))
        else:
            get_axis_obj.set_major_formatter(mticker.FormatStrFormatter(fmt_str))

    # Tick styling: size and rotation
    tick_kw: dict[str, Any] = {}
    if ink is not None:
        tick_kw["colors"] = ink
    eff_tick_size = axis.tick_size if axis.tick_size is not None else theme.font_size
    if eff_tick_size is not None:
        tick_kw["labelsize"] = float(eff_tick_size)
    if axis.tick_rotation is not None:
        tick_kw["rotation"] = float(axis.tick_rotation)
    if tick_kw:
        ax.tick_params(axis=which, **tick_kw)

    # Tick-label font family (tick_params has no family kwarg — set it on the
    # label objects directly, figure-locally, no global rcParams mutation).
    if theme.font_family is not None:
        for lbl in ax.get_xticklabels() if which == "x" else ax.get_yticklabels():
            lbl.set_fontfamily(theme.font_family)

    # Spine / axis label color (axis.color overrides theme.foreground)
    if ink is not None:
        if which == "x":
            ax.spines["bottom"].set_edgecolor(ink)
            ax.spines["top"].set_edgecolor(ink)
        else:
            ax.spines["left"].set_edgecolor(ink)
            ax.spines["right"].set_edgecolor(ink)


def apply_title(ax: Axes, spec: PlotSpec, theme: Theme) -> None:
    """Set the axes title **in the theme's ink, size and family**.

    Shared with the animation renderer rather than re-derived there, because the
    copy that path grew was a plain ``ax.set_title(spec.title)``: the theme's
    foreground reached a title only through :func:`_apply_theme_to_figure`, which
    colours it *if it already exists*, and the animator sets it afterwards.  So
    one spec rendered ``#e6e6e6`` as a ``.png`` and near-black as the same
    frame of a ``.gif`` — a talk slide and a paper figure that do not match.
    """
    if not spec.title:
        return
    title_kw: dict[str, Any] = {}
    if theme.foreground is not None:
        title_kw["color"] = theme.foreground
    eff_title_size = theme.title_size if theme.title_size is not None else theme.font_size
    if eff_title_size is not None:
        title_kw["fontsize"] = float(eff_title_size)
    if theme.font_family is not None:
        title_kw["fontfamily"] = theme.font_family
    ax.set_title(spec.title, **title_kw)


def _apply_axes(ax: Axes, spec: PlotSpec, preset: _KindPreset, theme: Theme) -> None:
    """Apply both axes, the title, and the aspect ratio to ``ax``."""
    _apply_axis(ax, spec.x, "x", theme)
    _apply_axis(ax, spec.y, "y", theme)
    apply_title(ax, spec, theme)
    aspect = spec.aspect if spec.aspect != "auto" else preset.aspect
    if aspect == "equal":
        ax.set_aspect("equal", adjustable="box")
    if spec._axes_hidden():
        ax.set_axis_off()


def _aspect_matched_cax(ax: Axes, location: str | None) -> Axes | None:
    """Return a colorbar axes **tied to an aspect-locked axes' drawn box**, or ``None``.

    Measured on a basin image over ``x in [-3, 3]``, ``y in [-0.6, 0.6]``: the
    colorbar came out roughly **three times the height of the picture**, with
    the picture squashed into the middle third — the figure looked broken, and
    it is one of the library's headline outputs.  ``Figure.colorbar(..., ax=ax)``
    sizes the bar from the axes' *rectangle*, but an equal-aspect image with
    ``adjustable="box"`` draws inside a smaller box than its rectangle, and the
    rectangle itself is re-laid-out afterwards — so no single-pass ``shrink=``
    can be exact.  ``axes_grid1``'s divider positions the bar through the parent's
    own locator, which is evaluated *after* ``apply_aspect``, so the two heights
    agree by construction (measured: 93.1 px vs 93.1 px).

    The bar is an **inset in axes coordinates**, which are the drawn box after
    ``apply_aspect`` by definition — so it tracks the picture under constrained
    layout too (``axes_grid1``'s divider does not).

    Returns ``None`` — leaving matplotlib's own sizing, with the
    :func:`_aspect_shrink` correction — for a 3-D axes, a free-aspect axes, a
    top/bottom bar (whose tick labels need layout space an inset cannot reserve),
    and, deliberately, **whenever the mismatch is small**: confining the change
    to the figures that were visibly broken leaves every currently-fine figure
    byte-identical.
    """
    if getattr(ax, "name", "") == "3d" or ax.get_adjustable() != "box":
        return None
    if str(location or "right") not in ("right", "left"):
        return None
    try:
        float(ax.get_aspect())
    except (TypeError, ValueError):  # "auto" — nothing to match
        return None
    if _aspect_shrink(ax, location) > _ASPECT_MISMATCH_FLOOR:
        return None
    x0 = 1.03 if str(location or "right") == "right" else -0.10
    return ax.inset_axes((x0, 0.0, 0.04, 1.0), transform=ax.transAxes)


#: How far the drawn box may fall short of its cell before the colorbar is
#: re-anchored to the picture.  Above it the two are close enough that
#: matplotlib's own placement reads correctly, and moving it would rewrite
#: figures that were never wrong.
_ASPECT_MISMATCH_FLOOR = 0.75


def _aspect_shrink(ax: Axes, location: str | None) -> float:
    """Return the colorbar ``shrink`` that matches an aspect-locked axes' drawn box.

    ``fig.colorbar(..., ax=ax)`` sizes the bar from the axes' *rectangle*, but an
    equal-aspect image with ``adjustable="box"`` draws inside a **smaller** box
    than its rectangle.  Measured on a basin image over ``x in [-3, 3]``,
    ``y in [-0.6, 0.6]``: the colorbar came out roughly **three times the height
    of the picture**, with the picture squashed into the middle third — the
    figure looked broken, and it is one of the library's headline outputs.

    Returns 1.0 (matplotlib's own default, so nothing moves) whenever the axes is
    free to fill its rectangle.
    """
    if ax.get_adjustable() != "box":
        return 1.0
    try:
        ratio = float(ax.get_aspect())
    except (TypeError, ValueError):  # "auto"
        return 1.0
    figure = ax.get_figure(root=True)
    if figure is None:  # pragma: no cover - an axes always has a figure here
        return 1.0
    fig_w, fig_h = figure.get_size_inches()
    # ``original=True`` is load-bearing: ``Figure.colorbar(..., ax=ax)`` sizes the
    # bar from the axes' ORIGINAL rectangle, and the active position may or may
    # not have been shrunk by ``apply_aspect`` yet depending on whether anything
    # has drawn.  Reading the same rectangle the colorbar does makes the ratio
    # correct either way.
    box = ax.get_position(original=True)
    rect_w, rect_h = float(box.width) * float(fig_w), float(box.height) * float(fig_h)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    data_w, data_h = abs(float(x1) - float(x0)), abs(float(y1) - float(y0)) * ratio
    if not (data_w > 0.0 and data_h > 0.0 and rect_w > 0.0 and rect_h > 0.0):
        return 1.0
    along_x = str(location) in ("top", "bottom")
    if along_x:
        drawn = min(rect_w, rect_h * data_w / data_h)
        return float(np.clip(drawn / rect_w, 0.05, 1.0))
    drawn = min(rect_h, rect_w * data_h / data_w)
    return float(np.clip(drawn / rect_h, 0.05, 1.0))


def _apply_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable | None,
    colorbar: Colorbar | None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Attach a colorbar for ``mappable`` honouring a :class:`Colorbar` spec.

    Honors ``colorbar.label_size``, and — for a **discrete** (categorical)
    colorbar — turns the numeric ramp into a genuine categorical legend: one tick
    at the centre of each swatch, labelled with the category's own name.

    A basin diagram is the motivating case.  Its colour channel is an *attractor
    id*, not a quantity, but the colorbar read ``0.5 / 1.5 / 2.5`` — the
    :class:`~matplotlib.colors.BoundaryNorm` bin edges — which names nothing and
    implies an ordering the data does not have.  The category names come from
    ``spec.meta["category_labels"]`` (``{value: label}``), which the emitter
    records alongside the ``palette_index`` mapping it already carries; absent
    that, the integer value itself is the label.
    """
    if mappable is None or colorbar is None or not colorbar.show:
        return
    cax = _aspect_matched_cax(ax, colorbar.location)
    if cax is not None:
        cb = fig.colorbar(mappable, cax=cax, location=colorbar.location)
    else:
        cb = fig.colorbar(
            mappable,
            ax=ax,
            location=colorbar.location,
            shrink=_aspect_shrink(ax, colorbar.location),
        )
    if colorbar.label:
        label_kw: dict[str, Any] = {}
        if colorbar.label_size is not None:
            label_kw["fontsize"] = float(colorbar.label_size)
        cb.set_label(colorbar.label, **label_kw)
    if colorbar.ticks is not None:
        cb.set_ticks(list(colorbar.ticks))
    elif colorbar.discrete:
        _apply_categorical_ticks(cb, mappable, meta)
    if colorbar.tickformat is not None:
        import matplotlib.ticker as mticker

        fmt_str = colorbar.tickformat
        if "{" in fmt_str:
            cb.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(fmt_str))
        else:
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt_str))
    if colorbar.label_size is not None:
        cb.ax.tick_params(labelsize=float(colorbar.label_size))


def _apply_categorical_ticks(
    cb: Any, mappable: ScalarMappable, meta: dict[str, Any] | None
) -> None:
    """Tick a discrete colorbar once per swatch, labelled by category name.

    Reads the :class:`~matplotlib.colors.BoundaryNorm` the discrete image built
    (:func:`_make_discrete_cmap_norm`): its boundaries are ``value ± 0.5``, so the
    swatch centres are the integer category values themselves.  Labels come from
    ``meta["category_labels"]`` when the emitter supplied them.
    """
    import matplotlib.colors as mcolors

    norm = getattr(mappable, "norm", None)
    if not isinstance(norm, mcolors.BoundaryNorm):
        return
    bounds = np.asarray(norm.boundaries, dtype=float)
    if bounds.size < 2:
        return
    centres = 0.5 * (bounds[:-1] + bounds[1:])
    # ``_make_discrete_cmap_norm`` builds the boundaries as ``value + 0.5`` per
    # unique value, so bin ``i`` represents ``bounds[i + 1] - 0.5``.  Recover the
    # category from the *upper* edge, not from the bin centre: with
    # non-contiguous labels (a basin field of ``[-1, 1, 2]``) the bins are uneven
    # and a centre rounds to the wrong value (0.5 -> "0" instead of "1").
    values = [int(round(float(b) - 0.5)) for b in bounds[1:]]
    labels_map: dict[int, str] = {}
    if meta:
        raw = meta.get("category_labels")
        if isinstance(raw, dict):
            labels_map = {int(k): str(v) for k, v in raw.items()}
    cb.set_ticks(list(centres))
    cb.set_ticklabels([labels_map.get(v, str(v)) for v in values])


def _apply_legend(ax: Axes, spec: PlotSpec, theme: Theme) -> None:
    """Draw the per-layer legend, honouring all enriched :class:`Legend` fields.

    Now honors ``legend.font_size``, ``legend.ncol``, and ``legend.frame`` in
    addition to the existing ``location`` / ``title``.
    """
    if spec.legend is None or not spec.legend.show:
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    leg = spec.legend
    legend_kw: dict[str, Any] = {
        "loc": leg.location,
        "ncols": int(leg.ncol),
        "frameon": bool(leg.frame),
    }
    if leg.title:
        legend_kw["title"] = leg.title
    eff_font_size = leg.font_size if leg.font_size is not None else theme.font_size
    if eff_font_size is not None:
        legend_kw["fontsize"] = float(eff_font_size)
    ax.legend(**legend_kw)


_TEXT_ONLY_STYLE_KEYS = frozenset({"fontsize", "fontweight", "fontstyle", "ha", "va", "rotation"})


def _annotation_line_style(style: dict[str, Any]) -> dict[str, Any]:
    """Drop text-only keys a line artist (``axvline``/``axhline``) would reject."""
    return {k: v for k, v in style.items() if k not in _TEXT_ONLY_STYLE_KEYS}


def _annotation_text_style(style: dict[str, Any]) -> dict[str, Any]:
    """Keep only the style keys :meth:`~matplotlib.axes.Axes.text` accepts.

    A ``vline`` / ``hline`` style is a *line* style (it may carry ``linestyle`` /
    ``linewidth``, which ``text`` rejects); the inline label inherits the shared
    ``color`` / ``alpha`` / ``fontsize`` / ``fontweight`` so it matches its line.
    """
    keep = ("color", "alpha", "fontsize", "fontweight", "fontstyle")
    return {k: style[k] for k in keep if k in style}


def _apply_annotations(ax: Axes, annotations: list[Annotation]) -> None:
    """Draw reference lines / text / spans (``vline`` / ``hline`` / ``text`` / ``span``).

    For ``vline`` / ``hline`` the ``text`` is drawn as an inline label at the top
    (vline) / right (hline) edge of the axes *and* registered as the line's legend
    label, so the label is visible whether or not a legend is shown.  Line-only and
    text-only style keys are routed to the right artist, so a caller may put a
    ``fontsize`` on a ``vline`` style without breaking the line.
    """
    for ann in annotations:
        style = dict(ann.style)
        line_style = _annotation_line_style(style)
        if ann.kind == "vline" and ann.x is not None:
            ax.axvline(ann.x, label=ann.text or None, **line_style)
            if ann.text:
                # x in data coords, y in axes fraction (top edge), label up the line.
                ax.text(
                    ann.x,
                    0.99,
                    ann.text,
                    transform=ax.get_xaxis_transform(),
                    ha="right",
                    va="top",
                    rotation=90,
                    rotation_mode="anchor",
                    **_annotation_text_style(style),
                )
        elif ann.kind == "hline" and ann.y is not None:
            ax.axhline(ann.y, label=ann.text or None, **line_style)
            if ann.text:
                # y in data coords, x in axes fraction (right edge).
                ax.text(
                    0.99,
                    ann.y,
                    ann.text,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    **_annotation_text_style(style),
                )
        elif ann.kind == "text" and ann.x is not None and ann.y is not None:
            ax.text(ann.x, ann.y, ann.text, **style)
        elif ann.kind == "span" and ann.span is not None:
            lo, hi = ann.span
            if ann.axis == "y":
                ax.axhspan(lo, hi, **line_style)
            else:
                ax.axvspan(lo, hi, **line_style)


# ---------------------------------------------------------------------------
# The render entry point
# ---------------------------------------------------------------------------


def render(
    spec: PlotSpec,
    *,
    figsize: tuple[float, float] | None = None,
    path: str | Path | None = None,
    ax: Axes | None = None,
    **_kw: Any,
) -> Figure | FuncAnimation | Path:
    """Render a 2-D :class:`~tsdynamics.viz.spec.PlotSpec` to a matplotlib Figure.

    Builds a :class:`~matplotlib.figure.Figure` with the Agg canvas (no
    ``pyplot``), resolves the spec's theme, applies it figure-locally (no global
    ``rcParams`` mutation), draws every :class:`~tsdynamics.viz.spec.Layer` through
    :data:`MARK_DISPATCH`, then applies the axes, colorbar, legend and annotations
    the spec carries.  The spec's semantic kind is normalised through
    :func:`tsdynamics.viz.render.normalize_kind` so an alias / mark spelling
    resolves to the preset table.

    Parameters
    ----------
    spec : PlotSpec
        The backend-agnostic spec to draw.  Its per-call tweaks
        (relabel/rescale/limits/ticks/colorize/style/recolor/theme/…) are
        already baked into the typed axes / colorbar / theme, so honouring those
        honours the tweaks.
    figsize : tuple of float, optional
        ``(width, height)`` in inches; matplotlib's default when ``None``.
    path : str or Path, optional
        Write the artifact here and return the :class:`~pathlib.Path` instead of
        the figure — the same ``render(path=...)`` contract the plotly and data
        backends already honored.  matplotlib is the **default** backend, so
        without this ``spec.render(path=...)`` (and therefore
        ``result.plot(path=...)``, whose keyword table blesses ``path``) accepted
        the request, returned a Figure and wrote **nothing**.  A raster / vector
        extension goes through ``Figure.savefig``; ``.mp4`` / ``.gif`` on an
        animated spec go through the animation's own writer.
    ax : matplotlib.axes.Axes, optional
        Draw into an axes **you** own instead of building a figure — the escape
        hatch for "put this plot in my paper figure"::

            fig, axs = plt.subplots(1, 2)
            ts.viz.plot(duffing_basins).render(ax=axs[0])
            traj.plot(ax=axs[1])

        The single-panel drawing body already worked on any axes (it is what the
        composite renderer calls per panel); only the plumbing was missing.  The
        spec's theme is still applied **figure-locally** to your figure, and the
        return value is that figure, so ``ax=`` composes with the rest of your
        matplotlib code rather than replacing it.

        A 3-D spec needs a 3-D axes (``subplot_kw={"projection": "3d"}``); an
        animated or composite spec drives a whole figure and so cannot be given
        one axes.  Both raise rather than drawing something misleading.

        .. versionadded:: 6.0
    **_kw
        Forwarded but unused backend keywords (kept for a uniform renderer
        signature).

    Returns
    -------
    matplotlib.figure.Figure or pathlib.Path
        The rendered figure (a single axes), ready to ``savefig`` / embed — or
        the written ``path`` when one was given.  A 3-D spec (``ndim == 3`` / a
        ``z`` axis / a ``LINE3D`` / ``SURFACE3D`` mark) is dispatched to the
        :mod:`._threed` renderer.  With ``ax=``, the axes' own figure.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``ax`` is given for an animated or composite spec, or for a 3-D spec
        on a 2-D axes (or vice versa).
    """
    from . import _threed

    if ax is not None:
        return _render_into(spec, ax, path)

    result: Figure | FuncAnimation
    if spec.is_animated:
        from . import _anim

        result = _anim.render_animation(spec, figsize=figsize)
    elif spec.is_composite:
        result = _render_composite(spec, figsize=figsize)
    elif _threed.is_three_d(spec):
        result = _threed.render_3d(spec, figsize=figsize)
    else:
        figsize, dpi, layout = figure_geometry(spec, figsize)
        fig = new_figure(figsize, dpi, layout)
        ax = fig.add_subplot(1, 1, 1)
        _draw_2d_panel(fig, ax, spec)
        result = fig
    if path is None:
        return result
    return _write(result, path)


def _render_into(spec: PlotSpec, ax: Axes, path: str | Path | None) -> Figure | Path:
    """Draw ``spec`` into a caller-owned ``ax`` and return its figure (or ``path``).

    The ``render(ax=...)`` body.  Everything it refuses, it refuses because the
    alternative is a plausible-looking wrong picture: an animation and a composite
    each drive a *figure* (many axes / a frame loop), and a 3-D spec on a 2-D axes
    would silently drop the depth coordinate.
    """
    from . import _threed

    if spec.is_animated:
        raise InvalidParameterError(
            "ax= cannot render an animated spec: an animation drives a whole figure "
            "(it owns the frame loop). Render it without ax= and use the returned "
            "FuncAnimation, or drop the animation with spec.animation = None."
        )
    if spec.is_composite:
        raise InvalidParameterError(
            "ax= cannot render a COMPOSITE spec: it needs one axes per panel. "
            "Render a single panel into your axes — spec.panels[i].render(ax=ax)."
        )
    is_3d_axes = getattr(ax, "name", "") == "3d"
    if _threed.is_three_d(spec) and not is_3d_axes:
        raise InvalidParameterError(
            "ax= got a 2-D axes for a 3-D spec; the depth coordinate would be "
            'dropped. Create one with plt.subplots(subplot_kw={"projection": "3d"}).'
        )
    if is_3d_axes and not _threed.is_three_d(spec):
        raise InvalidParameterError(
            "ax= got a 3-D axes for a 2-D spec. Create a plain axes with plt.subplots()."
        )

    figure = cast("Figure", ax.get_figure())
    if is_3d_axes:
        _threed._draw_3d_panel(figure, ax, spec)
    else:
        _draw_2d_panel(figure, ax, spec)
    if path is None:
        return figure
    return _write(figure, path)


def _write(result: Figure | FuncAnimation, path: str | Path) -> Path:
    """Write a rendered figure / animation to ``path`` and return it.

    A ``FuncAnimation`` (uniquely carrying ``to_jshtml``) writes a movie through
    its own ``save`` when the extension is one; anything else is a still of the
    animation's underlying figure.  Raises rather than silently producing an
    empty file for an extension matplotlib cannot write.
    """
    out = Path(path)
    ext = out.suffix.lower()
    if hasattr(result, "to_jshtml") and ext in _MOVIE_EXTENSIONS:
        result.save(str(out))  # type: ignore[union-attr]
        return out
    figure = getattr(result, "_fig", None) or getattr(result, "figure", result)
    savefig = getattr(figure, "savefig", None)
    if savefig is None:  # pragma: no cover - defensive
        raise InvalidParameterError(
            f"the matplotlib backend cannot write {out.name}: no savable figure was produced."
        )
    savefig(str(out))
    return out


def _draw_2d_panel(
    fig: Figure, ax: Axes, spec: PlotSpec, theme: Theme | None = None, *, colorbar: bool = True
) -> ScalarMappable | None:
    """Draw one 2-D spec's layers + axes/colorbar/legend/annotations onto ``ax``.

    The single-panel body of :func:`render`, factored out so the composite
    renderer can draw each panel into its own axes of a shared figure.

    Step 1: resolve theme and apply it figure-locally (background, color cycle,
    font, default grid).  Step 2: draw layers with normalized per-layer style
    overriding theme defaults.  Step 3: apply enriched Axis/Legend/Colorbar fields.

    ``theme`` lets the composite renderer pass an *inherited* theme (its own) for a
    panel that has none, **without mutating the panel spec** — when ``None`` the
    panel's own resolved theme is used.

    ``colorbar=False`` skips step 3's colorbar and returns the mappable instead,
    so a ``share_color=True`` composite can attach **one** bar spanning every
    panel at the figure edge rather than inside whichever panel happened to keep
    it (which, on a 2x2, landed the "figure-level" bar between panels 1 and 2 of
    the top row).

    Returns
    -------
    ScalarMappable or None
        The colour-mapped artist this panel drew, if any.
    """
    theme = theme if theme is not None else _resolve_theme(spec)
    preset = _preset_for(normalize_kind(spec.kind))

    # Step 1: apply theme presentation
    _apply_theme_to_figure(fig, ax, theme)
    _apply_theme_color_cycle(ax, theme, spec)

    # Step 2: draw layers
    mappable: ScalarMappable | None = None
    for layer in spec.layers:
        drawer = MARK_DISPATCH.get(PlotKind(layer.kind))
        if drawer is None:
            continue
        produced = drawer(ax, layer, spec, preset, theme)
        if produced is not None:
            mappable = produced

    # Step 3: apply axes (grid, labels, ticks, tickformat, …), colorbar, legend, annotations
    _apply_theme_grid(ax, spec, theme)
    _apply_axes(ax, spec, preset, theme)
    if colorbar:
        _apply_colorbar(fig, ax, mappable, spec.colorbar, spec.meta)
    _apply_legend(ax, spec, theme)
    _apply_annotations(ax, spec.annotations)
    return mappable


def _apply_shared_colorbar(
    fig: Figure, axes: list[Axes], mappable: ScalarMappable, colorbar: Colorbar
) -> None:
    """Attach **one** colorbar spanning every panel, at the figure edge.

    ``share_color=True`` unifies the colour range in the IR
    (:func:`~tsdynamics.viz.compose._unify_colour`) and leaves the
    :class:`~tsdynamics.viz.spec.Colorbar` on the first colour-bearing panel,
    which a per-panel renderer draws *inside that panel* — so the "one
    figure-level colorbar" of a 2x2 landed between panels 1 and 2 of the top
    row, which is not a layout anyone would put in a paper.  Handing ``ax=`` the
    whole axes list makes matplotlib steal space from all of them and put the bar
    on the figure's edge, which is what the keyword promises.
    """
    if not colorbar.show:
        return
    bar = fig.colorbar(mappable, ax=axes, location=colorbar.location or "right")
    if colorbar.label:
        bar.set_label(colorbar.label)
    if colorbar.ticks is not None:
        bar.set_ticks(list(colorbar.ticks))


def _hide_inner_tick_labels(axes: list[Axes], *, share_x: bool, share_y: bool) -> None:
    """Strip the redundant inner tick labels of a shared-axis grid.

    ``sharex=``/``sharey=`` link the *limits*; matplotlib only hides the inner
    labels when you go through ``plt.subplots(...)``, which this renderer does
    not (it builds axes one at a time so a 3-D panel can sit beside a 2-D one).
    So ``share_x=True`` produced a stack with the x tick labels repeated under
    every panel — and removing that repetition is the main reason to ask for
    shared axes in a paper figure.
    """
    for ax in axes:
        spec = getattr(ax, "get_subplotspec", lambda: None)()
        if spec is None or getattr(ax, "name", "") == "3d":
            continue
        if share_x and not spec.is_last_row():
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
        if share_y and not spec.is_first_col():
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")


def _composite_grid(layout: Any, n: int) -> tuple[int, int]:
    """Return the ``(rows, cols)`` subplot grid for a composite's :class:`Layout`."""
    mode = getattr(layout, "mode", "stack")
    if mode == "row":
        return 1, n
    if mode == "grid":
        rows = getattr(layout, "rows", None)
        cols = getattr(layout, "cols", None)
        if rows and cols:
            return int(rows), int(cols)
        c = int(np.ceil(np.sqrt(n)))
        r = int(np.ceil(n / c))
        return r, c
    return n, 1  # "stack" (default): one column


def _leaf_extent(spec: PlotSpec) -> tuple[int, int]:
    """Return ``(rows, cols)`` of a nested composite measured in LEAF panels.

    The default figure size is ``(cols * 5.0, rows * 3.2)``; reading that off the
    *top-level* grid of a nested tree sizes a 2x2 built as ``(a|b)/(c|d)`` — whose
    top-level grid is 2x1 — as a two-row single-column page, so every panel is
    drawn squeezed into half the width it needs.  Measuring the leaves gives the
    2x2 the page a 2x2 wants.
    """
    if not spec.is_composite:
        return (1, 1)
    rows, cols = _composite_grid(spec.layout, len(spec.panels))
    extents = [_leaf_extent(p) for p in spec.panels]
    heights = [
        max((extents[i][0] for i in range(len(extents)) if i // cols == r), default=1)
        for r in range(rows)
    ]
    widths = [
        max((extents[i][1] for i in range(len(extents)) if i % cols == c), default=1)
        for c in range(cols)
    ]
    return (sum(heights), sum(widths))


def _place_composite(
    fig: Figure,
    cell: Any,
    spec: PlotSpec,
    inherited_theme: Theme,
) -> list[Axes]:
    """Draw one composite level into ``cell``, recursing into nested composites.

    ``cell`` is the enclosing :class:`~matplotlib.gridspec.SubplotSpec` this
    arrangement occupies, or ``None`` for the whole figure.  A panel that is
    itself a composite gets its cell subdivided
    (``cell.subgridspec(rows, cols)``) and is placed by a recursive call, which
    is what makes the parentheses a user typed the layout they get: in
    ``(a|b)/c`` the outer stack is 2x1, its first cell holds a 1x2 row, and ``c``
    occupies the whole second cell — **spanning the full width**, as a single
    panel on a row of its own should.

    ``share_x`` / ``share_y`` / ``share_color`` are honoured **per arrangement**:
    each composite shares within its own subtree, against its own anchor, because
    that is the arrangement the caller asked the question of.

    Returns
    -------
    list of Axes
        Every drawable axes created for this subtree, in layout order.
    """
    from . import _threed

    theme = spec._theme if spec._theme is not None else inherited_theme
    panels = spec.panels
    layout = spec.layout
    rows, cols = _composite_grid(layout, len(panels))
    gs = fig.add_gridspec(rows, cols) if cell is None else cell.subgridspec(rows, cols)

    share_x = bool(getattr(layout, "share_x", False))
    share_y = bool(getattr(layout, "share_y", False))
    share_color = bool(getattr(layout, "share_color", False))
    anchor: Axes | None = None
    flat: list[Axes] = []
    shared_mappable: ScalarMappable | None = None
    shared_colorbar: Colorbar | None = None
    for i, panel in enumerate(panels):
        # Resolve the effective theme LOCALLY (panel theme > composite theme).
        # Do NOT write it back onto the panel spec — rendering must never mutate
        # its input, so a re-render under a different composite theme stays
        # correct and a caller's ``panel._theme is None`` survives the render.
        effective_theme = panel._theme if panel._theme is not None else theme
        sub = gs[i // cols, i % cols]
        if panel.is_composite:
            flat.extend(_place_composite(fig, sub, panel, effective_theme))
            continue
        threed = _threed.is_three_d(panel)
        sub_kw: dict[str, Any] = {}
        if not threed and anchor is not None:
            if share_x:
                sub_kw["sharex"] = anchor
            if share_y:
                sub_kw["sharey"] = anchor
        ax = fig.add_subplot(sub, projection=("3d" if threed else None), **sub_kw)
        flat.append(ax)
        if threed:
            _threed._draw_3d_panel(fig, ax, panel, effective_theme)
        else:
            produced = _draw_2d_panel(fig, ax, panel, effective_theme, colorbar=not share_color)
            if share_color and panel.colorbar is not None and produced is not None:
                shared_mappable = shared_mappable or produced
                shared_colorbar = shared_colorbar or panel.colorbar
            if anchor is None:
                anchor = ax
    if share_color and shared_mappable is not None and shared_colorbar is not None:
        _apply_shared_colorbar(fig, flat, shared_mappable, shared_colorbar)
    _hide_inner_tick_labels(flat, share_x=share_x, share_y=share_y)
    return flat


def _render_composite(spec: PlotSpec, *, figsize: tuple[float, float] | None) -> Figure:
    """Tile a ``COMPOSITE`` spec's ``panels`` into one figure per its ``layout``.

    Each panel is a single-panel :class:`~tsdynamics.viz.spec.PlotSpec` drawn into
    its own axes (a 3-D panel gets an ``mplot3d`` axes); 2-D panels optionally
    share x / y per the :class:`~tsdynamics.viz.spec.Layout`.

    A composite spec may carry its own ``theme``; each panel inherits it when the
    panel has no theme of its own (``panel.theme or composite.theme or get_theme()``).
    """
    from mpl_toolkits import mplot3d  # noqa: F401 — registers the "3d" projection

    # Inherit composite theme onto panels that have no own theme
    composite_theme = _resolve_theme(spec)

    panels = spec.panels
    if not panels:
        # A 0-panel COMPOSITE used to render as a blank figure — a silent no-op
        # that told the caller nothing while looking like a successful plot (it
        # is what ``__plot_spec__(kind="composite")`` produced, discarding the
        # trajectory).  Fail loudly instead.  ``PlotSpec`` enforces the
        # COMPOSITE <=> panels invariant at construction, so this is a backstop.
        raise InvalidParameterError(
            "cannot render a COMPOSITE spec with no panels: build it with "
            "tsdynamics.viz.plot(..., layout=...), which always attaches panels."
        )
    rows, cols = _leaf_extent(spec)
    # Explicit figsize > spec.meta["figsize"] > theme.figsize > the grid-derived
    # default.  (A theme's figsize is a *panel-independent* page size, so it is
    # deliberately allowed to win over the grid heuristic but not over an
    # explicit request.)
    figsize, dpi, layout_engine = figure_geometry(spec, figsize, theme=composite_theme)
    if figsize is None:
        figsize = (cols * 5.0, rows * 3.2)

    fig = new_figure(figsize, dpi, layout_engine)

    # Apply composite-level background to the figure
    if composite_theme.background is not None:
        fig.patch.set_facecolor(composite_theme.background)

    _place_composite(fig, None, spec, composite_theme)
    if spec.title:
        title_kw: dict[str, Any] = {}
        if composite_theme.foreground is not None:
            title_kw["color"] = composite_theme.foreground
        if composite_theme.title_size is not None:
            title_kw["fontsize"] = float(composite_theme.title_size)
        fig.suptitle(spec.title, **title_kw)
    return fig


def render_result(spec: PlotSpec, **kw: Any) -> RenderResult:
    """Render ``spec`` and wrap the figure in a :class:`RenderResult`.

    A thin convenience over :func:`render` for callers that want the typed
    envelope (figure handle + backend name + kind) rather than a bare
    :class:`~matplotlib.figure.Figure`.
    """
    fig = render(spec, **kw)
    return RenderResult(backend="matplotlib", figure=fig, kind=PlotKind(spec.kind))
