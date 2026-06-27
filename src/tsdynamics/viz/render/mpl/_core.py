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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

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
    PlotKind.SPECTROGRAM: _KindPreset(cmap="magma", norm="log"),
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


def _apply_theme_color_cycle(ax: Any, theme: Theme) -> None:
    """Set the axes color cycle from the theme's palette."""
    ax.set_prop_cycle(color=list(theme.palette))


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
# Style coercion helpers
# ---------------------------------------------------------------------------


def _canon_style(layer: Layer, theme: Theme) -> dict[str, Any]:
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

    Returns
    -------
    dict
        Matplotlib artist kwargs (``color``, ``lw``, ``linestyle``, ``marker``,
        ``ms``, ``alpha``, ``zorder``, …).
    """
    canon = normalize_style(layer.style, warn=False)
    kw: dict[str, Any] = {}

    if "color" in canon:
        kw["color"] = canon["color"]
    if "alpha" in canon:
        kw["alpha"] = float(canon["alpha"])
    if "zorder" in canon:
        kw["zorder"] = int(canon["zorder"])
    # NOTE: ``fill`` / ``fillalpha`` are AREA/ENSEMBLE_FAN-only knobs and are
    # consumed directly by ``_draw_area`` (which reads them off ``canon``).  They
    # are deliberately NOT forwarded here: a ``Line2D`` (``ax.plot``) and a
    # ``PathCollection`` (``ax.scatter``) both reject a ``fill=`` kwarg, so
    # leaking it crashes a fully-styled LINE/SCATTER layer.

    # linewidth: canonical key → mpl "linewidth"
    if "linewidth" in canon:
        kw["linewidth"] = float(canon["linewidth"])
    elif theme.line_width is not None:
        kw["linewidth"] = float(theme.line_width)

    # linestyle: canonical → mpl spelling
    if "linestyle" in canon:
        kw["linestyle"] = _LINESTYLE_MPL.get(str(canon["linestyle"]), canon["linestyle"])

    # marker: canonical → mpl spelling
    if "marker" in canon:
        kw["marker"] = _MARKER_MPL.get(str(canon["marker"]), canon["marker"])

    # markersize: canonical key → mpl "markersize"
    if "markersize" in canon:
        kw["markersize"] = float(canon["markersize"])
    elif theme.marker_size is not None:
        kw["markersize"] = float(theme.marker_size)

    return kw


def _line_kwargs(layer: Layer, theme: Theme) -> dict[str, Any]:
    """Collect matplotlib line kwargs from a layer's style + theme defaults."""
    kw = _canon_style(layer, theme)
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
    kw = _line_kwargs(layer, theme)
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
    lw = canon.get("linewidth") or (theme.line_width if theme.line_width is not None else None)
    if lw is not None:
        lc.set_linewidth(float(lw))
    if "zorder" in canon:
        lc.set_zorder(int(canon["zorder"]))
    if "alpha" in canon:
        lc.set_alpha(float(canon["alpha"]))
    if layer.label is not None:
        lc.set_label(layer.label)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


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
    elif "markersize" in canon:
        kw["s"] = float(canon["markersize"])
    elif theme.marker_size is not None:
        kw["s"] = float(theme.marker_size)
    if c is not None:
        kw["c"] = c
        kw["cmap"] = _resolve_cmap(spec, layer, preset)
        kw["norm"] = _make_norm(_resolve_norm(spec, preset), spec.clim)
        return ax.scatter(x, y, **kw)
    if "color" in canon:
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
    _tab20_size = 20  # tab20: swatch i centres at (2i+1)/40

    colors: list[Any] = []
    for v in unique_vals:
        v_int = int(v)
        if diverged_color is not None and v_int == -1:
            colors.append(mcolors.to_rgba(diverged_color))
        elif v_int in palette_index:
            swatch = palette_index[v_int]
            colors.append(base_cmap((2 * swatch + 1) / (2 * _tab20_size)))
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
    """
    x = layer.data.get("x")
    y = layer.data.get("y")
    if x is not None and y is not None:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if xa.ndim == 1 and ya.ndim == 1 and xa.size >= 2 and ya.size >= 2:
            return (float(xa[0]), float(xa[-1]), float(ya[0]), float(ya[-1]))
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
    if layer.label is not None:
        kw["label"] = layer.label
    if show_fill:
        ax.fill_between(x, lo, hi, **kw)
    if y is not None and "lo" in layer.data:
        # A central line through a fan/band, when a distinct ``y`` is supplied.
        line_kw = {k: v for k, v in kw.items() if k in ("color", "label")}
        ax.plot(x, y, **line_kw)
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

    # Spine / axis label color (axis.color overrides theme.foreground)
    if ink is not None:
        if which == "x":
            ax.spines["bottom"].set_edgecolor(ink)
            ax.spines["top"].set_edgecolor(ink)
        else:
            ax.spines["left"].set_edgecolor(ink)
            ax.spines["right"].set_edgecolor(ink)


def _apply_axes(ax: Axes, spec: PlotSpec, preset: _KindPreset, theme: Theme) -> None:
    """Apply both axes, the title, and the aspect ratio to ``ax``."""
    _apply_axis(ax, spec.x, "x", theme)
    _apply_axis(ax, spec.y, "y", theme)
    if spec.title:
        title_kw: dict[str, Any] = {}
        if theme.foreground is not None:
            title_kw["color"] = theme.foreground
        eff_title_size = theme.title_size if theme.title_size is not None else theme.font_size
        if eff_title_size is not None:
            title_kw["fontsize"] = float(eff_title_size)
        if theme.font_family is not None:
            title_kw["fontfamily"] = theme.font_family
        ax.set_title(spec.title, **title_kw)
    aspect = spec.aspect if spec.aspect != "auto" else preset.aspect
    if aspect == "equal":
        ax.set_aspect("equal", adjustable="box")
    if spec._axes_hidden():
        ax.set_axis_off()


def _apply_colorbar(
    fig: Figure, ax: Axes, mappable: ScalarMappable | None, colorbar: Colorbar | None
) -> None:
    """Attach a colorbar for ``mappable`` honouring a :class:`Colorbar` spec.

    Now also honors ``colorbar.label_size`` (the new enriched field).
    """
    if mappable is None or colorbar is None or not colorbar.show:
        return
    cb = fig.colorbar(mappable, ax=ax, location=colorbar.location)
    if colorbar.label:
        label_kw: dict[str, Any] = {}
        if colorbar.label_size is not None:
            label_kw["fontsize"] = float(colorbar.label_size)
        cb.set_label(colorbar.label, **label_kw)
    if colorbar.ticks is not None:
        cb.set_ticks(list(colorbar.ticks))
    if colorbar.tickformat is not None:
        import matplotlib.ticker as mticker

        fmt_str = colorbar.tickformat
        if "{" in fmt_str:
            cb.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(fmt_str))
        else:
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt_str))
    if colorbar.label_size is not None:
        cb.ax.tick_params(labelsize=float(colorbar.label_size))


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


def _apply_annotations(ax: Axes, annotations: list[Annotation]) -> None:
    """Draw reference lines / text / spans (``vline`` / ``hline`` / ``text`` / ``span``)."""
    for ann in annotations:
        style = dict(ann.style)
        if ann.kind == "vline" and ann.x is not None:
            ax.axvline(ann.x, label=ann.text or None, **style)
        elif ann.kind == "hline" and ann.y is not None:
            ax.axhline(ann.y, label=ann.text or None, **style)
        elif ann.kind == "text" and ann.x is not None and ann.y is not None:
            ax.text(ann.x, ann.y, ann.text, **style)
        elif ann.kind == "span" and ann.span is not None:
            lo, hi = ann.span
            if ann.axis == "y":
                ax.axhspan(lo, hi, **style)
            else:
                ax.axvspan(lo, hi, **style)


# ---------------------------------------------------------------------------
# The render entry point
# ---------------------------------------------------------------------------


def render(
    spec: PlotSpec, *, figsize: tuple[float, float] | None = None, **_kw: Any
) -> Figure | FuncAnimation:
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
    **_kw
        Forwarded but unused backend keywords (kept for a uniform renderer
        signature).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure (a single axes), ready to ``savefig`` / embed.  A
        3-D spec (``ndim == 3`` / a ``z`` axis / a ``LINE3D`` / ``SURFACE3D``
        mark) is dispatched to the :mod:`._threed` renderer.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from . import _threed

    if spec.is_animated:
        from . import _anim

        return _anim.render_animation(spec, figsize=figsize)

    if spec.is_composite:
        return _render_composite(spec, figsize=figsize)

    if _threed.is_three_d(spec):
        return _threed.render_3d(spec, figsize=figsize)

    # Resolve meta figsize / dpi
    meta_figsize = spec.meta.get("figsize") if isinstance(spec.meta, dict) else None
    if figsize is None and meta_figsize is not None:
        w, h = meta_figsize
        if w is not None and h is not None:
            figsize = (float(w), float(h))

    dpi: float | None = None
    if isinstance(spec.meta, dict) and "dpi" in spec.meta:
        dpi = float(spec.meta["dpi"])

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    _draw_2d_panel(fig, ax, spec)
    return fig


def _draw_2d_panel(fig: Figure, ax: Axes, spec: PlotSpec, theme: Theme | None = None) -> None:
    """Draw one 2-D spec's layers + axes/colorbar/legend/annotations onto ``ax``.

    The single-panel body of :func:`render`, factored out so the composite
    renderer can draw each panel into its own axes of a shared figure.

    Step 1: resolve theme and apply it figure-locally (background, color cycle,
    font, default grid).  Step 2: draw layers with normalized per-layer style
    overriding theme defaults.  Step 3: apply enriched Axis/Legend/Colorbar fields.

    ``theme`` lets the composite renderer pass an *inherited* theme (its own) for a
    panel that has none, **without mutating the panel spec** — when ``None`` the
    panel's own resolved theme is used.
    """
    theme = theme if theme is not None else _resolve_theme(spec)
    preset = _preset_for(normalize_kind(spec.kind))

    # Step 1: apply theme presentation
    _apply_theme_to_figure(fig, ax, theme)
    _apply_theme_color_cycle(ax, theme)

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
    _apply_colorbar(fig, ax, mappable, spec.colorbar)
    _apply_legend(ax, spec, theme)
    _apply_annotations(ax, spec.annotations)


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


def _render_composite(spec: PlotSpec, *, figsize: tuple[float, float] | None) -> Figure:
    """Tile a ``COMPOSITE`` spec's ``panels`` into one figure per its ``layout``.

    Each panel is a single-panel :class:`~tsdynamics.viz.spec.PlotSpec` drawn into
    its own axes (a 3-D panel gets an ``mplot3d`` axes); 2-D panels optionally
    share x / y per the :class:`~tsdynamics.viz.spec.Layout`.

    A composite spec may carry its own ``theme``; each panel inherits it when the
    panel has no theme of its own (``panel.theme or composite.theme or get_theme()``).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits import mplot3d  # noqa: F401 — registers the "3d" projection

    from . import _threed

    # Inherit composite theme onto panels that have no own theme
    composite_theme = _resolve_theme(spec)

    panels = spec.panels
    layout = spec.layout
    if not panels:
        # A composite with no panels is degenerate; emit a valid (empty) figure.
        fig = Figure(figsize=figsize)
        FigureCanvasAgg(fig)
        fig.add_subplot(1, 1, 1)
        return fig
    rows, cols = _composite_grid(layout, len(panels))
    if figsize is None:
        figsize = (cols * 5.0, rows * 3.2)

    # Composite-level dpi
    dpi: float | None = None
    if isinstance(spec.meta, dict) and "dpi" in spec.meta:
        dpi = float(spec.meta["dpi"])

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)

    # Apply composite-level background to the figure
    if composite_theme.background is not None:
        fig.patch.set_facecolor(composite_theme.background)

    share_x = bool(getattr(layout, "share_x", False))
    share_y = bool(getattr(layout, "share_y", False))
    anchor: Axes | None = None
    for i, panel in enumerate(panels):
        # Resolve the effective theme LOCALLY (panel theme > composite theme).
        # Do NOT write it back onto the panel spec — rendering must never mutate
        # its input, so a re-render under a different composite theme stays
        # correct and a caller's ``panel._theme is None`` survives the render.
        effective_theme = panel._theme if panel._theme is not None else composite_theme
        threed = _threed.is_three_d(panel)
        sub_kw: dict[str, Any] = {}
        if not threed and anchor is not None:
            if share_x:
                sub_kw["sharex"] = anchor
            if share_y:
                sub_kw["sharey"] = anchor
        ax = fig.add_subplot(rows, cols, i + 1, projection=("3d" if threed else None), **sub_kw)
        if threed:
            _threed._draw_3d_panel(fig, ax, panel, effective_theme)
        else:
            _draw_2d_panel(fig, ax, panel, effective_theme)
            if anchor is None:
                anchor = ax
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
