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

3-D marks (``LINE3D`` / ``SURFACE3D``) are drawn by a *follow-up* stream
(VIZ-MPL-3D).  This core raises a clear :class:`NotImplementedError` for a 3-D
mark so the capability declaration (``supports_3d=True``) routes 3-D here while
the drawing is filled in later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import Annotation, Axis, Colorbar, Layer, PlotKind, PlotSpec
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
}

_DEFAULT_PRESET = _KindPreset()


def _preset_for(kind: PlotKind) -> _KindPreset:
    """Return the preset for ``kind`` (the neutral default if none is declared)."""
    return KIND_PRESETS.get(kind, _DEFAULT_PRESET)


# ---------------------------------------------------------------------------
# Style coercion helpers
# ---------------------------------------------------------------------------

#: Neutral style keys that map straight onto matplotlib artist kwargs.
_PASSTHROUGH_STYLE = ("color", "alpha", "lw", "linewidth", "linestyle", "ls", "zorder")


def _line_kwargs(layer: Layer) -> dict[str, Any]:
    """Collect matplotlib line kwargs from a layer's neutral style."""
    style = layer.style
    kw: dict[str, Any] = {}
    for key in _PASSTHROUGH_STYLE:
        if key in style:
            kw[key] = style[key]
    if layer.label is not None:
        kw["label"] = layer.label
    return kw


def _resolve_cmap(spec: PlotSpec, layer: Layer, preset: _KindPreset) -> str | None:
    """Pick the colormap: layer style > spec colorbar > kind preset > backend default."""
    cmap = layer.style.get("cmap")
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

#: A mark-drawing function takes ``(ax, layer, spec, preset)`` and draws the
#: layer, returning a colour-mappable artist when it produced one (for the
#: colorbar) else ``None``.
_MarkDrawer = Callable[["Axes", Layer, PlotSpec, _KindPreset], "ScalarMappable | None"]


def _channel(layer: Layer, name: str) -> np.ndarray | None:
    """Return a layer's channel array as float, or ``None`` if absent."""
    arr = layer.data.get(name)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _draw_line(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
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
        return _draw_colored_line(ax, x, y, c, spec, layer, preset)
    ax.plot(x, y, **_line_kwargs(layer))
    return None


def _draw_colored_line(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    spec: PlotSpec,
    layer: Layer,
    preset: _KindPreset,
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
    lw = layer.style.get("lw", layer.style.get("linewidth"))
    if lw is not None:
        lc.set_linewidth(lw)
    if layer.label is not None:
        lc.set_label(layer.label)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def _draw_scatter(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
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
    kw: dict[str, Any] = {}
    if "alpha" in layer.style:
        kw["alpha"] = layer.style["alpha"]
    if "marker" in layer.style:
        kw["marker"] = layer.style["marker"]
    if layer.label is not None:
        kw["label"] = layer.label
    if size is not None:
        kw["s"] = size
    elif "s" in layer.style:
        kw["s"] = layer.style["s"]
    if c is not None:
        kw["c"] = c
        kw["cmap"] = _resolve_cmap(spec, layer, preset)
        kw["norm"] = _make_norm(_resolve_norm(spec, preset), spec.clim)
        return ax.scatter(x, y, **kw)
    if "color" in layer.style:
        kw["color"] = layer.style["color"]
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
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
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
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
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
    kw: dict[str, Any] = {}
    if "color" in layer.style:
        kw["color"] = layer.style["color"]
    if "alpha" in layer.style:
        kw["alpha"] = layer.style["alpha"]
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


def _draw_bar(ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset) -> ScalarMappable | None:
    """Draw a ``BAR`` mark — values ``y`` at positions ``x`` / ``cat``."""
    y = _channel(layer, "y")
    if y is None:
        return None
    x = _channel(layer, "cat")
    if x is None:
        x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    kw: dict[str, Any] = {}
    if "color" in layer.style:
        kw["color"] = layer.style["color"]
    if "alpha" in layer.style:
        kw["alpha"] = layer.style["alpha"]
    if layer.label is not None:
        kw["label"] = layer.label
    ax.bar(x, y, **kw)
    return None


def _draw_area(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
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
    kw: dict[str, Any] = {"alpha": layer.style.get("alpha", 0.3)}
    if "color" in layer.style:
        kw["color"] = layer.style["color"]
    if layer.label is not None:
        kw["label"] = layer.label
    ax.fill_between(x, lo, hi, **kw)
    if y is not None and "lo" in layer.data:
        # A central line through a fan/band, when a distinct ``y`` is supplied.
        line_kw = {k: v for k, v in kw.items() if k in ("color", "label")}
        ax.plot(x, y, **line_kw)
    return None


def _draw_errorbar(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
) -> ScalarMappable | None:
    """Draw an ``ERRORBAR`` mark — ``y`` vs ``x`` with symmetric ``err`` bars."""
    y = _channel(layer, "y")
    if y is None:
        return None
    x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    err = _channel(layer, "err")
    kw: dict[str, Any] = {"fmt": layer.style.get("marker", "o")}
    if "color" in layer.style:
        kw["color"] = layer.style["color"]
    if "alpha" in layer.style:
        kw["alpha"] = layer.style["alpha"]
    if layer.label is not None:
        kw["label"] = layer.label
    ax.errorbar(x, y, yerr=err, **kw)
    return None


def _draw_quiver(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
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
    if c is not None:
        cmap = _resolve_cmap(spec, layer, preset)
        norm = _make_norm(_resolve_norm(spec, preset), spec.clim)
        q = ax.quiver(x, y, u, v, c, cmap=cmap, norm=norm)
        return q
    kw: dict[str, Any] = {}
    if "color" in layer.style:
        kw["color"] = layer.style["color"]
    ax.quiver(x, y, u, v, **kw)
    return None


def _draw_3d_unsupported(
    ax: Axes, layer: Layer, spec: PlotSpec, preset: _KindPreset
) -> ScalarMappable | None:
    """Raise for a 3-D mark — drawn by the follow-up VIZ-MPL-3D stream."""
    raise NotImplementedError(
        f"the matplotlib reference renderer draws 2-D marks only; {layer.kind.value!r} "
        "(3-D) is added by the VIZ-MPL-3D stream."
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


def _apply_axis(ax: Axes, axis: Axis, which: str) -> None:
    """Apply one :class:`~tsdynamics.viz.spec.Axis` (label/scale/limits/ticks/categories)."""
    set_label = ax.set_xlabel if which == "x" else ax.set_ylabel
    set_scale = ax.set_xscale if which == "x" else ax.set_yscale
    set_lim = ax.set_xlim if which == "x" else ax.set_ylim
    set_ticks = ax.set_xticks if which == "x" else ax.set_yticks
    set_ticklabels = ax.set_xticklabels if which == "x" else ax.set_yticklabels

    if axis.label:
        set_label(axis.label)
    if axis.scale in ("log", "symlog"):
        set_scale(axis.scale)
    elif axis.scale == "categorical" and axis.categories is not None:
        positions = np.arange(len(axis.categories), dtype=float)
        set_ticks(positions)
        set_ticklabels(list(axis.categories))
    if axis.limits is not None:
        set_lim(axis.limits[0], axis.limits[1])
    if axis.ticks is not None:
        set_ticks(list(axis.ticks))


def _apply_axes(ax: Axes, spec: PlotSpec, preset: _KindPreset) -> None:
    """Apply both axes, the title, and the aspect ratio to ``ax``."""
    _apply_axis(ax, spec.x, "x")
    _apply_axis(ax, spec.y, "y")
    if spec.title:
        ax.set_title(spec.title)
    aspect = spec.aspect if spec.aspect != "auto" else preset.aspect
    if aspect == "equal":
        ax.set_aspect("equal", adjustable="box")
    if spec._axes_hidden():
        ax.set_axis_off()


def _apply_colorbar(
    fig: Figure, ax: Axes, mappable: ScalarMappable | None, colorbar: Colorbar | None
) -> None:
    """Attach a colorbar for ``mappable`` honouring a :class:`Colorbar` spec."""
    if mappable is None or colorbar is None or not colorbar.show:
        return
    cb = fig.colorbar(mappable, ax=ax, location=colorbar.location)
    if colorbar.label:
        cb.set_label(colorbar.label)
    if colorbar.ticks is not None:
        cb.set_ticks(list(colorbar.ticks))


def _apply_legend(ax: Axes, spec: PlotSpec) -> None:
    """Draw the per-layer legend when the spec asks for one and labels exist."""
    if spec.legend is None or not spec.legend.show:
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    title = spec.legend.title or None
    ax.legend(loc=spec.legend.location, title=title)


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
    ``pyplot``), draws every :class:`~tsdynamics.viz.spec.Layer` through
    :data:`MARK_DISPATCH`, then applies the axes, colorbar, legend and
    annotations the spec carries.  The spec's semantic kind is normalised through
    :func:`tsdynamics.viz.render.normalize_kind` so an alias / mark spelling
    resolves to the preset table.

    Parameters
    ----------
    spec : PlotSpec
        The backend-agnostic spec to draw.  Its per-call tweaks
        (relabel/rescale/limits/ticks/colorize) are already baked into the typed
        axes / colorbar, so honouring those honours the tweaks.
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

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    _draw_2d_panel(fig, ax, spec)
    return fig


def _draw_2d_panel(fig: Figure, ax: Axes, spec: PlotSpec) -> None:
    """Draw one 2-D spec's layers + axes/colorbar/legend/annotations onto ``ax``.

    The single-panel body of :func:`render`, factored out so the composite
    renderer can draw each panel into its own axes of a shared figure.
    """
    preset = _preset_for(normalize_kind(spec.kind))
    mappable: ScalarMappable | None = None
    for layer in spec.layers:
        drawer = MARK_DISPATCH.get(PlotKind(layer.kind))
        if drawer is None:
            continue
        produced = drawer(ax, layer, spec, preset)
        if produced is not None:
            mappable = produced
    _apply_axes(ax, spec, preset)
    _apply_colorbar(fig, ax, mappable, spec.colorbar)
    _apply_legend(ax, spec)
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
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits import mplot3d  # noqa: F401 — registers the "3d" projection

    from . import _threed

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
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)

    share_x = bool(getattr(layout, "share_x", False))
    share_y = bool(getattr(layout, "share_y", False))
    anchor: Axes | None = None
    for i, panel in enumerate(panels):
        threed = _threed.is_three_d(panel)
        sub_kw: dict[str, Any] = {}
        if not threed and anchor is not None:
            if share_x:
                sub_kw["sharex"] = anchor
            if share_y:
                sub_kw["sharey"] = anchor
        ax = fig.add_subplot(rows, cols, i + 1, projection=("3d" if threed else None), **sub_kw)
        if threed:
            _threed._draw_3d_panel(fig, ax, panel)
        else:
            _draw_2d_panel(fig, ax, panel)
            if anchor is None:
                anchor = ax
    if spec.title:
        fig.suptitle(spec.title)
    return fig


def render_result(spec: PlotSpec, **kw: Any) -> RenderResult:
    """Render ``spec`` and wrap the figure in a :class:`RenderResult`.

    A thin convenience over :func:`render` for callers that want the typed
    envelope (figure handle + backend name + kind) rather than a bare
    :class:`~matplotlib.figure.Figure`.
    """
    fig = render(spec, **kw)
    return RenderResult(backend="matplotlib", figure=fig, kind=PlotKind(spec.kind))
