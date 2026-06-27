"""matplotlib 3-D marks + camera (stream VIZ-MPL-3D).

The 2-D reference renderer (:mod:`._core`) dispatches a 3-D
:class:`~tsdynamics.viz.spec.PlotSpec` here: a spec is 3-D when its ``ndim`` is
3, it carries a ``z`` axis, or any layer is a ``LINE3D`` / ``SURFACE3D`` mark.
This module draws those marks on an ``mplot3d`` ``Axes3D`` (object-oriented API,
Agg canvas — no ``pyplot``), sets an equal box aspect and the camera, and reuses
the 2-D core's colorbar / legend application.

Theme and style application follow the same three-step contract as :mod:`._core`:

1. Resolve the theme (``spec.theme or get_theme(None)``).
2. Apply theme figure-locally (background, color cycle, font sizes).
3. Per-layer canonical style (via ``normalize_style``) overrides theme defaults;
   ``zorder``, ``alpha``, ``color``, ``linewidth``, ``linestyle``, ``marker``,
   ``markersize`` are all honored.

The component triple is whatever the spec's ``x`` / ``y`` / ``z`` channels carry
(the producer chooses it — e.g. an arbitrary, non-first-three triple for a
Lorenz-96), so this renderer is triple-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import PlotKind, PlotSpec
from ...style import normalize_style
from ._core import (
    _LINESTYLE_MPL,
    _MARKER_MPL,
    _apply_colorbar,
    _apply_theme_color_cycle,
    _apply_theme_to_figure,
    _resolve_theme,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["is_three_d", "render_3d"]

# Default camera (matplotlib's own default elev/azim) when the spec carries none.
_DEFAULT_ELEV = 30.0
_DEFAULT_AZIM = -60.0


def is_three_d(spec: PlotSpec) -> bool:
    """Whether ``spec`` needs 3-D drawing (ndim 3 / a ``z`` axis / a 3-D mark)."""
    if spec.ndim == 3 or spec.z is not None:
        return True
    return any(
        PlotKind(layer.kind) in (PlotKind.LINE3D, PlotKind.SURFACE3D) for layer in spec.layers
    )


def _f(arr: Any) -> np.ndarray:
    """Coerce a channel to a float ``ndarray``."""
    return np.asarray(arr, dtype=float)


def _cmap(spec: PlotSpec) -> str | None:
    """Return the colormap a colour-mapped 3-D layer uses, if the spec sets one."""
    return spec.colorbar.cmap if spec.colorbar is not None else None


def _3d_style(layer: Any, theme: Any) -> dict[str, Any]:
    """Return canonical mpl kwargs for a 3-D layer from its style + theme defaults."""
    canon = normalize_style(layer.style, warn=False)
    kw: dict[str, Any] = {}
    if "color" in canon:
        kw["color"] = canon["color"]
    if "alpha" in canon:
        kw["alpha"] = float(canon["alpha"])
    if "zorder" in canon:
        kw["zorder"] = int(canon["zorder"])
    if "linewidth" in canon:
        kw["lw"] = float(canon["linewidth"])
    elif theme.line_width is not None:
        kw["lw"] = float(theme.line_width)
    if "linestyle" in canon:
        kw["linestyle"] = _LINESTYLE_MPL.get(str(canon["linestyle"]), canon["linestyle"])
    if "marker" in canon:
        kw["marker"] = _MARKER_MPL.get(str(canon["marker"]), canon["marker"])
    if "markersize" in canon:
        kw["ms"] = float(canon["markersize"])
    elif theme.marker_size is not None:
        kw["ms"] = float(theme.marker_size)
    return kw


def _draw_line3d(ax: Any, layer: Any, spec: PlotSpec, theme: Any) -> Any:
    """Draw a 3-D line; colour it by the ``c`` channel via a ``Line3DCollection``."""
    x, y, z = _f(layer.data["x"]), _f(layer.data["y"]), _f(layer.data["z"])
    kw = _3d_style(layer, theme)
    # Rename for plot() which uses 'lw' but we use 'linewidth' in other places
    c = layer.data.get("c")
    if c is not None:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=_cmap(spec), label=layer.label)
        lc.set_array(_f(c)[:-1])
        if spec.clim is not None:
            lc.set_clim(*spec.clim)
        if "lw" in kw:
            lc.set_linewidth(kw["lw"])
        if "alpha" in kw:
            lc.set_alpha(kw["alpha"])
        if "zorder" in kw:
            lc.set_zorder(kw["zorder"])
        ax.add_collection3d(lc)
        ax.auto_scale_xyz(x, y, z)
        return lc
    plot_kw = {k: v for k, v in kw.items() if k in ("color", "lw", "alpha", "linestyle", "zorder")}
    ax.plot(x, y, z, label=layer.label, **plot_kw)
    return None


def _draw_scatter3d(ax: Any, layer: Any, spec: PlotSpec, theme: Any) -> Any:
    """Draw a 3-D scatter, honouring the ``c`` (colour) and ``size`` channels."""
    x, y, z = _f(layer.data["x"]), _f(layer.data["y"]), _f(layer.data["z"])
    kw = _3d_style(layer, theme)
    scatter_kw: dict[str, Any] = {}
    if "color" in kw:
        scatter_kw["color"] = kw["color"]
    if "alpha" in kw:
        scatter_kw["alpha"] = kw["alpha"]
    if "marker" in kw:
        scatter_kw["marker"] = kw["marker"]
    if "zorder" in kw:
        scatter_kw["zorder"] = kw["zorder"]
    c = layer.data.get("c")
    size = layer.data.get("size")
    if size is not None:
        scatter_kw["s"] = _f(size)
    elif "ms" in kw:
        scatter_kw["s"] = kw["ms"]
    if c is not None:
        scatter_kw["c"] = _f(c)
        scatter_kw["cmap"] = _cmap(spec)
    sc = ax.scatter(x, y, z, label=layer.label, **scatter_kw)
    if c is not None and spec.clim is not None:
        sc.set_clim(*spec.clim)
    return sc if c is not None else None


def _draw_surface3d(ax: Any, layer: Any, spec: PlotSpec, theme: Any) -> Any:
    """Draw a parametric surface; ``x``/``y`` may be 1-D axes or 2-D meshes."""
    x, y, z = _f(layer.data["x"]), _f(layer.data["y"]), _f(layer.data["z"])
    if x.ndim == 1 and y.ndim == 1 and z.ndim == 2:
        xx, yy = np.meshgrid(x, y)
    else:
        xx, yy = x, y
    surf = ax.plot_surface(xx, yy, z, cmap=_cmap(spec) or "viridis")
    if spec.clim is not None:
        surf.set_clim(*spec.clim)
    return surf


#: 3-D layer mark → drawing function (now takes theme as extra arg).
#: ``LINE`` / ``MARKERS`` in a 3-D spec are drawn as their 3-D counterparts
#: (they carry a ``z`` channel).
_MARK_3D: dict[PlotKind, Any] = {
    PlotKind.LINE3D: _draw_line3d,
    PlotKind.LINE: _draw_line3d,
    PlotKind.SURFACE3D: _draw_surface3d,
    PlotKind.SCATTER: _draw_scatter3d,
    PlotKind.MARKERS: _draw_scatter3d,
}


def _apply_3d_axes(ax: Any, spec: PlotSpec, theme: Any) -> None:
    """Apply the three axis labels/limits, the title, equal box aspect, and camera.

    Also applies theme font sizes and foreground color to the 3-D axes.
    """
    label_kw: dict[str, Any] = {}
    if theme.foreground is not None:
        label_kw["color"] = theme.foreground
    if theme.font_size is not None:
        label_kw["fontsize"] = float(theme.font_size)

    ax.set_xlabel(spec.x.label, **label_kw)
    ax.set_ylabel(spec.y.label, **label_kw)
    if spec.z is not None:
        ax.set_zlabel(spec.z.label, **label_kw)
    if spec.x.limits is not None:
        ax.set_xlim(*spec.x.limits)
    if spec.y.limits is not None:
        ax.set_ylim(*spec.y.limits)
    if spec.z is not None and spec.z.limits is not None:
        ax.set_zlim(*spec.z.limits)
    if spec.title:
        title_kw: dict[str, Any] = {}
        if theme.foreground is not None:
            title_kw["color"] = theme.foreground
        if theme.title_size is not None:
            title_kw["fontsize"] = float(theme.title_size)
        ax.set_title(spec.title, **title_kw)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    elev, azim = _DEFAULT_ELEV, _DEFAULT_AZIM
    camera = spec.meta.get("camera") if isinstance(spec.meta, dict) else None
    if isinstance(camera, dict):
        elev = float(camera.get("elev", elev))
        azim = float(camera.get("azim", azim))
    ax.view_init(elev=elev, azim=azim)
    if spec._axes_hidden():
        ax.set_axis_off()
    if theme.foreground is not None:
        ax.tick_params(colors=theme.foreground, which="both")


def render_3d(spec: PlotSpec, *, figsize: tuple[float, float] | None = None) -> Figure:
    """Render a 3-D ``spec`` to a matplotlib Figure on an ``mplot3d`` axes.

    Builds a :class:`~matplotlib.figure.Figure` (Agg canvas, no ``pyplot``) with a
    single ``projection="3d"`` axes, draws every layer through :data:`_MARK_3D`,
    then applies the axes / camera and reuses the 2-D core's colorbar + legend.

    The theme is resolved (``spec.theme or get_theme(None)``) and applied
    figure-locally — no global ``rcParams`` mutation.

    Parameters
    ----------
    spec : PlotSpec
        A 3-D spec (see :func:`is_three_d`).
    figsize : tuple of float, optional
        ``(width, height)`` in inches; matplotlib's default when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits import mplot3d  # noqa: F401 — registers the "3d" projection

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
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _draw_3d_panel(fig, ax, spec)
    return fig


def _draw_3d_panel(fig: Any, ax: Any, spec: PlotSpec) -> None:
    """Draw one 3-D spec's marks + axes/colorbar/legend onto an ``mplot3d`` ``ax``.

    The single-panel body of :func:`render_3d`, factored out so the composite
    renderer can draw a 3-D panel into its own axes of a shared figure.

    Resolves the spec's theme and applies it figure-locally (background, color
    cycle, font); each layer's canonical style overrides theme defaults.
    """
    from ._core import _apply_legend

    theme = _resolve_theme(spec)
    _apply_theme_to_figure(fig, ax, theme)
    _apply_theme_color_cycle(ax, theme)

    mappable = None
    for layer in spec.layers:
        drawer = _MARK_3D.get(PlotKind(layer.kind))
        if drawer is None:
            continue
        produced = drawer(ax, layer, spec, theme)
        if produced is not None:
            mappable = produced

    _apply_3d_axes(ax, spec, theme)
    _apply_colorbar(fig, ax, mappable, spec.colorbar)
    _apply_legend(ax, spec, theme)
