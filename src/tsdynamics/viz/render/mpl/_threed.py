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

from ...producers import autostyle_enabled, autostyle_line
from ...spec import PlotKind, PlotSpec
from ...style import Theme, normalize_style
from ._core import (
    _LINESTYLE_MPL,
    _MARKER_MPL,
    _apply_colorbar,
    _apply_theme_color_cycle,
    _apply_theme_to_figure,
    _KindPreset,
    _make_norm,
    _resolve_cmap,
    _resolve_norm,
    _resolve_theme,
    figure_geometry,
    new_figure,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["is_three_d", "render_3d"]

# Default camera (matplotlib's own default elev/azim) when the spec carries none.
_DEFAULT_ELEV = 30.0
_DEFAULT_AZIM = -60.0


def is_three_d(spec: PlotSpec) -> bool:
    """Whether ``spec`` needs 3-D drawing (ndim 3 / a ``z`` axis / a 3-D mark).

    A thin alias for :attr:`~tsdynamics.viz.spec.PlotSpec.is_three_d`, the single
    definition every backend and the capability check now share.  It used to be a
    byte-identical copy of that predicate; the copies could drift, and a renderer
    that disagreed with the dispatcher about what "3-D" means routes a spec to an
    axes it cannot draw on.  The name is kept because it is this module's public
    entry point (``_core`` / ``_anim`` / the plotly composite ask it).
    """
    return spec.is_three_d


def _f(arr: Any) -> np.ndarray:
    """Coerce a channel to a float ``ndarray``."""
    return np.asarray(arr, dtype=float)


#: The 3-D marks carry no per-kind colour preset — a 3-D spec's colour comes from
#: the layer, the spec's colorbar, or the backend default, in that order.
_NO_PRESET = _KindPreset()


def _cmap(spec: PlotSpec, layer: Any = None) -> str | None:
    """Return the colormap this 3-D layer uses — **layer style first**.

    Shares :func:`~._core._resolve_cmap` with the 2-D renderer rather than
    re-deriving the precedence, because the copy that used to live here read
    only ``spec.colorbar.cmap``: ``ts.plot(traj, color_by="time",
    cmap="plasma")`` wrote ``cmap`` onto the *layer* (where the front door puts
    every style key) and the 3-D path never looked there, so the picture came
    out viridis with **no warning** — a declared style key, silently dropped, on
    the 106-of-142 catalogue systems that are 3-D.  The identical call in 2-D
    honored it, which is what made the drop invisible.
    """
    if layer is None:
        return spec.colorbar.cmap if spec.colorbar is not None else None
    return _resolve_cmap(spec, layer, _NO_PRESET)


def _norm(spec: PlotSpec) -> Any:
    """Return the matplotlib colour norm for this spec (``clim`` folded in)."""
    return _make_norm(_resolve_norm(spec, _NO_PRESET), spec.clim)


def _3d_style(
    layer: Any, theme: Any, *, n: int | None = None, autostyle: bool = True
) -> dict[str, Any]:
    """Return canonical mpl kwargs for a 3-D layer from its style + theme defaults.

    When ``n`` (the curve's sample count) is given, the theme's ``line_width`` /
    full-opacity defaults are resolved through
    :func:`~tsdynamics.viz.producers.autostyle_line` — see that function for why
    a constant stroke is the wrong default for a long chaotic trajectory.  An
    explicit ``linewidth`` / ``alpha`` on the layer always wins.
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
    if "linewidth" in canon:
        kw["lw"] = float(canon["linewidth"])
    elif auto_lw is not None:
        kw["lw"] = float(auto_lw)
    if "linestyle" in canon:
        kw["linestyle"] = _LINESTYLE_MPL.get(str(canon["linestyle"]), canon["linestyle"])
    if "marker" in canon:
        kw["marker"] = _MARKER_MPL.get(str(canon["marker"]), canon["marker"])
        # Mirror the 2-D ``_canon_style``: ``filled=False`` is a hollow marker.
        # (The 3-D *scatter* spells it with facecolors/edgecolors; see
        # ``_draw_scatter3d``, which builds its own kwargs and ignores this one.)
        if canon.get("filled") is False:
            kw["markerfacecolor"] = "none"
    if "markersize" in canon:
        kw["ms"] = float(canon["markersize"])
    elif theme.marker_size is not None:
        kw["ms"] = float(theme.marker_size)
    return kw


def _xyz(layer: Any) -> tuple[Any, Any, Any]:
    """Read a layer's ``x``/``y``/``z``, lifting a **flat** layer onto ``z = 0``.

    CONTRACT §6.5 [M18]: a hand-built geometry is stamped
    :data:`~tsdynamics.viz._frames.FrameSpace.FREE` — *"I did not say what space
    this is"* — and overlays with anything, so ``ts.plot(traj,
    "phase_portrait").add(ts.viz.draw({"x": …, "y": …}, "line"))`` must draw.
    Before this it raised a bare ``KeyError: 'z'`` from inside the renderer,
    which punished precisely the person who used the escape hatch.  A 2-D curve
    in a 3-D box has exactly one honest reading: the ``z = 0`` plane.
    """
    x, y = _f(layer.data["x"]), _f(layer.data["y"])
    raw = layer.data.get("z")
    z = np.zeros_like(x) if raw is None else _f(raw)
    return x, y, z


def _draw_line3d(ax: Any, layer: Any, spec: PlotSpec, theme: Any) -> Any:
    """Draw a 3-D line; colour it by the ``c`` channel via a ``Line3DCollection``."""
    x, y, z = _xyz(layer)
    kw = _3d_style(layer, theme, n=int(x.size), autostyle=autostyle_enabled(spec))
    # Rename for plot() which uses 'lw' but we use 'linewidth' in other places
    c = layer.data.get("c")
    if c is not None:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(
            segments, cmap=_cmap(spec, layer), norm=_norm(spec), label=layer.label
        )
        lc.set_array(_f(c)[:-1])
        if "lw" in kw:
            lc.set_linewidth(kw["lw"])
        if "alpha" in kw:
            lc.set_alpha(kw["alpha"])
        if "zorder" in kw:
            lc.set_zorder(kw["zorder"])
        ax.add_collection3d(lc)
        ax.auto_scale_xyz(x, y, z)
        return lc
    # ``marker`` / ``ms`` / ``markerfacecolor`` belong in this whitelist: the 2-D
    # LINE path honors them, ``STYLE_KEYS["marker"].honored_by`` claims matplotlib
    # unconditionally, and 3-D is 106 of the 136 catalogue ODEs — so leaving them
    # out made ``.style(marker="o")`` on a 3-D trajectory an accepted, unwarned,
    # invisible request.  They only ever appear when the caller asked for them.
    plot_kw = {
        k: v
        for k, v in kw.items()
        if k in ("color", "lw", "alpha", "linestyle", "zorder", "marker", "ms", "markerfacecolor")
    }
    ax.plot(x, y, z, label=layer.label, **plot_kw)
    return None


def _draw_scatter3d(ax: Any, layer: Any, spec: PlotSpec, theme: Any) -> Any:
    """Draw a 3-D scatter, honouring the ``c`` (colour) and ``size`` channels."""
    x, y, z = _xyz(layer)
    kw = _3d_style(layer, theme)
    canon = normalize_style(layer.style, warn=False)
    scatter_kw: dict[str, Any] = {}
    # ``filled=False`` draws a hollow/open marker (the unstable-fixed-point
    # convention).  ``STYLE_KEYS["filled"].honored_by`` claims matplotlib
    # unconditionally, but only the 2-D scatter honored it, so a genuinely 3-D
    # spec both ignored the request *and* — because the claim is unconditional —
    # emitted no VisualizationDegraded warning about it.  Same spelling as the
    # 2-D path: no facecolor, ink in the edge.
    if canon.get("filled") is False:
        scatter_kw["facecolors"] = "none"
        scatter_kw["edgecolors"] = canon.get("color", theme.foreground or "C0")
        scatter_kw.setdefault("linewidths", 1.2)
    elif "color" in kw:
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
        # ``ms`` is a marker *diameter* (pt, the Line2D convention) but scatter's
        # ``s`` is an *area* (pt²) — square it, exactly as the 2-D path does, so a
        # canonical ``markersize`` means the same thing in 2-D and 3-D.
        scatter_kw["s"] = float(kw["ms"]) ** 2
    if c is not None:
        scatter_kw["c"] = _f(c)
        scatter_kw["cmap"] = _cmap(spec, layer)
        scatter_kw["norm"] = _norm(spec)
    sc = ax.scatter(x, y, z, label=layer.label, **scatter_kw)
    return sc if c is not None else None


def _draw_surface3d(ax: Any, layer: Any, spec: PlotSpec, theme: Any) -> Any:
    """Draw a parametric surface; ``x``/``y`` may be 1-D axes or 2-D meshes."""
    x, y, z = _f(layer.data["x"]), _f(layer.data["y"]), _f(layer.data["z"])
    if x.ndim == 1 and y.ndim == 1 and z.ndim == 2:
        xx, yy = np.meshgrid(x, y)
    else:
        xx, yy = x, y
    surf = ax.plot_surface(xx, yy, z, cmap=_cmap(spec, layer) or "viridis", norm=_norm(spec))
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


def _apply_3d_annotations(ax: Any, spec: PlotSpec, theme: Any) -> None:
    """Draw ``spec.annotations`` onto a 3-D axes.

    The 3-D renderer used to **silently drop** every annotation: the 2-D core
    called ``_apply_annotations``, this module never did.  A reference line or a
    marked value on a 3-D spec simply vanished, on the majority (106 of 136) of
    the catalogue's ODEs.

    The 2-D helper cannot be reused verbatim — ``Axes3D.text`` takes
    ``(x, y, z, s)`` and ``get_xaxis_transform`` has no 3-D meaning — so the
    primitives are re-expressed in 3-D:

    - ``text`` is placed at ``(x, y)`` on the mid-``z`` plane;
    - ``vline`` / ``hline`` become a **reference line** drawn across the axes at
      the constant coordinate, on the mid-``z`` plane (in 3-D the honest object
      would be a plane, but a plane occludes the attractor — a line reads);
    - ``span`` is a 2-D band with no 3-D analogue that does not occlude, so it is
      skipped.

    Note the annotations are applied **after** the layers and axis limits, and
    the limits are frozen first, so an annotation never rescales the view.
    """
    annotations = list(spec.annotations)
    if not annotations:
        return
    xlo, xhi = ax.get_xlim3d()
    ylo, yhi = ax.get_ylim3d()
    zlo, zhi = ax.get_zlim3d()
    zmid = 0.5 * (zlo + zhi)
    default_color = theme.foreground if theme.foreground is not None else "0.4"
    for ann in annotations:
        style = dict(ann.style)
        color = style.get("color", default_color)
        alpha = float(style.get("alpha", 0.7))
        if ann.kind == "text" and ann.x is not None and ann.y is not None:
            ax.text(float(ann.x), float(ann.y), zmid, ann.text, color=color)
        elif ann.kind == "vline" and ann.x is not None:
            ax.plot(
                [float(ann.x), float(ann.x)],
                [ylo, yhi],
                [zmid, zmid],
                color=color,
                alpha=alpha,
                linestyle="--",
                label=ann.text or None,
            )
        elif ann.kind == "hline" and ann.y is not None:
            ax.plot(
                [xlo, xhi],
                [float(ann.y), float(ann.y)],
                [zmid, zmid],
                color=color,
                alpha=alpha,
                linestyle="--",
                label=ann.text or None,
            )
        # "span" has no non-occluding 3-D analogue — deliberately skipped.
    ax.set_xlim3d(xlo, xhi)
    ax.set_ylim3d(ylo, yhi)
    ax.set_zlim3d(zlo, zhi)


def _apply_3d_panes(ax: Any, theme: Any) -> None:
    """Theme the three background panes + grid of an ``mplot3d`` axes.

    Without this a dark theme produced a **light-grey 3-D box floating in a dark
    page**: ``_apply_theme_to_figure`` sets the *figure* and 2-D axes facecolor,
    but ``mplot3d`` draws its own ``xaxis.pane`` / ``yaxis.pane`` / ``zaxis.pane``
    quads, which keep matplotlib's own light default regardless of the theme.
    """
    background = theme.background
    if background is None and theme.grid_color is None:
        return
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane = getattr(axis, "pane", None)
        if pane is not None and background is not None:
            pane.set_facecolor(background)
            pane.set_edgecolor(theme.grid_color or theme.foreground or background)
            pane.set_alpha(1.0)
        if theme.grid_color is not None:
            axis._axinfo["grid"]["color"] = theme.grid_color


def _apply_3d_scale_and_ticks(ax: Any, axis: Any, which: str) -> None:
    """Apply one axis's ``scale`` / ``ticks`` / ``tickformat`` to a 3-D axes.

    The 3-D renderer used to apply the three **labels** and the three **limits**
    and nothing else, so six of the seventeen figure keywords — ``xticks`` /
    ``yticks`` / ``zticks`` / ``xscale`` / ``yscale`` / ``zscale`` — were honoured
    on a 2-D axes and silently dropped on a 3-D one.  Measured before v6 round 8:
    ``ts.plot(traj, components=("x","y","z"), zticks=[10, 40])`` rendered
    ``ax.get_zticks() == [-10, 0, 10, 20, 30, 40, 50]`` and the picture was
    bit-identical to the one without the keyword.  106 of the catalogue's ODEs
    are 3-D, so this was the *common* axes.

    ``Axes3D`` takes ``set_zscale`` / ``set_zticks`` / ``ax.zaxis`` exactly as the
    flat axes takes their x/y counterparts, so this is the same application the
    2-D core does (``_core._apply_axis``), reached for all three axes.
    """
    import matplotlib.ticker as mticker

    if axis is None:
        return
    if axis.scale in ("log", "symlog"):
        getattr(ax, f"set_{which}scale")(axis.scale)
    elif axis.scale == "categorical" and axis.categories is not None:
        getattr(ax, f"set_{which}ticks")(np.arange(len(axis.categories), dtype=float))
        getattr(ax, f"set_{which}ticklabels")(list(axis.categories))
    if axis.ticks is not None:
        getattr(ax, f"set_{which}ticks")(list(axis.ticks))
    if axis.tickformat is not None:
        formatter = (
            mticker.StrMethodFormatter(axis.tickformat)
            if "{" in axis.tickformat
            else mticker.FormatStrFormatter(axis.tickformat)
        )
        getattr(ax, f"{which}axis").set_major_formatter(formatter)


def _apply_3d_axes(ax: Any, spec: PlotSpec, theme: Any) -> None:
    """Apply the three axes (labels, limits, scales, ticks), title, aspect and camera.

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
    # Scales and ticks BEFORE the limits: setting a log scale resets the view
    # interval, so a limit applied first would be thrown away.
    for axis, which in ((spec.x, "x"), (spec.y, "y"), (spec.z, "z")):
        _apply_3d_scale_and_ticks(ax, axis, which)
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
    from mpl_toolkits import mplot3d  # noqa: F401 — registers the "3d" projection

    figsize, dpi, layout = figure_geometry(spec, figsize)
    fig = new_figure(figsize, dpi, layout)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _draw_3d_panel(fig, ax, spec)
    return fig


def _draw_3d_panel(fig: Any, ax: Any, spec: PlotSpec, theme: Theme | None = None) -> None:
    """Draw one 3-D spec's marks + axes/colorbar/legend onto an ``mplot3d`` ``ax``.

    The single-panel body of :func:`render_3d`, factored out so the composite
    renderer can draw a 3-D panel into its own axes of a shared figure.

    Resolves the spec's theme and applies it figure-locally (background, color
    cycle, font); each layer's canonical style overrides theme defaults.

    ``theme`` lets the composite renderer pass an *inherited* theme (its own) for a
    panel that has none, **without mutating the panel spec** — when ``None`` the
    panel's own resolved theme is used.
    """
    from ._core import _apply_legend

    theme = theme if theme is not None else _resolve_theme(spec)
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
    _apply_3d_panes(ax, theme)
    # 3-D used to silently drop ``spec.annotations`` — the 2-D core applied them,
    # this one never called any helper — so a reference line / marked value on a
    # 3-D spec (106 of the 136 catalogue ODEs are 3-D) vanished with no warning.
    _apply_3d_annotations(ax, spec, theme)
    _apply_colorbar(fig, ax, mappable, spec.colorbar, spec.meta)
    _apply_legend(ax, spec, theme)
