"""The plotly 3-D interactive marks + camera (stream PLOTLY-3D).

The plotly 2-D renderer (:mod:`._core`) dispatches a 3-D
:class:`~tsdynamics.viz.spec.PlotSpec` here: a spec is 3-D when its ``ndim`` is
3, it carries a ``z`` axis, or any layer is a ``LINE3D`` / ``SURFACE3D`` mark.
This module draws those marks as ``graph_objects`` 3-D traces — ``go.Scatter3d``
(a ``mode="lines"`` poly-line or a ``mode="markers"`` colour-by-``c`` scatter) and
``go.Surface`` (a parametric surface) — on an **orbitable** ``scene`` (drag to
rotate, scroll to zoom), so a Lorenz attractor is explorable in the browser.

The renderer uses plotly's ``graph_objects`` API **only** (no ``plotly.express``,
no ``kaleido``).  The component triple is whatever the spec's ``x`` / ``y`` /
``z`` channels carry (the producer chooses it — an arbitrary, non-first-three
triple for a Lorenz-96), so this renderer is triple-agnostic.

The resolved :class:`~tsdynamics.viz.style.Theme` is applied to the 3-D scene:
``paper_bgcolor`` / ``plot_bgcolor`` from ``theme.background``, ``layout.font``
from ``theme.font_family`` / ``theme.font_size``.  Per-layer canonical style keys
(``linewidth`` → ``line.width``, ``linestyle`` → ``line.dash``,
``marker`` / ``markersize`` → ``marker.symbol`` / ``marker.size``,
``alpha`` → ``opacity``, ``cmap`` → ``colorscale``) are honored exactly as in
the 2-D core.

The camera is read from ``spec.meta["camera"]`` (the ``eye`` / ``up`` plotly
spelling — each a ``{"x", "y", "z"}`` mapping); when absent plotly's own default
camera applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import PlotKind, PlotSpec
from ...style import Theme, normalize_style

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["MARK_DISPATCH_3D", "build_3d_traces", "is_three_d", "render_3d", "scene_layout"]


def is_three_d(spec: PlotSpec) -> bool:
    """Whether ``spec`` needs 3-D drawing (ndim 3 / a ``z`` axis / a 3-D mark)."""
    if spec.ndim == 3 or spec.z is not None:
        return True
    return any(
        PlotKind(layer.kind) in (PlotKind.LINE3D, PlotKind.SURFACE3D) for layer in spec.layers
    )


def _resolve_theme(spec: PlotSpec) -> Theme:
    """Return the effective theme for ``spec`` (spec-level or active global)."""
    return spec.resolved_theme


def _canon_style(layer: Any) -> dict[str, Any]:
    """Return the normalized (canonical-key) style dict for a layer."""
    return normalize_style(layer.style, warn=False)


def _f(arr: Any) -> np.ndarray:
    """Coerce a channel to a float ``ndarray``."""
    return np.asarray(arr, dtype=float)


def _xyz(layer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Resolve the ``(x, y, z)`` coordinate triple for a 3-D mark, or ``None``."""
    if "x" not in layer.data or "y" not in layer.data or "z" not in layer.data:
        return None
    return _f(layer.data["x"]), _f(layer.data["y"]), _f(layer.data["z"])


def _colorscale(spec: PlotSpec, layer: Any) -> str | None:
    """Pick the plotly colorscale: layer style > spec colorbar cmap > ``None``."""
    style = _canon_style(layer)
    cmap = style.get("cmap")
    if cmap is not None:
        return str(cmap)
    if spec.colorbar is not None and spec.colorbar.cmap is not None:
        return spec.colorbar.cmap
    return None


def _colorbar_dict(spec: PlotSpec) -> dict[str, Any] | None:
    """Build a plotly ``colorbar`` dict from the spec's :class:`Colorbar`, if shown."""
    cbar = spec.colorbar
    if cbar is None or not cbar.show:
        return None
    out: dict[str, Any] = {}
    if cbar.label:
        out["title"] = cbar.label
    if cbar.ticks is not None:
        out["tickvals"] = [float(t) for t in cbar.ticks]
    if cbar.label_size is not None:
        out["titlefont"] = {"size": float(cbar.label_size)}
    return out


def _clim_kwargs(spec: PlotSpec) -> dict[str, float]:
    """Return ``cmin`` / ``cmax`` from the spec's clim (empty when unset)."""
    if spec.clim is None:
        return {}
    lo, hi = spec.clim
    return {"cmin": float(lo), "cmax": float(hi)}


def _build_line3d(layer: Any, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``LINE3D`` trace — a ``go.Scatter3d`` poly-line, optional colour-by-``c``.

    A plain line is ``mode="lines"``; a colour-by-``c`` line renders as
    ``mode="lines+markers"`` with the scalar field mapped onto the colour-scaled
    markers (plotly colours markers, not line segments), so hover shows the scalar.
    Per-layer canonical style keys ``linewidth`` (→ ``line.width``), ``linestyle``
    (→ ``line.dash``, via :data:`_DASH_MAP`), and ``alpha`` (→ ``opacity``) are
    honored; the theme's ``line_width`` default fills in when no explicit width is set.
    """
    import plotly.graph_objects as go

    xyz = _xyz(layer)
    if xyz is None:
        return []
    x, y, z = xyz
    style = _canon_style(layer)
    theme = _resolve_theme(spec)
    line: dict[str, Any] = {}
    if "color" in style:
        line["color"] = style["color"]
    width = style.get("linewidth")
    if width is None and theme.line_width is not None:
        width = theme.line_width
    if width is not None:
        line["width"] = float(width)
    dash = _DASH_MAP.get(str(style.get("linestyle", "")))
    if dash is not None:
        line["dash"] = dash
    opacity = float(style.get("alpha", 1.0))
    c = layer.data.get("c")
    if c is not None and _f(c).size == z.size:
        marker: dict[str, Any] = {
            "color": _f(c),
            "colorscale": _colorscale(spec, layer),
            "showscale": True,
            **_clim_kwargs(spec),
        }
        cbar = _colorbar_dict(spec)
        if cbar is not None:
            marker["colorbar"] = cbar
        return [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                line=line,
                marker=marker,
                name=layer.label,
                showlegend=layer.label is not None,
                opacity=opacity,
            )
        ]
    return [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=line,
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=opacity,
        )
    ]


def _build_scatter3d(layer: Any, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a 3-D ``SCATTER`` / ``MARKERS`` ``go.Scatter3d``, colour-/size-mapped.

    A ``LINE`` / ``MARKERS`` / ``SCATTER`` mark in a 3-D spec carries a ``z``
    channel and draws as a 3-D scatter, honouring the ``c`` (colour), ``size``,
    ``markersize`` (canonical key), and ``alpha`` channels.  Falls back to the
    theme's ``marker_size`` default.
    """
    import plotly.graph_objects as go

    xyz = _xyz(layer)
    if xyz is None:
        return []
    x, y, z = xyz
    style = _canon_style(layer)
    theme = _resolve_theme(spec)
    marker: dict[str, Any] = {}
    size_arr = layer.data.get("size")
    if size_arr is not None:
        marker["size"] = _f(size_arr)
    else:
        ms = style.get("markersize")
        if ms is None and theme.marker_size is not None:
            ms = theme.marker_size
        if ms is not None:
            marker["size"] = float(ms)
    c = layer.data.get("c")
    if c is not None:
        marker["color"] = _f(c)
        marker["colorscale"] = _colorscale(spec, layer)
        marker["showscale"] = True
        marker.update(_clim_kwargs(spec))
        cbar = _colorbar_dict(spec)
        if cbar is not None:
            marker["colorbar"] = cbar
    elif "color" in style:
        marker["color"] = style["color"]
    return [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=marker,
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=float(style.get("alpha", 1.0)),
        )
    ]


def _build_surface3d(layer: Any, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``SURFACE3D`` ``go.Surface``; ``x`` / ``y`` may be 1-D axes or 2-D meshes.

    plotly's ``go.Surface`` takes a 2-D ``z`` over coordinate vectors / meshes
    ``x`` / ``y``; the colour follows ``z`` (or the spec colorscale / clim).
    """
    import plotly.graph_objects as go

    xyz = _xyz(layer)
    if xyz is None:
        return []
    x, y, z = xyz
    kw: dict[str, Any] = {
        "z": z,
        "x": x,
        "y": y,
        "colorscale": _colorscale(spec, layer),
        # go.Surface takes cmin/cmax directly (it has NO zmin/zmax — those belong
        # to Heatmap/Contour/Image), so pass the colour range verbatim.
        **_clim_kwargs(spec),
    }
    cbar = _colorbar_dict(spec)
    if cbar is None and spec.colorbar is not None and not spec.colorbar.show:
        kw["showscale"] = False
    elif cbar is not None:
        kw["colorbar"] = cbar
    return [go.Surface(**kw)]


#: plotly ``line.dash`` names (canonical → plotly).
_DASH_MAP: dict[str, str] = {
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
    "dashdot": "dashdot",
    # legacy / mpl short spellings kept for layers that bypassed normalize_style
    "-": "solid",
    "--": "dash",
    ":": "dot",
    "-.": "dashdot",
}

#: 3-D layer mark → trace builder.  ``LINE`` / ``MARKERS`` / ``SCATTER`` in a 3-D
#: spec draw as their 3-D counterparts (they carry a ``z`` channel).
MARK_DISPATCH_3D: dict[PlotKind, Any] = {
    PlotKind.LINE3D: _build_line3d,
    PlotKind.LINE: _build_line3d,
    PlotKind.SURFACE3D: _build_surface3d,
    PlotKind.SCATTER: _build_scatter3d,
    PlotKind.MARKERS: _build_scatter3d,
}


def _scene(spec: PlotSpec) -> dict[str, Any]:
    """Build the plotly ``scene`` layout — axis titles/ranges, aspect, camera.

    Reads the camera from ``spec.meta["camera"]`` (the plotly ``eye`` / ``up``
    spelling, each a ``{"x", "y", "z"}`` mapping); absent keys keep plotly's own
    default camera.  ``aspect="equal"`` requests a cube (``aspectmode="cube"``).
    The theme's ``foreground`` color is applied to the scene axes (title, ticks,
    line) when set.
    """
    theme = _resolve_theme(spec)
    scene: dict[str, Any] = {
        "xaxis": _scene_axis(spec.x, theme),
        "yaxis": _scene_axis(spec.y, theme),
    }
    if spec.z is not None:
        scene["zaxis"] = _scene_axis(spec.z, theme)
    scene["aspectmode"] = "cube" if spec.aspect == "equal" else "auto"
    # Background: a 3-D plot's background is the ``bgcolor`` of each axis pane.
    if theme.background is not None:
        bg = theme.background
        for k in ("xaxis", "yaxis", "zaxis"):
            if k in scene:
                scene[k]["backgroundcolor"] = bg
        scene["bgcolor"] = bg
    camera = spec.meta.get("camera") if isinstance(spec.meta, dict) else None
    if isinstance(camera, dict):
        cam: dict[str, Any] = {}
        eye = camera.get("eye")
        if isinstance(eye, dict):
            cam["eye"] = {k: float(eye[k]) for k in ("x", "y", "z") if k in eye}
        up = camera.get("up")
        if isinstance(up, dict):
            cam["up"] = {k: float(up[k]) for k in ("x", "y", "z") if k in up}
        center = camera.get("center")
        if isinstance(center, dict):
            cam["center"] = {k: float(center[k]) for k in ("x", "y", "z") if k in center}
        if cam:
            scene["camera"] = cam
    if spec._axes_hidden():
        for k in ("xaxis", "yaxis", "zaxis"):
            if k in scene:
                scene[k] = {**scene[k], "visible": False}
    return scene


def _scene_axis(axis: Any, theme: Theme | None = None) -> dict[str, Any]:
    """Build one plotly ``scene`` axis dict (title + range + theming)."""
    out: dict[str, Any] = {}
    if axis.label:
        title: dict[str, Any] = {"text": axis.label}
        if axis.label_size is not None:
            title["font"] = {"size": float(axis.label_size)}
        elif theme is not None and theme.font_size is not None:
            title["font"] = {"size": float(theme.font_size)}
        if theme is not None and theme.foreground is not None:
            title.setdefault("font", {})["color"] = theme.foreground
        out["title"] = title
    if axis.limits is not None:
        out["range"] = [float(axis.limits[0]), float(axis.limits[1])]
    # Foreground ink for axis ticks / lines.
    if theme is not None and theme.foreground is not None:
        out["tickcolor"] = theme.foreground
        out["linecolor"] = theme.foreground
    if axis.tick_size is not None:
        out["tickfont"] = {"size": float(axis.tick_size)}
    elif theme is not None and theme.font_size is not None:
        out["tickfont"] = {"size": float(theme.font_size)}
    return out


def build_3d_traces(spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build every 3-D trace for ``spec`` through :data:`MARK_DISPATCH_3D`.

    Factored out of :func:`render_3d` so the composite renderer can draw a 3-D
    panel's ``go.Scatter3d`` / ``go.Surface`` traces into one ``scene`` cell of a
    :func:`plotly.subplots.make_subplots` grid (rather than its own figure).
    """
    traces: list[go.BaseTraceType] = []
    for layer in spec.layers:
        builder = MARK_DISPATCH_3D.get(PlotKind(layer.kind))
        if builder is None:
            continue
        traces.extend(builder(layer, spec))
    return traces


def scene_layout(spec: PlotSpec) -> dict[str, Any]:
    """Return the plotly ``scene`` layout dict for a 3-D ``spec``.

    The public wrapper over the private :func:`_scene` builder, so the composite
    renderer can attach a panel's scene (axes / camera / aspect / theme) to the
    right ``sceneN`` slot of a subplot grid.
    """
    return _scene(spec)


def render_3d(spec: PlotSpec, **_kw: Any) -> go.Figure:
    """Render a 3-D :class:`~tsdynamics.viz.spec.PlotSpec` to an orbitable plotly Figure.

    Builds a :class:`plotly.graph_objects.Figure`, adds every layer's 3-D traces
    through :data:`MARK_DISPATCH_3D` (``go.Scatter3d`` / ``go.Surface``), then
    applies the ``scene`` (three axis titles / ranges, cube aspect for an
    ``"equal"`` spec, and the ``spec.meta["camera"]`` eye / up), the title, and the
    legend.  The resolved :class:`~tsdynamics.viz.style.Theme` is applied to the
    figure layout (``paper_bgcolor`` / ``plot_bgcolor``, ``layout.font``).

    Parameters
    ----------
    spec : PlotSpec
        A 3-D spec (see :func:`is_three_d`).
    **_kw
        Forwarded but unused backend keywords (kept for a uniform renderer
        signature).

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive 3-D figure (orbit / zoom), ready to ``show`` / embed.
    """
    import plotly.graph_objects as go

    from ._core import _theme_layout

    theme = _resolve_theme(spec)
    fig = go.Figure()
    for trace in build_3d_traces(spec):
        fig.add_trace(trace)

    show_legend = spec.legend is not None and spec.legend.show
    layout: dict[str, Any] = {"scene": _scene(spec), "showlegend": show_legend}
    # Apply theme-level presentation (background, font).
    layout.update(_theme_layout(theme))
    if spec.title:
        title_dict: dict[str, Any] = {"text": spec.title}
        if theme.title_size is not None:
            title_dict["font"] = {"size": float(theme.title_size)}
        layout["title"] = title_dict
    if show_legend and spec.legend is not None and spec.legend.title:
        leg: dict[str, Any] = {"title": {"text": spec.legend.title}}
        if spec.legend.font_size is not None:
            leg["font"] = {"size": float(spec.legend.font_size)}
        if not spec.legend.frame:
            leg["bgcolor"] = "rgba(0,0,0,0)"
            leg["bordercolor"] = "rgba(0,0,0,0)"
        layout["legend"] = leg
    fig.update_layout(**layout)
    return fig
