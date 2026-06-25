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

The camera is read from ``spec.meta["camera"]`` (the ``eye`` / ``up`` plotly
spelling — each a ``{"x", "y", "z"}`` mapping); when absent plotly's own default
camera applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import PlotKind, PlotSpec

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["MARK_DISPATCH_3D", "is_three_d", "render_3d"]


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


def _xyz(layer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Resolve the ``(x, y, z)`` coordinate triple for a 3-D mark, or ``None``."""
    if "x" not in layer.data or "y" not in layer.data or "z" not in layer.data:
        return None
    return _f(layer.data["x"]), _f(layer.data["y"]), _f(layer.data["z"])


def _colorscale(spec: PlotSpec, layer: Any) -> str | None:
    """Pick the plotly colorscale: layer style > spec colorbar cmap > ``None``."""
    cmap = layer.style.get("cmap")
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
    """
    import plotly.graph_objects as go

    xyz = _xyz(layer)
    if xyz is None:
        return []
    x, y, z = xyz
    style = layer.style
    line: dict[str, Any] = {}
    if "color" in style:
        line["color"] = style["color"]
    if "lw" in style or "linewidth" in style:
        line["width"] = style.get("lw", style.get("linewidth"))
    opacity = style.get("alpha", 1.0)
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
    channel and draws as a 3-D scatter, honouring the ``c`` (colour) and ``size``
    channels.
    """
    import plotly.graph_objects as go

    xyz = _xyz(layer)
    if xyz is None:
        return []
    x, y, z = xyz
    marker: dict[str, Any] = {}
    size = layer.data.get("size")
    if size is not None:
        marker["size"] = _f(size)
    elif "s" in layer.style:
        marker["size"] = layer.style["s"]
    c = layer.data.get("c")
    if c is not None:
        marker["color"] = _f(c)
        marker["colorscale"] = _colorscale(spec, layer)
        marker["showscale"] = True
        marker.update(_clim_kwargs(spec))
        cbar = _colorbar_dict(spec)
        if cbar is not None:
            marker["colorbar"] = cbar
    elif "color" in layer.style:
        marker["color"] = layer.style["color"]
    return [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=marker,
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=layer.style.get("alpha", 1.0),
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
    """
    scene: dict[str, Any] = {
        "xaxis": _scene_axis(spec.x),
        "yaxis": _scene_axis(spec.y),
    }
    if spec.z is not None:
        scene["zaxis"] = _scene_axis(spec.z)
    scene["aspectmode"] = "cube" if spec.aspect == "equal" else "auto"
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


def _scene_axis(axis: Any) -> dict[str, Any]:
    """Build one plotly ``scene`` axis dict (title + range) from a typed :class:`Axis`."""
    out: dict[str, Any] = {}
    if axis.label:
        out["title"] = {"text": axis.label}
    if axis.limits is not None:
        out["range"] = [float(axis.limits[0]), float(axis.limits[1])]
    return out


def render_3d(spec: PlotSpec, **_kw: Any) -> go.Figure:
    """Render a 3-D :class:`~tsdynamics.viz.spec.PlotSpec` to an orbitable plotly Figure.

    Builds a :class:`plotly.graph_objects.Figure`, adds every layer's 3-D traces
    through :data:`MARK_DISPATCH_3D` (``go.Scatter3d`` / ``go.Surface``), then
    applies the ``scene`` (three axis titles / ranges, cube aspect for an
    ``"equal"`` spec, and the ``spec.meta["camera"]`` eye / up), the title, and the
    legend.  The result is a draggable / zoomable 3-D figure.

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

    fig = go.Figure()
    for layer in spec.layers:
        builder = MARK_DISPATCH_3D.get(PlotKind(layer.kind))
        if builder is None:
            continue
        for trace in builder(layer, spec):
            fig.add_trace(trace)

    show_legend = spec.legend is not None and spec.legend.show
    layout: dict[str, Any] = {"scene": _scene(spec), "showlegend": show_legend}
    if spec.title:
        layout["title"] = {"text": spec.title}
    if show_legend and spec.legend is not None and spec.legend.title:
        layout["legend"] = {"title": {"text": spec.legend.title}}
    fig.update_layout(**layout)
    return fig
