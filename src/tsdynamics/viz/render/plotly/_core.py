"""The plotly 2-D interactive renderer core (stream PLOTLY-RENDER).

This module turns a backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into a
:class:`plotly.graph_objects.Figure` — an interactive (pan / zoom / hover) figure
suitable for notebooks and the web.  It uses plotly's ``graph_objects`` API
**only**: no ``plotly.express`` (a tidy-frame layer TSDynamics' numeric arrays do
not need) and no ``kaleido`` (static-image export is the JSON / image exporters'
concern, not this live-figure renderer).

The plotly backend is *partial*: it draws the 2-D marks (``LINE`` / ``SCATTER`` /
``MARKERS`` / ``IMAGE`` / ``HISTOGRAM`` / ``BAR`` / ``AREA`` / ``ERRORBAR``) and
the 2-D semantic kinds, and **declines** the 3-D marks (``LINE3D`` /
``SURFACE3D``) so dispatch falls back to the matplotlib reference renderer.  Its
:class:`~tsdynamics.viz.render.caps.RendererCapabilities` (built in the package
``register`` hook) advertise exactly that set.

The pieces
----------
- :data:`MARK_DISPATCH` — layer mark → trace builder.  Each maps one
  :class:`~tsdynamics.viz.spec.Layer` to a ``graph_objects`` trace, reading the
  layer's channel data (``x`` / ``y`` / ``c`` / ``lo`` / ``hi`` / ``err`` /
  ``cat`` / ``size``) and neutral style keys.
- :func:`render` — the entry point: build a :class:`~plotly.graph_objects.Figure`,
  normalise the spec's semantic kind, add every layer's traces, apply axes
  (label / scale / limits / categories), colorbar (cmap / clim), legend, and the
  ``vline`` / ``hline`` / ``text`` / ``span`` annotations (as layout shapes /
  annotations).  Returns the :class:`~plotly.graph_objects.Figure`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import Annotation, Axis, Layer, PlotKind, PlotSpec
from .. import normalize_kind

if TYPE_CHECKING:
    import plotly.graph_objects as go


__all__ = ["MARK_DISPATCH", "render"]


# ---------------------------------------------------------------------------
# Channel / style helpers
# ---------------------------------------------------------------------------


def _channel(layer: Layer, name: str) -> np.ndarray | None:
    """Return a layer's channel array as float, or ``None`` if absent."""
    arr = layer.data.get(name)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _line_style(layer: Layer) -> dict[str, Any]:
    """Build a plotly ``line`` dict from a layer's neutral style keys."""
    style = layer.style
    line: dict[str, Any] = {}
    color = style.get("color")
    if color is not None:
        line["color"] = color
    width = style.get("lw", style.get("linewidth"))
    if width is not None:
        line["width"] = width
    dash = _DASH_MAP.get(str(style.get("linestyle", style.get("ls", ""))))
    if dash is not None:
        line["dash"] = dash
    return line


#: matplotlib-flavoured linestyle spellings → plotly ``dash`` names.
_DASH_MAP: dict[str, str] = {
    "-": "solid",
    "solid": "solid",
    "--": "dash",
    "dashed": "dash",
    ":": "dot",
    "dotted": "dot",
    "-.": "dashdot",
    "dashdot": "dashdot",
}


def _marker_symbol(style: dict[str, Any]) -> str | None:
    """Map a matplotlib-flavoured marker code to a plotly symbol, if recognised."""
    marker = style.get("marker")
    if marker is None:
        return None
    return _MARKER_MAP.get(str(marker), "circle")


#: matplotlib marker codes → plotly marker symbols (unknown → ``"circle"``).
_MARKER_MAP: dict[str, str] = {
    "o": "circle",
    ".": "circle",
    "s": "square",
    "^": "triangle-up",
    "v": "triangle-down",
    "x": "x",
    "+": "cross",
    "d": "diamond",
    "D": "diamond",
    "*": "star",
}


def _colorscale(spec: PlotSpec, layer: Layer) -> str | None:
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
    """Return ``cmin`` / ``cmax`` (or ``zmin`` callers rename) from the spec's clim."""
    if spec.clim is None:
        return {}
    lo, hi = spec.clim
    return {"cmin": float(lo), "cmax": float(hi)}


# ---------------------------------------------------------------------------
# Mark trace builders (one per layer mark) — each returns a list of traces
# ---------------------------------------------------------------------------

#: A mark-trace builder takes ``(layer, spec)`` and returns the plotly traces it
#: produced (a list so a mark may emit more than one, e.g. an AREA band).
_MarkBuilder = Callable[[Layer, PlotSpec], "list[go.BaseTraceType]"]


def _xy(layer: Layer) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve ``(x, y)`` for a line/scatter mark; ``x`` defaults to the index."""
    y = _channel(layer, "y")
    if y is None:
        return None
    x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    return x, y


def _build_line(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``LINE`` trace — a poly-line, optionally colour-by-``c``."""
    import plotly.graph_objects as go

    xy = _xy(layer)
    if xy is None:
        return []
    x, y = xy
    c = _channel(layer, "c")
    if c is not None and c.size == y.size:
        # Colour-by-``c``: plotly colours markers, not line segments, so render a
        # thin line beneath a colour-mapped marker overlay (hover shows the scalar).
        marker: dict[str, Any] = {
            "color": c,
            "colorscale": _colorscale(spec, layer),
            "showscale": True,
            **_clim_kwargs(spec),
        }
        cbar = _colorbar_dict(spec)
        if cbar is not None:
            marker["colorbar"] = cbar
        return [
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=_line_style(layer),
                marker=marker,
                name=layer.label,
                showlegend=layer.label is not None,
                opacity=layer.style.get("alpha", 1.0),
            )
        ]
    return [
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=_line_style(layer),
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=layer.style.get("alpha", 1.0),
        )
    ]


def _build_scatter(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``SCATTER`` / ``MARKERS`` trace, optionally colour-/size-mapped."""
    import plotly.graph_objects as go

    xy = _xy(layer)
    if xy is None:
        return []
    x, y = xy
    c = _channel(layer, "c")
    size = _channel(layer, "size")
    marker: dict[str, Any] = {}
    symbol = _marker_symbol(layer.style)
    if symbol is not None:
        marker["symbol"] = symbol
    if size is not None:
        marker["size"] = size
    elif "s" in layer.style:
        marker["size"] = layer.style["s"]
    if c is not None:
        marker["color"] = c
        marker["colorscale"] = _colorscale(spec, layer)
        marker["showscale"] = True
        marker.update(_clim_kwargs(spec))
        cbar = _colorbar_dict(spec)
        if cbar is not None:
            marker["colorbar"] = cbar
    elif "color" in layer.style:
        marker["color"] = layer.style["color"]
    return [
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=layer.style.get("alpha", 1.0),
        )
    ]


def _build_image(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build an ``IMAGE`` mark as a ``go.Heatmap`` from a 2-D ``z`` (or ``c``)."""
    import plotly.graph_objects as go

    z = layer.data.get("z")
    if z is None:
        z = layer.data.get("c")
    if z is None:
        return []
    img = np.asarray(z, dtype=float)
    kw: dict[str, Any] = {
        "z": img,
        "colorscale": _colorscale(spec, layer),
        **{k.replace("c", "z", 1): v for k, v in _clim_kwargs(spec).items()},
    }
    x = layer.data.get("x")
    y = layer.data.get("y")
    if x is not None:
        kw["x"] = np.asarray(x, dtype=float)
    if y is not None:
        kw["y"] = np.asarray(y, dtype=float)
    cbar = _colorbar_dict(spec)
    if cbar is None and spec.colorbar is not None and not spec.colorbar.show:
        kw["showscale"] = False
    elif cbar is not None:
        kw["colorbar"] = cbar
    return [go.Heatmap(**kw)]


def _build_histogram(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``HISTOGRAM`` mark.

    Pre-binned (``x`` = centres, ``y`` = counts) becomes a ``go.Bar``; raw
    samples (only ``x``) become a ``go.Histogram`` plotly bins itself.
    """
    import plotly.graph_objects as go

    x = _channel(layer, "x")
    if x is None:
        return []
    y = _channel(layer, "y")
    color = layer.style.get("color")
    opacity = layer.style.get("alpha", 1.0)
    if y is not None:
        return [
            go.Bar(
                x=x,
                y=y,
                name=layer.label,
                showlegend=layer.label is not None,
                marker={"color": color} if color is not None else {},
                opacity=opacity,
            )
        ]
    return [
        go.Histogram(
            x=x,
            name=layer.label,
            showlegend=layer.label is not None,
            marker={"color": color} if color is not None else {},
            opacity=opacity,
        )
    ]


def _build_bar(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``BAR`` mark — values ``y`` at positions ``x`` / ``cat``."""
    import plotly.graph_objects as go

    y = _channel(layer, "y")
    if y is None:
        return []
    x = _channel(layer, "cat")
    if x is None:
        x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    color = layer.style.get("color")
    return [
        go.Bar(
            x=x,
            y=y,
            name=layer.label,
            showlegend=layer.label is not None,
            marker={"color": color} if color is not None else {},
            opacity=layer.style.get("alpha", 1.0),
        )
    ]


def _build_area(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build an ``AREA`` mark — a shaded ``lo <= hi`` band over ``x``.

    Emits an invisible ``lo`` boundary trace plus a ``hi`` trace with
    ``fill="tonexty"`` so the band between them is shaded; a central ``y`` line is
    drawn through the fan when a distinct ``y`` is supplied alongside the edges.
    """
    import plotly.graph_objects as go

    x = _channel(layer, "x")
    lo = _channel(layer, "lo")
    hi = _channel(layer, "hi")
    y = _channel(layer, "y")
    if x is None:
        ref = lo if lo is not None else (hi if hi is not None else y)
        if ref is None:
            return []
        x = np.arange(ref.size, dtype=float)
    if lo is None:
        lo = y if y is not None else np.zeros_like(x)
    if hi is None:
        hi = y if y is not None else np.zeros_like(x)
    color = layer.style.get("color")
    alpha = layer.style.get("alpha", 0.3)
    traces: list[go.BaseTraceType] = [
        go.Scatter(x=x, y=lo, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"),
        go.Scatter(
            x=x,
            y=hi,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=color,
            opacity=alpha,
            name=layer.label,
            showlegend=layer.label is not None,
        ),
    ]
    if y is not None and "lo" in layer.data:
        traces.append(go.Scatter(x=x, y=y, mode="lines", line=_line_style(layer), showlegend=False))
    return traces


def _build_errorbar(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build an ``ERRORBAR`` mark — ``y`` vs ``x`` with symmetric ``error_y``."""
    import plotly.graph_objects as go

    y = _channel(layer, "y")
    if y is None:
        return []
    x = _channel(layer, "x")
    if x is None:
        x = np.arange(y.size, dtype=float)
    err = _channel(layer, "err")
    error_y = {"type": "data", "array": err, "visible": True} if err is not None else None
    marker: dict[str, Any] = {}
    symbol = _marker_symbol(layer.style)
    if symbol is not None:
        marker["symbol"] = symbol
    if "color" in layer.style:
        marker["color"] = layer.style["color"]
    return [
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            error_y=error_y,
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=layer.style.get("alpha", 1.0),
        )
    ]


def _build_quiver(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``QUIVER`` mark — line segments ``(x, y) -> (x+u, y+v)`` plus tips.

    plotly's ``graph_objects`` has no native quiver trace (that lives in the
    figure-factory layer this renderer avoids), so arrows are drawn as a single
    ``Scatter`` line trace of ``NaN``-separated segments with a marker tip trace.
    """
    import plotly.graph_objects as go

    x = _channel(layer, "x")
    y = _channel(layer, "y")
    u = _channel(layer, "u")
    v = _channel(layer, "v")
    if x is None or y is None or u is None or v is None:
        return []
    xe = x + u
    ye = y + v
    seg_x = np.empty(x.size * 3)
    seg_y = np.empty(y.size * 3)
    seg_x[0::3], seg_x[1::3], seg_x[2::3] = x, xe, np.nan
    seg_y[0::3], seg_y[1::3], seg_y[2::3] = y, ye, np.nan
    color = layer.style.get("color")
    return [
        go.Scatter(
            x=seg_x,
            y=seg_y,
            mode="lines",
            line={"color": color} if color is not None else {},
            name=layer.label,
            showlegend=layer.label is not None,
        ),
        go.Scatter(
            x=xe, y=ye, mode="markers", marker={"size": 4, "color": color}, showlegend=False
        ),
    ]


#: Layer mark → trace builder.  The 3-D marks (``LINE3D`` / ``SURFACE3D``) are
#: deliberately absent: the capability declaration declines them so dispatch
#: falls back to the matplotlib reference renderer rather than this backend
#: drawing a degraded 2-D projection.
MARK_DISPATCH: dict[PlotKind, _MarkBuilder] = {
    PlotKind.LINE: _build_line,
    PlotKind.SCATTER: _build_scatter,
    PlotKind.MARKERS: _build_scatter,
    PlotKind.IMAGE: _build_image,
    PlotKind.HISTOGRAM: _build_histogram,
    PlotKind.BAR: _build_bar,
    PlotKind.AREA: _build_area,
    PlotKind.ERRORBAR: _build_errorbar,
    PlotKind.QUIVER: _build_quiver,
}


# ---------------------------------------------------------------------------
# Axis / legend / annotation application
# ---------------------------------------------------------------------------

#: Spec axis scale → plotly axis ``type`` (``"categorical"`` is handled inline).
_AXIS_TYPE: dict[str, str] = {"linear": "linear", "log": "log", "symlog": "linear"}


def _axis_layout(axis: Axis) -> dict[str, Any]:
    """Build a plotly axis-layout dict from a typed :class:`Axis`."""
    out: dict[str, Any] = {}
    if axis.label:
        out["title"] = {"text": axis.label}
    if axis.scale == "categorical" and axis.categories is not None:
        out["type"] = "category"
        out["tickmode"] = "array"
        out["tickvals"] = list(range(len(axis.categories)))
        out["ticktext"] = list(axis.categories)
    else:
        out["type"] = _AXIS_TYPE.get(axis.scale, "linear")
    if axis.limits is not None:
        lo, hi = axis.limits
        if axis.scale == "log":
            lo = float(np.log10(lo)) if lo > 0 else lo
            hi = float(np.log10(hi)) if hi > 0 else hi
        out["range"] = [lo, hi]
    if axis.ticks is not None and axis.scale != "categorical":
        out["tickmode"] = "array"
        out["tickvals"] = [float(t) for t in axis.ticks]
    if axis.tickformat is not None:
        out["tickformat"] = axis.tickformat
    return out


def _annotation_shapes(
    annotations: list[Annotation],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Translate annotations into plotly layout ``shapes`` + ``annotations`` lists."""
    shapes: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    for ann in annotations:
        if ann.kind == "vline" and ann.x is not None:
            shapes.append(_line_shape("x", ann.x, ann.style))
            if ann.text:
                texts.append(
                    {"x": ann.x, "y": 1.0, "yref": "paper", "text": ann.text, "showarrow": False}
                )
        elif ann.kind == "hline" and ann.y is not None:
            shapes.append(_line_shape("y", ann.y, ann.style))
            if ann.text:
                texts.append(
                    {"x": 1.0, "xref": "paper", "y": ann.y, "text": ann.text, "showarrow": False}
                )
        elif ann.kind == "text" and ann.x is not None and ann.y is not None:
            texts.append({"x": ann.x, "y": ann.y, "text": ann.text, "showarrow": False})
        elif ann.kind == "span" and ann.span is not None:
            shapes.append(_span_shape(ann.axis, ann.span, ann.style))
    return shapes, texts


def _line_shape(axis: str, value: float, style: dict[str, Any]) -> dict[str, Any]:
    """Build a full-extent reference-line shape on ``axis`` at ``value``."""
    line = {}
    if "color" in style:
        line["color"] = style["color"]
    if axis == "x":
        base = {"type": "line", "x0": value, "x1": value, "y0": 0, "y1": 1, "yref": "paper"}
    else:
        base = {"type": "line", "y0": value, "y1": value, "x0": 0, "x1": 1, "xref": "paper"}
    base["line"] = line
    return base


def _span_shape(axis: str, span: tuple[float, float], style: dict[str, Any]) -> dict[str, Any]:
    """Build a shaded band shape across ``axis`` between ``span`` edges."""
    lo, hi = span
    fill = style.get("color", "gray")
    opacity = style.get("alpha", 0.2)
    if axis == "y":
        return {
            "type": "rect",
            "y0": lo,
            "y1": hi,
            "x0": 0,
            "x1": 1,
            "xref": "paper",
            "fillcolor": fill,
            "opacity": opacity,
            "line": {"width": 0},
        }
    return {
        "type": "rect",
        "x0": lo,
        "x1": hi,
        "y0": 0,
        "y1": 1,
        "yref": "paper",
        "fillcolor": fill,
        "opacity": opacity,
        "line": {"width": 0},
    }


# ---------------------------------------------------------------------------
# The render entry point
# ---------------------------------------------------------------------------


def render(spec: PlotSpec, **_kw: Any) -> go.Figure:
    """Render a 2-D :class:`~tsdynamics.viz.spec.PlotSpec` to a plotly Figure.

    Builds a :class:`plotly.graph_objects.Figure`, adds every
    :class:`~tsdynamics.viz.spec.Layer`'s traces through :data:`MARK_DISPATCH`,
    then applies the typed axes (label / scale / limits / categories), the
    legend, and the ``vline`` / ``hline`` / ``text`` / ``span`` annotations as
    plotly layout shapes / annotations.  The spec's semantic kind is normalised
    through :func:`tsdynamics.viz.render.normalize_kind` (an alias / mark spelling
    resolves to a real kind), and an ``"equal"`` aspect maps to a constrained
    y-axis so phase portraits / sections / images keep their geometry.

    Parameters
    ----------
    spec : PlotSpec
        The backend-agnostic spec to draw.  Its per-call tweaks
        (relabel / rescale / limits / ticks / colorize) are already baked into the
        typed axes / colorbar, so honouring those honours the tweaks.
    **_kw
        Forwarded but unused backend keywords (kept for a uniform renderer
        signature).

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive figure (pan / zoom / hover), ready to ``show`` / embed /
        export to HTML.

    Notes
    -----
    A 3-D spec (``ndim == 3`` / a ``z`` axis / a ``LINE3D`` / ``SURFACE3D`` mark)
    is dispatched to the :mod:`._threed` renderer, which draws orbitable
    ``go.Scatter3d`` / ``go.Surface`` traces on a 3-D ``scene``.
    """
    import plotly.graph_objects as go

    from . import _threed

    normalize_kind(spec.kind)  # validate / canonicalise the semantic kind

    if _threed.is_three_d(spec):
        return _threed.render_3d(spec, **_kw)

    fig = go.Figure()
    for layer in spec.layers:
        builder = MARK_DISPATCH.get(PlotKind(layer.kind))
        if builder is None:
            continue
        for trace in builder(layer, spec):
            fig.add_trace(trace)

    xaxis = _axis_layout(spec.x)
    yaxis = _axis_layout(spec.y)
    if spec.aspect == "equal":
        yaxis["scaleanchor"] = "x"
        yaxis["scaleratio"] = 1

    show_legend = spec.legend is not None and spec.legend.show
    layout: dict[str, Any] = {
        "xaxis": xaxis,
        "yaxis": yaxis,
        "showlegend": show_legend,
    }
    if spec.title:
        layout["title"] = {"text": spec.title}
    if show_legend and spec.legend is not None and spec.legend.title:
        layout["legend"] = {"title": {"text": spec.legend.title}}

    shapes, texts = _annotation_shapes(spec.annotations)
    if shapes:
        layout["shapes"] = shapes
    if texts:
        layout["annotations"] = texts

    fig.update_layout(**layout)
    return fig
