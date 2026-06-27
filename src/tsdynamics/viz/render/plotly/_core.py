"""The plotly 2-D interactive renderer core (stream PLOTLY-RENDER).

This module turns a backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into a
:class:`plotly.graph_objects.Figure` — an interactive (pan / zoom / hover) figure
suitable for notebooks and the web.  It uses plotly's ``graph_objects`` API
**only**: no ``plotly.express`` (a tidy-frame layer TSDynamics' numeric arrays do
not need) and no ``kaleido`` (static-image export is the JSON / image exporters'
concern, not this live-figure renderer).

The plotly backend draws the 2-D marks (``LINE`` / ``SCATTER`` / ``MARKERS`` /
``IMAGE`` / ``HISTOGRAM`` / ``BAR`` / ``AREA`` / ``ERRORBAR``) here, dispatches
3-D specs to :mod:`._threed`, and optionally exports the interactive figure as
self-contained HTML for web / mkdocs embedding.  Only the animation kinds are
still declined; the package-level
:class:`~tsdynamics.viz.render.caps.RendererCapabilities` advertise that set.

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
  annotations).  The spec's resolved :class:`~tsdynamics.viz.style.Theme` is
  applied to the figure layout (``paper_bgcolor`` / ``plot_bgcolor``,
  ``layout.font``, axis ``showgrid``).  Returns the :class:`~plotly.graph_objects.Figure`.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import Annotation, Axis, Layer, PlotKind, PlotSpec
from ...style import Theme, normalize_style
from .. import normalize_kind

if TYPE_CHECKING:
    import plotly.graph_objects as go


__all__ = ["MARK_DISPATCH", "build_2d_traces", "render"]


# ---------------------------------------------------------------------------
# Canonical style → plotly mapping helpers
# ---------------------------------------------------------------------------

#: Canonical linestyle names → plotly ``dash`` names.
_DASH_MAP: dict[str, str] = {
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
    "dashdot": "dashdot",
    # legacy / mpl short spellings (kept for layers that still carry them)
    "-": "solid",
    "--": "dash",
    ":": "dot",
    "-.": "dashdot",
}

#: Canonical marker-shape names → plotly marker symbols.
#: Unknown → ``"circle"`` (the dispatcher warns via §5.2 honest negotiation).
_MARKER_MAP: dict[str, str] = {
    "circle": "circle",
    "square": "square",
    "triangle": "triangle-up",
    "diamond": "diamond",
    "cross": "cross",
    "x": "x",
    "star": "star",
    "none": "circle",  # plotly has no "none" — smallest visible marker
    # legacy / mpl single-char spellings (style.py canonicalises, but a layer
    # that bypassed normalize_style may still carry these)
    "o": "circle",
    ".": "circle",
    "s": "square",
    "^": "triangle-up",
    "v": "triangle-down",
    "d": "diamond",
    "D": "diamond",
    "+": "cross",
    "*": "star",
}


def _resolve_theme(spec: PlotSpec) -> Theme:
    """Return the effective theme for ``spec`` (spec-level or active global)."""
    return spec.resolved_theme


def _canon_style(layer: Layer) -> dict[str, Any]:
    """Return the normalized (canonical-key) style dict for a layer.

    Renderers call with ``warn=False`` — the dispatcher already emitted the
    consolidated :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`.
    """
    return normalize_style(layer.style, warn=False)


# ---------------------------------------------------------------------------
# Channel / style helpers
# ---------------------------------------------------------------------------


def _channel(layer: Layer, name: str) -> np.ndarray | None:
    """Return a layer's channel array as float, or ``None`` if absent."""
    arr = layer.data.get(name)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _line_style(layer: Layer, theme: Theme | None = None) -> dict[str, Any]:
    """Build a plotly ``line`` dict from a layer's canonical style keys.

    Falls back to theme-level ``line_width`` when no per-layer width is set.
    """
    style = _canon_style(layer)
    line: dict[str, Any] = {}
    color = style.get("color")
    if color is not None:
        line["color"] = color
    width = style.get("linewidth")
    if width is None and theme is not None and theme.line_width is not None:
        width = theme.line_width
    if width is not None:
        line["width"] = float(width)
    dash = _DASH_MAP.get(str(style.get("linestyle", "")))
    if dash is not None:
        line["dash"] = dash
    return line


def _marker_symbol(style: dict[str, Any]) -> str | None:
    """Map a canonical marker name to a plotly symbol (``None`` ⇒ omit from trace)."""
    marker = style.get("marker")
    if marker is None:
        return None
    return _MARKER_MAP.get(str(marker), "circle")


def _colorscale(spec: PlotSpec, layer: Layer) -> str | None:
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
    """Return ``cmin`` / ``cmax`` (or ``zmin`` callers rename) from the spec's clim."""
    if spec.clim is None:
        return {}
    lo, hi = spec.clim
    return {"cmin": float(lo), "cmax": float(hi)}


def _opacity(style: dict[str, Any]) -> float:
    """Resolve the ``alpha`` / ``opacity`` canonical key (default 1.0)."""
    return float(style.get("alpha", 1.0))


def _marker_size(style: dict[str, Any], theme: Theme | None = None) -> float | None:
    """Resolve the ``markersize`` canonical key, falling back to theme default."""
    sz = style.get("markersize")
    if sz is None and theme is not None:
        sz = theme.marker_size
    return float(sz) if sz is not None else None


def _apply_zorder(traces: list[go.BaseTraceType], style: dict[str, Any]) -> None:
    """Set the canonical ``zorder`` draw order on every 2-D trace, in place.

    plotly 6.x supports a trace-level ``.zorder`` on the 2-D cartesian traces
    (``Scatter`` / ``Bar`` / ``Histogram`` / ``Heatmap``) — higher draws on top —
    so the canonical ``zorder`` style key is honored by mapping it directly.  3-D
    traces (``Scatter3d`` / ``Surface``) have no ``zorder`` knob (their draw order
    is the intrinsic depth of the scene), so the 3-D renderer never calls this.
    """
    if "zorder" not in style:
        return
    z = int(style["zorder"])
    for trace in traces:
        # All cartesian 2-D traces this renderer emits carry ``zorder``; guard
        # defensively so a future trace type without it cannot raise.
        if "zorder" in trace:
            trace.zorder = z


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
    style = _canon_style(layer)
    theme = _resolve_theme(spec)
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
        traces: list[go.BaseTraceType] = [
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=_line_style(layer, theme),
                marker=marker,
                name=layer.label,
                showlegend=layer.label is not None,
                opacity=_opacity(style),
            )
        ]
        _apply_zorder(traces, style)
        return traces
    traces = [
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=_line_style(layer, theme),
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=_opacity(style),
        )
    ]
    _apply_zorder(traces, style)
    return traces


def _build_scatter(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build a ``SCATTER`` / ``MARKERS`` trace, optionally colour-/size-mapped."""
    import plotly.graph_objects as go

    xy = _xy(layer)
    if xy is None:
        return []
    x, y = xy
    style = _canon_style(layer)
    theme = _resolve_theme(spec)
    c = _channel(layer, "c")
    size_arr = _channel(layer, "size")
    marker: dict[str, Any] = {}
    symbol = _marker_symbol(style)
    if symbol is not None:
        marker["symbol"] = symbol
    if size_arr is not None:
        marker["size"] = size_arr
    else:
        ms = _marker_size(style, theme)
        if ms is not None:
            marker["size"] = ms
    if c is not None:
        marker["color"] = c
        marker["colorscale"] = _colorscale(spec, layer)
        marker["showscale"] = True
        marker.update(_clim_kwargs(spec))
        cbar = _colorbar_dict(spec)
        if cbar is not None:
            marker["colorbar"] = cbar
    elif "color" in style:
        marker["color"] = style["color"]
    traces: list[go.BaseTraceType] = [
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=_opacity(style),
        )
    ]
    _apply_zorder(traces, style)
    return traces


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
    traces: list[go.BaseTraceType] = [go.Heatmap(**kw)]
    _apply_zorder(traces, _canon_style(layer))
    return traces


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
    style = _canon_style(layer)
    color = style.get("color")
    opacity = _opacity(style)
    if y is not None:
        bars: list[go.BaseTraceType] = [
            go.Bar(
                x=x,
                y=y,
                name=layer.label,
                showlegend=layer.label is not None,
                marker={"color": color} if color is not None else {},
                opacity=opacity,
            )
        ]
        _apply_zorder(bars, style)
        return bars
    hist: list[go.BaseTraceType] = [
        go.Histogram(
            x=x,
            name=layer.label,
            showlegend=layer.label is not None,
            marker={"color": color} if color is not None else {},
            opacity=opacity,
        )
    ]
    _apply_zorder(hist, style)
    return hist


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
    style = _canon_style(layer)
    color = style.get("color")
    traces: list[go.BaseTraceType] = [
        go.Bar(
            x=x,
            y=y,
            name=layer.label,
            showlegend=layer.label is not None,
            marker={"color": color} if color is not None else {},
            opacity=_opacity(style),
        )
    ]
    _apply_zorder(traces, style)
    return traces


#: A neutral fallback band color when an ``AREA`` layer carries no ``color``.
_DEFAULT_FILL_RGB: tuple[int, int, int] = (31, 119, 180)  # the default-palette blue


def _rgba(color: str | None, alpha: float) -> str:
    """Compose an ``rgba(...)`` fill string from a CSS color + an opacity in [0, 1].

    Baking the fill opacity into the ``fillcolor`` (rather than the trace-level
    ``opacity``) is what lets ``AREA`` honor a *separate* ``fillalpha`` for the
    band while a central line / the edges keep their own opacity.  A ``#rrggbb``
    hex or an ``rgb(r,g,b)`` color is parsed to its channels; anything else (a CSS
    name like ``"blue"``) is wrapped in plotly's own ``rgba`` over the parsed
    channels when possible, else falls back to the neutral palette blue so the
    band is always genuinely alpha-blended.
    """
    a = max(0.0, min(1.0, float(alpha)))
    r, g, b = _DEFAULT_FILL_RGB
    if isinstance(color, str):
        c = color.strip()
        if c.startswith("#") and len(c) == 7:
            with contextlib.suppress(ValueError):
                r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        elif c.startswith("#") and len(c) == 4:
            with contextlib.suppress(ValueError):
                r, g, b = int(c[1] * 2, 16), int(c[2] * 2, 16), int(c[3] * 2, 16)
        elif c.lower().startswith("rgb(") and c.endswith(")"):
            parts = c[4:-1].split(",")
            if len(parts) == 3:
                with contextlib.suppress(ValueError):
                    r, g, b = (int(float(p)) for p in parts)
        else:
            # A CSS name (e.g. "blue") — resolve its channels so the alpha is
            # genuinely honored rather than dropped.  matplotlib's color table is
            # the one always-present resolver for the full CSS name set (plotly's
            # own ``validate_colors`` does not parse names); falling through to the
            # neutral fill keeps the band alpha-blended if even that is absent.
            with contextlib.suppress(Exception):
                from matplotlib.colors import to_rgb

                fr, fg, fb = to_rgb(c)
                r, g, b = int(round(fr * 255)), int(round(fg * 255)), int(round(fb * 255))
    return f"rgba({r}, {g}, {b}, {a})"


def _build_area(layer: Layer, spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build an ``AREA`` mark — a shaded ``lo <= hi`` band over ``x``.

    Emits an invisible ``lo`` boundary trace plus a ``hi`` trace with
    ``fill="tonexty"`` so the band between them is shaded; a central ``y`` line is
    drawn through the fan when a distinct ``y`` is supplied alongside the edges.

    The band's opacity is the canonical **``fillalpha``** style key (AREA-only,
    default ``0.3``), baked into the ``rgba`` ``fillcolor`` so it is a genuine
    fill opacity independent of any line ``alpha``.  The canonical **``fill``**
    key (a bool, AREA / ENSEMBLE_FAN only) suppresses the shaded band when
    ``False`` — the edges / central line still draw, just unfilled.  Neither
    ``fill`` nor ``fillalpha`` is ever forwarded onto a line / scatter trace.
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
    style = _canon_style(layer)
    color = style.get("color")
    # fillalpha (AREA-only) is the band opacity; default 0.3 when unset.
    fillalpha = float(style.get("fillalpha", 0.3))
    # fill=False suppresses the shaded band (edges / line still draw).
    show_fill = bool(style.get("fill", True))
    fillcolor = _rgba(color, fillalpha) if show_fill else None
    hi_kw: dict[str, Any] = {
        "x": x,
        "y": hi,
        "mode": "lines",
        "line": {"width": 0},
        "name": layer.label,
        "showlegend": layer.label is not None,
    }
    if show_fill:
        hi_kw["fill"] = "tonexty"
        hi_kw["fillcolor"] = fillcolor
    traces: list[go.BaseTraceType] = [
        go.Scatter(x=x, y=lo, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"),
        go.Scatter(**hi_kw),
    ]
    theme = _resolve_theme(spec)
    if y is not None and "lo" in layer.data:
        traces.append(
            go.Scatter(x=x, y=y, mode="lines", line=_line_style(layer, theme), showlegend=False)
        )
    _apply_zorder(traces, style)
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
    style = _canon_style(layer)
    theme = _resolve_theme(spec)
    marker: dict[str, Any] = {}
    symbol = _marker_symbol(style)
    if symbol is not None:
        marker["symbol"] = symbol
    if "color" in style:
        marker["color"] = style["color"]
    ms = _marker_size(style, theme)
    if ms is not None:
        marker["size"] = ms
    traces: list[go.BaseTraceType] = [
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            error_y=error_y,
            name=layer.label,
            showlegend=layer.label is not None,
            opacity=_opacity(style),
        )
    ]
    _apply_zorder(traces, style)
    return traces


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
    style = _canon_style(layer)
    color = style.get("color")
    traces: list[go.BaseTraceType] = [
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
    _apply_zorder(traces, style)
    return traces


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


def _axis_layout(axis: Axis, theme: Theme | None = None) -> dict[str, Any]:
    """Build a plotly axis-layout dict from a typed :class:`Axis`.

    Honors the enriched fields added in the styling redesign:
    ``grid`` (per-axis grid visibility, overrides theme default),
    ``color`` (axis ink — spine, ticks, labels), ``label_size``, ``tick_size``,
    ``tick_rotation``.  When a field is ``None`` the theme / plotly default wins.
    """
    out: dict[str, Any] = {}
    if axis.label:
        title: dict[str, Any] = {"text": axis.label}
        if axis.label_size is not None:
            title["font"] = {"size": float(axis.label_size)}
        elif theme is not None and theme.font_size is not None:
            title["font"] = {"size": float(theme.font_size)}
        out["title"] = title
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

    # ── enriched styling fields ──────────────────────────────────────────────
    # Grid visibility: Axis.grid > theme.grid > plotly default (off).
    grid_on = axis.grid
    if grid_on is None and theme is not None:
        grid_on = theme.grid
    if grid_on is not None:
        out["showgrid"] = bool(grid_on)
        if grid_on and theme is not None:
            if theme.grid_color is not None:
                out["gridcolor"] = theme.grid_color
            if theme.grid_alpha is not None:
                # plotly uses ``gridwidth`` for grid line width, not alpha; express
                # alpha as opacity on the *line color* via rgba when we have a hex.
                # For simplicity store it as a layout meta and apply via ``gridcolor``
                # only if we can.  The safe path: leave plotly to alpha-blend normally.
                pass  # grid_alpha honored by mpl; plotly gridcolor handles it visually

    # Axis ink / color (spine, label, ticks).
    ink = axis.color
    if ink is None and theme is not None:
        ink = theme.foreground
    if ink is not None:
        out["tickcolor"] = ink
        out["linecolor"] = ink
        out["title"] = {
            **out.get("title", {}),
            "font": {**out.get("title", {}).get("font", {}), "color": ink},
        }

    # Tick font size.
    if axis.tick_size is not None:
        out["tickfont"] = {"size": float(axis.tick_size)}
    elif theme is not None and theme.font_size is not None:
        out["tickfont"] = {"size": float(theme.font_size)}

    # Tick label rotation.
    if axis.tick_rotation is not None:
        out["tickangle"] = float(axis.tick_rotation)

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
# Theme → plotly layout translation
# ---------------------------------------------------------------------------


def _theme_layout(theme: Theme) -> dict[str, Any]:
    """Build the theme portion of a plotly ``update_layout`` dict.

    Translates:
    - ``background`` → ``paper_bgcolor`` + ``plot_bgcolor``
    - ``foreground`` → ``font.color``
    - ``font_family`` → ``font.family``
    - ``font_size`` → ``font.size``
    """
    layout: dict[str, Any] = {}
    if theme.background is not None:
        layout["paper_bgcolor"] = theme.background
        layout["plot_bgcolor"] = theme.background
    font: dict[str, Any] = {}
    if theme.foreground is not None:
        font["color"] = theme.foreground
    if theme.font_family is not None:
        font["family"] = theme.font_family
    if theme.font_size is not None:
        font["size"] = float(theme.font_size)
    if font:
        layout["font"] = font
    return layout


def _legend_layout(spec: PlotSpec, theme: Theme) -> dict[str, Any]:
    """Build the legend portion of a plotly ``update_layout`` dict.

    Honors :class:`~tsdynamics.viz.spec.Legend` enriched fields:
    ``font_size`` (legend entry font), ``ncol`` (``traceorder`` approximation),
    ``frame`` (``bgcolor`` / ``bordercolor`` to toggle the box).
    """
    leg = spec.legend
    if leg is None or not leg.show:
        return {}
    out: dict[str, Any] = {}
    if leg.title:
        out["title"] = {"text": leg.title}
    font: dict[str, Any] = {}
    if leg.font_size is not None:
        font["size"] = float(leg.font_size)
    elif theme.font_size is not None:
        font["size"] = float(theme.font_size)
    if theme.foreground is not None:
        font["color"] = theme.foreground
    if font:
        out["font"] = font
    # ncol: plotly does not have a direct "columns" knob; it mirrors an implicit
    # horizontal orientation when ncol > 1.
    if leg.ncol > 1:
        out["orientation"] = "h"
    # frame: suppress the legend box via bgcolor/bordercolor transparency.
    if not leg.frame:
        out["bgcolor"] = "rgba(0,0,0,0)"
        out["bordercolor"] = "rgba(0,0,0,0)"
    return out


# ---------------------------------------------------------------------------
# 2-D panel builders (factored so the composite renderer can reuse them)
# ---------------------------------------------------------------------------


def build_2d_traces(spec: PlotSpec) -> list[go.BaseTraceType]:
    """Build every 2-D trace for ``spec`` through :data:`MARK_DISPATCH`.

    Factored out of :func:`render` so the composite renderer can draw a 2-D
    panel's traces into one cell of a :func:`plotly.subplots.make_subplots` grid
    (rather than its own figure).  A layer whose mark has no builder is skipped.
    """
    traces: list[go.BaseTraceType] = []
    for layer in spec.layers:
        builder = MARK_DISPATCH.get(PlotKind(layer.kind))
        if builder is None:
            continue
        traces.extend(builder(layer, spec))
    return traces


def _build_2d_layout(spec: PlotSpec) -> dict[str, Any]:
    """Build the single-panel 2-D ``update_layout`` dict (axes / legend / annotations).

    Mirrors the inline body :func:`render` used before the composite refactor:
    the typed x / y axes (with an ``"equal"`` aspect mapped to a constrained
    y-axis and hidden axes honoured), the legend, the title, and the
    ``vline`` / ``hline`` / ``text`` / ``span`` annotations as layout shapes /
    annotations.  The resolved :class:`~tsdynamics.viz.style.Theme` is applied
    (background, font, grid defaults) before the per-axis overrides.
    """
    theme = _resolve_theme(spec)
    xaxis = _axis_layout(spec.x, theme)
    yaxis = _axis_layout(spec.y, theme)
    if spec.aspect == "equal":
        yaxis["scaleanchor"] = "x"
        yaxis["scaleratio"] = 1
    if spec._axes_hidden():
        xaxis["visible"] = False
        yaxis["visible"] = False

    show_legend = spec.legend is not None and spec.legend.show
    layout: dict[str, Any] = {
        "xaxis": xaxis,
        "yaxis": yaxis,
        "showlegend": show_legend,
    }
    # Apply theme-level presentation.
    layout.update(_theme_layout(theme))

    if spec.title:
        title_dict: dict[str, Any] = {"text": spec.title}
        if theme.title_size is not None:
            title_dict["font"] = {"size": float(theme.title_size)}
        layout["title"] = title_dict
    if show_legend:
        leg_dict = _legend_layout(spec, theme)
        if leg_dict:
            layout["legend"] = leg_dict

    shapes, texts = _annotation_shapes(spec.annotations)
    if shapes:
        layout["shapes"] = shapes
    if texts:
        layout["annotations"] = texts
    return layout


# ---------------------------------------------------------------------------
# The render entry point
# ---------------------------------------------------------------------------


def render(
    spec: PlotSpec,
    *,
    html: bool = False,
    path: str | os.PathLike[str] | None = None,
    full_html: bool | None = None,
    include_plotlyjs: str | bool = "cdn",
    **_kw: Any,
) -> Any:
    """Render a 2-D :class:`~tsdynamics.viz.spec.PlotSpec` to a plotly Figure.

    Builds a :class:`plotly.graph_objects.Figure`, adds every
    :class:`~tsdynamics.viz.spec.Layer`'s traces through :data:`MARK_DISPATCH`,
    then applies the typed axes (label / scale / limits / categories), the
    legend, and the ``vline`` / ``hline`` / ``text`` / ``span`` annotations as
    plotly layout shapes / annotations.  The spec's semantic kind is normalised
    through :func:`tsdynamics.viz.render.normalize_kind` (an alias / mark spelling
    resolves to a real kind), and an ``"equal"`` aspect maps to a constrained
    y-axis so phase portraits / sections / images keep their geometry.

    The resolved :class:`~tsdynamics.viz.style.Theme` (``spec.resolved_theme``:
    the spec's own theme or the active global default) is applied to the plotly
    layout:
    ``paper_bgcolor`` / ``plot_bgcolor``, ``layout.font`` (family + size),
    axis ``showgrid`` (with ``gridcolor``).  Per-layer canonical style keys
    (``linewidth`` → ``line.width``, ``linestyle`` → ``line.dash``,
    ``marker`` → ``marker.symbol``, ``markersize`` → ``marker.size``,
    ``alpha`` → ``opacity``, ``cmap`` → ``colorscale``) are translated from
    their canonical names to plotly idioms.  Legend enriched fields
    (``font_size``, ``ncol``, ``frame``) are honored where plotly allows.

    Parameters
    ----------
    spec : PlotSpec
        The backend-agnostic spec to draw.  Its per-call tweaks
        (relabel / rescale / limits / ticks / colorize / style / theme) are
        already baked into the typed axes / colorbar / theme / layers, so
        honouring those honours the tweaks.
    html : bool, optional
        When ``True`` return the self-contained interactive **HTML string**
        instead of the live figure — so ``result.plot(backend="plotly", html=True)``
        yields HTML ready to drop into a web page.  Default ``False``.
    path : str or os.PathLike, optional
        When given, **write** the self-contained interactive HTML to this file
        and return the :class:`pathlib.Path`.  Implies HTML export.
    full_html : bool, optional
        Whether the exported HTML is a standalone document or a bare fragment.
        Default ``False`` for ``html=True``, ``True`` for ``path=``.
    include_plotlyjs : str or bool, optional
        How to provide the plotly bundle; ``"cdn"`` (the default) references it
        from a CDN.
    **_kw
        Forwarded but unused backend keywords (kept for a uniform renderer
        signature).

    Returns
    -------
    plotly.graph_objects.Figure or str or pathlib.Path
        The interactive figure by default; the HTML **string** when ``html`` is
        set; the written :class:`pathlib.Path` when ``path`` is given.

    Notes
    -----
    A 3-D spec (``ndim == 3`` / a ``z`` axis / a ``LINE3D`` / ``SURFACE3D`` mark)
    is dispatched to the :mod:`._threed` renderer, which draws orbitable
    ``go.Scatter3d`` / ``go.Surface`` traces on a 3-D ``scene``.
    """
    import plotly.graph_objects as go

    from . import _threed

    normalize_kind(spec.kind)  # validate / canonicalise the semantic kind

    if spec.is_animated:
        from ._anim import animated_html, build_animated_figure

        # HTML export → the smooth, rotatable real-time (requestAnimationFrame +
        # restyle) animation; a live figure (notebook) → the frames player.
        if path is not None or html:
            return animated_html(
                spec,
                path=path,
                html=html,
                full_html=full_html,
                include_plotlyjs=include_plotlyjs,
            )
        return build_animated_figure(spec)
    if spec.is_composite:
        from ._composite import render_composite

        return render_composite(
            spec,
            html=html,
            path=path,
            full_html=full_html,
            include_plotlyjs=include_plotlyjs,
            **_kw,
        )
    if _threed.is_three_d(spec):
        return _threed.render_3d(spec, **_kw)
    else:
        fig = go.Figure()
        for trace in build_2d_traces(spec):
            fig.add_trace(trace)
        fig.update_layout(**_build_2d_layout(spec))

    if path is not None:
        # Write a self-contained interactive HTML file (standalone by default).
        from ._html import write_html

        return write_html(
            fig,
            path,
            full_html=True if full_html is None else full_html,
            include_plotlyjs=include_plotlyjs,
        )
    if html:
        # Return the embeddable HTML fragment (no kernel needed to render it).
        from ._html import to_html

        return to_html(
            fig,
            full_html=False if full_html is None else full_html,
            include_plotlyjs=include_plotlyjs,
        )
    return fig
