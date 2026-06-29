"""Analysis-facing :class:`~tsdynamics.viz.spec.PlotSpec` builder (the result-side seam).

The trajectory side funnels every plot through the parameterised
:mod:`tsdynamics.viz.producers`; the **analysis** side — the ~two-dozen
:class:`~tsdynamics.analysis._result.AnalysisResult` ``to_plot_spec`` methods —
used to each hand-assemble the same ``Layer(...)`` / ``Axis(...)`` /
``PlotSpec(...)`` boilerplate inline.  This module is the analysis counterpart of
the producers: a handful of small, engine-free, backend-free helpers that
encapsulate the recurring build-layers / set-axes / wrap-and-copy-meta patterns,
so a result class describes *what* it draws (a scaling fit, a diagnostic curve, a
sparse recurrence scatter, a bar readout, an image) and the IR vocabulary lives
**here**, in one place, instead of being threaded through every result module.

Like the producers it imports the :mod:`tsdynamics.viz.spec` IR **lazily** (every
function-body import is deferred), so building a result — or merely importing
:mod:`tsdynamics` — never pulls a plotting library.

The catalogue
-------------
Two layers of helper:

- **Layer / annotation constructors** — :func:`line`, :func:`line3d`,
  :func:`scatter`, :func:`markers`, :func:`bar`, :func:`image`, :func:`area`,
  :func:`errorbar`, :func:`histogram` (one per low-level mark a result draws) and
  :func:`vline` / :func:`hline` / :func:`span` / :func:`text` (the reference
  overlays).  A result builds its drawables with these and never names a raw
  :class:`~tsdynamics.viz.spec.PlotKind` mark.
- **Spec assemblers** — :func:`spec` (the general "resolve the kind, set the
  labelled axes, attach legend / colorbar / annotations / meta" wrapper every
  bespoke site shares) plus the named-pattern shortcuts :func:`scaling_fit` (the
  log--log SCATTER + fit-region MARKERS + fit LINE schema the dimension /
  Lyapunov-from-data estimators emit) and :func:`diagonal` (the ``y = x``
  reference line a return map / cobweb draws under its scatter).

Every produced spec uses only the closed
:class:`~tsdynamics.viz.spec.Layer` channel vocabulary and the frozen
:class:`~tsdynamics.viz.spec.PlotKind` enum, so it round-trips losslessly through
:meth:`~tsdynamics.viz.spec.PlotSpec.to_dict` /
:meth:`~tsdynamics.viz.spec.PlotSpec.from_dict` exactly as the hand-built specs
did — this is a *refactor*, output-equivalent by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tsdynamics.viz.spec import Annotation, Layer, PlotKind, PlotSpec

__all__ = [
    "area",
    "bar",
    "diagonal",
    "errorbar",
    "histogram",
    "hline",
    "image",
    "line",
    "line3d",
    "markers",
    "scaling_fit",
    "scatter",
    "spec",
    "text",
    "vline",
]


# ---------------------------------------------------------------------------
# Layer constructors (one per low-level mark a result draws)
# ---------------------------------------------------------------------------


def _layer(
    mark: str,
    data: Mapping[str, Any],
    *,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build one :class:`~tsdynamics.viz.spec.Layer` (the lazy-import choke point)."""
    from tsdynamics.viz.spec import Layer, PlotKind

    return Layer(
        PlotKind(mark),
        dict(data),
        label=label,
        style=dict(style) if style else {},
    )


def line(
    x: Any,
    y: Any,
    *,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build a 2-D ``LINE`` layer over ``(x, y)``."""
    return _layer("line", {"x": x, "y": y}, label=label, style=style)


def line3d(
    x: Any,
    y: Any,
    z: Any,
    *,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build a 3-D ``LINE3D`` layer over ``(x, y, z)``."""
    return _layer("line3d", {"x": x, "y": y, "z": z}, label=label, style=style)


def scatter(
    x: Any,
    y: Any,
    *,
    cat: Any = None,
    c: Any = None,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build a 2-D ``SCATTER`` point cloud over ``(x, y)``.

    ``cat`` (per-point category index, paired with a categorical palette) and
    ``c`` (per-point scalar colour field) are optional extra channels.
    """
    data: dict[str, Any] = {"x": x, "y": y}
    if cat is not None:
        data["cat"] = cat
    if c is not None:
        data["c"] = c
    return _layer("scatter", data, label=label, style=style)


def markers(
    x: Any,
    y: Any,
    *,
    z: Any = None,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build a ``MARKERS`` layer over ``(x, y)`` (a few points), optionally 3-D (``z``)."""
    data: dict[str, Any] = {"x": x, "y": y}
    if z is not None:
        data["z"] = z
    return _layer("markers", data, label=label, style=style)


def bar(
    y: Any,
    *,
    x: Any = None,
    cat: Any = None,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build a ``BAR`` layer of heights ``y``, positioned by ``x`` (index) or ``cat`` (category)."""
    data: dict[str, Any] = {"y": y}
    if cat is not None:
        data["cat"] = cat
    if x is not None:
        data["x"] = x
    return _layer("bar", data, label=label, style=style)


def image(
    c: Any,
    *,
    x: Any = None,
    y: Any = None,
    z: Any = None,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build an ``IMAGE`` layer with color field ``c`` (and optional ``x`` / ``y`` / ``z``)."""
    data: dict[str, Any] = {"c": c}
    if x is not None:
        data["x"] = x
    if y is not None:
        data["y"] = y
    if z is not None:
        data["z"] = z
    return _layer("image", data, label=label, style=style)


def area(
    x: Any,
    y: Any,
    *,
    lo: Any,
    hi: Any,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build an ``AREA`` band layer (``lo <= hi`` envelope around ``y`` over ``x``)."""
    return _layer("area", {"x": x, "y": y, "lo": lo, "hi": hi}, label=label, style=style)


def errorbar(
    x: Any,
    y: Any,
    err: Any,
    *,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build an ``ERRORBAR`` layer (symmetric ``err`` magnitudes over ``(x, y)``)."""
    return _layer("errorbar", {"x": x, "y": y, "err": err}, label=label, style=style)


def histogram(
    x: Any,
    *,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build a ``HISTOGRAM`` layer over the sample values ``x``."""
    return _layer("histogram", {"x": x}, label=label, style=style)


# ---------------------------------------------------------------------------
# Annotation constructors (reference lines / shaded bands / text overlays)
# ---------------------------------------------------------------------------


def vline(x: float, *, text: str = "", style: Mapping[str, Any] | None = None) -> Annotation:
    """Build a vertical reference line at constant ``x``."""
    from tsdynamics.viz.spec import Annotation

    return Annotation(kind="vline", x=float(x), text=text, style=dict(style) if style else {})


def hline(y: float, *, text: str = "", style: Mapping[str, Any] | None = None) -> Annotation:
    """Build a horizontal reference line at constant ``y``."""
    from tsdynamics.viz.spec import Annotation

    return Annotation(kind="hline", y=float(y), text=text, style=dict(style) if style else {})


def span(
    lo: float,
    hi: float,
    *,
    axis: str = "x",
    text: str = "",
    style: Mapping[str, Any] | None = None,
) -> Annotation:
    """Build a shaded band between ``lo`` and ``hi`` along ``axis`` (a rejection tail / fit region)."""
    from tsdynamics.viz.spec import Annotation

    return Annotation(
        kind="span",
        span=(float(lo), float(hi)),
        axis=axis,  # type: ignore[arg-type]
        text=text,
        style=dict(style) if style else {},
    )


def text(x: float, y: float, label: str, *, style: Mapping[str, Any] | None = None) -> Annotation:
    """Build a text overlay at ``(x, y)``."""
    from tsdynamics.viz.spec import Annotation

    return Annotation(
        kind="text", x=float(x), y=float(y), text=label, style=dict(style) if style else {}
    )


# ---------------------------------------------------------------------------
# Spec assemblers
# ---------------------------------------------------------------------------


def spec(
    kind: str | None,
    default: PlotKind | str,
    *,
    layers: Sequence[Layer],
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str | None = None,
    title: str = "",
    ndim: int | None = None,
    aspect: str = "auto",
    xlimits: tuple[float, float] | None = None,
    ylimits: tuple[float, float] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    xcategories: Sequence[str] | None = None,
    xticks: Sequence[float] | None = None,
    annotations: Sequence[Annotation] | None = None,
    legend: bool | Any = False,
    colorbar: Any = None,
    clim: tuple[float, float] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> PlotSpec:
    """Assemble a finished :class:`~tsdynamics.viz.spec.PlotSpec` from labelled axes + layers.

    The single "resolve the semantic kind, build the labelled
    :class:`~tsdynamics.viz.spec.Axis` pair (or triple), attach the optional
    legend / colorbar / annotations / clim and copy meta" wrapper every bespoke
    ``to_plot_spec`` opens and closes with.  ``kind`` is the caller's override (or
    ``None``) and ``default`` the result's natural kind — resolved through the
    closed :class:`~tsdynamics.viz.spec.PlotKind` vocabulary exactly as before.

    ``ndim`` defaults to ``3`` when a ``zlabel`` is given, else ``2``.  ``legend``
    accepts a ready :class:`~tsdynamics.viz.spec.Legend`, or ``True`` for a default
    one, or ``False`` for none.  The ``meta`` mapping is shallow-copied onto the
    spec (provenance is never shared by reference).
    """
    from tsdynamics.viz.spec import Axis, Legend, PlotKind, PlotSpec

    spec_kind = PlotKind(kind) if kind is not None else PlotKind(default)
    z_axis = Axis(label=zlabel) if zlabel is not None else None
    resolved_ndim = ndim if ndim is not None else (3 if z_axis is not None else 2)

    if legend is True:
        legend_obj: Any = Legend()
    elif legend is False:
        legend_obj = None
    else:
        legend_obj = legend

    return PlotSpec(
        kind=spec_kind,
        ndim=resolved_ndim,  # type: ignore[arg-type]
        aspect=aspect,  # type: ignore[arg-type]
        title=title,
        x=Axis(
            label=xlabel,
            scale=xscale,  # type: ignore[arg-type]
            limits=xlimits,
            categories=xcategories,
            ticks=xticks,
        ),
        y=Axis(label=ylabel, scale=yscale, limits=ylimits),  # type: ignore[arg-type]
        z=z_axis,
        layers=list(layers),
        legend=legend_obj,
        colorbar=colorbar,
        clim=clim,
        annotations=list(annotations) if annotations else [],
        meta=dict(meta) if meta else {},
    )


def scaling_fit(
    kind: str | None,
    x: Any,
    y: Any,
    *,
    fit_region: tuple[int, int],
    slope: float,
    line_y: Any | None = None,
    intercept: float | None = None,
    curve_label: str = "curve",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    default: PlotKind | str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> PlotSpec:
    r"""Build the log--log ``SCALING_FIT`` schema shared by every scaling estimator.

    The recurring three-layer figure: a ``SCATTER`` of the whole ``(x, y)`` curve,
    a ``MARKERS`` highlight of the fitted scaling region ``x[lo:hi+1]``, and a
    ``LINE`` of the fit of the given ``slope`` drawn across the fit region — the
    schema the fractal-dimension estimators, ``LyapunovFromData`` and the
    uncertainty exponent all emit, so a single ``result.plot.scaling()`` renders
    them identically.

    The fit line is given **either** as a ready ``line_y`` array (already evaluated
    at ``[x[lo], x[hi]]``, e.g. when the line is anchored to a centroid) **or** via
    ``intercept`` (then ``y = intercept + slope * [x[lo], x[hi]]``).  The
    fit-region layers are added only when the curve is non-empty and the region is
    valid (``hi >= lo``).

    Parameters
    ----------
    kind : str or None
        The caller's semantic-kind override.
    x, y : array-like
        The log--log curve.
    fit_region : tuple of int
        ``(lo, hi)`` **inclusive** index bounds of the fitted scaling region.
    slope : float
        The estimated slope (the dimension / exponent) — used in the fit-line
        label ``slope = <slope>``.
    line_y : array-like, optional
        The fit-line ordinates at ``[x[lo], x[hi]]``; mutually exclusive with
        ``intercept``.
    intercept : float, optional
        The fit-line intercept (``y = intercept + slope * x``); used when
        ``line_y`` is not given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = fit_region
    layers: list[Layer] = [scatter(x, y, label=curve_label)]
    if x.size and hi >= lo:
        layers.append(markers(x[lo : hi + 1], y[lo : hi + 1], label="fit region"))
        fit_x = np.array([x[lo], x[hi]], dtype=float)
        if line_y is not None:
            fit_y = np.asarray(line_y, dtype=float)
        else:
            fit_y = float(intercept if intercept is not None else 0.0) + slope * fit_x
        layers.append(line(fit_x, fit_y, label=f"slope = {slope:.3g}"))
    return spec(
        kind,
        default if default is not None else "scaling_fit",
        layers=layers,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        meta=meta,
    )


def diagonal(x: Any, y: Any, *, label: str = "$v_{n+1}=v_n$") -> Layer:
    r"""Build the ``y = x`` reference ``LINE`` spanning the combined range of ``x`` and ``y``.

    The fixed-point locus a first-return map / cobweb draws under its scatter — a
    line from ``(m, m)`` to ``(M, M)`` where ``[m, M]`` is the range of both
    series combined.
    """
    both = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    diag = np.array([float(both.min()), float(both.max())])
    return line(diag, diag, label=label)
