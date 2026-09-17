r"""The phase-plane transforms — nullclines, direction fields, streamlines, tr-det.

Five model-only transforms, all drawn on a 2-D **slice** of state space and all
thin adapters over :mod:`tsdynamics.analysis.planar`:

=====================  ==========  ==================================================
transform              default     what it draws
=====================  ==========  ==================================================
``nullclines``         ``line``    the curves :math:`f_i = 0`
``vector_field``       ``quiver``  unit arrows (``normalize=False`` for true lengths)
``flow_speed``         ``image``   :math:`\|f\|` as a backdrop under the arrows
``streamlines``        ``line``    integral curves of the sliced field
``trace_determinant``  ``points``  the equilibria on the :math:`(\tau, \Delta)` plane
=====================  ==========  ==================================================

The payoff is that they **compose**.  Frame identity means a direction field, its
nullclines, its equilibria and an orbit are four drawings of one ``state2(x, y)``
plane, so::

    ts.plot(VanDerPol(), "vector_field", "nullclines", "streamlines")

is one set of axes, in role order (field under curves under markers), and the
nullclines pass through the equilibria — which is a correctness check on both at
once, because they are computed by completely different code.

Every one of them needs the right-hand side at points no trajectory visited, so
every one is ``source="model"``: handed a bare
:class:`~tsdynamics.data.Trajectory` they raise, naming the system they need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

import tsdynamics.analysis.planar as _planar

from .._frames import FrameSpace, OverlayRole
from ..spec import PlotKind
from ._base import Geometry, Part, Presentation, make_frame
from ._registry import plot_transform

__all__ = [
    "flow_speed",
    "nullclines",
    "streamlines",
    "trace_determinant",
    "vector_field",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _join(curves: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    r"""Join polylines into one ``NaN``-separated pair of channels.

    Several disconnected branches of one curve — the two lines of a
    Lotka-Volterra nullcline, the sixty streamlines of a field — are **one
    object**, and drawing them as N layers makes the renderer give each branch
    its own palette colour and its own legend entry.  A ``NaN`` between branches
    breaks the stroke on every backend that draws lines (verified on
    matplotlib, plotly, JSON and three.js, whose payload bounds ignore the
    separators), so one nullcline is one colour and one legend entry, which is
    what the picture means.
    """
    if not curves:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    gap = np.array([np.nan])
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for k, seg in enumerate(curves):
        arr = np.asarray(seg, dtype=float)
        if k:
            xs.append(gap)
            ys.append(gap)
        xs.append(arr[:, 0])
        ys.append(arr[:, 1])
    return np.concatenate(xs), np.concatenate(ys)


def _slice_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an estimator's provenance onto the spec, unrenamed and complete.

    Every auto-chosen default a model transform made — the window and how it was
    picked, the frozen base state, the grid, the horizon — travels to
    ``spec.meta`` under the estimator's own key.  A model plot that quietly chose
    its own region is the one that shows half the story with nothing to say so,
    and a plot layer that renames the estimator's keys on the way is the reason
    nobody can find them.
    """
    return dict(meta)


def _demo_flow() -> Any:
    """Return the small, fast, deterministic planar flow the gate draws on."""
    from tsdynamics.systems.continuous.population_dynamics import LotkaVolterra

    return LotkaVolterra()


#: A window that contains both Lotka-Volterra equilibria — ``(0, 0)`` and the
#: coexistence centre at ``(4, 2.75)`` — so every example figure is non-degenerate.
_DEMO_WINDOW: dict[str, Any] = {"xlim": (0.0, 8.0), "ylim": (0.0, 6.0)}


# ---------------------------------------------------------------------------
# nullclines
# ---------------------------------------------------------------------------


@plot_transform(
    name="nullclines",
    # It evaluates the vector field at points no trajectory visits, through an
    # analysis that requires a continuous SYSTEM (measured: a bare callable is
    # refused by the estimator itself).
    subjects=("flow",),
    source="model",
    kind=PlotKind.PHASE_PORTRAIT_2D,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points"),
    presentation=Presentation(aspect="equal"),
    analysis="tsdynamics.analysis.planar.nullclines",
    example=lambda primitive: (_demo_flow(), {**_DEMO_WINDOW, "grid": 41}),
    doc="The curves f_i = 0 on a 2-D slice; their crossings are the equilibria.",
)
def nullclines(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 201,
    components: Sequence[int | str] | None = None,
) -> Geometry:
    r"""Draw the nullclines of a 2-D slice — one labelled curve per component.

    Where :math:`\dot x = 0` the flow is vertical; where :math:`\dot y = 0` it is
    horizontal; where they cross, it stops.  That last fact is the check worth
    making every time: overlay ``fixed_points`` and the equilibrium markers must
    sit exactly on the crossings, and the two were computed by entirely separate
    code (marching squares over a lattice versus multi-start Newton).

    Parameters
    ----------
    system : ContinuousSystem
        The flow.  A bare trajectory cannot serve this: the zero set is a
        statement about points no orbit visited.
    plane : sequence of int or str, optional
        The two coordinates of the slice, by name (``("x", "z")``) or index.
    at : array-like, optional
        The state the off-plane coordinates are frozen at.  Defaults to the
        system's ``default_ic``, else the origin — never a random draw.
    xlim, ylim : tuple of float, optional
        The window.  Auto-chosen (and recorded in ``spec.meta``) when omitted.
    grid : int or tuple of int, optional
        Marching-squares resolution.  Default ``201``.
    components : sequence of int or str, optional
        Which components' zero sets to draw.  ``None`` draws the two in-plane
        ones — the textbook pair.

    Returns
    -------
    Geometry

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If no component's zero set crosses the window at all.  Drawing an empty
        figure with a full legend would say "these nullclines are here" about
        curves that are not.
    """
    from tsdynamics.errors import InvalidParameterError

    _i, _j, labels = _planar.resolve_plane(system, plane)
    window_x, window_y, meta = _planar.window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)
    meta["grid"] = grid
    found = _planar.nullclines(
        system,
        plane=plane,
        at=at,
        xlim=window_x,
        ylim=window_y,
        grid=grid,
        components=components,
    )
    parts = [
        Part({"x": x, "y": y}, label=f"{nc.label}' = 0")
        for nc in found
        for x, y in [_join(nc.curves)]
        if x.size
    ]
    if not parts:
        raise InvalidParameterError(
            f"no nullcline of {type(system).__name__} crosses the window "
            f"x={window_x}, y={window_y}. Widen it with xlim=/ylim=, or raise grid= "
            "if the curve is thinner than a lattice cell."
        )
    return Geometry(
        "nullclines",
        make_frame(FrameSpace.STATE2, labels),
        parts,
        axis_labels=labels,
        axis_limits=(window_x, window_y),
        legend=True,
        meta=_slice_meta(meta),
    )


# ---------------------------------------------------------------------------
# vector_field (direction_field is its alias)
# ---------------------------------------------------------------------------


def _field_of_callable(
    rhs: Any,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    grid: int | tuple[int, int],
    normalize: bool,
    labels: tuple[str, str] | None,
) -> Geometry:
    """Sample a bare ``rhs([x, y]) -> [u, v]`` on a lattice (the no-System path)."""
    from tsdynamics.errors import InvalidParameterError

    from ._data import _quiver_channels

    if xlim is None or ylim is None:
        raise InvalidParameterError(
            "a bare right-hand side carries no state space, so vector_field cannot infer "
            "the window: pass xlim=(lo, hi) and ylim=(lo, hi). (Handed the *system* it "
            "infers them, like flow_speed does.)"
        )
    names = labels or ("x", "y")
    n = int(grid[0]) if isinstance(grid, tuple) else int(grid)
    return Geometry(
        "vector_field",
        make_frame(FrameSpace.STATE2, names),
        channels=_quiver_channels(rhs, xlim, ylim, n, normalize),
        axis_labels=names,
        axis_limits=(xlim, ylim),
    )


@plot_transform(
    name="vector_field",
    # It evaluates the vector field at points no trajectory visits: a continuous
    # system, or a bare right-hand side f(u, t) with no system around it.
    subjects=("flow", "function"),
    aliases=("direction_field",),
    source="model",
    kind=PlotKind.VECTOR_FIELD,
    frame=FrameSpace.STATE2,
    role=OverlayRole.FIELD,
    default_primitive="quiver",
    primitives=("quiver",),
    presentation=Presentation(aspect="equal"),
    analysis="tsdynamics.analysis.planar.flow_field",
    example=lambda primitive: (_demo_flow(), {**_DEMO_WINDOW, "grid": 7}),
    doc="The right-hand side as arrows on a lattice; unit-length by default.",
)
def vector_field(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 21,
    normalize: bool = True,
    color_by: str | None = None,
    labels: tuple[str, str] | None = None,
) -> Geometry:
    """Draw the vector field of a 2-D slice as arrows.

    Both textbook pictures live here, because they differ only in one drawing
    decision.  ``normalize=True`` (the default) draws **unit** arrows — the
    *direction field*, which is what a textbook draws and what reads correctly
    when the speed varies by orders of magnitude across the window (a relaxation
    oscillator draws one enormous arrow and a field of dots otherwise).
    ``normalize=False`` draws the arrows at their true lengths — the *vector
    field* proper, which shows where the flow is fast.

    It takes the **system** — which is the repair that made the guessable name
    the working one (``ts.plot(vdp, "vector_field")`` used to answer ``TypeError:
    vector_field() missing 2 required keyword-only arguments: 'xlim' and
    'ylim'``, while its sibling ``flow_speed`` inferred them) — so it can slice a
    flow of any dimension: ``plane=("x", "z")`` with ``at=``
    fixes the off-plane coordinates and samples the field on that plane.

    Parameters
    ----------
    system : ContinuousSystem
    plane, at, xlim, ylim
        The slice and window; see :func:`nullclines`.
    grid : int or tuple of int, optional
        Arrows per axis.  Default ``21``.
    normalize : bool, optional
        Unit-length arrows.  Default ``True``.
    color_by : {"speed"}, optional
        Colour the arrows by a scalar.  ``"speed"`` uses the **true** magnitude
        ``|f|``, so a normalized field still shows where the flow is fast — the
        best of both pictures.  ``None`` (default) draws plain arrows.

    Returns
    -------
    Geometry

    Notes
    -----
    On a slice of a higher-dimensional flow the arrows are the in-plane
    components of :math:`f` with the other coordinates frozen at ``at``: a
    correct picture of a 2-D slice, and *not* a picture of the full flow, whose
    trajectories leave the plane.  The frozen state is recorded in ``spec.meta``.
    """
    from tsdynamics.errors import InvalidParameterError
    from tsdynamics.families import SystemBase

    if not isinstance(system, SystemBase) and callable(system):
        # The field itself, handed in as a bare ``rhs([x, y]) -> [u, v]``.  One
        # name, two subject shapes — and they are the same mathematical object,
        # so this is not a second spelling of one concept: a caller who has the
        # right-hand side but no System should not be turned away from the
        # transform that draws right-hand sides.
        return _field_of_callable(system, xlim, ylim, grid, normalize, labels)
    field = _planar.flow_field(system, plane=plane, at=at, xlim=xlim, ylim=ylim, grid=grid)
    u, v = field.u, field.v
    if normalize:
        mag = np.where(field.speed < np.finfo(float).tiny, 1.0, field.speed)
        u, v = u / mag, v / mag
    gx, gy = np.meshgrid(field.xs, field.ys)
    channels: dict[str, np.ndarray] = {
        "x": gx.ravel(),
        "y": gy.ravel(),
        "u": u.ravel(),
        "v": v.ravel(),
    }
    color_label: str | None = None
    if color_by is not None:
        if color_by != "speed":
            raise InvalidParameterError(
                f"color_by={color_by!r} is not a field this transform computes; the only "
                "colour channel a direction field carries is 'speed' (the true |f|)."
            )
        channels["c"] = field.speed.ravel()
        color_label = "|f|"
    return Geometry(
        "vector_field",
        make_frame(FrameSpace.STATE2, field.labels),
        channels=channels,
        axis_labels=field.labels,
        axis_limits=(field.meta["xlim"], field.meta["ylim"]),
        color_label=color_label,
        meta={**_slice_meta(field.meta), "normalize": bool(normalize)},
    )


# ---------------------------------------------------------------------------
# flow_speed
# ---------------------------------------------------------------------------


@plot_transform(
    name="flow_speed",
    # It evaluates the vector field at points no trajectory visits, through an
    # analysis that requires a continuous SYSTEM (measured: a bare callable is
    # refused by the estimator itself).
    subjects=("flow",),
    source="model",
    kind=PlotKind.PHASE_PORTRAIT_2D,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="image",
    primitives=("image", "contour", "surface3d"),
    presentation=Presentation(aspect="equal", cmap="viridis", autocolor=True),
    analysis="tsdynamics.analysis.planar.flow_field",
    example=lambda primitive: (_demo_flow(), {**_DEMO_WINDOW, "grid": 12}),
    doc="The speed |f| of the flow as a scalar backdrop.",
)
def flow_speed(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 121,
    log: bool = False,
) -> Geometry:
    r"""Draw the speed :math:`\|f\|` of a 2-D slice as a scalar backdrop.

    The natural companion to a direction field: normalizing the arrows throws
    the magnitude away, and this puts it back as a colour field underneath.  Its
    near-zero valleys locate the equilibria and the slow branches of a
    relaxation oscillator; its ridges are where the flow is fast.

    Parameters
    ----------
    system : ContinuousSystem
    plane, at, xlim, ylim
        The slice and window; see :func:`nullclines`.
    grid : int or tuple of int, optional
        Lattice resolution.  Default ``121``.
    log : bool, optional
        Draw :math:`\log_{10}\|f\|` instead.  Default ``False``.  The speed of a
        real flow spans several decades, so the linear field is often one bright
        corner; the log field shows the structure.  Zero speed becomes ``NaN``
        (an equilibrium has no logarithm), which draws as a hole rather than as
        a false minimum.

    Returns
    -------
    Geometry
    """
    field = _planar.flow_field(system, plane=plane, at=at, xlim=xlim, ylim=ylim, grid=grid)
    values = field.speed
    label = "|f|"
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.log10(np.where(values > 0.0, values, np.nan))
        label = "log10 |f|"
    return Geometry(
        "flow_speed",
        make_frame(FrameSpace.STATE2, field.labels),
        channels={
            "x": field.xs,
            "y": field.ys,
            "z": values,
            "c": values.ravel(),
        },
        axis_labels=field.labels,
        axis_limits=(field.meta["xlim"], field.meta["ylim"]),
        color_label=label,
        meta={**_slice_meta(field.meta), "log": bool(log)},
    )


# ---------------------------------------------------------------------------
# streamlines
# ---------------------------------------------------------------------------


@plot_transform(
    name="streamlines",
    # It evaluates the vector field at points no trajectory visits, through an
    # analysis that requires a continuous SYSTEM (measured: a bare callable is
    # refused by the estimator itself).
    subjects=("flow",),
    source="model",
    kind=PlotKind.PHASE_PORTRAIT_2D,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="line",
    primitives=("line", "points"),
    presentation=Presentation(aspect="equal"),
    analysis="tsdynamics.analysis.planar.streamlines",
    example=lambda primitive: (_demo_flow(), {**_DEMO_WINDOW, "seeds": 3, "steps": 30}),
    doc="Integral curves of the sliced field, seeded on a lattice.",
)
def streamlines(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    seeds: int | tuple[int, int] = 8,
    length: float | None = None,
    steps: int = 200,
    both_ways: bool = True,
) -> Geometry:
    """Integral curves of a 2-D slice, seeded on a lattice.

    A **line** primitive, not a new ``STREAM`` mark, and the choice was made on
    the evidence rather than by preference.  ``matplotlib.streamplot`` would give
    arrow-decorated curves on one backend; a ``STREAM`` mark would then have to
    be grown on plotly, JSON and three.js as well, and it would carry no
    information a ``LINE`` does not — a streamline *is* a polyline.  Integrating
    here also uses the system's own right-hand side rather than a plotting
    library's interpolation of a pre-sampled lattice, which is exactly what
    matters near a separatrix, where the interpolated field and the real one
    part company.

    Every curve is one ``NaN``-separated layer, so the whole field is one colour
    and one legend entry.

    Parameters
    ----------
    system : ContinuousSystem
    plane, at, xlim, ylim
        The slice and window; see :func:`nullclines`.
    seeds : int or tuple of int, optional
        Seeds per axis, inset from the window edge.  Default ``8``.
    length : float, optional
        Arc length per direction.  ``None`` uses the window's diagonal.
    steps : int, optional
        Steps per direction.  Default ``200``.
    both_ways : bool, optional
        March backwards as well as forwards, so a streamline through a seed
        reaches structure on both sides of it.  Default ``True``.

    Returns
    -------
    Geometry
    """
    from tsdynamics.errors import InvalidParameterError

    _i, _j, labels = _planar.resolve_plane(system, plane)
    window_x, window_y, meta = _planar.window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)
    meta["seeds"] = seeds
    curves = _planar.streamlines(
        system,
        plane=plane,
        at=at,
        xlim=window_x,
        ylim=window_y,
        seeds=seeds,
        length=length,
        steps=steps,
        both_ways=both_ways,
    )
    x, y = _join(curves)
    if x.size == 0:
        raise InvalidParameterError(
            f"every streamline seed of {type(system).__name__} sits at a stagnation point of "
            f"the window x={window_x}, y={window_y}; there is no curve to draw. Move the "
            "window, or raise seeds=."
        )
    return Geometry(
        "streamlines",
        make_frame(FrameSpace.STATE2, labels),
        channels={"x": x, "y": y},
        label="streamlines",
        axis_labels=labels,
        axis_limits=(window_x, window_y),
        meta={**_slice_meta(meta), "n_streamlines": len(curves)},
    )


# ---------------------------------------------------------------------------
# trace_determinant
# ---------------------------------------------------------------------------


def _demo_trace_determinant() -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: Lotka-Volterra's two equilibria, supplied explicitly.

    Passing ``points=`` skips the multi-start Newton search, so the example is
    milliseconds rather than the better part of a second — and it still exercises
    the whole geometry (a saddle, a centre, the parabola and both axes).
    """
    return _demo_flow(), {"points": [[0.0, 0.0], [4.0, 2.75]]}


@plot_transform(
    name="trace_determinant",
    # It evaluates the vector field at points no trajectory visits, through an
    # analysis that requires a continuous SYSTEM (measured: a bare callable is
    # refused by the estimator itself).
    subjects=("flow",),
    source="model",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.PARAM2,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="points",
    primitives=("points",),
    presentation=Presentation(aspect="auto", legend=True),
    analysis="tsdynamics.analysis.planar.trace_determinant",
    example=lambda primitive: _demo_trace_determinant(),
    doc="A system's equilibria placed on the (tr J, det J) stability plane.",
    # ``points`` rather than ``markers``: on every backend the two lower to the
    # same scatter drawing, and ``markers`` is reserved for the equilibrium /
    # tipping-point *overlays* whose glyphs are stability-coded.  Nothing is
    # lost in the picture and the reserve stays honest.
)
def trace_determinant(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    points: Any | None = None,
    trace_range: tuple[float, float] | None = None,
    samples: int = 201,
    **fixed_point_kwargs: Any,
) -> Geometry:
    r"""Draw the trace-determinant stability plane, with a system's equilibria on it.

    The classic teaching figure, and one nothing in Python ships.  Every planar
    linearization is a single point :math:`(\tau, \Delta)` and its whole
    qualitative behaviour is read off from where that point falls:

    - :math:`\Delta < 0` (below the horizontal axis): **saddle**;
    - above the parabola :math:`\Delta = \tau^2/4`: complex eigenvalues, a
      **spiral** — a **centre** exactly on the vertical axis;
    - between the two: real same-sign eigenvalues, a **node**;
    - and the sign of :math:`\tau` decides stable from unstable.

    The geometry draws the parabola and both axes as pinned ``line`` parts, and
    the equilibria grouped by class — so the legend reads *saddle*, *centre*,
    *stable focus*, and each marker's position proves the label.

    Parameters
    ----------
    system : ContinuousSystem
    plane : sequence of int or str, optional
        For a flow with more than two coordinates, the 2x2 Jacobian sub-block to
        take the invariants of, i.e. the linearization **within that slice**.
    points : array-like, optional
        Equilibria, shape ``(n, dim)``.  ``None`` finds them with
        ``fixed_points`` (extra keywords are forwarded to it), which is the slow
        part — pass known equilibria to skip it.
    trace_range : tuple of float, optional
        The trace range the parabola spans.  ``None`` covers the equilibria.
    samples : int, optional
        Points on the parabola.  Default ``201``.
    **fixed_point_kwargs
        Forwarded to ``fixed_points``.

    Returns
    -------
    Geometry

    Notes
    -----
    The stability *regions* are delimited by the drawn curves rather than named
    in the figure: the plot vocabulary has no text mark, so a region label would
    have to be an annotation the geometry layer cannot carry today.  The legend
    names every equilibrium's class instead, which is the information a reader
    actually takes away.
    """
    result = _planar.trace_determinant(
        system,
        plane=plane,
        points=points,
        trace_range=trace_range,
        samples=samples,
        **fixed_point_kwargs,
    )
    taus = result.parabola[:, 0]
    dets = result.determinant
    det_lo = float(min(0.0, np.min(dets) if dets.size else 0.0))
    det_hi = float(max(np.max(result.parabola[:, 1]), np.max(dets) if dets.size else 0.0))
    span = max(det_hi - det_lo, 1.0)

    parts: list[Part] = [
        Part(
            {"x": taus, "y": result.parabola[:, 1]},
            label="det = tr^2 / 4",
            style={"linestyle": "solid"},
            primitive="line",
        ),
        Part(
            {"x": taus, "y": np.zeros_like(taus)},
            label="det = 0",
            style={"linestyle": "dashed", "color": "#888888"},
            primitive="line",
        ),
        Part(
            {
                "x": np.zeros(2),
                "y": np.array([det_lo - 0.05 * span, det_hi + 0.05 * span]),
            },
            label="tr = 0",
            style={"linestyle": "dashed", "color": "#888888"},
            primitive="line",
        ),
    ]
    for name in dict.fromkeys(result.classes):
        mask = np.array([cls == name for cls in result.classes], dtype=bool)
        parts.append(Part({"x": result.trace[mask], "y": dets[mask]}, label=name))

    return Geometry(
        "trace_determinant",
        make_frame(FrameSpace.PARAM2, ("tr J", "det J")),
        parts,
        axis_labels=("tr J", "det J"),
        axis_limits=(result.trace_range, (det_lo - 0.05 * span, det_hi + 0.05 * span)),
        legend=True,
        meta=dict(result.meta),
    )
