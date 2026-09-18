r"""The scalar-field transforms — FTLE / LCS, escape time, transient time, density.

Four transforms that paint a **number per initial condition** over a 2-D slice.
Three of them are the same engine call with three different reductions, which is
why they arrive together:

===================  ===========  =====================================================
transform            source       what the colour means
===================  ===========  =====================================================
``ftle``             ``model``    finite-time Lyapunov exponent; ridges are LCS
``escape_time``      ``model``    time to leave the window (fractal boundaries)
``transient_time``   ``model``    time to settle onto an attractor
``invariant_density``  ``data``   the natural measure of the attractor
===================  ===========  =====================================================

The first three are cheap because the engine already fans an ensemble out over
its thread pool: 10,201 initial conditions integrated to :math:`T = 1` is a few
milliseconds, so an FTLE field is that one call plus :func:`numpy.gradient` and
a 2x2 eigenvalue in closed form.

``invariant_density`` is the odd one out and belongs here anyway: it is the same
*shape* of answer (a scalar over a region) computed from data instead of from
the model, and it is the transform that finally gives the ``HISTOGRAM`` mark a
producer — the mark has sat in the frozen vocabulary with zero of them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import tsdynamics.analysis.planar as _planar

from .._frames import FrameSpace, OverlayRole
from .._visibility import listing_dir
from ..spec import PlotKind
from ._base import Geometry, Presentation, make_frame
from ._registry import plot_transform

__all__ = ["escape_time", "ftle", "invariant_density", "transient_time"]

__dir__ = listing_dir(__all__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _field_geometry(name: str, result: _planar.ScalarField) -> Geometry:
    """Wrap a :class:`~tsdynamics.analysis.planar.ScalarField` as drawable geometry.

    One shape for all three ensemble fields: 1-D ``x`` / ``y`` sample
    coordinates (which set an image's pixel extent and a contour's coordinates)
    plus the 2-D ``z`` field and its ravelled ``c`` twin, so ``image``,
    ``contour`` and ``surface3d`` all draw the same numbers.
    """
    return Geometry(
        name,
        make_frame(FrameSpace.STATE2, result.labels),
        channels={
            "x": result.xs,
            "y": result.ys,
            "z": result.values,
            "c": result.values.ravel(),
        },
        axis_labels=result.labels,
        axis_limits=(result.meta["xlim"], result.meta["ylim"]),
        color_label=result.label,
        meta=dict(result.meta),
    )


def _demo_flow() -> Any:
    """Return the small, fast, deterministic planar flow the gate draws on."""
    from tsdynamics.systems.continuous.population_dynamics import LotkaVolterra

    return LotkaVolterra()


#: A window containing both Lotka-Volterra equilibria, so no example is degenerate.
_DEMO_WINDOW: dict[str, Any] = {"xlim": (0.5, 8.0), "ylim": (0.5, 6.0)}


def _demo_settling_flow() -> Any:
    """Return a planar flow whose orbits genuinely *settle*, for the transient example.

    The default Brusselator sits past its Hopf bifurcation (``b > 1 + a^2``) and
    relaxes onto a limit cycle, whose speed never becomes small — so the default
    "the flow has slowed to a stop" arrival test would never fire and the example
    would be a field of ``NaN``.  Below the threshold the equilibrium is a stable
    focus and every orbit really does arrive, which is the case this transform is
    for.
    """
    from tsdynamics.systems.continuous.chem_bio_systems import Brusselator

    return Brusselator(b=1.5)


def _demo_orbit() -> Any:
    """Return a short deterministic orbit for the ``invariant_density`` example."""
    return _demo_flow().run(final_time=40.0, dt=0.02, ic=[4.0, 1.5])


# ---------------------------------------------------------------------------
# ftle
# ---------------------------------------------------------------------------


@plot_transform(
    name="ftle",
    # The analysis door spells it ``ftle_field``; one concept must not
    # need two words depending on which door you are at, so the analysis
    # spelling is an ALIAS here (one record, one matrix row).
    aliases=("ftle_field",),
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
    presentation=Presentation(aspect="equal", cmap="inferno", autocolor=True),
    analysis="tsdynamics.analysis.planar.ftle_field",
    example=lambda primitive: (_demo_flow(), {**_DEMO_WINDOW, "grid": 12, "final_time": 1.0}),
    doc="The finite-time Lyapunov exponent field; its ridges are the LCS.",
)
def ftle(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 101,
    final_time: float = 1.0,
    backward: bool = False,
    **integrate_kwargs: Any,
) -> Geometry:
    r"""Paint the finite-time Lyapunov exponent field — whose ridges are the LCS.

    Every lattice point is integrated for a time :math:`T`; the in-plane
    deformation gradient of that flow map is taken by finite differences, and the
    colour is :math:`\log\sqrt{\lambda_{\max}(F^\top F)} / |T|` — the largest
    stretching a small blob at that point experiences over the horizon.

    **The ridges are the Lagrangian coherent structures**: forward time draws the
    *repelling* ones (stable manifolds, and for a multistable system the basin
    boundary itself), ``backward=True`` the *attracting* ones.  Overlaying a
    forward FTLE field on a basin diagram of the same system is the check that
    the field is right, and it is a good deal more informative than either alone
    — the basin image says *which*, the FTLE ridge says *how sharply*.

    Parameters
    ----------
    system : ContinuousSystem
    plane, at, xlim, ylim
        The slice and window; see
        :func:`~tsdynamics.viz.transforms.planar.nullclines`.
    grid : int or tuple of int, optional
        Lattice resolution.  Default ``101``.  The gradient is a finite
        difference on this lattice, so a ridge thinner than a cell is smeared,
        never sharpened — raising ``grid`` is what sharpens a ridge, not raising
        ``final_time``.
    final_time : float, optional
        The horizon :math:`T`.  Default ``1.0``.  The field genuinely depends on
        it (there is no :math:`T \to \infty` picture to converge to), so it is a
        stated choice and is recorded in ``spec.meta``.
    backward : bool, optional
        Integrate the time-reversed flow: attracting structures.  Default
        ``False``.
    **integrate_kwargs
        Forwarded to the engine ensemble (``method``, ``rtol``, ``backend``, ...).

    Returns
    -------
    Geometry
    """
    return _field_geometry(
        "ftle",
        _planar.ftle_field(
            system,
            plane=plane,
            at=at,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            final_time=final_time,
            backward=backward,
            **integrate_kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# escape_time
# ---------------------------------------------------------------------------


@plot_transform(
    name="escape_time",
    # The analysis door spells it ``escape_time_field``; one concept must not
    # need two words depending on which door you are at, so the analysis
    # spelling is an ALIAS here (one record, one matrix row).
    aliases=("escape_time_field",),
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
    presentation=Presentation(aspect="equal", cmap="magma", autocolor=True),
    analysis="tsdynamics.analysis.planar.escape_time_field",
    example=lambda primitive: (
        _demo_flow(),
        {**_DEMO_WINDOW, "grid": 12, "final_time": 2.0, "chunks": 4},
    ),
    doc="How long each initial condition takes to leave the region.",
)
def escape_time(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 101,
    final_time: float = 20.0,
    chunks: int = 40,
    escape: Callable[[np.ndarray], np.ndarray] | float | None = None,
    **integrate_kwargs: Any,
) -> Geometry:
    """Paint the escape-time field — the classic fractal-basin-boundary picture.

    Colour each initial condition by how long its orbit stays in the region.
    Near a fractal boundary the answer varies wildly between neighbouring cells,
    so the level sets accumulate there and the picture resolves structure a
    two-colour basin diagram flattens away.

    Parameters
    ----------
    system : ContinuousSystem
    plane, at, xlim, ylim, grid
        The slice, window and resolution.
    final_time : float, optional
        The horizon.  Default ``20``.  A point that has not escaped by then is
        ``NaN`` — "it did not escape within the horizon", which is honest, rather
        than a bright pixel claiming it escaped exactly at the end.
    chunks : int, optional
        Time resolution: the answer is quantised to ``final_time / chunks``,
        recorded in ``spec.meta``.  Default ``40``.  The march is *one*
        integration split into that many segments, not that many integrations.
    escape : callable or float, optional
        What escaping means.  ``None`` (default) is *left the drawn window*; a
        float is an in-plane radius from the window centre; a callable takes the
        ``(n, dim)`` states and returns an ``(n,)`` boolean.
    **integrate_kwargs
        Forwarded to the engine ensemble.

    Returns
    -------
    Geometry
    """
    return _field_geometry(
        "escape_time",
        _planar.escape_time_field(
            system,
            plane=plane,
            at=at,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            final_time=final_time,
            chunks=chunks,
            escape=escape,
            **integrate_kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# transient_time
# ---------------------------------------------------------------------------


@plot_transform(
    name="transient_time",
    # The analysis door spells it ``transient_time_field``; one concept must not
    # need two words depending on which door you are at, so the analysis
    # spelling is an ALIAS here (one record, one matrix row).
    aliases=("transient_time_field",),
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
    analysis="tsdynamics.analysis.planar.transient_time_field",
    example=lambda primitive: (
        _demo_settling_flow(),
        {
            "xlim": (0.2, 2.5),
            "ylim": (0.5, 3.0),
            "grid": 12,
            "final_time": 12.0,
            "chunks": 8,
            "tol": 0.3,
        },
    ),
    doc="How long each initial condition takes to settle onto its attractor.",
)
def transient_time(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 101,
    final_time: float = 20.0,
    chunks: int = 40,
    tol: float | None = None,
    settled: Callable[[np.ndarray], np.ndarray] | None = None,
    **integrate_kwargs: Any,
) -> Geometry:
    """Paint the transient-time field — how long the approach to the attractor takes.

    Escape time asks when an orbit **leaves** a set; this asks when it
    **arrives**.  The default arrival test is that the flow has slowed to a stop
    (``|f(u)| < tol``), which is the right test when the attractors are
    equilibria and the wrong one for a limit cycle, where the speed never becomes
    small.  For those, pass ``settled=`` — a ball around a known attractor, a
    section crossing, whatever "arrived" means for that system.  Which test ran
    is recorded in ``spec.meta`` rather than guessed at silently.

    Parameters
    ----------
    system : ContinuousSystem
    plane, at, xlim, ylim, grid, final_time, chunks
        As in :func:`escape_time`.
    tol : float, optional
        The speed below which the orbit counts as settled.  ``None`` (the
        default) takes 1% of the median speed over the drawn lattice — a flow
        has no universal speed scale — and records the number in ``spec.meta``.
    settled : callable, optional
        An explicit arrival test over the ``(n, dim)`` states.  Overrides ``tol``.
    **integrate_kwargs
        Forwarded to the engine ensemble.

    Returns
    -------
    Geometry
    """
    return _field_geometry(
        "transient_time",
        _planar.transient_time_field(
            system,
            plane=plane,
            at=at,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            final_time=final_time,
            chunks=chunks,
            tol=tol,
            settled=settled,
            **integrate_kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# invariant_density
# ---------------------------------------------------------------------------


def _series(
    subject: Any, components: Sequence[int | str] | None
) -> tuple[list[np.ndarray], list[str]]:
    """Pull the requested component series (and their labels) off any subject.

    Accepts a :class:`~tsdynamics.data.Trajectory`, a system (which is integrated
    — a ``data`` transform accepts a model because a model gives you data for
    free), or a bare array.
    """
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    if hasattr(subject, "y") and hasattr(subject, "t"):
        values = np.asarray(subject.y, dtype=float)
        system = getattr(subject, "system", None)
        names: tuple[str, ...] | None = getattr(system, "variables", None)
    elif hasattr(subject, "run"):
        raise InvalidInputError(
            "invariant_density takes samples: integrate or iterate the system first "
            "(the transform will not silently pick a run length, a transient or a seed "
            "for you), then pass the trajectory."
        )
    else:
        values = np.atleast_2d(np.asarray(subject, dtype=float))
        if values.shape[0] == 1 and values.shape[1] > 1:
            values = values.T
        names = None

    if values.ndim != 2:
        raise InvalidInputError(
            f"invariant_density needs a (n_samples, n_components) array, got shape {values.shape}."
        )
    dim = values.shape[1]
    wanted: list[int]
    if components is None:
        wanted = [0]
    else:
        wanted = []
        for item in components:
            if isinstance(item, str):
                if not names or item not in names:
                    raise InvalidParameterError(
                        f"component {item!r} is not one of "
                        f"{list(names) if names else 'the data declares no variable names'}."
                    )
                wanted.append(names.index(item))
            else:
                idx = int(item)
                if not -dim <= idx < dim:
                    raise InvalidParameterError(
                        f"component {idx} is out of range for {dim} components."
                    )
                wanted.append(idx % dim)
    if len(wanted) not in (1, 2):
        raise InvalidParameterError(
            f"invariant_density draws the measure of one component (a density curve) or two "
            f"(an image); {len(wanted)} components were selected."
        )
    labels = [names[k] if names else f"x{k}" for k in wanted]
    return [values[:, k] for k in wanted], labels


@plot_transform(
    name="invariant_density",
    source="data",
    # Two genuinely different pictures: one component is a density CURVE (a
    # diagnostic in ``scaling``), two are the measure on a plane (``state2``).
    # It used to declare ``state2`` with ``ndim=(1, 2)`` — a two-axis space with
    # one axis — and that malformation was the root cause of an overlay refusal
    # naming ``state2`` on both sides of a "these differ" message.
    frame=(FrameSpace.SCALING, FrameSpace.STATE2),
    role=OverlayRole.BASE,
    default_primitive="histogram",
    primitives=("histogram", "line", "steps", "image", "contour", "surface3d"),
    presentation=Presentation(cmap="magma"),
    analysis="tsdynamics.analysis.planar.invariant_density",
    example=lambda primitive: (
        _demo_orbit(),
        {"components": ("x", "y"), "bins": 24}
        if primitive in ("image", "contour", "surface3d")
        else {"bins": 24},
    ),
    doc="The natural measure of an orbit: a density curve, or a 2-D measure image.",
)
def invariant_density(
    subject: Any,
    *,
    components: Sequence[int | str] | None = None,
    bins: int | tuple[int, int] = 200,
    span: tuple[float, float] | None = None,
) -> Geometry:
    r"""Paint the natural measure of an orbit — the density it spends its time at.

    The invariant density is what an attractor *is*, statistically.  One
    component gives the familiar density curve; two give the measure on that
    projection of the attractor, which is also the right way to draw a very large
    point cloud (ten million orbit points as markers are a black rectangle;
    binned, they are a picture).

    It is one of the few plots in a dynamics library with an **exact** reference:
    the logistic map at :math:`r = 4` has

    .. math::  \rho(x) = \frac{1}{\pi \sqrt{x(1-x)}} ,

    so the drawn curve can be checked against truth rather than against itself.

    Parameters
    ----------
    subject : Trajectory or array-like
        The samples.  A system is refused: choosing the run length, the transient
        and the initial condition changes the answer, so those are the caller's
        to state.
    components : sequence, optional
        One component (a density curve, the default: the first) or two (a 2-D
        measure image), by name or index.
    bins : int or tuple of int, optional
        Bins per axis.  Default ``200``.
    span : tuple, optional
        The range to bin over — ``(lo, hi)`` for one component,
        ``((xlo, xhi), (ylo, yhi))`` for two.  ``None`` uses the data range.

    Returns
    -------
    Geometry
        1-D: channels ``x`` (bin centres) and ``y`` (density), drawable as a
        ``histogram`` / ``line`` / ``steps``.  2-D: ``x`` / ``y`` / ``z`` / ``c``,
        drawable as an ``image`` / ``contour`` / ``surface3d``.  The declared row
        is narrowed per geometry, so asking for an ``image`` of a 1-D density
        raises instead of drawing a one-pixel-tall stripe.
    """
    series, labels = _series(subject, components)
    meta: dict[str, Any] = {"n_samples": int(series[0].size), "bins": bins}

    if len(series) == 1:
        centres, density, _edges = _planar.invariant_density(
            series[0],
            bins=int(bins) if not isinstance(bins, tuple) else int(bins[0]),
            range=_as_pair(span),
        )
        return Geometry(
            "invariant_density",
            make_frame(FrameSpace.SCALING, labels),
            channels={"x": centres, "y": density},
            label=f"density of {labels[0]}",
            axis_labels=(labels[0], "density"),
            kind=PlotKind.DIAGNOSTIC_CURVE,
            chosen_primitive="histogram",
            primitives=("histogram", "line", "steps"),
            aspect="auto",
            meta=meta,
        )

    xs, ys, density2d = _planar.invariant_density_2d(
        series[0], series[1], bins=bins, range=_as_pairs(span)
    )
    return Geometry(
        "invariant_density",
        make_frame(FrameSpace.STATE2, labels),
        channels={"x": xs, "y": ys, "z": density2d, "c": density2d.ravel()},
        axis_labels=(labels[0], labels[1]),
        kind=PlotKind.PHASE_PORTRAIT_2D,
        chosen_primitive="image",
        primitives=("image", "contour", "surface3d"),
        aspect="equal",
        color_label="density",
        meta=meta,
    )


def _as_pair(span: Any) -> tuple[float, float] | None:
    """Coerce a 1-D ``span=`` declaration to ``(lo, hi)``."""
    from tsdynamics.errors import InvalidParameterError

    if span is None:
        return None
    values = list(span)
    if len(values) != 2 or any(isinstance(v, (list, tuple)) for v in values):
        raise InvalidParameterError(f"span= for a one-component density is (lo, hi), got {span!r}.")
    return float(values[0]), float(values[1])


def _as_pairs(span: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Coerce a 2-D ``span=`` declaration to ``((xlo, xhi), (ylo, yhi))``."""
    from tsdynamics.errors import InvalidParameterError

    if span is None:
        return None
    values = list(span)
    if len(values) != 2 or not all(
        isinstance(v, (list, tuple, np.ndarray)) and len(list(v)) == 2 for v in values
    ):
        raise InvalidParameterError(
            f"span= for a two-component density is ((xlo, xhi), (ylo, yhi)), got {span!r}."
        )
    first, second = (list(v) for v in values)
    return (float(first[0]), float(first[1])), (float(second[0]), float(second[1]))
