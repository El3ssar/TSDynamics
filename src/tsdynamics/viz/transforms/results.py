"""Transforms for the things an *analysis* produces — and for a batch of orbits.

Four transforms whose subject is a measured structure rather than a raw orbit:

===================  ===========  =====================================================
transform            source       what it draws
===================  ===========  =====================================================
``recurrence``       ``data``     the recurrence plot R(i, j) of a trajectory
``orbit_diagram``    ``data``     the asymptotic orbit against a swept parameter
``basins``           ``model``    the basin label image over a 2-D region
``ensemble_fan``     ``data``     an ensemble's median with its spread as a band
===================  ===========  =====================================================

They exist because *plotting a result* and *composing a result with something
else* were different capabilities before v6: every one of these had a result
class that could draw itself, and none had a transform — so none could be
overlaid, gridded, restyled by name, or asked for a different primitive.  Each is
one registration, adapting an estimator that already exists in
:mod:`tsdynamics.analysis`; none owns any new math.

They also **claim the four primitives that no row claimed**: ``boundary``
(``basins``), ``band`` (``ensemble_fan``), plus the ``bars`` / ``errorbars`` rows
the spectrum and scaling transforms carry.  A primitive nobody can reach is a
drawing the library cannot do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._frames import FrameSpace, OverlayRole
from ..spec import PlotKind
from ._base import Geometry, Part, Presentation, make_frame
from ._registry import register

__all__ = ["basins", "ensemble_fan", "orbit_diagram", "recurrence"]


def _demo_orbit(n: int = 160) -> Any:
    """Return a short deterministic orbit for the gate's examples."""
    from tsdynamics.systems.continuous.chaotic_attractors import Rossler

    return Rossler().run(final_time=20.0, dt=20.0 / n, ic=[1.0, 1.0, 1.0])


def _demo_map() -> Any:
    """Return the logistic map — the orbit-diagram example everyone recognises."""
    from tsdynamics.systems.discrete.population_maps import Logistic

    return Logistic()


def _demo_bistable() -> Any:
    """Return a small 2-D map with a basin structure, for the basins example."""
    from tsdynamics.systems.discrete.chaotic_maps import Henon

    return Henon()


# ---------------------------------------------------------------------------
# recurrence
# ---------------------------------------------------------------------------


@register(
    source="data",
    frame=FrameSpace.GRID2,
    kind=PlotKind.RECURRENCE_PLOT,
    primitives=("image", "contour"),
    role=OverlayRole.FIELD,
    labels=("i", "j"),
    presentation=Presentation(aspect="equal", cmap="Greys"),
    analysis="tsdynamics.analysis.recurrence_matrix",
    example=lambda primitive: (_demo_orbit(80), {"recurrence_rate": 0.05}),
    doc="The recurrence plot R(i, j) — where the orbit revisits its own neighbourhood.",
)
def recurrence(subject: Any, *, recurrence_rate: float = 0.05, **kwargs: Any) -> Any:
    """Draw the recurrence plot of a trajectory.

    A phase-space method, not a series statistic: ``R(i, j) = 1`` when the orbit
    at time ``j`` is within ``epsilon`` of where it was at time ``i``, so the
    diagonals are recurrences and their lengths are what DET and L_max measure.

    There was no route from a bare trajectory to this picture before v6 — the
    estimator's *result* could draw itself, but a transform is what makes it
    composable (``ts.plot(traj, "recurrence")``, a grid of them across a
    parameter, ``primitive="contour"`` for the level sets of a thresholded
    distance field).

    Parameters
    ----------
    subject : Trajectory or array
        The orbit.
    recurrence_rate : float, optional
        Target fraction of recurrent pairs; ``epsilon`` is calibrated to it.
    **kwargs
        Forwarded to :func:`tsdynamics.analysis.recurrence_matrix` (``epsilon``,
        ``metric``, ``theiler``).

    Returns
    -------
    Geometry
    """
    from tsdynamics.analysis import recurrence_matrix

    result = recurrence_matrix(subject, recurrence_rate=recurrence_rate, **kwargs)
    dense = np.asarray(result.matrix.todense(), dtype=float)
    index = np.arange(dense.shape[0], dtype=float)
    return Geometry(
        "recurrence",
        make_frame(FrameSpace.GRID2, ("i", "j")),
        channels={"x": index, "y": index, "z": dense, "c": dense.ravel()},
        axis_labels=("i", "j"),
        color_label="recurrence",
        meta={"epsilon": float(result.epsilon), "recurrence_rate": float(result.recurrence_rate)},
    )


# ---------------------------------------------------------------------------
# orbit_diagram
# ---------------------------------------------------------------------------


@register(
    source="data",
    frame=FrameSpace.PARAM1,
    kind=PlotKind.ORBIT_DIAGRAM,
    primitives=("points", "density"),
    example=lambda primitive: (_demo_map(), {"param": "r", "values": (3.4, 4.0, 60), "points": 40}),
    doc="The asymptotic orbit against a swept parameter — the bifurcation cascade.",
)
def orbit_diagram(
    subject: Any,
    *,
    param: str = "r",
    values: Sequence[float] | tuple[float, float, int] | None = None,
    points: int = 100,
    transient: int = 500,
    component: int = 0,
    **kwargs: Any,
) -> Geometry:
    """Sweep a parameter and draw the asymptotic orbit at each value.

    The picture the library is *for*, and it had no transform: the result class
    could draw itself, so a cascade could not be overlaid on anything, gridded
    beside anything, or drawn as a density (which is what a three-million-point
    diagram needs — as markers it is a black rectangle).

    Parameters
    ----------
    subject : DiscreteMap, PoincareMap or StroboscopicMap
        Any discrete view.  A flow wrapped in ``sys.poincare(...)`` gives the
        bifurcation diagram of the flow.
    param : str, optional
        The parameter to sweep.
    values : sequence or (lo, hi, n), optional
        The values.  A three-tuple is read as ``linspace(lo, hi, n)``; ``None``
        sweeps the declared parameter's neighbourhood.
    points : int, optional
        Asymptotic points recorded per value.
    transient : int, optional
        Iterations discarded before recording.
    component : int, optional
        Which state component to record.
    **kwargs
        Forwarded to :func:`tsdynamics.analysis.orbit_diagram`.

    Returns
    -------
    Geometry
    """
    from tsdynamics.analysis import orbit_diagram as _orbit_diagram

    if values is None:
        current = float(subject.params[param])
        swept = np.linspace(0.5 * current, 1.5 * current, 100)
    elif len(values) == 3 and isinstance(values[2], (int, np.integer)):
        swept = np.linspace(float(values[0]), float(values[1]), int(values[2]))
    else:
        swept = np.asarray(values, dtype=float)
    result = _orbit_diagram(
        subject, param, swept, points_per_value=points, transient=transient, **kwargs
    )
    orbit = np.asarray(result.points, dtype=float)[:, :, component]
    xs = np.repeat(swept, orbit.shape[1])
    ys = orbit.ravel()
    finite = np.isfinite(xs) & np.isfinite(ys)
    return Geometry(
        "orbit_diagram",
        make_frame(FrameSpace.PARAM1, (param,)),
        channels={"x": xs[finite], "y": ys[finite]},
        axis_labels=(param, f"x{component}"),
        style={"markersize": 0.5, "alpha": 0.5},
        meta={"param": param, "points_per_value": int(points), "transient": int(transient)},
    )


# ---------------------------------------------------------------------------
# basins
# ---------------------------------------------------------------------------


@register(
    source="model",
    frame=FrameSpace.STATE2,
    kind=PlotKind.BASINS_IMAGE,
    primitives=("image", "boundary", "contour"),
    role=OverlayRole.FIELD,
    presentation=Presentation(aspect="equal", cmap="tab10", discrete=True),
    analysis="tsdynamics.analysis.basins",
    example=lambda primitive: (_demo_bistable(), {"region": ((-2.0, 2.0, 16), (-2.0, 2.0, 16))}),
    doc="Which attractor each initial condition reaches — the basin label image.",
)
def basins(subject: Any, *, region: Any = None, **kwargs: Any) -> Geometry:
    """Paint the basins of attraction over a 2-D region.

    ``boundary`` is the primitive that makes this row worth having: category
    labels have no meaningful intermediate value, so the *edge* of a basin is
    where a cell differs from its neighbour, not where an interpolant crosses a
    threshold — which is why a basin boundary overlays cleanly on an FTLE field
    or a portrait, and a filled image does not.

    Parameters
    ----------
    subject : system
        The map or flow.
    region : sequence, optional
        One ``(lo, hi[, n])`` pair per state component — the same plain-Python
        region every door in the library reads.
    **kwargs
        Forwarded to :func:`tsdynamics.analysis.basins`.

    Returns
    -------
    Geometry
    """
    from tsdynamics.analysis.basins.basins import basins as _basins

    if region is None:
        raise _needs_region(subject)
    result = _basins(subject, region, **kwargs)
    labels = np.asarray(result.labels, dtype=float)
    if result.grid is None:  # pragma: no cover - defensive
        raise _needs_region(subject)
    axes = [np.asarray(a, dtype=float) for a in result.grid.axes()][:2]
    names = tuple(result.meta.get("variables") or ("x", "y"))[:2]
    return Geometry(
        "basins",
        make_frame(FrameSpace.STATE2, names),
        channels={"x": axes[0], "y": axes[1], "z": labels, "c": labels.ravel()},
        axis_labels=names,
        axis_limits=(
            (float(axes[0][0]), float(axes[0][-1])),
            (float(axes[1][0]), float(axes[1][-1])),
        ),
        color_label="attractor",
        meta=dict(result.meta),
    )


def _needs_region(subject: Any) -> Exception:
    """Build the error a basin plot raises when it was not told where to look."""
    from tsdynamics.errors import InvalidParameterError

    return InvalidParameterError(
        "basins needs the region to paint: one (lo, hi, n) pair per state component, e.g. "
        f"ts.plot({type(subject).__name__.lower()}, 'basins', "
        "region=[(-2, 2, 60), (-2, 2, 60)])."
    )


# ---------------------------------------------------------------------------
# ensemble_fan
# ---------------------------------------------------------------------------


@register(
    source="data",
    frame=FrameSpace.TIME,
    kind=PlotKind.ENSEMBLE_FAN,
    primitives=("band",),
    labels=("t", "x"),
    presentation=Presentation(legend=True),
    example=lambda primitive: (_demo_batch(), {}),
    doc="An ensemble's median with its spread as a shaded envelope.",
)
def ensemble_fan(
    subject: Any, *, component: int = 0, spread: tuple[float, float] = (10.0, 90.0)
) -> Geometry:
    """Draw the spread of an ensemble of trajectories as a band around its median.

    The right picture for a stochastic system or a spread of initial conditions:
    fifty overplotted curves are a smear, and their median plus a percentile
    envelope is the statement the smear was trying to make.  It is also what
    finally gives :data:`~tsdynamics.viz.spec.PlotKind.ENSEMBLE_FAN` a producer —
    the mark existed in the frozen vocabulary with nothing able to emit it.

    Parameters
    ----------
    subject : TrajectoryBatch, sequence of Trajectory, or (n, T) array
        The ensemble.
    component : int, optional
        Which state component to summarise.
    spread : tuple of float, optional
        The percentile envelope.  Default the 10th-90th.

    Returns
    -------
    Geometry
    """
    t, band = _ensemble_array(subject, component)
    lo, hi = (np.percentile(band, p, axis=0) for p in spread)
    median = np.median(band, axis=0)
    return Geometry(
        "ensemble_fan",
        make_frame(FrameSpace.TIME, ("t",)),
        parts=[
            # The envelope carries no ``y``: a band that also names a centre line
            # is drawn twice and legended twice, which reads as two ensembles.
            Part(
                {"x": t, "lo": lo, "hi": hi},
                label=f"{spread[0]:g}-{spread[1]:g}%",
                style={"alpha": 0.25},
                primitive="band",
            ),
            Part({"x": t, "y": median}, label="median", primitive="line"),
        ],
        axis_labels=("t", f"x{component}"),
        legend=True,
        meta={"members": int(band.shape[0]), "spread": tuple(float(p) for p in spread)},
    )


def _ensemble_array(subject: Any, component: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, values)`` — ``values`` is ``(members, T)`` — from any ensemble shape."""
    from tsdynamics.errors import InvalidInputError

    members = list(getattr(subject, "trajectories", None) or subject)
    if members and hasattr(members[0], "y"):
        t = np.asarray(members[0].t, dtype=float)
        rows = [np.atleast_2d(np.asarray(m.y, dtype=float))[:, component] for m in members]
        return t, np.asarray(rows, dtype=float)
    arr = np.asarray(subject, dtype=float)
    if arr.ndim != 2:
        raise InvalidInputError(
            "ensemble_fan needs an ensemble: a TrajectoryBatch, a list of trajectories, or an "
            f"(members, T) array; got shape {arr.shape}."
        )
    return np.arange(arr.shape[1], dtype=float), arr


def _demo_batch() -> Any:
    """Return a tiny deterministic ensemble for the gate's example."""
    t = np.linspace(0.0, 6.0, 60)
    rng = np.random.default_rng(0)
    return np.array([np.sin(t) + 0.2 * rng.standard_normal(t.size) for _ in range(12)])
