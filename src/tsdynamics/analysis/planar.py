r"""Planar-flow estimators — the numerics behind the model-only plots.

Everything in this module answers one question: *what does the vector field do
at points the trajectory never visited?*  That is the line between a **data**
plot and a **model** plot, and it is the reason these estimators live here and
not in :mod:`tsdynamics.viz` — a transform is a thin adapter and owns no new
math.

Seven quantities, all defined on a 2-D **slice** of state space:

============================  ==============================================
:func:`flow_field`            the right-hand side sampled on a lattice
:func:`nullclines`            the curves :math:`f_i = 0` (marching squares)
:func:`streamlines`           integral curves of the sliced field
:func:`trace_determinant`     equilibria placed on the :math:`(\tau, \Delta)` plane
:func:`ftle_field`            finite-time Lyapunov exponents; ridges are LCS
:func:`escape_time_field`     first time the orbit leaves the window
:func:`transient_time_field`  first time the orbit settles
============================  ==============================================

plus :func:`invariant_density` / :func:`invariant_density_2d`, which are pure
data (a histogram of an orbit) and are here because the natural measure is a
phase-space object, not a series statistic.

The slice
---------
A planar flow is drawn on its own two coordinates and nothing is lost.  A
higher-dimensional flow is drawn on a **frozen slice**: two coordinates vary
over the window and the rest are held at ``at``.  Everything this module
computes on such a slice is a property of *that slice*, not of the full flow —
an FTLE field of a Lorenz ``(x, z)`` slice is the stretching of the frozen
2-D field, and the honest statement is on every function that takes ``at``.
The one exception is noted where it matters: :func:`escape_time_field` and
:func:`transient_time_field` integrate the **full** system from initial
conditions that lie in the slice, so they are properties of the real flow.

Not registered as analyses
--------------------------
None of these self-registers into ``registry.analyses``: like
:mod:`tsdynamics.analysis.sampling`, they are tools a plot needs rather than
quantifiers a user asks for by name.  They are plain functions returning plain
records, directly testable without a plotting backend.

References
----------
Strogatz, S. H. (1994). *Nonlinear Dynamics and Chaos*.  Addison-Wesley.
    Nullclines, the trace-determinant plane, and the phase-plane method.
Haller, G. (2015). Lagrangian coherent structures. *Annu. Rev. Fluid Mech.*
    **47**, 137-162.  doi:10.1146/annurev-fluid-010313-141322.  FTLE ridges.
Shadden, S. C., Lekien, F. & Marsden, J. E. (2005). Definition and properties
    of Lagrangian coherent structures.  *Physica D* **212**, 271-304.
    doi:10.1016/j.physd.2005.10.007.  The Cauchy-Green / FTLE definition used
    in :func:`ftle_field`.
Lai, Y.-C. & Tel, T. (2011). *Transient Chaos*.  Springer.
    doi:10.1007/978-1-4419-6987-3.  Escape-time and transient-time fields.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from tsdynamics.errors import InvalidInputError, InvalidParameterError, remedy

from ._common import reject_system
from ._result_json import _sig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.families import ContinuousSystem

__all__ = [
    "FlowField",
    "Nullcline",
    "ScalarField",
    "TraceDeterminant",
    "classify_linear",
    "escape_time_field",
    "flow_field",
    "ftle_field",
    "invariant_density",
    "invariant_density_2d",
    "nullclines",
    "resolve_plane",
    "streamlines",
    "trace_determinant",
    "transient_time_field",
    "window_for",
]


def __dir__() -> list[str]:
    """Show the planar toolkit, not this module's imports.

    ``planar`` is advertised as a capability category on ``ts.analysis.<TAB>``,
    so it is a namespace a user tab-completes into.  Without this, a plain module
    listing hands them ``np`` / ``warnings`` / ``dataclass`` / ``field`` /
    ``Any`` / ``Callable`` / ``Sequence`` and the private grid helpers alongside
    the actual estimators — the same clutter the curated packages exist to
    remove, one dot deeper.  Everything dropped stays reachable by name.
    """
    return sorted(__all__)


#: A polyline: ``(n, 2)`` of ``(x, y)`` points in the plane of the slice.
#:
#: It is an **alias of** :class:`numpy.ndarray`, not a class — nothing is ever an
#: instance of ``Curve`` and ``Curve`` resolves from no public home, so the public
#: signatures below spell ``numpy.ndarray`` outright rather than advertising a
#: name a reader of ``help()`` cannot look up.  The alias stays as documentation
#: of the *shape* convention for in-module readers.
Curve = np.ndarray


# ---------------------------------------------------------------------------
# The slice: which two coordinates, held where, over what window
# ---------------------------------------------------------------------------


def _require_flow(system: Any, what: str) -> ContinuousSystem:
    """Return ``system`` if it is a flow with a symbolic right-hand side, else raise.

    The greppable model-transform rule made concrete: these estimators call
    ``_rhs_numeric`` at points that are not in any input array, so a bare
    :class:`~tsdynamics.data.Trajectory` — or a discrete map, which has no
    vector field at all — cannot serve them.
    """
    if not hasattr(system, "_rhs_numeric"):
        # Every other wrong-subject door in the analysis layer ends with a line
        # the reader can run; these eight did not, so "pass the system itself"
        # was the whole of the advice.  A Trajectory knows its own system, and a
        # map genuinely has no vector field — two different situations, two
        # different lines.
        held = type(system).__name__
        if getattr(system, "system", None) is not None:
            fix = remedy(f"ts.analysis.{what}(traj.system)")
        elif getattr(system, "family", None) == "map":
            fix = remedy(
                "ts.analysis.find(system)",
                lead=(
                    "A map advances by iteration and has no vector field, so there is "
                    "nothing here to evaluate. What a map DOES answer:"
                ),
            )
        else:
            fix = remedy(f"ts.analysis.{what}(system)")
        raise InvalidInputError(
            f"{what} needs a continuous system (it evaluates the right-hand side at points "
            f"that are not in any trajectory), but got {held}." + fix
        )
    return system  # type: ignore[no-any-return]


def resolve_plane(
    system: Any, plane: Sequence[int | str] = (0, 1)
) -> tuple[int, int, tuple[str, str]]:
    """Resolve a ``plane=`` declaration to ``(i, j, (label_i, label_j))``.

    The friendly spelling of every model transform, matching the Poincaré
    section API: name the two coordinates (``("x", "z")``) or index them
    (``(0, 2)``).  Names are resolved against the system's ``variables``.

    Parameters
    ----------
    system : ContinuousSystem
        The system whose coordinates are being named.
    plane : sequence of int or str, optional
        The two coordinates spanning the slice.  Default ``(0, 1)``.

    Returns
    -------
    tuple
        ``(i, j, (label_i, label_j))`` — the two indices and their display
        labels (the declared variable names when the system has them).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``plane`` is not a pair, names a coordinate the system does not
        declare, indexes out of range, or names the same coordinate twice.
    """
    _require_flow(system, "resolve_plane")
    items = list(plane)
    if len(items) != 2:
        raise InvalidParameterError(
            f"plane= names the two coordinates of the slice, so it needs exactly 2 entries, "
            f"got {len(items)}: {items!r}."
        )
    names: tuple[str, ...] | None = getattr(type(system), "variables", None)
    dim = int(getattr(system, "dim", 2))
    # ``plane=`` is TWO WORDS in this library: here it is the pair of *view axes*
    # of a slice, and at ``poincare_section`` / ``PoincareMap`` it is a cutting
    # *section* ``(axis, offset[, direction])``.  A section spelling reaching this
    # door used to be read as axes and draw a confident, wrong picture — measured,
    # ``flow_field(lorenz, plane=("y", 0.0))`` returned a FlowField of the x-y
    # plane with no complaint.  An offset is a *value*, so it is detectable: a
    # non-integral number, or a third entry, can only be a section.
    for item in items:
        # A float can only be an OFFSET: a coordinate is named or indexed, and an
        # index is an int.  Checking the type rather than the value is what makes
        # ``("y", 0.0)`` — the commonest section spelling, and an exactly integral
        # number — detectable at all.
        if isinstance(item, (float, np.floating)):
            raise InvalidParameterError(
                f"plane={tuple(items)!r} looks like a Poincaré SECTION — (axis, offset) — "
                f"but here plane= names the two view AXES of the slice, so {item!r} would "
                f"have to be a coordinate index (an int) or a coordinate name.\n"
                f"    plane=(0, 1)                       # the first two coordinates\n"
                f'    plane=("x", "z")                   # ...by name\n'
                f"    ts.analysis.poincare_section(system, {tuple(items)!r})   # a section"
            )
    out: list[int] = []
    for item in items:
        if isinstance(item, str):
            if not names or item not in names:
                raise InvalidParameterError(
                    f"plane component {item!r} is not one of "
                    f"{list(names) if names else 'the system declares no variables'}; "
                    "use an index instead."
                )
            out.append(names.index(item))
        else:
            idx = int(item)
            if not -dim <= idx < dim:
                raise InvalidParameterError(
                    f"plane component {idx} is out of range for a {dim}-dimensional system."
                )
            out.append(idx % dim)
    if out[0] == out[1]:
        raise InvalidParameterError(
            f"plane={items!r} names the same coordinate twice; a slice needs two distinct axes."
        )
    labels = tuple(names[k] if names else f"x{k}" for k in out)
    return out[0], out[1], (labels[0], labels[1])


def _base_state(system: Any, at: Any | None) -> np.ndarray:
    """Return the state the off-plane coordinates are frozen at.

    ``at=None`` uses the system's declared ``default_ic`` when it has one and
    the origin otherwise — **never** a random draw, because a plot whose picture
    changes between two identical calls is not a plot of anything.
    """
    dim = int(getattr(system, "dim", 2))
    if at is not None:
        arr = np.asarray(at, dtype=float).ravel()
        if arr.size != dim:
            raise InvalidParameterError(
                f"at= is the state the off-plane coordinates are held at, so it needs "
                f"{dim} entries, got {arr.size}."
            )
        return arr
    declared = getattr(type(system), "_default_ic", None)
    if declared is None:
        return np.zeros(dim, dtype=float)
    return np.asarray(declared, dtype=float).reshape(dim)


def window_for(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    pad: float = 0.25,
    pilot_time: float = 40.0,
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    """Choose the ``(xlim, ylim)`` window to draw a slice over, and say how.

    Auto-choosing a region is the one place a model plot can be *confidently
    wrong* — a pilot-orbit bounding box on a two-well system collapses onto one
    well — so the choice is (a) always overridable, (b) always recorded.  The
    returned mapping goes straight into ``spec.meta`` and is surfaced in the
    repr, which is what makes an auto-chosen window auditable instead of
    invisible.

    The heuristic, in order: an explicit limit always wins; otherwise the
    padded bounding box of a short pilot orbit from the system's declared
    initial condition, **unioned with the base state and the origin** so an
    orbit that lives in one corner does not hide the structure at the centre;
    and if the pilot orbit diverges or the system declares nothing,
    ``(-2, 2)``.

    Parameters
    ----------
    system : ContinuousSystem
    plane : sequence of int or str, optional
        The two coordinates of the slice.
    at : array-like, optional
        The frozen base state (see :func:`resolve_plane`).
    xlim, ylim : tuple of float, optional
        Explicit limits.  Either may be given alone.
    pad : float, optional
        Fractional padding around the pilot box.  Default ``0.25``.
    pilot_time : float, optional
        Integration time of the pilot orbit.  Default ``40``.

    Returns
    -------
    tuple
        ``(xlim, ylim, meta)``.
    """
    _require_flow(system, "window_for")
    i, j, _ = resolve_plane(system, plane)
    base = _base_state(system, at)
    meta: dict[str, Any] = {
        "window_source": "explicit",
        "plane": (i, j),
        "at": base.tolist(),
    }
    if xlim is not None and ylim is not None:
        out_x = (float(xlim[0]), float(xlim[1]))
        out_y = (float(ylim[0]), float(ylim[1]))
        meta["xlim"], meta["ylim"] = out_x, out_y
        return out_x, out_y, meta

    box: np.ndarray | None = None
    if getattr(type(system), "_default_ic", None) is not None:
        try:
            traj = system.run(final_time=float(pilot_time), dt=float(pilot_time) / 400.0)
            pts = np.asarray(traj.y, dtype=float)
            if np.isfinite(pts).all():
                box = np.array(
                    [
                        [pts[:, i].min(), pts[:, i].max()],
                        [pts[:, j].min(), pts[:, j].max()],
                    ]
                )
        except Exception:  # noqa: BLE001 - a diverging pilot is a fallback, not a failure
            box = None

    if box is None:
        auto_x = (-2.0, 2.0)
        auto_y = (-2.0, 2.0)
        meta["window_source"] = "default box (no pilot orbit available)"
    else:
        anchors = np.array([[base[i], base[j]], [0.0, 0.0]])
        lo = np.minimum(box[:, 0], anchors.min(axis=0))
        hi = np.maximum(box[:, 1], anchors.max(axis=0))
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        lo, hi = lo - pad * span, hi + pad * span
        auto_x = (float(lo[0]), float(hi[0]))
        auto_y = (float(lo[1]), float(hi[1]))
        meta["window_source"] = (
            f"pilot orbit (T={float(pilot_time):g}) union the base state and the origin, "
            f"padded {pad:.0%}"
        )
    out_x = (float(xlim[0]), float(xlim[1])) if xlim is not None else auto_x
    out_y = (float(ylim[0]), float(ylim[1])) if ylim is not None else auto_y
    if xlim is not None or ylim is not None:
        meta["window_source"] = f"partly explicit; the rest from {meta['window_source']}"
    meta["xlim"] = out_x
    meta["ylim"] = out_y
    return out_x, out_y, meta


def _sliced_rhs(
    system: Any, i: int, j: int, base: np.ndarray
) -> Callable[[float, float], tuple[float, float]]:
    """Return ``g(x, y) -> (f_i, f_j)``, the right-hand side restricted to the slice."""
    rhs = system._rhs_numeric()
    state = np.array(base, dtype=float, copy=True)

    def g(x: float, y: float) -> tuple[float, float]:
        state[i] = x
        state[j] = y
        out = np.asarray(rhs(state, 0.0), dtype=float)
        return float(out[i]), float(out[j])

    return g


def _grid_axes(
    xlim: tuple[float, float], ylim: tuple[float, float], grid: int | tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(xs, ys)`` sample coordinates of a lattice over the window."""
    nx, ny = (grid, grid) if isinstance(grid, (int, np.integer)) else (int(grid[0]), int(grid[1]))
    if nx < 2 or ny < 2:
        raise InvalidParameterError(f"grid= needs at least 2 samples per axis, got {(nx, ny)}.")
    return (
        np.linspace(xlim[0], xlim[1], nx),
        np.linspace(ylim[0], ylim[1], ny),
    )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowField:
    """The right-hand side sampled on a lattice over a 2-D slice.

    Attributes
    ----------
    xs, ys : ndarray
        The sample coordinates, shapes ``(nx,)`` and ``(ny,)``.
    u, v : ndarray
        The in-plane components of the right-hand side, shape ``(ny, nx)``
        (row-major, so ``u[r, c]`` sits at ``(xs[c], ys[r])`` — the layout an
        image expects).
    speed : ndarray
        ``hypot(u, v)`` **before** any normalization, shape ``(ny, nx)``.  Kept
        separately so a direction field (unit arrows) can still colour by, or be
        drawn over, the true magnitude.
    labels : tuple of str
        The two coordinate labels.
    meta : dict
        Provenance: the plane, the base state, the window and how it was chosen.
    """

    xs: np.ndarray
    ys: np.ndarray
    u: np.ndarray
    v: np.ndarray
    speed: np.ndarray
    labels: tuple[str, str]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Nullcline:
    r"""The zero level set of one right-hand-side component on a slice.

    Attributes
    ----------
    index : int
        Which component of the right-hand side vanishes on these curves.
    label : str
        The coordinate that component is the derivative of (``"x"`` for
        :math:`\\dot x = 0`).
    curves : list of ndarray
        The polylines, each ``(n, 2)``.  A nullcline is generally several
        disconnected branches, and joining them would draw a curve that is not
        in the field.
    """

    index: int
    label: str
    curves: list[np.ndarray]

    def __len__(self) -> int:
        """Return the number of disconnected branches."""
        return len(self.curves)


@dataclass(frozen=True)
class ScalarField:
    """A scalar quantity sampled on a lattice over a 2-D slice.

    The shared return type of :func:`ftle_field`, :func:`escape_time_field` and
    :func:`transient_time_field` — three different reductions of the same
    ensemble march, so they return the same shape of answer.

    Attributes
    ----------
    xs, ys : ndarray
        The sample coordinates, shapes ``(nx,)`` and ``(ny,)``.
    values : ndarray
        The field, shape ``(ny, nx)``.  ``NaN`` marks a sample the reduction is
        not defined at (an orbit that never escaped, a diverged integration).
    label : str
        What the numbers are, for the colorbar.
    labels : tuple of str
        The two coordinate labels.
    meta : dict
        Provenance, including every auto-chosen default.
    """

    xs: np.ndarray
    ys: np.ndarray
    values: np.ndarray
    label: str
    labels: tuple[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Render the ANSWER, like every other result in the library.

        Seven of the fifty registered analyses return one of these, and it
        carried the default dataclass repr — so printing it dumped two
        coordinate arrays and the whole value matrix into the terminal, in a
        library where every other result is a crafted one-line readout.
        """
        ny, nx = (int(n) for n in np.asarray(self.values).shape[:2])
        xs = np.asarray(self.xs, dtype=float)
        ys = np.asarray(self.ys, dtype=float)
        span = ""
        if xs.size and ys.size:
            span = (
                f"  over {self.labels[0]} ∈ [{_sig(xs.min(), 3)}, {_sig(xs.max(), 3)}], "
                f"{self.labels[1]} ∈ [{_sig(ys.min(), 3)}, {_sig(ys.max(), 3)}]"
            )
        v = np.asarray(self.values, dtype=float)
        blank = float(np.mean(~np.isfinite(v))) * 100.0 if v.size else 0.0
        finite = v[np.isfinite(v)]
        reading = (
            f"{_sig(float(finite.min()), 3)} … {_sig(float(finite.max()), 3)}"
            if finite.size
            else "all NaN"
        )
        head = f"ScalarField  {self.label}  ·  {ny}×{nx} lattice{span}"
        return f"{head}\n    {reading}   ·   {blank:.0f}% not defined (NaN)"

    def __str__(self) -> str:
        """Return the repr — ``print(field)`` and the REPL agree."""
        return repr(self)


@dataclass(frozen=True)
class TraceDeterminant:
    r"""Equilibria placed on the trace-determinant stability plane.

    Attributes
    ----------
    trace, determinant : ndarray
        One entry per equilibrium — the trace and determinant of the Jacobian
        there, shape ``(n,)``.
    classes : list of str
        The linear classification of each equilibrium: ``"saddle"``,
        ``"stable node"``, ``"unstable node"``, ``"stable focus"``,
        ``"unstable focus"``, ``"centre"`` or ``"degenerate"``.
    points : ndarray
        The equilibria themselves, shape ``(n, dim)`` — so a caller can find
        which point a marker belongs to.
    parabola : ndarray
        The discriminant curve :math:`\\Delta = \\tau^2 / 4` sampled over the
        drawn trace range, shape ``(m, 2)``.  Above it the eigenvalues are
        complex (spirals), below it real (nodes and saddles).
    trace_range : tuple of float
        The trace range the parabola was sampled over.
    meta : dict
    """

    trace: np.ndarray
    determinant: np.ndarray
    classes: list[str]
    points: np.ndarray
    parabola: np.ndarray
    trace_range: tuple[float, float]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# flow_field / nullclines / streamlines
# ---------------------------------------------------------------------------


#: The six field analyses that sample a rectangular WINDOW, and therefore accept
#: ``region=`` — the one state-space-box grammar every other door in
#: ``ts.analysis`` already speaks.
def window_from_region(
    region: Any,
    plane: Sequence[int | str],
    system: Any,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    grid: Any,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, Any]:
    """Translate ``region=`` into this module's ``xlim`` / ``ylim`` / ``grid``.

    ``ts.analysis`` had **two** grammars for "this box of state space": the
    region doors (``basins`` / ``fixed_points`` / ``attractors`` /
    ``basin_fractions`` / ``expansion_entropy``) read one ``(lo, hi[, n])`` pair
    per state component, and the eight field analyses read ``plane=`` +
    ``xlim=`` + ``ylim=`` + ``grid=``.  Crossing them produced the one error in
    the library that taught nothing — a bare ``TypeError: takes 1 positional
    argument but 2 were given`` — on the only door that breaks its own grammar.

    The translation is exact: a region carries the same information, indexed by
    state component, so the two plane axes select their pairs out of it.

    Raises
    ------
    InvalidParameterError
        If ``region`` is given together with the window it would set, or if it
        does not carry a pair for each plane axis.
    """
    if region is None:
        return xlim, ylim, grid
    if xlim is not None or ylim is not None:
        raise InvalidParameterError(
            "region= already says where to sample, so xlim=/ylim= would be a second "
            "answer to the same question. Pass one or the other."
            + remedy(
                "ts.analysis.ftle_field(system, region=[(-2.0, 2.0, 60), (-2.0, 2.0, 60)])",
                "ts.analysis.ftle_field(system, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), grid=60)",
            )
        )
    from tsdynamics.data import as_region

    box = as_region(region, dim=getattr(system, "dim", None))
    lo = np.atleast_1d(np.asarray(getattr(box, "lo", ()), dtype=float))
    hi = np.atleast_1d(np.asarray(getattr(box, "hi", ()), dtype=float))
    counts = getattr(box, "shape", None)
    i, j, _ = resolve_plane(system, plane)
    ax = [i, j]
    if max(ax) >= lo.size:
        raise InvalidParameterError(
            f"region= carries {lo.size} bound(s) and the plane {tuple(plane)!r} needs "
            f"component {max(ax)}: pass one (lo, hi[, n]) pair per state component."
            + remedy("ts.analysis.ftle_field(system, region=[(-2.0, 2.0), (-2.0, 2.0)])")
        )
    window_x = (float(lo[ax[0]]), float(hi[ax[0]]))
    window_y = (float(lo[ax[1]]), float(hi[ax[1]]))
    if counts is not None:
        nx, ny = (int(counts[a]) for a in ax)
        grid = nx if nx == ny else (nx, ny)
    return window_x, window_y, grid


def flow_field(
    system: Any,
    region: Any | None = None,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 21,
) -> FlowField:
    """Sample the right-hand side on a lattice over a 2-D slice.

    The one estimator behind both textbook pictures: a **vector field** draws
    ``(u, v)`` at true magnitude, a **direction field** draws them unit-normalized
    (which is what a textbook draws, because a field whose speed varies by three
    orders of magnitude shows one long arrow and a lot of dots).  Normalization
    is a drawing decision, so it is not done here — :attr:`FlowField.speed` keeps
    the magnitudes either way.

    Parameters
    ----------
    system : ContinuousSystem
        The flow.  A trajectory is not enough: this evaluates the right-hand
        side at lattice points no orbit visited.
    plane : sequence of int or str, optional
        The two coordinates spanning the slice.
    at : array-like, optional
        The state the off-plane coordinates are frozen at.
    region : sequence of (lo, hi[, n]), optional
        Where to sample, in the ONE state-space-box grammar every ``region=``
        door in the library reads: one ``(lo, hi)`` bound — or one
        ``(lo, hi, n)`` triple — **per state component**, of which the two
        ``plane`` axes are used.  It is the **second positional** argument, so
        ``ts.analysis.flow_field(system, [(-2, 2, 60), (-2, 2, 60)])`` reads exactly
        like ``ts.analysis.basins(system, ...)``.

        Two grammars for "this box of state space" is one too many: crossing them
        used to produce a bare ``TypeError: takes 1 positional argument but 2
        were given``, the one error in ``ts.analysis`` that taught nothing.
        Equivalent to ``xlim=`` / ``ylim=`` / the lattice size; passing both
        raises.

        .. versionadded:: 6.0
    xlim, ylim : tuple of float, optional
        The window.  ``None`` auto-chooses via :func:`window_for` and records
        how.
    grid : int or tuple of int, optional
        Samples per axis.  Default ``21``.

    Returns
    -------
    FlowField
    """
    _require_flow(system, "flow_field")
    xlim, ylim, grid = window_from_region(region, plane, system, xlim, ylim, grid)
    i, j, labels = resolve_plane(system, plane)
    base = _base_state(system, at)
    xlim, ylim, meta = window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)
    xs, ys = _grid_axes(xlim, ylim, grid)
    g = _sliced_rhs(system, i, j, base)

    u = np.empty((ys.size, xs.size), dtype=float)
    v = np.empty_like(u)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            u[r, c], v[r, c] = g(float(x), float(y))
    meta.update(plane=(i, j), at=base.tolist(), grid=(int(xs.size), int(ys.size)))
    return FlowField(xs, ys, u, v, np.hypot(u, v), labels, meta)


def nullclines(
    system: Any,
    region: Any | None = None,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 201,
    components: Sequence[int | str] | None = None,
) -> list[Nullcline]:
    r"""Find the nullclines of a 2-D slice — the curves :math:`f_i = 0`.

    Marching squares (``contourpy``, a matplotlib hard dependency, so no new
    requirement) on each right-hand-side component sampled over the window.  The
    result is a set of **polylines**, which is why nullclines draw on every
    backend without a new mark: they are lines.

    The intersections of the :math:`\dot x = 0` and :math:`\dot y = 0` curves are
    the equilibria of the slice, which is the strongest cheap correctness check
    available — overlay :func:`~tsdynamics.analysis.fixedpoints.fixed.fixed_points`
    and the markers must sit on the crossings.

    Parameters
    ----------
    system : ContinuousSystem
    plane : sequence of int or str, optional
        The two coordinates spanning the slice.
    at : array-like, optional
        The state the off-plane coordinates are frozen at.
    region : sequence of (lo, hi[, n]), optional
        Where to sample, in the ONE state-space-box grammar every ``region=``
        door in the library reads: one ``(lo, hi)`` bound — or one
        ``(lo, hi, n)`` triple — **per state component**, of which the two
        ``plane`` axes are used.  It is the **second positional** argument, so
        ``ts.analysis.nullclines(system, [(-2, 2, 60), (-2, 2, 60)])`` reads exactly
        like ``ts.analysis.basins(system, ...)``.

        Two grammars for "this box of state space" is one too many: crossing them
        used to produce a bare ``TypeError: takes 1 positional argument but 2
        were given``, the one error in ``ts.analysis`` that taught nothing.
        Equivalent to ``xlim=`` / ``ylim=`` / the lattice size; passing both
        raises.

        .. versionadded:: 6.0
    xlim, ylim : tuple of float, optional
        The window (auto-chosen and recorded when omitted).
    grid : int or tuple of int, optional
        Marching-squares resolution.  Default ``201``; the curve is only as
        smooth as this lattice, and a nullcline with a near-vertical branch (a
        relaxation oscillator) wants more.
    components : sequence of int or str, optional
        Which right-hand-side components to take the zero set of.  ``None``
        (default) takes both in-plane ones — the textbook pair.  On a slice of a
        higher-dimensional flow the off-plane components have zero sets too, and
        naming them draws those.

    Returns
    -------
    list of Nullcline
        One entry per component, in the order requested.

    Notes
    -----
    On a slice of a flow with more than two coordinates these are the nullclines
    of the **frozen** field, not of the full system: the surfaces
    :math:`f_i = 0` cut the slice in these curves, and the crossings of the two
    in-plane ones are equilibria of the slice, not necessarily of the flow.
    """
    from contourpy import contour_generator

    _require_flow(system, "nullclines")
    xlim, ylim, grid = window_from_region(region, plane, system, xlim, ylim, grid)
    i, j, labels = resolve_plane(system, plane)
    base = _base_state(system, at)
    xlim, ylim, _ = window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)
    xs, ys = _grid_axes(xlim, ylim, grid)

    dim = int(getattr(system, "dim", 2))
    wanted: list[int]
    if components is None:
        wanted = [i, j]
    else:
        names: tuple[str, ...] | None = getattr(type(system), "variables", None)
        wanted = []
        for item in components:
            if isinstance(item, str):
                if not names or item not in names:
                    raise InvalidParameterError(
                        f"component {item!r} is not one of "
                        f"{list(names) if names else 'the system declares no variables'}."
                    )
                wanted.append(names.index(item))
            else:
                wanted.append(int(item) % dim)

    rhs = system._rhs_numeric()
    state = np.array(base, dtype=float, copy=True)
    fields = np.empty((dim, ys.size, xs.size), dtype=float)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            state[i] = x
            state[j] = y
            fields[:, r, c] = np.asarray(rhs(state, 0.0), dtype=float)

    gx, gy = np.meshgrid(xs, ys)
    all_labels: tuple[str, ...] | None = getattr(type(system), "variables", None)
    out: list[Nullcline] = []
    for k in wanted:
        gen = contour_generator(gx, gy, fields[k])
        curves = [
            np.asarray(seg, dtype=float)
            for seg in gen.lines(0.0)
            if np.asarray(seg).ndim == 2 and np.asarray(seg).shape[0] >= 2
        ]
        name = all_labels[k] if all_labels else f"x{k}"
        out.append(Nullcline(index=k, label=name, curves=curves))
    return out


def streamlines(
    system: Any,
    region: Any | None = None,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    seeds: int | tuple[int, int] = 8,
    length: float | None = None,
    steps: int = 200,
    both_ways: bool = True,
) -> list[np.ndarray]:
    """Integrate the sliced field from a seed lattice — the integral curves.

    Each streamline is an **arc-length-parametrized** integral curve: the field
    is unit-normalized before stepping (classical fourth-order Runge-Kutta), so
    the curve's shape is the field's geometry and its sampling is uniform in
    distance rather than in time.  That is the standard definition of a
    streamline, and it is what makes the picture readable for a relaxation
    oscillator, where a time-parametrized curve spends almost all of its samples
    on the slow branches.

    Marching stops at the window edge or at a stagnation point (where the field
    has no direction to follow).  With ``both_ways`` each seed is marched
    backwards as well and the two halves are joined, so a streamline through a
    seed reaches equilibria on both sides.

    Parameters
    ----------
    system : ContinuousSystem
    region : sequence of (lo, hi[, n]), optional
        Where to sample, in the ONE state-space-box grammar every ``region=``
        door in the library reads: one ``(lo, hi)`` bound — or one
        ``(lo, hi, n)`` triple — **per state component**, of which the two
        ``plane`` axes are used.  It is the **second positional** argument, so
        ``ts.analysis.streamlines(system, [(-2, 2, 60), (-2, 2, 60)])`` reads exactly
        like ``ts.analysis.basins(system, ...)``.

        Two grammars for "this box of state space" is one too many: crossing them
        used to produce a bare ``TypeError: takes 1 positional argument but 2
        were given``, the one error in ``ts.analysis`` that taught nothing.
        Equivalent to ``xlim=`` / ``ylim=`` / the lattice size; passing both
        raises.

        .. versionadded:: 6.0
    plane, at, xlim, ylim
        The slice and window (see :func:`flow_field`).
    seeds : int or tuple of int, optional
        How many starting points per axis — the streamline lattice is
        ``seeds x seeds``, inset from the edges.  Default ``8``.

        It is a **lattice size**, not an RNG seed: nothing here is random, and
        this function takes no ``seed=``.  The two words are one letter apart and
        live in the same namespace, so read this one as "how many seed points".
    length : float, optional
        Arc length to integrate each direction.  ``None`` (default) uses the
        window's diagonal, which is the length that just crosses the picture.
    steps : int, optional
        Steps per direction.  Default ``200``.
    both_ways : bool, optional
        March backwards as well as forwards.  Default ``True``.

    Returns
    -------
    list of ndarray
        One ``(n, 2)`` polyline per seed that produced one (a seed sitting
        exactly on an equilibrium produces none).

    Notes
    -----
    **Why polylines and not a new mark.**  ``matplotlib.streamplot`` would give a
    prettier arrow-decorated result on one backend, at the cost of a ``STREAM``
    mark that plotly, JSON and three.js would each have to grow — and a
    streamline *is* a polyline, so the mark would carry no information the
    ``LINE`` mark does not.  Integrating here also uses the system's own
    right-hand side rather than a plotting library's interpolation of a
    pre-sampled lattice, which matters exactly where the picture is interesting:
    near a separatrix, where an interpolated field and the real one part company.
    """
    _require_flow(system, "streamlines")
    # ``seeds`` is this function's lattice size (there is no ``grid=`` here), so
    # a region's per-axis counts land on it.
    xlim, ylim, seeds = window_from_region(region, plane, system, xlim, ylim, seeds)
    i, j, _ = resolve_plane(system, plane)
    base = _base_state(system, at)
    xlim, ylim, _ = window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)

    nx, ny = (
        (seeds, seeds) if isinstance(seeds, (int, np.integer)) else (int(seeds[0]), int(seeds[1]))
    )
    if nx < 1 or ny < 1:
        raise InvalidParameterError(f"seeds= needs at least 1 per axis, got {(nx, ny)}.")
    # Inset the seed lattice by half a cell so no seed sits on the boundary,
    # where it would be marched straight out of the window in one direction.
    sx = np.linspace(xlim[0], xlim[1], nx + 2)[1:-1]
    sy = np.linspace(ylim[0], ylim[1], ny + 2)[1:-1]

    span = float(np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    arc = float(length) if length is not None else span
    h = arc / max(int(steps), 1)
    tiny = 1e-12 * max(span, 1.0)

    field = _batched_sliced_rhs(system, i, j, base)
    gx, gy = np.meshgrid(sx, sy)
    seed_points = np.column_stack([gx.ravel(), gy.ravel()])

    def march(sign: float) -> list[list[tuple[float, float]]]:
        """March **every** seed in lockstep, each stopping on its own condition.

        Every seed takes the same number of Runge-Kutta stages per step, so the
        marches vectorize exactly: one right-hand-side call per stage for the
        whole lattice instead of one per seed per stage.  Each seed keeps its own
        ``alive`` flag, so a curve that leaves the window or reaches an
        equilibrium stops exactly where a per-seed loop would stop it.  The
        arithmetic per point is unchanged, so the curves are bit-identical to the
        per-point path (pinned by
        ``test_the_batched_sliced_field_is_the_per_point_field_exactly``).
        """
        tracks: list[list[tuple[float, float]]] = [
            [(float(p[0]), float(p[1]))] for p in seed_points
        ]
        position = seed_points.copy()
        alive = np.ones(seed_points.shape[0], dtype=bool)
        for _ in range(int(steps)):
            live = np.nonzero(alive)[0]
            if live.size == 0:
                break
            p = position[live]
            with np.errstate(all="ignore"):
                k1, ok1 = _unit_batch(field(p), sign, tiny)
                k2, ok2 = _unit_batch(field(p + 0.5 * h * k1), sign, tiny)
                k3, ok3 = _unit_batch(field(p + 0.5 * h * k2), sign, tiny)
                k4, ok4 = _unit_batch(field(p + h * k3), sign, tiny)
                # How far the direction turned across the step, measured
                # directly: the stages are unit vectors, so ``k1 . k4`` is the
                # cosine of the total turn.  A fixed-arc-length march that
                # strides *through* an equilibrium (where the direction reverses
                # over a distance the step cannot resolve) would otherwise emit a
                # segment tangent to nothing — a straight line drawn across a
                # fixed point, a picture of something that does not happen.  A
                # streamline ends at an equilibrium; stopping is the correct
                # answer, not a fallback.
                aligned = k1[:, 0] * k4[:, 0] + k1[:, 1] * k4[:, 1] >= _MIN_ALIGN
                delta = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                turn = np.hypot(delta[:, 0], delta[:, 1])
                # Re-projecting onto unit length restores the exact arc-length
                # parametrization the unit field defines (|u'| = 1), so samples
                # are evenly spaced in distance rather than nearly so.
                nxt = p + h * delta / turn[:, None]
            advance = ok1 & ok2 & ok3 & ok4 & aligned & (turn >= _MIN_TURN)
            advance &= np.isfinite(nxt).all(axis=1)
            inside = (
                (nxt[:, 0] >= xlim[0])
                & (nxt[:, 0] <= xlim[1])
                & (nxt[:, 1] >= ylim[0])
                & (nxt[:, 1] <= ylim[1])
            )
            for local, row in enumerate(live):
                if not advance[local]:
                    alive[row] = False
                    continue
                tracks[row].append((float(nxt[local, 0]), float(nxt[local, 1])))
                if inside[local]:
                    position[row] = nxt[local]
                else:
                    alive[row] = False  # the step that leaves is drawn, then stop
        return tracks

    forward = march(+1.0)
    backward = march(-1.0) if both_ways else [[(float(p[0]), float(p[1]))] for p in seed_points]
    out: list[np.ndarray] = []
    for back, fore in zip(backward, forward, strict=True):
        pts = back[::-1] + fore[1:]
        if len(pts) >= 2:
            out.append(np.asarray(pts, dtype=float))
    return out


#: The smallest ``cos`` of the turn a single step may make (60 degrees).  The
#: Runge-Kutta stages are unit vectors, so ``k1 . k4`` *is* that cosine — a
#: scale-free measure needing no per-system tuning.  Below it the step no longer
#: resolves the curve, which in practice means the march has reached an
#: equilibrium; raise ``steps=`` to resolve a genuinely tight turn.
_MIN_ALIGN: float = 0.5

#: A second, weaker guard on the same quantity: the norm of the averaged stage
#: directions, which collapses when the stages disagree.  Kept because it is the
#: divisor of the unit re-projection and so must not be near zero.
_MIN_TURN: float = 0.5


def _batched_sliced_rhs(
    system: Any, i: int, j: int, base: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``G(points) -> (m, 2)``, the sliced right-hand side over a whole batch.

    The same broadcasting trick :func:`_batched_speed` uses, for the same reason:
    a streamline lattice costs ``seeds x directions x steps x 4`` right-hand-side
    evaluations — 102,400 at the defaults — and doing them one Python call at a
    time is the whole cost of the plot.  Falls back to the per-point callable if
    the lambdified kernel declines a batch.
    """
    per_point = _sliced_rhs(system, i, j, base)

    def by_point(points: np.ndarray) -> np.ndarray:
        return np.array(
            [per_point(float(p[0]), float(p[1])) for p in np.atleast_2d(points)], dtype=float
        )

    try:
        fn, _, control_names = system._build_lambdified()
        values = np.array([float(system.params[name]) for name in control_names])
    except Exception:  # noqa: BLE001 - a non-standard kernel keeps the slow path
        return by_point

    dim = int(getattr(system, "dim", 2))

    def batched(points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        m = pts.shape[0]
        state = np.tile(np.asarray(base, dtype=float), (m, 1))
        state[:, i] = pts[:, 0]
        state[:, j] = pts[:, 1]
        parts = [state, np.zeros((m, 1))]
        if values.size:
            parts.append(np.tile(values, (m, 1)))
        try:
            out = np.asarray(fn(np.column_stack(parts)), dtype=float).reshape(m, dim)
        except Exception:  # noqa: BLE001 - defensive; the per-point path is exact too
            return by_point(pts)
        return np.column_stack([out[:, i], out[:, j]])

    return batched


def _unit_batch(uv: np.ndarray, sign: float, tiny: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit-normalize a batch of field vectors; report which ones had a direction.

    The batch twin of :func:`_unit`.  A stagnation row (or a non-finite one) gets
    the zero vector and a ``False`` flag rather than a ``NaN``, so the arithmetic
    of the *other* rows is never contaminated and the caller can stop exactly
    that curve.
    """
    magnitude = np.hypot(uv[:, 0], uv[:, 1])
    ok = np.isfinite(magnitude) & (magnitude > tiny)
    safe = np.where(ok, magnitude, 1.0)
    unit = np.where(ok[:, None], sign * uv / safe[:, None], 0.0)
    return unit, ok


def _unit(uv: tuple[float, float], sign: float, tiny: float) -> tuple[float, float] | None:
    """Return the unit vector along ``uv`` (times ``sign``), or ``None`` at a stagnation point."""
    u, v = uv
    mag = float(np.hypot(u, v))
    if not np.isfinite(mag) or mag <= tiny:
        return None
    return (sign * u / mag, sign * v / mag)


# ---------------------------------------------------------------------------
# trace_determinant
# ---------------------------------------------------------------------------

#: The linear classification of a 2x2 Jacobian from its trace and determinant.
_CLASSES = (
    "saddle",
    "stable node",
    "unstable node",
    "stable focus",
    "unstable focus",
    "centre",
    "degenerate",
)


def classify_linear(trace: float, determinant: float, *, tol: float = 1e-9) -> str:
    r"""Classify a planar linearization from its trace and determinant.

    The reading of the trace-determinant plane, in one function:
    ``det < 0`` is a saddle; above the parabola ``det = tr^2/4`` the eigenvalues
    are complex (a focus, or a centre when the trace vanishes); below it they are
    real (a node); and the trace's sign decides stability.

    Parameters
    ----------
    trace, determinant : float
        The invariants of the 2x2 Jacobian.
    tol : float, optional
        How close to zero counts as zero.  Default ``1e-9``.

    Returns
    -------
    str
        One of ``"saddle"``, ``"stable node"``, ``"unstable node"``,
        ``"stable focus"``, ``"unstable focus"``, ``"centre"``, ``"degenerate"``.
    """
    if determinant < -tol:
        return "saddle"
    if abs(determinant) <= tol:
        return "degenerate"
    disc = trace * trace - 4.0 * determinant
    if disc < -tol:
        if abs(trace) <= tol:
            return "centre"
        return "stable focus" if trace < 0 else "unstable focus"
    if abs(trace) <= tol:
        return "degenerate"
    return "stable node" if trace < 0 else "unstable node"


def trace_determinant(
    system: Any,
    *,
    plane: Sequence[int | str] = (0, 1),
    points: Any | None = None,
    trace_range: tuple[float, float] | None = None,
    samples: int = 201,
    **fixed_point_kwargs: Any,
) -> TraceDeterminant:
    r"""Place a system's equilibria on the trace-determinant stability plane.

    The classic teaching figure: every planar linearization is one point
    :math:`(\tau, \Delta)`, the parabola :math:`\Delta = \tau^2/4` separates
    nodes from spirals, the axis :math:`\Delta = 0` separates them both from
    saddles, and :math:`\tau = 0` separates stable from unstable.  Reading a
    system's equilibria off that one diagram is how the phase-plane chapter of
    every textbook classifies them; nothing in Python ships it.

    Parameters
    ----------
    system : ContinuousSystem
        The flow whose equilibria are classified.
    plane : sequence of int or str, optional
        For a system with more than two coordinates, the 2x2 sub-block of the
        Jacobian to take the invariants of — i.e. the linearization *within the
        slice*.  For a planar system this is the whole Jacobian.
    points : array-like, optional
        Equilibria to use, shape ``(n, dim)``.  ``None`` (default) calls
        :func:`~tsdynamics.analysis.fixedpoints.fixed.fixed_points`; extra
        keyword arguments are forwarded to it.
    trace_range : tuple of float, optional
        The trace range the parabola is drawn over.  ``None`` covers the
        equilibria with a margin, and falls back to ``(-2, 2)`` when there are
        none.
    samples : int, optional
        Points on the parabola.  Default ``201``.
    **fixed_point_kwargs
        Forwarded to ``fixed_points`` when ``points`` is not given.

    Returns
    -------
    TraceDeterminant

    Notes
    -----
    For a flow with more than two coordinates the ``(tr, det)`` of a 2x2
    sub-block is **not** the full linearization: the classification it yields is
    that of the flow restricted to the slice.  The full spectrum is
    ``FixedPoint.eigenvalues``.
    """
    _require_flow(system, "trace_determinant")
    i, j, _ = resolve_plane(system, plane)

    if points is None:
        from tsdynamics.analysis.fixedpoints.fixed import fixed_points

        found = fixed_points(system, **fixed_point_kwargs)
        # ``found.points`` is the ``(n, dim)`` matrix of the equilibria: indexing
        # a result collection gives NUMBERS in v6, so there is no ``fp.x`` to
        # gather here (the records are ``found.details``).
        pts = np.asarray(found.points, dtype=float).reshape(len(found), -1)
    else:
        pts = np.atleast_2d(np.asarray(points, dtype=float))

    traces = np.empty(pts.shape[0], dtype=float)
    dets = np.empty(pts.shape[0], dtype=float)
    classes: list[str] = []
    for k, x in enumerate(pts):
        jac = np.asarray(system.jacobian(x, 0.0), dtype=float)
        block = jac[np.ix_([i, j], [i, j])]
        traces[k] = float(np.trace(block))
        dets[k] = float(np.linalg.det(block))
        classes.append(classify_linear(traces[k], dets[k]))

    if trace_range is None:
        if traces.size and np.isfinite(traces).any():
            # Always span tau = 0: the vertical axis is what separates stable
            # from unstable, so a window that excludes it draws the diagram
            # without the line the reader is looking for.
            lo = float(min(0.0, np.nanmin(traces)))
            hi = float(max(0.0, np.nanmax(traces)))
            span = max(hi - lo, 1.0)
            trace_range = (lo - 0.4 * span, hi + 0.4 * span)
        else:
            trace_range = (-2.0, 2.0)
    taus = np.linspace(trace_range[0], trace_range[1], int(samples))
    parabola = np.column_stack([taus, taus**2 / 4.0])
    return TraceDeterminant(
        trace=traces,
        determinant=dets,
        classes=classes,
        points=pts,
        parabola=parabola,
        trace_range=(float(trace_range[0]), float(trace_range[1])),
        meta={"plane": (i, j), "n_equilibria": int(pts.shape[0])},
    )


# ---------------------------------------------------------------------------
# The ensemble fields: ftle / escape time / transient time
# ---------------------------------------------------------------------------


def _slice_ics(base: np.ndarray, i: int, j: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Build the ``(ny*nx, dim)`` initial conditions of a lattice over the slice."""
    gx, gy = np.meshgrid(xs, ys)
    ics = np.tile(base, (gx.size, 1))
    ics[:, i] = gx.ravel()
    ics[:, j] = gy.ravel()
    return ics


def ftle_field(
    system: Any,
    region: Any | None = None,
    *,
    plane: Sequence[int | str] = (0, 1),
    at: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int | tuple[int, int] = 101,
    final_time: float = 1.0,
    backward: bool = False,
    **integrate_kwargs: Any,
) -> ScalarField:
    r"""Compute the finite-time Lyapunov exponent field over a 2-D slice.

    For each lattice point :math:`x_0` the flow map :math:`\phi_T` is evaluated
    by integrating the **full** system for a time :math:`T`; the in-plane
    deformation gradient :math:`F = \partial\phi_T/\partial x_0` is taken by
    finite differences on the lattice (:func:`numpy.gradient`), the right
    Cauchy-Green tensor is :math:`C = F^\top F`, and

    .. math::  \sigma(x_0) = \frac{1}{|T|}\,\log\sqrt{\lambda_{\max}(C)} .

    **Ridges of** :math:`\sigma` **are the Lagrangian coherent structures** — the
    material curves that organise transport.  Forward time reveals *repelling*
    structures (the stable manifolds); ``backward=True`` integrates backwards and
    reveals *attracting* ones.  For a system with two attractors the forward
    ridge sits on the basin boundary, which is the cheapest available check that
    the field is right.

    One ensemble call does the whole lattice: 10,201 initial conditions to
    ``T = 1`` takes a few milliseconds on the compiled engine.

    Parameters
    ----------
    system : ContinuousSystem
    region : sequence of (lo, hi[, n]), optional
        Where to sample, in the ONE state-space-box grammar every ``region=``
        door in the library reads: one ``(lo, hi)`` bound — or one
        ``(lo, hi, n)`` triple — **per state component**, of which the two
        ``plane`` axes are used.  It is the **second positional** argument, so
        ``ts.analysis.ftle_field(system, [(-2, 2, 60), (-2, 2, 60)])`` reads exactly
        like ``ts.analysis.basins(system, ...)``.

        Two grammars for "this box of state space" is one too many: crossing them
        used to produce a bare ``TypeError: takes 1 positional argument but 2
        were given``, the one error in ``ts.analysis`` that taught nothing.
        Equivalent to ``xlim=`` / ``ylim=`` / the lattice size; passing both
        raises.

        .. versionadded:: 6.0
    plane, at, xlim, ylim
        The slice and window (see :func:`flow_field`).
    grid : int or tuple of int, optional
        Lattice resolution.  Default ``101``.  The finite-difference gradient
        means the field is only as sharp as this lattice: a ridge thinner than a
        cell is smeared, never sharpened.
    final_time : float, optional
        The integration time :math:`T`.  Default ``1.0``.  This is a *choice*,
        not a parameter of the system — the field genuinely depends on it — so it
        is recorded in ``meta``.
    backward : bool, optional
        Integrate backwards in time (attracting structures).  Default ``False``.
    **integrate_kwargs
        Forwarded to :func:`tsdynamics._engine.run.ensemble` (``method``,
        ``rtol``, ``dt``, ``backend``, ...).

    Returns
    -------
    ScalarField
        ``values`` is :math:`\sigma`, in inverse time units.  A lattice point
        whose trajectory diverged is ``NaN`` and stays ``NaN`` (a gradient
        touching one is ``NaN`` too, rather than being quietly filled in).

    Notes
    -----
    On a slice of a higher-dimensional flow the initial conditions lie in the
    slice but the trajectories leave it; the deformation gradient taken here is
    the in-plane block of the full one, i.e. the stretching *of the slice*
    measured in the plane it was drawn on.  That is the standard 2-D FTLE
    section and it is what the ridges of a 3-D flow's section mean.
    """
    from tsdynamics._engine import run as _run

    _require_flow(system, "ftle_field")
    xlim, ylim, grid = window_from_region(region, plane, system, xlim, ylim, grid)
    i, j, labels = resolve_plane(system, plane)
    base = _base_state(system, at)
    xlim, ylim, meta = window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)
    xs, ys = _grid_axes(xlim, ylim, grid)
    if float(final_time) == 0.0:
        raise InvalidParameterError("final_time= is the FTLE horizon and must be non-zero.")

    ics = _slice_ics(base, i, j, xs, ys)
    horizon = float(abs(final_time))
    if backward:
        final = _backward_final(system, ics, horizon, integrate_kwargs)
    else:
        final = _run.ensemble(system, ics, final_time=horizon, **integrate_kwargs)
    final = np.asarray(final, dtype=float)

    fx = final[:, i].reshape(ys.size, xs.size)
    fy = final[:, j].reshape(ys.size, xs.size)
    # np.gradient returns d/d(row) then d/d(col); rows are y, columns are x.
    dfx_dy, dfx_dx = np.gradient(fx, ys, xs)
    dfy_dy, dfy_dx = np.gradient(fy, ys, xs)

    # C = F^T F for the 2x2 F = [[dfx_dx, dfx_dy], [dfy_dx, dfy_dy]], then the
    # larger eigenvalue in closed form (the analytic root is exact and avoids a
    # per-cell eigensolve over ~10^4 cells).
    c11 = dfx_dx**2 + dfy_dx**2
    c22 = dfx_dy**2 + dfy_dy**2
    c12 = dfx_dx * dfx_dy + dfy_dx * dfy_dy
    half = 0.5 * (c11 + c22)
    radius = np.sqrt(np.maximum((0.5 * (c11 - c22)) ** 2 + c12**2, 0.0))
    lam_max = half + radius
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.log(np.sqrt(np.maximum(lam_max, np.finfo(float).tiny))) / horizon
    sigma = np.where(np.isfinite(sigma), sigma, np.nan)

    meta.update(
        plane=(i, j),
        at=base.tolist(),
        grid=(int(xs.size), int(ys.size)),
        ftle_time=float(final_time),
        backward=bool(backward),
    )
    return ScalarField(xs, ys, sigma, "FTLE", labels, meta)


def _backward_final(
    system: Any, ics: np.ndarray, horizon: float, kwargs: dict[str, Any]
) -> np.ndarray:
    r"""Integrate a batch **backwards** by ``horizon`` on the time-reversed flow.

    The engine integrates forward only, so backward time is taken the way it is
    defined — on the flow with :math:`f \\to -f` — and the reversal is done where
    the flow is actually described, in the **symbolic** right-hand side: the
    system's equations are negated and lowered to their own engine tape via the
    public :func:`~tsdynamics._engine.compile.lower_expressions`, exactly as the
    variational (Lyapunov) lowering does.  The reversed tape is then handed to
    the ordinary ensemble path, so backward FTLE costs the same as forward FTLE
    and runs on the same compiled engine.

    Wrapping the *system* in a negating proxy was the obvious alternative and it
    does not work: family detection walks the MRO, so a proxy is not a
    ``ContinuousSystem`` and the problem builder refuses it.
    """
    from tsdynamics._engine import run as _run

    problem = _reversed_problem(system)
    return np.asarray(_run.ensemble(problem, ics, final_time=horizon, **kwargs), dtype=float)


def _reversed_problem(system: Any) -> Any:
    """Build an :class:`~tsdynamics._engine.problem.ODEProblem` for ``-f``.

    Lowered ``jacobian=True`` so a flow whose ``_default_method`` is an implicit
    kernel still integrates: a pre-built problem is never re-lowered by
    ``run.ensemble``, so the Jacobian block has to be there from the start.
    """
    import symengine

    from tsdynamics._engine.compile import lower_expressions
    from tsdynamics._engine.problem import ODEProblem
    from tsdynamics._engine.symbols import state_time_symbols

    y, t_sym = state_time_symbols()
    dim = int(system.dim)
    struct_vals = system._structural_vals()
    control_names = list(system._control_params())
    control_syms = {name: symengine.Symbol(f"p{i}") for i, name in enumerate(control_names)}
    raw = list(type(system)._equations(y, t_sym, **{**struct_vals, **control_syms}))

    u_syms = [symengine.Symbol(f"u{j}") for j in range(dim)]
    t_canon = symengine.Symbol("t")
    subs: dict[Any, Any] = {y(j): u_syms[j] for j in range(dim)}
    subs[t_sym] = t_canon
    exprs = [-symengine.sympify(expr).subs(subs) for expr in raw]

    tape = lower_expressions(
        exprs,
        u_syms,
        param_syms=[control_syms[name] for name in control_names],
        time_sym=t_canon,
        jacobian=True,
        control_names=control_names,
    )
    return ODEProblem(tape=tape, ic=np.zeros(dim, dtype=float), t0=0.0, system=system)


def _march_first_time(
    system: Any,
    ics: np.ndarray,
    *,
    predicate: Callable[[np.ndarray], np.ndarray],
    final_time: float,
    chunks: int,
    integrate_kwargs: dict[str, Any],
) -> np.ndarray:
    """Return the first chunk time at which ``predicate`` holds, or ``NaN``.

    One ensemble march, split into ``chunks`` segments, each continuing from the
    previous segment's states — so the total integration work is one run to
    ``final_time``, not ``chunks`` of them.  A row that has already satisfied the
    predicate is frozen (its recorded time never changes) and its subsequent
    divergence cannot un-record it, which is exactly what a fractal escape-time
    picture needs.

    **Only the rows still in play are advanced.**  A finished row's answer can
    never change, so dropping it from the batch is exactly answer-preserving (the
    engine's batch is bit-identical to the same members run one at a time), and
    it is the difference between a plot and a hang: the rows that satisfy an
    *escape* predicate are precisely the ones running away to infinity, and
    integrating a runaway to ``final_time`` at the default tolerances costs
    exponentially more per chunk as its state grows.  Measured on the default
    ``ts.plot(LotkaVolterra(), "escape_time")`` window at a mere 11x11 lattice,
    marching the dead rows too took **172 s** (71 s in the last chunk alone, with
    the escaped states at 7e9 and climbing); advancing only the pending rows
    takes **0.04 s** and returns a bit-identical field.

    The resolution of the answer is ``final_time / chunks`` — this is a sampled
    first-passage time, not a root-found one, and the transforms record the
    chunk size in ``meta`` so the quantisation is visible.
    """
    from tsdynamics._engine import run as _run

    n = ics.shape[0]
    out = np.full(n, np.nan, dtype=float)
    state = np.array(ics, dtype=float, copy=True)
    step = float(final_time) / max(int(chunks), 1)
    done = np.asarray(predicate(state), dtype=bool)
    out[done] = 0.0
    for k in range(int(chunks)):
        live = np.flatnonzero(~done)
        if live.size == 0:
            break
        t0 = k * step
        advanced = np.asarray(
            _run.ensemble(system, state[live], t0=t0, final_time=t0 + step, **integrate_kwargs),
            dtype=float,
        )
        state[live] = advanced
        hit = np.asarray(predicate(advanced), dtype=bool)
        out[live[hit]] = t0 + step
        done[live] = hit | ~np.isfinite(advanced).all(axis=1)
    return out


def escape_time_field(
    system: Any,
    region: Any | None = None,
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
) -> ScalarField:
    """Compute the escape-time field: how long each start takes to leave a region.

    The classic picture of transient chaos and of a fractal basin boundary: an
    initial condition near the boundary lingers for a long time before it commits
    to one exit, so the field's level sets accumulate on the boundary and the
    picture resolves structure that a two-colour basin diagram cannot.

    Parameters
    ----------
    system : ContinuousSystem
    region : sequence of (lo, hi[, n]), optional
        Where to sample, in the ONE state-space-box grammar every ``region=``
        door in the library reads: one ``(lo, hi)`` bound — or one
        ``(lo, hi, n)`` triple — **per state component**, of which the two
        ``plane`` axes are used.  It is the **second positional** argument, so
        ``ts.analysis.escape_time_field(system, [(-2, 2, 60), (-2, 2, 60)])`` reads exactly
        like ``ts.analysis.basins(system, ...)``.

        Two grammars for "this box of state space" is one too many: crossing them
        used to produce a bare ``TypeError: takes 1 positional argument but 2
        were given``, the one error in ``ts.analysis`` that taught nothing.
        Equivalent to ``xlim=`` / ``ylim=`` / the lattice size; passing both
        raises.

        .. versionadded:: 6.0
    plane, at, xlim, ylim
        The slice and window (see :func:`flow_field`).
    grid : int or tuple of int, optional
        Lattice resolution.  Default ``101``.
    final_time : float, optional
        How long to wait before giving up.  Default ``20``.  A point that has
        not escaped by then is ``NaN``, which is honest: "it did not escape
        within the horizon", not "it escaped at the horizon".
    chunks : int, optional
        Time resolution: the answer is quantised to ``final_time / chunks``.
        Default ``40``.
    escape : callable or float, optional
        What "escaped" means.  A callable takes the ``(n, dim)`` states and
        returns an ``(n,)`` boolean.  A float is a radius: escaped when the
        **in-plane** distance from the window centre exceeds it.  ``None``
        (default) is *left the drawn window*, in-plane — the picture the window
        is already showing.
    **integrate_kwargs
        Forwarded to :func:`tsdynamics._engine.run.ensemble`.

    Returns
    -------
    ScalarField
    """
    _require_flow(system, "escape_time_field")
    xlim, ylim, grid = window_from_region(region, plane, system, xlim, ylim, grid)
    i, j, labels = resolve_plane(system, plane)
    base = _base_state(system, at)
    xlim, ylim, meta = window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)
    xs, ys = _grid_axes(xlim, ylim, grid)
    ics = _slice_ics(base, i, j, xs, ys)

    predicate, described = _escape_predicate(escape, i, j, xlim, ylim)
    times = _march_first_time(
        system,
        ics,
        predicate=predicate,
        final_time=float(final_time),
        chunks=int(chunks),
        integrate_kwargs=integrate_kwargs,
    )
    meta.update(
        plane=(i, j),
        at=base.tolist(),
        grid=(int(xs.size), int(ys.size)),
        final_time=float(final_time),
        time_resolution=float(final_time) / max(int(chunks), 1),
        escape=described,
    )
    return ScalarField(xs, ys, times.reshape(ys.size, xs.size), "escape time", labels, meta)


def _escape_predicate(
    escape: Callable[[np.ndarray], np.ndarray] | float | None,
    i: int,
    j: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Resolve the ``escape=`` declaration into ``(predicate, description)``."""
    if callable(escape):
        return escape, "caller-supplied predicate"
    if escape is not None:
        radius = float(escape)
        cx = 0.5 * (xlim[0] + xlim[1])
        cy = 0.5 * (ylim[0] + ylim[1])

        def by_radius(states: np.ndarray) -> np.ndarray:
            d = np.hypot(states[:, i] - cx, states[:, j] - cy)
            return ~np.isfinite(d) | (d > radius)

        return by_radius, f"in-plane distance from the window centre > {radius:g}"

    def by_window(states: np.ndarray) -> np.ndarray:
        x, y = states[:, i], states[:, j]
        inside = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        return ~inside | ~np.isfinite(x) | ~np.isfinite(y)

    return by_window, "left the drawn window (in-plane)"


def transient_time_field(
    system: Any,
    region: Any | None = None,
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
) -> ScalarField:
    """Compute the transient-time field: how long each start takes to settle.

    The complement of :func:`escape_time_field`: escape time is the first time an
    orbit **leaves** a set, transient time is the first time it **arrives**.  The
    default arrival test is that the flow has slowed to a stop —
    ``|f(u)| < tol`` — which is the right test for a system whose attractors are
    equilibria (a two-well potential, a multistable planar flow), and the wrong
    one for a limit cycle, where the speed never becomes small.  For those, pass
    an explicit ``settled=`` predicate.  The library says which test it used in
    ``meta`` rather than guessing silently.

    Parameters
    ----------
    system : ContinuousSystem
    region : sequence of (lo, hi[, n]), optional
        Where to sample, in the ONE state-space-box grammar every ``region=``
        door in the library reads: one ``(lo, hi)`` bound — or one
        ``(lo, hi, n)`` triple — **per state component**, of which the two
        ``plane`` axes are used.  It is the **second positional** argument, so
        ``ts.analysis.transient_time_field(system, [(-2, 2, 60), (-2, 2, 60)])`` reads exactly
        like ``ts.analysis.basins(system, ...)``.

        Two grammars for "this box of state space" is one too many: crossing them
        used to produce a bare ``TypeError: takes 1 positional argument but 2
        were given``, the one error in ``ts.analysis`` that taught nothing.
        Equivalent to ``xlim=`` / ``ylim=`` / the lattice size; passing both
        raises.

        .. versionadded:: 6.0
    plane, at, xlim, ylim, grid, final_time, chunks
        As in :func:`escape_time_field`.
    tol : float, optional
        The speed below which the orbit counts as settled.  ``None`` (the
        default) reads the scale off the field itself — 1% of the median speed
        over the sampled lattice — and records the number it chose in ``meta``,
        because an absolute speed threshold means nothing without knowing the
        flow's own scale.
    settled : callable, optional
        An explicit arrival test taking the ``(n, dim)`` states and returning an
        ``(n,)`` boolean — a ball around a known attractor, a section crossing, a
        sign condition.  Overrides ``tol``.
    **integrate_kwargs
        Forwarded to :func:`tsdynamics._engine.run.ensemble`.

    Returns
    -------
    ScalarField
    """
    _require_flow(system, "transient_time_field")
    xlim, ylim, grid = window_from_region(region, plane, system, xlim, ylim, grid)
    i, j, labels = resolve_plane(system, plane)
    base = _base_state(system, at)
    xlim, ylim, meta = window_for(system, plane=plane, at=at, xlim=xlim, ylim=ylim)
    xs, ys = _grid_axes(xlim, ylim, grid)
    ics = _slice_ics(base, i, j, xs, ys)

    if settled is not None:
        predicate = settled
        described = "caller-supplied arrival test"
    else:
        speed_of = _batched_speed(system)
        if tol is None:
            # A speed threshold is a *scale*, and a flow has no universal one:
            # a fixed 1e-3 makes this field almost entirely "did not settle" on a
            # system whose speeds are of order 1e-2, and almost entirely zero on
            # one of order 1e3.  So the default is read off the field itself — a
            # percent of the median speed over the lattice being drawn — and, like
            # the window, it is *recorded* rather than silently applied.
            sampled = speed_of(ics)
            finite = sampled[np.isfinite(sampled)]
            threshold = 0.01 * float(np.median(finite)) if finite.size else 1e-3
            described = f"speed |f(u)| < {threshold:g} (1% of the median lattice speed)"
        else:
            threshold = float(tol)
            described = f"speed |f(u)| < {threshold:g}"

        def by_speed(states: np.ndarray) -> np.ndarray:
            return np.asarray(speed_of(states) < threshold, dtype=bool)

        predicate = by_speed

    times = _march_first_time(
        system,
        ics,
        predicate=predicate,
        final_time=float(final_time),
        chunks=int(chunks),
        integrate_kwargs=integrate_kwargs,
    )
    _warn_if_nothing_arrived(times, described)
    meta.update(
        plane=(i, j),
        at=base.tolist(),
        grid=(int(xs.size), int(ys.size)),
        final_time=float(final_time),
        time_resolution=float(final_time) / max(int(chunks), 1),
        settled=described,
    )
    return ScalarField(xs, ys, times.reshape(ys.size, xs.size), "transient time", labels, meta)


#: Below this finite fraction, a transient-time field is not a picture — it is a
#: blank rectangle, and the caller is told why.
_ARRIVAL_FLOOR = 0.05


def _warn_if_nothing_arrived(times: np.ndarray, described: str) -> None:
    """Warn when the arrival test was satisfied almost nowhere.

    ``NaN`` means *did not settle within the horizon*, which is an honest answer
    and stays the answer — but a field that is 99% ``NaN`` renders as an empty
    rectangle, and the reader has no way to tell "nothing settles here" from "the
    plot is broken".  The overwhelmingly common cause is the documented one: the
    default arrival test is ``|f(u)| -> 0``, which detects an **equilibrium** and
    can never be satisfied on a **limit cycle**, where the speed stays bounded
    away from zero.  Measured: Van der Pol at the default settings leaves 1 cell
    of 3721 finite, while the same system below its Hopf threshold leaves all of
    them.  So say so, once, and name the two ways out.
    """
    if times.size == 0:
        return
    arrived = float(np.isfinite(times).mean())
    if arrived >= _ARRIVAL_FLOOR:
        return
    warnings.warn(
        f"transient_time_field: the arrival test ({described}) was satisfied at "
        f"{arrived:.1%} of the lattice, so the field is almost entirely NaN and will "
        "render nearly blank. The default test detects an equilibrium; on a limit "
        "cycle the speed never becomes small. Pass an explicit settled= predicate "
        "(a ball around the attractor, a section crossing), raise tol=, or lengthen "
        "final_time=.",
        RuntimeWarning,
        stacklevel=3,
    )


def _batched_speed(system: Any) -> Callable[[np.ndarray], np.ndarray]:
    r"""Return ``speed(states) -> |f|`` evaluated over a **whole batch** at once.

    The arrival test runs once per lattice point per chunk — on a 101x101 field
    over 40 chunks that is 400,000 evaluations, and calling the per-point
    ``_rhs_numeric`` that many times costs seconds of pure Python.  The
    lambdified right-hand side the library already builds broadcasts over a
    leading axis, so one call does the whole batch; measured on a Brusselator
    field this is the difference between 10 s and 0.1 s, bit-for-bit identically
    (verified: ``max|batched - per-row| == 0``).

    Falls back to the per-row loop if the lambdified callable declines a batch,
    so a system with an unusual kernel still gets the right answer, slowly.
    """
    per_row = system._rhs_numeric()

    def by_row(states: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(states, dtype=float))
        return np.array(
            [
                float(np.linalg.norm(per_row(row, 0.0))) if np.isfinite(row).all() else np.inf
                for row in rows
            ]
        )

    try:
        fn, _, control_names = system._build_lambdified()
        values = np.array([float(system.params[name]) for name in control_names])
    except Exception:  # noqa: BLE001 - a non-standard kernel keeps the slow path
        return by_row

    def batched(states: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(states, dtype=float))
        n = rows.shape[0]
        parts = [rows, np.zeros((n, 1))]
        if values.size:
            parts.append(np.tile(values, (n, 1)))
        try:
            out = np.asarray(fn(np.column_stack(parts)), dtype=float).reshape(n, -1)
        except Exception:  # noqa: BLE001 - defensive; the per-row path is exact too
            return by_row(rows)
        speed = np.linalg.norm(out, axis=1)
        # A diverged row has no meaningful speed: call it "not settled" rather
        # than let a NaN comparison decide it by accident.
        return np.where(np.isfinite(speed), speed, np.inf)

    return batched


# ---------------------------------------------------------------------------
# The natural measure
# ---------------------------------------------------------------------------


def invariant_density(
    values: Any, *, bins: int = 200, range: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Estimate the natural measure of one observable — a normalized orbit histogram.

    The invariant density is what the attractor *is*, statistically: the fraction
    of its time the orbit spends in each part of state space.  For the logistic
    map at :math:`r = 4` it is known in closed form,

    .. math::  \rho(x) = \frac{1}{\pi\sqrt{x(1-x)}} ,

    which makes it one of the few plots in a dynamics library with an exact
    reference to check against.

    Parameters
    ----------
    values : array-like
        The samples — one component of a long orbit.
    bins : int, optional
        Number of bins.  Default ``200``.
    range : tuple of float, optional
        ``(lo, hi)``.  ``None`` uses the data range.

    Returns
    -------
    tuple of ndarray
        ``(centres, density, edges)`` — the bin centres, the density
        (integrating to 1 over the range), and the bin edges.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If ``values`` is a system rather than its output (this measures a point
        set, so it needs the run first), or if no finite sample survives.
    """
    # It is the ONE data-first analysis in this module, and it was the only one
    # with no guard: a system fell through to ``np.asarray(..., dtype=float)``
    # and surfaced as "float() argument must be ... not 'Lorenz'".
    reject_system(values, analysis="invariant_density")
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise InvalidInputError("invariant_density needs at least one finite sample.")
    counts, edges = np.histogram(arr, bins=int(bins), range=range, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts, edges


def invariant_density_2d(
    x: Any,
    y: Any,
    *,
    bins: int | tuple[int, int] = 200,
    range: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the natural measure on a 2-D projection of the attractor.

    Also the right way to draw a very large point cloud: ten million orbit points
    drawn as markers are a black rectangle, and binned are a picture of the
    measure.

    Parameters
    ----------
    x, y : array-like
        The two components.
    bins : int or tuple of int, optional
        Bins per axis.  Default ``200``.
    range : tuple, optional
        ``((xlo, xhi), (ylo, yhi))``.  ``None`` uses the data range.

    Returns
    -------
    tuple of ndarray
        ``(xs, ys, density)`` — the bin centres along each axis and the density,
        shape ``(ny, nx)`` (rows are ``y``, as an image expects).
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size:
        raise InvalidInputError(
            f"invariant_density_2d needs two series of the same length, got {xa.size} and {ya.size}."
        )
    good = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[good], ya[good]
    if xa.size == 0:
        raise InvalidInputError("invariant_density_2d needs at least one finite sample pair.")
    counts, xedges, yedges = np.histogram2d(xa, ya, bins=bins, range=range, density=True)
    xs = 0.5 * (xedges[:-1] + xedges[1:])
    ys = 0.5 * (yedges[:-1] + yedges[1:])
    return xs, ys, counts.T
