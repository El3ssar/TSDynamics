"""Poincaré (first-return) map of a continuous-time system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.engine.run import Event

from tsdynamics.engine.events import _DIRECTION_WORDS  # noqa: F401  # re-export for back-compat
from tsdynamics.engine.events import _normalize_event_direction as _normalize_direction
from tsdynamics.errors import ConvergenceError, invalid_value
from tsdynamics.families import Trajectory

from . import _crossings
from ._base import DerivedSystem

__all__ = ["PoincareMap", "PoincareSection", "auto_plane"]


# ---------------------------------------------------------------------------
# Friendly section spelling — named axes + direction words
# ---------------------------------------------------------------------------
# The direction vocabulary (``_DIRECTION_WORDS``) and its coercion helper
# (imported above as ``_normalize_direction``) are the single canonical pair in
# :mod:`tsdynamics.engine.events` — the engine layer sits *below* ``derived`` in
# the import graph, so importing them here introduces no cycle.  ``_DIRECTION_WORDS``
# is re-imported (rather than only used) to keep the historical
# ``poincare._DIRECTION_WORDS`` path resolvable.


#: Probe run used to choose a section when the caller names none.  Short and
#: coarse on purpose: the choice only needs the *shape* of the attractor, not an
#: accurate orbit, and this cost is paid once per :class:`PoincareMap`.
_AUTO_PROBE_FINAL_TIME = 100.0
_AUTO_PROBE_TRANSIENT = 50.0
_AUTO_PROBE_DT = 0.01

#: How many probe orbits to try before giving up.  A system with no ``default_ic``
#: starts from a fresh **random** state on every run, and for a few catalogue
#: attractors (SprottF and friends) a random start simply escapes to infinity —
#: the same reason ``iterate`` / ``basins`` retry rather than report the first
#: divergence as the system's verdict.  Each retry therefore draws a new start.
_AUTO_PROBE_ATTEMPTS = 4

#: Solver kernel for the probe: **fixed-step**, the same choice (and the same
#: reason) as the engine crossing march.  The probe needs the *shape* of the
#: attractor, not an accurate orbit, and a fixed-step kernel makes its cost
#: bounded and predictable — 3000 steps, ~4 ms.  An adaptive kernel does not: at
#: the v6 default tolerances a random start that wanders off a catalogue
#: attractor (SprottF) grinds for **over 100 seconds** shrinking its step before
#: reporting the divergence, which would turn "omit the plane" from a convenience
#: into a hang.  A system that declares an *implicit* default is stiff, where the
#: explicit kernel is unstable rather than merely inaccurate, so it keeps its own.
_AUTO_PROBE_METHOD = "rk4"
_IMPLICIT_METHODS: frozenset[str] = frozenset({"bdf", "rosenbrock", "trbdf2", "sdirk2"})


def _probe_method(system: Any) -> str:
    """Return the fixed-step probe kernel, unless this system declares a stiff one."""
    declared = str(getattr(type(system), "_default_method", "") or "").lower()
    return declared if declared in _IMPLICIT_METHODS else _AUTO_PROBE_METHOD


def auto_plane(system: Any) -> tuple[int, float]:
    """Choose a section plane for ``system`` by looking at where its orbit goes.

    A Poincaré map has no meaningful default section in the abstract — but
    "no default" used to mean ``PoincareMap(Lorenz())`` answered with Python's
    raw binder error (*missing 1 required positional argument: 'plane'*), which
    tells a new user nothing about what a plane even is.  So the section is
    **chosen and recorded**, the way :func:`~tsdynamics.analysis.orbit_diagram`
    chooses and records a flow's discrete view.

    The rule, and why it is this one: integrate a short probe orbit past a
    transient, then take the component with the largest **interquartile range**
    *among those the orbit repeatedly returns to the median of*, and section it
    at that median.

    * The median is the only offset a component is *guaranteed* to reach — half
      the sampled orbit lies on each side of it — so the chosen plane cannot miss
      **the probe orbit**, which is the failure mode a naive default (``x = 0``)
      walks into.  Note the exact claim: it is about the orbit that was probed.
      A system with no ``default_ic`` draws a fresh random start on every run, so
      the map's own march begins somewhere else, and on a system whose orbits do
      *not* all fall onto one attractor — a conservative flow, where each start
      sits on its own torus — the plane chosen for one orbit can genuinely miss
      the next (measured: ArnoldBeltramiChildress, ArnoldWeb).  Pin ``ic=`` on
      the system, or name the plane, whenever there is no single attractor to
      agree about.
    * The IQR, rather than the range or the variance, picks the component the
      orbit genuinely spreads across rather than the one with the largest
      excursion.  On Rössler that is the difference between the classical
      ``x``/``y`` section and a useless one through the rare ``z`` spike; the two
      robust statistics agree with the textbook choice on the systems that have
      one (Lorenz → ``z ≈ 22``, Rössler → ``x ≈ -0.6``).
    * **Reached once is not enough** — a section has to be crossed again and
      again, so a candidate must recross its median at least
      :data:`_MIN_RECROSSINGS` times.  Spread alone picks exactly the wrong
      component on a *driven* system: a carried drive phase (``z' = omega``) has
      the largest IQR of any coordinate by construction, because it sweeps
      monotonically across the whole window — yet it crosses its median once,
      ever, so ``PoincareMap(Duffing()).run(5)`` marched to
      ``max_time = 1e4`` and raised.  Measured over the catalogue, **30 of 137
      flows** (Duffing, ForcedVanDerPol, BickleyJet, DoubleGyre, …) chose such a
      coordinate under a spread-only rule; requiring recrossings moves them onto
      a component that really is a section (Duffing → ``x``).

    The probe is a few milliseconds for an ordinary flow; for a high-dimensional
    method-of-lines *field* it is the cost of integrating that field, which is
    seconds — name a plane there.

    Parameters
    ----------
    system : System or Trajectory
        The continuous system to be sectioned — a **copy** is probed, so the
        caller's live state and time are untouched — or measured data, whose
        samples are used directly (no probe run).

    Returns
    -------
    tuple
        ``(component_index, offset)`` — the resolved plane, in the raw form
        :meth:`PoincareMap._parse_plane` consumes.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If the probe cannot run, if its orbit is too flat to section (a system
        that settled on a fixed point has no crossings to find), or if no
        component recrosses its median often enough to be a section.  The message
        names concrete planes to pass instead, because at that point the caller
        does have to choose.
    """
    name = _display_name(system)
    is_data = isinstance(system, Trajectory)
    if not is_data:
        _refuse_a_family_with_no_section(system, name)
    stretch = 1.0
    while True:
        y = (
            np.atleast_2d(np.asarray(system.y, dtype=float))
            if is_data
            else _probe_orbit(system, name, stretch)
        )
        if y.ndim != 2 or y.shape[0] < 2 or not np.isfinite(y).all():
            raise _no_plane_error(system, f"the samples of {name} contain no usable orbit")

        spread, flat = _component_spread(y)
        if spread[int(np.argmax(spread))] <= flat:
            raise _no_plane_error(
                system,
                f"the orbit of {name} is flat — it sits at a fixed point, so nothing crosses "
                "any section",
            )
        chosen = _widest_recrossed_component(y, spread, flat)
        if chosen is not None:
            return chosen
        # A slow relaxation oscillator (FitzHughNagumo, CellCycle, CircadianRhythm)
        # simply has fewer than `_MIN_RECROSSINGS` periods inside the short window,
        # so a system gets ONE longer look before it is refused.  Measured data is
        # all the data there is, and a method-of-lines *field* would pay ten times
        # a probe that already costs seconds, so neither gets the second look —
        # refusing with a message beats a multi-minute stall on a call the caller
        # did not know was expensive.
        if is_data or stretch > 1.0 or int(getattr(system, "dim", 0)) > _AUTO_PROBE_RETRY_MAX_DIM:
            break
        stretch = _AUTO_PROBE_LONG_FACTOR
    raise _no_plane_error(
        system,
        f"no component of {name}'s orbit recrosses its own median more than "
        f"{_MIN_RECROSSINGS - 1} times, so none of them is a section the map could return "
        "to — which is what a monotonically advancing coordinate (a carried drive phase) "
        "looks like",
    )


#: How many times a component must recross its median in the probe window before
#: it is accepted as a section.  A Poincaré map is a *return* map, so one crossing
#: is not a section: the map would find it and then march to ``max_time`` looking
#: for the second.  Four is the smallest count that rejects every monotone
#: coordinate in the catalogue while accepting every genuine oscillation in it
#: (the tightest legitimate value measured is 5, Duffing's ``x``, against 0-3 for
#: every drive phase).
_MIN_RECROSSINGS = 4

#: Factor by which the probe window is stretched for a second look before giving
#: up.  Only the systems the short window cannot decide pay it, and the probe is
#: fixed-step, so the cost is exactly proportional: ~0.1 s for an ordinary flow.
_AUTO_PROBE_LONG_FACTOR = 10.0

#: State size above which the second, ten-times-longer look is not taken.  The
#: cost of a probe is the cost of integrating the system, and the catalogue splits
#: cleanly: every ordinary flow probes in 0.01-0.4 s (the largest,
#: KuramotoSivashinsky, has 32 states), while the method-of-lines *fields* take
#: 5.6 s at 1024 states and 13.5 s at 4608 — where a tenfold window would mean
#: minutes.  The bound is on the **dimension** rather than on the measured
#: elapsed time so that which systems are refused does not depend on how fast the
#: machine is; a field is exactly the case whose docstring already says "name a
#: plane here".
_AUTO_PROBE_RETRY_MAX_DIM = 64


def _component_spread(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-component interquartile range, and the width below which it is *flat*."""
    q1, q3 = np.percentile(y, [25.0, 75.0], axis=0)
    scale = float(np.max(np.abs(y))) or 1.0
    return np.asarray(q3 - q1, dtype=float), 1e-9 * scale


def _widest_recrossed_component(
    y: np.ndarray, spread: np.ndarray, flat: float
) -> tuple[int, float] | None:
    """Widest-spread component the orbit *returns to*, or ``None`` if there is none.

    Widest spread first, but only among components the orbit comes back to: a
    coordinate that sweeps past its median once (a drive phase) has the widest
    spread of all and is not a section.
    """
    for candidate in np.argsort(spread)[::-1]:
        if spread[candidate] <= flat:
            return None
        offset = float(np.median(y[:, candidate]))
        if _recrossings(y[:, candidate], offset) >= _MIN_RECROSSINGS:
            return int(candidate), offset
    return None


def _recrossings(column: np.ndarray, offset: float) -> int:
    """Count upward crossings of ``offset`` by ``column``.

    Samples sitting exactly on the offset are dropped rather than counted as a
    side, so a series that touches the plane and retreats does not register as
    two crossings.  Up and down crossings of a *median* differ by at most one, so
    counting one direction is enough whichever way the caller filters.
    """
    side = np.sign(np.asarray(column, dtype=float) - offset)
    side = side[side != 0.0]
    if side.size < 2:
        return 0
    return int(np.count_nonzero((side[:-1] < 0.0) & (side[1:] > 0.0)))


#: Magnitude above which a probe orbit is treated as having escaped rather than
#: settled.  ``isfinite`` is not enough: a start that runs away *slowly* returns a
#: perfectly finite orbit reaching ``1e20`` (measured, SprottF), and its median is
#: a plane the real attractor never comes near — a silently nonsensical section
#: instead of an error.  An **absolute** bound is used rather than a growth ratio
#: because a growth ratio false-rejects a system whose orbit legitimately spirals
#: out from near an equilibrium onto a small attractor (measured, Chua).  It is
#: the same kind of judgement, and for the same reason, as the engine's own
#: ``OVERFLOW_SCALE``: no catalogue attractor lives anywhere near it.
_PROBE_MAGNITUDE_LIMIT = 1e12


def _display_name(system: Any) -> str:
    """Name the subject the way the caller thinks of it (a system class, or "the data")."""
    return "the data" if isinstance(system, Trajectory) else type(system).__name__


def _looks_bounded(y: np.ndarray) -> bool:
    """Whether the probe orbit settled rather than running away over the window."""
    return float(np.max(np.abs(y))) <= _PROBE_MAGNITUDE_LIMIT


def _probe_orbit(system: Any, name: str, stretch: float = 1.0) -> np.ndarray:
    """Integrate a short probe orbit on a **copy**, retrying a diverged start.

    ``stretch`` multiplies the window (and the transient with it), for the second
    look a slow oscillator needs; the sampling interval is unchanged, so a
    stretched probe is proportionally longer and no coarser.
    """
    last: Exception | None = None
    for _ in range(_AUTO_PROBE_ATTEMPTS):
        try:
            traj = system.copy().run(
                final_time=_AUTO_PROBE_FINAL_TIME * stretch,
                dt=_AUTO_PROBE_DT,
                transient=_AUTO_PROBE_TRANSIENT * stretch,
                solver=_probe_method(system),
            )
        except Exception as err:  # noqa: BLE001 - any probe failure means "try again"
            last = err
            continue
        y = np.atleast_2d(np.asarray(traj.y, dtype=float))
        if y.size and np.isfinite(y).all() and _looks_bounded(y):
            return y
    raise _no_plane_error(
        system,
        f"every probe run of {name} failed or diverged"
        + (f" (last: {last})" if last is not None else "")
        + " — if it has no default_ic, the random start may be leaving the attractor's "
        "basin, so reinit(ic) it first",
    ) from last


def _refuse_a_family_with_no_section(system: Any, name: str) -> None:
    """Refuse a subject whose *family* has no continuous crossings, by name.

    A Poincaré section is a statement about a **flow**: it exists because a
    continuous orbit passes *through* a surface.  A discrete map jumps, so it has
    no crossing to refine, and an SDE's path is nowhere differentiable, so it has
    no well-defined transversal.  Both are a category mistake rather than a
    numerical failure — and without this check the probe simply *tries*, and the
    caller is told what the probe tripped over instead (``dt is not a valid
    Henon.run() keyword`` / ``unknown SDE method 'rk4'``) under a heading
    reading "every probe run failed or diverged … reinit(ic) it first", which
    diagnoses an initial condition when the real answer is "not this family".
    """
    from tsdynamics.errors import InvalidParameterError, remedy

    if getattr(system, "_is_discrete", False):
        raise InvalidParameterError(
            f"{name} is a discrete map, so it has no Poincaré section: a section counts the "
            "crossings of a *continuous* orbit through a surface, and a map jumps rather than "
            "crossing. It is already the discrete view a section would produce."
            + remedy(
                f"ts.orbit_diagram({name.lower()}, 'a', values)",
                f"ts.fixed_points({name.lower()})",
                lead="Analyse the map directly:",
            )
        )
    if hasattr(type(system), "_drift"):
        raise InvalidParameterError(
            f"{name} is a stochastic system, so it has no Poincaré section: its path is not "
            "differentiable, so a crossing has no well-defined direction and the refined "
            "crossing point is not reproducible."
            + remedy(
                "ts.poincare_section(deterministic_system, plane=('x', 0.0))",
                lead="Section the deterministic system instead:",
            )
        )


def _no_plane_error(system: Any, why: str) -> Any:
    """Build the "no plane, and none could be chosen" error, naming planes that work."""
    from tsdynamics.errors import InvalidParameterError, remedy

    name = _display_name(system)
    names = getattr(system, "variables", None)
    axis: Any = names[0] if names else 0
    components = tuple(names) if names else tuple(range(int(getattr(system, "dim", 0))))
    return InvalidParameterError(
        f"no section plane was given and one could not be chosen automatically: {why}. "
        f"The components of {name} are {components}, and a plane is (axis, value), "
        "(axis, value, direction) or (normal_vector, offset)."
        + remedy(
            f"ts.poincare_section(system, plane=({axis!r}, 0.0))",
            f"ts.PoincareMap(system, plane=({axis!r}, 0.0, 'down'))",
            lead="Name the section instead:",
        )
    )


def _resolve_section_plane(system: Any, plane: Any, direction: Any) -> tuple[tuple[Any, Any], int]:
    """Resolve a friendly ``plane`` spelling to the raw ``(axis, offset)`` form.

    Normalizes the accepted spellings to the ``(component_index, value)`` /
    ``(normal, offset)`` tuple :meth:`PoincareMap._parse_plane` consumes, and
    resolves the crossing direction:

    - ``None`` — no section named, so one is **chosen** by :func:`auto_plane`
      (and recorded by the caller);
    - ``(axis, offset)`` — ``axis`` is a component **name** (resolved against the
      system's ``variables``) or an integer index;
    - ``(axis, offset, direction)`` — the same, with the crossing direction as
      the third element (``"up"`` / ``"down"`` / ``"both"`` or a sign), which
      then **overrides** the ``direction`` argument;
    - ``(normal, offset)`` — an arbitrary normal **vector** (first element a
      sequence), passed through untouched.

    Returns the resolved ``(axis_or_normal, offset)`` tuple and the integer
    direction in ``{+1, 0, -1}``.  A name on a system without ``variables``, an
    unknown component name, a bad direction word, or a malformed ``plane`` raises
    :class:`~tsdynamics.errors.InvalidParameterError`.
    """
    if plane is None:
        return auto_plane(system), _normalize_direction(direction)
    if not isinstance(plane, (tuple, list)) or len(plane) not in (2, 3):
        raise invalid_value(
            "plane",
            value=plane,
            rule="must be (axis, offset), (axis, offset, direction), or (normal, offset)",
            hint="e.g. plane=('y', 0.0, 'up'), plane=(1, 0.0), or plane=([1, 0, 0], 0.0).",
        )
    axis, offset = plane[0], plane[1]
    if len(plane) == 3:
        direction = plane[2]
    if isinstance(axis, str):
        names = getattr(system, "variables", None)
        if names is None:
            raise invalid_value(
                "plane",
                value=axis,
                rule=f"names a component but {type(system).__name__} declares no `variables`",
                hint="pass an integer component index instead, e.g. plane=(1, 0.0).",
            )
        if axis not in names:
            raise invalid_value(
                "plane",
                value=axis,
                rule=f"is not a declared component of {type(system).__name__}",
                hint=f"choose one of {tuple(names)}.",
            )
        axis = names.index(axis)
    elif np.isscalar(axis):
        # A scalar index (incl. a numpy integer) → plain int, so the resolved
        # ``meta["plane"]`` carries built-in types on every path.  A normal
        # *vector* (a non-scalar sequence) is left untouched for _parse_plane.
        axis = int(axis)  # type: ignore[arg-type]  # np.isscalar guarantees int-able
    return (axis, offset), _normalize_direction(direction)


def _section_jsonable(obj: Any) -> Any:
    """Coerce a value to JSON-friendly types (arrays → lists), for ``to_dict``."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): _section_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_section_jsonable(v) for v in obj]
    return obj


#: Human labels for the resolved crossing direction, shown in :meth:`summary`.
_DIRECTION_LABELS: dict[int, str] = {1: "up (+)", -1: "down (−)", 0: "both"}


class PoincareSection(Trajectory):
    """A Poincaré surface of section — the crossing states, viz-ready.

    A thin :class:`~tsdynamics.data.Trajectory` subclass returned by
    :func:`~tsdynamics.analysis.poincare_section` and
    :meth:`PoincareMap.trajectory`.  It carries the **section intent**
    (:data:`~tsdynamics.viz.spec.PlotKind.POINCARE_SECTION`, in
    ``meta["plot_kind"]``) so a renderer draws the in-plane scatter rather than
    mistaking the full-dimensional crossing states for a flow line, and adds the
    self-describing result surface (:meth:`summary` / :meth:`to_dict`) on top of
    the ordinary trajectory affordances (``.t`` / ``.y`` / named components /
    :meth:`~tsdynamics.data.Trajectory.plot` / ``to_plot_spec``).

    ``t`` holds the continuous crossing times and ``y`` the full-dimensional
    crossing states; the section plane and direction live in ``meta["plane"]`` /
    ``meta["direction"]``.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # noqa: D105
        return f"PoincareSection(crossings={self.n_steps}, dim={self.dim})"

    def summary(self) -> str:
        """Return a human-readable readout: crossing count, dimension, plane, direction."""
        system = self.meta.get("system")
        header = "PoincareSection" + (f"  ({system})" if system else "")
        auto = (
            " (chosen automatically — name a plane to pin it)"
            if self.meta.get("plane_auto")
            else ""
        )
        lines = [
            header,
            f"  crossings = {self.n_steps}",
            f"  dim = {self.dim}",
            f"  plane = {self.meta.get('plane')}{auto}",
        ]
        direction = self.meta.get("direction")
        word = _DIRECTION_LABELS.get(direction) if direction is not None else None
        if word is not None:
            lines.append(f"  direction = {word}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the section (standard library only).

        Carries the crossing times / states, the crossing count, the section
        plane and direction, and the provenance ``meta`` — every value coerced to
        plain ``list`` / ``float`` / ``str`` so :func:`json.dumps` round-trips it.
        """
        return {
            "t": self.t.tolist(),
            "y": self.y.tolist(),
            "n_crossings": int(self.n_steps),
            "plane": _section_jsonable(self.meta.get("plane")),
            "direction": self.meta.get("direction"),
            "meta": _section_jsonable(self.meta),
        }


class PoincareMap(DerivedSystem):
    """
    Present a flow as the discrete map of its crossings through a hyperplane.

    One ``step()`` advances the underlying system until the trajectory
    crosses the section plane in the chosen direction, refines the crossing
    point by cubic Hermite interpolation of the bracketing samples (using the
    system's numeric RHS for endpoint derivatives — O(dt⁴) accuracy), and
    returns the full-dimensional crossing state.

    Because a ``PoincareMap`` *is* a discrete system, everything written for
    maps applies to flows through it — e.g. an orbit diagram over a
    ``PoincareMap`` is a bifurcation diagram of the flow.

    Parameters
    ----------
    system : System
        A continuous-time system (ODE or DDE).
    plane : tuple, optional
        The section, in any of three spellings: ``(axis, c)`` where ``axis`` is
        a component **name** (resolved against the system's ``variables``) or an
        integer index, for the section ``y_axis = c``; ``(axis, c, direction)``,
        the same with the crossing direction (``"up"`` / ``"down"`` / ``"both"``)
        as a third element; or ``(normal, offset)`` with an arbitrary normal
        vector, for the section ``normal · y = offset``.  Examples:
        ``plane=("y", 0.0)``, ``plane=("y", 0.0, "up")``, ``plane=(1, 0.0)``,
        ``plane=([1, 0, 0], 0.0)``.

        **Omit it and one is chosen** — see :func:`auto_plane` — from a short
        probe orbit: the widest-spread component *the orbit keeps returning to*,
        sectioned at its median (an offset the orbit provably straddles, and
        recrosses).  The choice is never silent: it
        lands in :attr:`plane`, in ``section.meta["plane"]`` with
        ``meta["plane_auto"] = True``, and in
        :meth:`PoincareSection.summary`.  Name a plane whenever you have one —
        auto is for exploring, not for a result you are going to publish.
    direction : {+1, -1, 0} or {"up", "down", "both"}
        Count only crossings with ``d(normal·y)/dt > 0`` (``+1`` / ``"up"``,
        default), ``< 0`` (``-1`` / ``"down"``), or both (``0`` / ``"both"``).
        A direction given inside ``plane`` (its third element) overrides this.
    dt : float
        March step used for crossing detection.  The refinement makes the
        crossing itself far more accurate than ``dt``; this only needs to be
        small enough not to skip crossings.
    max_time : float
        Raise if no crossing is found within this much time (e.g. the plane
        misses the attractor).

    Examples
    --------
    >>> pmap = PoincareMap(Rossler(), plane=("x", 0.0, "up"))
    >>> section = pmap.run(500)                # 500 crossings → PoincareSection
    >>> section.y.shape
    (500, 3)
    >>> pmap = PoincareMap(Lorenz())           # no plane named → one is chosen
    >>> pmap.plane_auto                        # ...and the choice is recorded
    True
    """

    def __init__(
        self,
        system: Any,
        plane: tuple[Any, ...] | None = None,
        *,
        direction: int | str = +1,
        dt: float = 0.01,
        max_time: float = 1e4,
    ) -> None:
        super().__init__(system)
        #: Whether :attr:`plane` was chosen by :func:`auto_plane` rather than named.
        self.plane_auto = plane is None
        #: The plane **as the user typed it**, kept beside the resolved
        #: ``(index, offset)``.  Without it two sections on different planes repr
        #: identically, because only the resolved form survived.
        self.plane_given = plane
        plane, direction = _resolve_section_plane(system, plane, direction)
        normal, offset = self._parse_plane(system.dim, plane)
        self.plane = plane
        self._normal = normal
        self._offset = offset
        self.direction = direction
        self.dt = float(dt)
        self.max_time = float(max_time)

        # Numeric RHS for Hermite endpoint derivatives; falls back to linear
        # interpolation (O(dt²)) for systems without one (e.g. DDEs).
        self._rhs = system._rhs_numeric() if hasattr(system, "_rhs_numeric") else None

        self._u_cross: np.ndarray | None = None
        self._t_cross: float | None = None
        self._n_cross = 0

    @staticmethod
    def _parse_plane(dim: int, plane: tuple[Any, ...]) -> tuple[np.ndarray, float]:
        """Lower a resolved ``plane`` to a unit ``(normal, offset)`` geometry.

        Accepts the ``(component_index, value)`` form (an axis-aligned section
        ``y[i] = value``) or the ``(normal, offset)`` form (an arbitrary normal
        **vector** and a scalar offset, normalized to a unit normal here).  A
        malformed tuple, an out-of-range index, or a zero normal raises
        :class:`~tsdynamics.errors.InvalidParameterError` (a ``ValueError``).
        """
        if len(plane) != 2:
            raise invalid_value(
                "plane",
                value=plane,
                rule="must be (component_index, value) or (normal, offset)",
            )
        first, second = plane
        if np.isscalar(first):
            i = int(first)  # type: ignore[arg-type]  # np.isscalar guarantees int-able
            if not 0 <= i < dim:
                raise invalid_value(
                    "plane",
                    value=i,
                    rule=f"component index out of range for dim={dim}",
                )
            normal = np.zeros(dim)
            normal[i] = 1.0
            return normal, float(second)
        normal = np.asarray(first, dtype=float).reshape(dim)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise invalid_value("plane", value=first, rule="normal must be non-zero")
        return normal / norm, float(float(second) / norm)

    def _rebuild(self, inner: Any) -> PoincareMap:
        rebuilt = PoincareMap(
            inner,
            self.plane,
            direction=self.direction,
            dt=self.dt,
            max_time=self.max_time,
        )
        # The re-parametrized map keeps the *same* section (rebuilding is a
        # parameter change, not a new choice), so it also keeps how that section
        # was arrived at — otherwise a swept auto section would report itself as
        # deliberate from the second value onwards.
        rebuilt.plane_auto = self.plane_auto
        rebuilt.plane_given = self.plane_given
        return rebuilt

    # --- section geometry ---

    def _g(self, u: np.ndarray) -> float:
        """Signed distance of state ``u`` from the section plane."""
        return float(self._normal @ u - self._offset)

    def _is_crossing(self, g_prev: float, g_now: float) -> bool:
        up = g_prev < 0.0 <= g_now
        down = g_prev > 0.0 >= g_now
        if self.direction > 0:
            return up
        if self.direction < 0:
            return down
        return up or down

    def _refine(
        self, t0: float, u0: np.ndarray, t1: float, u1: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Locate the crossing inside the bracket [t0, t1]."""
        from scipy.optimize import brentq

        g0, g1 = self._g(u0), self._g(u1)
        if self._rhs is None:
            # Linear interpolation — O(dt²)
            s = g0 / (g0 - g1)
            return t0 + s * (t1 - t0), u0 + s * (u1 - u0)

        # Cubic Hermite on [0, 1] with endpoint derivatives from the RHS — O(dt⁴)
        h = t1 - t0
        f0 = self._rhs(u0, t0)
        f1 = self._rhs(u1, t1)

        def u_at(s: float) -> np.ndarray:
            s2, s3 = s * s, s * s * s
            return cast(
                np.ndarray,
                (2 * s3 - 3 * s2 + 1) * u0
                + (s3 - 2 * s2 + s) * h * f0
                + (-2 * s3 + 3 * s2) * u1
                + (s3 - s2) * h * f1,
            )

        s_star = brentq(lambda s: self._g(u_at(s)), 0.0, 1.0, xtol=1e-14)
        return t0 + s_star * h, u_at(s_star)

    # --- System protocol (discrete view) ---

    @property
    def _is_discrete(self) -> bool:
        """A Poincaré map is a discrete view of the flow."""
        return True

    def _advance_to_crossing(self) -> None:
        sys = self.system
        u_prev = sys.state()
        t_prev = sys.time()
        g_prev = self._g(u_prev)
        deadline = t_prev + self.max_time

        while True:
            u = sys.step(self.dt)
            t = sys.time()
            if not np.isfinite(u).all():
                # A non-finite inner step makes every ``_is_crossing`` NaN-compare
                # false, so the march would otherwise spin the full ``max_time`` and
                # mis-report "plane may miss the attractor".  Surface the real cause.
                raise ConvergenceError(
                    f"PoincareMap: the inner flow diverged (non-finite state) at "
                    f"t={t:g} before a section crossing was found."
                )
            g = self._g(u)
            if self._is_crossing(g_prev, g):
                self._t_cross, self._u_cross = self._refine(t_prev, u_prev, t, u)
                self._n_cross += 1
                return
            if t >= deadline:
                raise ConvergenceError(
                    f"PoincareMap: no section crossing within max_time={self.max_time} "
                    f"(plane may miss the attractor, or direction={self.direction} is wrong)."
                )
            u_prev, t_prev, g_prev = u, t, g

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """Advance to the ``n``-th next crossing and return it (full-dim coords)."""
        n = int(n_or_dt) if n_or_dt is not None else 1
        for _ in range(n):
            self._advance_to_crossing()
        assert self._u_cross is not None  # set by _advance_to_crossing (n >= 1)
        return self._u_cross.copy()

    def state(self) -> np.ndarray:
        """Return the last crossing point (or the inner state before any crossing)."""
        if self._u_cross is not None:
            return self._u_cross.copy()
        return cast(np.ndarray, self.system.state())

    def set_state(self, u: Any) -> None:
        """Overwrite the inner flow state and reset crossing bookkeeping."""
        self.system.set_state(u)
        self._u_cross = None
        self._t_cross = None

    def time(self) -> float:
        """Return the continuous time of the last crossing (inner time before any)."""
        return self._t_cross if self._t_cross is not None else self.system.time()

    @property
    def crossing_count(self) -> int:
        """Return the number of crossings recorded so far."""
        return self._n_cross

    def as_events(self) -> list[Event]:
        """Return the section as a one-element ``[Event]`` for ``system.run(events=...)``.

        A Poincaré section *is* an event: the crossing of ``g(u) = normal·u −
        offset`` in the map's :attr:`direction`.  This exposes it as an
        :class:`~tsdynamics.engine.run.Event` so the general ``events=`` API
        reproduces the section — ``PoincareMap`` is one consumer of the same
        wired engine seam (stream WS-EVENTSAPI / WS-CROSSKERNEL).  Driven at the
        same fixed-step march (``method="rk4"`` at this map's ``dt``) from the
        same initial condition, the crossings of
        ``inner.run(events=pmap.as_events(), ...)`` match
        :meth:`trajectory` to the engine's refinement accuracy.

        Examples
        --------
        >>> pmap = PoincareMap(Rossler(), plane=("y", 0.0, "up"), dt=0.01)
        >>> sol = Rossler().run(final_time=400, dt=0.01, solver="rk4",
        ...                     events=pmap.as_events())
        >>> sol.meta["y_events"][0][:5].shape      # the same crossing states
        (5, 3)
        """
        from tsdynamics.engine.run import Event

        return [Event((self._normal.copy(), self._offset), direction=self.direction)]

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        """Restart the inner flow and clear crossing bookkeeping."""
        self.system.reinit(u, **kwargs)
        # Parameter values are baked into the numeric RHS — rebuild it so the
        # Hermite refinement matches the (possibly re-parametrized) dynamics.
        if hasattr(self.system, "_rhs_numeric"):
            self._rhs = self.system._rhs_numeric()
        self._u_cross = None
        self._t_cross = None
        self._n_cross = 0

    def _python_trajectory(self, steps: int, transient: int) -> tuple[np.ndarray, np.ndarray]:
        """Collect crossings with the per-``dt`` Python march (DDE / no-RHS path)."""
        for _ in range(transient):
            self._advance_to_crossing()
        times = np.empty(steps)
        points = np.empty((steps, self.system.dim))
        for k in range(steps):
            self._advance_to_crossing()
            times[k] = self._t_cross
            points[k] = self._u_cross
        return times, points

    def _engine_trajectory(
        self, steps: int, transient: int, backend: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect crossings with the wired Rust event engine (one call)."""
        be = backend if backend is not None else getattr(self.system, "_default_backend", "jit")
        ic = self.system.state()
        t0 = self.system.time()
        times, points, t_final, u_final = _crossings.section_crossings(
            self.system,
            self._normal,
            self._offset,
            direction=self.direction,
            n_crossings=steps,
            transient=transient,
            dt=self.dt,
            max_time=self.max_time,
            backend=be,
            ic=ic,
            t0=t0,
        )
        # Advance the inner flow past the marched crossings so a subsequent
        # ``step()`` continues forward (the per-``dt`` loop advanced it too); the
        # cursor is the span end, just past the collected crossings.
        self.system.reinit(u_final, t=t_final)
        if steps:
            self._u_cross = np.asarray(points[-1], dtype=float).copy()
            self._t_cross = float(times[-1])
            self._n_cross += transient + steps
        return times, points

    def __repr__(self) -> str:
        """Name the inner system AND the plane — two sections must not repr alike."""
        axis, offset = self.plane
        names = tuple(self.system.variables)
        label = names[axis] if isinstance(axis, int) and 0 <= axis < len(names) else str(axis)
        word = {1: " up", -1: " down"}.get(int(self.direction), "")
        return f"PoincareMap({type(self.system).__name__}, plane={label} = {float(offset):g}{word})"

    def run(
        self,
        steps: int = 100,
        *,
        ic: Any | None = None,
        transient: int = 0,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PoincareSection:
        """
        Collect crossings as a :class:`PoincareSection` — **from a fresh start**.

        ``t`` holds the continuous crossing times; ``y`` the full-dimensional
        crossing states.  ``transient`` crossings are discarded first.  The
        returned :class:`PoincareSection` is a :class:`~tsdynamics.data.Trajectory`
        carrying section intent (so a renderer draws the in-plane scatter) plus a
        ``.summary()`` / ``.to_dict()`` readout.

        For an ordinary (non-stiff) ODE on the compiled engine this marches the
        whole attractor and refines every crossing in **one engine call** (the
        wired Rust ``integrate_events``, stream WS-CROSSKERNEL) — ~100× faster than
        the per-``dt`` Python loop it replaces.  DDEs, systems without a numeric
        RHS, stiff defaults, and ``backend="reference"`` keep the Python loop.  The
        engine path is answer-identical to that loop's fixed-step (``rk4``)
        refinement; see :mod:`tsdynamics.derived._crossings`.

        Parameters
        ----------
        steps : int
            Number of crossings to collect.
        transient : int
            Number of leading crossings to discard.
        backend : {"jit", "interp", "reference"}, optional
            Engine evaluator for the fast path; defaults to the inner system's
            ``_default_backend`` (``"jit"`` for every concrete family since v6).
            ``"reference"`` forces the pure-Python loop.

        Returns
        -------
        PoincareSection
            The collected crossings (continuous times in ``t``, full-dimensional
            states in ``y``), carrying section plot intent and a
            ``.summary()`` / ``.to_dict()`` readout.

        Raises
        ------
        ConvergenceError
            If the inner flow diverges (non-finite state) or no crossing is found
            within ``max_time`` of marching — the plane misses the attractor, or
            the crossing :attr:`direction` is wrong.

        Notes
        -----
        **Live-cursor semantics.**  ``run`` advances the *inner* system as
        a side effect, so a subsequent :meth:`step` continues forward rather than
        re-yielding the crossings just collected.  The two collection paths leave
        the inner cursor at slightly different places: the Python loop stops just
        past the **last collected** crossing, while the engine path stops at the
        **span end** it marched to (which can be a little beyond the last
        crossing).  Both invariants — that ``step()`` resumes *after* the
        collected crossings — hold; do not rely on the exact cursor offset.  Call
        :meth:`reinit` first if you need a deterministic restart point.
        """
        # ``run`` is a fresh run on every family and every wrapper.  Before v6
        # this started from the inner flow's LIVE cursor, so ``pmap.run(5)``
        # twice returned different data; ``step()`` is the verb that continues.
        self.reinit(ic, **kwargs)

        if _crossings.engine_eligible(self.system, backend):
            from tsdynamics.engine.run import EngineNotAvailableError

            try:
                times, points = self._engine_trajectory(steps, transient, backend)
            except EngineNotAvailableError:
                times, points = self._python_trajectory(steps, transient)
        else:
            times, points = self._python_trajectory(steps, transient)

        meta = {
            "derived": "PoincareMap",
            # Section intent, so a renderer draws the 2-D in-plane scatter rather
            # than mistaking the full-dimensional crossing states for a flow line
            # (the string value of viz.PlotKind.POINCARE_SECTION).
            "plot_kind": "poincare_section",
            "plane": self.plane,
            # Recorded so an auto-chosen section is never a silent choice (the
            # same contract orbit_diagram keeps for its auto discrete view).
            "plane_auto": self.plane_auto,
            "direction": self.direction,
            "dt": self.dt,
            "system": type(self.system).__name__,
            "params": self.params.as_dict(),
        }
        return PoincareSection(t=times, y=points, system=self.system, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
