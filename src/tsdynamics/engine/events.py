"""Event subsystem — the ``run(events=[...])`` engine seam.

The crossing/event layer split out of :mod:`tsdynamics.engine.run` (the run-split
refactor).  It owns:

- :func:`crossings` — the low-level engine seam refining crossings of *one* event
  tape over one span (the wired ``tsdyn_engine::integrate_events``, stream
  WS-CROSSKERNEL); :class:`~tsdynamics.derived.poincare.PoincareMap` is its
  consumer via :func:`tsdynamics.derived._crossings.section_crossings`.
- :class:`Event` / :class:`EventSolution` and :func:`integrate_events` — the
  scipy-shaped ``events=[...]`` surface for arbitrary stopping (stream
  WS-EVENTSAPI), driving *both* the compiled engine (one :func:`crossings` per
  event) and the dependency-light :func:`scipy.integrate.solve_ivp` oracle.

Every name here stays reachable as ``tsdynamics.engine.run.<name>`` via re-export,
so this is a pure move.  Run-side helpers (``_engine``/``_primary_tape``/
``_run_continuous``/``resolve_backend``/``EngineNotAvailableError``/``_name``) are
late-imported from :mod:`tsdynamics.engine.run` inside the functions that need
them, so importing this module does not create an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tsdynamics.errors import ConvergenceError
from tsdynamics.utils.grids import make_output_grid

from .problem import ODEProblem, Problem


def crossings(
    problem: Problem,
    g_tape: Any,
    *,
    t0: float,
    t1: float,
    first_step: float,
    direction: int,
    method: str,
    rtol: float,
    atol: float,
    backend: str,
    terminal: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    """Detect crossings of an event function on the Rust engine (one span).

    Marches an ODE problem from ``t0`` to ``t1`` and returns every crossing of the
    single-output event tape ``g_tape`` (``g(u, t) = 0``) in ``direction``, refined
    by the engine's O(h⁴) cubic-Hermite dense output — the wired
    :func:`tsdyn_engine::integrate_events`, the native analogue of
    :meth:`~tsdynamics.derived.poincare.PoincareMap._refine` (stream
    WS-CROSSKERNEL).

    This is the low-level *engine seam* — geometry-free and span-at-a-time. The
    :func:`tsdynamics.derived._crossings.section_crossings` driver builds the plane
    tape, sizes the spans and accumulates the requested number of crossings on top
    of it; :class:`~tsdynamics.derived.poincare.PoincareMap` is its consumer.

    Parameters
    ----------
    problem : Problem
        An :class:`~tsdynamics.engine.problem.ODEProblem` whose ``ic`` is the
        state to start this span from (resume by rebuilding the problem with the
        previous ``u_final``).
    g_tape : Tape
        A single-output event tape over the full state, e.g. built by
        :func:`tsdynamics.engine.compile.lower_expressions` from
        ``normal · u - offset``.
    t0, t1 : float
        The integration span (forward only, ``t1 >= t0``).
    first_step : float
        The solver's first step.  With the fixed-step ``method="rk4"`` it *is* the
        detection step ``dt`` — the engine then marches a ``dt`` grid exactly like
        the Python :class:`PoincareMap`, so the crossings are answer-identical.
    direction : {+1, -1, 0}
        Count rising (``+1``), falling (``-1``) or either (``0``) crossings.
    method, rtol, atol : str, float, float
        Solver configuration, resolved as in :func:`integrate`.
    backend : {"interp", "jit"}
        The engine evaluator.  ``"reference"`` is rejected — the crossing engine is
        compiled-only; callers fall back to the Python loop for the no-engine case.
    terminal : bool, default False
        Stop at the first crossing (a terminal event) instead of collecting all.

    Returns
    -------
    times : ndarray, shape (K,)
        Refined crossing times, increasing.
    states : ndarray, shape (K, dim)
        Full-dimensional crossing states.
    t_final : float
        Time the run stopped (``t1`` or the terminal crossing).
    u_final : ndarray, shape (dim,)
        State at ``t_final`` — the resume point for the next span.
    terminated : bool
        Whether a terminal event stopped the run before ``t1``.

    Raises
    ------
    EngineNotAvailableError
        If the compiled engine is not importable.
    NotImplementedError
        For ``backend="reference"`` or a non-ODE problem.
    RuntimeError
        If the integration diverged or the step collapsed.
    """
    from .run import _engine, _name, _primary_tape, resolve_backend

    name = resolve_backend(backend)
    if name == "reference":
        raise NotImplementedError(
            "the crossing engine is compiled-only (no reference backend); the "
            "Python PoincareMap loop is the fallback for DDE / no-RHS / reference "
            "systems."
        )
    if not isinstance(problem, ODEProblem):
        raise NotImplementedError(
            f"crossings() integrates ODE problems only, not {problem.family!r}; "
            f"DDEs and SDEs use the Python crossing fallback."
        )
    eng = _engine()
    times, states, t_final, u_final, terminated = eng.integrate_events_dense(
        *_primary_tape(problem).to_arrays(),
        *g_tape.to_arrays(),
        np.ascontiguousarray(problem.ic, dtype=np.float64),
        problem.params_vec(),
        float(t0),
        float(t1),
        float(first_step),
        int(direction),
        bool(terminal),
        method,
        float(rtol),
        float(atol),
        name == "jit",
    )
    times = np.asarray(times, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    u_final = np.asarray(u_final, dtype=np.float64)
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(states))):
        raise ConvergenceError(
            f"{_name(problem)}: crossing detection diverged or produced non-finite "
            f"values before reaching the span end."
        )
    return times, states, float(t_final), u_final, bool(terminated)


# ---------------------------------------------------------------------------
# Event-aware integration — the general ``system.run(events=[...])`` API
# ---------------------------------------------------------------------------
#
# The :func:`crossings` seam refines crossings of *one* event tape over one
# span; this layer raises it to the scipy-shaped ``events=[...]`` surface a user
# (or an analysis — A-RQA, A-BASIN, arbitrary stopping) reaches for: several
# events at once, each with its own crossing ``direction`` and an optional
# ``terminal`` flag that stops the integration at the first crossing.
#
# An event is a *symbolic* scalar condition ``g(u, t) = 0`` so it lowers to the
# same one-output tape the Rust event engine watches (the
# :class:`~tsdynamics.derived.poincare.PoincareMap` section is one such event —
# see :meth:`PoincareMap.as_events`).  Because it is symbolic it serves *both*
# paths from one spec: the compiled engine (``integrate_events_dense`` per event)
# and the dependency-light :func:`scipy.integrate.solve_ivp` ``events=`` oracle —
# two independent implementations that the test-suite cross-checks.

#: Friendly spellings for the crossing-direction filter (kept byte-identical to
#: :data:`tsdynamics.derived.poincare._DIRECTION_WORDS`; duplicated here so the
#: engine layer stays below ``derived`` in the import graph).
_DIRECTION_WORDS: dict[str, int] = {
    "up": +1,
    "increasing": +1,
    "positive": +1,
    "+": +1,
    "down": -1,
    "decreasing": -1,
    "negative": -1,
    "-": -1,
    "both": 0,
    "either": 0,
    "all": 0,
}

#: Explicit kernels the compiled event engine drives; an implicit / stiff method
#: (``bdf``/``rosenbrock``/``trbdf2``) routes to the SciPy oracle instead (its
#: event root-finder over the dense output is the dependency-light fallback).
_EVENT_ENGINE_METHODS: frozenset[str] = frozenset({"rk4", "rk45", "tsit5", "dop853"})


def _normalize_event_direction(direction: Any) -> int:
    """Coerce a crossing-direction spec to one of ``{+1, 0, -1}``.

    Accepts the friendly words ``"up"`` / ``"down"`` / ``"both"`` (and a few
    synonyms) or any signed number (its sign is taken).
    """
    if isinstance(direction, str):
        from tsdynamics.errors import invalid_value

        key = direction.strip().lower()
        if key not in _DIRECTION_WORDS:
            raise invalid_value(
                "direction",
                value=direction,
                rule="must be 'up', 'down', or 'both'",
                hint="or pass a signed number: +1 (up), -1 (down), 0 (both).",
            )
        return _DIRECTION_WORDS[key]
    return int(np.sign(direction))


@dataclass
class Event:
    """A scipy-shaped event condition for :meth:`ContinuousSystem.run`'s ``events=``.

    An event watches a scalar condition ``g(u, t)`` along the flow and records
    every zero-crossing in the chosen :attr:`direction`; a :attr:`terminal` event
    stops the integration at its first crossing (the *arbitrary-stopping* use
    case).  The condition is **symbolic** so the *same* spec drives the compiled
    Rust event engine and the SciPy fallback.

    Parameters
    ----------
    condition : callable or tuple
        Either a symbolic callable ``g(y, t)`` (or ``g(y, t, **params)``) over the
        engine state accessor — ``y(i)`` is component ``i`` — returning **one**
        SymEngine expression (e.g. ``lambda y, t: y(2) - 27.0``); or a plane
        spelling shared with :class:`~tsdynamics.derived.poincare.PoincareMap`:
        ``(axis, offset)`` / ``(axis, offset, direction)`` with ``axis`` a
        component **name** (resolved against the system's ``variables``) or an
        integer index, or ``(normal, offset)`` with an arbitrary normal vector.
    direction : {+1, -1, 0} or {"up", "down", "both"}, default 0
        Keep only rising (``+1`` / ``"up"``), falling (``-1`` / ``"down"``) or
        either (``0``) crossings.  A direction given inside a plane tuple (its
        third element) overrides this.
    terminal : bool, default False
        Stop the integration at this event's first crossing.
    name : str, optional
        A label carried into the trajectory provenance.

    Notes
    -----
    A bare callable carrying ``.direction`` / ``.terminal`` attributes (the SciPy
    convention) and a plane tuple are both accepted directly in ``events=[...]``
    and coerced via :meth:`coerce`, so the class is optional sugar.

    Examples
    --------
    >>> from tsdynamics.engine.run import Event
    >>> above = Event(lambda y, t: y(2) - 27.0, direction="up")
    >>> escape = Event(("x", 1e3), terminal=True)        # stop if x crosses 1000
    """

    condition: Any
    direction: Any = 0
    terminal: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        self.terminal = bool(self.terminal)
        if isinstance(self.condition, (tuple, list)):
            # Plane spelling: pull a direction word out of a 3-tuple (it overrides
            # ``direction=``), and store the normalized 2-tuple ``(axis, offset)``.
            spec = tuple(self.condition)
            if len(spec) not in (2, 3):
                from tsdynamics.errors import invalid_value

                raise invalid_value(
                    "event condition",
                    value=self.condition,
                    rule="plane must be (axis, offset), (axis, offset, direction) or (normal, offset)",
                    hint="or pass a callable g(y, t) returning a SymEngine expression.",
                )
            if len(spec) == 3:
                self.direction = spec[2]
                self.condition = (spec[0], spec[1])
            else:
                self.condition = spec
        elif not callable(self.condition):
            from tsdynamics.errors import InvalidInputError

            raise InvalidInputError(
                f"event condition must be a callable g(y, t) or a plane tuple, "
                f"got {type(self.condition).__name__}."
            )
        self.direction = _normalize_event_direction(self.direction)

    @classmethod
    def coerce(cls, spec: Any) -> Event:
        """Return ``spec`` as an :class:`Event`, accepting the SciPy conventions.

        Passes an :class:`Event` through; reads ``.direction`` / ``.terminal``
        off a bare callable (the SciPy convention); and builds a plane event from
        a tuple.
        """
        if isinstance(spec, Event):
            return spec
        if callable(spec):
            return cls(
                spec,
                direction=getattr(spec, "direction", 0) or 0,
                terminal=bool(getattr(spec, "terminal", False)),
                name=getattr(spec, "__name__", None),
            )
        return cls(spec)

    def _expr(self, system: Any, y: Any, t_sym: Any, control_syms: dict[str, Any]) -> Any:
        """Build the SymEngine condition expression for ``system``."""
        import symengine

        cond = self.condition
        if callable(cond):
            import inspect

            try:
                params = inspect.signature(cond).parameters.values()
                takes_params = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
            except (TypeError, ValueError):  # un-introspectable callable
                takes_params = False
            if takes_params:
                return cond(y, t_sym, **control_syms)
            return cond(y, t_sym)
        axis_or_normal, offset = cond
        dim = int(system.dim)
        if np.isscalar(axis_or_normal):
            idx = _resolve_event_axis(system, axis_or_normal)
            return y(idx) - float(offset)
        normal = np.asarray(axis_or_normal, dtype=float).reshape(dim)
        expr: Any = symengine.sympify(0)
        for i in range(dim):
            coeff = float(normal[i])
            if coeff != 0.0:
                expr = expr + coeff * y(i)
        return expr - float(offset)


def _resolve_event_axis(system: Any, axis: Any) -> int:
    """Resolve a plane ``axis`` (component name or index) to an integer index."""
    from tsdynamics.errors import invalid_value

    dim = int(system.dim)
    if isinstance(axis, str):
        names = getattr(system, "variables", None)
        if not names or axis not in names:
            raise invalid_value(
                "event axis",
                value=axis,
                rule=f"names a component but {type(system).__name__} has no such variable",
                hint=f"choose one of {tuple(names)} or pass an integer index."
                if names
                else "pass an integer component index instead.",
            )
        return int(names.index(axis))
    idx = int(axis)
    if not 0 <= idx < dim:
        raise invalid_value(
            "event axis",
            value=axis,
            rule=f"is out of range for dim={dim}",
            hint=f"use an index in [0, {dim - 1}].",
        )
    return idx


@dataclass(frozen=True)
class EventSolution:
    """Result of an event-aware integration (the transport behind ``run(events=)``).

    Mirrors the SciPy ``solve_ivp`` event shape: a dense trajectory (``t`` / ``y``,
    truncated at the first terminal crossing) plus ``t_events`` / ``y_events`` —
    one array per input event, holding that event's crossing times / states.
    """

    t: np.ndarray
    y: np.ndarray
    t_events: list[np.ndarray]
    y_events: list[np.ndarray]
    terminated: bool
    events: list[Event] = field(default_factory=list)


def _event_tape(system: Any, event: Event) -> Any:
    """Lower an event's condition ``g(u, t)`` to a single-output tape.

    Mirrors :func:`tsdynamics.engine.compile.lower_ode`: the condition is built
    over the engine state accessor ``y(i)``, the time symbol and the system's
    control-parameter symbols, then ``y(i)`` is substituted to plain state inputs
    and the expression is lowered.  Declaring the *full* control-parameter list
    (in ``system._control_params()`` order, matching ``problem.params_vec()``)
    lets the shared parameter vector feed the event tape correctly even for a
    parameter-dependent condition.
    """
    import symengine

    from tsdynamics.engine.compile import lower_expressions
    from tsdynamics.engine.symbols import state_time_symbols

    y, t_sym = state_time_symbols()
    dim = int(system.dim)
    control_names = list(system._control_params())
    control_syms = {k: symengine.Symbol(k) for k in control_names}

    expr = event._expr(system, y, t_sym, control_syms)

    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    t_canon = symengine.Symbol("t")
    subs = {y(i): u_syms[i] for i in range(dim)}
    subs[t_sym] = t_canon
    expr = symengine.sympify(expr).subs(subs)

    return lower_expressions(
        [expr],
        u_syms,
        param_syms=[control_syms[k] for k in control_names],
        time_sym=t_canon,
        control_names=control_names,
    )


def integrate_events(
    problem: Problem,
    events: Any,
    *,
    final_time: float = 100.0,
    dt: float = 0.02,
    t0: float | None = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    backend: str = "interp",
) -> EventSolution:
    """Integrate an ODE while detecting a list of events — the ``events=`` engine seam.

    Each event's crossings are refined to the engine's O(h⁴) accuracy on the
    compiled backend (one :func:`crossings` call per event), or by SciPy's event
    root-finder on the ``"reference"`` / stiff fallback.  A terminal event stops
    the integration at its first crossing; the dense trajectory is then sampled
    on ``[t0, t_stop]`` at ``dt``.

    Parameters
    ----------
    problem : ODEProblem
        The lowered ODE whose ``ic`` is the start state.
    events : sequence
        :class:`Event` instances, bare ``g(y, t)`` callables (SciPy convention),
        or plane tuples — each coerced via :meth:`Event.coerce`.
    final_time, dt, t0, method, rtol, atol, backend
        As in :func:`integrate` (``method`` is resolved by the solver registry).

    Returns
    -------
    EventSolution

    Raises
    ------
    InvalidInputError
        If ``problem`` is not an ODE, or ``events`` is empty.
    """
    from tsdynamics.errors import InvalidInputError

    from .run import resolve_backend

    backend = resolve_backend(backend)
    if not isinstance(problem, ODEProblem):
        raise InvalidInputError(
            f"events= is an ODE feature; got a {problem.family!r} problem. "
            f"(Maps have no continuous crossings; DDEs/SDEs are not supported.)"
        )
    evs = [Event.coerce(e) for e in events]
    if not evs:
        from tsdynamics.errors import invalid_value

        raise invalid_value(
            "events",
            value=events,
            rule="must be a non-empty sequence of events",
            hint="pass at least one Event, g(y, t) callable, or plane tuple.",
        )

    from tsdynamics import solvers

    method = solvers.resolve(method).name
    system = problem.system
    start = problem.t0 if t0 is None else float(t0)
    event_tapes = [_event_tape(system, e) for e in evs]

    if backend in ("interp", "jit") and method in _EVENT_ENGINE_METHODS:
        from .run import EngineNotAvailableError

        try:
            return _engine_events(
                problem,
                evs,
                event_tapes,
                final_time=float(final_time),
                dt=float(dt),
                t0=start,
                method=method,
                rtol=rtol,
                atol=atol,
                backend=backend,
            )
        except EngineNotAvailableError:
            pass  # fall through to the dependency-light SciPy oracle
    return _reference_events(
        problem,
        evs,
        event_tapes,
        final_time=float(final_time),
        dt=float(dt),
        t0=start,
        method=method,
        rtol=rtol,
        atol=atol,
    )


def _engine_events(
    problem: ODEProblem,
    evs: list[Event],
    event_tapes: list[Any],
    *,
    final_time: float,
    dt: float,
    t0: float,
    method: str,
    rtol: float,
    atol: float,
    backend: str,
) -> EventSolution:
    """Collect each event's crossings on the compiled engine + a dense trajectory.

    Terminal events are probed first (``terminal=True``) to find the earliest
    stop ``t_stop``; every event's crossings are then gathered over ``[t0,
    t_stop]`` and the dense trajectory integrated on that span — so the returned
    trajectory truncates exactly like SciPy's, and the crossings lie on it.
    """
    from .run import _run_continuous

    dim = int(problem.dim)
    n = len(evs)

    # Phase 1 — earliest terminal crossing fixes the stop time.
    t_stop = float(final_time)
    terminated = False
    terminal_cross: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, (e, gt) in enumerate(zip(evs, event_tapes, strict=True)):
        if not e.terminal:
            continue
        times, states, t_final, _u_final, term = crossings(
            problem,
            gt,
            t0=t0,
            t1=final_time,
            first_step=dt,
            direction=e.direction,
            method=method,
            rtol=rtol,
            atol=atol,
            backend=backend,
            terminal=True,
        )
        if term and times.size:
            terminal_cross[i] = (times[:1], states[:1])
            if float(times[0]) < t_stop:
                t_stop = float(times[0])
                terminated = True
    # Drop terminal crossings that would have fired *after* the actual stop.
    for i in [k for k, (tc, _) in terminal_cross.items() if float(tc[0]) > t_stop + 1e-12]:
        del terminal_cross[i]

    # Phase 2 — every event's crossings over the integrated span [t0, t_stop].
    t_events: list[np.ndarray] = [np.empty(0) for _ in range(n)]
    y_events: list[np.ndarray] = [np.empty((0, dim)) for _ in range(n)]
    for i, (e, gt) in enumerate(zip(evs, event_tapes, strict=True)):
        if e.terminal:
            if i in terminal_cross:
                tc, sc = terminal_cross[i]
                t_events[i] = np.asarray(tc, dtype=float)
                y_events[i] = np.asarray(sc, dtype=float).reshape(-1, dim)
            continue
        times, states, _t_final, _u_final, _term = crossings(
            problem,
            gt,
            t0=t0,
            t1=t_stop,
            first_step=dt,
            direction=e.direction,
            method=method,
            rtol=rtol,
            atol=atol,
            backend=backend,
            terminal=False,
        )
        t_events[i] = np.asarray(times, dtype=float)
        y_events[i] = np.asarray(states, dtype=float).reshape(-1, dim)

    # Phase 3 — the dense trajectory on the (possibly truncated) span.
    t_eval = make_output_grid(t0, t_stop, dt)
    y = _run_continuous(problem, t_eval, method=method, rtol=rtol, atol=atol, backend=backend)
    return EventSolution(
        t=t_eval, y=y, t_events=t_events, y_events=y_events, terminated=terminated, events=evs
    )


def _reference_events(
    problem: ODEProblem,
    evs: list[Event],
    event_tapes: list[Any],
    *,
    final_time: float,
    dt: float,
    t0: float,
    method: str,
    rtol: float,
    atol: float,
) -> EventSolution:
    """Detect events with SciPy's ``solve_ivp(events=...)`` — the wheel-free oracle.

    A faithful, *independent* implementation of the engine path (SciPy's Brent
    root-finder over its own dense output): the events test-suite cross-checks the
    two.  Reuses the reference ODE RHS (the lowered tape evaluated in pure
    Python) and wraps each event tape as a SciPy event callable carrying the
    SciPy ``.terminal`` / ``.direction`` attributes.
    """
    from scipy.integrate import solve_ivp

    from .compile import eval_tape
    from .reference import _scipy_method

    dim = int(problem.dim)
    tape = problem.tape
    p = problem.params_vec()
    ic = np.asarray(problem.ic, dtype=np.float64).ravel()
    scipy_method = _scipy_method(method)

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        return eval_tape(tape, u, p, t)

    def make_event(g_tape: Any) -> Any:
        def gfun(t: float, u: np.ndarray) -> float:
            return float(eval_tape(g_tape, u, p, t)[0])

        return gfun

    scipy_events = []
    for e, gt in zip(evs, event_tapes, strict=True):
        f = make_event(gt)
        f.terminal = e.terminal
        f.direction = e.direction
        scipy_events.append(f)

    t_eval = make_output_grid(t0, final_time, dt)
    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        ic,
        t_eval=t_eval,
        method=scipy_method,
        rtol=rtol,
        atol=atol,
        events=scipy_events,
    )
    if not sol.success:
        raise ConvergenceError(f"reference event integration failed: {sol.message}")

    t_events = [np.asarray(te, dtype=float) for te in sol.t_events]
    y_events = [np.asarray(ye, dtype=float).reshape(-1, dim) for ye in sol.y_events]
    terminated = any(e.terminal and t_events[i].size for i, e in enumerate(evs))
    return EventSolution(
        t=np.asarray(sol.t, dtype=float),
        y=np.ascontiguousarray(sol.y.T),
        t_events=t_events,
        y_events=y_events,
        terminated=terminated,
        events=evs,
    )
