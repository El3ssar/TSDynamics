"""Run entry points — compile a system, pick a backend, integrate.

This is the top of the engine seam: the functions a user (or the family base
classes, stream C-FAM) call to actually *run* a system on the Rust engine.
Each entry point turns a system into a :class:`~tsdynamics.engine.problem.Problem`
(:mod:`tsdynamics.engine.problem`), resolves a backend, and dispatches.

Backends
--------
The two production evaluators behind the frozen ``Evaluator`` seam are selected
by name:

- ``"interp"`` — the zero-warmup tape interpreter (``tsdyn-vm``).  The default:
  no compilation latency, ideal for sweeps, tests, and small/medium systems.
- ``"jit"`` — the Cranelift native-code evaluator (``tsdyn-jit``), for large or
  long-running problems.  Numerically identical to the interpreter.
- ``"auto"`` — resolves to ``"interp"`` today (a size/run-length heuristic for
  promoting to ``"jit"`` is a later refinement, ROADMAP §11).

Both run inside the compiled extension :mod:`tsdynamics._rust` (stream E7).
Until that extension is built, ``"interp"``/``"jit"`` raise
:class:`EngineNotAvailableError`.  A third backend, ``"reference"``, needs no compiled
engine: it evaluates the lowered tape in pure Python (the
:mod:`~tsdynamics.engine.compile` reference evaluator) and delegates ODE
time-stepping to SciPy — the dependency-light oracle the lowering is validated
against, and a usable fallback for RHS evaluation and small ODE/map runs.

What this module does *not* own: the solver kernels and the integrate loop
(those are the Rust ``tsdyn-solvers`` / ``tsdyn-engine`` crates) and the FFI
marshalling (``tsdyn-core`` → :mod:`tsdynamics._rust`, stream E7).  It owns the
Python-side orchestration: problem construction, backend resolution, and
wrapping the engine's output as a :class:`~tsdynamics.families.Trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tsdynamics.errors import BackendError, ConvergenceError
from tsdynamics.utils.grids import make_output_grid

if TYPE_CHECKING:
    from tsdynamics.families import Trajectory

from .compile import eval_tape, eval_tape_jac
from .problem import (
    DDEProblem,
    MapProblem,
    ODEProblem,
    Problem,
    SDEProblem,
    build_problem,
)

__all__ = [
    "BACKENDS",
    "EngineNotAvailableError",
    "Event",
    "EventSolution",
    "crossings",
    "ensemble",
    "eval_jac",
    "eval_rhs",
    "integrate",
    "integrate_events",
    "make_ode_stepper",
    "resolve_backend",
    "sde_ensemble_final",
    "sde_integrate_dense",
    "step_advance",
    "step_advance_to_event",
]

#: The selectable backend names.  ``"interp"`` / ``"jit"`` run on the compiled
#: engine; ``"reference"`` is the pure-Python oracle/fallback.
BACKENDS: frozenset[str] = frozenset({"interp", "jit", "reference"})


class EngineNotAvailableError(BackendError):
    """The compiled Rust engine (:mod:`tsdynamics._rust`, stream E7) is not available.

    Raised when an ``"interp"`` / ``"jit"`` run is requested but the extension
    module is not importable.  Use ``backend="reference"`` for pure-Python ODE/map
    runs, or reinstall the compiled wheel (``pip install tsdynamics``).

    Subclasses :class:`~tsdynamics.errors.BackendError` (itself a
    :class:`RuntimeError`), so it joins the documented ``TSDynamicsError``
    hierarchy while existing ``except RuntimeError`` handlers keep catching it.
    """


def resolve_backend(backend: str) -> str:
    """Normalise a backend name to one of :data:`BACKENDS`.

    ``"auto"`` resolves to ``"interp"`` (the zero-warmup default; a heuristic
    promotion to ``"jit"`` is a later refinement).

    Parameters
    ----------
    backend : str
        ``"interp"``, ``"jit"``, ``"reference"``, or ``"auto"``.

    Returns
    -------
    str
        A canonical name in :data:`BACKENDS`.

    Raises
    ------
    InvalidParameterError
        If ``backend`` is not a recognised name.  Subclasses :class:`ValueError`,
        so existing ``except ValueError`` handlers keep catching it.
    """
    name = str(backend).lower()
    if name == "auto":
        return "interp"
    if name not in BACKENDS:
        from tsdynamics.errors import invalid_value

        raise invalid_value(
            "backend",
            backend,
            options=[*sorted(BACKENDS), "auto"],
            hint="'interp'/'jit' run on the Rust engine; 'reference' is the pure-Python oracle.",
        )
    return name


def _engine() -> Any:
    """Return the compiled engine module, or raise :class:`EngineNotAvailableError`.

    Indirected through this single accessor so callers (and tests) have one seam
    to the FFI surface — the engine binding is whatever
    :mod:`tsdynamics._rust` (stream E7) exposes.
    """
    try:
        from tsdynamics import _rust
    except ImportError as err:  # pragma: no cover - until E7 builds the wheel
        raise EngineNotAvailableError(
            "the Rust engine extension (tsdynamics._rust) is not built; "
            "use backend='reference' for pure-Python ODE/map evaluation, or "
            "reinstall the compiled wheel (`pip install tsdynamics`)."
        ) from err
    return _rust


# ---------------------------------------------------------------------------
# Problem coercion
# ---------------------------------------------------------------------------


def _as_problem(obj: Any, **build_kwargs: Any) -> Problem:
    """Return ``obj`` if it is already a Problem, else build one from a system."""
    if isinstance(obj, ODEProblem | MapProblem | DDEProblem | SDEProblem):
        return obj
    return build_problem(obj, **build_kwargs)


# ---------------------------------------------------------------------------
# Pointwise RHS / Jacobian evaluation (runnable today)
# ---------------------------------------------------------------------------


def eval_rhs(
    system_or_problem: Any,
    u: Any,
    t: float = 0.0,
    *,
    backend: str = "reference",
    **build_kwargs: Any,
) -> np.ndarray:
    """Evaluate ``du/dt`` (or the next state, for a map) once at ``(u, t)``.

    The strongest, tolerance-tight signal that a system *lowers* correctly — it
    has no chaotic-divergence caveat.  With ``backend="reference"`` (default)
    the lowered tape is evaluated in pure Python; with ``"interp"``/``"jit"`` the
    compiled engine evaluates it (raising :class:`EngineNotAvailableError` until the
    extension is built).

    Parameters
    ----------
    system_or_problem : SystemBase or Problem
        The system to lower (any extra ``build_kwargs`` go to the Problem
        factory) or an already-built Problem.
    u : array-like, shape (dim,)
        State to evaluate at.  For an SDE this evaluates the drift.
    t : float, default 0.0
        Time.
    backend : str, default "reference"
        ``"reference"``, ``"interp"``, or ``"jit"``.

    Returns
    -------
    ndarray, shape (dim,)
    """
    backend = resolve_backend(backend)
    problem = _as_problem(system_or_problem, **build_kwargs)
    tape = _primary_tape(problem)
    if backend == "reference":
        return eval_tape(tape, u, problem.params_vec(), float(t))
    eng = _engine()
    arrays = tape.to_arrays()
    return np.asarray(
        eng.eval_rhs(
            *arrays, np.asarray(u, dtype=np.float64).ravel(), problem.params_vec(), float(t)
        )
    )


def eval_jac(
    system_or_problem: Any,
    u: Any,
    t: float = 0.0,
    *,
    backend: str = "reference",
    **build_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``(du/dt, Jacobian)`` once at ``(u, t)``.

    Requires a Problem whose tape carries a Jacobian (build with
    ``with_jacobian=True``).  With ``backend="reference"`` the lowered tape is
    evaluated in pure Python.

    Returns
    -------
    (ndarray, ndarray)
        The derivative ``(dim,)`` and the row-major ``(dim, dim)`` Jacobian.
    """
    backend = resolve_backend(backend)
    problem = _as_problem(system_or_problem, with_jacobian=True, **build_kwargs)
    tape = _primary_tape(problem)
    if backend == "reference":
        return eval_tape_jac(tape, u, problem.params_vec(), float(t))
    eng = _engine()
    arrays = tape.to_arrays()
    d, j = eng.eval_jac(
        *arrays, np.asarray(u, dtype=np.float64).ravel(), problem.params_vec(), float(t)
    )
    return np.asarray(d), np.asarray(j).reshape(tape.dim, tape.dim)


def _primary_tape(problem: Problem) -> Any:
    """Return the RHS/next-state tape of a problem (the drift tape for an SDE)."""
    if isinstance(problem, SDEProblem):
        return problem.drift
    return problem.tape


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def _recommend_method(problem: Problem) -> Any:
    """Resolve ``method="auto"`` to a kernel by a-priori auto-stiffness selection.

    The auto-stiffness wiring (ticket FIX-AUTOSTIFF): probe the problem's Jacobian
    spectrum at its start state and let the solver registry recommend an implicit
    kernel (``bdf``) on a stiff RHS or the explicit default (``rk45``) otherwise
    (:func:`tsdynamics.solvers.recommend`).  The probe is the one-point heuristic
    :func:`tsdynamics.solvers.is_stiff` — useful but not a guarantee (see its
    docstring): the verdict is read at the integration's *own* start state
    (``problem.ic``), so a system declares a stiff default via ``_default_method``
    when the heuristic is too coarse for it.

    Maps iterate without a solver kernel, so ``"auto"`` is a no-op there: it
    resolves to the explicit ODE default the map branch ignores (keeping a literal
    ``method="auto"`` from raising on a map rather than meaning anything).

    Parameters
    ----------
    problem : Problem
        The already-lowered problem (built once by :func:`integrate`).

    Returns
    -------
    solvers.Resolution
        The recommended kernel, carrying its ``build_kwargs``.
    """
    from tsdynamics import solvers

    if isinstance(problem, MapProblem):
        return solvers.resolve(solvers.DEFAULT_METHOD["ode"])
    return solvers.recommend(
        problem.system,
        family=problem.family,
        ic=problem.ic,
        t=getattr(problem, "t0", 0.0),
    )


def _resolve_method_for(method: str, problem: Problem) -> Any:
    """Resolve ``method=`` to a :class:`solvers.Resolution` for an already-built problem.

    The single home for the ``method=`` contract shared by :func:`integrate` and
    :func:`ensemble`, so ``"auto"`` means the same thing through both entry points
    (diagnosis #11/#13).  A literal ``method="auto"`` triggers a-priori
    auto-stiffness selection over *problem* (:func:`_recommend_method` →
    :func:`solvers.recommend`); every other name normalises spellings/aliases
    ("RK45" → "rk45", "dopri5" → "rk45") and rejects unknown or v2-only names
    ("LSODA") with a listing error + a stiff hint.  Resolving is harmless for the
    families that ignore ``method=`` (maps fold params in; SDE/DDE batches are
    refused by the caller before stepping).
    """
    from tsdynamics import solvers

    if solvers.normalize(method) == "auto":
        return _recommend_method(problem)
    return solvers.resolve(method)


def integrate(
    system_or_problem: Any,
    *,
    final_time: float = 100.0,
    dt: float = 0.02,
    t0: float | None = None,
    ic: Any = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    backend: str = "interp",
    history: Any = None,
    **build_kwargs: Any,
) -> Trajectory:
    """Integrate a system on the engine and return a :class:`~tsdynamics.families.Trajectory`.

    The single family-polymorphic engine entry point: every family's engine path
    (ODE, map and DDE) funnels here, so FFI marshalling, divergence guards and
    engine-path provenance live in one place rather than being re-implemented per
    family.  The family is detected from the built :class:`Problem` and dispatched
    to the matching runner.  Diagonal-Itô **SDEs are the exception** — the generic
    seam cannot carry their noise seed and step-as-noise-scale, so they are
    refused here (use :func:`sde_integrate_dense`).

    Parameters
    ----------
    system_or_problem : SystemBase or Problem
        The system (lowered to a Problem here) or a pre-built Problem.
    final_time : float, default 100.0
        End of the integration window.  For a map this is the (rounded) number of
        iterations.
    dt : float, default 0.02
        Output sampling interval (the internal stepper is adaptive for adaptive
        methods).  Ignored for a map.
    t0 : float, optional
        Start time; defaults to the Problem's ``t0``.
    ic : array-like, optional
        Initial state (only used when building a Problem from a system).
    method : str, default "RK45"
        Solver name, resolved by the solver registry (stream C-SOLV).  The
        special value ``"auto"`` selects a kernel by a-priori auto-stiffness:
        the Jacobian spectrum at the start state is probed and an implicit
        kernel (``bdf``) is used on a stiff RHS, the explicit default (``rk45``)
        otherwise (:func:`tsdynamics.solvers.recommend`; a one-point heuristic).
    rtol, atol : float
        Solver tolerances.
    backend : str, default "interp"
        ``"interp"``, ``"jit"`` (compiled engine), or ``"reference"``
        (pure-Python; ODE and map only).
    history : callable, optional
        For a DDE only — ``h(s) -> sequence`` defining the past for ``s <= 0``;
        ``None`` is a constant past equal to the resolved initial state.

    Returns
    -------
    Trajectory

    Raises
    ------
    EngineNotAvailableError
        For ``"interp"``/``"jit"`` when :mod:`tsdynamics._rust` is not built.
    NotImplementedError
        For ``backend="reference"`` on a DDE or SDE (the reference integrator
        covers ODEs and maps only), and for any SDE problem (the generic seam
        cannot carry the noise seed/step — use :func:`sde_integrate_dense`).
    """
    from tsdynamics.families import Trajectory

    backend = resolve_backend(backend)

    # Resolve the solver name to a canonical engine kernel (the C-SOLV → C-FAM
    # wiring), via the shared :func:`_resolve_method_for` contract (so ``"auto"``
    # behaves identically in :func:`ensemble`).  A literal ``method="auto"``
    # triggers a-priori **auto-stiffness** selection: the problem is lowered first
    # so the Jacobian spectrum at the start state can be probed
    # (``solvers.recommend`` → ``_recommend_method``), picking the implicit ``bdf``
    # kernel on a stiff RHS and the explicit ``rk45`` otherwise.  Every other name
    # normalises spellings/aliases (e.g. "RK45" → "rk45", "dopri5" → "rk45") and
    # rejects unknown or v2-only names (e.g. "LSODA") with a listing error + a
    # stiff hint pointing at "bdf".  Maps ignore `method` and an SDE problem is
    # refused below, so resolving is harmless there.
    problem = _as_problem(system_or_problem, ic=ic, **build_kwargs)
    resolution = _resolve_method_for(method, problem)
    method = resolution.name

    # The implicit ODE kernels (bdf/rosenbrock/trbdf2) need ∂f/∂u on the tape, or
    # the engine refuses the step.  Rebuild once with the Jacobian when the
    # resolved method needs it and the tape lacks it — only for an ODE built from
    # a system (DDEs drive explicit kernels only; maps fold params in; a pre-built
    # Problem we cannot re-lower, so the engine raises its clear guard instead).
    if (
        resolution.build_kwargs.get("with_jacobian")
        and isinstance(problem, ODEProblem)
        and not problem.tape.has_jacobian
        and "with_jacobian" not in build_kwargs
        and not isinstance(system_or_problem, ODEProblem)
    ):
        problem = _as_problem(system_or_problem, ic=ic, with_jacobian=True, **build_kwargs)

    if isinstance(problem, MapProblem):
        steps = int(round(final_time))
        t_arr, y = _run_map(problem, steps, backend)
        meta = _provenance(
            problem,
            backend=backend,
            steps=steps,
            ic=np.asarray(problem.ic, dtype=np.float64).copy(),
        )
    else:
        start = problem.t0 if t0 is None else float(t0)
        t_eval = make_output_grid(start, final_time, dt)
        t_arr = t_eval
        if isinstance(problem, DDEProblem):
            y, ic0 = _run_dde(
                problem,
                t_eval,
                history=history,
                dt=dt,
                method=method,
                rtol=rtol,
                atol=atol,
                backend=backend,
            )
            meta = _provenance(
                problem,
                backend=backend,
                method=method,
                dt=dt,
                rtol=rtol,
                atol=atol,
                ic=np.asarray(ic0, dtype=np.float64).copy(),
                history="callable" if history is not None else "constant",
            )
        else:  # ODEProblem (an SDEProblem is rejected inside _run_continuous)
            y = _run_continuous(
                problem, t_eval, method=method, rtol=rtol, atol=atol, backend=backend
            )
            meta = _provenance(
                problem,
                backend=backend,
                method=method,
                dt=dt,
                t0=start,
                rtol=rtol,
                atol=atol,
                ic=np.asarray(problem.ic, dtype=np.float64).copy(),
            )

    return Trajectory(t=t_arr, y=y, system=problem.system, meta=meta)


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


def _run_continuous(
    problem: Problem,
    t_eval: np.ndarray,
    *,
    method: str,
    rtol: float,
    atol: float,
    backend: str,
) -> np.ndarray:
    """Integrate a continuous-time problem (ODE) and sample at ``t_eval``."""
    if backend == "reference":
        if not isinstance(problem, ODEProblem):
            raise NotImplementedError(
                f"the reference integrator covers ODEs (and maps) only, not "
                f"{problem.family!r}; use backend='interp'/'jit' (the Rust engine) for "
                f"DDE integration, or StochasticSystem.integrate for SDEs."
            )
        return _reference_ode(problem, t_eval, method=method, rtol=rtol, atol=atol)

    if isinstance(problem, SDEProblem):
        # On the engine path: the generic integrate seam carries neither the SDE
        # noise seed nor the step-as-noise-scale, and the drift tape is ODE-shaped
        # (n_state == dim), so dispatching here would silently integrate the
        # *deterministic* drift as an ODE. Route SDEs through their seeded entry
        # point instead. (The reference path above already rejected SDEs.)
        raise NotImplementedError(
            "run.integrate cannot carry the SDE noise seed/step; use "
            "StochasticSystem.integrate(backend=...) for diagonal-Itô SDEs, or "
            "run.sde_integrate_dense(problem, t_eval, dt=, method=, seed=, backend=)."
        )

    eng = _engine()
    y = _engine_integrate_dense(
        eng,
        _primary_tape(problem).to_arrays(),
        problem.ic,
        problem.params_vec(),
        t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        jit=(backend == "jit"),
    )
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ConvergenceError(
            f"{_name(problem)}: integration diverged or the step collapsed before "
            f"reaching the final time."
        )
    return y


def _step_continuous(
    tape_arrays: tuple[Any, ...],
    ic: np.ndarray,
    params_vec: np.ndarray,
    t_eval: np.ndarray,
    *,
    method: str,
    rtol: float,
    atol: float,
    jit: bool,
    name: str,
) -> np.ndarray:
    """Integrate one dense ODE span from pre-marshalled tape arrays — the per-step seam.

    The lean inner core of
    :meth:`~tsdynamics.families.continuous.ContinuousSystem.step`: it performs
    exactly the engine call and finiteness guard :func:`_run_continuous` does for an
    ODE on the compiled engine, but takes the already-marshalled tape wire arrays
    (cached once at ``reinit`` — they are loop-invariant) plus the live parameter
    vector, skipping the per-call tape re-marshalling, ``Problem`` construction and
    output-grid build that the full :func:`integrate` entry point pays.  It is
    therefore **answer-identical** to :func:`_run_continuous` over the same inputs
    (stream WS-INVHOIST); it serves only the constant-/small-``dt`` stepping loops
    (Poincaré refinement, basins over flows, the streaming protocol), never a
    user-facing :class:`~tsdynamics.families.Trajectory` (which keeps its
    provenance via :func:`integrate`).

    Parameters
    ----------
    tape_arrays : tuple
        The RHS tape's engine wire arrays (:meth:`Tape.to_arrays`), marshalled once
        by the caller and reused across the loop.
    ic : ndarray, shape (dim,)
        The state to start this span from.
    params_vec : ndarray
        Live control-parameter values, read per step so a mid-loop parameter change
        still takes effect — identical to the pre-hoist path.
    t_eval : ndarray
        The (two-node) output grid for this span.
    method, rtol, atol, jit : str, float, float, bool
        Solver configuration, resolved once by ``reinit``.
    name : str
        System name, for the divergence error message.

    Returns
    -------
    ndarray, shape (len(t_eval), dim)

    Raises
    ------
    ConvergenceError
        If the integration diverged or the step collapsed (non-finite output) —
        the same loud-divergence contract as :func:`_run_continuous`.  Subclasses
        :class:`RuntimeError`, so existing ``except RuntimeError`` handlers (the
        divergence convention) keep catching it.
    """
    eng = _engine()
    y = np.asarray(
        _engine_integrate_dense(
            eng, tape_arrays, ic, params_vec, t_eval, method=method, rtol=rtol, atol=atol, jit=jit
        ),
        dtype=np.float64,
    )
    if not np.isfinite(y).all():
        raise ConvergenceError(
            f"{name}: integration diverged or the step collapsed before reaching the final time."
        )
    return y


# ---------------------------------------------------------------------------
# Resumable ODE stepper handle (stream WS-STEPPER)
# ---------------------------------------------------------------------------
#
# The durable replacement for the per-`dt` :func:`_step_continuous` core: an
# opaque engine handle (``tsdynamics._rust.OdeStepper``) that owns the built tape
# evaluator + solver once and carries the live integration point across calls. A
# constant-/small-``dt`` stepping loop (``ContinuousSystem.step()``, Poincaré
# refinement, basins over flows) then never rebuilds or re-marshals the tape — only
# the live state is threaded — yet the numerics are unchanged: ``advance(dt)`` is
# byte-for-byte identical to the per-`dt` :func:`_step_continuous` it supersedes
# (the engine re-seeds a fresh solver/state each ``dt``, so the adaptive controller
# behaves exactly as the released path; verified bit-for-bit in the Rust test
# ``stepper_advance_reproduces_per_dt_integrate_dense_bit_for_bit``).


def make_ode_stepper(
    tape_arrays: tuple[Any, ...],
    ic: np.ndarray,
    t0: float,
    *,
    method: str,
    rtol: float,
    atol: float,
    jit: bool,
) -> Any:
    """Build a resumable ``OdeStepper`` engine handle over an ODE tape.

    The opaque handle (``tsdynamics._rust.OdeStepper``) owns the built tape
    evaluator + the resolved solver/tolerances once and carries the live
    integration point ``(u, t)`` across :func:`step_advance` /
    :func:`step_advance_to_event` calls — the durable amortisation behind
    :meth:`~tsdynamics.families.continuous.ContinuousSystem.step` (stream
    WS-STEPPER). It is engine-only (``"interp"`` / ``"jit"``); the ``method`` is
    resolved by the engine's solver registry and an implicit kernel needs a tape
    built ``with_jacobian=True`` (rejected at construction).

    Parameters
    ----------
    tape_arrays : tuple
        The ODE tape's engine wire arrays (:meth:`Tape.to_arrays`).
    ic : ndarray, shape (dim,)
        The state to start from.
    t0 : float
        The start time.
    method : str
        Solver kernel name (already registry-canonical, e.g. ``"rk45"``).
    rtol, atol : float
        Adaptive-kernel tolerances.
    jit : bool
        Select the Cranelift evaluator (``True``) or the interpreter (``False``).

    Returns
    -------
    OdeStepper
        The opaque engine handle.

    Raises
    ------
    EngineNotAvailableError
        If :mod:`tsdynamics._rust` is not built.
    """
    eng = _engine()
    return eng.OdeStepper(
        *tape_arrays,
        np.ascontiguousarray(ic, dtype=np.float64),
        float(t0),
        method,
        float(rtol),
        float(atol),
        bool(jit),
    )


def step_advance(stepper: Any, dt: float, params_vec: np.ndarray, *, name: str) -> np.ndarray:
    """Advance an :class:`OdeStepper` handle by one ``dt`` and return the new state.

    The lean per-step seam :meth:`ContinuousSystem.step` calls: it advances the
    durable handle one ``dt`` from the live state (reading ``params_vec`` live, so a
    mid-loop parameter change still takes effect) and applies the same loud
    finiteness/divergence guard :func:`_step_continuous` did.  **Answer-identical**
    to the released per-``dt`` path (the engine reproduces it bit-for-bit), at a
    fraction of the per-call cost — the tape is marshalled once, into the handle.

    Raises
    ------
    ConvergenceError
        If the segment diverged or the step collapsed (a non-finite advance).
        Subclasses :class:`RuntimeError`, so existing ``except RuntimeError``
        divergence handlers keep catching it.
    """
    u = np.asarray(stepper.advance(float(dt), params_vec), dtype=np.float64)
    if not np.isfinite(u).all():
        raise ConvergenceError(
            f"{name}: integration diverged or the step collapsed before reaching the final time."
        )
    return u


def step_advance_to_event(
    stepper: Any,
    g_tape: Any,
    *,
    max_span: float,
    first_step: float,
    direction: int,
    params_vec: np.ndarray,
) -> tuple[bool, float, np.ndarray, int]:
    """March an :class:`OdeStepper` handle to the next crossing of ``g_tape``.

    The resumable crossing primitive (the durable analogue of :func:`crossings` for
    a single event): the handle marches up to ``max_span`` ahead of its live time
    and stops at the first refined crossing of the single-output event tape
    ``g_tape`` (``g(u, t) = 0``) in ``direction`` (``+1`` rising / ``-1`` falling /
    ``0`` either), carrying the adaptive step across the whole search. ``first_step``
    seeds the solver (with the fixed-step ``rk4`` it *is* the detection step ``dt``,
    reproducing the Python ``PoincareMap`` dt-grid march).

    Returns
    -------
    (found, t_cross, u_cross, direction) : (bool, float, ndarray, int)
        On a hit, the refined crossing and its sign, with the handle advanced one
        marching step *past* it (so a repeated call finds the *next* crossing); with
        no hit ``found`` is ``False``, the handle is advanced to ``t + max_span``,
        and ``u_cross`` is a zero placeholder.

    Raises
    ------
    ConvergenceError
        If the search diverged before a crossing or ``max_span``.
    """
    found, t_cross, u_cross, dir_ = stepper.advance_to_event(
        *g_tape.to_arrays(),
        float(max_span),
        float(first_step),
        int(direction),
        params_vec,
    )
    u_cross = np.asarray(u_cross, dtype=np.float64)
    if not (np.isfinite(t_cross) and np.isfinite(u_cross).all()):
        raise ConvergenceError(
            "OdeStepper.advance_to_event: crossing search diverged or produced "
            "non-finite values before the span end."
        )
    return bool(found), float(t_cross), u_cross, int(dir_)


def _run_dde(
    problem: DDEProblem,
    t_eval: np.ndarray,
    *,
    history: Any,
    dt: float,
    method: str,
    rtol: float,
    atol: float,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a DDE on the Rust method-of-steps engine (stream E-DDE).

    The delay system lowers (via :func:`tsdynamics.engine.compile.lower_dde`) to a
    tape over ``dim + n_slots`` inputs whose extra inputs are the delay slots; the
    engine fills those from a history buffer it interpolates with cubic Hermite,
    reusing the explicit solver kernels for each step.  Only constant delays
    lower; a state-dependent delay raises ``TapeCompileError`` at build time.

    Returns ``(y, ic0)`` — the sampled trajectory and the ``t0`` state used (the
    constant past, or ``history(0)`` for a callable past) so the caller can record
    it in provenance.
    """
    if backend == "reference":
        raise NotImplementedError(
            f"{_name(problem)}: backend='reference' has no DDE integrator; use "
            f"backend='interp'/'jit' (the Rust engine)."
        )
    try:
        eng = _engine()
    except EngineNotAvailableError as err:
        raise EngineNotAvailableError(
            f"{_name(problem)}: the Rust DDE engine (tsdynamics._rust) is not built; "
            f"install the compiled wheel (`pip install tsdynamics`) to integrate DDEs."
        ) from err

    dim = problem.dim
    max_delay = problem.max_delay
    # The t0 state and the past on [-max_delay, 0]; a constant past (no history
    # callable) is a single sample the engine treats as constant.
    past_t, past_y, ic0 = _sample_past(history, problem.ic, dim, max_delay, dt)

    slot_components = np.array([s.component for s in problem.delay_slots], dtype=np.int32)
    slot_delays = np.array([s.delay for s in problem.delay_slots], dtype=np.float64)

    y = np.asarray(
        eng.integrate_dde_dense(
            *problem.tape.to_arrays(),
            slot_components,
            slot_delays,
            np.ascontiguousarray(ic0, dtype=np.float64),
            np.ascontiguousarray(past_t, dtype=np.float64),
            np.ascontiguousarray(past_y.ravel(), dtype=np.float64),
            np.ascontiguousarray(t_eval, dtype=np.float64),
            method,
            float(rtol),
            float(atol),
            backend == "jit",
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(y)):
        raise ConvergenceError(
            f"{_name(problem)}: DDE integration diverged or the step collapsed before "
            f"reaching the final time (backend={backend!r}, method={method!r})."
        )
    return y, ic0


def _sample_past(
    history: Any,
    ic_arr: np.ndarray,
    dim: int,
    max_delay: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the past on ``[-max_delay, 0]``, returning ``(past_t, past_y, ic0)``.

    A ``None`` history is a constant past equal to ``ic_arr`` — a single sample
    the engine reads as constant.  A callable is sampled densely (so the engine's
    finite-difference Hermite tangents stay accurate), with the ``t0`` state taken
    as ``history(0)``.
    """
    if history is None:
        past_t = np.array([0.0], dtype=float)
        past_y = np.asarray(ic_arr, dtype=float).reshape(1, dim)
        return past_t, past_y, np.asarray(ic_arr, dtype=float)

    # Dense sampling: step no coarser than the output dt and well below the delay,
    # capped so a tiny delay or dt cannot explode the sample count.
    step = min(dt, max_delay / 200.0) if max_delay > 0.0 else dt
    step = max(step, 1e-9)
    n = max(2, int(np.ceil(max_delay / step)) + 1)
    n = min(n, 200_000)
    past_t = np.linspace(-max_delay, 0.0, n)
    past_y = np.array(
        [np.asarray(history(s), dtype=float).reshape(dim) for s in past_t],
        dtype=float,
    )
    return past_t, past_y, past_y[-1].copy()


def _run_map(problem: MapProblem, steps: int, backend: str) -> tuple[np.ndarray, np.ndarray]:
    """Iterate a map for ``steps`` steps; return ``(step_index, states)``.

    The step index starts at the problem's ``n0`` so a warm-restart map carries
    a meaningful iteration count on its trajectory.
    """
    if backend == "reference":
        y = _reference_map(problem, steps)
    else:
        eng = _engine()
        diverged_msg = (
            f"{_name(problem)}: map diverged or produced a non-finite state before "
            f"reaching {steps} iterations."
        )
        try:
            y = np.asarray(
                eng.iterate_map(
                    *problem.tape.to_arrays(), problem.ic, int(steps), backend == "jit"
                ),
                dtype=np.float64,
            )
        except RuntimeError as exc:
            # The compiled map loop raises (EngineError::Diverged → RuntimeError)
            # at the first non-finite iterate — the engine's diverge-loudly
            # contract. Re-raise *that* with the system name so the message matches
            # every other boundary (ODE/DDE/reference). A non-divergence
            # RuntimeError (e.g. a JIT compile failure from backend="jit") is a
            # different fault and must propagate unchanged, not be mislabelled as a
            # numerical blow-up.
            if "diverg" not in str(exc).lower():
                raise
            raise ConvergenceError(diverged_msg) from exc
        # Defense-in-depth: should the binding ever return NaN instead of raising,
        # still refuse to hand back a silently poisoned trajectory.
        if not np.all(np.isfinite(y)):
            raise ConvergenceError(diverged_msg)
    return np.arange(problem.n0, problem.n0 + steps), y


def ensemble(
    system_or_problem: Any,
    ics: Any,
    *,
    final_time: float = 100.0,
    dt: float = 0.02,
    t0: float | None = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    backend: str = "interp",
    **build_kwargs: Any,
) -> np.ndarray:
    """Integrate a batch of initial conditions and return their final states.

    ``ics`` is ``(n, dim)``; each row is integrated from ``t0`` to ``final_time``
    and its final state returned as a row of the ``(n, dim)`` result.  The
    compiled engine fans this out over a rayon thread pool (stream E5); the
    reference backend loops in Python.  A diverging trajectory yields a row of
    ``NaN`` rather than aborting the batch.

    Parameters
    ----------
    system_or_problem : SystemBase or Problem
    ics : array-like, shape (n, dim)
        The batch of initial conditions.
    dt : float, default 0.02
        Integration cadence.  For an adaptive method this seeds the controller's
        first trial step (it adapts away from it); for the fixed-step ``rk4`` it
        *is* the step taken for the whole run.  Matching :func:`integrate`'s
        ``dt`` so a fixed-step method gives the same trajectory through the dense
        and ensemble paths.  Consumed only by the compiled engine (the reference
        backend integrates adaptively via SciPy and ignores it).
    final_time, t0, method, rtol, atol, backend
        As in :func:`integrate`.

    Returns
    -------
    ndarray, shape (n, dim)
        Final states (rows of ``NaN`` for diverged trajectories).
    """
    backend = resolve_backend(backend)

    problem = _as_problem(system_or_problem, **build_kwargs)

    # Canonicalise the solver name exactly as :func:`integrate` does, via the
    # shared :func:`_resolve_method_for` contract — so an alias ("dopri5" →
    # "rk45"), a v2-only name ("LSODA"), and the auto-stiffness ``method="auto"``
    # are all handled identically across the engine entry points (diagnosis
    # #11/#13: ``"auto"`` is no longer an "unknown solver method" error here).
    # Maps and SDEs ignore `method`, so resolving is harmless.
    method = _resolve_method_for(method, problem).name

    ics = np.ascontiguousarray(ics, dtype=np.float64)
    if ics.ndim != 2 or ics.shape[1] != problem.dim:
        from tsdynamics.errors import invalid_value

        raise invalid_value("ics", ics.shape, rule=f"must be (n, {problem.dim})")

    # Maps and SDEs are dispatched before the (ODE-shaped) start-time logic below:
    # a MapProblem has no ``t0``, and an SDE batch needs its own seeded entry
    # point (the generic path would integrate the drift as a deterministic ODE).
    if isinstance(problem, SDEProblem):
        raise NotImplementedError(
            "run.ensemble cannot carry the SDE noise seed/step; use "
            "StochasticSystem.ensemble(...) for diagonal-Itô SDEs, or "
            "run.sde_ensemble_final(problem, ics, t0=, t1=, dt=, method=, seed=, backend=)."
        )
    if isinstance(problem, DDEProblem):
        # A DDE batch would need a per-trajectory history buffer; the engine has
        # no batched method-of-steps path. Refuse rather than fan the (extended,
        # delay-slot) tape out through the ODE ensemble FFI and return garbage.
        raise NotImplementedError(
            "run.ensemble has no DDE path (the engine integrates delay systems one "
            "trajectory at a time); use DelaySystem.integrate(backend='interp') per "
            "initial condition."
        )
    if isinstance(problem, MapProblem):
        steps = int(round(final_time))
        if backend == "reference":
            return _reference_map_ensemble(problem, ics, steps)
        eng = _engine()
        return np.asarray(
            _engine_map_ensemble_final(eng, problem, ics, steps, jit=(backend == "jit")),
            dtype=np.float64,
        )

    start = problem.t0 if t0 is None else float(t0)

    if backend == "reference":
        return _reference_ensemble(
            problem, ics, start, final_time, method=method, rtol=rtol, atol=atol
        )
    eng = _engine()
    return np.asarray(
        _engine_ensemble_final(
            eng,
            problem,
            ics,
            start,
            float(final_time),
            first_step=float(dt),
            method=method,
            rtol=rtol,
            atol=atol,
            jit=(backend == "jit"),
        ),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# SDE integration (diagonal-Itô; drift + diffusion tapes)
# ---------------------------------------------------------------------------
#
# The SDE engine entry points.  Unlike :func:`integrate` / :func:`ensemble`,
# these carry the two SDE-specific knobs — the fixed step ``dt`` (which *is* the
# noise scale ``√dt``) and the noise ``seed`` — and drive the two-tape engine
# call (drift + diffusion).  The family base class
# (:class:`~tsdynamics.families.stochastic.StochasticSystem`) calls these for its
# ``backend="interp"/"jit"`` path and wraps the result as a Trajectory with
# provenance; the pure-Python reference path stays in the family.


def sde_integrate_dense(
    problem: SDEProblem,
    t_eval: np.ndarray,
    *,
    dt: float,
    method: str,
    seed: int,
    backend: str = "interp",
) -> np.ndarray:
    """Integrate one diagonal-Itô SDE trajectory on the engine.

    Parameters
    ----------
    problem : SDEProblem
        The lowered drift + diffusion tapes plus runtime context.
    t_eval : ndarray
        Output grid; the first row of the result is the state at ``t_eval[0]``.
    dt : float
        Fixed step *and* noise scale (each Wiener increment is ``~ N(0, dt)``).
    method : str
        SDE kernel name (``"euler_maruyama"`` / ``"milstein"``).
    seed : int
        Seed for the noise stream (a ``u64``).
    backend : str, default "interp"
        ``"interp"`` or ``"jit"``.  ``"reference"`` has no engine SDE integrator
        (the pure-Python reference lives in the family).

    Returns
    -------
    ndarray, shape (len(t_eval), dim)

    Raises
    ------
    EngineNotAvailableError
        If :mod:`tsdynamics._rust` is not built.
    NotImplementedError
        If ``backend="reference"`` (use the family's reference integrator).
    RuntimeError
        If the trajectory diverged before the final time.
    """
    backend = resolve_backend(backend)
    if backend == "reference":
        raise NotImplementedError(
            "the engine SDE path needs backend='interp'/'jit'; the pure-Python "
            "reference SDE integrator lives in StochasticSystem.integrate."
        )
    eng = _engine()
    y = eng.integrate_sde_dense(
        *problem.drift.to_arrays(),
        *problem.diffusion.to_arrays(),
        np.ascontiguousarray(problem.ic, dtype=np.float64),
        problem.params_vec(),
        np.ascontiguousarray(t_eval, dtype=np.float64),
        method,
        float(dt),
        int(seed),
        backend == "jit",
    )
    return np.asarray(y, dtype=np.float64)


def sde_ensemble_final(
    problem: SDEProblem,
    ics: np.ndarray,
    *,
    t0: float,
    t1: float,
    dt: float,
    method: str,
    seed: int,
    backend: str = "interp",
) -> np.ndarray:
    """Integrate a batch of SDE initial conditions to ``t1`` and return final states.

    ``ics`` is ``(n, dim)``.  Trajectory ``i`` is seeded by ``seed_for(seed, i)``
    inside the engine, so the batch is **parallel == serial** bit-for-bit; a
    diverging trajectory yields a row of ``NaN`` rather than aborting the batch.

    Parameters
    ----------
    problem : SDEProblem
    ics : ndarray, shape (n, dim)
    t0, t1 : float
        Integration window.
    dt, method, seed, backend
        As in :func:`sde_integrate_dense` (``seed`` is the ensemble base seed).

    Returns
    -------
    ndarray, shape (n, dim)
        Final states (rows of ``NaN`` for diverged trajectories).
    """
    backend = resolve_backend(backend)
    if backend == "reference":
        raise NotImplementedError(
            "the engine SDE ensemble needs backend='interp'/'jit'; the pure-Python "
            "reference SDE ensemble lives in StochasticSystem.ensemble."
        )
    eng = _engine()
    ics = np.ascontiguousarray(ics, dtype=np.float64)
    y = eng.integrate_sde_ensemble_final(
        *problem.drift.to_arrays(),
        *problem.diffusion.to_arrays(),
        ics,
        problem.params_vec(),
        float(t0),
        float(t1),
        method,
        float(dt),
        int(seed),
        backend == "jit",
    )
    return np.asarray(y, dtype=np.float64)


# ---------------------------------------------------------------------------
# Engine dispatch (the E7 binding surface)
# ---------------------------------------------------------------------------
#
# These marshal a Problem to the call the compiled extension exposes.  The exact
# signatures are finalised by stream E7 (tsdyn-core / tsdynamics._rust); they are
# isolated here so that is the only thing E7 needs to match.  The payload is the
# tape wire arrays + the runtime vectors + the solver options.


def _engine_integrate_dense(
    eng: Any,
    tape_arrays: tuple[Any, ...],
    ic: np.ndarray,
    params_vec: np.ndarray,
    t_eval: np.ndarray,
    *,
    method: str,
    rtol: float,
    atol: float,
    jit: bool,
) -> np.ndarray:
    """Dispatch a dense single-trajectory integration to the engine.

    ``tape_arrays`` is the RHS tape's wire tuple (:meth:`Tape.to_arrays` — the
    drift tape for an SDE).  Taking the marshalled arrays as an argument rather than
    deriving them here lets a hot stepping loop pass arrays it marshalled once
    (:func:`_step_continuous`) instead of rebuilding the tuple per step (stream
    WS-INVHOIST), while the single-shot :func:`integrate` path marshals them inline
    at the call site.
    """
    return np.asarray(
        eng.integrate_dense(
            *tape_arrays,
            np.ascontiguousarray(ic, dtype=np.float64),
            params_vec,
            np.ascontiguousarray(t_eval, dtype=np.float64),
            method,
            float(rtol),
            float(atol),
            bool(jit),
        )
    )


def _engine_ensemble_final(
    eng: Any,
    problem: Problem,
    ics: np.ndarray,
    t0: float,
    t1: float,
    *,
    first_step: float,
    method: str,
    rtol: float,
    atol: float,
    jit: bool,
) -> np.ndarray:
    """Dispatch a parallel ensemble integration (final states) to the engine.

    ``first_step`` is the integration cadence (the user's ``dt``): only the first
    trial step for an adaptive kernel, but the step for the whole run for the
    fixed-step ``rk4`` — so the ensemble and dense (:func:`integrate`) paths take
    identical steps for fixed-step methods.
    """
    return np.asarray(
        eng.integrate_ensemble_final(
            *_primary_tape(problem).to_arrays(),
            ics,
            problem.params_vec(),
            float(t0),
            float(t1),
            float(first_step),
            method,
            float(rtol),
            float(atol),
            bool(jit),
        )
    )


def _engine_map_ensemble_final(
    eng: Any, problem: MapProblem, ics: np.ndarray, steps: int, *, jit: bool
) -> np.ndarray:
    """Dispatch a parallel map ensemble (final iterates) to the engine.

    Map parameters fold into the tape (``n_param == 0``), so there is no
    parameter vector; a diverging trajectory comes back as a ``NaN`` row.
    ``jit`` picks the native-code evaluator over the interpreter (bit-for-bit
    identical results).
    """
    return np.asarray(
        eng.iterate_ensemble_final(
            *problem.tape.to_arrays(),
            np.ascontiguousarray(ics, dtype=np.float64),
            int(steps),
            bool(jit),
        )
    )


# ---------------------------------------------------------------------------
# Reference backend (pure Python; the lowering oracle / fallback)
# ---------------------------------------------------------------------------


def _reference_ode(
    problem: ODEProblem,
    t_eval: np.ndarray,
    *,
    method: str,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Integrate the lowered ODE tape with SciPy (the dependency-light oracle).

    Delegates time-stepping to :func:`scipy.integrate.solve_ivp` over the
    reference-evaluated tape RHS — the same pattern the cross-validation harness
    uses for its reference backend.  This is *not* one of the engine's solver
    kernels; it is the pure-Python validation/fallback path.
    """
    from scipy.integrate import solve_ivp

    t_eval = np.ascontiguousarray(t_eval, dtype=np.float64)
    if t_eval.size == 0:
        return np.empty((0, problem.dim), dtype=np.float64)
    tape = problem.tape
    p = problem.params_vec()
    scipy_method = _scipy_method(method)

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        return eval_tape(tape, u, p, t)

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.asarray(problem.ic, dtype=np.float64).ravel(),
        t_eval=t_eval,
        method=scipy_method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise ConvergenceError(f"reference ODE integration failed: {sol.message}")
    return np.ascontiguousarray(sol.y.T)


def _reference_map(problem: MapProblem, steps: int) -> np.ndarray:
    """Iterate the lowered map tape in pure Python for ``steps`` steps.

    Diverges loudly: a non-finite iterate raises (naming the 0-based iterate
    index) rather than returning silent inf/NaN rows.  This matches the
    finiteness guard the ODE path already applies in :func:`_run_continuous`, and
    the diverge-loudly contract the compiled map loop will enforce (stream
    E-MAP, whose agreed shape is a ``NonFinite { step }`` error keyed by the same
    0-based iterate index).  So the pure-Python reference oracle and the engine
    agree on divergence handling — the cross-validation harness compares the two
    directly.
    """
    tape = problem.tape
    out = np.empty((steps, problem.dim), dtype=np.float64)
    x = np.asarray(problem.ic, dtype=np.float64).ravel()
    for i in range(steps):
        x = eval_tape(tape, x)
        if not np.all(np.isfinite(x)):
            raise ConvergenceError(
                f"{_name(problem)}: map diverged — non-finite state at iteration {i} "
                f"(0-based, of {steps} requested)."
            )
        out[i] = x
    return out


def _reference_map_ensemble(problem: MapProblem, ics: np.ndarray, steps: int) -> np.ndarray:
    """Loop the reference map iterator over a batch; NaN row on divergence.

    The pure-Python twin of the engine map ensemble: each IC is iterated to its
    ``f^{steps}`` final state; a diverged orbit becomes a ``NaN`` row (mirroring
    the engine's per-trajectory isolation) rather than aborting the batch.
    """
    out = np.empty_like(ics)
    if steps <= 0:
        out[:] = ics
        return out
    for i, ic in enumerate(ics):
        sub = MapProblem(
            tape=problem.tape,
            ic=np.asarray(ic, dtype=np.float64),
            system=problem.system,
        )
        try:
            out[i] = _reference_map(sub, steps)[-1]
        except (RuntimeError, ValueError):
            out[i] = np.nan
    return out


def _reference_ensemble(
    problem: Problem,
    ics: np.ndarray,
    t0: float,
    t1: float,
    *,
    method: str,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Loop the reference ODE integrator over a batch; NaN row on divergence."""
    if not isinstance(problem, ODEProblem):
        raise NotImplementedError(
            f"the reference ensemble covers ODEs only, not {problem.family!r}; use "
            f"backend='interp'/'jit'."
        )
    t_eval = np.array([t0, t1], dtype=np.float64)
    out = np.empty_like(ics)
    for i, ic in enumerate(ics):
        sub = ODEProblem(tape=problem.tape, ic=ic, t0=t0, system=problem.system)
        try:
            y = _reference_ode(sub, t_eval, method=method, rtol=rtol, atol=atol)
            out[i] = y[-1]
        except (RuntimeError, ValueError):
            out[i] = np.nan
    return out


#: Canonical engine kernel name → SciPy ``solve_ivp`` method (reference backend
#: only — the pure-Python oracle).  Keys are the names :func:`solvers.resolve`
#: produces (lower-case), so the reference path sees the same canonical method the
#: engine does.  SciPy has no Tsitouras / fixed-RK4 / Rosenbrock-W / TR-BDF2
#: integrator, so those map to the nearest-character SciPy solver (an explicit RK
#: for the explicit kernels, an implicit one for the implicit kernels) — close
#: enough for an oracle, since the reference backend validates the *tape*, not the
#: exact stepper.
_SCIPY_METHOD: dict[str, str] = {
    "rk45": "RK45",
    "tsit5": "RK45",
    "rk4": "RK45",
    "dop853": "DOP853",
    "bdf": "BDF",
    "rosenbrock": "Radau",
    "trbdf2": "BDF",
}


def _scipy_method(method: str) -> str:
    """Map a canonical engine kernel name to a SciPy ``solve_ivp`` method, or raise.

    The reference backend's ODE oracle is :func:`scipy.integrate.solve_ivp`, which
    only knows the methods in :data:`_SCIPY_METHOD`.  A kernel with no SciPy
    equivalent (an SDE method reaching the ODE oracle, a plugin kernel with no
    mapping) gets a clear :class:`~tsdynamics.errors.InvalidParameterError`
    naming the method and pointing at the engine backends — instead of letting a
    raw SciPy ``ValueError`` ("Unknown method …") surface from inside the oracle.
    """
    try:
        return _SCIPY_METHOD[method]
    except KeyError:
        from tsdynamics.errors import invalid_value

        raise invalid_value(
            "method",
            method,
            rule="has no reference (SciPy) equivalent",
            hint=f"Use backend='interp'/'jit' (the Rust engine) for this method, or "
            f"choose a reference-capable one from {sorted(_SCIPY_METHOD)}.",
        ) from None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _provenance(problem: Problem, **extra: Any) -> dict[str, Any]:
    """Build the provenance dict attached to an engine-produced Trajectory."""
    system = problem.system
    if system is not None and hasattr(system, "_provenance"):
        prov = system._provenance(family=problem.family, engine="rust", **extra)
        return cast("dict[str, Any]", prov)
    return {"family": problem.family, "engine": "rust", **extra}


def _name(problem: Problem) -> str:
    """Return a readable name for error messages."""
    return type(problem.system).__name__ if problem.system is not None else problem.family


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
