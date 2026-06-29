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

Module layout (the run-split refactor)
--------------------------------------
``run.py`` keeps the ``integrate`` / ``ensemble`` orchestration, the pointwise
``eval_rhs`` / ``eval_jac`` seam, backend resolution (``resolve_backend`` /
``_engine`` / ``BACKENDS`` / :class:`EngineNotAvailableError`) and the shared
problem-coercion / naming / provenance helpers.  The rest was split out into
focused submodules and is **re-exported here** so every name stays reachable at
its historical path ``tsdynamics.engine.run.<name>``:

- :mod:`~tsdynamics.engine.run_methods` — ``method=`` resolution + auto-stiffness
  (``_resolve_method_for`` / ``_recommend_method`` / ``_resolve_method_and_prepare``).
- :mod:`~tsdynamics.engine._families` — the per-family runners (``_run_continuous``
  / ``_step_continuous`` / ``_run_dde`` / ``_run_map`` / ``_sample_past``) and the
  low-level ``_engine_*`` FFI shims.
- :mod:`~tsdynamics.engine.stepper` — the resumable ``OdeStepper`` API
  (``make_ode_stepper`` / ``step_advance`` / ``step_advance_to_event``).
- :mod:`~tsdynamics.engine.sde_run` — the SDE dense/ensemble seam
  (``sde_integrate_dense`` / ``sde_ensemble_final``).
- :mod:`~tsdynamics.engine.events` — the event subsystem (``crossings`` / ``Event``
  / ``EventSolution`` / ``integrate_events``).
- :mod:`~tsdynamics.engine.reference` — the pure-Python reference oracle
  (``_reference_*`` / ``_scipy_method``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tsdynamics.errors import BackendError
from tsdynamics.utils.grids import make_output_grid

if TYPE_CHECKING:
    from tsdynamics.families import Trajectory

from ._families import (  # noqa: F401
    _engine_ensemble_final,
    _engine_integrate_dense,
    _engine_map_ensemble_final,
    _run_dde,
    _run_map,
    _sample_past,
)

# The run-split refactor moved the focused concerns out of this module into the
# submodules below.  Re-export every name they own so all of them stay reachable
# at their historical path ``tsdynamics.engine.run.<name>`` — a pure move, no
# public-surface change.  (The split-out modules late-import the shared run-side
# helpers — ``_engine``/``resolve_backend``/``_name``/``_primary_tape``/… — from
# this module inside their functions, so these module-level imports do not create
# an import cycle.)  ``_run_continuous`` / ``_step_continuous`` carry the explicit
# ``X as X`` re-export spelling because the rest of ``src`` imports them by that
# private name (``continuous.py`` / ``events.py``) and they are not in ``__all__``,
# so ``mypy --strict`` (``--no-implicit-reexport``) needs them marked explicit; the
# names in ``__all__`` (and the test-only privates) are fine under plain ``F401``.
from ._families import _run_continuous as _run_continuous
from ._families import _step_continuous as _step_continuous
from .compile import eval_tape, eval_tape_jac

# Event subsystem (engine/events.py) and the pure-Python reference oracle
# (engine/reference.py).
from .events import (  # noqa: F401
    _DIRECTION_WORDS,
    _EVENT_ENGINE_METHODS,
    Event,
    EventSolution,
    _engine_events,
    _event_tape,
    _normalize_event_direction,
    _reference_events,
    _resolve_event_axis,
    crossings,
    integrate_events,
)
from .problem import (
    DDEProblem,
    MapProblem,
    ODEProblem,
    Problem,
    SDEProblem,
    build_problem,
)
from .reference import (  # noqa: F401
    _SCIPY_METHOD,
    _reference_ensemble,
    _reference_map,
    _reference_map_ensemble,
    _reference_ode,
    _scipy_method,
)
from .run_methods import (  # noqa: F401
    _recommend_method,
    _resolve_method_and_prepare,
    _resolve_method_for,
)
from .sde_run import (  # noqa: F401
    sde_ensemble_final,
    sde_integrate_dense,
)
from .stepper import (  # noqa: F401
    make_ode_stepper,
    step_advance,
    step_advance_to_event,
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


def _primary_tape(problem: Problem) -> Any:
    """Return the RHS/next-state tape of a problem (the drift tape for an SDE)."""
    if isinstance(problem, SDEProblem):
        return problem.drift
    return problem.tape


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


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


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
        special value ``"auto"`` selects a kernel by a-priori auto-stiffness
        **for ODEs**: the Jacobian spectrum at the start state is probed and an
        implicit kernel (``bdf``) is used on a stiff RHS, the explicit default
        (``rk45``) otherwise (:func:`tsdynamics.solvers.recommend`; a one-point
        heuristic).  The probe reads the Jacobian at the resolved initial state
        (``problem.ic``) but always at ``t=0.0`` — the ``t0=`` argument is not
        threaded into it — so a non-autonomous system whose stiffness varies in
        time is classified by its ``t=0`` spectrum.  This only affects kernel
        selection, never the integrated trajectory.  For maps (no kernel) and
        DDEs (explicit method-of-steps, whose instantaneous-Jacobian probe would
        ignore the delay terms), ``"auto"`` is a no-op resolving to the family
        default — see :func:`_recommend_method`.
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
    InvalidParameterError
        For an unknown ``backend`` or ``method`` (incl. v2-only names), a
        non-positive ``dt``, or a non-forward window (``final_time <= t0``).
        Subclasses :class:`ValueError`.
    ConvergenceError
        If the integration diverged or the step collapsed before the final time
        (a non-finite result).  Subclasses :class:`RuntimeError`.
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
    # Resolve method= and (for an implicit ODE kernel) rebuild the tape with the
    # Jacobian, via the one shared contract ensemble() uses too.
    method, problem = _resolve_method_and_prepare(
        system_or_problem, problem, method, build_kwargs, ic=ic
    )

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

    The same up-front ``method=`` / ``dt`` / window validation as
    :func:`integrate` runs *before* any per-trajectory work: an unknown or
    v2-only solver name, a non-positive ``dt``, or a non-forward window
    (``final_time <= t0``) raises :class:`~tsdynamics.errors.InvalidParameterError`
    here rather than slipping through to the solver as a silent no-step or
    backwards run.

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
        backend integrates adaptively via SciPy and ignores it).  Must be strictly
        positive (validated up front for the flow path).
    final_time, t0, method, rtol, atol, backend
        As in :func:`integrate`.  ``method="auto"`` selects a kernel by the same
        a-priori auto-stiffness probe as :func:`integrate`, with one documented
        caveat: the batch is *not* probed per-row.  The stiffness verdict is read
        once, at the system's **default** initial condition (``problem.ic``, since
        ``ensemble`` builds the Problem without a single ``ic=``) and at ``t=0.0``
        (the start time ``t0`` is not threaded into the probe).  This only ever
        affects which kernel is chosen — never the integrated results — so a stiff
        batch whose default IC sits in a non-stiff region should pass an explicit
        ``method=`` (or the system should declare ``_default_method``).

    Returns
    -------
    ndarray, shape (n, dim)
        Final states (rows of ``NaN`` for diverged trajectories).

    Raises
    ------
    InvalidParameterError
        For an unknown/v2-only ``method``, a non-positive ``dt``, a non-forward
        window, or an ``ics`` array that is not ``(n, dim)``.  Subclasses
        :class:`ValueError`.
    ConvergenceError
        Never raised for a single diverged trajectory (those become ``NaN``
        rows); a batch-wide engine failure surfaces through the engine seam.
    EngineNotAvailableError
        For ``"interp"``/``"jit"`` when :mod:`tsdynamics._rust` is not built.
    NotImplementedError
        For an SDE problem (use :func:`sde_ensemble_final`) or a DDE problem (the
        engine has no batched method-of-steps path — integrate one IC at a time).
    """
    backend = resolve_backend(backend)

    problem = _as_problem(system_or_problem, **build_kwargs)

    # Canonicalise the solver name AND (for an implicit ODE kernel) rebuild the
    # tape with the Jacobian — the exact same shared contract :func:`integrate`
    # uses, so an alias ("dopri5" → "rk45"), a v2-only name ("LSODA"), and the
    # auto-stiffness ``method="auto"`` behave identically across both entry points
    # (diagnosis #11/#13).  Without the rebuild a stiff ``ensemble(method="auto"/
    # "bdf")`` would dispatch a Jacobian-less tape and the engine would refuse it.
    # Maps/SDEs ignore `method` (the rebuild is ODE-only), so resolving is harmless.
    method, problem = _resolve_method_and_prepare(system_or_problem, problem, method, build_kwargs)

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

    # Up-front window/dt validation, identical to the checks :func:`integrate`
    # runs through ``make_output_grid`` for the flow path: a non-positive ``dt``
    # or a non-forward window is a typed ``InvalidParameterError`` raised here,
    # before any (per-trajectory) engine work, rather than slipping through to the
    # solver as a silent no-step / backwards run.  ``make_output_grid`` is the one
    # validation chokepoint both entry points share; the grid it returns is
    # discarded (the ensemble engine call needs only ``t0``/``t1``/``first_step``).
    make_output_grid(start, float(final_time), dt)

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
