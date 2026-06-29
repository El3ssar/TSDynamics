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
    # ``_engine_integrate_dense`` returns a ``float64`` ndarray (the single dtype
    # conversion point), so re-wrapping it with an explicit dtype here would be a
    # redundant pass over the array — consume it as-is.
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


# ---------------------------------------------------------------------------
# Basin / attractor recurrence FSM (stream perf/basin-march)
# ---------------------------------------------------------------------------
# The sequential Rust kernel that runs the *entire* per-initial-condition
# recurrence FSM (stepping + cell-binning + the shared-label early-out) in one
# engine call — the accelerated path behind ``find_attractors`` /
# ``basins_of_attraction``.  It is bit-identical to the Python ``_AttractorMapper``
# because every cell-check advances the SAME engine stepper the Python
# ``system.step(dt)`` drives (a fresh-solver ``integrate_dense`` over ``[t, t+dt]``
# for a flow, one map iteration for a map).  See the ``basin`` module in the engine
# crate for the why-sequential rationale (parallelism is a measured net loss).


def basin_march(
    tape_arrays: tuple[Any, ...],
    params_vec: np.ndarray,
    grid_lo: np.ndarray,
    grid_hi: np.ndarray,
    grid_counts: np.ndarray,
    seeds: np.ndarray,
    thresholds: tuple[int, int, int, int, int, int],
    *,
    is_discrete: bool,
    method: str = "rk45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt: float = 1.0,
    jit: bool = False,
) -> dict[str, Any]:
    """Run the whole basin recurrence FSM in the Rust engine, for every seed.

    Drives the sequential kernel (``tsdynamics._rust.basin_march_flow`` for a flow,
    ``basin_march_map`` for a map) over the shared cell tessellation, returning the
    per-seed labels plus the accumulated ``att_cells`` / ``bas_cells`` /
    ``att_points`` the basin layer rebuilds its
    :class:`~tsdynamics.analysis.basins.attractors._AttractorMapper` from.  The
    per-cell-check numerics are byte-for-byte the released stepping path, so the
    labels are bit-identical to the pure-Python loop.

    Parameters
    ----------
    tape_arrays : tuple
        The ODE/map tape's engine wire arrays (:meth:`Tape.to_arrays`).
    params_vec : ndarray
        Live control parameters (empty for a lowered map).
    grid_lo, grid_hi : ndarray, shape (dim,)
        Lower/upper corner of the recurrence cell box.
    grid_counts : ndarray of int, shape (dim,)
        Cells per axis.
    seeds : ndarray, shape (n_seeds, dim)
        Initial conditions to classify, in order (the order the shared labelling
        accumulates in — see the kernel note on why it must stay serial).
    thresholds : tuple of int
        ``(max_steps, mx_fnd, mx_loc, mx_att, mx_bas, mx_lost)`` — the six FSM
        thresholds (``consecutive_recurrences`` etc.).
    is_discrete : bool
        Whether ``tape_arrays`` is a map (drives the map kernel; ``dt``/``method``/
        tolerances are then ignored).
    method, rtol, atol, dt : optional
        Flow stepper configuration (``method`` registry-canonical; ``dt`` the
        per-cell-check step).
    jit : bool, default False
        Select the Cranelift evaluator.

    Returns
    -------
    dict
        ``{"labels", "att_cells", "bas_cells", "att_points", "dim"}`` where
        ``att_cells`` / ``bas_cells`` are ``{flat_cell_index: attractor_id}`` and
        ``att_points`` is ``{attractor_id: (m, dim) ndarray}``.

    Raises
    ------
    EngineNotAvailableError
        If :mod:`tsdynamics._rust` is not built.
    """
    eng = _engine()
    seeds = np.ascontiguousarray(seeds, dtype=np.float64)
    if seeds.ndim != 2:
        seeds = seeds.reshape(-1, len(grid_counts))
    grid_lo = np.ascontiguousarray(grid_lo, dtype=np.float64)
    grid_hi = np.ascontiguousarray(grid_hi, dtype=np.float64)
    grid_counts = np.ascontiguousarray(grid_counts, dtype=np.int64)
    params_vec = np.ascontiguousarray(params_vec, dtype=np.float64)
    max_steps, mx_fnd, mx_loc, mx_att, mx_bas, mx_lost = (int(x) for x in thresholds)

    if is_discrete:
        out = eng.basin_march_map(
            *tape_arrays,
            params_vec,
            grid_lo,
            grid_hi,
            grid_counts,
            seeds,
            max_steps,
            mx_fnd,
            mx_loc,
            mx_att,
            mx_bas,
            mx_lost,
            bool(jit),
        )
    else:
        out = eng.basin_march_flow(
            *tape_arrays,
            params_vec,
            str(method),
            float(rtol),
            float(atol),
            float(dt),
            grid_lo,
            grid_hi,
            grid_counts,
            seeds,
            max_steps,
            mx_fnd,
            mx_loc,
            mx_att,
            mx_bas,
            mx_lost,
            bool(jit),
        )
    (
        labels,
        att_keys,
        att_ids,
        bas_keys,
        bas_ids,
        point_ids,
        point_counts,
        points_flat,
        dim,
    ) = out
    dim = int(dim)
    att_cells = {int(k): int(v) for k, v in zip(att_keys, att_ids, strict=True)}
    bas_cells = {int(k): int(v) for k, v in zip(bas_keys, bas_ids, strict=True)}
    att_points: dict[int, np.ndarray] = {}
    off = 0
    points_flat = np.asarray(points_flat, dtype=np.float64)
    for pid, m in zip(point_ids, point_counts, strict=True):
        m = int(m)
        block = points_flat[off * dim : (off + m) * dim].reshape(m, dim)
        att_points[int(pid)] = np.array(block, dtype=np.float64)
        off += m
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "att_cells": att_cells,
        "bas_cells": bas_cells,
        "att_points": att_points,
        "dim": dim,
    }


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
