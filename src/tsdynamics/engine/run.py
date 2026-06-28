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

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tsdynamics.errors import BackendError, ConvergenceError
from tsdynamics.utils.grids import make_output_grid

if TYPE_CHECKING:
    from tsdynamics.families import Trajectory

from .compile import eval_tape, eval_tape_jac

# Event subsystem (engine/events.py) and the pure-Python reference oracle
# (engine/reference.py) were split out of this module (the run-split refactor).
# Re-export every name they own so all of them stay reachable at their historical
# path ``tsdynamics.engine.run.<name>`` — a pure move, no public-surface change.
# ``_reference_*`` are also used internally below by ``_run_continuous`` /
# ``_run_map`` / ``ensemble``; the remaining names are re-exports for back-compat.
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

    The auto-stiffness wiring (ticket FIX-AUTOSTIFF) applies to **ODEs only**:
    probe the problem's Jacobian spectrum at its start state and let the solver
    registry recommend an implicit kernel (``bdf``) on a stiff RHS or the explicit
    default (``rk45``) otherwise (:func:`tsdynamics.solvers.recommend`).  The probe
    is the one-point heuristic :func:`tsdynamics.solvers.is_stiff` — useful but not
    a guarantee (see its docstring): the verdict is read at the integration's *own*
    start state (``problem.ic``), so a system declares a stiff default via
    ``_default_method`` when the heuristic is too coarse for it.

    Maps and DDEs treat ``"auto"`` as an explicit, documented **no-op** — it
    resolves to the family default rather than probing:

    - **Maps** iterate without a solver kernel, so the resolved kernel is ignored
      by the map branch (``"auto"`` simply does not raise).
    - **DDEs** are driven by the method of steps, which reuses the *explicit*
      stage kernels only.  A stiffness probe would be meaningless **and** a trap:
      :func:`tsdynamics.solvers.is_stiff` reads only the *instantaneous* Jacobian
      ``∂f/∂u``, ignoring the delay terms that actually shape a DDE's spectrum, so
      a spurious "stiff" verdict could otherwise select the implicit
      ``rosenbrock`` the DDE engine cannot drive.  ``"auto"`` therefore resolves to
      the DDE default (``rk45``); a delay system that genuinely needs another
      explicit kernel should pass ``method=`` explicitly.

    SDEs never reach here (the SDE path refuses ``run.integrate`` upstream).

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

    # Maps (no kernel) and DDEs (explicit method-of-steps; no meaningful stiffness
    # probe) fall back to their family default — auto-stiffness is ODE-only.
    if isinstance(problem, MapProblem | DDEProblem):
        # A map has no solver family of its own (DEFAULT_METHOD has no "map" key)
        # and the map branch discards the resolved kernel entirely, so it borrows
        # the "ode" default purely to return a valid Resolution; only DDEs carry a
        # genuine family here.
        family = "dde" if isinstance(problem, DDEProblem) else "ode"
        return solvers.resolve(solvers.DEFAULT_METHOD[family], family=family)
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


def _resolve_method_and_prepare(
    system_or_problem: Any,
    problem: Problem,
    method: str,
    build_kwargs: dict[str, Any],
    **rebuild_extra: Any,
) -> tuple[str, Problem]:
    """Resolve ``method=`` and rebuild the ODE problem ``with_jacobian`` if the kernel needs it.

    The single shared contract for both :func:`integrate` and :func:`ensemble`, so
    ``method="auto"`` *and* the implicit-kernel Jacobian requirement behave
    identically through both entry points.  (This previously drifted: ``ensemble``
    resolved ``"auto"`` but skipped the ``with_jacobian`` rebuild, so a stiff
    ``ensemble(method="auto")`` dispatched a Jacobian-less tape and the engine
    refused it — factoring the two halves into one helper closes that gap.)

    The implicit ODE kernels (``bdf``/``rosenbrock``/``trbdf2``) need ``∂f/∂u`` on
    the tape or the engine refuses the step.  Rebuild once with the Jacobian when
    the resolved method needs it and the tape lacks it — only for an ODE built from
    a *system* (DDEs drive explicit kernels only; maps fold params in; a pre-built
    ``Problem`` we cannot re-lower, so the engine raises its clear guard instead).
    ``rebuild_extra`` carries entry-point-specific build kwargs (``integrate``
    passes ``ic=``; ``ensemble`` seeds each trajectory later, so it passes none).

    Returns the canonical method name and the (possibly rebuilt) problem.
    """
    resolution = _resolve_method_for(method, problem)
    if (
        resolution.build_kwargs.get("with_jacobian")
        and isinstance(problem, ODEProblem)
        and not problem.tape.has_jacobian
        and "with_jacobian" not in build_kwargs
        and not isinstance(system_or_problem, ODEProblem)
    ):
        problem = _as_problem(
            system_or_problem, with_jacobian=True, **rebuild_extra, **build_kwargs
        )
    return resolution.name, problem


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
        heuristic).  For maps (no kernel) and DDEs (explicit method-of-steps,
        whose instantaneous-Jacobian probe would ignore the delay terms),
        ``"auto"`` is a no-op resolving to the family default — see
        :func:`_recommend_method`.
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
