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

from typing import Any

import numpy as np

from tsdynamics.utils.grids import make_output_grid

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
    "ensemble",
    "eval_jac",
    "eval_rhs",
    "integrate",
    "resolve_backend",
    "sde_ensemble_final",
    "sde_integrate_dense",
]

#: The selectable backend names.  ``"interp"`` / ``"jit"`` run on the compiled
#: engine; ``"reference"`` is the pure-Python oracle/fallback.
BACKENDS: frozenset[str] = frozenset({"interp", "jit", "reference"})


class EngineNotAvailableError(RuntimeError):
    """The compiled Rust engine (:mod:`tsdynamics._rust`, stream E7) is not available.

    Raised when an ``"interp"`` / ``"jit"`` run is requested but the extension
    module is not importable.  Use ``backend="reference"`` for pure-Python RHS
    evaluation and small ODE/map runs, or the family's own ``integrate`` for the
    v2 backends, until the engine wheel is built.
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
    ValueError
        If ``backend`` is not a recognised name.
    """
    name = str(backend).lower()
    if name == "auto":
        return "interp"
    if name not in BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {sorted(BACKENDS)} (or 'auto'). "
            f"'interp'/'jit' run on the Rust engine; 'reference' is the pure-Python oracle."
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
            "the Rust engine extension (tsdynamics._rust, stream E7) is not built; "
            "use backend='reference' for pure-Python evaluation, or the family's own "
            "integrate() for the v2 backends."
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
):
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
        Solver name, resolved by the solver registry (stream C-SOLV).
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
    problem = _as_problem(system_or_problem, ic=ic, **build_kwargs)

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
        eng, problem, t_eval, method=method, rtol=rtol, atol=atol, jit=(backend == "jit")
    )
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise RuntimeError(
            f"{_name(problem)}: integration diverged or the step collapsed before "
            f"reaching the final time."
        )
    return y


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
    lower; a state-dependent delay raises ``TapeCompileError`` at build time (use
    the family's ``backend="jitcdde"``).

    Returns ``(y, ic0)`` — the sampled trajectory and the ``t0`` state used (the
    constant past, or ``history(0)`` for a callable past) so the caller can record
    it in provenance.
    """
    if backend == "reference":
        raise NotImplementedError(
            f"{_name(problem)}: backend='reference' has no DDE integrator; use "
            f"backend='interp'/'jit' (the Rust engine), or the family's "
            f"integrate(backend='jitcdde') for the v2 backend."
        )
    try:
        eng = _engine()
    except EngineNotAvailableError as err:
        # The generic accessor steers callers to backend='reference', which a DDE
        # rejects one branch up — give the DDE-correct fallback (jitcdde) instead.
        raise EngineNotAvailableError(
            f"{_name(problem)}: the Rust DDE engine (tsdynamics._rust, stream E7) is not "
            f"built; use the family's integrate(backend='jitcdde') for the v2 DDE backend."
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
        raise RuntimeError(
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
                eng.iterate_map(*problem.tape.to_arrays(), problem.ic, int(steps)),
                dtype=np.float64,
            )
        except RuntimeError as exc:
            # The compiled map loop raises (EngineError::Diverged → RuntimeError)
            # at the first non-finite iterate — the engine's diverge-loudly
            # contract. Re-raise with the system name so the message matches every
            # other boundary (ODE/DDE/reference).
            raise RuntimeError(diverged_msg) from exc
        # Defense-in-depth: should the binding ever return NaN instead of raising,
        # still refuse to hand back a silently poisoned trajectory.
        if not np.all(np.isfinite(y)):
            raise RuntimeError(diverged_msg)
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
    ics = np.ascontiguousarray(ics, dtype=np.float64)
    if ics.ndim != 2 or ics.shape[1] != problem.dim:
        raise ValueError(f"ics must be (n, {problem.dim}); got shape {ics.shape}")

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
        return np.asarray(_engine_map_ensemble_final(eng, problem, ics, steps), dtype=np.float64)

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
    problem: Problem,
    t_eval: np.ndarray,
    *,
    method: str,
    rtol: float,
    atol: float,
    jit: bool,
) -> np.ndarray:
    """Dispatch a dense single-trajectory integration to the engine."""
    return eng.integrate_dense(
        *_primary_tape(problem).to_arrays(),
        np.ascontiguousarray(problem.ic, dtype=np.float64),
        problem.params_vec(),
        np.ascontiguousarray(t_eval, dtype=np.float64),
        method,
        float(rtol),
        float(atol),
        bool(jit),
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
    return eng.integrate_ensemble_final(
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


def _engine_map_ensemble_final(
    eng: Any, problem: MapProblem, ics: np.ndarray, steps: int
) -> np.ndarray:
    """Dispatch a parallel map ensemble (final iterates) to the engine.

    Map parameters fold into the tape (``n_param == 0``), so there is no
    parameter vector; a diverging trajectory comes back as a ``NaN`` row.
    """
    return eng.iterate_ensemble_final(
        *problem.tape.to_arrays(),
        np.ascontiguousarray(ics, dtype=np.float64),
        int(steps),
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
    scipy_method = _SCIPY_METHOD.get(method, method)

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
        raise RuntimeError(f"reference ODE integration failed: {sol.message}")
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
            raise RuntimeError(
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


#: TSDynamics solver name → SciPy ``solve_ivp`` method (reference backend only).
_SCIPY_METHOD: dict[str, str] = {
    "RK45": "RK45",
    "dopri5": "RK45",
    "DOP853": "DOP853",
    "dop853": "DOP853",
    "LSODA": "LSODA",
    "lsoda": "LSODA",
    "BDF": "BDF",
    "bdf": "BDF",
    "Radau": "Radau",
    "radau": "Radau",
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _provenance(problem: Problem, **extra: Any) -> dict:
    """Build the provenance dict attached to an engine-produced Trajectory."""
    system = problem.system
    if system is not None and hasattr(system, "_provenance"):
        return system._provenance(family=problem.family, engine="rust", **extra)
    return {"family": problem.family, "engine": "rust", **extra}


def _name(problem: Problem) -> str:
    """Return a readable name for error messages."""
    return type(problem.system).__name__ if problem.system is not None else problem.family
