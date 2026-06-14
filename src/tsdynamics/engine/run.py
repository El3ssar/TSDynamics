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


def _make_t_eval(t0: float, tf: float, dt: float) -> np.ndarray:
    """Uniform output grid from ``t0`` to ``tf`` inclusive (matches the families)."""
    t_arr = np.arange(t0, tf, dt)
    if t_arr.size == 0 or t_arr[-1] < tf - 1e-12:
        t_arr = np.append(t_arr, tf)
    return t_arr


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
    **build_kwargs: Any,
):
    """Integrate a system on the engine and return a :class:`~tsdynamics.families.Trajectory`.

    Parameters
    ----------
    system_or_problem : SystemBase or Problem
        The system (lowered to a Problem here) or a pre-built Problem.
    final_time : float, default 100.0
        End of the integration window.
    dt : float, default 0.02
        Output sampling interval (the internal stepper is adaptive for adaptive
        methods).
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

    Returns
    -------
    Trajectory

    Raises
    ------
    EngineNotAvailableError
        For ``"interp"``/``"jit"`` when :mod:`tsdynamics._rust` is not built.
    NotImplementedError
        For ``backend="reference"`` on a DDE or SDE (the reference integrator
        covers ODEs and maps only).
    """
    from tsdynamics.families import Trajectory

    backend = resolve_backend(backend)
    problem = _as_problem(system_or_problem, ic=ic, **build_kwargs)

    if isinstance(problem, MapProblem):
        steps = int(round(final_time))
        t_arr, y = _run_map(problem, steps, backend)
    else:
        start = problem.t0 if t0 is None else float(t0)
        t_eval = _make_t_eval(start, final_time, dt)
        y = _run_continuous(problem, t_eval, method=method, rtol=rtol, atol=atol, backend=backend)
        t_arr = t_eval

    return Trajectory(
        t=t_arr,
        y=y,
        system=problem.system,
        meta=_provenance(problem, backend=backend, method=method, dt=dt, rtol=rtol, atol=atol),
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
    """Integrate a continuous-time problem (ODE/DDE/SDE) and sample at ``t_eval``."""
    if backend == "reference":
        if not isinstance(problem, ODEProblem):
            raise NotImplementedError(
                f"the reference integrator covers ODEs (and maps) only, not "
                f"{problem.family!r}; use backend='interp'/'jit' (the Rust engine) for "
                f"DDE/SDE integration."
            )
        return _reference_ode(problem, t_eval, method=method, rtol=rtol, atol=atol)

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


def _run_map(problem: MapProblem, steps: int, backend: str) -> tuple[np.ndarray, np.ndarray]:
    """Iterate a map for ``steps`` steps; return ``(step_index, states)``.

    The step index starts at the problem's ``n0`` so a warm-restart map carries
    a meaningful iteration count on its trajectory.
    """
    if backend == "reference":
        y = _reference_map(problem, steps)
    else:
        eng = _engine()
        y = np.asarray(
            eng.iterate_map(*problem.tape.to_arrays(), problem.ic, int(steps)), dtype=np.float64
        )
        # Backstop mirroring _run_continuous: the compiled map loop (stream E-MAP)
        # is expected to raise on a non-finite iterate, but guard here too so the
        # engine map path can never hand back a silently poisoned trajectory,
        # whatever way the binding surfaces divergence.
        if not np.all(np.isfinite(y)):
            raise RuntimeError(
                f"{_name(problem)}: map diverged or produced a non-finite state before "
                f"reaching {steps} iterations."
            )
    return np.arange(problem.n0, problem.n0 + steps), y


def ensemble(
    system_or_problem: Any,
    ics: Any,
    *,
    final_time: float = 100.0,
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
            method=method,
            rtol=rtol,
            atol=atol,
            jit=(backend == "jit"),
        ),
        dtype=np.float64,
    )


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
    method: str,
    rtol: float,
    atol: float,
    jit: bool,
) -> np.ndarray:
    """Dispatch a parallel ensemble integration (final states) to the engine."""
    return eng.integrate_ensemble_final(
        *_primary_tape(problem).to_arrays(),
        ics,
        problem.params_vec(),
        float(t0),
        float(t1),
        method,
        float(rtol),
        float(atol),
        bool(jit),
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
