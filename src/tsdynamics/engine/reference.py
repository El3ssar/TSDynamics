"""Reference backend — the pure-Python lowering oracle / dependency-light fallback.

The ``backend="reference"`` path: it needs no compiled engine, evaluating the
lowered IR tape in pure Python (the :mod:`tsdynamics.engine.compile` reference
evaluator) and delegating ODE time-stepping to SciPy.  It is the dependency-light
oracle the lowering is validated against, and a usable fallback for ODE/map runs.

Split out of :mod:`tsdynamics.engine.run` (the run-split refactor); every name
here stays reachable as ``tsdynamics.engine.run.<name>`` via re-export, so this is
a pure move.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.errors import ConvergenceError

from .compile import eval_tape
from .problem import MapProblem, ODEProblem, Problem


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
    from .run import _name

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
