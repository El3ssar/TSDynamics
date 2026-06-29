"""Per-family runners + the low-level engine FFI shims.

Split out of :mod:`tsdynamics.engine.run` (the run-split refactor); every name
here stays reachable as ``tsdynamics.engine.run.<name>`` via re-export, so this is
a pure move.

This module owns:

- the per-family integrate/iterate runners :func:`_run_continuous` (ODE),
  :func:`_run_dde` (the Rust method-of-steps) and :func:`_run_map`, plus the lean
  per-step ODE core :func:`_step_continuous` (stream WS-INVHOIST) and the DDE
  past-sampler :func:`_sample_past`;
- the low-level engine dispatch shims (the E7 binding surface)
  :func:`_engine_integrate_dense` / :func:`_engine_ensemble_final` /
  :func:`_engine_map_ensemble_final` that marshal a Problem to the call the
  compiled extension exposes.

Run-side helpers (``_engine``/``_name``/``_primary_tape``/
``EngineNotAvailableError``) are late-imported from :mod:`tsdynamics.engine.run`
inside the functions that need them, so importing this module does not create an
import cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.errors import ConvergenceError

from .problem import DDEProblem, MapProblem, ODEProblem, Problem, SDEProblem
from .reference import _reference_map


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
    from .reference import _reference_ode
    from .run import _engine, _name, _primary_tape

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
    from .run import _engine

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
    from .run import EngineNotAvailableError, _engine, _name

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
    from .run import _engine, _name

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

    The returned array is typed ``float64`` here, the single conversion point —
    :func:`_run_continuous` consumes it as-is rather than re-wrapping it.
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
        ),
        dtype=np.float64,
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
    from .run import _primary_tape

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
