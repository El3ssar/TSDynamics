"""Resumable ODE stepper handle (stream WS-STEPPER).

Split out of :mod:`tsdynamics.engine.run` (the run-split refactor); every name
here stays reachable as ``tsdynamics.engine.run.<name>`` via re-export, so this is
a pure move.

The durable replacement for the per-``dt`` :func:`tsdynamics.engine._families._step_continuous`
core: an opaque engine handle (``tsdynamics._rust.OdeStepper``) that owns the built
tape evaluator + solver once and carries the live integration point across calls. A
constant-/small-``dt`` stepping loop (``ContinuousSystem.step()``, Poincaré
refinement, basins over flows) then never rebuilds or re-marshals the tape — only
the live state is threaded — yet the numerics are unchanged: ``advance(dt)`` is
byte-for-byte identical to the per-``dt`` ``_step_continuous`` it supersedes (the
engine re-seeds a fresh solver/state each ``dt``, so the adaptive controller behaves
exactly as the released path; verified bit-for-bit in the Rust test
``stepper_advance_reproduces_per_dt_integrate_dense_bit_for_bit``).

The engine accessor (``_engine``) is late-imported from
:mod:`tsdynamics.engine.run` inside the functions that need it, so importing this
module does not create an import cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.errors import ConvergenceError


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
    from .run import _engine

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
