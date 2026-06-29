"""SDE engine seam — the diagonal-Itô dense/ensemble entry points.

Split out of :mod:`tsdynamics.engine.run` (the run-split refactor); every name
here stays reachable as ``tsdynamics.engine.run.<name>`` via re-export, so this is
a pure move.

Unlike :func:`tsdynamics.engine.run.integrate` / :func:`tsdynamics.engine.run.ensemble`,
these carry the two SDE-specific knobs — the fixed step ``dt`` (which *is* the
noise scale ``√dt``) and the noise ``seed`` — and drive the two-tape engine call
(drift + diffusion).  The family base class
(:class:`~tsdynamics.families.stochastic.StochasticSystem`) calls these for its
``backend="interp"/"jit"`` path and wraps the result as a Trajectory with
provenance; the pure-Python reference path stays in the family.

The backend resolver + engine accessor (``resolve_backend``/``_engine``) are
late-imported from :mod:`tsdynamics.engine.run` inside the functions that need
them, so importing this module does not create an import cycle.
"""

from __future__ import annotations

import numpy as np

from .problem import SDEProblem


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
    from .run import _engine, resolve_backend

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
    from .run import _engine, resolve_backend

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
