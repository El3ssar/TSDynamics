"""Pick a smooth, non-pixelated output ``dt`` for a system's documentation plot.

The viewer (:mod:`threejs_viewer`) and the static figure renderer (:mod:`figures`)
want an output sampling step that is coarse enough to keep the inlined payload /
PNG light, yet fine enough that the attractor curve reads as smooth rather than
faceted.  :func:`choose_plot_dt` runs a short *pilot* integration of the system and
hands its samples to :func:`tsdynamics.analysis.sampling.estimate_dt_from_sagitta`,
which returns the largest stride whose per-triple *sagitta* (the bow of the curve
off its local chord) stays inside a geometric tolerance — i.e. the coarsest step
that does not visibly straighten the curve.

The helper is deliberately *robust*: an editorial ``plot_dt`` override short-circuits
the heuristic, and **any** failure (a system that will not integrate from a random
start, a too-short pilot, an import problem) falls back to ``dt0`` rather than
raising — a docs build must never break on the choice of a plotting step.
"""

from __future__ import annotations

import numpy as np


def choose_plot_dt(
    entry,
    *,
    final_time: float,
    dt0: float,
    override: float | None = None,
    epsilon: float = 0.04,
) -> float:
    """Return a smooth output ``dt`` for ``entry`` (an editorial override, else sagitta).

    Parameters
    ----------
    entry
        A registry ``SystemEntry`` (carries ``.cls`` / ``.family``).
    final_time, dt0
        Pilot-run integration window and base step (``dt0`` is also the fallback).
    override
        An editorial ``plot_dt`` override; returned verbatim when a positive float.
    epsilon
        Geometric (sagitta) tolerance — larger ⇒ coarser ``dt``.  Default ``0.04``.

    Returns
    -------
    float
        The chosen output ``dt`` (always ``>= dt0``), or ``dt0`` on any failure.
    """
    if override is not None and float(override) > 0.0:
        return float(override)
    try:
        from tsdynamics.analysis.sampling import estimate_dt_from_sagitta

        sys_obj = entry.cls()
        if entry.family == "dde":
            traj = sys_obj.integrate(
                final_time=final_time,
                dt=dt0,
                history=lambda s: [0.8 + 0.2 * np.sin(0.2 * s)] * sys_obj.dim,
            )
        elif entry.family == "map":
            traj = sys_obj.iterate(steps=int(final_time / dt0), max_retries=15)
        else:
            traj = sys_obj.integrate(final_time=final_time, dt=dt0, backend="interp", method="rk4")
        y = np.asarray(traj.y, dtype=float)
        if y.ndim != 2 or len(y) < 50 or not np.all(np.isfinite(y)):
            return float(dt0)
        result = estimate_dt_from_sagitta(y, float(dt0), epsilon=epsilon)
        return max(float(dt0), float(result.delta_t))
    except Exception:  # noqa: BLE001 — never let a plotting-step choice break the build
        return float(dt0)
