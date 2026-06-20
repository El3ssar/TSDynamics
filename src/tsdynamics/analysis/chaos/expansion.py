r"""
Expansion entropy (Hunt & Ott 2015) — a definition of chaos via volume growth.

For a region :math:`S` of state space, expansion entropy measures the
exponential growth rate of the *expansion* the linearised dynamics produce on
trajectories that remain in :math:`S`.  Writing :math:`G(A)` for the product of
the singular values of a matrix :math:`A` that exceed 1 (the volume growth of
the unit ball under :math:`A`, restricted to expanding directions), and
:math:`DF^{t}` for the fundamental/tangent matrix accumulated over :math:`t`,

.. math::

    E(t) = \frac{1}{N}\!\!\sum_{\substack{i:\ \text{orbit stays in } S}}\!\! G\big(DF^{t}_i\big),
    \qquad H = \lim_{t\to\infty} \frac{\ln E(t)}{t},

estimated by sampling :math:`N` initial conditions uniformly in :math:`S` and
reading :math:`H` as the slope of :math:`\ln E(t)` against :math:`t`.  Positive
:math:`H` is chaos.  For a uniformly expanding map :math:`H` is exact — the tent
map with unit height has :math:`|f'| \equiv 2`, so :math:`E(t) = 2^t` and
:math:`H = \ln 2`.

Supported systems are discrete maps (exact tangent map) and flows (RK4
variational fundamental matrix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tsdynamics.data import Box, sampler
from tsdynamics.families import ContinuousSystem, DiscreteMap

from . import _common as _c

__all__ = ["ExpansionEntropyResult", "expansion_entropy"]


@dataclass(frozen=True)
class ExpansionEntropyResult:
    r"""An expansion-entropy estimate with the growth curve it was read from.

    ``float(result)`` is the entropy :math:`H` (the fitted slope), so the result
    drops straight into arithmetic and thresholds.

    Attributes
    ----------
    entropy : float
        The estimated expansion entropy :math:`H` (slope of :math:`\ln E` vs
        :math:`t`).
    stderr : float
        Standard error of the slope over the fitted range.
    times : ndarray
        The :math:`t` grid (iterations for maps, time for flows).
    log_growth : ndarray
        :math:`\ln E(t)` at each :math:`t`.
    n_samples : int
        Number of initial conditions sampled in the region.
    n_survivors : int
        How many of them stayed in the region for the whole run.
    fit_slice : tuple[int, int]
        Inclusive ``(lo, hi)`` indices of the fitted range in ``times``.
    intercept : float
        Intercept of the fitted line.
    """

    entropy: float
    stderr: float
    times: np.ndarray = field(repr=False)
    log_growth: np.ndarray = field(repr=False)
    n_samples: int
    n_survivors: int
    fit_slice: tuple[int, int]
    intercept: float

    def __float__(self) -> float:
        """Return the entropy number, so the result drops into arithmetic."""
        return float(self.entropy)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ExpansionEntropyResult(entropy={self.entropy:.4g} ± {self.stderr:.2g}, "
            f"survivors={self.n_survivors}/{self.n_samples})"
        )


def expansion_entropy(
    system: Any,
    region: Any = None,
    *,
    n_samples: int = 1000,
    steps: int | None = None,
    final_time: float | None = None,
    dt: float | None = None,
    fit_range: tuple[int, int] | None = None,
    seed: int | None = 0,
    n_internal: int = 5,
) -> ExpansionEntropyResult:
    r"""Estimate the expansion entropy :math:`H` of a map or flow on a region.

    Parameters
    ----------
    system : DiscreteMap or ContinuousSystem
        The system whose expansion to measure.
    region : Box, (lo, hi), or None
        The restricting region :math:`S`.  ``None`` uses the (10%-expanded)
        bounding box of a burn-in orbit.
    n_samples : int, default 1000
        Number of initial conditions sampled uniformly in the region.
    steps : int, optional
        Number of iterations (maps).  Default 15.  (Kept modest: the raw tangent
        product is not renormalised, so very long horizons overflow.)
    final_time : float, optional
        Integration time (flows).  Default 5.0.
    dt : float, optional
        Recording step for flows.  Default 0.1.  Not valid for maps.
    fit_range : (int, int), optional
        Inclusive index range in the :math:`t` grid to fit the slope over.
        Default skips :math:`t = 0` and uses the rest.
    seed : int, optional
        Seed for the initial-condition sampling.
    n_internal : int, default 5
        RK4 sub-steps per ``dt`` for flows.

    Returns
    -------
    ExpansionEntropyResult

    Examples
    --------
    >>> float(expansion_entropy(Tent(params={"mu": 1.0}), Box([0.0], [1.0])))
    0.69...                                              # ln 2, exact

    References
    ----------
    Hunt & Ott (2015), *Chaos* 25, 097618 ("Defining chaos").
    """
    if isinstance(system, DiscreteMap):
        mode = "map"
    elif isinstance(system, ContinuousSystem):
        mode = "flow"
    else:
        raise NotImplementedError(
            f"expansion_entropy supports discrete maps and continuous flows, not "
            f"{type(system).__name__}."
        )

    box = _c._resolve_region(system, region)
    if box.dim != system.dim:
        raise ValueError(
            f"region dimension ({box.dim}) does not match system dimension ({system.dim})."
        )
    draw = sampler(box, seed=seed)
    ics = np.array([draw() for _ in range(int(n_samples))])

    if mode == "map":
        if dt is not None:
            raise ValueError("dt has no meaning for a discrete map — omit it.")
        n_steps = 15 if steps is None else int(steps)
        times, log_growth, survivors = _expansion_map(system, ics, box, n_steps)
    else:
        if steps is not None:
            raise ValueError("steps applies to maps; use final_time/dt for a flow.")
        t_end = 5.0 if final_time is None else float(final_time)
        step_dt = 0.1 if dt is None else float(dt)
        times, log_growth, survivors = _expansion_flow(
            system, ics, box, t_end, step_dt, int(n_internal)
        )

    lo, hi = _resolve_fit_range(times, log_growth, fit_range)
    # Fit only over finite ln E(t): once every sample has left S, E = 0 and ln E
    # = -inf (a contiguous tail in practice, but mask defensively rather than
    # rely on that invariant).
    t_fit, y_fit = times[lo : hi + 1], log_growth[lo : hi + 1]
    finite = np.isfinite(y_fit)
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError(
            "expansion entropy: fewer than two finite ln E(t) points in the fit range "
            "(too few survivors stayed in the region — enlarge it or shorten the horizon)."
        )
    slope, intercept, stderr = _c._linfit(t_fit[finite], y_fit[finite])
    return ExpansionEntropyResult(
        entropy=slope,
        stderr=stderr,
        times=times,
        log_growth=log_growth,
        n_samples=int(n_samples),
        n_survivors=int(survivors),
        fit_slice=(lo, hi),
        intercept=intercept,
    )


def _resolve_fit_range(
    times: np.ndarray, log_growth: np.ndarray, fit_range: tuple[int, int] | None
) -> tuple[int, int]:
    """Choose the fit window: explicit, else skip ``t=0`` and any non-finite tail."""
    n = times.size
    if fit_range is not None:
        lo, hi = int(fit_range[0]), int(fit_range[1])
        if not (0 <= lo < hi < n):
            raise ValueError(f"fit_range {fit_range} out of bounds for {n} points.")
        return lo, hi
    finite = np.nonzero(np.isfinite(log_growth))[0]
    if finite.size < 2:
        raise ValueError("expansion entropy: fewer than two finite ln E(t) points to fit.")
    lo = int(finite[0])
    if lo == 0 and finite.size > 2:
        lo = int(finite[1])  # drop t=0 (ln E = 0 trivially)
    hi = int(finite[-1])
    return lo, hi


def _expansion_map(
    system: Any, ics: np.ndarray, box: Box, n_steps: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate ``G(DF^t)`` for a map; ``DF^t = J(x_{t-1})...J(x_0)`` per sample."""
    step, jac = _c._map_fns(system)
    dim = int(system.dim)
    n = ics.shape[0]
    eye = np.eye(dim)
    mats = [eye.copy() for _ in range(n)]
    states = [np.asarray(ic, dtype=float).ravel() for ic in ics]
    alive = np.ones(n, dtype=bool)

    times = np.arange(0, n_steps + 1, dtype=float)
    ln_e = np.empty(n_steps + 1)
    ln_e[0] = 0.0  # E(0) = mean of G(I) = 1
    for t in range(1, n_steps + 1):
        total = 0.0
        for i in range(n):
            if not alive[i]:
                continue
            x = states[i]
            mats[i] = jac(x) @ mats[i]
            x = step(x)
            states[i] = x
            if not box.contains(x):
                alive[i] = False
                continue
            total += _c.expansion_volume(mats[i])
        e = total / n
        ln_e[t] = np.log(e) if e > 0.0 else -np.inf
    return times, ln_e, int(np.count_nonzero(alive))


def _expansion_flow(
    system: Any, ics: np.ndarray, box: Box, final_time: float, dt: float, n_internal: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate ``G(Phi(t))`` for a flow; ``Phi`` is the RK4 fundamental matrix per sample."""
    rhs, jac = _c._flow_fns(system)
    dim = int(system.dim)
    n = ics.shape[0]
    n_steps = int(round(final_time / dt))
    h = dt / max(1, n_internal)

    states = [np.asarray(ic, dtype=float).ravel() for ic in ics]
    mats = [np.eye(dim) for _ in range(n)]
    alive = np.ones(n, dtype=bool)
    t_local = np.zeros(n)

    times = np.empty(n_steps + 1)
    times[0] = 0.0
    ln_e = np.empty(n_steps + 1)
    ln_e[0] = 0.0
    for s in range(1, n_steps + 1):
        total = 0.0
        for i in range(n):
            if not alive[i]:
                continue
            x, m, t = states[i], mats[i], t_local[i]
            for _ in range(n_internal):
                x, m = _c._rk4_variational(rhs, jac, x, m, t, h)
                t += h
            states[i], mats[i], t_local[i] = x, m, t
            if not box.contains(x):
                alive[i] = False
                continue
            total += _c.expansion_volume(m)
        e = total / n
        times[s] = s * dt
        ln_e[s] = np.log(e) if e > 0.0 else -np.inf
    return times, ln_e, int(np.count_nonzero(alive))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
