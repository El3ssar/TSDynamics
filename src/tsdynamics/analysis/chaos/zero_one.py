r"""
The 0--1 test for chaos (Gottwald & Melbourne 2004, 2009).

A binary order/chaos diagnostic that acts directly on a *scalar observable*
:math:`\phi_j` of the dynamics — no phase-space reconstruction, no Jacobian.
For a frequency :math:`c` it drives the skew translation

.. math::

    p_c(n) = \sum_{j=1}^{n} \phi_j \cos(jc), \qquad
    q_c(n) = \sum_{j=1}^{n} \phi_j \sin(jc),

and measures the asymptotic growth of the mean-square displacement of
:math:`(p_c, q_c)`.  Regular dynamics keep :math:`(p_c, q_c)` bounded (the
displacement is bounded, growth rate :math:`K_c \approx 0`); chaotic dynamics
make it diffuse like a random walk (linear growth, :math:`K_c \approx 1`).  The
returned :math:`K` is the median of :math:`K_c` over many frequencies, obtained
by the regularised mean-square displacement + correlation method of the 2009
paper (more robust than fitting a growth exponent).

The observable should be sampled so successive values are not strongly
correlated (every iterate for a map; a suitable stride for a flow).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._result import AnalysisResult, ScalarResult
from . import _common as _c

__all__ = ["zero_one_test"]


def _observable(
    system: Any,
    component: int | None,
    *,
    final_time: float | None,
    n: int | None,
    dt: float | None,
    transient: float | None,
    ic: Any | None,
) -> np.ndarray:
    """Resolve the scalar observable from a System (integrate/iterate it) or data.

    A System produces its own decorrelated series — every iteration for a map or
    discrete view (Poincaré / stroboscopic), the ``dt``-grid for a flow.  A
    measured :class:`~tsdynamics.data.Trajectory` / ``ndarray`` is read directly
    (the ``data`` overload); the horizon keywords then do not apply.
    """
    is_system = hasattr(system, "is_discrete") and (
        hasattr(system, "trajectory")
        or hasattr(system, "iterate")
        or hasattr(system, "integrate")
        or hasattr(system, "_step")
    )
    if not is_system:
        if any(v is not None for v in (final_time, n, dt, transient, ic)):
            raise ValueError(
                "zero_one_test: final_time/n/dt/transient/ic apply only when the first "
                "argument is a System; a measured series / Trajectory is used as-is."
            )
        return _c._as_observable(system, component)
    if system.is_discrete:
        skip = int(transient) if transient is not None else 0
        count = int(n) if n is not None else 5000
        kw: dict[str, Any] = {"transient": skip}
        if ic is not None and hasattr(system, "iterate"):
            kw["ic"] = ic
        return _c._as_observable(system.trajectory(count, **kw), component)
    # continuous flow — sample on the dt grid (successive samples must be
    # decorrelated for the test to be meaningful; a coarse dt, or a Poincaré /
    # stroboscopic view passed as ``system``, gives the cleanest K).
    if n is not None:
        raise ValueError("zero_one_test: n is for maps/discrete views; a flow uses final_time.")
    horizon = float(final_time) if final_time is not None else 1000.0
    burn = float(transient) if transient is not None else 0.0
    step = float(dt) if dt is not None else 0.1
    traj = system.integrate(final_time=horizon + burn, dt=step, ic=ic)
    if burn:
        traj = traj.after(burn)
    return _c._as_observable(traj, component)


def zero_one_test(
    system: Any,
    *,
    component: int | None = None,
    final_time: float | None = None,
    n: int | None = None,
    dt: float | None = None,
    transient: float | None = None,
    ic: Any | None = None,
    n_c: int = 100,
    c_range: tuple[float, float] = (np.pi / 5.0, 4.0 * np.pi / 5.0),
    n_cut: int | None = None,
    seed: int | None = 0,
    return_distribution: bool = False,
) -> ScalarResult | tuple[ScalarResult, np.ndarray]:
    r"""Run the 0--1 test for chaos on a system or a measured observable.

    Parameters
    ----------
    system : System, Trajectory, or array-like
        A dynamical system (integrated / iterated internally to produce the
        observable, like :func:`~tsdynamics.analysis.chaos.gali`), or a measured
        1-D series / :class:`~tsdynamics.data.Trajectory` used directly (the
        ``data`` overload).  For a flow pass a coarse ``dt`` — or a Poincaré /
        stroboscopic view as ``system`` — so successive samples are decorrelated.
    component : int, optional
        Column to use when a multi-component system / trajectory is passed.
    final_time : float, optional
        Integration horizon for a flow (system input).  Default 1000.0.
    n : int, optional
        Number of iterations for a map / discrete view (system input).
        Default 5000.
    dt : float, optional
        Sampling / integration step for a flow (system input).  Default 0.1.
    transient : float, optional
        Discarded before recording — a flow burn-in **time**, a map / discrete
        burn-in in **steps** (system input).
    ic : array-like, optional
        Initial condition (system input).
    n_c : int, default 100
        Number of random frequencies :math:`c` drawn from ``c_range``.
    c_range : (float, float), default ``(pi/5, 4*pi/5)``
        Interval the frequencies are drawn from.  The default avoids the
        resonances near :math:`c = 0, \pi` (Gottwald & Melbourne 2009).
    n_cut : int, optional
        Largest displacement lag used in the mean-square displacement.  Default
        ``N // 10`` (the rule of thumb: stay well below the series length).
    seed : int, optional
        Seed for the frequency draw (makes :math:`K` reproducible).
    return_distribution : bool, default False
        If true, also return the per-frequency :math:`K_c` array.

    Returns
    -------
    ScalarResult or (ScalarResult, ndarray)
        The median growth rate :math:`K \in [0, 1]` (``~0`` regular, ``~1``
        chaotic) as a drop-in for its ``float`` value (``result > 0.9`` and
        ``float(result)`` work) carrying ``.meta``; with ``return_distribution``
        also the ``K_c`` values.

    Examples
    --------
    >>> zero_one_test(Logistic(params={"r": 4.0}), n=5000) > 0.9     # chaotic
    True
    >>> x = Logistic(params={"r": 4.0}).iterate(steps=5000).component("x")
    >>> zero_one_test(x) > 0.9          # the data overload
    True

    References
    ----------
    Gottwald & Melbourne (2004), *Proc. R. Soc. A* 460, 603--611.
    Gottwald & Melbourne (2009), *SIAM J. Appl. Dyn. Syst.* 8, 129--145.
    """
    phi = _observable(
        system, component, final_time=final_time, n=n, dt=dt, transient=transient, ic=ic
    )
    n_pts = phi.size
    if n_pts < 200:
        raise ValueError(
            f"the 0-1 test needs a long series to be meaningful; got {n_pts} points (need >= 200)."
        )
    if n_cut is None:
        n_cut = n_pts // 10
    n_cut = int(max(1, min(n_cut, n_pts - 1)))
    if n_c < 1:
        raise ValueError(f"n_c must be >= 1, got {n_c}.")

    rng = np.random.default_rng(seed)
    c_values = rng.uniform(c_range[0], c_range[1], size=int(n_c))

    j = np.arange(1, n_pts + 1, dtype=float)
    lags = np.arange(1, n_cut + 1, dtype=float)
    mean_phi_sq = float(np.mean(phi)) ** 2

    k_c = np.empty(int(n_c))
    for idx, c in enumerate(c_values):
        p = np.cumsum(phi * np.cos(j * c))
        q = np.cumsum(phi * np.sin(j * c))
        msd = np.empty(n_cut)
        for li in range(n_cut):
            lag = li + 1
            dp = p[lag:] - p[:-lag]
            dq = q[lag:] - q[:-lag]
            msd[li] = np.mean(dp * dp + dq * dq)
        # Regularised mean-square displacement: subtract the oscillatory term so
        # only the (diffusive) trend remains (Gottwald & Melbourne 2009, eq. 2.6).
        osc = mean_phi_sq * (1.0 - np.cos(lags * c)) / (1.0 - np.cos(c))
        d = msd - osc
        k_c[idx] = _c._pearson(lags, d)

    k = float(np.median(k_c))
    result = ScalarResult(value=k, meta=AnalysisResult.build_meta(system, analysis="zero_one_test"))
    return (result, k_c) if return_distribution else result


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
