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

from . import _common as _c

__all__ = ["zero_one_test"]


def zero_one_test(
    observable: Any,
    *,
    component: int | None = None,
    n_c: int = 100,
    c_range: tuple[float, float] = (np.pi / 5.0, 4.0 * np.pi / 5.0),
    n_cut: int | None = None,
    seed: int | None = 0,
    return_distribution: bool = False,
) -> float | tuple[float, np.ndarray]:
    r"""Run the 0--1 test for chaos on a scalar observable.

    Parameters
    ----------
    observable : array-like or Trajectory
        A 1-D series, or a :class:`~tsdynamics.data.Trajectory` (pass
        ``component`` to select a column of a multi-component trajectory).  A
        live system is rejected — integrate/iterate it first.
    component : int, optional
        Column to use when a multi-component trajectory is passed.
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
    float or (float, ndarray)
        The median growth rate :math:`K \in [0, 1]` (``~0`` regular, ``~1``
        chaotic); with ``return_distribution`` also the ``K_c`` values.

    Examples
    --------
    >>> x = Logistic(params={"r": 4.0}).iterate(steps=5000).component("x")
    >>> zero_one_test(x) > 0.9          # chaotic
    True

    References
    ----------
    Gottwald & Melbourne (2004), *Proc. R. Soc. A* 460, 603--611.
    Gottwald & Melbourne (2009), *SIAM J. Appl. Dyn. Syst.* 8, 129--145.
    """
    phi = _c._as_observable(observable, component)
    n = phi.size
    if n < 200:
        raise ValueError(
            f"the 0-1 test needs a long series to be meaningful; got {n} points (need >= 200)."
        )
    if n_cut is None:
        n_cut = n // 10
    n_cut = int(max(1, min(n_cut, n - 1)))
    if n_c < 1:
        raise ValueError(f"n_c must be >= 1, got {n_c}.")

    rng = np.random.default_rng(seed)
    c_values = rng.uniform(c_range[0], c_range[1], size=int(n_c))

    j = np.arange(1, n + 1, dtype=float)
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
    return (k, k_c) if return_distribution else k


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
