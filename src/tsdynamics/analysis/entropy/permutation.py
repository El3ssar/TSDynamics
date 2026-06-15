r"""
Permutation entropy and its amplitude-weighted variant.

References
----------
Bandt, C. & Pompe, B. (2002). Permutation entropy: a natural complexity measure
for time series. *Phys. Rev. Lett.* **88**, 174102.

Fadlallah, B., Chen, B., Keil, A. & Príncipe, J. (2013). Weighted-permutation
entropy: a complexity measure for time series incorporating amplitude
information. *Phys. Rev. E* **87**, 022911.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import OrdinalPatterns, Shannon, as_series, entropy

__all__ = ["permutation_entropy", "weighted_permutation_entropy"]


def permutation_entropy(
    x: Any,
    m: int = 3,
    tau: int = 1,
    *,
    base: float = 2.0,
    normalize: bool = True,
    component: int | str | None = None,
) -> float:
    r"""
    Permutation entropy of a time series (Bandt & Pompe 2002).

    The Shannon entropy of the distribution of *ordinal patterns* — the relative
    rankings within every length-``m`` embedded window.  It is invariant under
    any strictly monotonic transform of the data and needs no amplitude
    thresholds, which makes it a robust complexity measure for noisy
    experimental series.

    Parameters
    ----------
    x : array-like or Trajectory
        Scalar time series (or a component of a multivariate one).
    m : int, default 3
        Ordinal (embedding) order.  Typical choices are ``3 ≤ m ≤ 7``; the
        series should satisfy ``n ≫ m!`` for the estimate to be meaningful.
    tau : int, default 1
        Embedding delay.
    base : float, default 2.0
        Logarithm base (``2`` → bits).
    normalize : bool, default True
        Divide by ``log_base(m!)`` so the result lies in ``[0, 1]`` (``0`` for a
        perfectly ordered signal, ``1`` for uniform pattern usage).
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    float
        Permutation entropy.

    Examples
    --------
    >>> permutation_entropy(np.arange(1000))          # monotone → 0
    0.0
    >>> rng = np.random.default_rng(0)
    >>> permutation_entropy(rng.random(10000))        # white noise → ≈ 1
    0.99...
    """
    return entropy(
        x,
        outcomes=OrdinalPatterns(m, tau),
        measure=Shannon(base),
        normalize=normalize,
        component=component,
    )


def weighted_permutation_entropy(
    x: Any,
    m: int = 3,
    tau: int = 1,
    *,
    base: float = 2.0,
    normalize: bool = True,
    component: int | str | None = None,
) -> float:
    r"""
    Weighted permutation entropy (Fadlallah et al. 2013).

    Like :func:`permutation_entropy`, but each ordinal pattern is weighted by the
    variance of its embedding window before the probabilities are formed.  This
    restores sensitivity to amplitude — high-variance (often more dynamically
    relevant) windows count more, so abrupt large-amplitude events are no longer
    treated like tiny fluctuations sharing the same ordinal pattern.

    Parameters
    ----------
    x, m, tau, base, normalize, component
        As in :func:`permutation_entropy`.

    Returns
    -------
    float
        Weighted permutation entropy.
    """
    series = as_series(x, component)
    space = OrdinalPatterns(m, tau)
    labels, weights = space.window_variance(series)
    weighted = np.bincount(labels, weights=weights, minlength=space.cardinality)
    total = weighted.sum()
    if total == 0:
        # All windows are flat (zero variance) → no amplitude information.
        return 0.0
    p = weighted / total
    h = Shannon(base).apply(p)
    if normalize:
        hmax = Shannon(base).maximum(space.cardinality)
        return h / hmax if hmax > 0 else 0.0
    return h
