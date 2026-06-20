r"""
Sample entropy and approximate entropy.

These are correlation-integral statistics rather than members of the
outcome-space family: they count how often short template windows recur within
a tolerance ``r``, and are reported in nats.

References
----------
Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
*Proc. Natl. Acad. Sci. USA* **88**, 2297–2301.

Richman, J. S. & Moorman, J. R. (2000). Physiological time-series analysis using
approximate entropy and sample entropy. *Am. J. Physiol. Heart Circ. Physiol.*
**278**, H2039–H2049.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import as_series

__all__ = ["approximate_entropy", "sample_entropy"]


def _embed(x: np.ndarray, m: int, tau: int, n_templates: int) -> np.ndarray:
    """First ``n_templates`` length-``m`` delay vectors of ``x``."""
    idx = np.arange(n_templates)[:, None] + np.arange(m)[None, :] * tau
    return x[idx]


def _resolve_r(x: np.ndarray, r: float | None) -> float:
    if r is None:
        return 0.2 * float(x.std())
    return float(r)


def sample_entropy(
    x: Any,
    m: int = 2,
    r: float | None = None,
    tau: int = 1,
    *,
    component: int | str | None = None,
) -> float:
    r"""
    Sample entropy (Richman & Moorman 2000).

    ``SampEn = -ln(A / B)``, where ``B`` and ``A`` count template pairs (excluding
    self-matches) that stay within Chebyshev tolerance ``r`` over windows of
    length ``m`` and ``m+1`` respectively.  Excluding self-matches removes the
    bias of approximate entropy, making the statistic largely independent of
    series length.  Larger values mean less regularity.

    Parameters
    ----------
    x : array-like or Trajectory
        Scalar time series (or a component of a multivariate one).
    m : int, default 2
        Template length.
    r : float, optional
        Tolerance (Chebyshev radius).  Defaults to ``0.2 · std(x)``.
    tau : int, default 1
        Embedding delay.
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    float
        Sample entropy in nats.  Returns ``inf`` when no length-``m+1`` template
        pair matches (no regularity detected at that scale).

    Raises
    ------
    ValueError
        If the series is too short or no length-``m`` template pair matches
        (``r`` too small).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> sample_entropy(rng.random(2000))   # white noise → high
    2.2...
    """
    series = as_series(x, component)
    n = series.size
    n_templates = n - m * tau  # common index set for m and m+1
    if n_templates <= 1:
        raise ValueError(f"series too short: need > {m * tau + 1} samples for m={m}, tau={tau}.")
    rad = _resolve_r(series, r)

    emb_m = _embed(series, m, tau, n_templates)
    emb_m1 = _embed(series, m + 1, tau, n_templates)

    b_count = 0  # length-m matches (ordered, self excluded)
    a_count = 0  # length-(m+1) matches
    for i in range(n_templates):
        dist_m = np.max(np.abs(emb_m - emb_m[i]), axis=1)
        within = dist_m <= rad
        within[i] = False  # exclude self-match
        b_count += int(within.sum())
        if within.any():
            dist_m1 = np.max(np.abs(emb_m1[within] - emb_m1[i]), axis=1)
            a_count += int(np.count_nonzero(dist_m1 <= rad))

    if b_count == 0:
        raise ValueError("no length-m template matches — increase r or lengthen the series.")
    if a_count == 0:
        return float("inf")
    return -np.log(a_count / b_count)


def approximate_entropy(
    x: Any,
    m: int = 2,
    r: float | None = None,
    tau: int = 1,
    *,
    component: int | str | None = None,
) -> float:
    r"""
    Approximate entropy (Pincus 1991).

    ``ApEn = Φ^m(r) − Φ^{m+1}(r)``, where ``Φ^m`` averages ``ln C_i^m`` and
    ``C_i^m`` is the fraction of length-``m`` templates within tolerance ``r`` of
    template ``i`` (*including* the self-match).  Self-matching keeps the
    logarithms finite but biases the estimate downward for short series — prefer
    :func:`sample_entropy` when that bias matters.

    Parameters
    ----------
    x : array-like or Trajectory
        Scalar time series (or a component of a multivariate one).
    m : int, default 2
        Template length.
    r : float, optional
        Tolerance (Chebyshev radius).  Defaults to ``0.2 · std(x)``.
    tau : int, default 1
        Embedding delay.
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    float
        Approximate entropy in nats.
    """
    series = as_series(x, component)
    n = series.size
    rad = _resolve_r(series, r)

    def phi(mm: int) -> float:
        n_templates = n - (mm - 1) * tau
        if n_templates <= 0:
            raise ValueError(f"series too short for m={mm}, tau={tau}.")
        emb = _embed(series, mm, tau, n_templates)
        log_sum = 0.0
        for i in range(n_templates):
            dist = np.max(np.abs(emb - emb[i]), axis=1)
            c = np.count_nonzero(dist <= rad)  # includes self-match (i)
            log_sum += np.log(c / n_templates)
        return log_sum / n_templates

    return float(phi(m) - phi(m + 1))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
