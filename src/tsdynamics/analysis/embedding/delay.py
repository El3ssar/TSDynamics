r"""
Choosing the embedding delay :math:`\tau`.

The delay trades redundancy against irrelevance: too small and successive
coordinates are nearly identical (the reconstruction collapses onto the
diagonal); too large and they become causally unrelated.  Two standard criteria
are provided:

- **Autocorrelation** — the linear measure.  A common rule takes :math:`\tau` as
  the first lag at which the autocorrelation drops to :math:`1/e`, or its first
  zero crossing.
- **Time-delayed mutual information** (Fraser & Swinney, 1986) — the nonlinear
  measure, and the more widely recommended one.  The first local *minimum* of
  :math:`I(\tau)` marks the delay at which :math:`x_{i+\tau}` adds the most new
  information about the state while still being dynamically related to
  :math:`x_i`.

:func:`optimal_delay` returns a single recommended :math:`\tau`;
:func:`mutual_information` and :func:`autocorrelation` expose the underlying
curves for inspection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._result import ArrayResult, CountResult
from ._common import _as_series

__all__ = ["autocorrelation", "mutual_information", "optimal_delay"]


def autocorrelation(
    data: Any, *, max_delay: int = 50, component: int | str | None = None
) -> np.ndarray:
    r"""Normalised autocorrelation function up to ``max_delay``.

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``component``).
    max_delay : int, default 50
        Largest lag returned.  Clamped to ``N - 1``.
    component : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    ndarray, shape (max_delay + 1,)
        ``acf[k]`` is the autocorrelation at lag ``k`` (``acf[0] == 1``).

    Notes
    -----
    Computed via FFT (Wiener--Khinchin) on the mean-subtracted series and
    normalised by the zero-lag value, so it is an unbiased-in-mean estimate
    (each lag divided by the full zero-lag variance, the standard convention).
    """
    x = _as_series(data, component=component)
    n = x.size
    max_delay = int(max_delay)
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative.")
    max_delay = min(max_delay, n - 1)

    x = x - x.mean()
    var = float(x @ x)
    if var == 0.0:
        raise ValueError("series is constant; autocorrelation is undefined.")

    # Linear (non-circular) autocorrelation via zero-padded FFT.
    size = int(2 ** np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(x, size)
    acf_full = np.fft.irfft(f * np.conj(f), size)[: max_delay + 1]
    return acf_full / var


def _auto_bins(n: int) -> int:
    """Default histogram bin count for the mutual-information estimate."""
    return int(np.clip(np.sqrt(n / 5.0), 16, 128))


def mutual_information(
    data: Any,
    *,
    max_delay: int = 50,
    bins: int | None = None,
    base: float = np.e,
    component: int | str | None = None,
) -> ArrayResult:
    r"""Time-delayed mutual information :math:`I(\tau)` up to ``max_delay``.

    The histogram estimator of

    .. math::

        I(\tau) = \sum_{a,b} p_{ab}(\tau)\,
                  \log\frac{p_{ab}(\tau)}{p_a\, p_b},

    where :math:`p_{ab}` is the joint distribution of :math:`(x_i, x_{i+\tau})`
    over an equal-width 2-D histogram and :math:`p_a, p_b` its marginals.

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``component``).
    max_delay : int, default 50
        Largest lag returned.  Clamped to ``N - 2``.
    bins : int, optional
        Number of histogram bins per axis.  Default: a sample-size-dependent
        rule, ``clip(sqrt(N/5), 16, 128)``.
    base : float, default ``e``
        Logarithm base — ``e`` for nats, ``2`` for bits.  Only rescales the
        curve; the location of the first minimum is unaffected.
    component : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    ArrayResult
        Behaves as an ``(max_delay + 1,)`` ``ndarray``: ``mi[k]`` is
        :math:`I(k)`; ``mi[0]`` is the entropy of the (binned) series itself
        (its self-information).

    References
    ----------
    A. M. Fraser and H. L. Swinney, "Independent coordinates for strange
    attractors from mutual information", *Phys. Rev. A* **33**, 1134 (1986).
    """
    x = _as_series(data, component=component)
    n = x.size
    max_delay = int(max_delay)
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative.")
    max_delay = min(max_delay, n - 2)
    nbins = int(bins) if bins is not None else _auto_bins(n)
    if nbins < 2:
        raise ValueError("bins must be >= 2.")
    log = np.log if base == np.e else (lambda v: np.log(v) / np.log(base))

    # A shared, fixed bin grid over the series range keeps marginals consistent
    # across lags (Fraser--Swinney use one partition of the data range).
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        raise ValueError("series is constant; mutual information is undefined.")
    edges = np.linspace(lo, hi, nbins + 1)
    # Pre-bin every sample once; the lagged pair (a, b) just indexes shifted views.
    codes = np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)

    mi = np.empty(max_delay + 1, dtype=float)
    for tau in range(max_delay + 1):
        a = codes[: n - tau]
        b = codes[tau:] if tau > 0 else codes
        joint = np.zeros((nbins, nbins), dtype=float)
        np.add.at(joint, (a, b), 1.0)
        total = joint.sum()
        joint /= total
        p_a = joint.sum(axis=1)
        p_b = joint.sum(axis=0)
        mask = joint > 0.0
        outer = p_a[:, None] * p_b[None, :]
        mi[tau] = float(np.sum(joint[mask] * log(joint[mask] / outer[mask])))
    return ArrayResult(values=mi, meta={"analysis": "mutual_information", "max_delay": max_delay})


def _first_local_min(curve: np.ndarray) -> int | None:
    """Index of the first interior local minimum (strict on the rising side)."""
    for k in range(1, curve.size - 1):
        if curve[k] <= curve[k - 1] and curve[k] < curve[k + 1]:
            return k
    return None


def optimal_delay(
    data: Any,
    *,
    method: str = "mi",
    max_delay: int = 50,
    bins: int | None = None,
    component: int | str | None = None,
) -> CountResult:
    r"""Recommend an embedding delay :math:`\tau` (in samples).

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``component``).
    method : {"mi", "acf", "acf_zero"}, default "mi"
        - ``"mi"`` — first local minimum of the time-delayed mutual information
          (Fraser & Swinney); the recommended nonlinear criterion.
        - ``"acf"`` — first lag where the autocorrelation falls to ``1/e``.
        - ``"acf_zero"`` — first lag where the autocorrelation crosses zero.
    max_delay : int, default 50
        Largest lag considered.
    bins : int, optional
        Histogram bins for the mutual-information estimate (``method="mi"``).
    component : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    CountResult
        The recommended delay (behaves as an ``int``), always ``>= 1``.

    Notes
    -----
    If no first minimum / crossing is found within ``max_delay`` (e.g. a
    slowly-decaying curve), the criterion's global fallback is used — the
    location of the smallest mutual information, or ``max_delay`` for the
    autocorrelation rules — so a usable delay is always returned.
    """
    method = method.lower()
    if method == "mi":
        mi = np.asarray(
            mutual_information(data, max_delay=max_delay, bins=bins, component=component)
        )
        k = _first_local_min(mi)
        if k is None:  # monotone / no interior dip: fall back to the global minimum
            k = int(np.argmin(mi[1:])) + 1 if mi.size > 1 else 1
        tau = max(int(k), 1)
    elif method in ("acf", "acf_zero"):
        acf = autocorrelation(data, max_delay=max_delay, component=component)
        if method == "acf":
            below = np.flatnonzero(acf[1:] <= 1.0 / np.e)
        else:
            below = np.flatnonzero(acf[1:] <= 0.0)
        # never crosses → longest lag available
        tau = int(below[0]) + 1 if below.size else max(int(acf.size) - 1, 1)
    else:
        raise ValueError(f"unknown method {method!r}; use 'mi', 'acf', or 'acf_zero'.")

    return CountResult(value=int(tau), meta={"analysis": "optimal_delay", "method": method})


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
