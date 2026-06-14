r"""
Multiscale entropy: coarse-grain a series across temporal scales, then quantify.

References
----------
Costa, M., Goldberger, A. L. & Peng, C.-K. (2002). Multiscale entropy analysis
of complex physiologic time series. *Phys. Rev. Lett.* **89**, 068102.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .core import as_series
from .sample import sample_entropy

__all__ = ["coarse_grain", "multiscale_entropy"]


def coarse_grain(x: Any, scale: int, *, component: int | str | None = None) -> np.ndarray:
    r"""
    Non-overlapping coarse-graining at a given temporal scale (Costa et al. 2002).

    Splits the series into consecutive, non-overlapping windows of ``scale``
    samples and replaces each by its mean, yielding a series of length
    ``⌊n / scale⌋``.

    Parameters
    ----------
    x : array-like or Trajectory
        Scalar time series.
    scale : int
        Window length, ``scale ≥ 1`` (``scale = 1`` returns the series itself).
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    numpy.ndarray
        The coarse-grained series.
    """
    if scale < 1:
        raise ValueError("scale must be ≥ 1.")
    series = as_series(x, component)
    if scale == 1:
        return series.copy()
    n_windows = series.size // scale
    if n_windows == 0:
        raise ValueError(f"series of length {series.size} too short for scale {scale}.")
    return series[: n_windows * scale].reshape(n_windows, scale).mean(axis=1)


def multiscale_entropy(
    x: Any,
    scales: int | Iterable[int] = 20,
    *,
    entropy_fn: Callable[..., float] = sample_entropy,
    r: float | None = None,
    r_factor: float = 0.15,
    component: int | str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    r"""
    Multiscale entropy: an entropy measured on coarse-grained copies of a series.

    For each scale the series is coarse-grained (:func:`coarse_grain`) and
    ``entropy_fn`` is applied.  Following Costa et al. (2002), when the chosen
    entropy takes a tolerance ``r`` (e.g. :func:`sample_entropy`,
    :func:`approximate_entropy`) it is **fixed across all scales** from the
    *original* series' standard deviation, so the profile reflects structure
    rather than the shrinking variance of the coarse-grained signal.

    Pure (1/f-like) processes hold a roughly flat profile while uncorrelated
    white noise decays with scale — the discriminating feature multiscale
    entropy was designed to expose.

    Parameters
    ----------
    x : array-like or Trajectory
        Scalar time series.
    scales : int or iterable of int, default 20
        An ``int`` ``S`` expands to ``1, 2, …, S``; an iterable is used verbatim.
    entropy_fn : callable, default :func:`sample_entropy`
        Single-series entropy applied at each scale.  Any of this package's
        scalar entropies works (sample, approximate, permutation, dispersion).
    r : float, optional
        Explicit tolerance for ``r``-based entropies (overrides ``r_factor``).
    r_factor : float, default 0.15
        When ``r`` is not given and ``entropy_fn`` accepts an ``r`` argument, use
        ``r_factor · std(original)``.
    component : int or str, optional
        Component selector for multi-component input.
    **kwargs
        Forwarded to ``entropy_fn`` (e.g. ``m=``, ``tau=``).

    Returns
    -------
    numpy.ndarray
        Entropy at each requested scale (same order as ``scales``).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> mse = multiscale_entropy(rng.standard_normal(5000), scales=5)
    >>> bool(mse[0] > mse[-1])    # white noise decays with scale
    True
    """
    series = as_series(x, component)
    scale_list = list(range(1, int(scales) + 1)) if isinstance(scales, int) else list(scales)

    # Fix the tolerance from the original series if the entropy uses one.
    call_kwargs = dict(kwargs)
    takes_r = "r" in inspect.signature(entropy_fn).parameters
    if takes_r and "r" not in call_kwargs:
        call_kwargs["r"] = r if r is not None else r_factor * float(series.std())

    out = np.empty(len(scale_list), dtype=float)
    for k, s in enumerate(scale_list):
        out[k] = entropy_fn(coarse_grain(series, s), **call_kwargs)
    return out
