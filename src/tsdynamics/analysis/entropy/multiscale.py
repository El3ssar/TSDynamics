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

from .._result import ArrayResult
from .core import as_series
from .sample import sample_entropy

__all__ = ["coarse_grain", "multiscale_entropy", "multiscale_entropy_plot_spec"]


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
    return np.asarray(series[: n_windows * scale].reshape(n_windows, scale).mean(axis=1))


def multiscale_entropy(
    data: Any,
    scales: int | Iterable[int] = 20,
    *,
    entropy_fn: Callable[..., Any] = sample_entropy,
    r: float | None = None,
    r_factor: float = 0.15,
    component: int | str | None = None,
    **kwargs: Any,
) -> ArrayResult:
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
    data : array-like or Trajectory
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
        Forwarded to ``entropy_fn`` (e.g. ``dimension=``, ``delay=``).

    Returns
    -------
    ArrayResult
        Entropy at each requested scale (same order as ``scales``), a drop-in for
        the bare array (``mse[0]``, ``mse.shape``, ``np.asarray(mse)``) that also
        carries ``.meta``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> mse = multiscale_entropy(rng.standard_normal(5000), scales=5)
    >>> bool(mse[0] > mse[-1])    # white noise decays with scale
    True
    """
    series = as_series(data, component)
    scale_list = list(range(1, int(scales) + 1)) if isinstance(scales, int) else list(scales)

    # Fix the tolerance from the original series if the entropy uses one.
    call_kwargs = dict(kwargs)
    takes_r = "r" in inspect.signature(entropy_fn).parameters
    if takes_r and "r" not in call_kwargs:
        call_kwargs["r"] = r if r is not None else r_factor * float(series.std())

    out = np.empty(len(scale_list), dtype=float)
    for k, s in enumerate(scale_list):
        out[k] = float(entropy_fn(coarse_grain(series, s), **call_kwargs))
    return ArrayResult(
        values=out,
        meta={"analysis": "multiscale_entropy", "scales": [int(s) for s in scale_list]},
    )


def multiscale_entropy_plot_spec(result: ArrayResult, kind: str | None = None) -> Any:
    r"""Describe a multiscale-entropy profile as a backend-agnostic :class:`PlotSpec`.

    Renders the result of :func:`multiscale_entropy` as a *complexity curve* —
    the entropy measured at each temporal scale factor against that scale
    (Costa et al. 2002).  The scale factors are read from the result's
    ``meta["scales"]`` when present (the order :func:`multiscale_entropy`
    records), else taken as ``1, 2, …`` against the entropy index.  The shape of
    this curve is the discriminating feature: a flat / rising profile signals
    long-range correlation, a profile that decays with scale signals
    uncorrelated noise.

    The spec carries a ``LINE`` plus a ``MARKERS`` layer over the same
    ``(scale, entropy)`` points, under the ``COMPLEXITY_CURVE`` semantic kind.
    The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls
    a plotting library; this is a pure viz adapter and does not touch the
    estimator.

    Parameters
    ----------
    result : ArrayResult
        The profile returned by :func:`multiscale_entropy` — entropy at each
        requested scale, with ``meta["scales"]`` carrying the scale factors.
    kind : str, optional
        Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
        value).  ``None`` uses ``COMPLEXITY_CURVE``.

    Returns
    -------
    PlotSpec
        A ``COMPLEXITY_CURVE`` spec of entropy against scale factor.

    Raises
    ------
    ValueError
        If the result carries no entropy values (nothing to plot).
    """
    from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

    spec_kind = PlotKind(kind) if kind is not None else PlotKind.COMPLEXITY_CURVE
    y = np.asarray(result.values, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("multiscale-entropy profile is empty: nothing to plot.")
    meta = dict(result.meta) if result.meta else {}
    scales = meta.get("scales")
    if scales is not None and len(scales) == y.size:
        x = np.asarray(scales, dtype=float)
    else:
        x = np.arange(1.0, y.size + 1.0)
    layers = [
        Layer(PlotKind.LINE, {"x": x, "y": y}, label="entropy"),
        Layer(PlotKind.MARKERS, {"x": x, "y": y}, label="scales"),
    ]
    return PlotSpec(
        kind=spec_kind,
        ndim=2,
        title="Multiscale entropy",
        x=Axis(label="scale factor"),
        y=Axis(label="entropy"),
        layers=layers,
        legend=Legend(),
        meta=meta,
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
