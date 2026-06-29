r"""
Correlation sum and the Grassberger--Procaccia correlation dimension.

The correlation sum (Grassberger & Procaccia, *Physica D* **9**, 189, 1983)

.. math::

    C(r) = \frac{2}{N_{\text{pairs}}}
           \sum_{i < j} \Theta\!\big(r - \lVert x_i - x_j \rVert\big)

counts the fraction of point pairs closer than ``r``.  On a fractal it scales as
:math:`C(r) \sim r^{D_2}`, so the correlation dimension :math:`D_2` is the slope
of :math:`\log C(r)` against :math:`\log r` in the scaling region.

Temporally adjacent samples of a flow are spuriously close in state space and
inflate :math:`C(r)` at small ``r``, biasing :math:`D_2` downward.  The **Theiler
window** ``w`` (Theiler, *Phys. Rev. A* **34**, 2427, 1986) removes this by
counting only pairs with :math:`|i - j| > w`.

The pair counting is done with a k-d tree
(:meth:`scipy.spatial.cKDTree.count_neighbors`) — the box/tree-assisted range
search that replaces the naive :math:`O(N^2)` double loop — with an exact
:math:`O(Nw)` correction for the excluded near-diagonal pairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import invalid_value
from ._common import DimensionResult, _as_points, _default_radii, _metric_p, _pnorm
from ._scaling import fit_scaling_region

__all__ = ["correlation_dimension", "correlation_sum"]

#: Absolute floor on the number of points a Grassberger--Procaccia estimate needs.
#: Below this the correlation sum is dominated by its handful of pairs and the
#: scaling-region fit collapses to a spurious near-zero slope (it silently
#: returned ``D_2 ~= 0`` for, e.g., eight points).  Not a sufficiency guarantee —
#: a reliable estimate wants far more — just the floor below which the answer is
#: meaningless, so the input is rejected rather than fabricated.
_MIN_CORR_POINTS = 32


def _near_diagonal_distances(points: np.ndarray, w: int, p: float) -> np.ndarray:
    """Distances of all pairs ``(i, i+off)`` with ``1 <= off <= w`` (the Theiler band)."""
    n = points.shape[0]
    chunks = [_pnorm(points[:-off] - points[off:], p) for off in range(1, w + 1) if off < n]
    return np.concatenate(chunks) if chunks else np.empty(0)


def correlation_sum(
    data: Any,
    radii: np.ndarray | None = None,
    *,
    theiler: int = 0,
    metric: str | float = "euclidean",
    n_radii: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Correlation sum :math:`C(r)` over a grid of radii.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set (a :class:`~tsdynamics.data.Trajectory` or a raw array; a
        1-D series is treated as a single component).
    radii : ndarray, optional
        Radii at which to evaluate :math:`C(r)`.  Default: a data-adaptive
        log-spaced grid (:func:`~tsdynamics.analysis.dimensions._common._default_radii`).
    theiler : int, default 0
        Exclude pairs with :math:`|i - j| \le w`.  Use a few autocorrelation
        times for densely sampled flows; 0 for already-decorrelated point sets.
    metric : str or float, default "euclidean"
        Distance metric (``"euclidean"``, ``"chebyshev"``, ``"manhattan"``, or a
        Minkowski exponent).
    n_radii : int, default 24
        Number of radii when ``radii`` is not given.

    Returns
    -------
    (radii, C) : tuple[ndarray, ndarray]
        The radii and the corresponding correlation-sum values, normalised so
        ``C`` lies in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the Theiler window leaves no valid pairs.
    """
    return _correlation_sum_from_points(
        _as_points(data), radii, theiler=theiler, metric=metric, n_radii=n_radii
    )


def _correlation_sum_from_points(
    points: np.ndarray,
    radii: np.ndarray | None = None,
    *,
    theiler: int = 0,
    metric: str | float = "euclidean",
    n_radii: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation sum :math:`C(r)` for an already-coerced ``(N, dim)`` point set.

    The shared core of :func:`correlation_sum` (which coerces ``data`` first) and
    :func:`correlation_dimension` (which coerces once and reuses the array), so
    the point set is validated/copied a single time per estimate.
    """
    from scipy.spatial import cKDTree

    n = points.shape[0]
    w = int(theiler)
    if w < 0:
        raise ValueError("theiler must be non-negative.")
    p = _metric_p(metric)
    if radii is None:
        radii = _default_radii(points, p=p, n_radii=n_radii)
    radii = np.asarray(radii, dtype=float)

    order = np.argsort(radii)
    rs = radii[order]

    tree = cKDTree(points)
    # count_neighbors counts ordered pairs incl. the N zero-distance self-pairs:
    #   counts = N + 2 * (#unordered pairs i<j within r)
    counts = tree.count_neighbors(tree, rs, p=p).astype(float)
    pairs_le = (counts - n) / 2.0

    total_valid = n * (n - 1) / 2.0
    if w > 0:
        near = _near_diagonal_distances(points, w, p)
        near_sorted = np.sort(near)
        excluded_le = np.searchsorted(near_sorted, rs, side="right").astype(float)
        pairs_le = pairs_le - excluded_le
        total_valid -= near.size

    if total_valid <= 0.0:
        raise ValueError(
            f"Theiler window w={w} excludes every pair for N={n}; reduce it or add data."
        )

    c_sorted = pairs_le / total_valid
    c = np.empty_like(c_sorted)
    c[order] = c_sorted
    return radii, c


def correlation_dimension(
    data: Any,
    *,
    theiler: int = 0,
    metric: str | float = "euclidean",
    radii: np.ndarray | None = None,
    n_radii: int = 24,
    min_window: int = 5,
    tol: float = 1.5,
) -> DimensionResult:
    r"""Grassberger--Procaccia correlation dimension :math:`D_2`.

    Computes the correlation sum, then reads :math:`D_2` off the slope of
    :math:`\log C(r)` vs :math:`\log r` in the automatically selected scaling
    region (:func:`~tsdynamics.analysis.dimensions._scaling.fit_scaling_region`).

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    theiler : int, default 0
        Theiler window — exclude pairs with :math:`|i - j| \le w` (see
        :func:`correlation_sum`).  The default ``0`` suits a point set; flow
        users on a densely sampled trajectory should set a Theiler window to
        exclude temporally correlated neighbours, otherwise :math:`D_2` is biased
        downward.
    metric : str or float, default "euclidean"
        Distance metric.
    radii : ndarray, optional
        Explicit radii; default is a data-adaptive log-spaced grid.
    n_radii : int, default 24
        Number of radii when ``radii`` is not given.
    min_window : int, default 5
        Minimum number of radii in the fitted scaling region.
    tol : float, default 1.5
        Scaling-region residual tolerance (see
        :func:`~tsdynamics.analysis.dimensions._scaling.fit_scaling_region`).

    Returns
    -------
    DimensionResult
        ``float(result)`` is :math:`D_2`; the curve and selected window are
        carried for inspection.

    References
    ----------
    P. Grassberger and I. Procaccia, "Characterization of strange attractors",
    *Phys. Rev. Lett.* **50**, 346 (1983).

    Examples
    --------
    >>> d = correlation_dimension(lorenz_traj, theiler=50)   # doctest: +SKIP
    >>> float(d)                                                    # doctest: +SKIP
    2.05...

    Raises
    ------
    InvalidParameterError
        If fewer than :data:`_MIN_CORR_POINTS` points are given — too few to
        resolve a scaling region (it previously returned a spurious ``D_2 ~= 0``).
    """
    # ``_as_points`` rejects <2 points / non-finite first (keeping those messages);
    # then reject a too-short-but-finite handful rather than fabricating a slope.
    # Coerce once and reuse the array for the correlation sum (no double scan).
    points = _as_points(data)
    n = points.shape[0]
    if n < _MIN_CORR_POINTS:
        raise invalid_value(
            "data length",
            n,
            rule=f"must be >= {_MIN_CORR_POINTS} points for a correlation-dimension estimate",
            hint=(
                "the Grassberger-Procaccia correlation sum cannot resolve a scaling "
                "region from so few points; pass a longer trajectory / series."
            ),
        )
    radii, c = _correlation_sum_from_points(
        points,
        radii=radii,
        theiler=theiler,
        metric=metric,
        n_radii=n_radii,
    )
    mask = (c > 0.0) & (c < 1.0)
    if mask.sum() < min_window:
        raise ValueError(
            f"only {int(mask.sum())} usable radii (need >= {min_window}); the correlation sum is "
            "saturated or empty over this grid. Pass a wider/denser `radii` or more data."
        )
    # ``radii`` is returned in the caller's original order (which need not be
    # monotone, since ``radii`` is a public parameter), so sort the masked pair
    # by ascending log-radius: ``fit_scaling_region`` scans contiguous index
    # windows and requires inputs ordered by increasing scale.  Mirrors the
    # ``np.argsort`` pattern in ``generalized.py`` / ``fixedmass.py``.
    x = np.log(radii[mask])
    y = np.log(c[mask])
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    fit = fit_scaling_region(x, y, min_window=min_window, tol=tol)
    return DimensionResult(
        estimate=fit.slope,
        stderr=fit.stderr,
        kind="correlation",
        abscissa=x,
        ordinate=y,
        fit_region=(fit.lo, fit.hi),
        intercept=fit.intercept,
        q=2.0,
        meta={"analysis": "correlation_dimension", "kind": "correlation", "q": 2.0},
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
