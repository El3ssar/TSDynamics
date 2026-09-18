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

from typing import Any, NamedTuple

import numpy as np

from .._common import runaway_meta
from ._common import (
    _DEFAULT_C_HI,
    DimensionResult,
    _as_points,
    _metric_p,
    _pnorm,
    _radii_for_c_window,
    _resolve_theiler,
    require_min_points,
)
from ._scaling import fit_scaling_region

__all__ = ["CorrelationSum", "correlation_dimension", "correlation_sum"]


class CorrelationSum(NamedTuple):
    r"""The correlation-sum curve: radii and :math:`C(r)`, each under its own name.

    A :class:`~typing.NamedTuple`, so it still unpacks exactly like the bare
    2-tuple it replaced — ``r, C = correlation_sum(x)`` — while ``help()``, the
    repr and tab-completion all say which array is which.  A returned pair of
    unlabelled arrays is a small puzzle every caller has to solve from the
    docstring; a named one is not.

    Attributes
    ----------
    radii : numpy.ndarray
        The radii :math:`r` the sum was evaluated at, ascending.
    sums : numpy.ndarray
        :math:`C(r)` at each radius, normalised to ``[0, 1]``.
    """

    radii: np.ndarray
    sums: np.ndarray


def _near_diagonal_distances(points: np.ndarray, w: int, p: float) -> np.ndarray:
    """Distances of all pairs ``(i, i+off)`` with ``1 <= off <= w`` (the Theiler band)."""
    n = points.shape[0]
    chunks = [_pnorm(points[:-off] - points[off:], p) for off in range(1, w + 1) if off < n]
    return np.concatenate(chunks) if chunks else np.empty(0)


def correlation_sum(
    data: Any,
    radii: np.ndarray | None = None,
    *,
    theiler: int | str = "auto",
    metric: str | float = "euclidean",
    n_radii: int = 24,
    c_lo: float | None = None,
    c_hi: float = _DEFAULT_C_HI,
) -> CorrelationSum:
    r"""Correlation sum :math:`C(r)` over a grid of radii — the raw curve.

    This is the *curve* :func:`correlation_dimension` fits a slope to, exposed
    on its own for anyone who wants to choose the scaling region by eye rather
    than let the fitter choose it.  ``correlation_dimension(data)`` carries the
    same data in log form (``.abscissa`` / ``.ordinate``), so reach for this one
    when you want :math:`r` and :math:`C(r)` themselves — a plot, a custom fit, a
    comparison against an analytic :math:`C(r)` — and for the *dimension* reach
    for :func:`correlation_dimension`.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set (a :class:`~tsdynamics.data.Trajectory` or a raw array; a
        1-D series is treated as a single component).
    radii : ndarray, optional
        Radii at which to evaluate :math:`C(r)`.  Default: a grid spanning the
        informative window of :math:`C(r)` itself — see ``c_lo`` / ``c_hi`` and
        :func:`~tsdynamics.analysis.dimensions._common._radii_for_c_window`.
    theiler : int or "auto", default "auto"
        Exclude pairs with :math:`|i - j| \le w`.  ``"auto"`` reads ``w`` off the
        **space--time separation profile** — the first lag at which points that
        far apart in time are typically half a typical pair-distance apart in
        state space (:func:`~tsdynamics.analysis.dimensions._common._auto_theiler`).
        That is ``1`` for an already-decorrelated point cloud or map orbit, and a
        few tens of samples for a densely-sampled flow.  Pass ``0`` for the raw,
        uncorrected sum.
    metric : str or float, default "euclidean"
        Distance metric (``"euclidean"``, ``"chebyshev"``, ``"manhattan"``, or a
        Minkowski exponent).
    n_radii : int, default 24
        Number of radii when ``radii`` is not given.
    c_lo : float, optional
        Lower target value of :math:`C(r)` for the automatic grid.  ``None``
        uses ``1e-4``, raised if that would put fewer than 200 pairs below the
        smallest radius.
    c_hi : float, default 0.1
        Upper target value of :math:`C(r)` for the automatic grid.

    Returns
    -------
    CorrelationSum
        A named 2-tuple ``(radii, sums)`` — the radii and the corresponding
        correlation-sum values, normalised so ``sums`` lies in ``[0, 1]``.  It
        unpacks like the plain tuple it replaced (``r, C = correlation_sum(x)``)
        and also names its halves, so a reader never has to count positions to
        find out which array is which.

    Raises
    ------
    ValueError
        If the Theiler window leaves no valid pairs, or ``theiler`` is neither a
        non-negative int nor ``"auto"``.
    """
    out_radii, c, _w = _correlation_sum_from_points(
        _as_points(data, analysis="correlation_sum"),
        radii,
        theiler=theiler,
        metric=metric,
        n_radii=n_radii,
        c_lo=c_lo,
        c_hi=c_hi,
    )
    return CorrelationSum(out_radii, c)


def _correlation_sum_from_points(
    points: np.ndarray,
    radii: np.ndarray | None = None,
    *,
    theiler: int | str = "auto",
    metric: str | float = "euclidean",
    n_radii: int = 24,
    c_lo: float | None = None,
    c_hi: float = _DEFAULT_C_HI,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(radii, C(radii), w)`` for an already-coerced ``(N, dim)`` point set.

    The shared core of :func:`correlation_sum` (which coerces ``data`` first) and
    :func:`correlation_dimension` (which coerces once and reuses the array), so
    the point set is validated/copied a single time per estimate.  The k-d tree
    and the Theiler band are built once and reused by both the pilot sweep that
    picks the default radii and the final evaluation.

    The *resolved* Theiler window is returned alongside the curve so the caller
    can record it: with ``theiler="auto"`` the window is data-dependent, and an
    estimate whose most consequential knob was chosen for the user must say which
    value it chose.
    """
    from scipy.spatial import cKDTree

    n = points.shape[0]
    w = _resolve_theiler(theiler, points)
    p = _metric_p(metric)

    tree = cKDTree(points)
    total_valid = n * (n - 1) / 2.0
    near_sorted: np.ndarray | None = None
    if w > 0:
        near_sorted = np.sort(_near_diagonal_distances(points, w, p))
        total_valid -= near_sorted.size
    if total_valid <= 0.0:
        raise ValueError(
            f"Theiler window w={w} excludes every pair for N={n}; reduce it or add data."
        )

    def c_of_r(rs: np.ndarray) -> np.ndarray:
        # count_neighbors counts ordered pairs incl. the N zero-distance
        # self-pairs:  counts = N + 2 * (#unordered pairs i<j within r)
        counts = tree.count_neighbors(tree, rs, p=p).astype(float)
        pairs_le = (counts - n) / 2.0
        if near_sorted is not None:
            pairs_le = pairs_le - np.searchsorted(near_sorted, rs, side="right").astype(float)
        return np.asarray(pairs_le / total_valid)

    if radii is None:
        radii = _radii_for_c_window(c_of_r, points, n_radii=n_radii, c_lo=c_lo, c_hi=c_hi)
    radii = np.asarray(radii, dtype=float)

    order = np.argsort(radii)
    c_sorted = c_of_r(radii[order])
    c = np.empty_like(c_sorted)
    c[order] = c_sorted
    return radii, c, w


def correlation_dimension(
    data: Any,
    *,
    theiler: int | str = "auto",
    metric: str | float = "euclidean",
    radii: np.ndarray | None = None,
    n_radii: int = 24,
    c_lo: float | None = None,
    c_hi: float = _DEFAULT_C_HI,
    min_window: int = 5,
    flatness: float = 1.5,
) -> DimensionResult:
    r"""Grassberger--Procaccia correlation dimension :math:`D_2`.

    Computes the correlation sum, then reads :math:`D_2` off the slope of
    :math:`\log C(r)` vs :math:`\log r` in the automatically selected scaling
    region (:func:`~tsdynamics.analysis.dimensions._scaling.fit_scaling_region`).

    **Not the same estimator as** ``generalized_dimension(data, 2.0)``, despite
    both being called :math:`D_2`: this one counts *pairwise distances* and the
    Rényi one counts *box occupancies*, and they disagree by more than round-off
    (measured on a Lorenz orbit: 2.09 vs 1.86).  Prefer this one — it is the
    standard :math:`D_2` estimator, it is far less sensitive to the box
    alignment, and it works down to smaller scales on a modest point set.  Reach
    for :func:`generalized_dimension` when you want the whole :math:`D_q`
    *spectrum* on one consistent box partition.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    theiler : int or "auto", default "auto"
        Theiler window — exclude pairs with :math:`|i - j| \le w` (see
        :func:`correlation_sum`).  ``"auto"`` reads the decorrelation time off
        the data, so a densely sampled flow is corrected without the caller
        having to know to ask; pass ``0`` for the uncorrected sum.
    metric : str or float, default "euclidean"
        Distance metric.
    radii : ndarray, optional
        Explicit radii.  The default grid spans the informative window of
        :math:`C(r)` — from ``c_lo`` to ``c_hi`` — rather than a fixed fraction
        of the attractor extent.
    n_radii : int, default 24
        Number of radii when ``radii`` is not given.
    c_lo, c_hi : float, optional
        Target :math:`C(r)` levels bracketing the automatic radius grid
        (defaults ``1e-4`` and ``0.1``); see :func:`correlation_sum`.
    min_window : int, default 5
        Minimum number of radii in the fitted scaling region.
    flatness : float, default 1.5
        How flat the fitted scaling region has to be: a window is admitted when
        its straight-line residual is within this factor of the flattest window
        found.  Larger admits wider but less straight regions.  (It is **not** a
        solver tolerance — that is what the v6 rename off ``tol=`` is for.)

    Returns
    -------
    DimensionResult
        ``float(result)`` is :math:`D_2`; the curve and selected window are
        carried for inspection.

    Notes
    -----
    Two defaults changed in v6, both because the old ones needed overriding to
    be right.  The radius grid used to span the 1st to the 50th percentile of the
    pair-distance distribution — i.e. up to :math:`C = 0.5`, deep inside the
    saturation bend — and the Theiler window used to default to ``0``, which
    counts temporally adjacent samples of a flow as genuine neighbours.  On the
    Hénon map the two together returned :math:`D_2 = 1.169` against the published
    :math:`1.220 \pm 0.005`; with the current defaults the same call returns
    ``1.206``, and — unlike before — stays there as the radius grid is refined.

    References
    ----------
    P. Grassberger and I. Procaccia, "Characterization of strange attractors",
    *Phys. Rev. Lett.* **50**, 346 (1983).

    J. Theiler, "Spurious dimension from correlation algorithms applied to
    limited time-series data", *Phys. Rev. A* **34**, 2427 (1986).

    Examples
    --------
    >>> d = correlation_dimension(lorenz_traj)                       # doctest: +SKIP
    >>> float(d)                                                     # doctest: +SKIP
    2.05...

    Raises
    ------
    InvalidParameterError
        If fewer than
        :data:`~tsdynamics.analysis.dimensions._common.MIN_DIMENSION_POINTS`
        points are given — too few to resolve a scaling region (it previously
        returned a spurious ``D_2 ~= 0``).
    """
    # ``_as_points`` rejects <2 points / non-finite first (keeping those messages);
    # then reject a too-short-but-finite handful rather than fabricating a slope.
    # Coerce once and reuse the array for the correlation sum (no double scan).
    points = _as_points(data, analysis="correlation_dimension")
    require_min_points(
        points,
        analysis="correlation_dimension",
        reason=(
            "the Grassberger-Procaccia correlation sum cannot resolve a scaling "
            "region from so few points"
        ),
    )
    radii, c, w = _correlation_sum_from_points(
        points,
        radii=radii,
        theiler=theiler,
        metric=metric,
        n_radii=n_radii,
        c_lo=c_lo,
        c_hi=c_hi,
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
    fit = fit_scaling_region(x, y, min_window=min_window, tol=flatness)
    return DimensionResult(
        estimate=fit.slope,
        stderr=fit.stderr,
        kind="correlation",
        abscissa=x,
        ordinate=y,
        fit_region=(fit.lo, fit.hi),
        intercept=fit.intercept,
        q=2.0,
        meta={
            "analysis": "correlation_dimension",
            "kind": "correlation",
            "q": 2.0,
            "theiler": w,
            # One coordinate means "points on a line" — the estimator answers
            # D ~ 1 whatever the attractor is, so the result says so.
            "n_components": int(points.shape[1]),
            # The escape stamp, when the point set is a runaway orbit: the
            # cleanest-looking fit in the library is the one taken through a
            # blow-up, so ``trusted`` reads this before it reads R^2.
            **runaway_meta(points, analysis="correlation_dimension"),
        },
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
