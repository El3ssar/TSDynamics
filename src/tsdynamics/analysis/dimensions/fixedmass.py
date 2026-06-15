r"""
Fixed-mass (nearest-neighbour) dimension estimator.

Where the correlation sum fixes a radius and counts the enclosed mass, the
fixed-mass method (Badii & Politi, *J. Stat. Phys.* **40**, 725, 1985;
Grassberger, *Phys. Lett. A* **107**, 101, 1985) fixes the mass — the neighbour
count :math:`k` — and measures the radius :math:`r_k` needed to enclose it.  On a
fractal, :math:`k/N \sim r_k^{D}`, so averaging over reference points,

.. math::

    \langle \log r_k \rangle \approx \frac{1}{D}\,\log k + \text{const},

and :math:`D` is the slope of :math:`\log k` against
:math:`\langle \log r_k \rangle` in the scaling region.

Because it adapts the radius to the local density, the fixed-mass estimator
keeps a usable signal-to-noise ratio into the sparse tails of an attractor where
fixed-radius counts run out of pairs.  A Theiler window excludes temporally
correlated neighbours, exactly as for the correlation sum.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import DimensionResult, as_points, metric_p
from ._scaling import fit_scaling_region

__all__ = ["fixed_mass_dimension"]


def _kth_valid_distances(
    dists: np.ndarray, idx: np.ndarray, ref_index: np.ndarray, w: int, k: int
) -> np.ndarray:
    """For each reference row, the distance to its ``k``-th Theiler-valid neighbour.

    A neighbour ``j`` of reference ``i`` is valid when ``|i - j| > w`` (this also
    drops the self-match at distance 0).  Rows without ``k`` valid neighbours in
    the queried block yield ``nan``.
    """
    valid = np.abs(idx - ref_index[:, None]) > w
    csum = np.cumsum(valid, axis=1)
    # The k-th valid neighbour is the column where the running count first hits k.
    hit = valid & (csum == k)
    has_k = hit.any(axis=1)
    col = np.argmax(hit, axis=1)  # first True per row (0 where none, masked out below)
    out = np.full(dists.shape[0], np.nan)
    rows = np.nonzero(has_k)[0]
    out[rows] = dists[rows, col[rows]]
    return out


def fixed_mass_dimension(
    data: Any,
    *,
    ks: np.ndarray | None = None,
    theiler_window: int = 0,
    metric: str | float = "euclidean",
    n_ref: int | None = 1500,
    n_ks: int = 16,
    min_window: int = 5,
    tol: float = 1.5,
    seed: int = 0,
) -> DimensionResult:
    r"""Fixed-mass (nearest-neighbour) dimension.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    ks : array-like of int, optional
        Neighbour counts (masses) to probe.  Default: a log-spaced integer grid
        from 1 to ``N // 10``.
    theiler_window : int, default 0
        Exclude neighbours with :math:`|i - j| \le w` (set for dense flows).
    metric : str or float, default "euclidean"
        Distance metric.
    n_ref : int or None, default 1500
        Number of reference points averaged over (randomly sub-sampled, seeded).
        ``None`` uses every point.
    n_ks : int, default 16
        Number of masses when ``ks`` is not given.
    min_window : int, default 5
        Minimum number of masses in the fitted scaling region.
    tol : float, default 1.5
        Scaling-region residual tolerance.
    seed : int, default 0
        Seed for the reference sub-sample (keeps the estimate reproducible).

    Returns
    -------
    DimensionResult
        ``float(result)`` is the dimension; ``x`` is :math:`\langle\log r_k\rangle`
        and ``y`` is :math:`\log k`.

    References
    ----------
    R. Badii and A. Politi, "Statistical description of chaotic attractors: The
    dimension function", *J. Stat. Phys.* **40**, 725 (1985).
    """
    from scipy.spatial import cKDTree

    points = as_points(data)
    n = points.shape[0]
    w = int(theiler_window)
    if w < 0:
        raise ValueError("theiler_window must be non-negative.")
    p = metric_p(metric)

    if ks is None:
        k_hi = max(n // 10, min_window + 1)
        ks = np.unique(np.round(np.logspace(0, np.log10(k_hi), n_ks)).astype(int))
    ks = np.asarray(ks, dtype=int)
    ks = ks[(ks >= 1) & (ks < n)]
    if ks.size < min_window:
        raise ValueError(
            f"only {ks.size} usable masses (need >= {min_window}); supply more data or `ks`."
        )
    k_max = int(ks.max())

    rng = np.random.default_rng(seed)
    if n_ref is not None and n_ref < n:
        ref_index = np.sort(rng.choice(n, size=n_ref, replace=False))
    else:
        ref_index = np.arange(n)

    tree = cKDTree(points)
    # Query enough neighbours that k_max valid ones survive removing the <=2w+1
    # self/near-diagonal matches, with headroom for non-adjacent excluded points.
    k_query = min(n, k_max + 2 * w + 5)
    dists, idx = tree.query(points[ref_index], k=k_query, p=p)
    dists = np.atleast_2d(dists)
    idx = np.atleast_2d(idx)

    mean_log_r = np.empty(ks.size)
    for m, k in enumerate(ks):
        rk = _kth_valid_distances(dists, idx, ref_index, w, int(k))
        rk = rk[np.isfinite(rk) & (rk > 0.0)]
        if rk.size == 0:
            raise ValueError(
                f"no valid {k}-th neighbour found; increase n_ref or reduce the Theiler window."
            )
        mean_log_r[m] = float(np.mean(np.log(rk)))

    # D is the slope of log(k) vs <log r_k>; <log r_k> increases with k, so it is
    # the (sorted-ascending) abscissa and the index windows are contiguous in scale.
    x = mean_log_r
    y = np.log(ks.astype(float))
    order = np.argsort(x)
    x, y = x[order], y[order]
    fit = fit_scaling_region(x, y, min_window=min_window, tol=tol)
    return DimensionResult(
        dimension=fit.slope,
        stderr=fit.stderr,
        kind="fixed_mass",
        x=x,
        y=y,
        fit_slice=(fit.lo, fit.hi),
        intercept=fit.intercept,
        q=None,
    )
