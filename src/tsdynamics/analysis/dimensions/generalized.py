r"""
Generalized (Rényi) dimensions :math:`D_q` by box counting.

Partition state space into a grid of boxes of side :math:`\epsilon` and let
:math:`p_i(\epsilon)` be the fraction of points in occupied box :math:`i`.  The
Rényi / generalized dimension spectrum (Hentschel & Procaccia, *Physica D*
**8**, 435, 1983; Grassberger, *Phys. Lett. A* **97**, 227, 1983) is

.. math::

    D_q = \frac{1}{q-1}\,
          \lim_{\epsilon \to 0} \frac{\log \sum_i p_i(\epsilon)^q}{\log \epsilon},
    \qquad
    D_1 = \lim_{\epsilon \to 0} \frac{\sum_i p_i \log p_i}{\log \epsilon}.

Special cases: :math:`D_0` is the box-counting (capacity) dimension, :math:`D_1`
the information dimension, :math:`D_2` the correlation dimension.  Each is the
slope of a partition ordinate against :math:`\log \epsilon` in the scaling
region.

Occupied boxes are found by integer-flooring the (shifted) coordinates and
taking unique rows, so cost scales with the number of *occupied* boxes, never
the exponential full-grid size.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import DimensionResult, _as_points, _default_scales
from ._scaling import fit_scaling_region

__all__ = [
    "box_counting_dimension",
    "dimension_spectrum",
    "generalized_dimension",
    "information_dimension",
]


def _occupancy(points: np.ndarray, eps: float, mins: np.ndarray) -> np.ndarray:
    """Point counts of the occupied boxes of side ``eps`` (origin at ``mins``)."""
    coords = np.floor((points - mins) / eps).astype(np.int64)
    _, counts = np.unique(coords, axis=0, return_counts=True)
    return counts


def _informative_mask(occ: list[np.ndarray], n: int, sat_frac: float) -> np.ndarray:
    r"""Scales where the box partition carries dimension information.

    Drops the degenerate ends of the box-counting curve: large boxes that have
    collapsed to one or two cells (no slope), and small boxes that have
    saturated to (nearly) one point each — a perfectly straight *flat* plateau
    of slope :math:`\approx 0` that would otherwise capture the scaling-region
    fit and return a spurious near-zero dimension.  A box partition is
    informative while ``2 <= n_boxes <= sat_frac * N``.
    """
    n_boxes = np.array([c.size for c in occ])
    return (n_boxes >= 2) & (n_boxes <= sat_frac * n)


def _partition_ordinate(counts: np.ndarray, n: int, q: float) -> float:
    r"""Ordinate whose slope vs :math:`\log\epsilon` is :math:`D_q`.

    Returns :math:`\sum_i p_i \log p_i` for :math:`q = 1` and
    :math:`\log(\sum_i p_i^q)/(q-1)` otherwise.
    """
    p = counts / n
    if abs(q - 1.0) < 1e-12:
        return float(np.sum(p * np.log(p)))
    return float(np.log(np.sum(p**q)) / (q - 1.0))


def _fit_masked(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, min_window: int, tol: float, what: str
):
    """Fit a scaling region on the informative (masked) sub-range of a curve."""
    if int(mask.sum()) < min_window:
        raise ValueError(
            f"only {int(mask.sum())} informative box sizes for {what} (need >= {min_window}); the "
            "scales are saturated or too coarse. Pass a wider/denser `scales` or more data."
        )
    return fit_scaling_region(x[mask], y[mask], min_window=min_window, tol=tol)


def generalized_dimension(
    data: Any,
    q: float = 2.0,
    *,
    scales: np.ndarray | None = None,
    n_scales: int = 18,
    sat_frac: float = 0.85,
    min_window: int = 5,
    tol: float = 1.5,
) -> DimensionResult:
    r"""Generalized (Rényi) dimension :math:`D_q` by box counting.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    q : float, default 2.0
        Rényi order.  ``q=0`` is box-counting, ``q=1`` information, ``q=2``
        correlation; non-integer ``q`` is allowed.
    scales : ndarray, optional
        Box sizes :math:`\epsilon`.  Default: a log-spaced grid spanning the
        attractor diameter
        (:func:`~tsdynamics.analysis.dimensions._common._default_scales`).
    n_scales : int, default 18
        Number of box sizes when ``scales`` is not given.
    sat_frac : float, default 0.85
        Drop scales whose occupied-box count exceeds ``sat_frac * N`` — the
        saturated small-box regime where each box holds about one point and the
        curve flattens spuriously.
    min_window : int, default 5
        Minimum number of box sizes in the fitted scaling region.
    tol : float, default 1.5
        Scaling-region residual tolerance.

    Returns
    -------
    DimensionResult
        ``float(result)`` is :math:`D_q`.

    References
    ----------
    H. G. E. Hentschel and I. Procaccia, "The infinite number of generalized
    dimensions of fractals and strange attractors", *Physica D* **8**, 435
    (1983).
    """
    points = _as_points(data)
    n = points.shape[0]
    mins = points.min(axis=0)
    if scales is None:
        scales = _default_scales(points, n_scales=n_scales)
    scales = np.asarray(scales, dtype=float)
    if np.any(scales <= 0.0):
        raise ValueError("box sizes (scales) must be positive.")

    order = np.argsort(scales)
    scales = scales[order]
    occ = [_occupancy(points, e, mins) for e in scales]
    x = np.log(scales)
    y = np.array([_partition_ordinate(c, n, q) for c in occ])
    mask = _informative_mask(occ, n, sat_frac)
    fit = _fit_masked(x, y, mask, min_window=min_window, tol=tol, what=f"D_{q:g}")
    where = np.nonzero(mask)[0]
    lo, hi = int(where[fit.lo]), int(where[fit.hi])
    return DimensionResult(
        dimension=fit.slope,
        stderr=fit.stderr,
        kind="generalized",
        x=x,
        y=y,
        fit_slice=(lo, hi),
        intercept=fit.intercept,
        q=float(q),
    )


def box_counting_dimension(data: Any, **kwargs: Any) -> DimensionResult:
    r"""Box-counting (capacity) dimension :math:`D_0`.

    Thin wrapper over :func:`generalized_dimension` with ``q=0``.  Keyword
    arguments are forwarded.
    """
    return generalized_dimension(data, 0.0, **kwargs)


def information_dimension(data: Any, **kwargs: Any) -> DimensionResult:
    r"""Information dimension :math:`D_1`.

    Thin wrapper over :func:`generalized_dimension` with ``q=1``.  Keyword
    arguments are forwarded.
    """
    return generalized_dimension(data, 1.0, **kwargs)


def dimension_spectrum(
    data: Any,
    qs: Any = None,
    *,
    scales: np.ndarray | None = None,
    n_scales: int = 18,
    sat_frac: float = 0.85,
    min_window: int = 5,
    tol: float = 1.5,
) -> dict[float, DimensionResult]:
    r"""Compute the :math:`D_q` spectrum over several Rényi orders.

    Computes box occupancies once per scale and reuses them across every ``q``,
    so the whole spectrum costs barely more than a single :math:`D_q`.  For a
    monofractal the spectrum is flat; a decreasing :math:`D_q` signals
    multifractality.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    qs : array-like, optional
        Rényi orders.  Default: ``[0, 1, 2, 3, 4, 5]``.
    scales : ndarray, optional
        Box sizes; default as in :func:`generalized_dimension`.
    n_scales, sat_frac, min_window, tol
        As in :func:`generalized_dimension`.

    Returns
    -------
    dict[float, DimensionResult]
        ``{q: DimensionResult}`` in the order of ``qs``.
    """
    points = _as_points(data)
    n = points.shape[0]
    mins = points.min(axis=0)
    if qs is None:
        qs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    qs = [float(q) for q in np.atleast_1d(qs)]
    if scales is None:
        scales = _default_scales(points, n_scales=n_scales)
    scales = np.asarray(scales, dtype=float)

    order = np.argsort(scales)
    scales = scales[order]
    occ = [_occupancy(points, e, mins) for e in scales]
    x = np.log(scales)
    mask = _informative_mask(occ, n, sat_frac)
    where = np.nonzero(mask)[0]

    out: dict[float, DimensionResult] = {}
    for q in qs:
        y = np.array([_partition_ordinate(c, n, q) for c in occ])
        fit = _fit_masked(x, y, mask, min_window=min_window, tol=tol, what=f"D_{q:g}")
        out[q] = DimensionResult(
            dimension=fit.slope,
            stderr=fit.stderr,
            kind="generalized",
            x=x,
            y=y,
            fit_slice=(int(where[fit.lo]), int(where[fit.hi])),
            intercept=fit.intercept,
            q=q,
        )
    return out


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
