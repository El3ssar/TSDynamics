r"""
Automated scaling-region selection for log--log dimension estimators.

Every fractal-dimension estimator in this subpackage reduces to a slope on a
log--log plot: :math:`\log C(r)` vs :math:`\log r` for the correlation sum,
:math:`\log Z_q(\epsilon)/(q-1)` vs :math:`\log\epsilon` for the generalized
dimensions, :math:`\log k` vs :math:`\langle\log r_k\rangle` for fixed mass.
The dimension is the slope of the *scaling region* — the straight middle of the
curve.  It bends away from that line at small scales (finite sampling / noise)
and at large scales (saturation at the attractor diameter), so the slope must be
read off the linear portion rather than the whole range.

:func:`fit_scaling_region` selects that portion automatically by scanning every
contiguous window of at least ``min_window`` points, keeping those whose
straight-line residual is within ``tol`` of the best, and returning the *widest*
such window (ties broken toward the smaller residual).  This favours a long,
clean linear stretch over a short near-perfect one — the standard goal when
reading a dimension off a log--log plot (Theiler 1990).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ScalingFit", "fit_scaling_region", "local_slopes"]


@dataclass(frozen=True)
class ScalingFit:
    """Result of a scaling-region least-squares fit.

    Attributes
    ----------
    slope : float
        Fitted slope (the dimension estimate).
    intercept : float
        Fitted intercept.
    lo, hi : int
        Inclusive index bounds of the selected window into the input arrays.
    stderr : float
        Standard error of the slope.
    npts : int
        Number of points in the selected window.
    """

    slope: float
    intercept: float
    lo: int
    hi: int
    stderr: float
    npts: int


def _lstsq_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Ordinary least-squares line fit; returns ``(slope, intercept, sigma)``.

    ``sigma`` is the residual standard deviation ``sqrt(SSR / (n - 2))`` (0 for a
    two-point window).  Returns ``inf`` sigma for a degenerate (zero-spread) x.
    """
    n = x.size
    xm = float(x.mean())
    ym = float(y.mean())
    dx = x - xm
    sxx = float(np.dot(dx, dx))
    if sxx <= 0.0:
        return float("nan"), float("nan"), float("inf")
    slope = float(np.dot(dx, y - ym) / sxx)
    intercept = ym - slope * xm
    resid = y - (slope * x + intercept)
    ssr = float(np.dot(resid, resid))
    sigma = float(np.sqrt(ssr / (n - 2))) if n > 2 else 0.0
    return slope, intercept, sigma


def _slope_stderr(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """Compute the standard error of the slope of a least-squares line."""
    n = x.size
    if n <= 2:
        return 0.0
    dx = x - x.mean()
    sxx = float(np.dot(dx, dx))
    if sxx <= 0.0:
        return float("inf")
    resid = y - (slope * x + intercept)
    sigma2 = float(np.dot(resid, resid)) / (n - 2)
    return float(np.sqrt(sigma2 / sxx))


def fit_scaling_region(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_window: int = 5,
    tol: float = 1.5,
) -> ScalingFit:
    r"""Select and fit the linear scaling region of a log--log curve.

    Scans every contiguous window of at least ``min_window`` points, fits a
    line to each, and keeps the windows whose residual standard deviation is
    within a factor ``tol`` of the smallest seen.  Among those it returns the
    widest window (ties broken toward the smaller residual), so a long clean
    stretch is preferred over a short near-perfect one.

    Parameters
    ----------
    x, y : ndarray
        Abscissa and ordinate of the log--log curve, ordered by increasing
        scale.  Must be the same length.
    min_window : int, default 5
        Minimum number of points in the fitted window.
    tol : float, default 1.5
        A window is a candidate when its residual sigma is at most
        ``tol * sigma_min``.  Larger ``tol`` admits wider but slightly less
        straight windows.

    Returns
    -------
    ScalingFit

    Raises
    ------
    ValueError
        If fewer than ``min_window`` points are supplied or ``x`` and ``y``
        differ in length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")
    n = x.size
    if min_window < 2:
        raise ValueError("min_window must be at least 2.")
    if n < min_window:
        raise ValueError(
            f"need at least min_window={min_window} points to fit a scaling region, got {n}. "
            "Widen the scale range, add more data, or lower min_window."
        )

    # (sigma, width, lo, hi, slope, intercept) for every admissible window.
    candidates: list[tuple[float, int, int, int, float, float]] = []
    for lo in range(0, n - min_window + 1):
        for hi in range(lo + min_window - 1, n):
            slope, intercept, sigma = _lstsq_line(x[lo : hi + 1], y[lo : hi + 1])
            if np.isfinite(sigma):
                candidates.append((sigma, hi - lo + 1, lo, hi, slope, intercept))

    if not candidates:
        raise ValueError("No admissible scaling window (x has no spread). Check the input scales.")

    sigma_min = min(c[0] for c in candidates)
    threshold = sigma_min * tol if sigma_min > 0.0 else 0.0
    # Widest window within the residual threshold; tie-break toward smaller sigma.
    kept = [c for c in candidates if c[0] <= threshold + 1e-300]
    best = max(kept, key=lambda c: (c[1], -c[0]))
    _, width, lo, hi, slope, intercept = best
    stderr = _slope_stderr(x[lo : hi + 1], y[lo : hi + 1], slope, intercept)
    return ScalingFit(slope=slope, intercept=intercept, lo=lo, hi=hi, stderr=stderr, npts=width)


def local_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise local slope ``dy/dx`` of a log--log curve.

    Centered differences (one-sided at the ends, via :func:`numpy.gradient`), so
    non-uniform spacing is handled correctly.  The plateau of this curve is the
    scaling region; inspecting it is the standard sanity check on any reported
    fractal dimension.

    Parameters
    ----------
    x, y : ndarray
        Same-length abscissa and ordinate.

    Returns
    -------
    ndarray
        Local slope at each point (``nan`` if fewer than two points).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.full(x.shape, np.nan)
    return np.gradient(y, x)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
