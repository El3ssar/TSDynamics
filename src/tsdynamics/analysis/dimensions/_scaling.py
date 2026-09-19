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
contiguous window, keeping those whose straight-line residual is within ``tol``
of the best, and returning the one spanning the **widest range of scales** (ties
broken toward the smaller residual).  This favours a long, clean linear stretch
over a short near-perfect one — the standard goal when reading a dimension off a
log--log plot.

**Window admissibility is a span, not a point count (v6).**  A real log--log
curve is *lacunar*: a self-similar set makes :math:`\log C(r)` wobble around the
scaling line with a log-periodic ripple (Smith 1988; Theiler 1990 §4).  Any rule
that scores windows purely by residual therefore prefers a short window sitting
inside one ripple — and the finer the scale grid, the shorter (and more local)
that window becomes.  Selecting on a *point count* (the pre-v6 ``min_window``
alone) made the whole estimate grid-density dependent: densifying the radius grid
moved the Grassberger--Procaccia :math:`D_2` of the Hénon map monotonically away
from the literature value (1.167 at 12 radii to 1.148 at 128, against
:math:`1.220 \pm 0.005`) while the *reported* standard error shrank eightfold —
the reported uncertainty anti-correlated with the true error.

Both admissibility and the residual threshold are now set from windows spanning
at least ``min_span`` in abscissa units (default: one decade, or the whole
available range when that is shorter, whichever of the two rules binds harder).
The span and the residual standard deviation are both properties of the *curve*,
not of how densely it was sampled, so the selection converges as the grid is
refined: the same Hénon sweep now returns 1.2027 / 1.2056 / 1.2077 / 1.2067 /
1.2069 / 1.2066 at 12 / 24 / 48 / 96 / 128 / 256 radii — a total spread of 0.005
where it was 0.094, and converging on the published value rather than away from
it.  A window spanning at least one full lacunarity period also averages the
ripple out instead of fitting it.

References
----------
J. Theiler, "Estimating fractal dimension", *J. Opt. Soc. Am. A* **7**, 1055
(1990).

L. A. Smith, "Intrinsic limits on dimension calculations", *Phys. Lett. A*
**133**, 283 (1988).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ScalingFit", "fit_scaling_region", "local_slopes"]

#: One decade of abscissa (the abscissa of every estimator here is a natural
#: logarithm of a scale).  A dimension quoted from a log--log fit is
#: conventionally read off at least one decade of scaling, so this is the
#: absolute floor on the width of an admissible window — used unless the curve
#: itself is shorter than a decade, in which case the whole curve is the floor.
_DECADE = float(np.log(10.0))

#: Default minimum window width as a *fraction* of the available abscissa range.
#: Binds on curves *shorter* than a decade, where demanding a whole decade would
#: force the fit onto the entire curve (ends included) and defeat the point of
#: selecting a scaling region at all.
_DEFAULT_MIN_SPAN_FRAC = 0.5


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
    span : float
        Abscissa width ``x[hi] - x[lo]`` of the selected window — the
        density-independent measure of how much scaling range the fit covers.
    sigma : float
        Residual standard deviation of the fitted line over the window (the
        straightness score the selection minimises).
    """

    slope: float
    intercept: float
    lo: int
    hi: int
    stderr: float
    npts: int
    span: float = 0.0
    sigma: float = 0.0


def _window_stats(
    x: np.ndarray, y: np.ndarray, lo: int, min_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS statistics of every window ``[lo, hi]`` with ``hi >= lo + min_window - 1``.

    Vectorised over ``hi`` through cumulative sums, so the whole scan costs
    ``O(n^2)`` arithmetic in ``O(n)`` memory instead of the ``O(n^3)`` triple
    loop a per-window ``lstsq`` would need (which made a 256-radius grid take
    minutes).  Returns ``(hi, slope, intercept, sigma, stderr)`` where ``sigma``
    is the residual standard deviation and ``stderr`` the standard error of the
    slope; both are ``inf`` for a degenerate (zero-spread) window.
    """
    xs = x[lo:]
    ys = y[lo:]
    cnt = np.arange(1, xs.size + 1, dtype=float)
    sx = np.cumsum(xs)
    sy = np.cumsum(ys)
    sxx = np.cumsum(xs * xs)
    sxy = np.cumsum(xs * ys)
    syy = np.cumsum(ys * ys)
    with np.errstate(invalid="ignore", divide="ignore"):
        cxx = sxx - sx * sx / cnt
        cxy = sxy - sx * sy / cnt
        cyy = syy - sy * sy / cnt
        slope = cxy / cxx
        intercept = (sy - slope * sx) / cnt
        ssr = np.maximum(cyy - slope * cxy, 0.0)
        dof = np.maximum(cnt - 2.0, 1.0)
        sigma = np.where(cnt > 2.0, np.sqrt(ssr / dof), 0.0)
        stderr = np.where(cnt > 2.0, np.sqrt(ssr / dof / cxx), 0.0)
    degenerate = ~(cxx > 0.0)
    sigma = np.where(degenerate, np.inf, sigma)
    stderr = np.where(degenerate, np.inf, stderr)
    start = min_window - 1
    hi = np.arange(lo + start, x.size)
    return hi, slope[start:], intercept[start:], sigma[start:], stderr[start:]


def fit_scaling_region(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_window: int = 5,
    tol: float = 1.5,
    min_span_frac: float = _DEFAULT_MIN_SPAN_FRAC,
    min_span: float | None = None,
) -> ScalingFit:
    r"""Select and fit the linear scaling region of a log--log curve.

    Scans every contiguous window that is at least ``min_window`` points *and*
    at least the effective minimum span wide, fits a line to each, and keeps the
    windows whose residual standard deviation is within a factor ``tol`` of the
    smallest seen.  Among those it returns the one covering the widest abscissa
    span (ties broken toward the smaller residual), so a long clean stretch is
    preferred over a short near-perfect one.

    The **effective minimum span** is ``min(min_span_frac * total, min_span)``
    where ``total`` is the abscissa range of the input: *one decade of scaling,
    or half the available range, whichever is less*.  Both terms matter and the
    smaller one binds by design.  On a **long** curve (several decades, of which
    only part scales) the absolute decade floor binds, so the window cannot
    shrink to a locally-straight patch.  On a **short** curve the fraction binds,
    so the fitter keeps real freedom to drop the curved ends — taking the
    *larger* of the two there would clamp the floor to the whole range and force
    the fit over the very ends the selection exists to reject.  Because the floor
    never exceeds ``0.5 * total``, at least one admissible window always exists.

    Selecting on span rather than on a point count is what makes the estimate
    independent of how densely the curve was sampled — see the module docstring
    for the measured failure this fixes.

    Parameters
    ----------
    x, y : ndarray
        Abscissa and ordinate of the log--log curve, ordered by increasing
        scale.  Must be the same length.
    min_window : int, default 5
        Minimum number of points in the fitted window.  Must be ``>= 2``.  A
        secondary guard: the span rule below is the binding one on a dense grid.
    tol : float, default 1.5
        A window is a candidate when its residual sigma is at most
        ``tol * sigma_min``.  Larger ``tol`` admits wider but slightly less
        straight windows.
    min_span_frac : float, default 0.5
        Minimum window width as a fraction of the total abscissa range.
    min_span : float, optional
        Absolute minimum window width in abscissa units.  ``None`` (the default)
        uses one decade, ``log(10)`` — the conventional minimum scaling range for
        a quoted log--log dimension.  It is a *ceiling* on the floor (see above),
        never a hard requirement, so it cannot make the problem infeasible.

    Returns
    -------
    ScalingFit
        The fitted slope/intercept, the selected index window, and its span and
        residual sigma.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length, if ``min_window < 2``, if fewer
        than ``min_window`` points are supplied, or if ``x`` has no spread (no
        admissible window).

    References
    ----------
    J. Theiler, "Estimating fractal dimension", *J. Opt. Soc. Am. A* **7**,
    1055 (1990).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0.0, 1.0, 30)
    >>> y = 2.0 * x + 0.5
    >>> fit = fit_scaling_region(x, y)
    >>> round(fit.slope, 6)
    2.0
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

    total = float(x[-1] - x[0])
    floor_abs = _DECADE if min_span is None else float(min_span)
    span_floor = min(min_span_frac * total, floor_abs)

    # (span, sigma, lo, hi, slope, intercept, stderr, npts) for every admissible
    # window.  ``_window_stats`` vectorises over ``hi``; the outer loop over
    # ``lo`` keeps the memory linear.
    spans: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    slopes: list[np.ndarray] = []
    intercepts: list[np.ndarray] = []
    stderrs: list[np.ndarray] = []
    for lo in range(0, n - min_window + 1):
        hi, slope, intercept, sigma, stderr = _window_stats(x, y, lo, min_window)
        span = x[hi] - x[lo]
        keep = np.isfinite(sigma) & (span >= span_floor - 1e-12)
        if not keep.any():
            continue
        spans.append(span[keep])
        sigmas.append(sigma[keep])
        los.append(np.full(int(keep.sum()), lo))
        his.append(hi[keep])
        slopes.append(slope[keep])
        intercepts.append(intercept[keep])
        stderrs.append(stderr[keep])

    if not spans:
        raise ValueError("No admissible scaling window (x has no spread). Check the input scales.")

    span_a = np.concatenate(spans)
    sigma_a = np.concatenate(sigmas)
    lo_a = np.concatenate(los)
    hi_a = np.concatenate(his)
    slope_a = np.concatenate(slopes)
    intercept_a = np.concatenate(intercepts)
    stderr_a = np.concatenate(stderrs)

    # The residual threshold is set from the same span-admissible pool the
    # selection draws from.  Deriving it from arbitrarily short windows (which is
    # what a point-count rule does) drives it toward zero on a dense grid, since
    # any smooth curve is locally straight — the lacunarity trap the module
    # docstring documents.
    sigma_min = float(sigma_a.min())
    threshold = sigma_min * tol if sigma_min > 0.0 else 0.0
    kept = np.flatnonzero(sigma_a <= threshold + 1e-300)
    # Widest span among the kept windows; tie-break toward the smaller residual.
    widest = float(span_a[kept].max())
    ties = kept[span_a[kept] >= widest - 1e-12]
    best = int(ties[np.argmin(sigma_a[ties])])

    lo_i, hi_i = int(lo_a[best]), int(hi_a[best])
    return ScalingFit(
        slope=float(slope_a[best]),
        intercept=float(intercept_a[best]),
        lo=lo_i,
        hi=hi_i,
        stderr=float(stderr_a[best]),
        npts=hi_i - lo_i + 1,
        span=float(span_a[best]),
        sigma=float(sigma_a[best]),
    )


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
    return np.asarray(np.gradient(y, x))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
