"""Maximal Lyapunov exponent from a measured time series.

Two closely-related neighbour-divergence estimators that need only a scalar (or
multivariate) recording — no equations of motion:

- **Kantz (1994)** — average, over reference points, of the log of the *mean*
  distance between the forward images of all neighbours found within a ball of
  radius ``eps``.  Robust to noise because it averages over a neighbourhood.
- **Rosenstein, Collins & De Luca (1993)** — tracks the single nearest
  neighbour of each reference point; cheaper, well-suited to short records.

Both build the ``S(k)`` *stretching curve* — the mean log divergence after ``k``
samples — whose slope over the linear scaling region is the maximal Lyapunov
exponent.  The signal is first reconstructed in an ``m``-dimensional delay
embedding (Takens 1981).

References
----------
H. Kantz, "A robust method to estimate the maximal Lyapunov exponent of a time
series", *Physics Letters A* **185** (1994) 77-87.

M. T. Rosenstein, J. J. Collins & C. J. De Luca, "A practical method for
calculating largest Lyapunov exponents from small data sets", *Physica D*
**65** (1993) 117-134.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidParameterError, remedy

from .._common import reject_system
from .._result import ScalingResult

__all__ = ["LyapunovFromData", "ScalingRegionWarning", "lyapunov_from_data"]

_TINY = float(np.finfo(float).tiny)

# ── automatic scaling-region selection: the tuned constants ──────────────────
# A plateau is admissible when every local slope inside it stays within
# ``_PLATEAU_TOL`` (relative) of the plateau mean...
_PLATEAU_TOL = 0.2
# ...and when that mean clears ``_SLOPE_FLOOR`` times the largest local slope on
# the curve.  The floor is what stops the *saturated tail* — a long, dead-flat,
# perfectly "stable-slope" plateau at slope ~0 — from being selected as the
# scaling region (it would report an exponent of ~0 for a chaotic series).
_SLOPE_FLOOR = 0.05
# Local slopes are read from a least-squares window this fraction of the curve
# wide (at least 3 points), so a single noisy sample cannot break a plateau.
_SLOPE_WINDOW_FRACTION = 20
# A plateau must span at least this fraction of the curve, and never fewer than
# _MIN_REGION_POINTS samples: on a short curve any three or four consecutive
# points of a smoothly-bending transient look "stable" at any usable tolerance.
_MIN_REGION_FRACTION = 10
_MIN_REGION_POINTS = 5

# Automatic reconstruction defaults (used when ``delay`` / ``k_max`` are None).
# The look-ahead is expressed in units of the embedding delay: the scaling region
# of every system measured for this module lives between roughly 5 and 20 delays,
# so 25 delays covers it with margin (Lorenz, Rössler, Hénon, logistic).
_KMAX_DELAYS = 25
_KMAX_FLOOR = 20
# Cost ceiling on the *automatic* look-ahead.  ``25 * delay`` is unbounded in the
# oversampling factor, and both estimators are unbounded in ``k_max``: the Kantz
# loop is linear in it and the Rosenstein path materialises an
# ``(n_ref, k_max + 1, m)`` array.  Measured on a Lorenz x-series, the automatic
# value is 425 at dt = 0.02 but 4125 at dt = 0.002 and 8225 at dt = 0.001 — the
# last is ~90 GB on the Rosenstein path and hours on the Kantz one, from a call
# that passed no ``k_max`` at all.  Where the cap binds it costs nothing in
# accuracy (Lorenz at dt = 0.005: 0.7396 at k_max = 1650 in 77 s versus 0.7523 at
# k_max = 500 in 22 s, against a true 0.9076), and where the capped look-ahead is
# genuinely too short the plateau search refuses (``trusted`` False) instead of
# reporting a transient slope.  A caller who wants the longer curve passes
# ``k_max`` explicitly and accepts the cost.
_KMAX_CEILING = 500
# An explicit `delay` this many times shorter than the data's own decorrelation
# time makes the delay embedding near-collinear — "neighbours" then are not
# dynamical neighbours and the exponent is badly biased.
_DEGENERATE_DELAY_FACTOR = 4


class ScalingRegionWarning(UserWarning):
    """The delay reconstruction cannot support a Lyapunov estimate.

    Emitted by :func:`lyapunov_from_data` when the requested ``delay`` is far
    shorter than the series' own decorrelation time: the delay embedding is then
    **near-collinear**, its "neighbours" are consecutive samples of one
    trajectory rather than dynamical neighbours, and the exponent read off it is
    badly biased.  This is the reconstruction that made an oversampled Lorenz
    series report 11.34 against a true 0.906.

    The *other* untrustworthy outcome — the automatic search finding no
    stable-slope plateau — is reported through the result rather than a warning
    (``result.trusted`` is ``False``, ``repr`` says ``UNTRUSTED``, and
    ``meta["scaling_region"]`` says what happened).  It is deliberately not a
    warning: a regular signal has no exponential scaling region *by definition*,
    so warning on it would fire on correct answers and train callers to suppress
    the category.
    """


@dataclass(frozen=True, eq=False)
class LyapunovFromData(ScalingResult):
    """Outcome of :func:`lyapunov_from_data`: the divergence curve and its slope.

    A :class:`~tsdynamics.analysis._result.ScalingResult` — the maximal Lyapunov
    exponent is read off the slope of the stretching curve, the same shape every
    fractal dimension and embedding diagnostic share — so it inherits the canonical
    ``estimate`` / ``abscissa`` / ``ordinate`` / ``fit_region`` schema, the result
    surface (``.meta`` / the readout ``repr`` / ``.to_dict()`` / the ``.plot`` seam) and
    ``float(result)`` (the exponent).  Domain-named ``@property`` aliases
    (:attr:`lyapunov`, :attr:`times`, :attr:`divergence`) preserve the original
    field names.

    Attributes
    ----------
    estimate : float
        Estimated maximal Lyapunov exponent (per unit time), the slope of
        ``ordinate`` against ``abscissa`` over ``fit_region``.  Aliased
        :attr:`lyapunov`.  ``float(result)`` returns it.
    abscissa : numpy.ndarray
        Relative times ``k * dt`` for ``k = 0 … k_max``.  Aliased :attr:`times`.
    ordinate : numpy.ndarray
        The stretching curve ``S(k)`` — mean log divergence after ``k`` samples.
        Aliased :attr:`divergence`.  Inspect ``abscissa`` vs ``ordinate`` to
        choose a scaling region and refine with an explicit ``fit=(lo, hi)``.
    fit_region : tuple[int, int]
        Inclusive index range into the curve used for the slope.
    embedding_dim, delay, theiler : int
        Reconstruction parameters actually used.
    n_reference : int
        Number of reference points that contributed (had a usable neighbour).
    method : str
        ``"kantz"`` or ``"rosenstein"``.
    trusted : bool
        ``False`` when the estimate is not a reading of a scaling region — the
        automatic search found no stable-slope plateau (the slope was then fitted
        over the whole curve as a fallback), or the delay embedding is
        near-collinear (which also raises a :class:`ScalingRegionWarning`).
        ``repr`` then says ``UNTRUSTED`` and ``meta["scaling_region"]`` records
        which happened.  **Check this flag** before believing an exponent from an
        unattended run.  An explicit ``fit=(lo, hi)`` takes ownership of the
        region, so it is always ``trusted`` unless the embedding is degenerate.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("lyapunov", "method")

    embedding_dim: int = 0
    delay: int = 0
    theiler: int = 0
    n_reference: int = 0
    method: str = "kantz"
    trusted: bool = True

    @property
    def lyapunov(self) -> float:
        """The estimated maximal Lyapunov exponent (alias of :attr:`estimate`)."""
        return float(self.estimate)

    @property
    def times(self) -> np.ndarray:
        """Relative times of the stretching curve (alias of :attr:`abscissa`)."""
        return self.abscissa

    @property
    def divergence(self) -> np.ndarray:
        """The stretching curve ``S(k)`` (alias of :attr:`ordinate`)."""
        return self.ordinate

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the divergence curve as a backend-agnostic :class:`PlotSpec`.

        Builds a ``SCALING_FIT`` spec — the stretching curve :math:`S(k)` (mean
        log-divergence) against time as a scatter, the fitted scaling region
        highlighted, and the line of slope :attr:`lyapunov` drawn over it — the
        same schema the fractal-dimension estimators emit, so a single
        ``result.plot.scaling()`` renders it.  The :mod:`tsdynamics.viz.spec`
        import is lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"scaling_fit"``).  ``None`` uses
            ``SCALING_FIT``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        t = np.asarray(self.times, dtype=float)
        s = np.asarray(self.divergence, dtype=float)
        lo, hi = self.fit_region
        # The line of slope `lyapunov` anchored to the fit-region centroid.
        line_y = None
        if t.size and hi >= lo:
            tc = float(np.mean(t[lo : hi + 1]))
            sc = float(np.mean(s[lo : hi + 1]))
            fit_x = np.array([t[lo], t[hi]], dtype=float)
            line_y = sc + self.lyapunov * (fit_x - tc)
        return pb.scaling_fit(
            kind,
            t,
            s,
            fit_region=self.fit_region,
            slope=self.lyapunov,
            line_y=line_y,
            curve_label="$S(k)$",
            xlabel="time",
            ylabel="mean log divergence $S(k)$",
            title=f"max. Lyapunov ({self.method}) = {self.lyapunov:.3g}",
        )

    def _quantity(self) -> str:
        r"""Return ``λ_max`` — the quantity the slope is."""
        return "λ_max"

    def _unit(self) -> str:
        """Return ``per unit time`` or ``per sample``, read off the curve itself.

        The stretching curve's abscissa is ``k * dt``, so its spacing **is** the
        sampling interval the estimator used.  A unit spacing means the input
        carried no time axis and the exponent is per sample (per iteration); any
        other spacing means it did and the exponent is per unit time.  Getting
        this wrong is a factor of ``1/dt`` — ~60× on a Lorenz run at
        ``dt = 0.02`` — so it is derived, not assumed.
        """
        t = np.asarray(self.abscissa, dtype=float)
        if t.size < 2:
            return ""
        return "per sample" if float(t[1] - t[0]) == 1.0 else "per unit time"

    def _interpretation(self) -> str | None:
        """Flag an estimate that is not a reading of a scaling region."""
        return None if self.trusted else "⚠ UNTRUSTED"

    def _context(self) -> str | None:
        """Return the settings that make the number meaningful."""
        return f"{self.method}, m={self.embedding_dim}, τ={self.delay}, {self.n_reference} ref pts"

    def _details(self) -> tuple[str, ...]:
        """Return the fit line, replaced by the remedy when untrusted."""
        if not self.trusted:
            return ("⚠ no clear scaling region — inspect .plot.scaling() and pass fit=(lo, hi)",)
        return super()._details()

    def _derived(self) -> dict[str, Any]:
        """Export the exponent, its unit and the fit diagnostics."""
        data = super()._derived()
        data.update(lyapunov=self.lyapunov, unit=self._unit())
        return data


def _delay_embed(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Delay-coordinate embedding of ``series`` into ``m`` blocks spaced by ``tau``.

    Row ``n`` is ``[x[n], x[n+tau], …, x[n+(m-1)tau]]`` (each ``x`` a channel
    vector for multivariate input), so the forward image of row ``n`` after
    ``k`` samples is simply row ``n + k``.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise InvalidParameterError("series must be 1D (scalar) or 2D (n_samples, n_channels).")
    n, d = x.shape
    span = (m - 1) * tau
    rows = n - span
    if rows <= 0:
        raise InvalidParameterError(
            f"series too short: {n} samples cannot fill an m={m}, tau={tau} embedding "
            f"(needs more than {span}). Use a longer series, or reduce dimension/delay."
        )
    emb = np.empty((rows, m * d), dtype=float)
    for j in range(m):
        emb[:, j * d : (j + 1) * d] = x[j * tau : j * tau + rows]
    return emb


def _slope_stderr(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """Return the standard error of a least-squares slope (0 when undefined).

    The textbook ``s_slope = sqrt( SSE / (n-2) / Sxx )``; returns ``0.0`` for
    fewer than three points or a degenerate abscissa, computed directly (no
    ``polyfit(cov=True)``) so it never raises a rank warning under
    ``filterwarnings=error``.
    """
    n = x.size
    if n < 3:
        return 0.0
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 0.0:
        return 0.0
    resid = y - (slope * x + intercept)
    sse = float(np.sum(resid**2))
    return float(np.sqrt(sse / (n - 2) / sxx))


def _auto_delay(series: np.ndarray) -> tuple[int, bool]:
    """Embedding delay: the first lag whose autocorrelation has fallen to ``1/e``.

    Returns ``(delay, found)``.  ``found`` is ``False`` when the autocorrelation
    never reached ``1/e`` within the searched range (a monotone or very
    long-correlated record): the capped lag is still returned as the best
    available default, but no *diagnosis* may be based on it — the series has no
    measured decorrelation time to compare a caller's ``delay`` against.

    The classical linear decorrelation-time criterion.  It is ``1`` for a map —
    successive iterates are already decorrelated — and grows with the
    oversampling factor of a flow, which is exactly the failure the old fixed
    ``delay=1`` default walked into: at ``dt = 0.02`` a Lorenz embedding
    ``[x(t), x(t+dt), x(t+2dt)]`` is very nearly collinear, so "neighbours" are
    not dynamical neighbours and the stretching curve has no scaling region at
    all.

    The ``1/e`` rule is preferred here over the first *zero* crossing (which a
    slowly-decorrelating flow reaches only after a full turn of the attractor —
    ``2.4`` time units for Lorenz, an order of magnitude too long) and over the
    first minimum of the mutual information (the standard nonlinear criterion for
    a flow, but meaningless on a map, where the estimator has no clean dip to
    find and returns noise).  Pass ``delay=`` explicitly to use
    :func:`~tsdynamics.analysis.embedding.optimal_delay` or any other rule.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:  # constant series — nothing to decorrelate
        return 1, False
    # Look no further than a tenth of the record: a delay longer than that
    # cannot be estimated reliably (and would leave too few embedded rows).
    max_lag = max(1, min(x.size // 10, 1000))
    threshold = 1.0 / np.e
    for lag in range(1, max_lag + 1):
        if float(np.dot(x[lag:], x[:-lag])) / denom <= threshold:
            return lag, True
    return max_lag, False


def _local_slopes(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Least-squares slope of ``y`` vs ``x`` over a sliding ``window``-point window.

    The window is centred where it fits and clamped at the ends, so the returned
    array is the same length as the curve.  Smoothing over a window (rather than
    taking raw differences) keeps one noisy sample from puncturing an otherwise
    flat plateau.
    """
    n = y.size
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        hi = min(n, max(i - half, 0) + window)
        lo = max(0, hi - window)
        xs = x[lo:hi]
        xm = xs.mean()
        dx = xs - xm
        sxx = float(np.dot(dx, dx))
        out[i] = float(np.dot(dx, y[lo:hi] - y[lo:hi].mean()) / sxx) if sxx > 0.0 else np.nan
    return out


def _auto_fit_region(x: np.ndarray, y: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    """Find the linear scaling region of a stretching curve.

    Returns ``(region, peak_local_slope)``; ``region`` is ``None`` when there is
    none.  ``peak_local_slope`` is the largest local slope on the curve — the
    curve's own scale of "fast divergence", against which :data:`_SLOPE_FLOOR`
    is measured; it is returned for inspection and for the unit tests, and the
    estimator itself only uses ``region``.

    The curve ``S(k)`` has three parts: an initial **transient** (the neighbour
    ball is not yet aligned with the unstable manifold — the local slope is large
    and falling fast), the **scaling region** (a genuine plateau of the local
    slope, whose height is the maximal Lyapunov exponent), and **saturation**
    (neighbours have reached the attractor diameter — the local slope decays to
    zero).  Anchoring the fit at ``k = 0``, as this function used to, fits the
    transient and biases the exponent high by up to an order of magnitude.

    The search returns the **longest** index window whose local slopes all stay
    within :data:`_PLATEAU_TOL` of the window's mean slope (ties broken toward
    the flatter, then the earlier window) *and* whose mean slope clears
    :data:`_SLOPE_FLOOR` times the largest local slope on the curve — the second
    condition is what rejects the saturated tail, which is otherwise the longest
    and flattest "plateau" on the curve.

    A plateau that runs to the **last** sample of the curve is rejected too: the
    scaling region is bounded above by the onset of saturation, and a window
    reaching the end of a truncated curve is indistinguishable from a transient
    that simply has not finished decaying.  That guard is what refuses the old
    ``dimension=3, delay=1, k_max=20`` Lorenz reconstruction instead of reading a
    4x-too-high exponent off its still-decaying transient.

    **Exactly what that guard is, and is not.**  It excludes the single final
    index — ``hi`` never reaches ``n - 1`` — so a window ending at ``n - 2`` is
    still accepted.  It is not a margin, and it does not keep the last sample out
    of the fit: :func:`_local_slopes` is a *centred* window clamped at the ends,
    so ``slopes[n - 2]`` already reads the final point.  The guard is the minimum
    that refuses a curve whose plateau is still running when the data stop; a
    genuinely conservative version would drop a whole window's worth of tail,
    which would move every reported exponent and is deliberately not done here.

    ``None`` means the curve has no such plateau: either it is entirely
    transient (too small a ``k_max``, or a degenerate embedding) or entirely
    saturated.  The caller must not silently report a slope in that case.
    """
    n = y.size
    min_len = max(_MIN_REGION_POINTS, -(-n // _MIN_REGION_FRACTION))
    if n < min_len:
        return None, 0.0
    window = max(3, n // _SLOPE_WINDOW_FRACTION)
    slopes = _local_slopes(x, y, window)
    if not np.all(np.isfinite(slopes)):
        return None, 0.0
    peak = float(np.max(slopes))
    if peak <= 0.0:
        return None, max(peak, 0.0)
    floor = _SLOPE_FLOOR * peak

    best: tuple[int, float, int, int] | None = None
    for lo in range(0, n - min_len):
        # Longest first: the first `hi` that qualifies for this `lo` is the
        # longest window starting there, so the inner loop breaks immediately.
        # `hi` stops at n - 2: a window touching the last sample is rejected
        # (the curve was truncated before saturation, so we cannot tell a
        # plateau from an unfinished transient).
        for hi in range(n - 2, lo + min_len - 2, -1):
            seg = slopes[lo : hi + 1]
            mean = float(seg.mean())
            if mean <= floor:
                continue
            dev = float(np.max(np.abs(seg - mean))) / mean
            if dev > _PLATEAU_TOL:
                continue
            length = hi - lo + 1
            if best is None or length > best[0] or (length == best[0] and dev < best[1]):
                best = (length, dev, lo, hi)
            break
    if best is None:
        return None, peak
    return (best[2], best[3]), peak


#: How far a trajectory's widest sample gap may stray from the typical one
#: (relative) and still count as a uniformly-sampled time axis.  Output grids are
#: built by accumulation, so the last gap can differ by a few ULP.
_UNIFORM_TIME_AXIS_SPREAD = 1e-9


def _sampling_interval_of(data: Any) -> float | None:
    """Read the sampling interval a *trajectory* already knows, or ``None``.

    A :class:`~tsdynamics.data.Trajectory` carries its own output spacing —
    in ``meta["dt"]`` when it came from a run, and in its ``t`` axis either
    way.  Reading it is the difference between an exponent per *sample* and
    an exponent per *time unit*: ``traj.lyap.from_data()`` on a Lorenz run
    sampled at ``dt = 0.02`` returned 0.0164 (per sample) where the answer
    that compares with ``sys.lyap.spectrum()`` is 1.04 (per time unit) — a
    silent factor of 60, from information the object was holding.

    A bare array carries no time axis, so it keeps the documented
    ``dt = 1.0`` (an exponent per sample / per iteration).
    """
    meta = getattr(data, "meta", None)
    t = getattr(data, "t", None)
    if meta is None or t is None:
        return None
    recorded = meta.get("dt") if hasattr(meta, "get") else None
    if recorded is not None:
        return float(recorded)
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or times.size < 2:
        return None
    steps = np.diff(times)
    step = float(np.median(steps))
    # A uniformity check, not a solver tolerance: how far the widest gap strays
    # from the typical one, relative to the typical one.
    spread = float(np.max(np.abs(steps - step))) / abs(step) if step else np.inf
    if step <= 0.0 or spread > _UNIFORM_TIME_AXIS_SPREAD:
        # A non-uniform time axis has no single sampling interval, and quietly
        # assuming one is how a wrong exponent gets reported as a right one.
        raise InvalidParameterError(
            "lyapunov_from_data: this trajectory's time axis is not uniformly "
            "spaced, so it has no single sampling interval to read. Pass the one "
            "you mean explicitly, e.g. lyapunov_from_data(data, dt=0.01)."
        )
    return step


def lyapunov_from_data(
    data: np.ndarray,
    *,
    dt: float | None = None,
    dimension: int = 5,
    delay: int | None = None,
    theiler: int | None = None,
    k_max: int | None = None,
    eps: float | None = None,
    n_neighbors: int = 1,
    method: str = "kantz",
    fit: tuple[int, int] | None = None,
) -> LyapunovFromData:
    r"""Estimate the maximal Lyapunov exponent from a time series.

    Reconstructs an ``m``-dimensional delay embedding of ``series``, measures how
    fast nearby trajectories diverge as a function of the look-ahead ``k``, and
    reads the exponent off the slope of the resulting log-divergence curve.

    Parameters
    ----------
    data : array_like
        1-D scalar series, or 2-D ``(n_samples, n_channels)`` for a multivariate
        recording.
    dt : float, optional
        Sampling interval (time between consecutive samples); the exponent is
        reported per unit of ``dt``.  Left unset it is **read from the data
        when the data knows it** — a
        :class:`~tsdynamics.data.Trajectory` carries its output spacing, so
        ``lyapunov_from_data(traj)`` answers in the system's own time units and
        is directly comparable with :func:`lyapunov_spectrum`.  A bare array has
        no time axis, so it keeps ``dt = 1.0``: an exponent per sample, which is
        also the right reading for a map (per iteration).
    dimension : int, default 5
        Embedding dimension.  Should be large enough to unfold the attractor
        (Takens' sufficient condition is ``m > 2 D``; a false-nearest-neighbour
        estimate is the data-driven choice) — too small underestimates the
        exponent and roughens the scaling region.
    delay : int, optional
        Embedding delay, in samples.  ``None`` (the default) reads it from the
        data: the first lag at which the autocorrelation has fallen to
        :math:`1/e` (see :func:`_auto_delay`, which also explains why that rule
        rather than the first zero crossing or the mutual-information minimum).
        That is ``1`` for a map and grows with the oversampling factor of a
        flow; the old fixed
        ``delay=1`` default produced a near-collinear embedding for any
        finely-sampled flow and, with it, a badly biased exponent.
    theiler : int, optional
        Theiler window (Theiler 1986): neighbours with ``|n - j| <= theiler`` are
        rejected so temporally-correlated points are not mistaken for dynamical
        neighbours.  Defaults to ``(dimension - 1) * delay`` (the embedding span).
    k_max : int, optional
        Number of forward samples over which divergence is tracked; the curve
        spans ``k = 0 … k_max``.  ``None`` (the default) uses ``25 * delay``
        (at least 20, never more than the series can support, and never more than
        :data:`_KMAX_CEILING` — both estimators cost time, and Rosenstein memory,
        linearly in ``k_max``, so an oversampling-driven value is capped with a
        :class:`ScalingRegionWarning`): the scaling region lives a few
        decorrelation times out, so a look-ahead fixed in *samples* is
        meaningless while one fixed in *delays* transfers between systems.
    eps : float, optional
        Neighbour-ball radius for ``method="kantz"``.  Defaults to ``0.1`` times
        the standard deviation pooled over *all* embedded coordinates.  This
        default assumes the coordinates are comparably scaled: for a strongly
        anisotropic embedding (channels on very different scales, or a
        multivariate recording with disparate units) the pooled std is dominated
        by the largest-variance coordinate and mis-scales the isotropic ball for
        the others, biasing the neighbour set — standardize/normalize the
        channels first (or pass an explicit ``eps``) in that case.  Ignored by
        ``"rosenstein"`` (which uses the single nearest neighbour).
    n_neighbors : int, default 1
        Minimum neighbours a reference point needs to contribute (Kantz only).
    method : {"kantz", "rosenstein"}, default "kantz"
        Divergence estimator (see module docstring).
    fit : tuple[int, int], optional
        Inclusive sample range ``(lo, hi)`` over which the slope is fit.  ``None``
        (the default) locates the linear scaling region automatically
        (:func:`_auto_fit_region`: the longest stable-local-slope plateau, above
        the transient and below saturation).  Passing ``fit`` explicitly takes
        ownership of the choice — no plateau search runs and no
        :class:`ScalingRegionWarning` is raised for it, whatever the window
        contains.

    Returns
    -------
    LyapunovFromData
        The estimated exponent, the full divergence curve, and the parameters
        used.  Casts to ``float`` as the exponent.

    Warns
    -----
    ScalingRegionWarning
        When the requested ``delay`` is far below the series' own decorrelation
        time — a near-collinear embedding, whose exponent is badly biased — or
        when the series is so heavily oversampled that the automatic ``k_max``
        had to be capped at :data:`_KMAX_CEILING` to stay affordable.

        The other untrustworthy outcome, a curve with **no** stable-slope plateau
        (all transient and/or all saturation, so the slope is a fallback fit over
        the whole curve), is reported through ``result.trusted`` /
        ``repr(result)`` / ``result.meta["scaling_region"]`` rather than a
        warning — see :class:`ScalingRegionWarning` for why.  Either way, do not
        use an untrusted number: inspect the curve and pass an explicit ``fit``,
        or improve the reconstruction.

    Raises
    ------
    InvalidParameterError
        If a reconstruction parameter is invalid (``dimension < 1``,
        ``delay < 1``, ``k_max < 2``, ``n_neighbors < 1``, ``dt <= 0``,
        ``theiler < 0``, an unknown ``method``, ``eps <= 0`` for a constant
        series, an out-of-bounds ``fit`` window, or a ``k_max`` too large / series
        too short to leave any forward image).
    ConvergenceError
        If no usable neighbour is found (no reference point has a neighbour
        within ``eps`` outside the Theiler window, or no nearest neighbour clears
        the Theiler window), or the fit window holds too few usable divergence
        points to fit a slope.

    Notes
    -----
    The estimate is only as good as the embedding and the chosen scaling region.
    Always look at ``result.times`` vs ``result.divergence``: a trustworthy
    estimate comes from a clear straight segment before the curve saturates.
    Measured at the defaults against independent truth (logistic ``r = 4``, where
    the exponent is exactly :math:`\ln 2`; Hénon, Lorenz, Chen and Rössler
    against variational spectra), the automatic region lands within **1 % on a
    map** and within **−22 % to +9 % on a flow**, depending on the observable and
    the sampling rate — a Kantz/Rosenstein reading of a scalar flow observable is
    a ~20 %-accurate quantity, not a precision one.  Treat it as "is there a
    positive exponent, and roughly how large", and use
    :func:`~tsdynamics.analysis.lyapunov.lyapunov_spectrum` when the system (not
    just a measured series) is in hand.

    Examples
    --------
    >>> import tsdynamics as ts
    >>> traj = ts.systems.Henon().run(6000, transient=500, ic=[0.1, 0.1])
    >>> res = ts.lyapunov_from_data(traj.y[:, 0], dimension=4, k_max=12, fit=(0, 6))
    >>> 0.30 < float(res) < 0.55      # ≈ 0.42
    True

    References
    ----------
    H. Kantz, "A robust method to estimate the maximal Lyapunov exponent of a
    time series", *Physics Letters A* **185** (1994) 77--87.

    M. T. Rosenstein, J. J. Collins & C. J. De Luca, "A practical method for
    calculating largest Lyapunov exponents from small data sets", *Physica D*
    **65** (1993) 117--134.
    """
    reject_system(data, analysis="lyapunov_from_data")
    dimension = int(dimension)
    n_neighbors = int(n_neighbors)
    if dt is None:
        # Unset: read it off the data when the data knows it, else per sample.
        measured = _sampling_interval_of(data)
        dt = 1.0 if measured is None else measured
    else:
        dt = float(dt)
    method = method.lower()
    if dimension < 1:
        raise InvalidParameterError("dimension (embedding dimension) must be >= 1.")
    # Reconstruction defaults are read from the data, not fixed in samples: a
    # delay and a look-ahead that are right for a map are wrong for the same
    # dynamics sampled 100x finer (see the `delay` / `k_max` docstrings).
    auto_delay, delay_measured = _auto_delay(data)
    delay = auto_delay if delay is None else int(delay)
    if delay < 1:
        raise InvalidParameterError("delay (embedding delay) must be >= 1.")
    degenerate_embedding = delay_measured and delay * _DEGENERATE_DELAY_FACTOR < auto_delay
    if degenerate_embedding:
        warnings.warn(
            f"lyapunov_from_data: delay={delay} is far below this series' own "
            f"decorrelation time (~{auto_delay} samples), so the delay embedding is "
            "near-collinear: its 'neighbours' are consecutive samples of one "
            "trajectory rather than dynamical neighbours, and the exponent read "
            "off it is badly biased (result.trusted is False). Leave delay unset "
            "to use the data-driven value, or pass a delay of that order.",
            ScalingRegionWarning,
            stacklevel=2,
        )
    if k_max is None:
        # The look-ahead is set by the *dynamics'* own decorrelation time, not by
        # whatever embedding delay the caller chose: an over-long delay on a map
        # (e.g. delay=5 on the logistic map, whose stretching curve saturates by
        # k = 10) would otherwise stretch k_max to 125 and drown the scaling
        # region in saturation.
        scale = auto_delay if delay_measured else delay
        span = (dimension - 1) * delay
        rows = int(np.shape(np.asarray(data))[0]) - span
        wanted = max(_KMAX_FLOOR, _KMAX_DELAYS * scale)
        # Every row within k_max of the end is unusable as a reference OR as a
        # neighbour (its k-ahead image does not exist), so a look-ahead of half
        # the rows throws away half the point cloud — and on a short, heavily
        # oversampled series (whose Theiler window is ``(m - 1) * delay``, itself
        # large) the survivors are too sparse for any of them to have a neighbour
        # inside ``eps``: a 1000-sample Lorenz x-series at dt = 0.02 raised
        # ConvergenceError at ``(rows - 2) // 2`` and estimates fine at a quarter.
        k_max = int(min(wanted, _KMAX_CEILING, max(2, (rows - 1) // 4)))
        if wanted > _KMAX_CEILING and k_max == _KMAX_CEILING:
            # The cap bound: say so, because it is the sampling rate that made the
            # automatic look-ahead unaffordable, and decimating is the real remedy.
            warnings.warn(
                f"lyapunov_from_data: this series decorrelates only after ~{auto_delay} "
                f"samples, so the automatic look-ahead would be k_max={wanted}; it is "
                f"capped at {_KMAX_CEILING} because the cost of both estimators grows "
                "with k_max (and the Rosenstein path's memory with it). The series is "
                "heavily oversampled for this estimator: decimate it (keep every "
                f"~{max(1, auto_delay // 10)}th sample and scale dt by the same "
                "factor), or pass k_max explicitly and accept the cost. Check "
                "result.trusted.",
                ScalingRegionWarning,
                stacklevel=2,
            )
    else:
        k_max = int(k_max)
    if k_max < 2:
        raise InvalidParameterError("k_max must be >= 2 to fit a slope.")
    if n_neighbors < 1:
        raise InvalidParameterError("n_neighbors must be >= 1.")
    if dt <= 0.0:
        raise InvalidParameterError("dt must be positive.")
    if method not in {"kantz", "rosenstein"}:
        raise InvalidParameterError(f"method must be 'kantz' or 'rosenstein', got {method!r}.")
    theiler = (dimension - 1) * delay if theiler is None else int(theiler)
    if theiler < 0:
        raise InvalidParameterError("theiler must be >= 0.")
    if fit is not None and not (0 <= int(fit[0]) < int(fit[1]) <= k_max):
        # Validated up front: an out-of-range window is a caller error and must
        # not be reported only after the (potentially minutes-long) neighbour search.
        raise InvalidParameterError(
            f"fit region {fit!r} must satisfy 0 <= lo < hi <= k_max ({k_max})."
        )

    from scipy.spatial import cKDTree

    emb = _delay_embed(data, dimension, delay)
    n_rows = emb.shape[0]
    last = n_rows - 1 - k_max  # references/neighbours need their k-ahead image to exist
    if last < 1:
        raise InvalidParameterError(
            "k_max is too large for the embedded series: no forward images remain. "
            "Use a longer series or reduce k_max, dimension, or delay."
        )
    tree = cKDTree(emb)

    if method == "kantz":
        if eps is None:
            # Pooled std over every embedded coordinate: assumes comparably-scaled
            # channels (see the `eps` docstring; standardize anisotropic input).
            eps = 0.1 * float(np.std(emb))
        eps = float(eps)
        if eps <= 0.0:
            raise InvalidParameterError("eps must be positive (series may be constant).")
        # One batched ball query for *all* candidate reference rows, then filter
        # each candidate list to neighbours inside eps, outside the Theiler window,
        # and whose k-ahead image still exists (j <= last). A reference point with
        # >= n_neighbors survivors contributes. This reproduces the per-point
        # ``query_ball_point`` loop exactly (same eps, same predicate, same order).
        cand_lists = tree.query_ball_point(emb[: last + 1], eps)
        ref_idx_list: list[int] = []
        # Flat (reference, neighbour) pair arrays plus per-reference neighbour
        # counts: the divergence average over each reference's neighbour set is a
        # segment-mean over these flat arrays (np.add.at grouping below), so the
        # per-k double Python loop collapses to one vectorised distance + reduce.
        ref_repeat_blocks: list[np.ndarray] = []
        neigh_blocks: list[np.ndarray] = []
        counts: list[int] = []
        for n in range(last + 1):
            cand = cand_lists[n]
            neigh = np.fromiter(
                (j for j in cand if j <= last and abs(j - n) > theiler),
                dtype=np.intp,
            )
            if neigh.size >= n_neighbors:
                ref_pos = len(ref_idx_list)
                ref_idx_list.append(n)
                ref_repeat_blocks.append(np.full(neigh.size, ref_pos, dtype=np.intp))
                neigh_blocks.append(neigh)
                counts.append(int(neigh.size))
        if not ref_idx_list:
            raise ConvergenceError(
                f"lyapunov_from_data: no reference point on the reconstructed "
                f"attractor has a neighbour within eps={eps:.3g} outside the Theiler "
                f"window ({theiler} samples), so there is no divergence to measure. "
                f"The series is too short or too sparsely sampled for a "
                f"{dimension}-dimensional reconstruction."
                + remedy(
                    f"ts.lyapunov_from_data(data, dimension=3, eps={4 * eps:.3g},"
                    f" theiler={max(1, theiler // 2)})",
                    lead="Widen the neighbourhood and lower the embedding dimension:",
                )
            )
        n_reference = len(ref_idx_list)
        ref_idx_arr = np.asarray(ref_idx_list, dtype=np.intp)
        ref_repeat = np.concatenate(ref_repeat_blocks)  # group id per pair
        neigh_flat = np.concatenate(neigh_blocks)  # neighbour row per pair
        counts_arr = np.asarray(counts, dtype=float)  # neighbours per reference
        divergence = np.empty(k_max + 1)
        for k in range(k_max + 1):
            # Pairwise distances between every reference and its neighbours at the
            # k-ahead image, in one vectorised pass over the flat pair arrays.
            diff = emb[ref_idx_arr[ref_repeat] + k] - emb[neigh_flat + k]
            d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            # Mean distance per reference (segment-sum / count), matching the
            # per-point ``d.mean()`` exactly.
            sums = np.zeros(n_reference)
            np.add.at(sums, ref_repeat, d)
            means = sums / counts_arr
            divergence[k] = float(np.mean(np.log(np.maximum(means, _TINY))))
    else:  # rosenstein
        # Examine the n_query nearest candidates per point; the cap is a
        # heuristic sized to clear the Theiler window. A reference point whose
        # candidates are *all* rejected (inside the window or past `last`) is
        # skipped rather than matched to a wrong neighbour, lowering n_reference.
        n_query = min(n_rows, 4 * theiler + 20)
        _, idx_all = tree.query(emb, k=n_query)
        idx_all = np.atleast_2d(idx_all)
        # Select, for each reference row, the FIRST candidate column (in the
        # tree's nearest-first order) that clears the Theiler window and whose
        # k-ahead image exists — exactly the inner break-loop, expressed as a
        # masked argmax. ``argmax`` returns the first True (the break target);
        # rows with no valid candidate are dropped (``any`` over the row is False).
        rows = idx_all[: last + 1, 1:]  # drop column 0 (the point itself)
        ref_grid = np.arange(last + 1, dtype=np.intp)[:, None]
        valid = (rows <= last) & (np.abs(rows - ref_grid) > theiler)
        has_neighbour = valid.any(axis=1)
        first_col = valid.argmax(axis=1)  # first valid column per row (0 if none)
        ref_arr = ref_grid[:, 0][has_neighbour]
        nn_arr = rows[ref_arr, first_col[has_neighbour]].astype(np.intp)
        if ref_arr.size == 0:
            raise ConvergenceError(
                "no nearest neighbour outside the Theiler window was found; "
                "use a longer series or shorten the Theiler window."
            )
        # All look-ahead images at once: index the reference / neighbour pairs at
        # every lag k via broadcasting, one norm reduction over the last axis.
        ks = np.arange(k_max + 1, dtype=np.intp)
        ref_at_k = emb[ref_arr[:, None] + ks]  # (n_ref, k_max+1, dim)
        nn_at_k = emb[nn_arr[:, None] + ks]
        diff = ref_at_k - nn_at_k
        d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))  # (n_ref, k_max+1)
        divergence = np.mean(np.log(np.maximum(d, _TINY)), axis=0)
        n_reference = int(ref_arr.size)

    times = np.arange(k_max + 1, dtype=float) * dt
    trusted = not degenerate_embedding
    if fit is None:
        region, _peak_slope = _auto_fit_region(times, divergence)
        if region is None:
            # No plateau: the curve is all transient and/or all saturation, so no
            # slope read off it is a scaling-region reading.  Fall back to the
            # whole curve and flag the result untrusted.
            trusted = False
            lo, hi = 0, k_max
            region_source = "none (fallback: fitted over the whole curve)"
        else:
            lo, hi = region
            region_source = "auto (stable-slope plateau)"
    else:
        lo, hi = int(fit[0]), int(fit[1])  # bounds already validated above
        region_source = "explicit"
    xfit = times[lo : hi + 1]
    yfit = divergence[lo : hi + 1]
    # The divergence curve is floored at ``log(_TINY)``, so it is always finite;
    # guard the slope fit anyway against a degenerate (single-point / collinear-x)
    # window so it raises a clean typed error rather than a numpy rank warning.
    if xfit.size < 2 or not np.any(np.isfinite(yfit)):
        raise ConvergenceError(
            "lyapunov_from_data: the fit window holds too few usable divergence points "
            "to fit a slope; widen `fit` or use a longer series / different embedding."
        )
    slope, intercept = (float(c) for c in np.polyfit(xfit, yfit, 1))
    stderr = _slope_stderr(xfit, yfit, slope, intercept)

    return LyapunovFromData(
        estimate=slope,
        stderr=stderr,
        abscissa=times,
        ordinate=divergence,
        fit_region=(lo, hi),
        intercept=intercept,
        embedding_dim=dimension,
        delay=delay,
        theiler=theiler,
        n_reference=n_reference,
        method=method,
        trusted=trusted,
        meta={
            "method": method,
            "dimension": dimension,
            "delay": delay,
            "theiler": theiler,
            "k_max": k_max,
            "n_reference": n_reference,
            "trusted": trusted,
            "scaling_region": region_source,
        },
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
