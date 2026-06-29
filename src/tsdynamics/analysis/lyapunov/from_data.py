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

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidParameterError

from .._result import ScalingResult

__all__ = ["LyapunovFromData", "lyapunov_from_data"]

_TINY = float(np.finfo(float).tiny)


@dataclass(frozen=True)
class LyapunovFromData(ScalingResult):
    """Outcome of :func:`lyapunov_from_data`: the divergence curve and its slope.

    A :class:`~tsdynamics.analysis._result.ScalingResult` — the maximal Lyapunov
    exponent is read off the slope of the stretching curve, the same shape every
    fractal dimension and embedding diagnostic share — so it inherits the canonical
    ``estimate`` / ``abscissa`` / ``ordinate`` / ``fit_region`` schema, the result
    surface (``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam) and
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
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("lyapunov", "method")

    embedding_dim: int = 0
    delay: int = 0
    theiler: int = 0
    n_reference: int = 0
    method: str = "kantz"

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

    def __repr__(self) -> str:  # noqa: D105
        lo, hi = self.fit_region
        return (
            f"LyapunovFromData(lyapunov={self.lyapunov:.4g}, method={self.method!r}, "
            f"m={self.embedding_dim}, tau={self.delay}, fit_region=({lo}, {hi}), "
            f"n_reference={self.n_reference})"
        )


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
            f"(needs more than {span})."
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


def _auto_fit_region(divergence: np.ndarray, *, min_len: int) -> tuple[int, int]:
    """Pick ``[0, knee]`` — the initial rise before the curve saturates.

    The stretching curve rises roughly linearly while neighbours separate
    exponentially, then bends over as they reach the attractor's size.  The knee
    is the first sample whose local slope drops below half the initial slope.
    """
    n = divergence.size
    if n <= min_len:
        return (0, n - 1)
    k = np.arange(n, dtype=float)
    s0 = float(np.polyfit(k[:min_len], divergence[:min_len], 1)[0])
    if not np.isfinite(s0) or s0 <= 0.0:
        return (0, n - 1)
    diffs = np.diff(divergence)
    hi = n - 1
    for i in range(min_len, diffs.size + 1):
        if diffs[i - 1] < 0.5 * s0:
            hi = i - 1
            break
    return (0, max(hi, min_len - 1))


def lyapunov_from_data(
    data: np.ndarray,
    *,
    dt: float = 1.0,
    dimension: int = 3,
    delay: int = 1,
    theiler: int | None = None,
    k_max: int = 20,
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
    dt : float, default 1.0
        Sampling interval (time between consecutive samples).  Use ``1.0`` for a
        map (the exponent is then per iteration).
    dimension : int, default 3
        Embedding dimension.  Choose it from the data — large enough to unfold
        the attractor (e.g. a false-nearest-neighbour estimate); too small
        underestimates the exponent.
    delay : int, default 1
        Embedding delay, in samples.  For oversampled flows pick it near the
        first minimum of the mutual information / first zero of the
        autocorrelation.
    theiler : int, optional
        Theiler window (Theiler 1986): neighbours with ``|n - j| <= theiler`` are
        rejected so temporally-correlated points are not mistaken for dynamical
        neighbours.  Defaults to ``(dimension - 1) * delay`` (the embedding span).
    k_max : int, default 20
        Number of forward samples over which divergence is tracked; the curve
        spans ``k = 0 … k_max``.
    eps : float, optional
        Neighbour-ball radius for ``method="kantz"``.  Defaults to ``0.1`` times
        the standard deviation of the embedded coordinates.  Ignored by
        ``"rosenstein"`` (which uses the single nearest neighbour).
    n_neighbors : int, default 1
        Minimum neighbours a reference point needs to contribute (Kantz only).
    method : {"kantz", "rosenstein"}, default "kantz"
        Divergence estimator (see module docstring).
    fit : tuple[int, int], optional
        Inclusive sample range ``(lo, hi)`` over which the slope is fit.  Defaults
        to an automatic scaling region (the initial linear rise).  Inspect the
        returned curve and set this explicitly for a reliable estimate.

    Returns
    -------
    LyapunovFromData
        The estimated exponent, the full divergence curve, and the parameters
        used.  Casts to ``float`` as the exponent.

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

    Examples
    --------
    >>> import tsdynamics as ts
    >>> traj = ts.Henon().trajectory(6000, transient=500, ic=[0.1, 0.1])
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
    dimension = int(dimension)
    delay = int(delay)
    k_max = int(k_max)
    n_neighbors = int(n_neighbors)
    dt = float(dt)
    method = method.lower()
    if dimension < 1:
        raise InvalidParameterError("dimension (embedding dimension) must be >= 1.")
    if delay < 1:
        raise InvalidParameterError("delay (embedding delay) must be >= 1.")
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
                "no reference point has a neighbour within eps outside the Theiler "
                "window; increase eps, lower dimension, or shorten the Theiler window."
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
    if fit is None:
        lo, hi = _auto_fit_region(divergence, min_len=max(3, (k_max + 1) // 3))
    else:
        lo, hi = int(fit[0]), int(fit[1])
        if not (0 <= lo < hi <= k_max):
            raise InvalidParameterError(
                f"fit region {fit!r} must satisfy 0 <= lo < hi <= k_max ({k_max})."
            )
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
        meta={
            "method": method,
            "dimension": dimension,
            "delay": delay,
            "theiler": theiler,
            "n_reference": n_reference,
        },
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
