r"""
Shared plumbing for the fractal-dimension estimators.

Holds the point-set coercion (:func:`_as_points`), metric handling, default
log-spaced scale grids, and the :class:`DimensionResult` container every
estimator returns.  The numerical estimators live in :mod:`.correlation`,
:mod:`.generalized` and :mod:`.fixedmass`; the scaling-region fit they all share
lives in :mod:`._scaling`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .._common import reject_system
from .._result import ScalingResult
from ._scaling import local_slopes

__all__ = ["DimensionResult"]


def _as_points(data: Any, *, analysis: str | None = None) -> np.ndarray:
    """Coerce a trajectory / array / point list to a ``(N, dim)`` float array.

    Accepts anything with a ``.y`` attribute (a
    :class:`~tsdynamics.data.Trajectory` — duck-typed to avoid an import cycle),
    a 2-D ``(N, dim)`` array, or a 1-D ``(N,)`` series (treated as a single
    scalar component, i.e. ``(N, 1)``).

    Parameters
    ----------
    data : Trajectory or array-like
        The point set.
    analysis : str, optional
        Name of the calling public function, used only to open the
        ``System``-was-passed error message.

    Returns
    -------
    ndarray, shape (N, dim)
        A contiguous ``float64`` copy-safe view of the points.

    Raises
    ------
    InvalidInputError
        If ``data`` is a ``System`` rather than measured data
        (:func:`~tsdynamics.analysis._common.reject_system`) -- every
        dimension estimator is data-first.
    ValueError
        If the data is not 1-D or 2-D, or has fewer than two points.
    """
    reject_system(data, analysis=analysis)
    y = getattr(data, "y", None)
    arr = np.asarray(y if y is not None else data, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"expected a (N, dim) point set or a 1-D series, got array of shape {arr.shape}."
        )
    if arr.shape[0] < 2:
        raise ValueError(f"need at least two points, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("point set contains non-finite values (nan/inf).")
    return np.ascontiguousarray(arr)


def _metric_p(metric: str | float) -> float:
    """Map a metric name (or a Minkowski exponent) to a ``scipy`` ``p`` value.

    Recognises ``"euclidean"`` (``p=2``), ``"manhattan"``/``"cityblock"``/``"l1"``
    (``p=1``), and ``"chebyshev"``/``"max"``/``"maximum"``/``"infinity"``
    (``p=inf``).  A number is passed through as the Minkowski exponent and must be
    ``>= 1`` — a smaller value is not a metric, and the estimators disagree on
    whether ``scipy`` even accepts it (``count_neighbors`` does, ``query`` does
    not), so it is rejected here for a single, predictable contract.
    """
    if isinstance(metric, (int, float)):
        p = float(metric)
        if p < 1.0:
            raise ValueError(f"Minkowski exponent must be >= 1 (a metric), got {metric!r}.")
        return p
    key = metric.lower()
    table = {
        "euclidean": 2.0,
        "l2": 2.0,
        "manhattan": 1.0,
        "cityblock": 1.0,
        "l1": 1.0,
        "chebyshev": float("inf"),
        "max": float("inf"),
        "maximum": float("inf"),
        "infinity": float("inf"),
        "inf": float("inf"),
    }
    if key not in table:
        raise ValueError(
            f"unknown metric {metric!r}; use 'euclidean', 'manhattan', 'chebyshev', "
            "or a numeric Minkowski exponent."
        )
    return table[key]


def _pnorm(diff: np.ndarray, p: float) -> np.ndarray:
    """Row-wise Minkowski-``p`` norm of a ``(M, dim)`` array of differences."""
    a = np.abs(diff)
    if p == float("inf"):
        return np.asarray(a.max(axis=1))
    if p == 1.0:
        return np.asarray(a.sum(axis=1))
    if p == 2.0:
        return np.asarray(np.sqrt(np.einsum("ij,ij->i", diff, diff)))
    return np.asarray(np.power(np.power(a, p).sum(axis=1), 1.0 / p))


def _diameter(points: np.ndarray) -> float:
    """Largest per-axis extent of the point set (a cheap diameter proxy)."""
    return float((points.max(axis=0) - points.min(axis=0)).max())


#: Default upper target for the automatic correlation-sum radius grid, expressed
#: as a value of :math:`C(r)` rather than a radius.  Above ~10% of all pairs the
#: correlation sum is already bending over toward its saturation at 1, and that
#: bend biases the fitted slope down.  (The pre-v6 default reached the *median*
#: pair distance, ``C = 0.5``, which is deep into the saturated regime; that
#: alone cost the Hénon :math:`D_2` about 0.05.)
_DEFAULT_C_HI = 0.1

#: Default lower target for the automatic radius grid.  ``C(r) = 1e-4`` is small
#: enough to sit well inside the scaling region of every reference set here, and
#: is raised for small samples by :data:`_MIN_PAIRS_AT_C_LO`.
_DEFAULT_C_LO = 1e-4

#: Statistical floor on the smallest radius: the grid never starts below the
#: radius enclosing this many pairs, so ``C(r_lo)`` is never estimated from a
#: handful of pairs.
_MIN_PAIRS_AT_C_LO = 200.0


def _radii_for_c_window(
    c_of_r: Any,
    points: np.ndarray,
    *,
    n_radii: int,
    c_lo: float | None = None,
    c_hi: float = _DEFAULT_C_HI,
    n_pilot: int = 32,
) -> np.ndarray:
    r"""Log-spaced radii spanning a target window of the correlation sum itself.

    The informative part of a Grassberger--Procaccia curve is delimited by
    :math:`C(r)`, not by a radius: below some :math:`C` there are too few pairs
    for the count to mean anything, and above :math:`C \approx 0.1` the sum is
    bending over toward its saturation at 1.  So the grid is chosen by inverting
    a cheap pilot :math:`C(r)` at the two target levels instead of guessing
    percentiles of the distance distribution.

    ``c_of_r`` is a callable mapping an array of radii to :math:`C(r)` — the
    caller passes its own tree-based correlation sum (with whatever Theiler
    window is in force), so the pilot and the final grid are consistent.

    Parameters
    ----------
    c_of_r : callable
        ``radii -> C(radii)``, monotone non-decreasing.
    points : ndarray, shape (N, dim)
        The point set (used only for its extent).
    n_radii : int
        Number of radii in the returned grid.
    c_lo : float, optional
        Lower target correlation-sum level.  ``None`` uses
        :data:`_DEFAULT_C_LO`, raised so at least :data:`_MIN_PAIRS_AT_C_LO`
        pairs fall below ``r_lo``.
    c_hi : float, default 0.1
        Upper target correlation-sum level.
    n_pilot : int, default 32
        Number of radii in the pilot sweep.

    Returns
    -------
    ndarray, shape (n_radii,)
    """
    n = points.shape[0]
    n_pairs = n * (n - 1) / 2.0
    lo_level = _DEFAULT_C_LO if c_lo is None else float(c_lo)
    lo_level = max(lo_level, _MIN_PAIRS_AT_C_LO / max(n_pairs, 1.0))
    hi_level = float(c_hi)
    if not (0.0 < lo_level < hi_level < 1.0):
        raise ValueError(
            f"correlation-sum targets must satisfy 0 < c_lo < c_hi < 1, got {lo_level} / {hi_level}."
        )

    diam = _diameter(points)
    if diam <= 0.0:
        raise ValueError("degenerate point set: zero extent in every dimension.")
    pilot = np.logspace(np.log10(diam * 1e-6), np.log10(diam), n_pilot)
    c = np.asarray(c_of_r(pilot), dtype=float)
    ok = c > 0.0
    if ok.sum() < 2:  # pragma: no cover - a point set with no resolvable pairs
        return np.logspace(np.log10(diam / 1000.0), np.log10(diam / 4.0), n_radii)
    # Invert log C -> log r.  ``np.interp`` needs a strictly increasing abscissa,
    # so collapse the flat stretches C(r) has wherever a radius band holds no
    # pair; the first occurrence of each level is the smallest radius reaching it.
    ly, first = np.unique(np.log(c[ok]), return_index=True)
    lx = np.log(pilot[ok])[first]
    r_lo = float(np.exp(np.interp(np.log(lo_level), ly, lx)))
    r_hi = float(np.exp(np.interp(np.log(hi_level), ly, lx)))
    if not (r_hi > r_lo > 0.0):  # degenerate / near-constant C: fall back to the pilot span
        r_lo, r_hi = float(pilot[ok][0]), float(pilot[ok][-1])
    return np.logspace(np.log10(r_lo), np.log10(r_hi), n_radii)


#: Cap on the lag at which the automatic Theiler window stops looking, and on
#: the window it may return: a window wider than a tenth of the series would
#: start removing a material fraction of *all* pairs, not just the temporally
#: correlated ones.
_MAX_AUTO_THEILER_FRAC = 0.1

#: Absolute cap on the automatic Theiler window, so a slowly-decorrelating
#: (e.g. quasi-periodic) series cannot silently consume the whole pair budget.
_MAX_AUTO_THEILER = 1000

#: A lag counts as "temporally decorrelated" once points that far apart in time
#: are typically at least this fraction of the *typical* inter-point distance
#: apart in state space.
_THEILER_SEPARATION_FRAC = 0.5

#: Reference points sampled when profiling the space--time separation, and pairs
#: sampled for the typical-distance reference.  Both are plenty for a median.
_THEILER_N_REF = 2000
_THEILER_N_PAIRS = 4000


def _auto_theiler(points: np.ndarray) -> int:
    r"""Theiler window from the space--time separation profile of the point set.

    Temporally adjacent samples of a densely-sampled flow are close in state
    space *because they are adjacent in time*, not because the invariant measure
    put them there.  Counting them inflates :math:`C(r)` at small ``r`` and
    flattens the log--log curve, biasing every neighbour-based dimension
    (Theiler 1986).  The remedy is to count only pairs with :math:`|i - j| > w`;
    the question is what ``w`` should be by default, and ``0`` — the pre-v6
    default — is the one answer that is never right for a densely sampled flow.

    The window is read off the **space--time separation profile** (Provenzale et
    al. 1992): the median distance :math:`d(k)` between samples ``k`` apart in
    time, compared with the median distance :math:`d_\infty` between random
    pairs.  ``w`` is the smallest lag whose points are already typically
    :data:`_THEILER_SEPARATION_FRAC` of :math:`d_\infty` apart — i.e. the lag at
    which temporal adjacency has stopped implying spatial proximity.  That is a
    more direct statement of what the window is for than an autocorrelation
    time, and it is scale-free, so it returns ``1`` for an already-decorrelated
    point cloud (a random sample, a chaotic map orbit) where a linear
    autocorrelation of a structured coordinate could still read as correlated.

    Lags are probed on a log grid, so the returned window is exact for small
    values and coarse for large ones — which is all a Theiler band needs.

    **When the profile does not saturate, the answer is ``w = 1``.**  A
    non-stationary or drifting orbit — the catalogue's ``Chirikov`` standard map
    is one, since its angle is not wrapped — has :math:`d(k)` still climbing at
    the largest inspectable lag, so no separation lag exists.  It is tempting to
    read the window off the profile's own top instead (``d(k) >= f * d(k_max)``),
    or to warn that the sample is undersampled.  Both are wrong, and the first is
    actively destructive: on a 3000-point drifting Chirikov orbit — an invariant
    curve, true :math:`D_2 = 1` — the estimate goes ``w=1`` → 1.03, ``w=50`` →
    1.64, ``w=150`` → 2.47, ``w=300`` → 3.48.  Excluding the near-diagonal band
    of a drifting curve removes exactly the pairs that carry its geometry and
    leaves the drift.  A window is justified only by a *demonstrated* separation
    of time scales, so with none demonstrated the correction is not applied.  The
    resolved window is reported in the result's ``meta["theiler"]``.

    Parameters
    ----------
    points : ndarray, shape (N, dim)
        The point set, **in sampling order**.

    Returns
    -------
    int
        The recommended Theiler window ``w >= 1``.

    References
    ----------
    J. Theiler, "Spurious dimension from correlation algorithms applied to
    limited time-series data", *Phys. Rev. A* **34**, 2427 (1986).

    A. Provenzale, L. A. Smith, R. Vio and G. Murante, "Distinguishing between
    low-dimensional dynamics and randomness in measured time series", *Physica D*
    **58**, 31 (1992).
    """
    n = points.shape[0]
    max_lag = int(min(_MAX_AUTO_THEILER, int(n * _MAX_AUTO_THEILER_FRAC)))
    if n < 16 or max_lag < 1:
        return 1

    rng = np.random.default_rng(0)  # fixed: the default window must be reproducible
    i = rng.integers(0, n, size=_THEILER_N_PAIRS)
    j = rng.integers(0, n, size=_THEILER_N_PAIRS)
    d_typical = float(np.median(_pnorm(points[i] - points[j], 2.0)))
    if not (d_typical > 0.0):  # pragma: no cover - degenerate sets are rejected upstream
        return 1

    target = _THEILER_SEPARATION_FRAC * d_typical
    step = max(1, n // _THEILER_N_REF)
    lags = np.unique(np.round(np.logspace(0.0, np.log10(max_lag), 60)).astype(int))
    for k in lags:
        idx = np.arange(0, n - int(k), step)
        if idx.size == 0:  # pragma: no cover - guarded by max_lag <= n // 10
            break
        if float(np.median(_pnorm(points[idx + int(k)] - points[idx], 2.0))) >= target:
            return int(k)
    return 1


def _resolve_theiler(theiler: int | str, points: np.ndarray) -> int:
    """Coerce the public ``theiler=`` argument (an int, or ``"auto"``) to a window."""
    if isinstance(theiler, str):
        if theiler.lower() != "auto":
            raise ValueError(f"unknown theiler {theiler!r}; use a non-negative int or 'auto'.")
        return _auto_theiler(points)
    w = int(theiler)
    if w < 0:
        raise ValueError("theiler must be non-negative.")
    return w


@dataclass(frozen=True, eq=False)
class DimensionResult(ScalingResult):
    r"""A fractal-dimension estimate with the log--log curve it was read from.

    Returned by every estimator in this subpackage.  A
    :class:`~tsdynamics.analysis._result.ScalingResult` — the dimension is the
    fitted slope of a log--log curve — so it inherits the canonical ``estimate`` /
    ``abscissa`` / ``ordinate`` / ``fit_region`` schema, the result surface
    (``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam) and behaves
    as the dimension number (``float(result)`` and comparisons).  Domain-named
    ``@property`` aliases (:attr:`dimension`, :attr:`x`, :attr:`y`,
    :attr:`fit_slice`) preserve the original field names.

    Attributes
    ----------
    estimate : float
        The estimated dimension (the fitted slope).  Aliased :attr:`dimension`.
    stderr : float
        Standard error of the slope over the selected scaling region.
    kind : str
        Which estimator produced it (``"correlation"``, ``"generalized"``,
        ``"fixed_mass"``).
    abscissa, ordinate : ndarray
        The log--log curve the slope was fitted to (log-radius vs log-C for the
        correlation sum; log-scale vs partition ordinate for the generalized
        dimensions; mean-log-radius vs log-mass for fixed mass).  Aliased
        :attr:`x`, :attr:`y`.
    fit_region : tuple[int, int]
        Inclusive ``(lo, hi)`` indices of the selected scaling region.  Aliased
        :attr:`fit_slice`.
    intercept : float
        Intercept of the fitted line.
    q : float or None
        Rényi order, for the generalized dimensions (``2.0`` for the correlation
        sum, ``None`` for fixed mass).
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("kind", "dimension", "stderr", "q")

    kind: str = ""
    q: float | None = None
    #: ``False`` when the estimate is self-evidently unresolved — currently, when the
    #: shared-partition Renyi spectrum increases with ``q``, which is impossible for
    #: any measure and therefore proves the box count has not converged at the scales
    #: used.  The number is still returned (refusing outright would make the estimator
    #: useless on exactly the systems people reach for), but it must not be read as an
    #: answer.  Mirrors ``LyapunovFromData.trusted``.
    trusted: bool = True

    @property
    def dimension(self) -> float:
        """The estimated dimension (alias of :attr:`estimate`)."""
        return float(self.estimate)

    @property
    def x(self) -> np.ndarray:
        """The abscissa of the scaling curve (alias of :attr:`abscissa`; see :attr:`kind`)."""
        return self.abscissa

    @property
    def y(self) -> np.ndarray:
        """The ordinate of the scaling curve (alias of :attr:`ordinate`; see :attr:`kind`)."""
        return self.ordinate

    @property
    def fit_slice(self) -> tuple[int, int]:
        """The selected scaling region (alias of :attr:`fit_region`)."""
        return self.fit_region

    @property
    def local_slopes(self) -> np.ndarray:
        """Pointwise local slope of the log--log curve (the diagnostic plateau)."""
        return local_slopes(self.x, self.y)

    @property
    def scaling_window(self) -> tuple[float, float]:
        """The ``(x_lo, x_hi)`` abscissa span of the selected scaling region."""
        lo, hi = self.fit_slice
        return float(self.x[lo]), float(self.x[hi])

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe this dimension estimate as a backend-agnostic :class:`PlotSpec`.

        Builds a ``SCALING_FIT`` spec — the log--log curve as a scatter layer, the
        selected scaling region highlighted, and the fitted line drawn from
        :attr:`intercept` and :attr:`dimension` — the same schema every scaling
        estimator emits, so a single ``result.plot.scaling()`` renders it.  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
        plotting library.

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

        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        # Axis/series labels differ by estimator kind (see the abscissa/ordinate
        # field docstring): correlation = log-radius vs log-C(r); generalized =
        # log-scale vs partition ordinate; fixed_mass = mean-log-radius vs the
        # digamma log-mass.  Unknown kinds fall back to neutral labels.
        xlabel, ylabel = {
            "correlation": (r"$\log r$", r"$\log C(r)$"),
            "generalized": (r"$\log \epsilon$", r"partition ordinate"),
            "fixed_mass": (r"$\langle \log r_k \rangle$", r"$\psi(k)$"),
        }.get(self.kind, (r"$\log$ scale", r"$\log$ measure"))
        q = "" if self.q is None else f" (q={self.q:g})"
        return pb.scaling_fit(
            kind,
            x,
            y,
            fit_region=self.fit_slice,
            slope=self.dimension,
            intercept=self.intercept,
            curve_label=ylabel,
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{self.kind} dimension{q}  D = {self.dimension:.3f}",
        )

    def __repr__(self) -> str:  # noqa: D105
        q = "" if self.q is None else f", q={self.q:g}"
        flag = "" if self.trusted else ", UNTRUSTED (unresolved: D_q rises with q)"
        return (
            f"DimensionResult(kind={self.kind!r}{q}, "
            f"dimension={self.dimension:.4g} ± {self.stderr:.2g}, "
            f"n_fit={self.fit_slice[1] - self.fit_slice[0] + 1}{flag})"
        )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
