r"""
Shared plumbing for the fractal-dimension estimators.

Holds the point-set coercion (:func:`_as_points`), metric handling, default
log-spaced scale grids, and the :class:`DimensionResult` container every
estimator returns.  The numerical estimators live in :mod:`.correlation`,
:mod:`.generalized` and :mod:`.fixedmass`; the scaling-region fit they all share
lives in :mod:`._scaling`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._scaling import local_slopes

__all__ = ["DimensionResult"]


def _as_points(data: Any) -> np.ndarray:
    """Coerce a trajectory / array / point list to a ``(N, dim)`` float array.

    Accepts anything with a ``.y`` attribute (a
    :class:`~tsdynamics.data.Trajectory` — duck-typed to avoid an import cycle),
    a 2-D ``(N, dim)`` array, or a 1-D ``(N,)`` series (treated as a single
    scalar component, i.e. ``(N, 1)``).

    Parameters
    ----------
    data : Trajectory or array-like
        The point set.

    Returns
    -------
    ndarray, shape (N, dim)
        A contiguous ``float64`` copy-safe view of the points.

    Raises
    ------
    ValueError
        If the data is not 1-D or 2-D, or has fewer than two points.
    """
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
        return a.max(axis=1)
    if p == 1.0:
        return a.sum(axis=1)
    if p == 2.0:
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))
    return np.power(np.power(a, p).sum(axis=1), 1.0 / p)


def _diameter(points: np.ndarray) -> float:
    """Largest per-axis extent of the point set (a cheap diameter proxy)."""
    return float((points.max(axis=0) - points.min(axis=0)).max())


def _default_radii(
    points: np.ndarray,
    *,
    p: float,
    n_radii: int = 24,
    lo_pct: float = 1.0,
    hi_pct: float = 50.0,
    n_sample_pairs: int = 4000,
    seed: int = 0,
) -> np.ndarray:
    """Data-adaptive log-spaced radii for the correlation sum.

    The bounds are read off a random sample of pairwise distances so the grid
    sits where the correlation sum is actually informative (above the smallest
    inter-point gaps, below the attractor diameter) rather than on an arbitrary
    fraction of the extent.

    Parameters
    ----------
    points : ndarray, shape (N, dim)
    p : float
        Minkowski exponent of the metric.
    n_radii : int, default 24
        Number of radii.
    lo_pct, hi_pct : float
        Percentiles of the sampled non-zero pair distances used as the lower
        and upper radius bounds.
    n_sample_pairs : int, default 4000
        Number of random pairs sampled to estimate the distance distribution.
    seed : int, default 0
        Seed for the pair sampling (keeps the default grid reproducible).

    Returns
    -------
    ndarray, shape (n_radii,)
    """
    n = points.shape[0]
    rng = np.random.default_rng(seed)
    m = min(n_sample_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    keep = i != j
    i, j = i[keep], j[keep]
    d = _pnorm(points[i] - points[j], p)
    d = d[d > 0.0]
    if d.size == 0:
        diam = _diameter(points) or 1.0
        return np.logspace(np.log10(diam / 1000.0), np.log10(diam / 4.0), n_radii)
    r_lo = np.percentile(d, lo_pct)
    r_hi = np.percentile(d, hi_pct)
    if not (r_lo > 0.0) or r_hi <= r_lo:
        r_hi = d.max()
        r_lo = max(d.min(), r_hi / 1000.0)
    return np.logspace(np.log10(r_lo), np.log10(r_hi), n_radii)


def _default_scales(
    points: np.ndarray,
    *,
    n_scales: int = 16,
    lo_frac: float = 1.0 / 200.0,
    hi_frac: float = 1.0 / 4.0,
) -> np.ndarray:
    """Log-spaced box sizes for the generalized (box-counting) dimensions.

    Spans ``[lo_frac, hi_frac]`` times the attractor diameter; the
    scaling-region fit trims the ends where boxes either each hold one point or
    merge into one.
    """
    diam = _diameter(points)
    if diam <= 0.0:
        raise ValueError("degenerate point set: zero extent in every dimension.")
    return np.logspace(np.log10(diam * lo_frac), np.log10(diam * hi_frac), n_scales)


@dataclass(frozen=True)
class DimensionResult:
    r"""A fractal-dimension estimate with the log--log curve it was read from.

    Returned by every estimator in this subpackage.  It behaves as the
    dimension number in arithmetic (``float(result)`` and comparisons), while
    carrying the full curve so the scaling region and local slopes can be
    inspected or plotted.

    Attributes
    ----------
    dimension : float
        The estimated dimension (the fitted slope).
    stderr : float
        Standard error of the slope over the selected scaling region.
    kind : str
        Which estimator produced it (``"correlation"``, ``"generalized"``,
        ``"fixed_mass"``).
    x, y : ndarray
        The log--log curve the slope was fitted to (log-radius vs log-C for the
        correlation sum; log-scale vs partition ordinate for the generalized
        dimensions; mean-log-radius vs log-mass for fixed mass).
    fit_slice : tuple[int, int]
        Inclusive ``(lo, hi)`` indices of the selected scaling region in
        ``x``/``y``.
    intercept : float
        Intercept of the fitted line.
    q : float or None
        Rényi order, for the generalized dimensions (``2.0`` for the correlation
        sum, ``None`` for fixed mass).
    """

    dimension: float
    stderr: float
    kind: str
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    fit_slice: tuple[int, int]
    intercept: float
    q: float | None = None

    def __float__(self) -> float:
        """Return the dimension number, so the result drops into arithmetic."""
        return float(self.dimension)

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
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.SCALING_FIT
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        lo, hi = self.fit_slice
        layers = [Layer(PlotKind.SCATTER, {"x": x, "y": y}, label=r"$\log C(r)$")]
        if x.size and hi >= lo:
            layers.append(
                Layer(
                    PlotKind.MARKERS, {"x": x[lo : hi + 1], "y": y[lo : hi + 1]}, label="fit region"
                )
            )
            fit_x = np.array([x[lo], x[hi]], dtype=float)
            layers.append(
                Layer(
                    PlotKind.LINE,
                    {"x": fit_x, "y": self.intercept + self.dimension * fit_x},
                    label=f"slope = {self.dimension:.3g}",
                )
            )
        q = "" if self.q is None else f" (q={self.q:g})"
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"{self.kind} dimension{q}  D = {self.dimension:.3f}",
            x=Axis(label=r"$\log r$"),
            y=Axis(label=r"$\log C(r)$"),
            layers=layers,
        )

    def __repr__(self) -> str:  # noqa: D105
        q = "" if self.q is None else f", q={self.q:g}"
        return (
            f"DimensionResult(kind={self.kind!r}{q}, "
            f"dimension={self.dimension:.4g} ± {self.stderr:.2g}, "
            f"n_fit={self.fit_slice[1] - self.fit_slice[0] + 1})"
        )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
