r"""
Shared plumbing for the recurrence / RQA toolkit.

Holds the point-set coercion (:func:`_as_points`), metric handling
(:func:`_metric_p`), the run-length extractors that turn a recurrence matrix's
diagonals and columns into line-length histograms (:func:`_run_lengths`,
:func:`_runs_from_sorted`), and the recurrence-rate → threshold inversion
(:func:`_threshold_for_rate`).  The estimators live in :mod:`.matrix`,
:mod:`.rqa` and :mod:`.windowed`.

The :class:`~tsdynamics.data.Trajectory` is duck-typed (``.y``) to avoid an
import cycle through :mod:`tsdynamics.families` / :mod:`tsdynamics.data`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

__all__: list[str] = []


def _as_points(data: Any) -> np.ndarray:
    """Coerce a trajectory / array / series to a ``(N, dim)`` float array.

    Accepts anything with a ``.y`` attribute (a
    :class:`~tsdynamics.data.Trajectory` — duck-typed to avoid an import cycle),
    a 2-D ``(N, dim)`` array, or a 1-D ``(N,)`` series (a scalar measurement,
    treated as ``(N, 1)``).  Recurrence analysis is most often run on a
    delay-embedded reconstruction (see :func:`tsdynamics.analysis.embed`); a raw
    scalar series is accepted directly so an embedding step is optional.

    Parameters
    ----------
    data : Trajectory or array-like
        The point set.

    Returns
    -------
    ndarray, shape (N, dim)
        A contiguous ``float64`` array of the points.

    Raises
    ------
    ValueError
        If the data is not 1-D or 2-D, has fewer than two points, or contains
        non-finite values.
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
    (``p=inf`` — the standard RQA choice).  A number is passed through as the
    Minkowski exponent and must be ``>= 1`` (a smaller value is not a metric).
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


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of ``True`` in a 1-D boolean array.

    Used for diagonal lines, where the recurrence values along a diagonal are
    materialised densely.  Returns an empty array when there are no runs.
    """
    if mask.size == 0:
        return np.empty(0, dtype=np.intp)
    padded = np.empty(mask.size + 2, dtype=bool)
    padded[0] = padded[-1] = False
    padded[1:-1] = mask
    diff = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return ends - starts


def _runs_from_sorted(indices: np.ndarray) -> np.ndarray:
    """Lengths of runs of consecutive integers in a sorted index array.

    Used for vertical lines: a sparse column's stored row indices are ascending,
    so a maximal block of consecutive rows is a vertical recurrence line.  Avoids
    materialising the column densely.
    """
    if indices.size == 0:
        return np.empty(0, dtype=np.intp)
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    bounds = np.concatenate(([0], splits, [indices.size]))
    return np.diff(bounds)


def _sample_valid_distances(
    points: np.ndarray, p: float, theiler: int, *, max_pairs: int, seed: int
) -> np.ndarray:
    r"""Distances of pairs ``(i, j)`` with ``|i - j| > theiler``.

    Exact (all valid pairs) when the point set is small, otherwise a random
    sample — enough to read a stable quantile for the rate → threshold map.
    """
    n = points.shape[0]
    total_pairs = n * (n - 1) // 2

    def _dist(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        diff = points[i] - points[j]
        if p == float("inf"):
            return cast(np.ndarray, np.abs(diff).max(axis=1))
        if p == 1.0:
            return cast(np.ndarray, np.abs(diff).sum(axis=1))
        if p == 2.0:
            return cast(np.ndarray, np.sqrt(np.einsum("ij,ij->i", diff, diff)))
        return cast(np.ndarray, np.power(np.power(np.abs(diff), p).sum(axis=1), 1.0 / p))

    if total_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
        valid = (j - i) > theiler
        i, j = i[valid], j[valid]
    else:
        rng = np.random.default_rng(seed)
        # Oversample then filter the Theiler band; for the usual w << n this keeps
        # essentially all draws.
        draw = min(4 * max_pairs, 8_000_000)
        i = rng.integers(0, n, size=draw)
        j = rng.integers(0, n, size=draw)
        valid = np.abs(i - j) > theiler
        i, j = i[valid][:max_pairs], j[valid][:max_pairs]
    if i.size == 0:
        raise ValueError(f"Theiler window w={theiler} leaves no valid pairs for N={n}; reduce it.")
    return _dist(i, j)


def _threshold_for_rate(
    points: np.ndarray,
    rate: float,
    p: float,
    theiler: int,
    *,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> float:
    r"""Threshold :math:`\varepsilon` giving a target recurrence rate.

    The recurrence rate is the matrix density :math:`RR = \#\{R_{ij}=1\}/N^2`
    (the line of identity and the Theiler band carry no recurrences here).  With
    :math:`M` valid unordered pairs within :math:`\varepsilon` the symmetric
    matrix has :math:`2M` ones, so the target is :math:`2M/N^2 = RR`, i.e.
    :math:`\varepsilon` is the ``frac``-quantile of the valid pair distances with
    ``frac = RR * N^2 / (2 * n_valid_pairs)``.
    """
    n = points.shape[0]
    # True count of valid unordered pairs (|i - j| > theiler), so the quantile
    # fraction is taken against the population, not the (possibly subsampled) draw.
    total_pairs = n * (n - 1) // 2
    excluded = theiler * n - theiler * (theiler + 1) // 2
    total_valid = total_pairs - excluded
    if total_valid <= 0:
        raise ValueError(f"Theiler window w={theiler} leaves no valid pairs for N={n}; reduce it.")
    d = _sample_valid_distances(points, p, theiler, max_pairs=max_pairs, seed=seed)
    frac = rate * n * n / (2.0 * total_valid)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(np.quantile(d, frac))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
