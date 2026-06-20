r"""
Choosing the embedding dimension :math:`m`.

Two complementary neighbour-based estimators of the smallest dimension at which a
delay reconstruction unfolds the attractor (no self-intersections):

- **Cao's averaged false neighbours** (Cao, 1997).  For each point it tracks how
  the distance to its nearest neighbour changes when one extra delay coordinate
  is added.  The dimension-averaged ratio :math:`E_1(d)=E(d+1)/E(d)` rises to and
  then **saturates at** :math:`1`; the smallest :math:`d` past which it stops
  changing is the minimum embedding dimension.  A second quantity
  :math:`E_2(d)` stays :math:`\approx 1` for *random* data and so separates
  determinism from noise — the advantage of Cao's method over a single threshold.
- **Kennel's false nearest neighbours** (Kennel, Brown & Abarbanel, 1992).  A
  neighbour is *false* if adding a coordinate pushes it far apart (its proximity
  was a projection artefact).  The false-neighbour fraction falls to zero at the
  correct dimension.

Both read a single scalar series and a fixed delay (use
:func:`~tsdynamics.analysis.embedding.delay.optimal_delay`), and return an
:class:`EmbeddingDimension` whose ``int(result)`` is the recommended :math:`m`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._common import _as_series

__all__ = [
    "EmbeddingDimension",
    "cao_dimension",
    "embedding_dimension",
    "false_nearest_neighbors",
]


@dataclass(frozen=True)
class EmbeddingDimension:
    r"""A minimum-embedding-dimension estimate with the curve it was read from.

    Behaves as the dimension integer in arithmetic (``int(result)``), while
    carrying the per-dimension diagnostic so the saturation/decay can be
    inspected or plotted.

    Attributes
    ----------
    dimension : int
        The recommended minimum embedding dimension :math:`m`.
    dims : ndarray
        The dimensions :math:`d` at which the diagnostic was evaluated.
    method : str
        ``"cao"`` or ``"fnn"``.
    delay : int
        The delay (in samples) used to build the reconstructions.
    afn_e1, afn_e2 : ndarray or None
        Cao's :math:`E_1(d)` (saturates to 1 at the right dimension) and
        :math:`E_2(d)` (stays near 1 for stochastic data).  ``None`` for FNN.
    fnn_fraction : ndarray or None
        Kennel's false-nearest-neighbour fraction per dimension (decays to 0).
        ``None`` for Cao.
    """

    dimension: int
    dims: np.ndarray = field(repr=False)
    method: str
    delay: int
    afn_e1: np.ndarray | None = field(default=None, repr=False)
    afn_e2: np.ndarray | None = field(default=None, repr=False)
    fnn_fraction: np.ndarray | None = field(default=None, repr=False)

    def __int__(self) -> int:
        """Return the recommended embedding dimension, so it drops into ``embed``."""
        return int(self.dimension)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"EmbeddingDimension(method={self.method!r}, dimension={self.dimension}, "
            f"delay={self.delay}, dims={self.dims[0]}..{self.dims[-1]})"
        )


def _delay_columns(x: np.ndarray, n_cols: int, tau: int, rows: int) -> np.ndarray:
    """``(rows, n_cols)`` matrix whose column ``j`` is ``x[j*tau : j*tau + rows]``."""
    out = np.empty((rows, n_cols), dtype=float)
    for j in range(n_cols):
        start = j * tau
        out[:, j] = x[start : start + rows]
    return out


def _nearest_neighbor(
    points: np.ndarray, *, p: float, theiler: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest neighbour of every point under Minkowski-``p``, with a Theiler window.

    Returns ``(nn_index, nn_distance, valid)``.  A point with no neighbour outside
    the temporal window ``|i - j| > theiler`` is marked invalid (``valid[i] ==
    False``); its ``nn_index`` is ``-1`` and ``nn_distance`` is ``nan``.
    """
    from scipy.spatial import cKDTree

    m = points.shape[0]
    tree = cKDTree(points)
    k = min(m, theiler + 2)
    dist, idx = tree.query(points, k=k, p=p)
    if dist.ndim == 1:  # k == 1 (degenerate, m == 1): no neighbour exists
        return np.full(m, -1), np.full(m, np.nan), np.zeros(m, dtype=bool)
    rows = np.arange(m)
    allowed = np.abs(idx - rows[:, None]) > theiler
    valid = allowed.any(axis=1)
    first = np.argmax(allowed, axis=1)  # first True column per row (0 where none → masked)
    nn = np.where(valid, idx[rows, first], -1)
    nn_dist = np.where(valid, dist[rows, first], np.nan)
    return nn, nn_dist, valid


def _require_rows(rows: int, max_dim: int, tau: int) -> None:
    if rows < 10:
        raise ValueError(
            f"series too short: only {rows} aligned reference points for max_dim={max_dim}, "
            f"delay={tau}. Provide more samples, or reduce max_dim/delay."
        )


def cao_dimension(
    data: Any,
    *,
    delay: int = 1,
    max_dim: int = 10,
    threshold: float = 0.9,
    theiler_window: int = 0,
    component: int | str | None = None,
) -> EmbeddingDimension:
    r"""Cao's averaged-false-neighbour minimum embedding dimension.

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``component``).
    delay : int, default 1
        Embedding delay :math:`\tau` in samples (use
        :func:`~tsdynamics.analysis.embedding.delay.optimal_delay`).
    max_dim : int, default 10
        Largest dimension :math:`d` at which :math:`E_1` is evaluated.
    threshold : float, default 0.9
        Saturation threshold: the estimate is the smallest :math:`d` with
        :math:`E_1(d) \ge` ``threshold`` (the onset of the plateau at 1).
    theiler_window : int, default 0
        Exclude temporally-close neighbours with :math:`|i-j| \le w`.  Set a few
        autocorrelation times for densely sampled flows.
    component : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    EmbeddingDimension
        ``int(result)`` is the recommended :math:`m`; ``result.afn_e1`` /
        ``result.afn_e2`` carry the curves.

    References
    ----------
    L. Cao, "Practical method for determining the minimum embedding dimension of
    a scalar time series", *Physica D* **110**, 43 (1997).
    """
    x = _as_series(data, component=component)
    tau, max_dim = int(delay), int(max_dim)
    if tau < 1:
        raise ValueError("delay must be >= 1.")
    if max_dim < 2:
        raise ValueError("max_dim must be >= 2 (E1 compares consecutive dimensions).")
    w = int(theiler_window)

    big = max_dim + 1  # need E up to d = max_dim + 1 to form E1(max_dim)
    rows = x.size - big * tau
    _require_rows(rows, max_dim, tau)
    cols = _delay_columns(x, big + 1, tau, rows)  # columns 0 .. max_dim+1

    e = np.full(big + 1, np.nan)  # E(d),  d = 1 .. big
    estar = np.full(big + 1, np.nan)  # E*(d), d = 1 .. big
    for d in range(1, big + 1):
        nn, dist_d, has = _nearest_neighbor(cols[:, :d], p=np.inf, theiler=w)
        extra = np.abs(cols[:, d] - cols[nn, d])  # |x_{i+dτ} - x_{n+dτ}|
        valid = has & (dist_d > 0.0)
        if valid.sum() < 5:
            raise ValueError(
                f"too few valid neighbours at dimension {d} "
                f"(Theiler window {w} too large, or degenerate data)."
            )
        dist_d1 = np.maximum(dist_d[valid], extra[valid])
        e[d] = float(np.mean(dist_d1 / dist_d[valid]))
        estar[d] = float(np.mean(extra[valid]))

    dims = np.arange(1, max_dim + 1)
    e1 = np.array([e[d + 1] / e[d] for d in dims])
    e2 = np.array([estar[d + 1] / estar[d] for d in dims])

    reached = np.flatnonzero(e1 >= threshold)
    m_star = int(dims[reached[0]]) if reached.size else int(dims[int(np.argmax(e1))])
    return EmbeddingDimension(
        dimension=m_star, dims=dims, method="cao", delay=tau, afn_e1=e1, afn_e2=e2
    )


def false_nearest_neighbors(
    data: Any,
    *,
    delay: int = 1,
    max_dim: int = 10,
    rtol: float = 15.0,
    atol: float = 2.0,
    threshold: float = 0.01,
    theiler_window: int = 0,
    component: int | str | None = None,
) -> EmbeddingDimension:
    r"""Kennel's false-nearest-neighbour minimum embedding dimension.

    A neighbour found in dimension :math:`d` is *false* when extending to
    :math:`d+1` either stretches the pair by more than ``rtol`` relative to their
    :math:`d`-dimensional distance, or pushes them apart by more than ``atol``
    relative to the attractor size :math:`R_A` (the series' standard deviation) —
    the second test catching neighbours whose :math:`d`-dimensional distance is
    essentially zero.

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``component``).
    delay : int, default 1
        Embedding delay :math:`\tau` in samples.
    max_dim : int, default 10
        Largest dimension evaluated.
    rtol : float, default 15.0
        First-criterion tolerance :math:`R_{\text{tol}}` (Kennel et al. 1992).
    atol : float, default 2.0
        Second-criterion tolerance :math:`A_{\text{tol}}`.
    threshold : float, default 0.01
        The estimate is the smallest :math:`d` whose false-neighbour fraction is
        ``<= threshold``.
    theiler_window : int, default 0
        Exclude temporally-close neighbours with :math:`|i-j| \le w`.
    component : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    EmbeddingDimension
        ``int(result)`` is the recommended :math:`m`; ``result.fnn_fraction``
        carries the decay curve.

    References
    ----------
    M. B. Kennel, R. Brown and H. D. I. Abarbanel, "Determining embedding
    dimension for phase-space reconstruction using a geometrical construction",
    *Phys. Rev. A* **45**, 3403 (1992).
    """
    x = _as_series(data, component=component)
    tau, max_dim = int(delay), int(max_dim)
    if tau < 1:
        raise ValueError("delay must be >= 1.")
    if max_dim < 1:
        raise ValueError("max_dim must be >= 1.")
    w = int(theiler_window)

    r_attractor = float(np.std(x))
    if r_attractor == 0.0:
        raise ValueError("series is constant; the false-neighbour test is undefined.")

    rows = x.size - max_dim * tau  # need column max_dim → x[max_dim*tau : ...]
    _require_rows(rows, max_dim, tau)
    cols = _delay_columns(x, max_dim + 1, tau, rows)  # columns 0 .. max_dim

    dims = np.arange(1, max_dim + 1)
    fractions = np.empty(dims.size, dtype=float)
    for k, d in enumerate(dims):
        nn, r_d, has = _nearest_neighbor(cols[:, :d], p=2.0, theiler=w)
        extra = np.abs(cols[:, d] - cols[nn, d])
        if has.sum() < 5:
            raise ValueError(
                f"too few valid neighbours at dimension {d} "
                f"(Theiler window {w} too large, or degenerate data)."
            )
        # Criterion 1 written as a product (no division) so R_d == 0 is handled:
        # an extra-coordinate jump with zero base distance is a false neighbour.
        crit1 = extra[has] > rtol * r_d[has]
        r_d1 = np.sqrt(r_d[has] ** 2 + extra[has] ** 2)
        crit2 = r_d1 > atol * r_attractor
        fractions[k] = float(np.mean(crit1 | crit2))

    reached = np.flatnonzero(fractions <= threshold)
    m_star = int(dims[reached[0]]) if reached.size else int(dims[int(np.argmin(fractions))])
    return EmbeddingDimension(
        dimension=m_star, dims=dims, method="fnn", delay=tau, fnn_fraction=fractions
    )


def embedding_dimension(
    data: Any,
    *,
    method: str = "cao",
    delay: int = 1,
    max_dim: int = 10,
    component: int | str | None = None,
    **kwargs: Any,
) -> EmbeddingDimension:
    """Estimate the minimum embedding dimension by the chosen method.

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``component``).
    method : {"cao", "fnn"}, default "cao"
        ``"cao"`` → :func:`cao_dimension`; ``"fnn"`` → :func:`false_nearest_neighbors`.
    delay : int, default 1
        Embedding delay in samples.
    max_dim : int, default 10
        Largest dimension evaluated.
    component : int or str, optional
        Component selector for a multi-component input.
    **kwargs
        Forwarded to the selected estimator (``threshold``, ``theiler_window``,
        and for FNN ``rtol`` / ``atol``).

    Returns
    -------
    EmbeddingDimension
    """
    method = method.lower()
    if method == "cao":
        return cao_dimension(data, delay=delay, max_dim=max_dim, component=component, **kwargs)
    if method == "fnn":
        return false_nearest_neighbors(
            data, delay=delay, max_dim=max_dim, component=component, **kwargs
        )
    raise ValueError(f"unknown method {method!r}; use 'cao' or 'fnn'.")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
