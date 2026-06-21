"""
State-space regions, samplers, and set distances.

The geometric primitives the attractor/basin layer is built on:

- :class:`Box`, :class:`Ball`, :class:`Grid` describe regions of state space,
  each with a ``contains`` predicate.
- :func:`sampler` turns a region into a thread-local, reproducible draw of
  initial conditions (Monte-Carlo basin sampling).
- :func:`grid_points` enumerates a region's grid (full-grid basin scans).
- :func:`set_distance` measures how far two point sets (e.g. candidate
  attractors) are apart — the matching primitive for deduplication and
  continuation.

These are pure NumPy/SciPy and consume nothing from the compiled backends, so
they work uniformly across every system family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "Ball",
    "Box",
    "Grid",
    "Region",
    "grid_points",
    "region",
    "sampler",
    "set_distance",
]


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Axis-aligned box ``[lo_i, hi_i]`` per dimension."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", np.asarray(self.lo, dtype=float))
        object.__setattr__(self, "hi", np.asarray(self.hi, dtype=float))
        if self.lo.shape != self.hi.shape:
            raise ValueError("Box lo and hi must have the same shape")
        if np.any(self.hi < self.lo):
            raise ValueError("Box requires hi >= lo componentwise")

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.lo.size

    def contains(self, u: Any) -> bool:
        """Whether point ``u`` lies in the box."""
        u = np.asarray(u, dtype=float)
        return bool(np.all(u >= self.lo) and np.all(u <= self.hi))


@dataclass(frozen=True)
class Ball:
    """Closed Euclidean ball of radius ``r`` about ``center``."""

    center: np.ndarray
    r: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))
        if self.r <= 0:
            raise ValueError("Ball radius must be positive")

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.center.size

    def contains(self, u: Any) -> bool:
        """Whether point ``u`` lies in the ball."""
        u = np.asarray(u, dtype=float)
        return bool(np.linalg.norm(u - self.center) <= self.r)


@dataclass(frozen=True)
class Grid:
    """Regular grid: ``counts[i]`` points spanning ``[lo_i, hi_i]`` per axis."""

    lo: np.ndarray
    hi: np.ndarray
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", np.asarray(self.lo, dtype=float))
        object.__setattr__(self, "hi", np.asarray(self.hi, dtype=float))
        object.__setattr__(self, "counts", tuple(int(c) for c in self.counts))
        if not (self.lo.size == self.hi.size == len(self.counts)):
            raise ValueError("Grid lo, hi, and counts must agree in length")
        if any(c < 1 for c in self.counts):
            raise ValueError("Grid counts must be >= 1")

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.lo.size

    @property
    def shape(self) -> tuple[int, ...]:
        """Per-axis node counts."""
        return self.counts

    def axes(self) -> list[np.ndarray]:
        """Per-axis coordinate vectors."""
        return [np.linspace(self.lo[i], self.hi[i], self.counts[i]) for i in range(self.dim)]

    def contains(self, u: Any) -> bool:
        """Whether point ``u`` lies in the grid's bounding box."""
        u = np.asarray(u, dtype=float)
        return bool(np.all(u >= self.lo) and np.all(u <= self.hi))


Region = Box | Ball | Grid


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def sampler(region: Region, *, seed: int | None = None) -> Callable[[], np.ndarray]:
    """
    Return a reproducible 0-argument sampler drawing points from ``region``.

    Box → uniform in the box; Ball → uniform in the ball (radial CDF, not the
    biased "uniform radius" trick); Grid → uniform over its bounding box (use
    :func:`grid_points` for the lattice itself).

    Parameters
    ----------
    region : Box, Ball, or Grid
    seed : int, optional
        Seeds a private ``numpy.random.Generator`` so draws are reproducible
        and independent of global RNG state.

    Returns
    -------
    callable
        ``draw() -> ndarray`` of shape ``(region.dim,)``.

    Examples
    --------
    >>> draw = sampler(Box([-1, -1], [1, 1]), seed=0)
    >>> draw().shape
    (2,)
    """
    rng = np.random.default_rng(seed)

    if isinstance(region, Box):
        lo, hi = region.lo, region.hi

        def draw_box() -> np.ndarray:
            return rng.uniform(lo, hi)

        return draw_box

    if isinstance(region, Ball):
        c, r, d = region.center, region.r, region.dim

        def draw_ball() -> np.ndarray:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            radius = r * rng.uniform() ** (1.0 / d)  # uniform-in-volume
            return c + radius * v

        return draw_ball

    if isinstance(region, Grid):
        lo, hi = region.lo, region.hi

        def draw_grid() -> np.ndarray:
            return rng.uniform(lo, hi)

        return draw_grid

    raise TypeError(f"unknown region type {type(region).__name__}")


def grid_points(grid: Grid) -> np.ndarray:
    """
    Enumerate every lattice point of ``grid`` (row-major / C order).

    Returns
    -------
    ndarray, shape ``(prod(counts), dim)``
        One row per grid node; reshape to ``grid.shape + (dim,)`` for a basin
        map laid out over the grid.
    """
    axes = grid.axes()
    if grid.dim == 1:
        return axes[0][:, None]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def region(spec: Any) -> Grid:
    """
    Build a :class:`Grid` from per-axis ``(lo, hi, n)`` triples.

    A terse constructor for the most common region — a regular lattice of
    initial conditions — so a basin / attractor scan reads
    ``region([(-2, 2, 200), (-2, 2, 200)])`` instead of the three-parallel-array
    ``Grid([-2, -2], [2, 2], (200, 200))``.

    Parameters
    ----------
    spec : sequence of (float, float, int)
        One ``(lo, hi, n)`` triple per state-space axis: the axis spans
        ``[lo, hi]`` with ``n`` evenly spaced grid points.

    Returns
    -------
    Grid

    Examples
    --------
    >>> g = region([(-2.0, 2.0, 200), (-2.0, 2.0, 200)])   # a 200x200 IC box
    >>> g.shape
    (200, 200)
    """
    triples = [tuple(axis) for axis in spec]
    if not triples or any(len(t) != 3 for t in triples):
        raise ValueError(
            "region() takes a non-empty sequence of (lo, hi, n) triples, one per "
            f"axis; got {spec!r}."
        )
    lo = np.array([float(t[0]) for t in triples], dtype=float)
    hi = np.array([float(t[1]) for t in triples], dtype=float)
    counts = tuple(int(t[2]) for t in triples)
    return Grid(lo=lo, hi=hi, counts=counts)


# ---------------------------------------------------------------------------
# Set distances
# ---------------------------------------------------------------------------

_SetMethod = Literal["centroid", "hausdorff", "minimum"]


def set_distance(
    a: Any,
    b: Any,
    *,
    method: _SetMethod = "centroid",
) -> float:
    """
    Distance between two point sets ``a`` and ``b`` (each ``(n, dim)``).

    Methods (Datseris & Wagemakers-style matching primitives):

    - ``"centroid"`` — Euclidean distance between the set centroids. O(n);
      the cheap default used for attractor matching across a continuation.
    - ``"hausdorff"`` — symmetric Hausdorff distance, a true metric:
      ``max(sup_a inf_b ‖a-b‖, sup_b inf_a ‖a-b‖)``. KD-tree accelerated.
    - ``"minimum"`` — the smallest pairwise distance (do the sets touch?).
      KD-tree accelerated.

    Accepts :class:`~tsdynamics.data.Trajectory` (uses ``.y``), arrays, or any
    array-like.
    """
    A = _as_points(a)
    B = _as_points(b)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"point sets live in different dimensions: {A.shape[1]} vs {B.shape[1]}")

    if method == "centroid":
        return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))

    from scipy.spatial import cKDTree

    tree_a, tree_b = cKDTree(A), cKDTree(B)
    if method == "minimum":
        d_ab, _ = tree_b.query(A, k=1)
        return float(np.min(d_ab))
    if method == "hausdorff":
        d_ab, _ = tree_b.query(A, k=1)
        d_ba, _ = tree_a.query(B, k=1)
        return float(max(np.max(d_ab), np.max(d_ba)))

    raise ValueError(f"unknown method {method!r}; use centroid, hausdorff, or minimum")


def _as_points(x: Any) -> np.ndarray:
    """Coerce a Trajectory / array-like to a 2-D ``(n, dim)`` point array."""
    y = getattr(x, "y", x)  # Trajectory → its state array
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D point set, got shape {arr.shape}")
    return arr


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
