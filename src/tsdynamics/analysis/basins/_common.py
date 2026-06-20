r"""
Shared primitives for the attractor / basin layer.

Two concerns live here so the headline modules stay focused:

- **State-space tessellation** (:class:`_CellGrid`): bin a continuous (or
  discrete) state into one of ``counts[i]`` cells per axis, with out-of-region
  detection.  The recurrence attractor finder
  (:mod:`tsdynamics.analysis.basins.attractors`) and the grid-based metrics
  (:mod:`tsdynamics.analysis.basins.metrics`) share this binning so a basin
  *image* and the recurrence *labels* are laid out on the same lattice.
- **Coercion / driving helpers**: turn a :class:`~tsdynamics.data.Grid`,
  :class:`~tsdynamics.data.Box` or :class:`~tsdynamics.data.Ball` into a
  recurrence grid; coerce a :class:`~tsdynamics.analysis.basins.basins.BasinsResult`
  (or a raw label array) to an integer label array for the quantifiers; and a
  thin reinit/step driver over the :class:`~tsdynamics.families.System` protocol.

Everything is pure NumPy/SciPy — no compiled backend is consumed, so the layer
works uniformly across every system family.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...data import Ball, Box, Grid

__all__: list[str] = []


# ---------------------------------------------------------------------------
# State-space tessellation
# ---------------------------------------------------------------------------


class _CellGrid:
    """
    A regular tessellation of a box ``[lo, hi]`` into ``counts[i]`` cells per axis.

    Unlike :class:`~tsdynamics.data.Grid` (which enumerates lattice *points*),
    a :class:`_CellGrid` bins a continuous position into a half-open cell
    ``[lo + k*delta, lo + (k+1)*delta)``.  It is the substrate the recurrence
    attractor finder walks: a trajectory is followed cell by cell and an
    attractor is the recurrent set of cells it settles into.

    Parameters
    ----------
    lo, hi : array-like, shape (dim,)
        Lower/upper corner of the tessellated box.
    counts : tuple of int
        Number of cells along each axis (``>= 1``).
    """

    def __init__(self, lo: Any, hi: Any, counts: tuple[int, ...]) -> None:
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.counts = tuple(int(c) for c in counts)
        if not (self.lo.size == self.hi.size == len(self.counts)):
            raise ValueError("CellGrid lo, hi, and counts must agree in length")
        if any(c < 1 for c in self.counts):
            raise ValueError("CellGrid counts must be >= 1")
        if np.any(self.hi <= self.lo):
            raise ValueError("CellGrid requires hi > lo componentwise")
        self.dim = self.lo.size
        self._n = np.asarray(self.counts, dtype=np.int64)
        self.delta = (self.hi - self.lo) / self._n

    @classmethod
    def from_grid(cls, grid: Grid) -> _CellGrid:
        """Build a :class:`_CellGrid` whose cell counts are ``grid.counts``."""
        return cls(grid.lo, grid.hi, grid.counts)

    def index(self, u: np.ndarray) -> tuple[int, ...] | None:
        """
        Bin point ``u`` to its cell key, or ``None`` if it lies outside the box.

        Returns a hashable tuple of per-axis cell indices (the dict key the
        recurrence finder stores labels under).  Range checks happen in the float
        domain *before* any integer cast, so an arbitrarily large (but finite)
        coordinate is rejected cleanly rather than overflowing ``int64``.
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            rel = (np.asarray(u, dtype=float) - self.lo) / self.delta
        if not np.all(np.isfinite(rel)):
            return None
        n = self._n.astype(float)
        if np.any(rel < 0.0) or np.any(rel > n):  # closed box [lo, hi]
            return None
        # the closed top face (rel == n) belongs to the last cell.
        floor = np.floor(np.minimum(rel, n - 1e-9))
        return tuple(int(k) for k in floor.astype(np.int64))

    def center(self, key: tuple[int, ...]) -> np.ndarray:
        """Return the centre point of the cell with index ``key``."""
        return self.lo + (np.asarray(key, dtype=float) + 0.5) * self.delta


# ---------------------------------------------------------------------------
# Region → recurrence grid
# ---------------------------------------------------------------------------


def _recurrence_grid(region: Box | Ball | Grid, resolution: int | tuple[int, ...]) -> _CellGrid:
    """
    Build the recurrence :class:`_CellGrid` covering ``region``.

    A :class:`~tsdynamics.data.Grid` keeps its own ``counts``; a
    :class:`~tsdynamics.data.Box` or :class:`~tsdynamics.data.Ball` is tessellated
    at ``resolution`` cells per axis (a scalar applies to every axis).

    Parameters
    ----------
    region : Box, Ball, or Grid
    resolution : int or tuple of int
        Cells per axis for a Box/Ball region (ignored for a Grid, which carries
        its own ``counts``).
    """
    if isinstance(region, Grid):
        return _CellGrid.from_grid(region)
    if isinstance(region, Box):
        lo, hi = region.lo, region.hi
    elif isinstance(region, Ball):
        lo = region.center - region.r
        hi = region.center + region.r
    else:
        raise TypeError(f"unsupported region type {type(region).__name__}")
    dim = lo.size
    counts = (resolution,) * dim if np.isscalar(resolution) else tuple(resolution)
    return _CellGrid(lo, hi, counts)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_label_array(basins: Any) -> np.ndarray:
    """
    Coerce a basin diagram to an integer label array.

    Accepts a :class:`~tsdynamics.analysis.basins.basins.BasinsResult` (uses its
    ``.labels``) or a raw integer array.  Labels are attractor ids ``>= 1`` with
    ``-1`` marking diverged / unlabelled cells (the convention every quantifier
    in this subpackage reads).
    """
    labels = getattr(basins, "labels", basins)
    arr = np.asarray(labels)
    if not np.issubdtype(arr.dtype, np.integer):
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded, equal_nan=False):
            raise ValueError("basin labels must be integers (attractor ids; -1 = diverged).")
        arr = rounded.astype(np.int64)
    # Drop degenerate (size-1) axes so a slice of a higher-dim system is treated
    # at its effective dimension (a 2-D image of a 4-D flow is genuinely 2-D).
    squeezed = np.squeeze(arr)
    return squeezed if squeezed.ndim >= 1 else arr


def _representative(points: np.ndarray) -> np.ndarray:
    """Return a single representative point (the centroid) of an attractor cloud."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return pts.mean(axis=0)


def _apply_merge(labels: np.ndarray, merge: dict[int, int]) -> np.ndarray:
    """Remap a label array through ``{old_id: canonical_id}`` (others unchanged)."""
    if not merge:
        return labels
    out = labels.copy()
    for old, new in merge.items():
        if old != new:
            out[labels == old] = new
    return out


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
