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

from typing import Any, cast

import numpy as np

from ...data import Ball, Box, Grid, as_region
from ...data.sampling import DEFAULT_REGION_RESOLUTION
from ...data.sampling import _region_example as _region_example
from ...errors import InvalidInputError, remedy
from .._common import reject_system

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
            raise InvalidInputError("CellGrid lo, hi, and counts must agree in length")
        if any(c < 1 for c in self.counts):
            raise InvalidInputError("CellGrid counts must be >= 1")
        if np.any(self.hi <= self.lo):
            # ``CellGrid`` is private and the caller has never typed it, so name
            # the AXIS and the argument they did type.  A pinned slice axis is the
            # common cause: the *seed* region may pin one (n == 1), but the
            # ``recurrence=`` box a trajectory must stay inside cannot be flat.
            flat = [i for i in range(self.lo.size) if self.hi[i] <= self.lo[i]]
            axes = ", ".join(str(i) for i in flat)
            raise InvalidInputError(
                f"the recurrence box is flat on axis {axes}: "
                f"lo={self.lo[flat].tolist()} hi={self.hi[flat].tolist()}. "
                "A trajectory has to live INSIDE this box, so every axis needs "
                "real width — it is the seed region, not this one, that pins a "
                "slice axis with a count of 1."
                + remedy(
                    "res = basins(system, [(-2.0, 2.0, 60), (-2.0, 2.0, 60), (0.0, 0.0, 1)],",
                    "             recurrence=[(-3.0, 3.0), (-3.0, 3.0), (-1.0, 1.0)])",
                    lead="Give the free axis width in recurrence=:",
                )
            )
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


def coerce_region(
    spec: Any,
    *,
    analysis: str,
    system: Any,
    want_grid: bool,
    args: str = "",
) -> Box | Ball | Grid:
    """Coerce the caller's ``region=`` argument — see :func:`tsdynamics.data.as_region`.

    A thin basin-layer adapter over the library's one region reading, kept so
    this subpackage's call sites stay short.  It adds nothing but the
    ``system``-sized error message; the grammar (one ``(lo, hi)`` bound or
    ``(lo, hi, n)`` triple **per state component**) lives in one place.
    """
    return as_region(
        spec,
        dim=int(getattr(system, "dim", 2) or 2),
        want_grid=want_grid,
        resolution=DEFAULT_REGION_RESOLUTION,
        analysis=analysis,
        system=system,
        args=args,
    )


def _recurrence_grid(
    region: Box | Ball | Grid, resolution: int | tuple[int, ...] = 100
) -> _CellGrid:
    """
    Build the recurrence :class:`_CellGrid` covering ``region``.

    A :class:`~tsdynamics.data.Grid` keeps its own ``counts``; a
    :class:`~tsdynamics.data.Box` or :class:`~tsdynamics.data.Ball` is tessellated
    at ``resolution`` cells per axis (a scalar applies to every axis).

    Parameters
    ----------
    region : Box, Ball, or Grid
    resolution : int or tuple of int, default 100
        Cells per axis for a Box/Ball region (ignored for a Grid, which carries
        its own ``counts`` — so the default is never consulted for a Grid).
    """
    if isinstance(region, Grid):
        return _CellGrid.from_grid(region)
    if isinstance(region, Box):
        lo, hi = region.lo, region.hi
    elif isinstance(region, Ball):
        lo = region.center - region.r
        hi = region.center + region.r
    else:
        raise InvalidInputError(
            f"a region must be a Box, a Ball or a Grid, got {type(region).__name__}."
            + remedy(
                "ts.analysis.basins(system, [(-2.0, 2.0, 200), (-2.0, 2.0, 200)])",
                lead="Bounds are accepted directly — one (lo, hi, n) triple per axis:",
            )
        )
    dim = lo.size
    counts: tuple[int, ...]
    if np.isscalar(resolution):
        counts = (cast(int, resolution),) * dim
    else:
        counts = tuple(cast("tuple[int, ...]", resolution))
    return _CellGrid(lo, hi, counts)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


#: The remedy the basin *metrics* need: they read an already computed label
#: image, so the generic "pass its trajectory" advice would send the caller the
#: wrong way entirely.
_BASIN_HINT = (
    "Compute the basins first and pass the result (or its label array):\n"
    "    res = basins(system, [(-2.0, 2.0, 200), (-2.0, 2.0, 200)])\n"
    "    {who}(res)"
)


def _as_label_array(basins: Any, *, analysis: str | None = None) -> np.ndarray:
    """
    Coerce a basin diagram to an integer label array.

    Accepts a :class:`~tsdynamics.analysis.basins.basins.BasinsResult` (uses its
    ``.labels``) or a raw integer array.  Labels are attractor ids ``>= 1`` with
    ``-1`` marking diverged / unlabelled cells (the convention every quantifier
    in this subpackage reads).

    A ``System`` is rejected up front: the basin *metrics* read an already
    computed label image, so the fix is to run
    :func:`~tsdynamics.analysis.basins` first — which is what the shared message
    builder says for a *result*-first analysis, so a named one is sent there and
    only an unnamed caller falls back to :data:`_BASIN_HINT`.  (The hand-written
    hint opened with "expects measured data", the wrong clause CONTRACT §5.6
    names: these metrics do not take measured data.)
    """
    if analysis:
        reject_system(basins, analysis=analysis)
    else:
        reject_system(basins, analysis=analysis, hint=_BASIN_HINT.format(who="metric"))
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
    return cast(np.ndarray, pts.mean(axis=0))


# ---------------------------------------------------------------------------
# Shared visualization palette
# ---------------------------------------------------------------------------

#: The categorical colormap the attractor/basin views share.  A 20-swatch
#: qualitative map gives every attractor id a distinct, repeatable colour.
PALETTE: str = "tab20"

#: The fixed colour the diverged / escape set (``DIVERGED == -1``) is drawn in,
#: kept out of the cyclic attractor palette so escape never aliases an attractor.
DIVERGED_COLOR: str = "lightgray"

#: Number of distinct swatches in :data:`PALETTE` (``tab20`` has 20).
_PALETTE_SIZE: int = 20


def _palette_index(attractor_id: int) -> int:
    """Map an attractor id (``>= 1``) to its swatch index in :data:`PALETTE`.

    Deterministic and shared by every basin-layer view, so the *same* attractor
    id maps to the *same* swatch across the :class:`AttractorSet` scatter and the
    basin image.  Ids are 1-based; the swatch index is ``(id - 1) mod 20``.
    """
    return (int(attractor_id) - 1) % _PALETTE_SIZE


def _palette_indices(ids: Any) -> dict[int, int]:
    """Return ``{attractor_id: palette swatch index}`` for a set of ids.

    The intra-result palette contract: a single source of truth both the
    :class:`AttractorSet` scatter and the :class:`BasinsResult` image key off, so
    a given id paints the same colour in both.
    """
    return {int(k): _palette_index(k) for k in ids}


def _category_labels(labels: Any) -> dict[int, str]:
    """Return ``{label value: display name}`` for a basin label field.

    A basin colour channel is an attractor *id*, so its colorbar should read as a
    categorical legend ("attractor 1", "attractor 2", "diverged") rather than as a
    numeric ramp over the ``BoundaryNorm`` bin edges.  Renderers pick this up from
    ``spec.meta["category_labels"]``.
    """
    _DIVERGED = -1  # attractors.DIVERGED; inlined to keep this leaf module import-free
    out: dict[int, str] = {}
    for v in np.unique(np.asarray(labels)):
        iv = int(v)
        out[iv] = "diverged" if iv == _DIVERGED else f"attractor {iv}"
    return out


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
