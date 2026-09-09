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

from ...data import Ball, Box, Grid
from ...data import region as _build_region
from ...errors import InvalidInputError, remedy
from .._common import reject_system

__all__: list[str] = []

#: Lattice nodes per axis when a caller writes a region as bare ``(lo, hi)``
#: bounds rather than ``(lo, hi, n)`` triples.  A *resolution* is not a modelling
#: choice — it trades picture detail against runtime and is recorded in the
#: result — so it is defaulted rather than demanded, and the triple form is
#: right there in the signature for anyone who wants to set it.
DEFAULT_REGION_RESOLUTION = 100


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


def _region_example(system: Any, *, triples: bool = True) -> str:
    """Return a region literal sized to ``system``, as source text.

    The example an error message shows must be the *caller's* region, not a
    stock 2-D one: it names as many axes as the system has state components, and
    spans a box wide enough to be worth trying.  A system whose ``dim`` cannot be
    read falls back to two axes.
    """
    dim = int(getattr(system, "dim", 2) or 2)
    axis = f"(-2.0, 2.0, {DEFAULT_REGION_RESOLUTION})" if triples else "(-2.0, 2.0)"
    if dim <= 3:
        return "[" + ", ".join([axis] * dim) + "]"
    return f"[{axis}] * {dim}"


def coerce_region(
    spec: Any,
    *,
    analysis: str,
    system: Any,
    want_grid: bool,
    alias: str | None = None,
    args: str = "",
) -> Box | Ball | Grid:
    """Coerce whatever the caller passed as ``region=`` into a region primitive.

    The basin layer's regions are :class:`~tsdynamics.data.Box` /
    :class:`~tsdynamics.data.Ball` / :class:`~tsdynamics.data.Grid` objects, but
    the *obvious* thing to type is the bounds themselves — so a sequence of
    per-axis ``(lo, hi, n)`` triples (the :func:`tsdynamics.data.region`
    spelling) or bare ``(lo, hi)`` bounds is accepted and built into the right
    primitive here.  A missing or unrecognisable region is rejected with the
    literal line to type for *this* system.

    Parameters
    ----------
    spec : Any
        The caller's ``region=`` argument: a region primitive, a sequence of
        ``(lo, hi, n)`` triples, a sequence of ``(lo, hi)`` bounds, or ``None``.
    analysis : str
        The function's own name, used to open the message.
    system : System
        The system being analysed — its ``dim`` sizes the suggested region.
    want_grid : bool
        ``True`` for a routine that scans a *lattice* (it needs counts, so bare
        bounds are filled in at :data:`DEFAULT_REGION_RESOLUTION` per axis);
        ``False`` for a routine that *samples* the region, where bare bounds
        become a :class:`~tsdynamics.data.Box` and no resolution is invented.
    alias : str, optional
        The shorter headline spelling of the same function, when it has one
        (``basins_of_attraction`` is exported as ``ts.basins``).  The message
        names **both**, and the runnable line uses the alias: a user who typed
        ``ts.basins`` must not be answered about a name they never typed, and a
        user who typed the long name must still recognise their own call.
    args : str, optional
        Source text for the positional arguments that sit *between* the system
        and the region in this function's signature (``continuation`` takes
        ``param, values`` first).  Without it the suggested line would have the
        wrong arity — a remedy that does not run is worse than none.

    Returns
    -------
    Box, Ball, or Grid

    Raises
    ------
    InvalidInputError
        If ``spec`` is ``None`` or is not a region the layer can build.
    """
    if isinstance(spec, (Box, Ball, Grid)):
        return spec

    who = f"{analysis}() (exported as ts.{alias})" if alias else f"{analysis}()"
    call = alias or analysis

    def _refuse(detail: str) -> InvalidInputError:
        return InvalidInputError(
            f"{who} needs a region: the box of initial conditions to "
            f"classify — {detail}."
            + remedy(
                f"ts.{call}(system, {args}{_region_example(system, triples=want_grid)})",
                lead=(
                    "Pass one (lo, hi, n) triple per state component:"
                    if want_grid
                    else "Pass one (lo, hi) bound per state component:"
                ),
            )
        )

    if spec is None:
        raise _refuse(
            "there is no natural default, because it depends on where your attractors live"
        )

    try:
        rows = [tuple(float(v) for v in axis) for axis in spec]
    except (TypeError, ValueError) as err:
        raise _refuse(f"got {type(spec).__name__}, which is not a region") from err
    if not rows or not all(len(row) == len(rows[0]) for row in rows):
        raise _refuse("the per-axis bounds must all have the same shape")

    width = len(rows[0])
    if width == 3:
        return _build_region([(lo, hi, int(n)) for lo, hi, n in rows])
    if width == 2:
        lo = np.array([row[0] for row in rows], dtype=float)
        hi = np.array([row[1] for row in rows], dtype=float)
        if not want_grid:
            return Box(lo, hi)
        return Grid(lo, hi, (DEFAULT_REGION_RESOLUTION,) * len(rows))
    raise _refuse(f"each axis needs (lo, hi) or (lo, hi, n), got {width} numbers")


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
                "ts.basins(system, [(-2.0, 2.0, 200), (-2.0, 2.0, 200)])",
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
    "    res = basins_of_attraction(system, Grid(lo, hi, counts))\n"
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
    :func:`~tsdynamics.analysis.basins_of_attraction` first.
    """
    reject_system(basins, analysis=analysis, hint=_BASIN_HINT.format(who=analysis or "metric"))
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
