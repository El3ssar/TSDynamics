"""
State-space regions, samplers, and set distances.

The geometric primitives the attractor/basin layer is built on:

- :class:`Box`, :class:`Ball`, :class:`Grid` describe regions of state space,
  each with a ``contains`` predicate.
- :func:`as_region` is **the** region reading in the library: one ``(lo, hi)``
  — or ``(lo, hi, n)`` — pair **per state component**.  Every public
  ``region=`` argument goes through it, so plain bounds reach every door and
  the three primitives above are accepted but never *required*.
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
    "as_region",
    "grid_points",
    "region",
    "sampler",
    "set_distance",
]

#: Lattice nodes per axis when a caller writes a region as bare ``(lo, hi)``
#: bounds rather than ``(lo, hi, n)`` triples.  A *resolution* is not a modelling
#: choice — it trades picture detail against runtime and is recorded in the
#: result — so it is defaulted rather than demanded, and the triple form is
#: right there in the signature for anyone who wants to set it.
DEFAULT_REGION_RESOLUTION = 100


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


def _coerce_query(u: Any, dim: int) -> tuple[np.ndarray, bool]:
    """Coerce a containment query to a 2-D ``(n, dim)`` array.

    A single point ``(dim,)`` becomes a one-row batch and the boolean flag
    ``True`` (so the caller can squeeze the result back to a scalar); a batch
    ``(n, dim)`` passes through with flag ``False``.  Any other shape — in
    particular a flat array whose length is not ``dim`` — raises, so a batch can
    never silently collapse into one conflated truth value.
    """
    arr = np.asarray(u, dtype=float)
    if arr.ndim == 0 and dim == 1:
        # A bare scalar is a valid single point of a 1-D region.
        return arr.reshape(1, 1), True
    if arr.ndim == 1 and arr.shape[0] == dim:
        return arr[None, :], True
    if arr.ndim == 2 and arr.shape[1] == dim:
        return arr, False
    raise ValueError(
        f"containment query must have shape ({dim},) for a single point or "
        f"(n, {dim}) for a batch; got shape {arr.shape}."
    )


@dataclass(frozen=True)
class Box:
    """Axis-aligned box ``[lo_i, hi_i]`` per dimension.

    A closed, axis-aligned hyper-rectangle of state space: a point lies in the
    box iff ``lo_i <= u_i <= hi_i`` for every axis ``i``.  It is one of the
    region primitives (alongside :class:`Ball` and :class:`Grid`) that the
    sampler / basin / attractor layer is built on.

    Parameters
    ----------
    lo, hi : array_like, shape (dim,)
        Per-axis lower and upper corners.  Coerced to ``float`` arrays; ``hi``
        must be ``>= lo`` componentwise.

    Raises
    ------
    ValueError
        If ``lo`` and ``hi`` differ in shape, or ``hi < lo`` on any axis.

    Examples
    --------
    >>> b = Box([-1.0, -1.0], [1.0, 1.0])
    >>> b.contains([0.0, 0.0])
    True
    >>> b.contains([[0.0, 0.0], [2.0, 0.0]])      # batch query → per-row mask
    array([ True, False])
    """

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

    def contains(self, u: Any) -> Any:
        """Whether point(s) ``u`` lie in the box.

        Parameters
        ----------
        u : array_like
            A single point of shape ``(dim,)`` **or** a batch of shape
            ``(n, dim)``.  Any other shape raises (so a batch can never silently
            collapse into one conflated truth value).

        Returns
        -------
        bool or ndarray of bool
            A scalar ``bool`` for a single point, or an ``(n,)`` boolean mask —
            row ``i`` is ``True`` iff point ``i`` lies in the box.

        Raises
        ------
        ValueError
            If ``u`` is neither a ``(dim,)`` point nor an ``(n, dim)`` batch.
        """
        pts, scalar = _coerce_query(u, self.dim)
        mask = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
        return bool(mask[0]) if scalar else mask


@dataclass(frozen=True)
class Ball:
    """Closed Euclidean ball of radius ``r`` about ``center``.

    The set of points within Euclidean distance ``r`` of ``center`` (inclusive).
    Together with :class:`Box` and :class:`Grid` it is one of the region
    primitives the sampler / basin / attractor layer is built on.

    Parameters
    ----------
    center : array_like, shape (dim,)
        Ball centre.  Coerced to a ``float`` array.
    r : float
        Radius; must be strictly positive.

    Raises
    ------
    ValueError
        If ``r <= 0``.

    Examples
    --------
    >>> ball = Ball([0.0, 0.0], r=1.0)
    >>> ball.contains([0.5, 0.5])
    True
    >>> ball.contains([[0.0, 0.0], [2.0, 0.0]])    # batch query → per-row mask
    array([ True, False])
    """

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

    def contains(self, u: Any) -> Any:
        """Whether point(s) ``u`` lie in the ball.

        Parameters
        ----------
        u : array_like
            A single point of shape ``(dim,)`` **or** a batch of shape
            ``(n, dim)``.  Any other shape raises.

        Returns
        -------
        bool or ndarray of bool
            A scalar ``bool`` for a single point, or an ``(n,)`` boolean mask.

        Raises
        ------
        ValueError
            If ``u`` is neither a ``(dim,)`` point nor an ``(n, dim)`` batch.
        """
        pts, scalar = _coerce_query(u, self.dim)
        mask = np.linalg.norm(pts - self.center, axis=1) <= self.r
        return bool(mask[0]) if scalar else mask


@dataclass(frozen=True)
class Grid:
    """Regular grid: ``counts[i]`` points spanning ``[lo_i, hi_i]`` per axis.

    A regular Cartesian lattice over an axis-aligned box: axis ``i`` carries
    ``counts[i]`` evenly spaced nodes spanning ``[lo_i, hi_i]`` (inclusive of
    both endpoints).  Use :func:`grid_points` to enumerate the lattice nodes and
    :func:`region` for a terse ``(lo, hi, n)``-per-axis constructor.

    Parameters
    ----------
    lo, hi : array_like, shape (dim,)
        Per-axis lower and upper bounds of the bounding box.
    counts : tuple of int
        Number of nodes per axis; each must be ``>= 1``.

    Raises
    ------
    ValueError
        If ``lo``, ``hi``, and ``counts`` disagree in length, or any count
        is ``< 1``.

    Examples
    --------
    >>> g = Grid([-1.0, -1.0], [1.0, 1.0], (3, 3))
    >>> g.shape
    (3, 3)
    >>> g.contains([0.0, 0.0])
    True
    """

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
        """Per-axis coordinate vectors (``dim`` arrays, ``counts[i]`` long)."""
        return [np.linspace(self.lo[i], self.hi[i], self.counts[i]) for i in range(self.dim)]

    def contains(self, u: Any) -> Any:
        """Whether point(s) ``u`` lie in the grid's bounding box.

        Membership is against the bounding box ``[lo_i, hi_i]`` — *not* exact
        coincidence with a lattice node.

        Parameters
        ----------
        u : array_like
            A single point of shape ``(dim,)`` **or** a batch of shape
            ``(n, dim)``.  Any other shape raises.

        Returns
        -------
        bool or ndarray of bool
            A scalar ``bool`` for a single point, or an ``(n,)`` boolean mask.

        Raises
        ------
        ValueError
            If ``u`` is neither a ``(dim,)`` point nor an ``(n, dim)`` batch.
        """
        pts, scalar = _coerce_query(u, self.dim)
        mask = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
        return bool(mask[0]) if scalar else mask


Region = Box | Ball | Grid


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def sampler(region: Any, *, seed: int | None = None) -> Callable[[], np.ndarray]:
    """
    Return a reproducible 0-argument sampler drawing points from ``region``.

    Box → uniform in the box; Ball → uniform in the ball (radial CDF, not the
    biased "uniform radius" trick); Grid → uniform over its bounding box (use
    :func:`grid_points` for the lattice itself).

    Parameters
    ----------
    region : Box, Ball, Grid, or sequence of (lo, hi) bounds
        Plain per-axis bounds are read through :func:`as_region`, so no library
        type has to be constructed to call this.
    seed : int, optional
        Seeds a private ``numpy.random.Generator`` so draws are reproducible
        and independent of global RNG state.

    Returns
    -------
    callable
        ``draw() -> ndarray`` of shape ``(region.dim,)``.

    Examples
    --------
    >>> draw = sampler([(-1, 1), (-1, 1)], seed=0)     # plain bounds
    >>> draw().shape
    (2,)
    >>> draw = sampler(Box([-1, -1], [1, 1]), seed=0)  # or the primitive
    >>> draw().shape
    (2,)
    """
    region = as_region(region, analysis="sampler")
    rng = np.random.default_rng(seed)

    if isinstance(region, Box):
        lo, hi = region.lo, region.hi

        def draw_box() -> np.ndarray:
            # np.asarray: the numpy stubs type the scalar overload of
            # Generator.uniform as float, so the array-bounds call needs the
            # explicit coercion to stay ndarray-typed.
            return np.asarray(rng.uniform(lo, hi))

        return draw_box

    if isinstance(region, Ball):
        c, r, d = region.center, region.r, region.dim

        def draw_ball() -> np.ndarray:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            radius = r * rng.uniform() ** (1.0 / d)  # uniform-in-volume
            return np.asarray(c + radius * v)

        return draw_ball

    if isinstance(region, Grid):
        lo, hi = region.lo, region.hi

        def draw_grid() -> np.ndarray:
            # np.asarray: see draw_box above (numpy stubs' scalar overload).
            return np.asarray(rng.uniform(lo, hi))

        return draw_grid

    raise TypeError(f"unknown region type {type(region).__name__}")


def grid_points(grid: Any, *, resolution: int = DEFAULT_REGION_RESOLUTION) -> np.ndarray:
    """
    Enumerate every lattice point of ``grid`` (row-major / C order).

    Parameters
    ----------
    grid : Grid, Box, Ball, or sequence of (lo, hi[, n]) bounds
        Plain per-axis bounds are read through :func:`as_region`, so no library
        type has to be constructed to call this.  Bounds given without a node
        count are filled in at ``resolution`` nodes per axis.
    resolution : int, default 100
        Nodes per axis used for bare ``(lo, hi)`` bounds (ignored for a
        :class:`Grid`, which carries its own ``counts``).

    Returns
    -------
    ndarray, shape ``(prod(counts), dim)``
        One row per grid node; reshape to ``grid.shape + (dim,)`` for a basin
        map laid out over the grid.

    Examples
    --------
    >>> grid_points([(-1.0, 1.0, 3), (-1.0, 1.0, 3)]).shape   # plain bounds
    (9, 2)
    >>> grid_points(Grid([-1.0], [1.0], (5,))).shape          # or the primitive
    (5, 1)
    """
    resolved = as_region(grid, want_grid=True, resolution=resolution, analysis="grid_points")
    if not isinstance(resolved, Grid):
        lo = resolved.lo if isinstance(resolved, Box) else resolved.center - resolved.r
        hi = resolved.hi if isinstance(resolved, Box) else resolved.center + resolved.r
        resolved = Grid(lo, hi, (int(resolution),) * int(lo.size))
    grid = resolved
    axes = grid.axes()
    if grid.dim == 1:
        return np.asarray(axes[0], dtype=float)[:, None]
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
# The one region reading
# ---------------------------------------------------------------------------


def _region_example(dim: int, *, triples: bool) -> str:
    """Return a region literal sized to ``dim`` state components, as source text.

    The example an error message shows must be the *caller's* region, not a
    stock 2-D one: it names as many axes as the system has state components, and
    spans a box wide enough to be worth trying.
    """
    axis = f"(-2.0, 2.0, {DEFAULT_REGION_RESOLUTION})" if triples else "(-2.0, 2.0)"
    if dim <= 3:
        return "[" + ", ".join([axis] * dim) + "]"
    return f"[{axis}] * {dim}"


#: Public home of each ``region=`` door, so a remedy line RESOLVES.  ``ts.basins``
#: stopped being a name in v6, and a message handing back a line that raises is
#: worse than one handing back nothing (§5.6 [M37]).
_REGION_DOOR_HOME: dict[str, str] = {
    "sampler": "ts.data.sampler",
    "grid_points": "ts.data.grid_points",
}


def _qualified_door(analysis: str | None) -> str:
    """Return the resolvable spelling of the door that is asking for a region."""
    if not analysis:
        return "ts.analysis.<name>"
    return _REGION_DOOR_HOME.get(analysis, f"ts.analysis.{analysis}")


def as_region(
    spec: Any,
    *,
    dim: int | None = None,
    want_grid: bool = False,
    resolution: int = DEFAULT_REGION_RESOLUTION,
    analysis: str | None = None,
    system: Any = None,
    args: str = "",
) -> Region:
    """Coerce a ``region=`` argument to a region primitive.

    **The** one region reading in the library: a region is one ``(lo, hi)``
    bound — or one ``(lo, hi, n)`` triple — **per state component**.  That is
    the grammar :func:`plt.xlim <matplotlib.pyplot.xlim>` and
    :func:`numpy.histogramdd`'s ``range=`` already taught the caller, and it is
    the ONE reading at every dimension.  **A pre-v6 corner pair is not a
    second grammar and is not detected** at ``dim == 2``: an asymmetric one such
    as ``([-3, -1], [2, 6])`` also parses per-axis, into a *different* box, and
    no signal distinguishes it from the perfectly ordinary
    ``[(-3, -1), (2, 6)]``.  Refusing it would refuse that ordinary call, so
    the per-axis reading simply wins — which is why there is exactly one
    grammar, stated here and in every ``region=`` docstring.

    A :class:`Box` / :class:`Ball` / :class:`Grid` is passed straight through —
    plain bounds are an *addition*, never a replacement.  Nothing else is
    accepted.

    Parameters
    ----------
    spec : Region or sequence of (lo, hi) or sequence of (lo, hi, n), or None
        The caller's ``region=`` argument.
    dim : int, optional
        The system's state dimension, used to size the suggested literal in an
        error message.  Read from ``system`` when omitted.
    want_grid : bool, default False
        ``True`` for a routine that scans a *lattice* — it needs counts, so
        bare bounds are filled in at ``resolution`` nodes per axis.  ``False``
        for a routine that *samples* the region, where bare bounds become a
        :class:`Box` and no resolution is invented.
    resolution : int, default 100
        Nodes per axis used to fill in bare bounds when ``want_grid``.
    analysis, system, args : optional
        Diagnostic context: the calling function's name, the system being
        analysed (sizes the suggested literal), and the source text of any
        positional arguments that sit *between* the system and the region in
        that signature (``continuation`` takes ``param, values`` first).
        Without ``args`` the suggested line would have the wrong arity, and a
        remedy that does not run is worse than none.

    Returns
    -------
    Box, Ball, or Grid

    Raises
    ------
    InvalidInputError
        If ``spec`` is ``None`` or is not a region that can be read per-axis.

    Examples
    --------
    >>> as_region([(-3.0, 3.0), (-3.0, 3.0)]).lo        # one bound per axis
    array([-3., -3.])
    >>> as_region([(-2.0, 2.0, 50)] * 2).shape          # one triple per axis
    (50, 50)
    >>> as_region(Box([-1.0, -1.0], [1.0, 1.0])) is not None   # a primitive passes through
    True
    """
    from ..errors import InvalidInputError, remedy

    if isinstance(spec, (Box, Ball, Grid)):
        return spec

    ndim = int(dim if dim is not None else (getattr(system, "dim", 2) or 2))
    who = f"{analysis}()" if analysis else "this analysis"
    call = _qualified_door(analysis)

    def _refuse(detail: str, *, slice_hint: bool = False) -> InvalidInputError:
        lines = [f"{call}(system, {args}{_region_example(ndim, triples=want_grid)})"]
        if slice_hint and want_grid and ndim > 2:
            # Imaging a slice of a higher-dimensional flow is the whole point of
            # the ``recurrence=`` box, and it is the recipe three of four users
            # reach for next.  Put it in the message, not only in the docstring.
            pinned = ", ".join(["(-2.0, 2.0, 60)"] * 2 + ["(0.0, 0.0, 1)"] * (ndim - 2))
            free = ", ".join(["(-3.0, 3.0)"] * ndim)
            lines += [
                f"{call}(system, [{pinned}],",
                f"{' ' * len(call)}  recurrence=[{free}])   # image a SLICE",
            ]
        return InvalidInputError(
            f"{who} needs a region: the box of state space to search — {detail}."
            + remedy(
                *lines,
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
    if width not in (2, 3):
        raise _refuse(
            f"each axis needs (lo, hi) or (lo, hi, n), got {width} numbers. "
            "A region is read one axis at a time, so a pair of corner points is "
            "spelled Box(lo_corner, hi_corner)"
        )

    declared = dim if dim is not None else getattr(system, "dim", None)
    if declared is not None and len(rows) != int(declared):
        # Caught HERE, in the caller's own vocabulary — the FFI used to answer
        # this with "seeds buffer length 800 is not a multiple of dim = 3", and
        # only when 2*n**2 happened not to divide by 3.
        raise _refuse(
            f"a region is read one axis at a time, so it needs one bound per "
            f"state component: {int(declared)} for this system, and "
            f"{len(rows)} were given",
            slice_hint=True,
        )

    lo = np.array([row[0] for row in rows], dtype=float)
    hi = np.array([row[1] for row in rows], dtype=float)
    # A corner *pair* — ``([-3, -3], [3, 3])`` — parses as per-axis rows too, but
    # every axis then spans nothing (or runs backwards).  That is never a region
    # anyone meant, and it is exactly the misreading that used to search a
    # zero-volume box in silence, so name it rather than build it.
    if np.any(hi < lo) or (len(rows) > 1 and np.all(hi == lo)):
        raise _refuse(
            "every axis came out empty or backwards, which is what a pair of "
            "corner points looks like when it is read one axis at a time"
        )

    if width == 3:
        return region([(a, b, int(n)) for a, b, n in rows])
    if not want_grid:
        return Box(lo, hi)
    return Grid(lo, hi, (int(resolution),) * len(rows))


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
