r"""
Attractor finding via recurrences.

The estimator tessellates state space into cells and follows a trajectory cell
by cell with a small finite-state machine: while it keeps landing in *new* cells
it is transient; once it recurrently re-visits cells it has located an
**attractor** (the recurrent cell set), and every transient cell that led there
is labelled as that attractor's **basin**.  Later initial conditions that wander
into an already-labelled cell inherit its attractor cheaply, so the cost falls as
state space fills in.

This is the recurrence approach of

    G. Datseris and A. Wagemakers, "Effortless estimation of basins of
    attraction", *Chaos* **32**, 023104 (2022).

with the cell-visitation idea going back to H. E. Nusse and J. A. Yorke,
*Dynamics: Numerical Explorations* (Springer, 1997).

:func:`attractors` runs the machine from a cloud of seeds and returns the
attractors it discovers; :class:`_AttractorMapper` is the reusable engine the
basin and continuation layers drive over a full grid.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from ...data import Ball, Box, Grid, sampler, set_distance
from ...errors import ConvergenceError
from ...utils.tolerances import BASIN_ATOL, BASIN_RTOL
from .._common import reject_data
from .._result import AnalysisResult, _ArrayBacked
from .._result_base import _MAX_ITEMS
from .._result_json import _pct, _state
from ._common import (
    DIVERGED_COLOR,
    PALETTE,
    _CellGrid,
    _palette_indices,
    _recurrence_grid,
    _representative,
    coerce_region,
    reject_unknown_fsm,
)

if TYPE_CHECKING:
    from ...data.sampling import _SetMethod

__all__ = [
    "Attractor",
    "AttractorSet",
    "attractors",
]


def _is_discrete(system: Any) -> bool:
    """Return whether ``system`` advances by iterations rather than by time.

    v6 replaced the public ``is_discrete`` flag with ``family`` (§3.4).  Reading
    the old name through ``getattr(system, "is_discrete", False)`` is the
    silent-wrong-answer shape §9.4 rule 3 names: ``False`` is a *legal* value, so
    a map would have been marched by ``dt`` instead of by one iterate with
    nothing raised.  The private ``_is_discrete`` is the fallback because it is
    the flag the five derived wrappers carry (a ``PoincareMap`` reports
    ``family == "ode"`` while being a discrete view of one).
    """
    return bool(getattr(system, "family", None) == "map" or getattr(system, "_is_discrete", False))


#: Label returned for an initial condition that leaves the region / never settles.
DIVERGED = -1

#: Relative step size below which the invariance march calls a state *stationary*
#: and stops early: the orbit has converged onto a fixed point, which is
#: invariant and can reach nothing further.  This is a convergence test on the
#: state, **not** a solver tolerance — the march's integration accuracy is
#: :data:`~tsdynamics.utils.tolerances.BASIN_RTOL` /
#: :data:`~tsdynamics.utils.tolerances.BASIN_ATOL`.
_STATIONARY_REL = 1e-12


def _is_stationary(state: np.ndarray, prev: np.ndarray) -> bool:
    """Whether ``state`` is unchanged from ``prev`` to :data:`_STATIONARY_REL`."""
    scale = max(float(np.max(np.abs(state))), 1.0)
    return bool(float(np.max(np.abs(state - prev))) <= _STATIONARY_REL * scale)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attractor(_ArrayBacked, AnalysisResult):
    """
    One located attractor: a point cloud of states sampled on it.

    **It IS its point cloud** (v6): ``np.asarray(attractor)`` is the
    ``(m, dim)`` array of sampled states, ``attractor[0]`` is the first of them,
    and ``len(attractor)`` is how many were sampled — so ``attractors(sys, …)[0]``
    hands back numbers and the class stays invisible, showing itself only in the
    repr.  The label and the coarse size are one dot away: :attr:`id`,
    :attr:`center`, :attr:`cells`.

    Attributes
    ----------
    id : int
        Integer label (``>= 1``) identifying this attractor within its set.
    points : ndarray, shape (m, dim)
        States sampled while the trajectory was on the attractor.  A fixed point
        collapses to one repeated point; a cycle/chaotic set spreads out.
        ``np.asarray(attractor)`` returns it.
    cells : int
        Number of distinct grid cells the attractor occupies (a coarse size).
    """

    #: The numbers this result *is* — see :class:`_ArrayBacked`.
    _array_field: ClassVar[str] = "points"

    id: int = 0
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False, compare=False)
    cells: int = 0

    @property
    def center(self) -> np.ndarray:
        """The centroid of the point cloud (an attractor representative)."""
        return _representative(self.points)

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return int(self.points.shape[1])

    def _answer(self) -> str:
        """Return the attractor's id and where in state space it sits."""
        if not np.asarray(self.points).size:
            return f"#{self.id} — no sampled points"
        return f"#{self.id} at {_state(self.center)}"

    def _context(self) -> str | None:
        """Return how much of the tessellation the attractor occupies."""
        bits = [b for b in (self._system_label(),) if b]
        bits.append(f"{self.cells} cells, {int(np.shape(self.points)[0])} sampled points")
        return ", ".join(bits)

    def _as_item(self) -> str:
        """Return the compact one-line form used inside an :class:`AttractorSet`."""
        where = (
            "no sampled points" if not np.asarray(self.points).size else f"at {_state(self.center)}"
        )
        return f"#{self.id}  {where}  {self.cells} cells"

    def _derived(self) -> dict[str, Any]:
        """Export the representative the repr reports — the answer, not a field.

        ``points`` is an ``(m, dim)`` cloud, so a table row cannot carry it; the
        centroid is what ``to_frame()`` needs for the row to say *where* the
        attractor is rather than only how big it is.
        """
        if not np.asarray(self.points).size:
            return {}
        return {"center": self.center, "dim": self.dim}


@dataclass(frozen=True)
class AttractorSet(AnalysisResult):
    """
    The attractors found in a region, indexing to **numbers**.

    ``aset[0]`` is the first attractor's ``(dim,)`` centre as a plain
    :class:`numpy.ndarray`, equal to ``np.asarray(aset)[0]``; iteration yields
    those centres.  The :class:`Attractor` records are :attr:`details` (by
    position) and :meth:`by_id` (by label), and :attr:`centers` / :attr:`cells` /
    :attr:`ids` read the same data column-wise, so nothing here needs a loop::

        aset = ts.analysis.attractors(system, region)
        aset[0]            # array([-1.,  0.])   the centre
        aset.centers       # (n, dim)
        aset.cells         # (n,) int
        aset.details[0]    # Attractor  #1  centre [-1, 0] · 3 cells
        aset.by_id(2)      # the attractor LABELLED 2, as a record

    Attributes
    ----------
    attractors : dict[int, Attractor]
        Located attractors, id → :class:`Attractor`.
    diverged : int
        How many seeds left the region / never settled.
    seeds : int
        How many seeds were classified in total.
    """

    attractors: dict[int, Attractor] = field(default_factory=dict, compare=False)
    diverged: int = 0
    seeds: int = 0

    def __len__(self) -> int:  # noqa: D105
        return len(self.attractors)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate the attractors' **centres** — numbers, not wrappers."""
        return iter(self.centers)

    def __getitem__(self, key: Any) -> Any:
        """Return the **centre** of the attractor at position ``key``, shape ``(dim,)``.

        Positional, like every Python sequence — ids start at 1, so ``aset[0]``
        once raised ``KeyError`` while ``aset[1]`` returned the *first*
        attractor, which reads as positional and is not.

        Numbers, not a wrapper (contract §4.2 rule 6): ``aset[0]`` used to hand
        back an :class:`Attractor`, so a caller met a second class on the way to
        a coordinate and ``np.asarray(aset)[0]`` was a *different* thing from
        ``np.asarray(aset[0])``.  Both are now the same ``(dim,)`` centre.  The
        records are :attr:`details`; :meth:`by_id` looks one up by its label.
        """
        return self.centers[key]

    @property
    def details(self) -> tuple[Attractor, ...]:
        """The :class:`Attractor` records, in the order ``[]`` indexes.

        ``aset[0]`` is the centre; ``aset.details[0]`` is the attractor it came
        from, with its repr, its ``points`` cloud and its ``cells`` count.  One
        name across every collection in the library, so it is learned once.
        """
        return tuple(self.attractors[k] for k in self.ids)

    def by_id(self, key: int) -> Attractor:
        """Return the **record** of the attractor labelled ``key``.

        ``[]`` indexes by position and hands back numbers; an id is something you
        read off a basin image and then want to know *about*, so this door hands
        back the :class:`Attractor` itself — the same object :attr:`details`
        holds.

        Raises
        ------
        KeyError
            If no attractor carries that id.
        """
        return self.attractors[int(key)]

    @property
    def ids(self) -> list[int]:
        """Sorted attractor ids."""
        return sorted(self.attractors)

    @property
    def centers(self) -> np.ndarray:
        """Stack of attractor representatives, shape ``(n_attractors, dim)``.

        The same array ``np.asarray(aset)`` returns: a SET arrays as one
        representative per member, while a MEMBER arrays as its own point cloud.
        """
        return np.array([self.attractors[k].center for k in self.ids])

    @property
    def cells(self) -> np.ndarray:
        """How many grid cells each attractor occupies, shape ``(n_attractors,)``."""
        return np.array([self.attractors[k].cells for k in self.ids], dtype=int)

    def match(self, point: Any, *, method: str = "centroid") -> int | None:
        """
        Return the id of the attractor closest to ``point`` (or ``None`` if empty).

        Uses :func:`tsdynamics.data.set_distance`; ``point`` may be a single state
        or a point cloud.
        """
        if not self.attractors:
            return None
        pts = np.atleast_2d(np.asarray(point, dtype=float))  # a single state -> (1, dim)
        dists = {
            k: set_distance(self.attractors[k].points, pts, method=cast("_SetMethod", method))
            for k in self.ids
        }
        return min(dists, key=dists.__getitem__)

    def __plot_spec__(self, kind: str | None = None) -> Any:
        r"""Describe the located attractors as a backend-agnostic :class:`PlotSpec`.

        Builds a ``PHASE_PORTRAIT_2D`` of every attractor's point cloud as one
        ``SCATTER`` layer (the first two state coordinates).  Each point carries
        a ``"cat"`` channel — the attractor id's swatch index in the shared
        categorical palette (``tab20``) — so the same id is drawn the same colour
        here and on the basin image (:meth:`BasinsResult.__plot_spec__`).  The
        palette name and the fixed diverged colour are recorded in ``meta`` so a
        renderer can reproduce the mapping; the colorbar is marked
        :attr:`~tsdynamics.viz.spec.Colorbar.discrete`.

        Each id's representative also seeds a category label, so the colour key
        reads as ``attractor 1``, ``attractor 2``, ….  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls
        a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"phase_portrait_2d"``).  ``None``
            uses ``PHASE_PORTRAIT_2D``.

        Returns
        -------
        PlotSpec

        Raises
        ------
        VisualizationNotInstalled
            If the set holds no attractor with a ≥ 2-D point cloud to scatter.
        """
        from tsdynamics.analysis._result import VisualizationNotInstalled
        from tsdynamics.viz.spec import Colorbar

        from .. import _plotbuilder as pb

        ids = self.ids
        swatch = _palette_indices(ids)

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        cats: list[np.ndarray] = []
        for aid in ids:
            pts = np.atleast_2d(np.asarray(self.attractors[aid].points, dtype=float))
            if pts.shape[1] < 2 or pts.shape[0] == 0:
                continue
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
            cats.append(np.full(pts.shape[0], swatch[aid], dtype=int))

        if not xs:
            raise VisualizationNotInstalled(
                "AttractorSet holds no attractor with a 2-D point cloud to scatter; "
                "export it with .to_dict() instead."
            )

        layer = pb.scatter(
            np.concatenate(xs),
            np.concatenate(ys),
            cat=np.concatenate(cats),
            label="attractors",
            style={"cmap": PALETTE},
        )
        meta = dict(self.meta) if self.meta else {}
        meta.update(
            palette=PALETTE,
            diverged_color=DIVERGED_COLOR,
            palette_index=swatch,
            palette_labels=[f"attractor {aid}" for aid in ids],
        )
        return pb.spec(
            kind,
            "phase_portrait_2d",
            layers=[layer],
            aspect="equal",
            xlabel="x1",
            ylabel="x2",
            title=f"attractors ({len(self)})",
            colorbar=Colorbar(label="attractor", cmap=PALETTE, discrete=True),
            meta=meta,
        )

    def _answer(self) -> str:
        """Return how many attractors were located, and the escape share."""
        n = len(self)
        found = f"{n} attractor" + ("s" if n != 1 else "") if n else "no attractor found"
        share = _pct(self.diverged / self.seeds) if self.seeds else "0.0%"
        return f"{found} · {share} of {self.seeds} seeds diverged"

    def _item_lines(self) -> tuple[str, ...]:
        """Return one line per located attractor, truncated like a collection.

        Labelled ``details[i]`` — the accessor that hands back the record the
        line describes.  ``aset[i]`` is the representative *point* (contract
        §4.2 rule 6), so a bare ``[i]`` invited ``aset[0].center`` and an
        ``AttributeError`` on an ``ndarray``.
        """
        ordered = [self.attractors[k] for k in self.ids]
        shown = [f"details[{i}] {a._as_item()}" for i, a in enumerate(ordered[:_MAX_ITEMS])]
        if len(ordered) > _MAX_ITEMS:
            shown.append(f"... [{len(ordered)} total]")
        return tuple(shown)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the attractor representatives as an ``(n_attractors, dim)`` array."""
        arr = np.asarray(self.centers, dtype=float)
        return arr.astype(dtype, copy=bool(copy)) if dtype is not None else arr

    def _full_extras(self) -> dict[str, Any]:
        """Export each attractor's LOCATION, keyed by id.

        The repr says *where* every attractor is and so does each member's own
        ``to_frame()``; the set's ``to_dict(full=True)`` serialised its children
        without their derived answers, so the one property that makes two runs
        comparable — the position — was the one the exported table dropped.
        """
        return {"centers": {int(k): self.attractors[k].center for k in self.ids}}


# ---------------------------------------------------------------------------
# The recurrence finite-state machine
# ---------------------------------------------------------------------------


class _AttractorMapper:
    r"""
    Map an initial condition to the attractor it converges to, via recurrences.

    Drive any :class:`~tsdynamics.families.System` (map or flow) over a shared
    :class:`~tsdynamics.analysis.basins._common._CellGrid`.  :meth:`map_ic` returns
    a positive integer attractor id (discovering a new one if needed) or
    :data:`DIVERGED` (``-1``).  Labels persist across calls, so a sweep over many
    initial conditions amortises: most settle by hitting an already-labelled cell.

    Parameters
    ----------
    system : System
        A discrete map or a continuous flow (DDE/SDE are not supported).
    cellgrid : _CellGrid
        The state-space tessellation the recurrences are detected on.
    dt : float, default 1.0
        Integration step between cell checks for a *flow* (ignored for a map,
        which advances one iteration per check).
    max_steps : int, default 10000
        Hard cap on steps per initial condition before giving up (``DIVERGED``).
    consecutive_recurrences : int, default 30
        New attractor declared after this many consecutive steps into
        already-visited (this-trajectory) cells.
    attractor_locate_steps : int, default 30
        Extra steps integrated once an attractor is declared, to flesh out its
        cell set.
    attractor_revisits : int, default 2
        Steps in a known attractor's cells before an initial condition is assigned
        to it.
    basin_revisits : int, default 10
        Steps in a known basin's cells before inheriting that basin (the
        sparse-labelling shortcut).
    lost_steps : int, default 20
        Consecutive steps outside the region before declaring divergence.
    """

    def __init__(
        self,
        system: Any,
        cellgrid: _CellGrid,
        *,
        dt: float = 1.0,
        max_steps: int = 10000,
        consecutive_recurrences: int = 30,
        attractor_locate_steps: int = 30,
        attractor_revisits: int = 2,
        basin_revisits: int = 10,
        lost_steps: int = 20,
    ) -> None:
        self.system = system
        self.grid = cellgrid
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.mx_fnd = int(consecutive_recurrences)
        self.mx_loc = int(attractor_locate_steps)
        self.mx_att = int(attractor_revisits)
        self.mx_bas = int(basin_revisits)
        self.mx_lost = int(lost_steps)

        self._discrete = _is_discrete(system)
        self._step_arg: int | float = 1 if self._discrete else self.dt

        # Persistent labels (sparse): cell key → attractor id.  These dicts are
        # read AND written by every ``map_ic`` call and carry the order-dependent
        # state that makes the sweep amortise (a later seed inherits an earlier
        # seed's label).  Mutating them from threads would both race and change the
        # result, so ``map_ic`` is driven strictly serially — see the measured
        # rationale in ``attractors``'s Notes for why thread-parallelism is a
        # net loss here, not just unsafe.
        self._att_cells: dict[tuple[int, ...], int] = {}
        self._bas_cells: dict[tuple[int, ...], int] = {}
        self._att_points: dict[int, list[np.ndarray]] = {}
        self._next_id = 1

    # -- driving the system through the protocol --

    def _reinit(self, ic: np.ndarray) -> None:
        """Reinitialise the driven system at initial condition ``ic``.

        A **flow** is re-seeded with the basin march's own tolerances
        (:data:`~tsdynamics.utils.tolerances.BASIN_RTOL` /
        :data:`~tsdynamics.utils.tolerances.BASIN_ATOL`) rather than
        ``ContinuousSystem.reinit``'s library default.  Two reasons, and both are
        load-bearing:

        * This Python FSM is the contractual **bit-identical oracle** for the
          Rust march (:func:`_try_rust_march`, guarded by
          ``tests/test_basin_kernel.py``).  The kernel is handed the tolerance
          explicitly, so the two must name the *same* constant or the equivalence
          silently breaks the moment either default moves.
        * The march is a *topological* classification — which cell the orbit
          settles in, at a cell size many orders above ``1e-6`` — driven over
          thousands of two-node ``[t, t+dt]`` integrations per image.  Measured,
          the v6 ODE default (``1e-9``/``1e-12``) costs 2.27x on a smooth
          two-well Duffing basin and 3.01x on a fractal magnetic-pendulum slice
          while changing **0.00 %** of the labels in either.

        A map has no solver tolerances (``DiscreteMap.reinit`` takes none), so it
        is re-seeded plainly.
        """
        ic_arr = np.asarray(ic, dtype=float)
        if self._discrete:
            self.system.reinit(ic_arr)
        else:
            self.system.reinit(ic_arr, rtol=BASIN_RTOL, atol=BASIN_ATOL)

    def _advance(self) -> np.ndarray | None:
        """Advance one step; ``None`` if the trajectory blew up (raised / non-finite).

        Only genuine *divergence* (a :class:`~tsdynamics.errors.ConvergenceError`,
        the maps' / flows' loud-divergence contract) and arithmetic overflow are
        treated as "gone".  An engine-unavailability failure
        (:class:`~tsdynamics.errors.BackendError` /
        :class:`~tsdynamics.engine.run.EngineNotAvailableError`) is *also* a
        ``RuntimeError`` but is **not** a divergence — it propagates rather than
        silently painting an all-diverged basin.
        """
        try:
            state = np.asarray(self.system.step(self._step_arg), dtype=float).reshape(-1)
        except (ConvergenceError, ArithmeticError):
            # maps/flows raise ConvergenceError on divergence; arithmetic overflow
            # (ArithmeticError covers FloatingPointError and OverflowError) is the
            # same "gone for good".  Engine-unavailability errors are not caught.
            return None
        return state if np.all(np.isfinite(state)) else None

    # -- the FSM --

    def map_ic(self, ic: Any) -> int:
        """Classify one initial condition; return its attractor id or ``DIVERGED``."""
        self._reinit(ic)
        visited: dict[tuple[int, ...], int] = {}
        trail: list[tuple[int, ...]] = []
        c = att_hit = bas_hit = lost = 0

        for it in range(self.max_steps):
            state = self._advance()
            if state is None:
                # the step blew up — the trajectory is gone for good.
                return DIVERGED
            cell = self.grid.index(state)

            if cell is None:
                # finite but outside the region: a transient excursion may return.
                lost += 1
                c = att_hit = bas_hit = 0
                if lost >= self.mx_lost:
                    return DIVERGED
                continue
            lost = 0

            known = self._att_cells.get(cell)
            if known is not None:
                att_hit += 1
                c = bas_hit = 0
                if att_hit >= self.mx_att:
                    self._label_basin(trail, known)
                    return known
                continue

            known = self._bas_cells.get(cell)
            if known is not None:
                bas_hit += 1
                c = att_hit = 0
                if bas_hit >= self.mx_bas:
                    self._label_basin(trail, known)
                    return known
                continue

            # an unlabelled cell
            att_hit = bas_hit = 0
            if cell in visited:
                c += 1
                if c >= self.mx_fnd:
                    return self._locate_attractor(trail)
            else:
                visited[cell] = it
                c = 0
            trail.append(cell)

        return DIVERGED

    def _locate_attractor(self, trail: list[tuple[int, ...]]) -> int:
        """Recurrence detected: integrate on to map the attractor, then label."""
        new_id = self._next_id
        self._next_id += 1
        att_cells: set[tuple[int, ...]] = set()
        points: list[np.ndarray] = []

        for _ in range(self.mx_loc):
            state = self._advance()
            if state is None:
                break
            cell = self.grid.index(state)
            if cell is None:
                break
            att_cells.add(cell)
            points.append(state.copy())

        if not att_cells:
            # Could not pin the attractor down (left the region while locating).
            self._next_id -= 1
            return DIVERGED

        for cell in att_cells:
            self._att_cells[cell] = new_id
            self._bas_cells.pop(cell, None)
        self._att_points[new_id] = points

        # transient cells that led here become basin cells (not the attractor's).
        for cell in trail:
            if cell not in self._att_cells:
                self._bas_cells.setdefault(cell, new_id)
        return new_id

    def _label_basin(self, trail: list[tuple[int, ...]], att_id: int) -> None:
        """Mark an initial condition's transient cells as ``att_id``'s basin."""
        for cell in trail:
            if cell not in self._att_cells:
                self._bas_cells.setdefault(cell, att_id)

    # -- proximity dedup --

    def merge_map(self, tol: float) -> dict[int, int]:
        """
        Group attractor ids whose point clouds are near-coincident *as sets*.

        The recurrence machine occasionally splits one attractor into two cell
        sets (e.g. a chaotic set approached from two sides).  A small ``tol``
        unions only near-coincident ids, leaving genuinely distinct attractors
        apart.  Two ids are merged when their clouds are within ``tol`` **both**
        by centroid and by closest approach (``set_distance(..., "minimum")``):
        a centroid alone cannot separate *concentric* attractors, which share
        one exactly — see the comment at the test.  Returns
        ``{old_id: canonical_id}``.
        """
        ids = sorted(self._att_points)
        parent = {k: k for k in ids}

        def find(a: int) -> int:
            """Union-find root of ``a`` with path compression."""
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        clouds = {k: np.asarray(self._att_points[k], dtype=float) for k in ids}
        centers = {k: _representative(clouds[k]) for k in ids}
        for i, a in enumerate(ids):
            for c in ids[i + 1 :]:
                if find(a) == find(c):
                    continue
                if float(np.linalg.norm(centers[a] - centers[c])) > tol:
                    continue
                # A coincident centroid does NOT mean a coincident set: two
                # *concentric* attractors share a centroid exactly.  Two nested
                # limit cycles (r = 1 and r = 3 of
                # ``r' = -r(r-1)(r-2)(r-3)``, ``theta' = 1``) both centre on the
                # origin, so the centroid test alone merged them into one
                # "attractor" whose point cloud straddles both rings — 1 reported
                # where the truth is 2.  Require the sets to actually touch as
                # well; for the point clouds of fixed-point attractors the two
                # distances coincide, so nothing that merged before stops.
                if set_distance(clouds[a], clouds[c], method="minimum") <= tol:
                    parent[find(c)] = find(a)
        return {k: find(k) for k in ids}

    # -- harvest --

    def attractor_set(
        self, diverged: int, seeds: int, *, merge: dict[int, int] | None = None
    ) -> AttractorSet:
        """Bundle the discovered attractors into an :class:`AttractorSet`."""
        merge = merge or {}
        cell_counts: dict[int, int] = {}
        for att_id in self._att_cells.values():
            cid = merge.get(att_id, att_id)
            cell_counts[cid] = cell_counts.get(cid, 0) + 1

        pooled: dict[int, list[np.ndarray]] = {}
        # Pool in ascending-id order so a merged attractor's centroid mean is a
        # deterministic, order-independent sum.  The Rust kernel harvests its
        # ``_att_points`` from a hash map (randomised iteration order), so without
        # this sort a merged ≥2-cloud pool would wobble by ~1 ULP run-to-run and
        # diverge from the Python path; the pure-Python loop is already id-ordered,
        # so this is a no-op there.  (``merge_map`` likewise sorts its ids.)
        for k, pts in sorted(self._att_points.items(), key=lambda kv: kv[0]):
            pooled.setdefault(merge.get(k, k), []).extend(pts)

        attractors = {
            cid: Attractor(
                id=cid,
                points=np.atleast_2d(np.asarray(pts, dtype=float)),
                cells=cell_counts.get(cid, 0),
            )
            for cid, pts in pooled.items()
        }
        return AttractorSet(attractors=attractors, diverged=diverged, seeds=seeds)

    # -- post-march invariance audit (see ``audit_attractors``) --

    def _verify(
        self,
        own: set[tuple[int, ...]],
        others: dict[tuple[int, ...], int],
        start: np.ndarray,
        budget: int,
    ) -> tuple[bool, set[int], str | None]:
        r"""March and report ``(is_invariant, other ids reached, failure cause)``.

        The recurrence FSM declares an attractor from *consecutive steps into
        already-visited cells*, which a merely **slow** orbit satisfies without
        recurring at all: a bottleneck crawl sits in one cell for more than
        ``consecutive_recurrences`` steps simply because ``dt`` moves it less than
        a cell width.  This march is the invariance test that tells the two apart
        — it asks the only question that distinguishes them, namely *does the
        orbit come back*:

        * it **left the region / blew up** → not invariant (a crawl on its way
          out; no invariant set can leave the region it is claimed to live in);
        * it **left the cell set and returned** → invariant (a genuine recurrent
          set: fixed point approached from outside, cycle, or chaotic set);
        * it **never left** within ``budget`` steps → invariant (trapped);
        * it **left and never returned** within ``budget`` → not invariant.

        Along the way every *other* located attractor's cell it enters is
        recorded.  An orbit *on* an attractor can never reach a different
        attractor (invariance), so a hit means the recurrence machine split one
        set into fragments — which is what :func:`audit_attractors` uses to fuse
        them, and what the centroid-proximity :meth:`merge_map` cannot see when
        the fragments are far apart (the Lorenz case).

        A state that stops changing to ``1e-12`` relative is a converged fixed
        point: it is invariant and can reach nothing, so the march exits early.

        The third element separates the two ways invariance can fail, because
        they call for opposite remedies and are indistinguishable in the result:
        ``"escape"`` — the orbit walked out of the caller's **region** (or blew
        up) — versus ``"transient"`` — it stayed inside the region but left the
        cell set and never came back.  Only the second is the slow-crawl case a
        coarser sampling fixes; the first is either a region that genuinely holds
        no attractor or one that merely *clips* a real one (a Lorenz box capped
        at ``z <= 40`` while the attractor reaches ``z ~ 48`` discards it and
        labels the whole grid diverged), and those two are **not** distinguishable
        from inside, so the warning names both rather than guessing.  ``None``
        when the set is invariant.
        """
        self._reinit(start)
        left = returned = False
        reached: set[int] = set()
        prev: np.ndarray | None = None
        for _ in range(budget):
            state = self._advance()
            if state is None:
                return False, reached, "escape"
            cell = self.grid.index(state)
            if cell is None:
                return False, reached, "escape"
            other = others.get(cell)
            if other is not None:
                reached.add(other)
            if cell in own:
                returned = returned or left
            else:
                left = True
            if not left and prev is not None and _is_stationary(state, prev):
                return True, reached, None  # numerically stationary: a converged fixed point
            if returned and (reached or not others):
                return True, reached, None  # verdict settled; nothing more to learn
            prev = state
        ok = returned or not left
        return ok, reached, None if ok else "transient"

    def _settles_to(self, start: np.ndarray, budget: int) -> int | None:
        r"""March from ``start`` and report which located attractor it settles onto.

        A **read-only** replay of :meth:`map_ic`'s settle rule — the same
        ``attractor_revisits`` / ``basin_revisits`` counters against the same
        persistent cell labels — with the two side effects removed: it never
        writes a basin label and never *locates* a new attractor.  So calling it
        during the audit cannot invent ids, reorder the ones already found, or
        change the label state the caller's basin image is built from.

        Consulting the **basin** labels as well as the attractor cells is not an
        optimisation, it is what makes the test usable on a chaotic set: after
        the march an attractor owns only the ~``attractor_locate_steps`` cells
        the locate pass walked, and demanding two *consecutive* iterates inside
        that handful is a test the Hénon attractor itself fails (measured: 2714
        of its own basin's seeds rejected).  The basin cells are exactly the
        "leads here" evidence the FSM already accumulated.

        Returns ``None`` when the orbit blows up, leaves the region for
        ``lost_steps`` consecutive steps, or reaches no labelled cell within
        ``budget``.

        This is the question the *attraction* half of the audit asks: it is run
        from points perturbed **off** a located set, and an attractor is
        precisely a set that pulls its whole neighbourhood back (see
        :func:`audit_attractors`).
        """
        self._reinit(start)
        att_hit = bas_hit = lost = 0
        for _ in range(budget):
            state = self._advance()
            if state is None:
                return None
            cell = self.grid.index(state)
            if cell is None:
                lost += 1
                att_hit = bas_hit = 0
                if lost >= self.mx_lost:
                    return None
                continue
            lost = 0
            known = self._att_cells.get(cell)
            if known is not None:
                att_hit += 1
                bas_hit = 0
                if att_hit >= self.mx_att:
                    return known
                continue
            known = self._bas_cells.get(cell)
            if known is not None:
                bas_hit += 1
                att_hit = 0
                if bas_hit >= self.mx_bas:
                    return known
                continue
            att_hit = bas_hit = 0
        return None

    def _attracts(self, group: set[int], start: np.ndarray, budget: int) -> bool:
        r"""Whether a *neighbourhood* of ``start`` is pulled back onto ``group``.

        Perturbs ``start`` by one cell width along :math:`\pm` each axis and asks
        :meth:`_settles_to` where each perturbed orbit ends up.  The set attracts
        when a **strict majority** of the probes that stay inside the region come
        back to some member of ``group``.

        Why a majority and not all of them.  An attractor has a basin of
        *positive measure* around it, so almost every point of a small sphere
        returns; a saddle or a repellor is approached only along its stable set,
        which has measure zero, so an axis probe returns only in the accident
        that the stable manifold is axis-aligned — at most half of the ``2*dim``
        probes.  Demanding *all* of them instead is not a stricter version of the
        same statement, it is a different and false one: measured, one of the
        four cell-scale probes off a point of the **Hénon** attractor
        (``y + 0.05`` at ``(0.023, 0.267)``) leaves the basin and escapes,
        because the basin is genuinely not one cell wide there — an all-probes
        rule discards the Hénon attractor and every one of its 2714 captured
        seeds.  The 1-D saddle-node case the audit exists for, in contrast,
        returns on **zero** probes: :math:`x = +0.1` of
        :math:`\dot x = -0.01 + x^2` escapes upward and falls to the *other*
        attractor downward.

        One cell width is the natural probe scale: it is the smallest
        displacement the march can resolve, and it is what the basin image is
        drawn at.  A smaller probe would not do — the orbit would still be inside
        the set's own cells while its deviation grew, and the labels would report
        it "settled" on the very set it is running away from.

        A perturbation that lands outside the region is skipped rather than
        counted as a failure: a set sitting on the boundary of the box the caller
        chose must not be condemned for the box.  When *every* probe lands
        outside there is nothing to test and the set is kept.
        """
        probes = returned = 0
        for axis in range(self.grid.dim):
            for sign in (1.0, -1.0):
                probe = np.array(start, dtype=float, copy=True)
                probe[axis] += sign * float(self.grid.delta[axis])
                if self.grid.index(probe) is None:
                    continue  # outside the caller's region: not this set's fault
                probes += 1
                returned += int(self._settles_to(probe, budget) in group)
        return probes == 0 or 2 * returned > probes

    # -- Rust-kernel reconstruction (stream perf/basin-march) --

    def _load_march(self, outcome: dict[str, Any]) -> None:
        """Repopulate the FSM label state from a :func:`run.basin_march` outcome.

        The Rust kernel returns the *same* accumulated labelling the Python
        :meth:`map_ic` loop would build (it drives the identical sequential FSM
        over the identical engine stepper), but flattened for the FFI: cells are
        **flat row-major indices** over the grid ``counts``.  Inverting that
        flattening back to the per-axis cell-key tuples (the dict keys
        :meth:`merge_map` / :meth:`attractor_set` read) reconstructs
        ``_att_cells`` / ``_bas_cells`` / ``_att_points`` exactly, so every
        downstream step (proximity merge, harvest, basin painting) runs unchanged.

        ``_next_id`` is advanced past every located id so a *subsequent* Python
        ``map_ic`` (a mixed Rust-then-Python run, e.g. an unsupported follow-up)
        keeps allocating fresh ids.
        """
        counts = self.grid.counts
        self._att_cells = {
            _unflatten_cell(flat, counts): int(aid) for flat, aid in outcome["att_cells"].items()
        }
        self._bas_cells = {
            _unflatten_cell(flat, counts): int(aid) for flat, aid in outcome["bas_cells"].items()
        }
        # The kernel returns each attractor's point cloud as one ``(m, dim)``
        # array; the Python mapper keeps a *list of rows* (``attractor_set`` calls
        # ``np.asarray`` on it), so unpack into rows to match that internal shape
        # exactly (a later ``map_ic`` may ``.extend`` the list).
        self._att_points = {
            int(aid): [np.asarray(row, dtype=float) for row in pts]
            for aid, pts in outcome["att_points"].items()
        }
        located = [int(aid) for aid in outcome["att_points"]]
        self._next_id = (max(located) + 1) if located else 1


def _unflatten_cell(flat: int, counts: tuple[int, ...]) -> tuple[int, ...]:
    """Invert the kernel's row-major flat cell index back to a per-axis cell key.

    The Rust :class:`CellGrid` folds the per-axis indices into one flat index with
    ``flat = flat * counts[i] + k[i]`` for ``i`` ascending (row-major over
    ``counts``); this is the exact inverse — peel the last axis first.
    """
    key = [0] * len(counts)
    f = int(flat)
    for i in range(len(counts) - 1, -1, -1):
        key[i] = f % counts[i]
        f //= counts[i]
    return tuple(key)


def audit_attractors(mapper: _AttractorMapper, labels: np.ndarray) -> np.ndarray:
    r"""Verify every located set is an *attractor*; fuse fragments, drop phantoms.

    The recurrence machine's only detection predicate is *``mx_fnd`` consecutive
    steps into already-visited cells*.  That is satisfied by a genuine attractor
    **and** by an orbit that is merely moving slower than one cell per ``dt`` — a
    saddle-node bottleneck, a slow manifold, the crawl past an unstable spiral.
    On :math:`\dot x = \mu + x^2` with :math:`\mu > 0`, where :math:`\dot x > 0`
    everywhere so **no** attractor exists and every orbit escapes, the undefended
    machine reports 2 attractors at ``dt=0.05`` and 4 at ``dt=0.02`` (and
    ``basin_fractions`` hands 45 % of the region to them).  Refining ``dt`` — the
    natural instinct — makes it *worse*, because the crawl covers less of a cell
    per step.

    This pass runs after the march (on the Rust and Python paths alike, so their
    bit-identity is untouched) and re-marches from each located set.  It asks
    **two** questions, and a set has to answer both:

    1. *Is it invariant?*  :meth:`~_AttractorMapper._verify` re-marches from the
       set and checks the orbit stays in it or comes back.  A crawl on its way
       out fails here.
    2. *Does it attract?*  :meth:`~_AttractorMapper._attracts` perturbs one cell
       width along :math:`\pm` each axis and checks every perturbed orbit is
       pulled back onto the set.  **Invariance alone is not attraction**, and
       testing only the first certifies every invariant set the march happens to
       land on exactly — including unstable ones.  On
       :math:`\dot x = -0.01 + x^2` over ``Grid([-1], [1], (401,))`` the two
       equilibria are :math:`x = -0.1` (stable, :math:`f' = -0.2`) and
       :math:`x = +0.1` (**unstable**, :math:`f' = +0.2`); the grid puts a seed
       on each of them exactly, both sit still, and an invariance-only audit
       reports 2 attractors where the truth is 1.  Perturbing off :math:`+0.1`
       escapes to :math:`+\infty` on one side and falls to :math:`-0.1` on the
       other, so the attraction test rejects it.

    Ids are then grouped by mutual reachability and each group resolved:

    * a group with at least one invariant member **that attracts** is **one**
      attractor — its members are fused into the lowest id (the non-invariant
      members contribute their *cells* as basin cells but not their *points*,
      since those are transient).  This is what collapses the Lorenz
      fragmentation (2–4 reported attractors at ``dt <= 0.1`` where the truth is
      1) that the centroid-proximity :meth:`~_AttractorMapper.merge_map` cannot
      see.  Attraction is required of *some* member rather than all of them: a
      fragment is only part of the set, so one certified neighbourhood is enough
      to establish that the group attracts, and demanding it of every fragment
      would risk discarding a real attractor.
    * a group with **no** invariant member, or none that attracts, is deleted and
      the initial conditions that were assigned to it become :data:`DIVERGED` —
      the honest label, since the march never certified them as settling onto an
      attractor.

    Parameters
    ----------
    mapper : _AttractorMapper
        The marched machine; its label state is mutated in place.
    labels : ndarray of int
        Per-seed ids from the march, remapped in place-equivalent fashion.

    Returns
    -------
    ndarray of int
        ``labels`` with fused ids rewritten and rejected ids set to
        :data:`DIVERGED`.

    Warns
    -----
    UserWarning
        When a group fails the **invariance** check by leaving the caller's
        region — either the region holds no attractor or it *clips* a real one,
        so the remedy is a bigger region, not ``dt`` / ``resolution``.
    UserWarning
        When a group fails the **invariance** check inside the region — the
        recurrence FSM was fooled by slow motion, i.e. ``dt`` is too small for
        the cell size (the orbit does not cross a cell per step).  Raise ``dt``,
        coarsen ``resolution``, or accept that the region holds no attractor.
    UserWarning
        When a group is invariant but fails the **attraction** check — a saddle
        or a repellor that a seed landed on exactly.  Nothing is wrong with the
        settings; the set is simply not an attractor.
    """
    cellsets: dict[int, set[tuple[int, ...]]] = {}
    for cell, aid in mapper._att_cells.items():
        cellsets.setdefault(aid, set()).add(cell)
    if not cellsets:
        return labels

    budget = max(int(mapper.max_steps), 1)
    invariant: dict[int, bool] = {}
    causes: dict[int, str] = {}
    edges: dict[int, set[int]] = {}
    starts: dict[int, np.ndarray] = {}
    for aid in sorted(cellsets):
        own = cellsets[aid]
        others = {c: a for c, a in mapper._att_cells.items() if a != aid}
        points = mapper._att_points.get(aid) or []
        start = np.asarray(points[-1] if points else mapper.grid.center(min(own)), dtype=float)
        starts[aid] = start
        ok, reached, cause = mapper._verify(own, others, start, budget)
        invariant[aid] = ok
        edges[aid] = reached
        if cause is not None:
            causes[aid] = cause

    verdicts: list[tuple[list[int], str | None]] = []
    for group in _reachability_groups(sorted(cellsets), edges):
        certified = [k for k in group if invariant[k]]
        if not certified:
            # "escape" (the orbit left the caller's region) and "transient" (it
            # stayed inside but never came back) both mean "not invariant", but
            # they need opposite remedies, so they are reported apart.
            escaped = any(causes.get(k) == "escape" for k in group)
            verdicts.append((group, "escape" if escaped else "invariance"))
            continue
        accept = _accepting_ids(mapper, group)
        attracts = any(mapper._attracts(accept, starts[k], budget) for k in certified)
        verdicts.append((group, None if attracts else "attraction"))
    return _resolve_groups(mapper, labels, verdicts, invariant)


def _accepting_ids(mapper: _AttractorMapper, group: list[int]) -> set[int]:
    r"""``group`` plus every located id whose cloud lies within one cell of it.

    A probe that comes back to a *touching* fragment has come back to the same
    set: the proximity merge (:meth:`~_AttractorMapper.merge_map`) runs only
    after this audit, so at audit time one attractor may still be carried under
    two ids — and a fixed point that lands exactly on a cell boundary is the
    ordinary way that happens.  Measured on a globally attracting 1-D map
    ``x -> 0.5 x`` over ``Box([-1], [1])`` at 100 cells: the fixed point sits on
    the boundary between cells 49 and 50, successive orbits pin it at
    :math:`+2 \times 10^{-6}` and :math:`-1.8 \times 10^{-6}`, and the two ids
    each recapture only their *own* side — 1 probe of 2, short of a majority, so
    both halves of a provably global attractor were discarded.

    One cell is the right radius, and no larger: it is the resolution at which
    the whole march is defined, so two clouds inside it are indistinguishable to
    everything downstream.  Widening it toward ``merge_tol`` is not wanted — a
    caller who set a small ``merge_tol`` asked for those sets to stay *separate*,
    and this must not quietly certify one by the other's basin.
    """
    reach = float(np.linalg.norm(mapper.grid.delta))
    own = np.atleast_2d(
        np.asarray([p for k in group for p in mapper._att_points.get(k, [])], dtype=float)
    )
    accept = set(group)
    if own.size == 0:
        return accept
    for aid, pts in mapper._att_points.items():
        if aid in accept or not pts:
            continue
        if (
            set_distance(np.atleast_2d(np.asarray(pts, dtype=float)), own, method="minimum")
            <= reach
        ):
            accept.add(aid)
    return accept


def _reachability_groups(ids: list[int], edges: dict[int, set[int]]) -> list[list[int]]:
    """Group ids into connected components of the reachability graph (union-find)."""
    parent = {k: k for k in ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in ids:
        for b in sorted(edges.get(a, ())):
            if b in parent and find(a) != find(b):
                parent[find(b)] = find(a)
    out: dict[int, list[int]] = {}
    for k in ids:
        out.setdefault(find(k), []).append(k)
    return [sorted(g) for g in sorted(out.values(), key=min)]


def _resolve_groups(
    mapper: _AttractorMapper,
    labels: np.ndarray,
    verdicts: list[tuple[list[int], str | None]],
    invariant: dict[int, bool],
) -> np.ndarray:
    """Fuse each surviving reachability group; drop the rejected ones.

    ``verdicts`` pairs each group with ``None`` (it is an attractor) or the name
    of the check it failed — ``"escape"``, ``"invariance"`` or ``"attraction"``
    — which selects the warning the caller is given.
    """
    rewrite: dict[int, int] = {}
    dropped: dict[str, list[int]] = {"invariance": [], "escape": [], "attraction": []}
    for group, reason in verdicts:
        if reason is not None:
            dropped[reason].extend(group)
            continue
        keep = [k for k in group if invariant[k]]
        canonical = keep[0]
        for k in group:
            if k == canonical:
                continue
            rewrite[k] = canonical
            # A non-invariant fragment's *cells* are transient (basin) cells of the
            # canonical attractor; only an invariant fragment's points belong to
            # the attractor's cloud, so pooling is restricted to those.
            if invariant[k]:
                mapper._att_points[canonical].extend(mapper._att_points.pop(k, []))
            else:
                mapper._att_points.pop(k, None)

    gone = dropped["invariance"] + dropped["escape"] + dropped["attraction"]
    for cell, aid in list(mapper._att_cells.items()):
        if aid in gone:
            del mapper._att_cells[cell]
        elif aid in rewrite:
            if invariant[aid]:
                mapper._att_cells[cell] = rewrite[aid]
            else:
                del mapper._att_cells[cell]
                mapper._bas_cells.setdefault(cell, rewrite[aid])
    for cell, aid in list(mapper._bas_cells.items()):
        if aid in gone:
            del mapper._bas_cells[cell]
        elif aid in rewrite:
            mapper._bas_cells[cell] = rewrite[aid]
    for aid in gone:
        mapper._att_points.pop(aid, None)

    out = labels
    if rewrite or gone:
        out = labels.copy()
        for old, new in rewrite.items():
            out[labels == old] = new
        for old in gone:
            out[labels == old] = DIVERGED
    _warn_dropped(labels, dropped)
    return out


#: What each audit rejection means, and what (if anything) the caller should do.
_DROP_ADVICE: dict[str, str] = {
    "invariance": (
        "re-marching from them stays inside the region but leaves the set and never "
        "returns, so they are slow transients (a bottleneck / slow manifold), not "
        "attractors. The recurrence machine mistakes slow motion for recurrence when dt "
        "is too small for the cell size — raise dt, coarsen resolution, or accept that "
        "this region holds no attractor."
    ),
    "escape": (
        "re-marching from them leaves the region you asked about (or diverges), so they "
        "are not invariant *in it*. Two situations look identical from inside and only "
        "you can tell them apart: the region genuinely holds no attractor (every orbit "
        "escapes), or the region CLIPS one that is real — so enlarge it and re-run "
        "before concluding there is nothing here. Changing dt or resolution addresses "
        "neither."
    ),
    "attraction": (
        "they are invariant but do not attract: perturbing one cell width off them "
        "escapes, stalls, or falls to a different attractor. That is a saddle or a "
        "repellor a seed landed on exactly (the recurrence predicate cannot tell an "
        "unstable invariant set from an attractor), not a settings problem."
    ),
}


#: How each rejection reason is *named* in the warning.  ``"escape"`` is a
#: sub-case of the invariance check, not a third check, so it carries the same
#: name; only the remedy text differs.
_DROP_LABEL: dict[str, str] = {
    "invariance": "invariance",
    "escape": "invariance",
    "attraction": "attraction",
}


def _warn_dropped(labels: np.ndarray, dropped: dict[str, list[int]]) -> None:
    """Emit one ``UserWarning`` per audit-rejection reason that fired."""
    for reason, ids in dropped.items():
        if not ids:
            continue
        n_seeds = int(np.sum(np.isin(labels, ids)))
        warnings.warn(
            f"discarded {len(ids)} located set(s) that failed the "
            f"{_DROP_LABEL[reason]} check: {_DROP_ADVICE[reason]} {n_seeds} initial "
            "condition(s) assigned to them are reported as diverged.",
            UserWarning,
            stacklevel=5,
        )


def _march_supported(system: Any, backend: str) -> bool:
    """Whether the Rust basin-march kernel can drive ``system`` on ``backend``.

    Supported only for an ODE flow or a discrete map (the kernel has no
    DDE/SDE path — those reach here as ``False`` and keep the Python loop) on a
    compiled-engine backend (``interp`` / ``jit``).  ``reference`` (the wheel-free
    oracle) and any non-finite-state family fall back, mirroring the Poincaré
    engine-march carve-outs.
    """
    if backend not in ("interp", "jit"):
        return False
    if _looks_unsupported(system):  # DDE / SDE — no finite-dimensional point
        return False
    # A flow or a map: both have a Rust kernel.  Anything else (a wrapped/custom
    # stepper with no lowerable tape) is caught when lowering fails below.
    return _is_discrete(system) or hasattr(system, "_equations")


def classify_seeds(
    mapper: _AttractorMapper,
    seeds: np.ndarray,
    *,
    backend: str = "jit",
    jit: bool = False,
) -> np.ndarray:
    """Classify a batch of seeds, accelerating the per-IC march in Rust when possible.

    The single seam both :func:`attractors` and :func:`basins`
    drive the recurrence FSM through.  When the run is *supported* — an ODE flow or
    a map on the compiled engine whose tape lowers — the **entire** per-seed march
    (stepping + cell-binning + the shared-label early-out) runs in one sequential
    Rust kernel call (:func:`tsdynamics.engine.run.basin_march`), and ``mapper`` is
    reloaded from its outcome (:meth:`_AttractorMapper._load_march`) so the existing
    ``merge_map`` / ``attractor_set`` / basin-painting post-processing is unchanged.
    The per-cell-check numerics are byte-for-byte the released ``system.step()``, so
    the returned labels are **bit-identical** to the pure-Python loop.

    Anything unsupported (the ``reference`` backend, a non-lowering ``_step``, a
    DDE/SDE) transparently falls back to the per-seed Python ``map_ic`` loop — the
    same algorithm, the same result — so this never changes an answer, only the
    speed of the supported path.

    Parameters
    ----------
    mapper : _AttractorMapper
        The (freshly constructed) recurrence machine; mutated in place so a caller
        runs the standard post-processing on it afterwards.
    seeds : ndarray, shape (n_seeds, dim)
        Initial conditions to classify, in the order the shared labelling
        accumulates in (kept strictly serial — see the kernel's why-sequential
        note).
    backend : {"jit", "interp", "reference"}, default "jit"
        ``"jit"`` / ``"interp"`` drive the Rust kernel (when supported);
        ``"reference"`` (or any other) forces the Python loop.
    jit : bool, default False
        Select the Cranelift evaluator for the supported Rust path.

    Returns
    -------
    ndarray of int, shape (n_seeds,)
        The per-seed attractor id (``>= 1``) or :data:`DIVERGED` (``-1``).
    """
    seeds = np.ascontiguousarray(seeds, dtype=np.float64).reshape(-1, mapper.grid.dim)
    system = mapper.system

    labels: np.ndarray | None = None
    if _march_supported(system, backend):
        outcome = _try_rust_march(mapper, seeds, jit=jit)
        if outcome is not None:
            mapper._load_march(outcome)
            labels = np.asarray(outcome["labels"], dtype=np.int64)

    if labels is None:
        # Fallback (the oracle): the per-seed Python FSM loop.
        labels = np.empty(seeds.shape[0], dtype=np.int64)
        for i, ic in enumerate(seeds):
            labels[i] = mapper.map_ic(ic)

    # The invariance audit runs on *both* paths, from the same reloaded label
    # state, so the Rust/Python bit-identity contract is untouched.
    return audit_attractors(mapper, labels)


def _try_rust_march(
    mapper: _AttractorMapper, seeds: np.ndarray, *, jit: bool
) -> dict[str, Any] | None:
    """Build the inputs, call the Rust kernel, and return its outcome (or ``None``).

    Returns ``None`` — signalling the caller to use the Python fallback — when the
    system's tape does not lower (a non-symbolic ``_step`` / ``_equations``) or the
    compiled engine is unavailable.  Resolves the method / Jacobian-carrying tape /
    tolerances exactly as :meth:`_AttractorMapper._reinit` does (``_default_method``
    plus :data:`~tsdynamics.utils.tolerances.BASIN_RTOL` /
    :data:`~tsdynamics.utils.tolerances.BASIN_ATOL`), so the kernel reproduces the
    Python oracle's stepping numerics bit-for-bit.  The two sites must name the
    *same* constants — see :meth:`_AttractorMapper._reinit` for why the march keeps
    its own, looser pair.
    """
    from tsdynamics import solvers
    from tsdynamics.engine import run
    from tsdynamics.engine.compile import TapeCompileError
    from tsdynamics.engine.problem import map_problem, ode_problem

    grid = mapper.grid
    thresholds = (
        mapper.max_steps,
        mapper.mx_fnd,
        mapper.mx_loc,
        mapper.mx_att,
        mapper.mx_bas,
        mapper.mx_lost,
    )
    system = mapper.system
    counts = np.asarray(grid.counts, dtype=np.int64)

    try:
        if mapper._discrete:
            mprob = map_problem(system)
            return run.basin_march(
                mprob.tape.to_arrays(),
                mprob.params_vec(),
                grid.lo,
                grid.hi,
                counts,
                seeds,
                thresholds,
                is_discrete=True,
                jit=jit,
            )
        # A flow: resolve the method the released ``step`` would use and, for an
        # implicit kernel, lower the Jacobian-carrying tape (the engine refuses an
        # implicit step without ∂f/∂u) — exactly as ``ContinuousSystem.reinit``.
        method = getattr(system, "_default_method", "RK45")
        resolution = solvers.resolve(method)
        oprob = ode_problem(system, **resolution.build_kwargs)
        return run.basin_march(
            oprob.tape.to_arrays(),
            oprob.params_vec(),
            grid.lo,
            grid.hi,
            counts,
            seeds,
            thresholds,
            is_discrete=False,
            method=resolution.name,
            rtol=BASIN_RTOL,
            atol=BASIN_ATOL,
            dt=mapper.dt,
            jit=jit,
        )
    except TapeCompileError:
        # A non-symbolic ``_step`` / ``_equations`` (e.g. the complex-arithmetic
        # Newton map) cannot lower — fall back to the Python loop, the oracle.
        return None
    except run.EngineNotAvailableError:
        # No compiled wheel — the pure-Python fallback still works.
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def attractors(
    system: Any,
    region: Grid | Box | Ball | Sequence[tuple[float, ...]] | None = None,
    *,
    resolution: int | tuple[int, ...] = 100,
    n_seeds: int = 1000,
    seed: int | None = 0,
    dt: float = 1.0,
    max_steps: int = 10000,
    merge_tol: float | None = None,
    **fsm: Any,
) -> AttractorSet:
    r"""
    Find the attractors a system has within a region, via recurrences.

    Tessellate ``region`` into cells, draw ``n_seeds`` random initial conditions
    from it, and follow each until it settles into a recurrent cell set (a new
    attractor) or inherits one already found (Datseris & Wagemakers, 2022).

    Parameters
    ----------
    system : System
        A discrete map or continuous flow.  Delay and stochastic systems are not
        supported (their state is not a finite-dimensional point).
    region : Box, Ball, or Grid
        Where to sample initial conditions and the box the recurrence cells cover.
    resolution : int or tuple of int, default 100
        Recurrence cells per axis when ``region`` is a Box/Ball (a Grid carries
        its own ``counts``).  Too coarse merges distinct attractors; too fine
        stops a chaotic trajectory from recurring — tune it to the attractor scale.
    n_seeds : int, default 1000
        Number of random initial conditions to classify.
    seed : int, optional
        Seed for the initial-condition sampler (reproducible).
    dt : float, default 1.0
        Integration step between cell checks for a flow (ignored for a map).
    max_steps : int, default 10000
        Per-seed step cap before declaring divergence.
    merge_tol : float, optional
        Merge attractors whose centroids lie within this distance (a split-set
        cleanup).  ``None`` uses two recurrence-cell diagonals; ``0`` disables it.
    **fsm
        Finite-state-machine thresholds forwarded to :class:`_AttractorMapper`
        (``consecutive_recurrences``, ``attractor_locate_steps``,
        ``attractor_revisits``, ``basin_revisits``, ``lost_steps``).

    Returns
    -------
    AttractorSet
        The located attractors plus how many seeds diverged.

    Raises
    ------
    TypeError
        If ``system`` is a delay or stochastic system (their state is not a
        finite-dimensional point the cell tessellation can bin).

    Warns
    -----
    UserWarning
        When a located set fails the invariance audit and is discarded — the
        recurrence predicate was fooled by slow motion (``dt`` too small for the
        cell size).  See :func:`audit_attractors`.

    Notes
    -----
    **Every located attractor is verified before it is returned**
    (:func:`audit_attractors`).  The bare recurrence predicate — ``mx_fnd``
    consecutive steps into already-visited cells — is also satisfied by an orbit
    that merely moves less than one cell per ``dt``, so a bottleneck, a slow
    manifold or the crawl past an unstable spiral would otherwise be reported as
    an attractor (measured: 2–4 phantom attractors on :math:`\dot x = \mu + x^2`,
    which provably has none; 2–4 fragments of the *one* Lorenz attractor at
    ``dt <= 0.1``).  The audit re-marches from each located set, fuses mutually
    reachable fragments and drops sets the orbit leaves and never returns to.

    The seed march is **sequential by design**, not a parallelism oversight: each
    seed is followed cell-by-cell, and the persistent cell labels (``_att_cells`` /
    ``_bas_cells``) accumulated by earlier seeds let later seeds settle cheaply by
    reaching an already-labelled cell.  That shared, order-dependent labelling state
    is what makes the sweep amortise — and is exactly why the march cannot be
    parallelised without changing the result or the determinism.

    On a supported engine run (an ODE flow or a map whose ``_step`` lowers, on the
    ``interp`` / ``jit`` backend) the whole per-IC march now runs in **one
    sequential Rust kernel call** (stream ``perf/basin-march``) — stepping,
    cell-binning and the shared-label early-out all in Rust, with no per-``dt``
    Python→FFI round-trip — so it is fast *without* parallelism.  The per-cell-check
    numerics are byte-for-byte the released ``system.step()``, so the result is
    bit-identical to the pure-Python loop, which stays the fallback (and the oracle)
    for ``reference``, a non-lowering ``_step``, and DDE/SDE systems.

    Why thread-parallelism is a *net loss* here, and why the kernel is therefore
    serial (measured, so the next reader does not re-derive it).  The shared
    early-out is the dominant work-saver: on the two-well Duffing 60×60 grid a
    serial march takes ~42k engine steps, whereas marching every seed independently
    to ``max_steps`` (the only way to lift the serial label dependency) takes
    ~1.4M — a **~34×** work inflation that 16 cores cannot recover (the independent
    full-march alone clocked ~5× *slower* than the whole serial run).  A
    "speculative march in parallel, fold in seed order" scheme would reproduce the
    labels bit-for-bit (each seed's state stream is a pure function of its IC) but
    pays exactly that 34× over-march, so it is abandoned.  Dense-block FFI batching
    is *not* an option either: the FSM checks the cell after every per-``dt``
    ``step()`` restart, and an adaptive dense-output block over several ``dt`` does
    not reproduce those per-``dt`` restart states bit-for-bit (it drifts at ~5e-7),
    so it would change the cell sequence and the labels.  The march stays serial
    because that is the algorithm; the Rust kernel makes that serial march cheap.

    References
    ----------
    G. Datseris and A. Wagemakers, "Effortless estimation of basins of
    attraction", *Chaos* **32**, 023104 (2022).
    """
    _reject_unsupported(system, "attractors")
    reject_unknown_fsm(fsm, analysis="attractors")
    region = coerce_region(region, analysis="attractors", system=system, want_grid=False)

    grid = _recurrence_grid(region, resolution)
    mapper = _AttractorMapper(system, grid, dt=dt, max_steps=max_steps, **fsm)
    draw = sampler(region, seed=seed)

    # Draw the whole seed cloud up front (the sampler order is unchanged, so the
    # classification order — and thus the shared, order-dependent labelling — is
    # identical to the per-seed draw-then-classify loop), then march it: one
    # sequential Rust kernel call on a supported engine run, else the per-seed
    # Python loop (the oracle).  Either way ``mapper`` carries the same FSM state.
    seeds = np.array([draw() for _ in range(int(n_seeds))], dtype=np.float64).reshape(-1, grid.dim)
    from ...engine.run import resolve_backend

    backend = resolve_backend(getattr(system, "_default_backend", "jit"))
    labels = classify_seeds(mapper, seeds, backend=backend, jit=backend == "jit")
    diverged = int(np.sum(labels == DIVERGED))
    merge = mapper.merge_map(resolve_merge_tol(grid, merge_tol))
    found = mapper.attractor_set(diverged=diverged, seeds=int(n_seeds), merge=merge)
    found, _ = canonical_relabel(found, merge)
    # Attach provenance without re-allocating the (potentially large) attractor
    # dict — ``replace`` reuses every field but ``meta``.
    return replace(found, meta=AnalysisResult.build_meta(system, analysis="attractors"))


def canonical_relabel(
    found: AttractorSet, merge: dict[int, int]
) -> tuple[AttractorSet, dict[int, int]]:
    """Renumber attractors by **where they are**, not by when they were found.

    The recurrence machine hands out ids in the order it stumbles on
    attractors, which is the order the seeds happen to be visited in — so the
    same physical state is ``#1`` when a parameter is swept one way and ``#2``
    swept back, and a two-panel figure paints one well in two colours.  The
    location is the one property of an attractor that does not depend on how it
    was found, so it is what orders them: ascending lexicographically by
    representative, ties broken by the larger attractor first and then by the
    old id, which keeps the result deterministic for *concentric* sets (two
    rings share a centroid exactly).

    Returns the renumbered set and the composed ``{old_id: new_id}`` map — the
    merge and the renumbering in one permutation, so the label image and the
    attractor set are remapped by the same table and cannot drift apart.

    Parameters
    ----------
    found : AttractorSet
        The merged set, straight from
        :meth:`_AttractorMapper.attractor_set`.
    merge : dict[int, int]
        The proximity-merge map its ids already went through.

    Returns
    -------
    tuple[AttractorSet, dict[int, int]]
    """
    ids = found.ids
    if not ids:
        return found, merge

    def key(k: int) -> tuple[Any, ...]:
        att = found.attractors[k]
        pts = np.asarray(att.points, dtype=float)
        where = tuple(att.center) if pts.size else ()
        return (where, -int(att.cells), int(k))

    renumber = {old: new for new, old in enumerate(sorted(ids, key=key), start=1)}
    if all(old == new for old, new in renumber.items()):
        return found, merge
    relabelled = AttractorSet(
        attractors={renumber[k]: replace(found.attractors[k], id=renumber[k]) for k in ids},
        diverged=found.diverged,
        seeds=found.seeds,
        meta=found.meta,
    )
    composed = {k: renumber.get(merge.get(k, k), merge.get(k, k)) for k in set(merge) | set(ids)}
    return relabelled, composed


def resolve_merge_tol(cellgrid: _CellGrid, merge_tol: float | None) -> float:
    """Resolve the proximity-merge tolerance (``None`` → two cell diagonals)."""
    if merge_tol is None:
        return 2.0 * float(np.linalg.norm(cellgrid.delta))
    return float(merge_tol)


def _looks_unsupported(system: Any) -> bool:
    """Return True for delay / stochastic systems (no finite-dimensional state)."""
    return hasattr(system, "_drift") or hasattr(system, "history") or hasattr(system, "_delays")


def _reject_unsupported(system: Any, fn_name: str) -> None:
    """Raise a uniform error for measured data and for delay / stochastic systems.

    Shared by the basin entry points (``attractors``,
    ``basins``, ``basin_fractions``, ``continuation``) so an
    unsupported first argument fails early with one clear message instead of
    opaquely inside the step loop.

    Measured data is checked first and separately: a basin is a property of the
    *model* -- it is found by launching fresh initial conditions and seeing where
    each one goes -- so a recorded trajectory cannot answer it at any resolution.
    One already-run trajectory used to reach the FSM and surface as
    ``AttributeError: 'Trajectory' object has no attribute 'reinit'``.
    """
    reject_data(system, analysis=fn_name)
    if not _is_discrete(system) and _looks_unsupported(system):
        raise TypeError(f"{fn_name} supports maps and flows, not delay/stochastic systems.")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
