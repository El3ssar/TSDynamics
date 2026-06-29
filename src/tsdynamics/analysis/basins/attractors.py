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

:func:`find_attractors` runs the machine from a cloud of seeds and returns the
attractors it discovers; :class:`_AttractorMapper` is the reusable engine the
basin and continuation layers drive over a full grid.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...data import Ball, Box, Grid, sampler, set_distance
from ...errors import ConvergenceError
from .._result import AnalysisResult
from ._common import (
    DIVERGED_COLOR,
    PALETTE,
    _CellGrid,
    _palette_indices,
    _recurrence_grid,
    _representative,
)

if TYPE_CHECKING:
    from ...data.sampling import _SetMethod

__all__ = [
    "Attractor",
    "AttractorSet",
    "find_attractors",
]

#: Label returned for an initial condition that leaves the region / never settles.
DIVERGED = -1


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attractor(AnalysisResult):
    """
    One located attractor: a point cloud of states sampled on it.

    Attributes
    ----------
    id : int
        Integer label (``>= 1``) identifying this attractor within its set.
    points : ndarray, shape (m, dim)
        States sampled while the trajectory was on the attractor.  A fixed point
        collapses to one repeated point; a cycle/chaotic set spreads out.
    cells : int
        Number of distinct grid cells the attractor occupies (a coarse size).
    """

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

    def __repr__(self) -> str:  # noqa: D105
        c = np.round(self.center, 4)
        return f"Attractor(id={self.id}, center={c.tolist()}, cells={self.cells})"


@dataclass(frozen=True)
class AttractorSet(AnalysisResult):
    """
    The attractors found in a region, keyed by integer id.

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

    def __iter__(self) -> Iterator[Attractor]:  # noqa: D105
        return iter(self.attractors.values())

    def __getitem__(self, key: int) -> Attractor:  # noqa: D105
        return self.attractors[key]

    @property
    def ids(self) -> list[int]:
        """Sorted attractor ids."""
        return sorted(self.attractors)

    @property
    def centers(self) -> np.ndarray:
        """Stack of attractor representatives, shape ``(n_attractors, dim)``."""
        return np.array([self.attractors[k].center for k in self.ids])

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

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the located attractors as a backend-agnostic :class:`PlotSpec`.

        Builds a ``PHASE_PORTRAIT_2D`` of every attractor's point cloud as one
        ``SCATTER`` layer (the first two state coordinates).  Each point carries
        a ``"cat"`` channel — the attractor id's swatch index in the shared
        categorical palette (``tab20``) — so the same id is drawn the same colour
        here and on the basin image (:meth:`BasinsResult.to_plot_spec`).  The
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
        from tsdynamics.viz.spec import Axis, Colorbar, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.PHASE_PORTRAIT_2D
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

        layer = Layer(
            PlotKind.SCATTER,
            {"x": np.concatenate(xs), "y": np.concatenate(ys), "cat": np.concatenate(cats)},
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
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            aspect="equal",
            title=f"attractors ({len(self)})",
            x=Axis(label="x1"),
            y=Axis(label="x2"),
            layers=[layer],
            colorbar=Colorbar(label="attractor", cmap=PALETTE, discrete=True),
            meta=meta,
        )

    def __repr__(self) -> str:  # noqa: D105
        return f"AttractorSet({len(self)} attractors, {self.diverged}/{self.seeds} diverged)"


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

        self._discrete = bool(getattr(system, "is_discrete", False))
        self._step_arg: int | float = 1 if self._discrete else self.dt

        # Persistent labels (sparse): cell key → attractor id.  These dicts are
        # read AND written by every ``map_ic`` call and carry the order-dependent
        # state that makes the sweep amortise (a later seed inherits an earlier
        # seed's label).  Mutating them from threads would both race and change the
        # result, so ``map_ic`` is driven strictly serially — see the measured
        # rationale in ``find_attractors``'s Notes for why thread-parallelism is a
        # net loss here, not just unsafe.
        self._att_cells: dict[tuple[int, ...], int] = {}
        self._bas_cells: dict[tuple[int, ...], int] = {}
        self._att_points: dict[int, list[np.ndarray]] = {}
        self._next_id = 1

    # -- driving the system through the protocol --

    def _reinit(self, ic: np.ndarray) -> None:
        """Reinitialise the driven system at initial condition ``ic``."""
        self.system.reinit(np.asarray(ic, dtype=float))

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
        Group attractor ids whose point clouds sit within ``tol`` (centroid).

        The recurrence machine occasionally splits one attractor into two cell
        sets (e.g. a chaotic set approached from two sides).  A small ``tol``
        unions only near-coincident ids, leaving genuinely distinct attractors
        apart.  Returns ``{old_id: canonical_id}``.
        """
        ids = sorted(self._att_points)
        parent = {k: k for k in ids}

        def find(a: int) -> int:
            """Union-find root of ``a`` with path compression."""
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        centers = {k: _representative(np.asarray(self._att_points[k], dtype=float)) for k in ids}
        for i, a in enumerate(ids):
            for c in ids[i + 1 :]:
                if find(a) == find(c):
                    continue
                if float(np.linalg.norm(centers[a] - centers[c])) <= tol:
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
        for k, pts in self._att_points.items():
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def find_attractors(
    system: Any,
    region: Box | Ball | Grid,
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

    Notes
    -----
    The seed loop is **sequential by design**, not a parallelism oversight: each
    seed is followed cell-by-cell through the system's step protocol (one engine
    round-trip per cell check), and the persistent cell labels (``_att_cells`` /
    ``_bas_cells``) accumulated by earlier seeds let later seeds settle cheaply by
    reaching an already-labelled cell.  That shared, order-dependent labelling
    state is what makes the sweep amortise — and is exactly why the loop cannot be
    parallelised without changing the result or the determinism.  The dominant
    cost is therefore ``n_seeds`` × (steps to settle) engine steps.

    Why thread-parallelism is a *net loss* here (measured, so the next reader does
    not re-derive it).  The shared early-out is the dominant work-saver: on the
    two-well Duffing 60×60 grid a serial run takes ~42k engine steps, whereas
    marching every seed independently to ``max_steps`` (the only way to lift the
    serial label dependency) takes ~1.4M — a **~34×** work inflation that 16 cores
    cannot recover (the independent full-march alone clocked ~5× *slower* than the
    whole serial run).  Worse, only ~36 % of the serial wall time is in the engine
    ``step()`` FFI; the other ~64 % is the inherently-serial FSM (cell binning +
    the shared-label dict reads/writes), so Amdahl caps any speedup even before the
    work-inflation penalty.  A "speculative march in parallel, fold in seed order"
    scheme would reproduce the labels bit-for-bit (each seed's state stream is a
    pure function of its IC) but pays exactly that 34× over-march, so it is
    abandoned.  Dense-block FFI batching is *not* an option either: the FSM checks
    the cell after every per-``dt`` ``step()`` restart, and an adaptive dense-output
    block over several ``dt`` does not reproduce those per-``dt`` restart states
    bit-for-bit (it drifts at ~5e-7), so it would change the cell sequence and the
    labels.  The loop stays serial because that is the algorithm.

    References
    ----------
    G. Datseris and A. Wagemakers, "Effortless estimation of basins of
    attraction", *Chaos* **32**, 023104 (2022).
    """
    _reject_unsupported(system, "find_attractors")

    grid = _recurrence_grid(region, resolution)
    mapper = _AttractorMapper(system, grid, dt=dt, max_steps=max_steps, **fsm)
    draw = sampler(region, seed=seed)

    diverged = 0
    for _ in range(int(n_seeds)):
        if mapper.map_ic(draw()) == DIVERGED:
            diverged += 1
    merge = mapper.merge_map(resolve_merge_tol(grid, merge_tol))
    found = mapper.attractor_set(diverged=diverged, seeds=int(n_seeds), merge=merge)
    # Attach provenance without re-allocating the (potentially large) attractor
    # dict — ``replace`` reuses every field but ``meta``.
    return replace(found, meta=AnalysisResult.build_meta(system, analysis="find_attractors"))


def resolve_merge_tol(cellgrid: _CellGrid, merge_tol: float | None) -> float:
    """Resolve the proximity-merge tolerance (``None`` → two cell diagonals)."""
    if merge_tol is None:
        return 2.0 * float(np.linalg.norm(cellgrid.delta))
    return float(merge_tol)


def _looks_unsupported(system: Any) -> bool:
    """Return True for delay / stochastic systems (no finite-dimensional state)."""
    return hasattr(system, "_drift") or hasattr(system, "history") or hasattr(system, "_delays")


def _reject_unsupported(system: Any, fn_name: str) -> None:
    """Raise a uniform ``TypeError`` for delay / stochastic systems.

    Shared by the basin entry points (``find_attractors``,
    ``basins_of_attraction``, ``basin_fractions``, ``continuation``) so an
    unsupported system fails early with one clear message instead of opaquely
    inside the step loop.
    """
    if getattr(system, "is_discrete", False) is False and _looks_unsupported(system):
        raise TypeError(f"{fn_name} supports maps and flows, not delay/stochastic systems.")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
