r"""
Basins of attraction and basin fractions (basin stability).

Two complementary views of "which attractor wins from where":

- :func:`basins_of_attraction` paints a full grid — every lattice point is
  classified, giving the basin *image* (the input to the basin-entropy,
  uncertainty-exponent and Wada quantifiers).
- :func:`basin_fractions` draws random initial conditions from a region and
  reports each attractor's share — the **basin stability** of

      P. J. Menck, J. Heitzig, N. Marwan and J. Kurths, "How basin stability
      complements the linear-stability paradigm", *Nature Physics* **9**, 89
      (2013),

i.e. the probability a random state converges to a given attractor, with a
Monte-Carlo standard error that depends only on the fraction and the sample
count (not the dimension).

Both reuse the recurrence finder in
:mod:`tsdynamics.analysis.basins.attractors`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...data import Ball, Box, Grid, grid_points, sampler
from ._common import _apply_merge, _recurrence_grid
from .attractors import DIVERGED, AttractorSet, _AttractorMapper, resolve_merge_tol

__all__ = [
    "BasinFractions",
    "BasinsResult",
    "basin_fractions",
    "basins_of_attraction",
]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasinsResult:
    """
    A basin diagram: every grid cell labelled by the attractor it reaches.

    Attributes
    ----------
    labels : ndarray of int
        Attractor id (``>= 1``) per grid cell, shaped like ``grid.shape``; ``-1``
        marks diverged / unsettled cells.
    grid : Grid
        The lattice the labels are laid out on.
    attractors : AttractorSet
        The attractors the labels refer to.
    """

    labels: np.ndarray = field(repr=False)
    grid: Grid
    attractors: AttractorSet

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape of the basin image."""
        return self.labels.shape

    @property
    def n_attractors(self) -> int:
        """Number of distinct attractors present in the image."""
        return int(np.sum(np.unique(self.labels) >= 1))

    @property
    def fractions(self) -> dict[int, float]:
        """Fraction of grid cells in each attractor's basin (diverged cells excluded).

        Keyed by attractor id (``>= 1``); the diverged share is reported separately
        by :attr:`diverged_fraction` (so this mirrors
        :attr:`BasinFractions.fractions`).
        """
        ids, counts = np.unique(self.labels, return_counts=True)
        total = self.labels.size
        return {int(k): float(c) / total for k, c in zip(ids, counts, strict=True) if k != DIVERGED}

    @property
    def diverged_fraction(self) -> float:
        """Fraction of cells that diverged / never settled."""
        return float(np.mean(self.labels == DIVERGED))

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"BasinsResult(shape={self.shape}, n_attractors={self.n_attractors}, "
            f"diverged={self.diverged_fraction:.3g})"
        )


@dataclass(frozen=True)
class BasinFractions:
    """
    Monte-Carlo basin stability: each attractor's share of a sampled region.

    Attributes
    ----------
    fractions : dict[int, float]
        Attractor id → fraction of sampled initial conditions converging to it.
    diverged : float
        Fraction of samples that diverged / never settled.
    n : int
        Number of initial conditions sampled.
    attractors : AttractorSet
        The attractors the ids refer to.
    """

    fractions: dict[int, float]
    diverged: float
    n: int
    attractors: AttractorSet

    @property
    def standard_error(self) -> dict[int, float]:
        r"""Binomial standard error :math:`\sqrt{p(1-p)/n}` per fraction."""
        return {k: float(np.sqrt(p * (1.0 - p) / self.n)) for k, p in self.fractions.items()}

    @property
    def dominant(self) -> int | None:
        """Id of the attractor with the largest basin (``None`` if all diverged)."""
        return max(self.fractions, key=self.fractions.get) if self.fractions else None

    def __getitem__(self, key: int) -> float:  # noqa: D105
        return self.fractions[key]

    def __repr__(self) -> str:  # noqa: D105
        body = ", ".join(f"{k}:{v:.3g}" for k, v in sorted(self.fractions.items()))
        return f"BasinFractions({{{body}}}, diverged={self.diverged:.3g}, n={self.n})"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def basins_of_attraction(
    system: Any,
    grid: Grid,
    *,
    recurrence: Box | Grid | None = None,
    recurrence_resolution: int | tuple[int, ...] = 100,
    dt: float = 1.0,
    max_steps: int = 10000,
    merge_tol: float | None = None,
    **fsm: Any,
) -> BasinsResult:
    r"""
    Classify every point of a grid by the attractor it converges to.

    Each lattice point is followed until it settles into a recurrent cell set
    (Datseris & Wagemakers, 2022); by default the grid doubles as the recurrence
    tessellation, so labels accumulate and most points settle cheaply by reaching
    an already-labelled cell.

    For a higher-dimensional flow whose basins are viewed on a slice (e.g. a
    position grid of a system that also carries velocities), pass a full-dimension
    ``recurrence`` box covering the whole trajectory range and let ``grid`` be the
    thin slice of initial conditions (free axes pinned with ``counts == 1``).

    Parameters
    ----------
    system : System
        A discrete map or continuous flow.
    grid : Grid
        The lattice of initial conditions (full state dimension; a slice pins free
        axes with ``counts == 1``).  Its ``counts`` set the recurrence resolution
        when ``recurrence`` is not given.
    recurrence : Box or Grid, optional
        Full-dimension region whose tessellation recurrences are detected on.
        Defaults to ``grid`` itself (correct for maps and full grids; required when
        ``grid`` is a degenerate slice).
    recurrence_resolution : int or tuple of int, default 100
        Recurrence cells per axis when ``recurrence`` is a Box.
    dt : float, default 1.0
        Integration step between cell checks for a flow (ignored for a map).
    max_steps : int, default 10000
        Per-point step cap before declaring divergence.
    merge_tol : float, optional
        Merge attractors whose centroids lie within this distance.  ``None`` uses
        two recurrence-cell diagonals; ``0`` disables it.
    **fsm
        Finite-state-machine thresholds forwarded to
        :class:`~tsdynamics.analysis.basins.attractors._AttractorMapper`.

    Returns
    -------
    BasinsResult
        The labelled basin image and the attractors it refers to.

    References
    ----------
    G. Datseris and A. Wagemakers, "Effortless estimation of basins of
    attraction", *Chaos* **32**, 023104 (2022).
    """
    if recurrence is None:
        cellgrid = _recurrence_grid(grid, grid.counts)
    else:
        cellgrid = _recurrence_grid(recurrence, recurrence_resolution)
    mapper = _AttractorMapper(system, cellgrid, dt=dt, max_steps=max_steps, **fsm)

    points = grid_points(grid)
    labels = np.empty(points.shape[0], dtype=np.int64)
    diverged = 0
    for i, p in enumerate(points):
        lab = mapper.map_ic(p)
        labels[i] = lab
        if lab == DIVERGED:
            diverged += 1

    merge = mapper.merge_map(resolve_merge_tol(cellgrid, merge_tol))
    labels = _apply_merge(labels.reshape(grid.shape), merge)
    attractors = mapper.attractor_set(diverged=diverged, seeds=points.shape[0], merge=merge)
    return BasinsResult(labels=labels, grid=grid, attractors=attractors)


def basin_fractions(
    system: Any,
    region: Box | Ball | Grid,
    *,
    n: int = 10000,
    resolution: int | tuple[int, ...] = 100,
    seed: int | None = 0,
    dt: float = 1.0,
    max_steps: int = 10000,
    merge_tol: float | None = None,
    **fsm: Any,
) -> BasinFractions:
    r"""
    Estimate basin stability: each attractor's share of a sampled region.

    Draw ``n`` random initial conditions from ``region`` and classify each; the
    fraction converging to an attractor estimates its basin stability (Menck et
    al., 2013).  The estimate is dimension-free — its standard error
    :math:`\sqrt{p(1-p)/n}` depends only on the fraction and ``n``.

    Parameters
    ----------
    system : System
        A discrete map or continuous flow.
    region : Box, Ball, or Grid
        The measure to sample initial conditions from (uniform over a Box/Ball, or
        a Grid's bounding box).
    n : int, default 10000
        Number of random initial conditions.
    resolution : int or tuple of int, default 100
        Recurrence cells per axis (a Grid uses its own ``counts``).
    seed : int, optional
        Seed for the sampler (reproducible).
    dt : float, default 1.0
        Integration step between cell checks for a flow (ignored for a map).
    max_steps : int, default 10000
        Per-sample step cap before declaring divergence.
    merge_tol : float, optional
        Merge attractors whose centroids lie within this distance.  ``None`` uses
        two recurrence-cell diagonals; ``0`` disables it.
    **fsm
        Finite-state-machine thresholds forwarded to
        :class:`~tsdynamics.analysis.basins.attractors._AttractorMapper`.

    Returns
    -------
    BasinFractions
        Fractions per attractor, the diverged share, and the attractors.

    References
    ----------
    P. J. Menck, J. Heitzig, N. Marwan and J. Kurths, "How basin stability
    complements the linear-stability paradigm", *Nature Physics* **9**, 89 (2013).
    """
    cellgrid = _recurrence_grid(region, resolution)
    mapper = _AttractorMapper(system, cellgrid, dt=dt, max_steps=max_steps, **fsm)
    draw = sampler(region, seed=seed)

    n = int(n)
    counts: dict[int, int] = {}
    diverged = 0
    for _ in range(n):
        lab = mapper.map_ic(draw())
        if lab == DIVERGED:
            diverged += 1
        else:
            counts[lab] = counts.get(lab, 0) + 1

    merge = mapper.merge_map(resolve_merge_tol(cellgrid, merge_tol))
    merged_counts: dict[int, int] = {}
    for k, c in counts.items():
        cid = merge.get(k, k)
        merged_counts[cid] = merged_counts.get(cid, 0) + c

    fractions = {k: c / n for k, c in merged_counts.items()}
    attractors = mapper.attractor_set(diverged=diverged, seeds=n, merge=merge)
    return BasinFractions(fractions=fractions, diverged=diverged / n, n=n, attractors=attractors)
