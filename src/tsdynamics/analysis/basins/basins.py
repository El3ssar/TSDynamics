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
from .._result import AnalysisResult
from ._common import (
    DIVERGED_COLOR,
    PALETTE,
    _apply_merge,
    _palette_indices,
    _recurrence_grid,
)
from .attractors import (
    DIVERGED,
    AttractorSet,
    _AttractorMapper,
    _reject_unsupported,
    classify_seeds,
    resolve_merge_tol,
)

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
class BasinsResult(AnalysisResult):
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

    labels: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int), repr=False, compare=False
    )
    grid: Grid | None = field(default=None, compare=False)
    attractors: AttractorSet = field(default_factory=AttractorSet, compare=False)

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

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe this basin diagram as a backend-agnostic :class:`PlotSpec`.

        Builds a ``BASINS_IMAGE`` spec — the integer label field as an image on
        an ``"equal"`` canvas, the grid axes giving the extent, plus a marker
        layer at the attractor representatives (for a 2-D image).  A **3-D slice**
        (a label cube with one degenerate ``counts == 1`` axis, as
        :func:`basins_of_attraction` paints when imaging a slice of a
        higher-dimensional flow) is squeezed to its two non-degenerate axes so it
        renders as a 2-D image; a genuinely 3-D label cube keeps all three axes
        on the spec.

        The image shares the attractor palette (``tab20``) with
        :meth:`AttractorSet.to_plot_spec`: the explicit ``{id: swatch index}``
        mapping is recorded in ``meta["palette_index"]`` (identical to the one the
        scatter carries), so a given attractor id is the same colour in both
        views; ``meta["palette"]`` / ``meta["diverged_color"]`` name the colormap
        and the fixed escape colour.  The :mod:`tsdynamics.viz.spec` import is
        lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"basins_image"``).  ``None`` uses
            ``BASINS_IMAGE``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Colorbar

        from .. import _plotbuilder as pb

        labels = np.asarray(self.labels)

        assert self.grid is not None  # a populated basin image always carries its grid
        lo, hi = self.grid.lo, self.grid.hi

        # Pick the two axes the image spans.  A degenerate (``counts == 1``) grid
        # axis is a pinned slice coordinate — drop it so a 3-D slice paints as a
        # plain 2-D image on its two free axes.
        axes = [a for a in range(labels.ndim) if labels.shape[a] > 1]
        if labels.ndim == 3 and len(axes) == 2:
            labels = np.squeeze(labels, axis=tuple(a for a in range(labels.ndim) if a not in axes))
        else:
            axes = list(range(min(labels.ndim, 2)))

        layers = [pb.image(labels, style={"cmap": PALETTE})]

        # Mark the attractor representatives on a 2-D image, projected onto the
        # two free axes.  Lower-dim grids skip the overlay.
        if labels.ndim == 2:
            centers = self.attractors.centers
            ax0, ax1 = (axes + [0, 1])[:2]
            if centers.size and centers.shape[1] > max(ax0, ax1):
                layers.append(
                    pb.markers(
                        centers[:, ax0],
                        centers[:, ax1],
                        label="attractors",
                        style={"marker": "*", "color": "black"},
                    )
                )

        ax0, ax1 = (axes + [0, 1])[:2]
        x_lim = (float(lo[ax0]), float(hi[ax0])) if lo.size > ax0 else None
        y_lim = (float(lo[ax1]), float(hi[ax1])) if lo.size > ax1 else None

        meta = dict(self.meta) if self.meta else {}
        meta.update(
            palette=PALETTE,
            diverged_color=DIVERGED_COLOR,
            palette_index=_palette_indices(self.attractors.ids),
        )
        return pb.spec(
            kind,
            "basins_image",
            layers=layers,
            aspect="equal",
            xlabel=f"x{ax0 + 1}",
            xlimits=x_lim,
            ylabel=f"x{ax1 + 1}",
            ylimits=y_lim,
            title=f"basins ({self.n_attractors} attractors)",
            colorbar=Colorbar(label="attractor", cmap=PALETTE, discrete=True),
            meta=meta,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"BasinsResult(shape={self.shape}, n_attractors={self.n_attractors}, "
            f"diverged={self.diverged_fraction:.3g})"
        )


@dataclass(frozen=True)
class BasinFractions(AnalysisResult):
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

    fractions: dict[int, float] = field(default_factory=dict, compare=False)
    diverged: float = 0.0
    n: int = 0
    attractors: AttractorSet = field(default_factory=AttractorSet, compare=False)

    @property
    def standard_error(self) -> dict[int, float]:
        r"""Binomial standard error :math:`\sqrt{p(1-p)/n}` per fraction."""
        return {k: float(np.sqrt(p * (1.0 - p) / self.n)) for k, p in self.fractions.items()}

    @property
    def dominant(self) -> int | None:
        """Id of the attractor with the largest basin (``None`` if all diverged)."""
        return max(self.fractions, key=self.fractions.__getitem__) if self.fractions else None

    def __getitem__(self, key: int) -> float:  # noqa: D105
        return self.fractions[key]

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the basin fractions as a backend-agnostic :class:`PlotSpec`.

        Builds a ``CATEGORICAL_BAR`` — one ``BAR`` per attractor id (plus a final
        bar for the diverged share when it is non-zero) over a categorical x-axis
        whose :attr:`~tsdynamics.viz.spec.Axis.categories` carry the labels
        (``attractor 1``, …, ``diverged``).  The ``"cat"`` channel holds the
        integer category index for each bar and ``"y"`` its basin fraction.  The
        bars are coloured from the shared attractor palette (``tab20``, recorded
        in ``meta["palette"]``), the diverged bar in the fixed diverged colour, so
        an id keeps its colour across the basin views.  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
        plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"categorical_bar"``).  ``None`` uses
            ``CATEGORICAL_BAR``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Colorbar

        from .. import _plotbuilder as pb

        ids = sorted(self.fractions)
        swatch = _palette_indices(ids)

        categories = [f"attractor {aid}" for aid in ids]
        heights = [float(self.fractions[aid]) for aid in ids]
        if self.diverged > 0.0:
            categories.append("diverged")
            heights.append(float(self.diverged))

        ticks = [float(i) for i in range(len(categories))]
        positions = np.asarray(ticks, dtype=float)
        layer = pb.bar(
            np.asarray(heights, dtype=float),
            cat=positions,
            label="basin fraction",
            style={"cmap": PALETTE},
        )
        meta = dict(self.meta) if self.meta else {}
        meta.update(
            palette=PALETTE,
            diverged_color=DIVERGED_COLOR,
            palette_index=swatch,
        )
        return pb.spec(
            kind,
            "categorical_bar",
            layers=[layer],
            xlabel="attractor",
            xscale="categorical",
            xcategories=categories,
            xticks=ticks,
            ylabel="basin fraction",
            ylimits=(0.0, 1.0),
            title="basin stability",
            colorbar=Colorbar(label="attractor", cmap=PALETTE, discrete=True),
            meta=meta,
        )

    def __repr__(self) -> str:  # noqa: D105
        body = ", ".join(f"{k}:{v:.3g}" for k, v in sorted(self.fractions.items()))
        return f"BasinFractions({{{body}}}, diverged={self.diverged:.3g}, n={self.n})"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def basins_of_attraction(
    system: Any,
    region: Grid,
    *,
    recurrence: Box | Grid | None = None,
    recurrence_resolution: int | tuple[int, ...] = 100,
    seed: int | None = 0,
    dt: float = 1.0,
    max_steps: int = 10000,
    merge_tol: float | None = None,
    **fsm: Any,
) -> BasinsResult:
    r"""
    Classify every point of a grid region by the attractor it converges to.

    Each lattice point is followed until it settles into a recurrent cell set
    (Datseris & Wagemakers, 2022); by default the region doubles as the recurrence
    tessellation, so labels accumulate and most points settle cheaply by reaching
    an already-labelled cell.

    For a higher-dimensional flow whose basins are viewed on a slice (e.g. a
    position grid of a system that also carries velocities), pass a full-dimension
    ``recurrence`` box covering the whole trajectory range and let ``region`` be the
    thin slice of initial conditions (free axes pinned with ``counts == 1``).

    Parameters
    ----------
    system : System
        A discrete map or continuous flow.
    region : Grid
        The lattice of initial conditions (full state dimension; a slice pins free
        axes with ``counts == 1``).  Build one with
        :func:`tsdynamics.data.region`.  Its ``counts`` set the recurrence
        resolution when ``recurrence`` is not given.
    recurrence : Box or Grid, optional
        Full-dimension region whose tessellation recurrences are detected on.
        Defaults to ``region`` itself (correct for maps and full grids; required
        when ``region`` is a degenerate slice).
    recurrence_resolution : int or tuple of int, default 100
        Recurrence cells per axis when ``recurrence`` is a Box.
    seed : int, optional
        Accepted for signature uniformity with :func:`find_attractors` /
        :func:`basin_fractions`; the full-grid scan is deterministic, so ``seed``
        does not change the labelling (it is recorded in provenance).
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

    Raises
    ------
    TypeError
        If ``system`` is a delay or stochastic system (unsupported by the
        recurrence finder).

    References
    ----------
    G. Datseris and A. Wagemakers, "Effortless estimation of basins of
    attraction", *Chaos* **32**, 023104 (2022).
    """
    _reject_unsupported(system, "basins_of_attraction")
    if recurrence is None:
        # region is a Grid → keeps its own counts (resolution arg is ignored).
        cellgrid = _recurrence_grid(region)
    else:
        cellgrid = _recurrence_grid(recurrence, recurrence_resolution)
    mapper = _AttractorMapper(system, cellgrid, dt=dt, max_steps=max_steps, **fsm)

    # Classify every lattice point.  On a supported engine run (an ODE flow / a map
    # whose ``_step`` lowers, ``interp`` / ``jit``) the whole grid marches in one
    # sequential Rust kernel call (stream ``perf/basin-march``) — bit-identical to,
    # and falling back on, the per-point Python loop.  ``grid_points`` order is the
    # classification order, so the shared labelling accumulates exactly as before.
    points = grid_points(region)
    from ...engine.run import resolve_backend

    backend = resolve_backend(getattr(system, "_default_backend", "interp"))
    labels = classify_seeds(mapper, points, backend=backend, jit=backend == "jit")
    diverged = int(np.sum(labels == DIVERGED))

    merge = mapper.merge_map(resolve_merge_tol(cellgrid, merge_tol))
    labels = _apply_merge(labels.reshape(region.shape), merge)
    attractors = mapper.attractor_set(diverged=diverged, seeds=points.shape[0], merge=merge)
    return BasinsResult(
        labels=labels,
        grid=region,
        attractors=attractors,
        meta=AnalysisResult.build_meta(system, analysis="basins_of_attraction", seed=seed),
    )


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

    Raises
    ------
    TypeError
        If ``system`` is a delay or stochastic system (unsupported by the
        recurrence finder).

    References
    ----------
    P. J. Menck, J. Heitzig, N. Marwan and J. Kurths, "How basin stability
    complements the linear-stability paradigm", *Nature Physics* **9**, 89 (2013).
    """
    _reject_unsupported(system, "basin_fractions")
    cellgrid = _recurrence_grid(region, resolution)
    mapper = _AttractorMapper(system, cellgrid, dt=dt, max_steps=max_steps, **fsm)
    draw = sampler(region, seed=seed)

    n = int(n)
    # Draw the whole sample up front (the sampler order — and so the labelling
    # order — is unchanged) and march it: one sequential Rust kernel call on a
    # supported engine run, else the per-sample Python loop (the oracle).  This
    # also accelerates :func:`continuation`, which sweeps ``basin_fractions``.
    samples = np.array([draw() for _ in range(n)], dtype=np.float64).reshape(-1, cellgrid.dim)
    from ...engine.run import resolve_backend

    backend = resolve_backend(getattr(system, "_default_backend", "interp"))
    labels = classify_seeds(mapper, samples, backend=backend, jit=backend == "jit")
    diverged = int(np.sum(labels == DIVERGED))
    counts: dict[int, int] = {}
    for lab in labels[labels != DIVERGED]:
        counts[int(lab)] = counts.get(int(lab), 0) + 1

    merge = mapper.merge_map(resolve_merge_tol(cellgrid, merge_tol))
    merged_counts: dict[int, int] = {}
    for k, c in counts.items():
        cid = merge.get(k, k)
        merged_counts[cid] = merged_counts.get(cid, 0) + c

    fractions = {k: c / n for k, c in merged_counts.items()}
    attractors = mapper.attractor_set(diverged=diverged, seeds=n, merge=merge)
    return BasinFractions(
        fractions=fractions,
        diverged=diverged / n,
        n=n,
        attractors=attractors,
        meta=AnalysisResult.build_meta(system, analysis="basin_fractions"),
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
