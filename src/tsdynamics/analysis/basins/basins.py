r"""
Basins of attraction and basin fractions (basin stability).

Two complementary views of "which attractor wins from where":

- :func:`basins` paints a full grid — every lattice point is
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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...data import Ball, Box, Grid, grid_points, sampler
from ...errors import InvalidInputError, remedy
from .._result import AnalysisResult
from .._result_json import _pct, _sig
from ._common import (
    DIVERGED_COLOR,
    PALETTE,
    _apply_merge,
    _category_labels,
    _palette_indices,
    _recurrence_grid,
    _region_example,
    coerce_region,
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
    "basins",
]


#: Below this fraction of *labelled* seeds, a basin diagram is not an answer —
#: it is a report that nothing settled.  Set at 1.0 for the headline case (every
#: single seed failed), which is what a wrong ``recurrence=`` box produces.
_ALL_DIVERGED = 1.0


def _warn_if_nothing_settled(
    diverged: int, total: int, *, analysis: str, system: Any, cellgrid: Any
) -> None:
    """Say so when no seed reached an attractor, and name the likely cause.

    A 100 %-diverged run is the one place this library could hand back a
    confident, pretty, **empty** result in silence: measured, an undriven Duffing
    with its phase axis pinned returned ``0 basins · 100.0% diverged`` in 0.0 s
    with no warning and plotted a blank image.  Nothing had diverged — the pinned
    phase advanced straight out of the recurrence box on the first step.
    """
    if total <= 0 or diverged < total * _ALL_DIVERGED:
        return
    import warnings

    box = ", ".join(
        f"[{lo:.4g}, {hi:.4g}]" for lo, hi in zip(cellgrid.lo, cellgrid.hi, strict=True)
    )
    warnings.warn(
        f"{analysis}: every one of the {total} seeds left the recurrence box "
        f"without settling, so this result has 0 basins and is not a measurement "
        f"of {type(system).__name__}. The usual cause is a recurrence box the "
        f"dynamics leaves immediately — a monotone component (a drive phase, an "
        f"unbounded coordinate) has no attractor to recur to. "
        f"The box searched was {box}. Widen it with recurrence=, or exclude the "
        f"monotone component from the system.",
        RuntimeWarning,
        stacklevel=3,
    )


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
        """Fraction of grid CELLS in each attractor's basin.

        A **census of this image**: every cell of the lattice, counted once, with
        the diverged/lost share reported separately by :attr:`diverged_fraction`.

        This is *not* the same statistic as :func:`basin_fractions`, despite the
        shared word, and the two measurably disagree -- on Van der Pol,
        ``{1: 0.9922}`` here against ``{1: 1.0}`` there.  That function is Monte
        Carlo **basin stability** (Menck et al. 2013): an estimate, with a
        standard error, of the share of a *measure* over the region, from
        randomly drawn initial conditions.  This one is an exact count of a
        particular grid, so it depends on the grid and carries no error bar --
        and it includes cells that never settled, which the sampler drops.

        Use this to read off the picture you just drew; use
        :func:`basin_fractions` to estimate how likely a randomly perturbed state
        is to end up on each attractor.
        """
        ids, counts = np.unique(self.labels, return_counts=True)
        total = self.labels.size
        return {int(k): float(c) / total for k, c in zip(ids, counts, strict=True) if k != DIVERGED}

    @property
    def diverged_fraction(self) -> float:
        """Fraction of cells that diverged / never settled."""
        return float(np.mean(self.labels == DIVERGED))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the **label image** — the basin diagram is that integer field.

        ``np.asarray(basins(...))`` used to be a 0-d *object* array holding the
        result, which plots as nothing.
        """
        arr = np.asarray(self.labels)
        if dtype is not None:
            arr = arr.astype(dtype, copy=bool(copy))
        elif copy:
            arr = arr.copy()
        return arr

    def __plot_spec__(self, kind: str | None = None) -> Any:
        """Describe this basin diagram as a backend-agnostic :class:`PlotSpec`.

        Builds a ``BASINS_IMAGE`` spec — the integer label field as an image on
        an ``"equal"`` canvas, the grid axes giving the extent, plus a marker
        layer at the attractor representatives (for a 2-D image).  A **3-D slice**
        (a label cube with one degenerate ``counts == 1`` axis, as
        :func:`basins` paints when imaging a slice of a
        higher-dimensional flow) is squeezed to its two non-degenerate axes so it
        renders as a 2-D image; a genuinely 3-D label cube keeps all three axes
        on the spec.

        The image shares the attractor palette (``tab20``) with
        :meth:`AttractorSet.__plot_spec__`: the explicit ``{id: swatch index}``
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

        # An IMAGE channel is indexed ``[row, column]`` == ``[y, x]`` by every
        # backend (matplotlib's ``imshow``, plotly's ``Heatmap``), but the label
        # field is indexed ``[state axis 0, state axis 1]`` — and this spec puts
        # state axis ``ax0`` on **x** (it is what ``xlimits`` and the attractor
        # marker layer below both use).  Without the transpose the image is drawn
        # rotated a quarter turn relative to its own axis labels, limits and
        # markers: paint attractor 2 over ``x > 0.8`` and the figure shows a band
        # at ``y > 0.8`` with the attractor-2 star sitting in attractor 1's colour.
        # The two layers of one plot contradicting each other is the proof; the
        # defect was invisible while the axes were labelled ``x1`` / ``x2`` and
        # became publishable-wrong once they carry the system's real variable
        # names.
        image = labels.T if labels.ndim == 2 else labels
        layers = [pb.image(image, style={"cmap": PALETTE})]

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
            # The colour channel is an attractor *id*, not a quantity — name each
            # swatch so the colorbar reads as a categorical legend rather than a
            # numeric ramp over BoundaryNorm bin edges (0.5 / 1.5 / 2.5 ...).
            category_labels=_category_labels(np.asarray(self.labels)),
        )
        return pb.spec(
            kind,
            "basins_image",
            layers=layers,
            aspect="equal",
            xlabel=pb.axis_labels(self.meta, (ax0, ax1))[0],
            xlimits=x_lim,
            ylabel=pb.axis_labels(self.meta, (ax0, ax1))[1],
            ylimits=y_lim,
            title=f"basins ({self.n_attractors} attractors)",
            colorbar=Colorbar(label="attractor", cmap=PALETTE, discrete=True),
            meta=meta,
        )

    def _answer(self) -> str:
        """Return the grid size and every basin's share of it."""
        labels = np.asarray(self.labels)
        if not labels.size:
            return "empty basin image"
        shape = "×".join(str(int(n)) for n in labels.shape)
        shares = self.fractions
        parts = " · ".join(f"#{k} {_pct(v)}" for k, v in sorted(shares.items()) if k >= 1)
        n = self.n_attractors
        basins = f"{n} basin" + ("s" if n != 1 else "")
        body = f"{basins}: {parts}" if parts else basins
        return f"{shape} grid · {body} · {_pct(self.diverged_fraction)} diverged"

    def _details(self) -> tuple[str, ...]:
        """Say so when the image is a slice through a higher-dimensional space.

        A degenerate (``counts == 1``) grid axis is a *pinned* coordinate, and
        the picture is then a 2-D slice of an N-D basin structure, not the whole
        of it — which changes what the fractions mean.  Nothing else in the
        result says so.
        """
        grid = self.grid
        counts = getattr(grid, "counts", None)
        if grid is None or counts is None:
            return ()
        pinned = [i for i, c in enumerate(counts) if int(c) == 1]
        if not pinned:
            return ()
        names = self.meta.get("variables") if self.meta else None
        labels = tuple(names) if names else ()

        def _name(i: int) -> str:
            return str(labels[i]) if i < len(labels) else f"axis {i}"

        pins = ", ".join(f"{_name(i)} pinned at {_sig(np.asarray(grid.lo)[i], 4)}" for i in pinned)
        return (f"slice: {pins}",)

    def _derived(self) -> dict[str, Any]:
        """Export the shares and counts the repr reports."""
        return {
            "n_attractors": self.n_attractors,
            "fractions": self.fractions,
            "diverged_fraction": self.diverged_fraction,
        }


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

    def __len__(self) -> int:
        """Return how many attractors have a share (contract §4.2 rule 6)."""
        return len(self.fractions)

    def __iter__(self) -> Iterator[float]:
        """Iterate the shares themselves, in ascending id order."""
        return iter(float(self.fractions[k]) for k in self.ids)

    def __getitem__(self, key: Any) -> Any:
        """Return the basin fraction at **position** ``key`` (or a list, for a slice).

        Positional, like every other collection in the library (contract §4.2
        rule 6).  ``[]`` used to be an *id* lookup here — the exact defect v6
        fixed one file away for
        :class:`~tsdynamics.analysis.results.AttractorSet` ("ids start at 1, so
        ``aset[0]`` raised ``KeyError``"), left in place on its sibling: ``bf[0]``
        raised ``KeyError: 0``, and because there was no ``__iter__`` the legacy
        sequence protocol made ``list(bf)`` and ``for x in bf`` raise it too.
        Look a share up by its attractor label with :meth:`by_id`.
        """
        ordered = [float(self.fractions[k]) for k in self.ids]
        return ordered[key]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the shares as a ``(n_attractors,)`` float array, in id order."""
        arr = np.array([float(self.fractions[k]) for k in self.ids], dtype=float)
        return arr.astype(dtype, copy=bool(copy)) if dtype is not None else arr

    @property
    def ids(self) -> list[int]:
        """Sorted attractor ids — the order ``[]``, iteration and ``np.asarray`` use."""
        return sorted(self.fractions)

    def __plot_spec__(self, kind: str | None = None) -> Any:
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

    def by_id(self, key: int) -> float:
        """Return the basin fraction of the attractor **labelled** ``key``.

        ``[]`` is positional (the sequence convention every collection here
        follows); this is the explicit id lookup, the same pairing
        :class:`~tsdynamics.analysis.results.AttractorSet` uses.  The whole
        ``{id: share}`` mapping is still :attr:`fractions`.

        Raises
        ------
        KeyError
            If no attractor carries that id.
        """
        return float(self.fractions[int(key)])

    def _answer(self) -> str:
        """Return every attractor's sampled share, with the Monte-Carlo error."""
        err = self.standard_error
        parts = " · ".join(
            f"#{k} {_pct(v)} ± {_pct(err.get(k, 0.0), 1)}"
            for k, v in sorted(self.fractions.items())
        )
        body = parts or "no attractor found"
        return f"{body} · {_pct(self.diverged)} diverged"

    def _context(self) -> str | None:
        """Return the subject and how many initial conditions were sampled."""
        bits = [b for b in (self._system_label(),) if b]
        bits.append(f"{self.n} samples")
        return ", ".join(bits)

    def _derived(self) -> dict[str, Any]:
        """Export the dominant basin and the sampling error the repr reports."""
        return {"dominant": self.dominant, "standard_error": self.standard_error}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def basins(
    system: Any,
    region: Grid | Box | Ball | Sequence[tuple[float, ...]] | None = None,
    *,
    recurrence: Box | Grid | Sequence[tuple[float, ...]] | None = None,
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
        Accepted for signature uniformity with :func:`attractors` /
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

    Warns
    -----
    UserWarning
        When a located set fails the invariance audit and is discarded — the
        recurrence predicate mistook slow motion for convergence (``dt`` too
        small for the cell size).  Every located attractor is verified before it
        is returned; see
        :func:`~tsdynamics.analysis.basins.attractors.audit_attractors`.

    References
    ----------
    G. Datseris and A. Wagemakers, "Effortless estimation of basins of
    attraction", *Chaos* **32**, 023104 (2022).
    """
    _reject_unsupported(system, "basins")
    region = coerce_region(
        region,
        analysis="basins",
        system=system,
        want_grid=True,
    )
    if not isinstance(region, Grid):
        raise InvalidInputError(
            f"basins paints one label per lattice point, so region must "
            f"be a Grid (a {type(region).__name__} carries no resolution)."
            + remedy(
                "ts.analysis.basins(system, "
                f"{_region_example(int(getattr(system, 'dim', 2) or 2), triples=True)})",
                lead="Say how many initial conditions per axis:",
            )
        )
    if recurrence is None:
        # region is a Grid → keeps its own counts (resolution arg is ignored).
        cellgrid = _recurrence_grid(region)
    else:
        cellgrid = _recurrence_grid(
            coerce_region(
                recurrence,
                analysis="basins",
                system=system,
                want_grid=False,
            ),
            recurrence_resolution,
        )
    mapper = _AttractorMapper(system, cellgrid, dt=dt, max_steps=max_steps, **fsm)

    # Classify every lattice point.  On a supported engine run (an ODE flow / a map
    # whose ``_step`` lowers, ``interp`` / ``jit``) the whole grid marches in one
    # sequential Rust kernel call (stream ``perf/basin-march``) — bit-identical to,
    # and falling back on, the per-point Python loop.  ``grid_points`` order is the
    # classification order, so the shared labelling accumulates exactly as before.
    points = grid_points(region)
    from ...engine.run import resolve_backend

    backend = resolve_backend(getattr(system, "_default_backend", "jit"))
    labels = classify_seeds(mapper, points, backend=backend, jit=backend == "jit")
    diverged = int(np.sum(labels == DIVERGED))

    _warn_if_nothing_settled(
        diverged, points.shape[0], analysis="basins", system=system, cellgrid=cellgrid
    )

    merge = mapper.merge_map(resolve_merge_tol(cellgrid, merge_tol))
    labels = _apply_merge(labels.reshape(region.shape), merge)
    attractors = mapper.attractor_set(diverged=diverged, seeds=points.shape[0], merge=merge)
    return BasinsResult(
        labels=labels,
        grid=region,
        attractors=attractors,
        meta=AnalysisResult.build_meta(
            system,
            analysis="basins",
            seed=seed,
            variables=getattr(system, "variables", None),
        ),
    )


def basin_fractions(
    system: Any,
    region: Grid | Box | Ball | Sequence[tuple[float, ...]] | None = None,
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

    **Not the same statistic as** ``basins(system, region).fractions``, despite
    the shared word: that is an exact *census* of one grid image (every cell
    counted once, diverged cells included in the total), this is a Monte Carlo
    estimate over a *measure* with a standard error.  They measurably disagree —
    on Van der Pol, ``{1: 1.0}`` here against ``{1: 0.9922}`` there.  Ask this
    one "how likely is a random perturbation to land here?"; ask that one "what
    does this picture show?".

    Parameters
    ----------
    system : System
        A discrete map or continuous flow.
    region : Box, Ball, or Grid
        The measure to sample initial conditions from (uniform over a Box/Ball, or
        a Grid's bounding box).
    n : int, default 10000
        Number of random **initial conditions** drawn from ``region`` — the same
        quantity ``attractors`` / ``fixed_points`` / ``periodic_orbits`` spell
        ``n_seeds``.  The standard error of every reported fraction is
        :math:`\sqrt{p(1-p)/n}`, so this is the accuracy knob.
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

    Warns
    -----
    UserWarning
        When a located set fails the invariance audit and is discarded — the
        recurrence predicate mistook slow motion for convergence (``dt`` too
        small for the cell size).  Every located attractor is verified before it
        is returned; see
        :func:`~tsdynamics.analysis.basins.attractors.audit_attractors`.

    References
    ----------
    P. J. Menck, J. Heitzig, N. Marwan and J. Kurths, "How basin stability
    complements the linear-stability paradigm", *Nature Physics* **9**, 89 (2013).
    """
    _reject_unsupported(system, "basin_fractions")
    region = coerce_region(region, analysis="basin_fractions", system=system, want_grid=False)
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

    backend = resolve_backend(getattr(system, "_default_backend", "jit"))
    labels = classify_seeds(mapper, samples, backend=backend, jit=backend == "jit")
    diverged = int(np.sum(labels == DIVERGED))
    _warn_if_nothing_settled(
        diverged, int(n), analysis="basin_fractions", system=system, cellgrid=cellgrid
    )
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
