r"""
Quantifiers of a basin diagram.

These read a basin *image* — a labelled grid from
:func:`~tsdynamics.analysis.basins.basins.basins_of_attraction` (or a raw integer
array) — and need no further integration, so they are cheap and exact on a
synthetic label grid:

- :func:`basin_entropy` — the basin entropy :math:`S_b` and boundary basin
  entropy :math:`S_{bb}` of Daza et al. (2016); :math:`S_{bb} > \log 2` is a
  sufficient condition for a fractal boundary.
- :func:`uncertainty_exponent` — the final-state-sensitivity exponent
  :math:`\alpha` of Grebogi, McDonald, Ott & Yorke (1983):
  :math:`f(\varepsilon)\sim\varepsilon^{\alpha}`, with boundary dimension
  :math:`D_0 = D - \alpha`.
- :func:`wada_property` — a grid test for the Wada property (Daza et al., 2015):
  every boundary cell sees all basins.
- :func:`resilience` — the minimal-fatal-shock distance (Halekotte & Feudel,
  2020): how far an attractor sits from its basin boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._result import AnalysisResult, ScalarResult
from ._common import _as_label_array
from .basins import BasinsResult

__all__ = [
    "BasinEntropy",
    "UncertaintyExponent",
    "WadaResult",
    "basin_entropy",
    "resilience",
    "uncertainty_exponent",
    "wada_property",
]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasinEntropy(AnalysisResult):
    r"""
    Basin entropy and boundary basin entropy of a basin diagram.

    Attributes
    ----------
    sb : float
        Basin entropy :math:`S_b` — the mean Gibbs entropy over all boxes.
    sbb : float
        Boundary basin entropy :math:`S_{bb}` — the mean over boundary boxes only
        (those holding more than one basin).  ``nan`` if there is no boundary box.
    n_boxes : int
        Number of boxes the grid was partitioned into.
    n_boundary_boxes : int
        Number of boxes containing more than one basin.
    box_size : int
        Box side length in cells.
    log_base : float
        Base of the logarithm (``e`` by default).
    fractal_boundary : bool
        ``True`` when :math:`S_{bb} > \log 2`, the sufficient fractal-boundary
        criterion of Daza et al. (2016).
    """

    sb: float = 0.0
    sbb: float = 0.0
    n_boxes: int = 0
    n_boundary_boxes: int = 0
    box_size: int = 0
    log_base: float = 0.0
    fractal_boundary: bool = False

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"BasinEntropy(Sb={self.sb:.4g}, Sbb={self.sbb:.4g}, "
            f"fractal_boundary={self.fractal_boundary})"
        )


@dataclass(frozen=True)
class UncertaintyExponent(AnalysisResult):
    r"""
    The uncertainty exponent of a basin boundary.

    Attributes
    ----------
    alpha : float
        Uncertainty exponent :math:`\alpha` (slope of :math:`\log f` vs
        :math:`\log\varepsilon`); ``0`` = boundary fills the space (maximally
        unpredictable), ``1`` = smooth boundary.
    boundary_dimension : float
        Box-counting dimension of the boundary, :math:`D_0 = D - \alpha`.
    state_dimension : int
        State-space (grid) dimension :math:`D`.
    epsilons : ndarray
        Perturbation radii used (in state-space units).
    f : ndarray
        Fraction of :math:`\varepsilon`-uncertain cells at each radius.
    r_squared : float
        Coefficient of determination of the log-log fit.
    """

    alpha: float = 0.0
    boundary_dimension: float = 0.0
    state_dimension: int = 0
    epsilons: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    f: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    r_squared: float = 0.0

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the uncertainty exponent as its log--log scaling fit.

        The uncertainty exponent *is* a scaling estimate:
        :math:`f(\varepsilon)\sim\varepsilon^{\alpha}` (Grebogi et al., 1983), so
        the natural figure is the ``SCALING_FIT`` of :math:`\log f` against
        :math:`\log\varepsilon` with the fitted slope :math:`\alpha`.  This builds
        that spec directly — a ``SCATTER`` of the curve, the fit region marked,
        and the fit line drawn from the slope :math:`\alpha` and an intercept
        recovered from the curve mean — so a single ``result.plot.scaling()``
        renders it like every other dimension / Lyapunov-from-data scaling result.
        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``SCALING_FIT``; the
            ``.plot.scaling()`` seam passes ``"scaling_fit"`` explicitly, which
            resolves to the same kind.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        eps = np.asarray(self.epsilons, dtype=float)
        f = np.asarray(self.f, dtype=float)
        positive = (eps > 0.0) & (f > 0.0)
        log_eps = np.log(eps[positive])
        log_f = np.log(f[positive])

        layers = [pb.scatter(log_eps, log_f, label="curve")]
        if log_eps.size:
            # The fitted line: slope alpha, intercept recovered so it passes
            # through the curve's centroid (log_f ≈ intercept + alpha * log_eps).
            intercept = float(np.mean(log_f) - self.alpha * np.mean(log_eps))
            fit_x = np.array([log_eps.min(), log_eps.max()], dtype=float)
            layers.append(
                pb.line(fit_x, intercept + self.alpha * fit_x, label=f"slope = {self.alpha:.3g}")
            )
        return pb.spec(
            kind,
            "scaling_fit",
            layers=layers,
            xlabel=r"$\log\varepsilon$",
            ylabel=r"$\log f$",
            title=type(self).__name__,
            meta=self.meta,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"UncertaintyExponent(alpha={self.alpha:.4g}, "
            f"D0={self.boundary_dimension:.4g}, R2={self.r_squared:.4g})"
        )


@dataclass(frozen=True)
class WadaResult(AnalysisResult):
    r"""
    A grid test for the Wada property of a basin diagram.

    Attributes
    ----------
    is_wada : bool
        ``True`` when there are at least three basins and the fraction of
        boundary cells seeing *all* basins reaches ``threshold`` at the largest
        radius (a sufficient grid criterion, not a proof).
    n_basins : int
        Number of attractor basins (colours) considered.
    radii : ndarray
        Chebyshev radii tested.
    fractions : ndarray
        Fraction of boundary cells whose neighbourhood contains every basin, per
        radius (:math:`W` of Daza et al., 2015).
    n_boundary_cells : int
        Number of boundary cells.
    threshold : float
        Acceptance fraction at the largest radius.
    """

    is_wada: bool = False
    n_basins: int = 0
    radii: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)
    fractions: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    n_boundary_cells: int = 0
    threshold: float = 0.0

    def __repr__(self) -> str:  # noqa: D105
        w = self.fractions[-1] if self.fractions.size else float("nan")
        return f"WadaResult(is_wada={self.is_wada}, n_basins={self.n_basins}, W={w:.3g})"


# ---------------------------------------------------------------------------
# Basin entropy
# ---------------------------------------------------------------------------


def basin_entropy(
    basins: Any, *, box_size: int = 5, base: float = np.e, include_diverged: bool = False
) -> BasinEntropy:
    r"""
    Basin entropy :math:`S_b` and boundary basin entropy :math:`S_{bb}`.

    Partition the basin image into non-overlapping boxes of ``box_size`` cells per
    axis.  Within box :math:`i`, with colour fractions :math:`p_{ij}`, the Gibbs
    entropy is :math:`S_i = -\sum_j p_{ij}\log p_{ij}`.  Then
    :math:`S_b = \langle S_i\rangle` over all boxes and
    :math:`S_{bb} = \langle S_i\rangle` over boundary boxes (more than one
    colour).  :math:`S_{bb} > \log 2` is sufficient for a fractal boundary, since
    a box straddling a smooth boundary holds at most two colours.

    Parameters
    ----------
    basins : BasinsResult or array-like of int
        The basin image (attractor ids ``>= 1``; ``-1`` marks diverged / escape).
    box_size : int, default 5
        Box side length in cells.
    base : float, default e
        Logarithm base.  With the default natural log the fractal threshold is
        :math:`\log 2 \approx 0.693`.
    include_diverged : bool, default False
        If ``False`` (the default), diverged cells (``-1``) are dropped before the
        per-box colour count — escape is not a basin, so it must not inflate the
        entropy as a spurious extra colour.  A box that is *entirely* diverged then
        holds no settled basin and contributes zero entropy, but still counts in
        the box total :math:`N` (Daza et al. (2016) average :math:`S_b` over all
        :math:`N` boxes).  Set ``True`` to count ``-1`` as its own colour (the
        legacy behaviour).

    Returns
    -------
    BasinEntropy

    Raises
    ------
    ValueError
        If ``box_size < 1`` or the label array yields no boxes (it is empty).

    References
    ----------
    A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin and M. A. F. Sanjuán,
    "Basin entropy: a new tool to analyze uncertainty in dynamical systems",
    *Scientific Reports* **6**, 31416 (2016).
    """
    labels = _as_label_array(basins)
    if box_size < 1:
        raise ValueError(f"box_size must be >= 1, got {box_size}")
    log = np.log(base)

    box_entropies: list[float] = []
    n_boundary = 0
    for block in _iter_blocks(labels, box_size):
        flat = block.reshape(-1)
        if flat.size == 0:
            continue  # a zero-area block can only arise from an empty grid edge.
        if not include_diverged:
            flat = flat[flat != -1]  # escape is not a colour
            if flat.size == 0:
                # A fully diverged box holds no settled basin → zero Gibbs entropy.
                # Daza et al. (2016) normalise S_b over *all* N boxes, so an empty
                # box must still count in N (as a zero), not be skipped.
                box_entropies.append(0.0)
                continue
        _, counts = np.unique(flat, return_counts=True)
        p = counts / flat.size
        s = float(-np.sum(p * np.log(p)) / log)
        box_entropies.append(s)
        if counts.size > 1:
            n_boundary += 1

    if not box_entropies:
        raise ValueError("no boxes to analyse (empty label array).")

    entropies = np.asarray(box_entropies)
    sb = float(entropies.mean())
    boundary_mask = entropies > 0.0
    sbb = float(entropies[boundary_mask].mean()) if n_boundary else float("nan")
    threshold = np.log(2.0) / log
    fractal = bool(np.isfinite(sbb) and sbb > threshold)
    return BasinEntropy(
        sb=sb,
        sbb=sbb,
        n_boxes=len(box_entropies),
        n_boundary_boxes=n_boundary,
        box_size=int(box_size),
        log_base=float(base),
        fractal_boundary=fractal,
        meta={"analysis": "basin_entropy", "box_size": int(box_size)},
    )


def _iter_blocks(labels: np.ndarray, box_size: int) -> Iterator[np.ndarray]:
    """Yield every (possibly partial) ``box_size``-cell block of ``labels``."""
    from itertools import product

    ranges = [range(0, n, box_size) for n in labels.shape]
    for origin in product(*ranges):
        sl = tuple(slice(o, o + box_size) for o in origin)
        yield labels[sl]


# ---------------------------------------------------------------------------
# Uncertainty exponent
# ---------------------------------------------------------------------------


def uncertainty_exponent(
    basins: Any,
    *,
    radii: tuple[int, ...] = (1, 2, 3, 4, 5),
    cell_size: float | np.ndarray = 1.0,
    include_diverged: bool = False,
) -> UncertaintyExponent:
    r"""
    Estimate the uncertainty exponent of a basin boundary.

    A cell is :math:`\varepsilon`-uncertain when at least one axis-neighbour at
    distance :math:`\varepsilon` carries a different basin label.  The fraction of
    such cells scales as :math:`f(\varepsilon)\sim\varepsilon^{\alpha}`
    (Grebogi et al., 1983); :math:`\alpha` is the slope of a log-log fit and the
    boundary box-counting dimension is :math:`D_0 = D - \alpha`.

    Parameters
    ----------
    basins : BasinsResult or array-like of int
        The basin image.
    radii : tuple of int, default (1, 2, 3, 4, 5)
        Perturbation radii in cells.
    cell_size : float or array-like, default 1.0
        Physical cell spacing (a scalar or per-axis); rescales ``epsilons`` only —
        the exponent is invariant to it.
    include_diverged : bool, default False
        If ``False`` (the default), diverged cells (``-1``) are excluded: a cell is
        :math:`\varepsilon`-uncertain only when a same-distance neighbour carries a
        *different settled basin*, and the fraction is taken over settled cells.
        Escape is not a basin, so a basin/escape interface is not a final-state
        boundary.  Set ``True`` to treat ``-1`` as just another label.

    Returns
    -------
    UncertaintyExponent

    Raises
    ------
    ValueError
        If fewer than two ``radii`` are given, or fewer than two non-zero
        :math:`f(\varepsilon)` values are available to fit a slope (no boundary).

    References
    ----------
    C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, "Final state sensitivity:
    an obstruction to predictability", *Physics Letters A* **99**, 415 (1983).
    """
    labels = _as_label_array(basins)
    radii = tuple(int(r) for r in radii)
    if len(radii) < 2:
        raise ValueError("need at least two radii to fit a slope.")

    valid = None if include_diverged else (labels != -1)  # -1 == DIVERGED/escape
    fractions = np.array([_uncertain_fraction(labels, m, valid) for m in radii])
    spacing = float(np.mean(np.atleast_1d(cell_size)))
    epsilons = np.asarray(radii, dtype=float) * spacing

    positive = fractions > 0.0
    if positive.sum() < 2:
        raise ValueError("not enough non-zero f(epsilon) values to fit (no boundary?).")

    log_eps = np.log(epsilons[positive])
    log_f = np.log(fractions[positive])
    slope, intercept = np.polyfit(log_eps, log_f, 1)
    fit = slope * log_eps + intercept
    ss_res = float(np.sum((log_f - fit) ** 2))
    ss_tot = float(np.sum((log_f - log_f.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    alpha = float(slope)
    dim = labels.ndim
    return UncertaintyExponent(
        alpha=alpha,
        boundary_dimension=float(dim - alpha),
        state_dimension=dim,
        epsilons=epsilons,
        f=fractions,
        r_squared=float(r2),
        meta={"analysis": "uncertainty_exponent", "state_dimension": int(dim)},
    )


def _neighbor_differs(labels: np.ndarray, m: int, *, valid: np.ndarray | None = None) -> np.ndarray:
    """Boolean array: an axis-neighbour at distance ``m`` carries a different label.

    When ``valid`` (a boolean mask) is supplied, a pair only counts as differing if
    *both* cells are valid — so diverged / escape cells (``valid=False``) never
    create a spurious boundary against a settled basin.
    """
    out = np.zeros(labels.shape, dtype=bool)
    nd = labels.ndim
    for axis in range(nd):
        n = labels.shape[axis]
        if m >= n:
            continue
        lo = [slice(None)] * nd
        hi = [slice(None)] * nd
        lo[axis] = slice(0, n - m)
        hi[axis] = slice(m, n)
        lo_t, hi_t = tuple(lo), tuple(hi)
        diff = labels[lo_t] != labels[hi_t]
        if valid is not None:
            diff &= valid[lo_t] & valid[hi_t]
        out[lo_t] |= diff
        out[hi_t] |= diff
    return out


def _uncertain_fraction(labels: np.ndarray, m: int, valid: np.ndarray | None) -> float:
    r"""Fraction of :math:`\varepsilon`-uncertain cells at radius ``m``.

    With ``valid`` given (diverged excluded), the fraction is over *settled* cells
    only; otherwise over every cell (legacy behaviour).
    """
    differs = _neighbor_differs(labels, m, valid=valid)
    if valid is None:
        return float(np.mean(differs))
    if not valid.any():
        return 0.0
    return float(np.mean(differs[valid]))


# ---------------------------------------------------------------------------
# Wada property
# ---------------------------------------------------------------------------


def wada_property(
    basins: Any,
    *,
    radii: tuple[int, ...] = (1, 2, 3, 4, 5),
    threshold: float = 0.9,
    include_diverged: bool = False,
) -> WadaResult:
    r"""
    Test a basin diagram for the Wada property on a grid.

    With three or more basins, a boundary cell is "Wada-complete" at radius
    :math:`r` when its Chebyshev-:math:`r` neighbourhood contains *every* basin
    colour.  The fraction :math:`W(r)` of such boundary cells tends to one for a
    genuine Wada boundary (Daza et al., 2015).  This is a sufficient grid test,
    not a topological proof.

    Parameters
    ----------
    basins : BasinsResult or array-like of int
        The basin image.  Diverged cells (``-1``) are ignored when counting
        colours.
    radii : tuple of int, default (1, 2, 3, 4, 5)
        Chebyshev radii (in cells) at which to grow the neighbourhood.
    threshold : float, default 0.9
        Minimum :math:`W` at the largest radius to call the boundary Wada.
    include_diverged : bool, default False
        If ``False`` (the default), a basin/escape (``-1``) interface is not
        counted as a boundary cell — escape is not a basin.  Set ``True`` to let a
        cell bordering ``-1`` count as boundary.  Wada colours are always the
        settled basins (``>= 1``) regardless.

    Returns
    -------
    WadaResult

    Raises
    ------
    ValueError
        If ``basins`` carries non-integer labels (attractor ids must be integers,
        with ``-1`` marking escape).

    References
    ----------
    A. Daza, A. Wagemakers, M. A. F. Sanjuán and J. A. Yorke, "Testing for basins
    of Wada", *Scientific Reports* **5**, 16579 (2015).
    """
    from scipy.ndimage import maximum_filter

    labels = _as_label_array(basins)
    colors = [int(c) for c in np.unique(labels) if c >= 1]
    radii = tuple(int(r) for r in radii)
    valid = None if include_diverged else (labels != -1)  # -1 == DIVERGED/escape
    boundary = _neighbor_differs(labels, 1, valid=valid)
    n_boundary = int(boundary.sum())

    if len(colors) < 3 or n_boundary == 0:
        return WadaResult(
            is_wada=False,
            n_basins=len(colors),
            radii=np.asarray(radii),
            fractions=np.zeros(len(radii)),
            n_boundary_cells=n_boundary,
            threshold=float(threshold),
            meta={"analysis": "wada_property"},
        )

    presence = {c: (labels == c) for c in colors}
    fractions = []
    for r in radii:
        size = 2 * r + 1
        has_all = np.ones(labels.shape, dtype=bool)
        for c in colors:
            within = maximum_filter(presence[c], size=size, mode="constant", cval=False)
            has_all &= within
        fractions.append(float(has_all[boundary].mean()))

    frac_arr = np.asarray(fractions)
    is_wada = bool(frac_arr[-1] >= threshold)
    return WadaResult(
        is_wada=is_wada,
        n_basins=len(colors),
        radii=np.asarray(radii),
        fractions=frac_arr,
        n_boundary_cells=n_boundary,
        threshold=float(threshold),
        meta={"analysis": "wada_property"},
    )


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def resilience(result: BasinsResult, attractor_id: int) -> ScalarResult:
    r"""
    Minimal-fatal-shock resilience of an attractor: its distance to the boundary.

    The smallest perturbation that pushes the attractor out of its own basin,
    estimated as the state-space distance from the attractor's representative to
    the nearest cell of another basin (Halekotte & Feudel, 2020).  Larger means
    more resilient.

    Parameters
    ----------
    result : BasinsResult
        A basin image carrying its grid and attractors.
    attractor_id : int
        Which attractor (basin label) to measure.

    Returns
    -------
    ScalarResult
        Distance from the attractor to its basin boundary (behaves as a
        ``float``), in state-space units.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`BasinsResult` (the grid + attractors are
        required to measure a state-space distance).
    ValueError
        If the labels are not laid out on the grid (a pre-squeezed slice), or
        ``attractor_id`` is absent from the basin image.

    Notes
    -----
    A *sliced* basin image (a label cube with one or more degenerate ``counts == 1``
    axes, e.g. a position-plane slice of a higher-dimensional flow) is collapsed to
    its free axes before the distance transform, so a pinned axis does not inject a
    spurious one-cell-away boundary that would cap the reported distance.

    References
    ----------
    L. Halekotte and U. Feudel, "Minimal fatal shocks in multistable complex
    networks", *Scientific Reports* **10**, 11374 (2020).  Builds on the integral
    stability of C. Mitra, J. Kurths and R. V. Donner, *Scientific Reports* **5**,
    16196 (2015).
    """
    from scipy.ndimage import distance_transform_edt

    if not isinstance(result, BasinsResult):
        raise TypeError("resilience needs a BasinsResult (it requires the grid + attractors).")
    labels = result.labels
    grid = result.grid
    assert grid is not None  # a BasinsResult fed to resilience always carries its grid
    if labels.shape != tuple(grid.shape):
        raise ValueError(
            f"resilience needs labels laid out on the grid: labels {labels.shape} vs "
            f"grid {tuple(grid.shape)} (do not pre-squeeze a sliced basin image)."
        )
    mask = labels == int(attractor_id)
    if not mask.any():
        raise ValueError(f"attractor id {attractor_id} is absent from the basin image.")

    counts = np.asarray(grid.shape, dtype=float)
    span = grid.hi - grid.lo
    spacing_full = np.where(counts > 1, span / np.maximum(counts - 1, 1), 1.0)

    # Drop degenerate (``counts == 1``) axes — a pinned slice coordinate of a
    # higher-dimensional flow.  Padding/EDT over such an axis would inject a
    # spurious one-cell-away False border (the axis has only one layer), capping
    # the reported distance.  We mirror ``_as_label_array``: collapse the slice to
    # its effective dimension, keeping only the free axes for the distance field
    # and the matching grid origin / spacing entries.
    free = np.flatnonzero(np.asarray(grid.shape) > 1)
    if free.size == 0:  # fully degenerate grid (a single cell): no boundary at all.
        free = np.arange(labels.ndim)
    mask_free = np.squeeze(mask, axis=tuple(a for a in range(labels.ndim) if a not in free))
    lo_free = grid.lo[free]
    spacing = spacing_full[free]

    # Pad the basin mask with a one-cell False border so the computational-domain
    # edge is itself a boundary.  Without this, a basin that runs to the grid edge
    # has no nearby background there and the EDT reports the (large) distance to
    # the far interior boundary instead — *overestimating* the minimal fatal shock
    # (the domain simply ends; we cannot claim resilience past what was computed).
    padded = np.pad(mask_free, 1, mode="constant", constant_values=False)
    edt = distance_transform_edt(padded, sampling=spacing)
    shape = np.asarray(mask_free.shape)

    def _edt_at(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(distance-to-boundary, clipped cell index) for each state in ``points``."""
        pts_free = np.atleast_2d(np.asarray(points, dtype=float))[:, free]
        idx = np.clip(np.rint((pts_free - lo_free) / spacing).astype(int), 0, shape - 1)
        return edt[tuple((idx + 1).T)], idx

    # Minimal fatal shock = the closest approach of the attractor to its basin
    # boundary, i.e. the MINIMUM distance-to-boundary over the attractor's spatial
    # extent (its sampled point cloud) — an extended attractor (limit cycle /
    # strange set) can graze the boundary far from its single representative.
    att = result.attractors[int(attractor_id)]
    pts = np.atleast_2d(np.asarray(att.points, dtype=float))
    if pts.size:
        dists, idx = _edt_at(pts)
        on_basin = mask_free[tuple(idx.T)]  # ignore stray points outside the basin
        value = float(np.min(dists[on_basin])) if np.any(on_basin) else float("nan")
    else:
        value = float("nan")
    if not np.isfinite(value):  # empty / off-basin cloud → fall back to the centre
        dist, _ = _edt_at(np.atleast_2d(att.center))
        value = float(dist[0])
    return ScalarResult(
        value=value,
        meta={"analysis": "resilience", "attractor_id": int(attractor_id)},
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
