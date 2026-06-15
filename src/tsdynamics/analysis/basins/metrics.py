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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._common import as_label_array
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
class BasinEntropy:
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

    sb: float
    sbb: float
    n_boxes: int
    n_boundary_boxes: int
    box_size: int
    log_base: float
    fractal_boundary: bool

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"BasinEntropy(Sb={self.sb:.4g}, Sbb={self.sbb:.4g}, "
            f"fractal_boundary={self.fractal_boundary})"
        )


@dataclass(frozen=True)
class UncertaintyExponent:
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

    alpha: float
    boundary_dimension: float
    state_dimension: int
    epsilons: np.ndarray = field(repr=False)
    f: np.ndarray = field(repr=False)
    r_squared: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"UncertaintyExponent(alpha={self.alpha:.4g}, "
            f"D0={self.boundary_dimension:.4g}, R2={self.r_squared:.4g})"
        )


@dataclass(frozen=True)
class WadaResult:
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

    is_wada: bool
    n_basins: int
    radii: np.ndarray
    fractions: np.ndarray = field(repr=False)
    n_boundary_cells: int
    threshold: float

    def __repr__(self) -> str:  # noqa: D105
        w = self.fractions[-1] if self.fractions.size else float("nan")
        return f"WadaResult(is_wada={self.is_wada}, n_basins={self.n_basins}, W={w:.3g})"


# ---------------------------------------------------------------------------
# Basin entropy
# ---------------------------------------------------------------------------


def basin_entropy(basins: Any, *, box_size: int = 5, base: float = np.e) -> BasinEntropy:
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
        The basin image (attractor ids; ``-1`` = diverged is treated as a colour).
    box_size : int, default 5
        Box side length in cells.
    base : float, default e
        Logarithm base.  With the default natural log the fractal threshold is
        :math:`\log 2 \approx 0.693`.

    Returns
    -------
    BasinEntropy

    References
    ----------
    A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin and M. A. F. Sanjuán,
    "Basin entropy: a new tool to analyze uncertainty in dynamical systems",
    *Scientific Reports* **6**, 31416 (2016).
    """
    labels = as_label_array(basins)
    if box_size < 1:
        raise ValueError(f"box_size must be >= 1, got {box_size}")
    log = np.log(base)

    box_entropies: list[float] = []
    n_boundary = 0
    for block in _iter_blocks(labels, box_size):
        flat = block.reshape(-1)
        if flat.size == 0:
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
    )


def _iter_blocks(labels: np.ndarray, box_size: int):
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

    Returns
    -------
    UncertaintyExponent

    References
    ----------
    C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, "Final state sensitivity:
    an obstruction to predictability", *Physics Letters A* **99**, 415 (1983).
    """
    labels = as_label_array(basins)
    radii = tuple(int(r) for r in radii)
    if len(radii) < 2:
        raise ValueError("need at least two radii to fit a slope.")

    fractions = np.array([float(np.mean(_neighbor_differs(labels, m))) for m in radii])
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
    )


def _neighbor_differs(labels: np.ndarray, m: int) -> np.ndarray:
    """Boolean array: an axis-neighbour at distance ``m`` carries a different label."""
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
        out[lo_t] |= diff
        out[hi_t] |= diff
    return out


# ---------------------------------------------------------------------------
# Wada property
# ---------------------------------------------------------------------------


def wada_property(
    basins: Any,
    *,
    radii: tuple[int, ...] = (1, 2, 3, 4, 5),
    threshold: float = 0.9,
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

    Returns
    -------
    WadaResult

    References
    ----------
    A. Daza, A. Wagemakers, M. A. F. Sanjuán and J. A. Yorke, "Testing for basins
    of Wada", *Scientific Reports* **5**, 16579 (2015).
    """
    from scipy.ndimage import maximum_filter

    labels = as_label_array(basins)
    colors = [int(c) for c in np.unique(labels) if c >= 1]
    radii = tuple(int(r) for r in radii)
    boundary = _neighbor_differs(labels, 1)
    n_boundary = int(boundary.sum())

    if len(colors) < 3 or n_boundary == 0:
        return WadaResult(
            is_wada=False,
            n_basins=len(colors),
            radii=np.asarray(radii),
            fractions=np.zeros(len(radii)),
            n_boundary_cells=n_boundary,
            threshold=float(threshold),
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
    )


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def resilience(result: BasinsResult, attractor_id: int) -> float:
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
    float
        Distance from the attractor to its basin boundary, in state-space units.

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
    spacing = np.where(counts > 1, span / np.maximum(counts - 1, 1), 1.0)

    edt = distance_transform_edt(mask, sampling=spacing)

    center = result.attractors[int(attractor_id)].center
    idx = np.rint((center - grid.lo) / spacing).astype(int)
    idx = np.clip(idx, 0, np.asarray(grid.shape) - 1)
    return float(edt[tuple(idx)])
