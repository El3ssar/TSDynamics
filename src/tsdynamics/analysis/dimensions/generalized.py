r"""
Generalized (Rényi) dimensions :math:`D_q` by box counting.

Partition state space into a grid of boxes of side :math:`\epsilon` and let
:math:`p_i(\epsilon)` be the fraction of points in occupied box :math:`i`.  The
Rényi / generalized dimension spectrum (Hentschel & Procaccia, *Physica D*
**8**, 435, 1983; Grassberger, *Phys. Lett. A* **97**, 227, 1983) is

.. math::

    D_q = \frac{1}{q-1}\,
          \lim_{\epsilon \to 0} \frac{\log \sum_i p_i(\epsilon)^q}{\log \epsilon},
    \qquad
    D_1 = \lim_{\epsilon \to 0} \frac{\sum_i p_i \log p_i}{\log \epsilon}.

Special cases: :math:`D_0` is the box-counting (capacity) dimension, :math:`D_1`
the information dimension, :math:`D_2` the correlation dimension.  Each is the
slope of a partition ordinate against :math:`\log \epsilon` in the scaling
region.

Occupied boxes are found by integer-flooring the (shifted) coordinates and
taking unique rows, so cost scales with the number of *occupied* boxes, never
the exponential full-grid size.

**Grid-origin debiasing.**  A box count at a single, fixed grid origin is biased
by where the box boundaries happen to fall relative to the point set: a cluster
straddling a boundary is split across two boxes, inflating :math:`N(\epsilon)`
and, with it, the apparent dimension.  Following the standard minimal-cover
prescription, for every scale the partition is evaluated over several grid-origin
offsets and the one yielding the **fewest occupied boxes** — the offset closest
to the true covering number :math:`N(\epsilon)` — is kept, and that winning
cover's box counts feed every :math:`D_q`.  This removes the alignment bias and
flattens the :math:`D_q` spectrum of a self-similar monofractal (e.g. the
middle-thirds Cantor set, :math:`D_q = \log 2/\log 3` for all :math:`q`).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np

from ...errors import (
    ConvergenceError,
    InvalidInputError,
    InvalidParameterError,
    invalid_value,
    remedy,
)
from ._common import DimensionResult, _as_points, _diameter
from ._scaling import ScalingFit, fit_scaling_region

#: Rationale shared by the ``q < 0`` guards: a box-counting :math:`D_q` for
#: negative order is dominated by the rarely-visited, under-sampled boxes (small
#: :math:`p_i` raised to a negative power blows up), so the partition-function
#: estimate is unreliable and divergent in practice — the regime the fixed-mass
#: estimators were designed for instead (Badii & Politi 1985).
#:
#: **The negative-order branch is a documented gap, not a hidden one.**  The
#: estimator that *would* reach it — a fixed-mass moment estimator, Badii &
#: Politi's :math:`D_q` for :math:`q < 0` — is not implemented here:
#: :func:`~tsdynamics.analysis.dimensions.fixedmass.fixed_mass_dimension` takes
#: no ``q`` and returns the :math:`q \to 1` (information) fixed-mass dimension
#: only.  So no exported function in this library computes :math:`D_{q<0}`, and
#: the guard says so rather than pointing at an estimator that cannot deliver it.
_NEGATIVE_Q_RULE = "must be >= 0 for the box-counting estimator"
_NEGATIVE_Q_HINT = (
    "Negative-order Renyi dimensions are dominated by rarely-visited boxes, where "
    "the box-counting partition function is unreliable; restrict to q >= 0. "
    "TSDynamics has no estimator for q < 0 at all (fixed_mass_dimension takes no q), "
    "so this is a gap in the library, not just in this function."
)

#: Absolute slack allowed before a rise in :math:`D_q` with :math:`q` counts as a
#: violation of the (exact) non-increasing property.  Calibrated by measuring the
#: largest *spurious* rise over ``q = 0..5`` on the reference sets whose true
#: spectrum is flat or decreasing — uniform 1/2/3-cubes, the middle-thirds Cantor
#: set, the Sierpinski gasket and the Henon attractor: worst case +0.033 (the
#: uniform square), most of them under +0.010.  An under-resolved box count of the
#: Lorenz attractor rises by 0.14-0.25, so this value separates the two with
#: headroom on both sides.
_MONOTONE_ABS_TOL = 0.05

#: ...and the rise must additionally clear this many combined standard errors, so
#: a very precise fit is held to a correspondingly tighter standard.
_MONOTONE_SIGMA = 3.0

__all__ = [
    "NonMonotoneSpectrumWarning",
    "box_counting_dimension",
    "dimension_spectrum",
    "dimension_spectrum_plot_spec",
    "generalized_dimension",
    "information_dimension",
]


class NonMonotoneSpectrumWarning(UserWarning):
    r"""A computed :math:`D_q` spectrum rose with :math:`q`, which cannot happen.

    The Rényi dimensions are non-increasing in :math:`q` for *every* measure —
    it is an exact consequence of the power-mean inequality, not a rule of thumb.
    A computed spectrum that rises is therefore a statement about the estimate,
    not about the attractor: at least one order (in practice the small-\ :math:`q`
    end, which is dominated by the rarely-visited boxes) has not resolved.

    Emitted as this warning by default, with the estimate returned but marked
    ``trusted=False`` (its ``repr`` says ``UNTRUSTED``).  Refusing outright would
    make the estimator useless on exactly the systems people reach for — a Lorenz
    trajectory is under-resolved for :math:`D_0` at any realistic sample size —
    so the number is handed back with the violation stated, and cannot be mistaken
    for a resolved answer.  Pass ``strict=True`` to raise
    :class:`~tsdynamics.errors.ConvergenceError` instead.
    """


#: Default grid-origin offsets (as fractions of the box side ``eps``) swept to
#: debias the box count.  ``0.0`` is the naive origin-at-``mins`` partition; the
#: remaining shifts move the box boundaries so a boundary-straddling cluster is
#: not split.  Five offsets keep the per-scale cost bounded while reliably
#: finding a near-minimal cover.
_DEFAULT_OFFSETS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)


def _occupied_counts(coords: np.ndarray) -> np.ndarray:
    """Per-occupied-box point counts of integer cell coordinates ``(M, dim)``.

    Equivalent to ``np.unique(coords, axis=0, return_counts=True)[1]`` but groups
    on a single 1-D mixed-radix key instead of the structured ``axis=0`` lexsort,
    which is markedly faster on the large ``(M, dim)`` coordinate blocks here.
    Each cell row is offset to a non-negative ``(M, dim)`` grid and encoded as
    ``sum_d coord_d * prod(span_<d)`` — a bijection on the occupied cells, so the
    grouping (and hence the count *multiset*) is identical.  Only the counts are
    consumed downstream (their order is irrelevant to ``sum p**q`` /
    ``sum p log p`` and to the box *count* ``size``), so returning them in key
    order rather than lexicographic row order changes nothing.

    The encoding fits one ``int64`` key unless the product of per-axis spans would
    overflow; in that (extreme, high-dimension / huge-extent) case it falls back
    to the exact structured ``np.unique`` so the result is unconditionally
    preserved.
    """
    if coords.shape[1] == 1:
        return np.unique(coords[:, 0], return_counts=True)[1]
    cell = coords - coords.min(axis=0)
    spans = cell.max(axis=0).astype(object) + 1  # per-axis number of occupied levels
    # Mixed-radix place values; Python-int (object) arithmetic detects overflow.
    stride = 1
    overflow = False
    strides: list[int] = []
    for s in spans:
        strides.append(stride)
        stride *= int(s)
        if stride > np.iinfo(np.int64).max:
            overflow = True
            break
    if overflow:
        return np.unique(coords, axis=0, return_counts=True)[1]
    key = (cell.astype(np.int64) * np.asarray(strides, dtype=np.int64)).sum(axis=1)
    return np.unique(key, return_counts=True)[1]


def _occupancy(points: np.ndarray, eps: float, mins: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Point counts of the occupied boxes of side ``eps``.

    The grid origin sits at ``mins - offset * eps`` — i.e. the box boundaries are
    shifted *down* by ``offset`` box widths.  ``offset=0.0`` recovers the naive
    origin-at-``mins`` partition.
    """
    coords = np.floor((points - mins) / eps + offset).astype(np.int64)
    return _occupied_counts(coords)


def _min_cover_occupancy(
    points: np.ndarray,
    eps: float,
    mins: np.ndarray,
    offsets: tuple[float, ...] = _DEFAULT_OFFSETS,
) -> np.ndarray:
    r"""Occupancy of the alignment-debiased (minimal-cover) box partition.

    Sweeps the grid origin over ``offsets`` (fractions of ``eps``) and returns the
    occupied-box point counts of the offset with the **fewest occupied boxes** —
    the cover closest to the true covering number :math:`N(\epsilon)`, removing
    the bias of any single fixed grid origin.  The winning cover's counts are
    returned whole, so every :math:`D_q` is read from one consistent partition
    per scale.
    """
    best: np.ndarray | None = None
    for off in offsets:
        counts = _occupancy(points, eps, mins, off)
        if best is None or counts.size < best.size:
            best = counts
    assert best is not None  # offsets is non-empty
    return best


#: Default minimum mean occupancy (points per occupied box) for a scale to count
#: as informative — the **small**-box cut.  :math:`N(\epsilon)` can never exceed
#: the sample size, so once the boxes hold about one point each the curve bends
#: to a flat plateau of slope 0.  The pre-v6 mask admitted scales all the way to
#: ``0.85 * N`` boxes — 1.2 points per box, i.e. deep inside that plateau — which
#: is why a uniform 3-cube of 20k points returned :math:`D_0 = 2.76`.
#: Measured on the reference panel (see the module docstring), ``5`` recovers a
#: 3-cube best; ``10`` is materially worse there and no better elsewhere.
_DEFAULT_MIN_OCCUPANCY = 5.0

#: Default minimum **linear resolution** ``diam / eps`` — the number of boxes
#: spanning the set's largest extent — the **large**-box cut.  A cover only a few
#: boxes across measures the bounding box, not the geometry: its integer count is
#: lacunarity, and on the Lorenz attractor that tail (down to the pre-v6 floor of
#: 8 *occupied* boxes, which is 2-3 boxes per axis) pulled :math:`D_2` from 2.00
#: to 1.70.  A resolution floor is the dimension-free way to state that cut: a
#: floor on the occupied-box *count* means "3 boxes across" for a 3-D set and
#: "8 boxes across" for a 1-D one.
_DEFAULT_MIN_RESOLUTION = 6.0


def _informative_mask(
    occ: list[np.ndarray],
    n: int,
    scales: np.ndarray,
    diam: float,
    min_occupancy: float,
    min_resolution: float,
) -> np.ndarray:
    r"""Scales where the box partition carries dimension information.

    Drops the degenerate ends of the box-counting curve.  At the **large**-box
    end (:data:`_DEFAULT_MIN_RESOLUTION`) the cover spans only a few boxes across
    the set, so its count reflects the bounding box rather than the scaling.  At
    the **small**-box end (:data:`_DEFAULT_MIN_OCCUPANCY`) the partition has
    saturated toward one point per box, and the resulting flat plateau is exactly
    what a straightness-based scaling-region fit likes best.

    A partition is informative while ``diam / eps >= min_resolution`` and the
    *mean* number of points per occupied box is at least ``min_occupancy``.  Both
    criteria are stated in dimension-free units (a linear resolution, an
    occupancy) rather than as box counts or fractions of ``N``, which mean
    something different in every embedding dimension.

    Used for a caller-supplied ``scales=`` grid; the default grid is built
    informative by construction (:func:`_informative_scales`).
    """
    n_boxes = np.array([c.size for c in occ])
    resolution = diam / np.asarray(scales, dtype=float)
    return (resolution >= min_resolution) & (n_boxes * min_occupancy <= n)


#: Pilot sweep resolution (points per decade) used to locate the saturation end
#: of the informative band.  ~14/decade pins ``eps_lo`` to within a factor 1.2,
#: well inside the width of the band itself.
_PILOT_PER_DECADE = 14.0


def _informative_scales(
    points: np.ndarray,
    mins: np.ndarray,
    diam: float,
    *,
    n_scales: int,
    min_occupancy: float,
    min_resolution: float,
) -> np.ndarray:
    r"""Default box sizes: ``n_scales`` log-spaced across the *informative* band.

    The band is bounded above by the resolution floor and below by the saturation
    floor (see :func:`_informative_mask`).  Its upper end is exact —
    ``eps_hi = diam / min_resolution`` — while its lower end depends on the
    geometry through :math:`N(\epsilon)`, so it is found by a cheap pilot sweep of
    naive-origin box counts.  Since :math:`N(\epsilon)` is non-increasing in
    ``eps``, the smallest pilot ``eps`` meeting the occupancy floor is that end.

    Placing the grid *inside* the band, rather than spanning a fixed fraction of
    the diameter and masking afterwards, is what makes the estimator usable in
    more than one or two dimensions.  A fixed span wastes most of its scales on
    the saturated and unresolved tails: a 5000-point square then keeps 4
    informative box sizes out of 24 — fewer than ``min_window`` — and the
    estimate fails outright, while here all ``n_scales`` are informative.

    The pilot deliberately uses the naive origin only (no offset sweep): the
    minimal-cover partition used for the estimate itself has *fewer* occupied
    boxes at every scale, so a band accepted from the pilot satisfies the
    occupancy floor a fortiori.

    Raises
    ------
    ValueError
        If the band is empty *or collapses to a single scale* — the sample is too
        small to box-count a set of this
        dimension at any scale.  ``min_resolution ** D`` boxes are needed at the
        coarse end and only ``N / min_occupancy`` are affordable, so the two
        floors cross once ``D`` grows past ``log(N / min_occupancy) /
        log(min_resolution)``.
    """
    n = points.shape[0]
    eps_hi = diam / min_resolution
    # A 1-D set is the worst case: it needs N / min_occupancy boxes across, so no
    # set can saturate below this eps.  Start the pilot there.
    eps_floor = diam * min_occupancy / n
    decades = max(np.log10(eps_hi / eps_floor), 0.0)
    n_pilot = int(max(8, round(decades * _PILOT_PER_DECADE)))
    pilot = np.logspace(np.log10(eps_floor), np.log10(eps_hi), n_pilot)
    counts = np.array([_occupancy(points, e, mins).size for e in pilot], dtype=float)
    ok = counts * min_occupancy <= n
    # The band is unusable both when *no* pilot scale clears the occupancy floor
    # and when only the coarsest one does (``eps_lo == eps_hi``: a single point,
    # not a range).  Both have the same cause — the set's dimension is too high
    # for this sample — so both get the same actionable message.  The second case
    # is reachable: a uniform 3-cube of 2000 points hits it.
    eps_lo = float(pilot[int(np.argmax(ok))]) if bool(ok.any()) else eps_hi
    if not eps_lo < eps_hi:
        raise ValueError(
            f"no box size resolves this set: at the coarsest usable scale (diam/{min_resolution:g}) "
            f"the cover already needs more than N/{min_occupancy:g} = {n / min_occupancy:.0f} boxes "
            f"for N={n} points. The set's dimension is too high for this sample size — supply more "
            "data, or lower min_resolution / min_occupancy (and distrust the result)."
        )
    return np.logspace(np.log10(eps_lo), np.log10(eps_hi), n_scales)


#: Half-width of the ``q ~= 1`` band on which the entropy (limit) form of the
#: partition ordinate is used.  ``log(sum p^q) / (q - 1)`` is a 0/0 at ``q = 1``
#: and remains catastrophically ill-conditioned around it: the numerator is
#: ``O(q - 1)`` and is formed by cancellation, so at ``|q - 1| = 1e-8`` it still
#: carries ~8 significant digits while at ``1e-12`` it carries ~4.  The limit
#: ``sum p log p`` is exact and cheap, so it is used on the whole band.
_Q1_BAND = 1e-8


def _coerce_q(q: Any, *, analysis: str, param: str = "q") -> float:
    """Coerce the Renyi order to a float, rejecting a non-numeric ``q`` by name.

    Without this the comparison ``q < 0.0`` a line later raises NumPy's
    ``'<' not supported between instances of 'str' and 'float'`` — which names
    neither the parameter nor the call.
    """
    try:
        value = float(q)
    except (TypeError, ValueError) as err:
        raise InvalidInputError(
            f"{param} is the Renyi order of the dimension (a number: 0 for "
            f"box-counting, 1 for information, 2 for correlation), got {q!r}."
            + remedy(
                f"ts.{analysis}(data, {param}=[0.0, 1.0, 2.0])"
                if param == "qs"
                else f"ts.{analysis}(data, {param}=2.0)"
            )
        ) from err
    if not np.isfinite(value):
        raise invalid_value(
            param,
            q,
            rule="must be finite (the Renyi order)",
            hint="the q -> inf limit is not computed by this estimator.",
        )
    return value


def _partition_ordinate(counts: np.ndarray, n: int, q: float) -> float:
    r"""Ordinate whose slope vs :math:`\log\epsilon` is :math:`D_q`.

    Returns the information (entropy) form :math:`\sum_i p_i \log p_i` for
    :math:`q \approx 1` — the :math:`q \to 1` limit of the Rényi definition,
    which is a 0/0 there — and :math:`\log(\sum_i p_i^q)/(q-1)` otherwise.
    """
    p = counts / n
    if abs(q - 1.0) < _Q1_BAND:
        return float(np.sum(p * np.log(p)))
    return float(np.log(np.sum(p**q)) / (q - 1.0))


def _fit_masked(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, min_window: int, tol: float, what: str
) -> ScalingFit:
    """Fit a scaling region on the informative (masked) sub-range of a curve."""
    if int(mask.sum()) < min_window:
        raise ValueError(
            f"only {int(mask.sum())} informative box sizes for {what} (need >= {min_window}); the "
            "scales are either saturated (fewer than the required points per box) or too coarse "
            "(too few occupied boxes). Pass a wider/denser `scales`, lower `min_occupancy`, or "
            "supply more data."
        )
    return fit_scaling_region(x[mask], y[mask], min_window=min_window, tol=tol)


def _monotonicity_violations(
    spectrum: dict[float, DimensionResult],
) -> list[tuple[float, float, float, float]]:
    r"""Pairs ``(q_lo, q_hi, D_lo, D_hi)`` where the spectrum rises significantly.

    :math:`D_q` is non-increasing in :math:`q` for every measure, so any rise is
    estimator error.  A rise counts only when it clears both an absolute slack
    (:data:`_MONOTONE_ABS_TOL`) and :data:`_MONOTONE_SIGMA` combined standard
    errors, so ordinary finite-sample scatter on a monofractal does not trip it.
    """
    qs = sorted(spectrum)
    out: list[tuple[float, float, float, float]] = []
    for a in range(len(qs)):
        for b in range(a + 1, len(qs)):
            lo, hi = spectrum[qs[a]], spectrum[qs[b]]
            rise = float(hi.estimate) - float(lo.estimate)
            slack = max(
                _MONOTONE_ABS_TOL,
                _MONOTONE_SIGMA * float(np.hypot(lo.stderr, hi.stderr)),
            )
            if rise > slack:
                out.append((qs[a], qs[b], float(lo.estimate), float(hi.estimate)))
    return out


def _report_monotonicity(
    spectrum: dict[float, DimensionResult], *, strict: bool, what: str
) -> bool:
    r"""Raise (or warn) when a computed :math:`D_q` spectrum increases with :math:`q`.

    Returns ``True`` when the spectrum is admissible (the estimate resolved) and
    ``False`` when it was reported as unresolved via a warning.

    The estimate, not the attractor, is what such a spectrum describes: see
    :class:`NonMonotoneSpectrumWarning`.  Almost always the small-\ :math:`q` end
    is the culprit, because :math:`D_0` weights every occupied box equally and so
    is dominated by the rarely-visited parts of the support that a finite sample
    resolves worst.
    """
    bad = _monotonicity_violations(spectrum)
    if not bad:
        return True
    detail = "; ".join(f"D_{a:g}={da:.4g} < D_{b:g}={db:.4g}" for a, b, da, db in bad)
    message = (
        f"{what}: the computed Renyi spectrum increases with q ({detail}), which is impossible "
        "-- D_q is non-increasing in q for every measure. The estimate has not resolved: the "
        "small-q orders are dominated by rarely-visited boxes, so the box count is under-sampled "
        "at the scales used. Use a longer trajectory, restrict `scales` to the resolved range, "
        "or use correlation_dimension (D_2), which converges far faster."
    )
    message += (
        " Pass strict=False to downgrade this to a warning and return the (unreliable) numbers "
        "anyway."
        if strict
        else " The values are returned so you can see the violation; treat them as unresolved."
    )
    if strict:
        raise ConvergenceError(message)
    warnings.warn(message, NonMonotoneSpectrumWarning, stacklevel=3)
    return False


def _spectrum_core(
    points: np.ndarray,
    qs: list[float],
    *,
    scales: np.ndarray | None,
    n_scales: int,
    min_occupancy: float,
    min_resolution: float,
    min_window: int,
    flatness: float,
    offsets: tuple[float, ...],
    analysis: str,
) -> dict[float, DimensionResult]:
    r"""Estimate :math:`D_q` for several orders off one shared box partition.

    The box occupancies (and the minimal-cover grid origin chosen per scale) are
    computed once and reused across every ``q``, so a whole spectrum costs barely
    more than a single order and every order is read from the *same* partition —
    which is what makes the cross-``q`` monotonicity check meaningful.
    """
    n = points.shape[0]
    mins = points.min(axis=0)
    diam = _diameter(points)
    if diam <= 0.0:
        raise ValueError("degenerate point set: zero extent in every dimension.")
    if scales is None:
        scales = _informative_scales(
            points,
            mins,
            diam,
            n_scales=n_scales,
            min_occupancy=min_occupancy,
            min_resolution=min_resolution,
        )
    scales = np.asarray(scales, dtype=float)
    if np.any(scales <= 0.0):
        raise ValueError("box sizes (scales) must be positive.")

    scales = scales[np.argsort(scales)]
    occ = [_min_cover_occupancy(points, e, mins, offsets) for e in scales]
    x = np.log(scales)
    mask = _informative_mask(occ, n, scales, diam, min_occupancy, min_resolution)
    where = np.nonzero(mask)[0]

    out: dict[float, DimensionResult] = {}
    for q in qs:
        y = np.array([_partition_ordinate(c, n, q) for c in occ])
        fit = _fit_masked(x, y, mask, min_window=min_window, tol=flatness, what=f"D_{q:g}")
        out[q] = DimensionResult(
            estimate=fit.slope,
            stderr=fit.stderr,
            kind="generalized",
            abscissa=x,
            ordinate=y,
            fit_region=(int(where[fit.lo]), int(where[fit.hi])),
            intercept=fit.intercept,
            q=float(q),
            meta={
                "analysis": analysis,
                "kind": "generalized",
                "q": float(q),
                "n_components": int(points.shape[1]),
            },
        )
    return out


def generalized_dimension(
    data: Any,
    q: float = 2.0,
    *,
    scales: np.ndarray | None = None,
    n_scales: int = 20,
    min_occupancy: float = _DEFAULT_MIN_OCCUPANCY,
    min_resolution: float = _DEFAULT_MIN_RESOLUTION,
    min_window: int = 5,
    flatness: float = 1.5,
    offsets: tuple[float, ...] = _DEFAULT_OFFSETS,
) -> DimensionResult:
    r"""Generalized (Rényi) dimension :math:`D_q` by box counting.

    The one Rényi estimator: :func:`box_counting_dimension` is this at
    :math:`q = 0` and :func:`information_dimension` is this at :math:`q = 1`,
    both by delegation, so those names cost nothing to learn and can never give a
    different answer.  :math:`D_2` from *this* function is the Rényi
    :math:`q = 2`, which is a **different estimator** from
    :func:`correlation_dimension` (box occupancies vs pairwise distances);
    prefer that one for :math:`D_2` alone.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    q : float, default 2.0
        Rényi order.  ``q=0`` is box-counting, ``q=1`` information, ``q=2``
        correlation; non-integer ``q`` is allowed.  Must be ``>= 0`` — negative
        orders are rejected, and no estimator in this library computes them (see
        :data:`_NEGATIVE_Q_HINT`).
    scales : ndarray, optional
        Box sizes :math:`\epsilon`.  Default: ``n_scales`` log-spaced across the
        *informative* band — every one of them resolved and unsaturated by
        construction (:func:`_informative_scales`).  A supplied grid is instead
        trimmed to that band by :func:`_informative_mask`.
    n_scales : int, default 20
        Number of box sizes when ``scales`` is not given.
    min_occupancy : float, default 5.0
        Minimum mean number of points per occupied box for a scale to be used.
        This is the saturation guard: at fewer than a few points per box the box
        count is capped by the sample size rather than set by the geometry, and
        the curve flattens.  (It replaces the pre-v6 ``sat_frac``, which cut at a
        fraction of ``N`` and so meant something different in every dimension.)
    min_resolution : float, default 6.0
        Minimum linear resolution ``diam / eps`` — how many boxes span the set's
        largest extent — for a scale to be used.  This is the large-box guard: a
        cover a few boxes across measures the bounding box, not the geometry.
    min_window : int, default 5
        Minimum number of box sizes in the fitted scaling region.
    flatness : float, default 1.5
        How flat the fitted scaling region has to be: a window is admitted when
        its straight-line residual is within this factor of the flattest window
        found.  (It is **not** a solver tolerance — hence the v6 rename.)
    offsets : tuple of float, default ``(0.0, 0.2, 0.4, 0.6, 0.8)``
        Grid-origin offsets (as fractions of each box side) swept per scale; the
        offset with the fewest occupied boxes (the minimal cover) is kept, which
        removes the alignment bias of a single fixed grid origin.  Pass
        ``(0.0,)`` to recover the naive origin-at-minimum partition.

    Returns
    -------
    DimensionResult
        ``float(result)`` is :math:`D_q`.

    Notes
    -----
    This is the raw single-order estimator: it cannot check itself, because the
    only cheap internal consistency test on a box-counting dimension is that
    :math:`D_q` is non-increasing in :math:`q`, which needs more than one order.
    :func:`box_counting_dimension` and :func:`dimension_spectrum` do run that
    check.  Small :math:`q` — and :math:`q = 0` above all — converge slowly,
    because they weight the rarely-visited boxes a finite sample resolves worst.

    References
    ----------
    H. G. E. Hentschel and I. Procaccia, "The infinite number of generalized
    dimensions of fractals and strange attractors", *Physica D* **8**, 435
    (1983).

    Raises
    ------
    InvalidParameterError
        If ``q < 0``: the box-counting partition function is unreliable for
        negative orders (the rarely-visited boxes dominate).
    """
    q = _coerce_q(q, analysis="generalized_dimension")
    if q < 0.0:
        raise invalid_value("q", q, rule=_NEGATIVE_Q_RULE, hint=_NEGATIVE_Q_HINT)
    return _spectrum_core(
        _as_points(data, analysis="generalized_dimension"),
        [float(q)],
        scales=scales,
        n_scales=n_scales,
        min_occupancy=min_occupancy,
        min_resolution=min_resolution,
        min_window=min_window,
        flatness=flatness,
        offsets=offsets,
        analysis="generalized_dimension",
    )[float(q)]


def box_counting_dimension(data: Any, *, strict: bool = False, **kwargs: Any) -> DimensionResult:
    r"""Box-counting (capacity) dimension :math:`D_0`, with a self-consistency check.

    Numerically **identical** to ``generalized_dimension(data, 0.0)`` — it calls
    it — so there is one estimator here, under the name the literature uses for
    :math:`q = 0`.  Choose this one when :math:`D_0` is the quantity you want;
    choose :func:`generalized_dimension` when :math:`q` is a variable you are
    sweeping, and :func:`dimension_spectrum` when you want the whole curve.

    :math:`D_0` is the count of occupied boxes,
    :math:`N(\epsilon) \sim \epsilon^{-D_0}` — the :math:`q = 0` member of
    :func:`generalized_dimension`, and the *hardest* order to estimate: weighting
    every occupied box equally makes it dominated by the sparsely-visited parts
    of the support, which are exactly what a finite sample resolves last.  An
    under-resolved box count is biased **down**, and silently so.

    So this function does not just return :math:`D_0`.  It also computes
    :math:`D_1` and :math:`D_2` from the *same* box partition (free — the
    occupancies are already there) and checks the exact inequality
    :math:`D_0 \ge D_1 \ge D_2`.  A violation means the estimate has not
    resolved: the number is still returned, but marked ``trusted=False`` and
    accompanied by a :class:`NonMonotoneSpectrumWarning` naming the violation.  On an 8000-point Lorenz trajectory, for
    instance, this fires: :math:`D_0 = 1.75` against :math:`D_2 = 2.00`, and the
    :math:`D_0` end is the one that is wrong -- an independent box count of the
    same points puts the :math:`\log N(\epsilon)` slope at ~1.75 and the
    :math:`q = 2` slope at ~2.00, so it is box counting's notoriously slow
    convergence at :math:`q = 0`, not a property of the attractor.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    strict : bool, default False
        ``True`` raises :class:`~tsdynamics.errors.ConvergenceError` when the
        :math:`D_0 \ge D_1 \ge D_2` check fails.  The default warns
        (:class:`NonMonotoneSpectrumWarning`) and returns the number with
        ``trusted=False``.
    **kwargs
        Forwarded to :func:`generalized_dimension` (``scales``, ``n_scales``,
        ``min_occupancy``, ``min_resolution``, ``min_window``, ``flatness``, ``offsets``).

    Returns
    -------
    DimensionResult
        ``float(result)`` is :math:`D_0`.  Identical to
        ``generalized_dimension(data, 0.0, **kwargs)`` whenever the check passes.

    Raises
    ------
    ConvergenceError
        Only when ``strict=True`` and the shared-partition spectrum increases
        with ``q``.
    """
    spectrum = _spectrum_for_wrapper(data, [0.0, 1.0, 2.0], "box_counting_dimension", kwargs)
    resolved = _report_monotonicity(spectrum, strict=strict, what="box_counting_dimension")
    return dataclasses.replace(spectrum[0.0], trusted=resolved)


def information_dimension(data: Any, *, strict: bool = False, **kwargs: Any) -> DimensionResult:
    r"""Information dimension :math:`D_1`, with a self-consistency check.

    Numerically **identical** to ``generalized_dimension(data, 1.0)`` — it calls
    it — under the name the literature uses for :math:`q = 1`.  See
    :func:`box_counting_dimension` for when to reach for which spelling.

    The :math:`q \to 1` limit of the Rényi family: the slope of the Shannon
    information :math:`\sum_i p_i \log p_i` against :math:`\log \epsilon`.
    (The Rényi formula is a 0/0 at :math:`q = 1`; the entropy form is its exact
    limit and is what is evaluated — see :func:`_partition_ordinate`.)

    Like :func:`box_counting_dimension` this also computes the neighbouring
    orders off the same partition and checks :math:`D_0 \ge D_1 \ge D_2`.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    strict : bool, default True
        Raise on a non-monotone spectrum; ``False`` warns instead.
    **kwargs
        Forwarded to :func:`generalized_dimension` (``scales``, ``n_scales``,
        ``min_occupancy``, ``min_resolution``, ``min_window``, ``flatness``, ``offsets``).

    Returns
    -------
    DimensionResult
        ``float(result)`` is :math:`D_1`.

    Raises
    ------
    ConvergenceError
        If the shared-partition spectrum increases with ``q`` (``strict=True``).
    """
    spectrum = _spectrum_for_wrapper(data, [0.0, 1.0, 2.0], "information_dimension", kwargs)
    resolved = _report_monotonicity(spectrum, strict=strict, what="information_dimension")
    return dataclasses.replace(spectrum[1.0], trusted=resolved)


def _spectrum_for_wrapper(
    data: Any, qs: list[float], analysis: str, kwargs: dict[str, Any]
) -> dict[float, DimensionResult]:
    """Shared body of the ``q``-fixed wrappers: one partition, several orders.

    Keeps the wrappers' ``**kwargs`` contract identical to
    :func:`generalized_dimension` (an unknown keyword still raises ``TypeError``
    from the signature below) while routing through the shared core.

    The one keyword intercepted by hand is the renamed ``tol=``: forwarding it
    would surface a **private** function's name (``_core_kwargs() got an
    unexpected keyword argument 'tol'``), which is not a line anyone can act on.
    """
    if "tol" in kwargs:
        raise InvalidParameterError(
            f"{analysis}() has no 'tol' keyword: it was renamed 'flatness', because it is "
            "the flatness of the fitted scaling region, not a solver tolerance like rtol/atol.\n"
            f"    ts.analysis.{analysis}(data, flatness={kwargs['tol']!r})"
        )
    return _spectrum_core(
        _as_points(data, analysis=analysis), qs, analysis=analysis, **_core_kwargs(**kwargs)
    )


def _core_kwargs(
    *,
    scales: np.ndarray | None = None,
    n_scales: int = 20,
    min_occupancy: float = _DEFAULT_MIN_OCCUPANCY,
    min_resolution: float = _DEFAULT_MIN_RESOLUTION,
    min_window: int = 5,
    flatness: float = 1.5,
    offsets: tuple[float, ...] = _DEFAULT_OFFSETS,
) -> dict[str, Any]:
    """Validate/default the shared estimator keywords (one place, one signature)."""
    return {
        "scales": scales,
        "n_scales": n_scales,
        "min_occupancy": min_occupancy,
        "min_resolution": min_resolution,
        "min_window": min_window,
        "flatness": flatness,
        "offsets": offsets,
    }


def dimension_spectrum(
    data: Any,
    qs: Any = None,
    *,
    scales: np.ndarray | None = None,
    n_scales: int = 20,
    min_occupancy: float = _DEFAULT_MIN_OCCUPANCY,
    min_resolution: float = _DEFAULT_MIN_RESOLUTION,
    min_window: int = 5,
    flatness: float = 1.5,
    offsets: tuple[float, ...] = _DEFAULT_OFFSETS,
    strict: bool = False,
) -> dict[float, DimensionResult]:
    r"""Compute the :math:`D_q` spectrum over several Rényi orders.

    Computes box occupancies once per scale and reuses them across every ``q``,
    so the whole spectrum costs barely more than a single :math:`D_q`.  For a
    monofractal the spectrum is flat; a *decreasing* :math:`D_q` signals
    multifractality.

    An *increasing* :math:`D_q` signals nothing about the attractor at all —
    :math:`D_q` is non-increasing in :math:`q` for every measure — so it means the
    estimate has not resolved.

    ``strict`` defaults to ``False`` *here* and to ``True`` on the single-value
    estimators (:func:`box_counting_dimension`, :func:`information_dimension`), and
    the asymmetry is deliberate.  Asking for one number leaves the caller no way to
    see that it is unresolved, so those refuse rather than hand back a figure that
    looks authoritative.  Asking for the whole spectrum returns the very evidence
    of the failure — the rising :math:`D_q` values are right there in the result —
    so it warns (:class:`NonMonotoneSpectrumWarning`) and returns them, which is
    what makes the diagnosis possible.  Box counting is slow to converge at small
    :math:`q`, so this fires on ordinary inputs (a Lorenz trajectory, a Gaussian
    sample); :func:`correlation_dimension` is the right tool when it does.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    qs : array-like, optional
        Rényi orders.  Default: ``[0, 1, 2, 3, 4, 5]``.
    scales : ndarray, optional
        Box sizes; default as in :func:`generalized_dimension`.
    n_scales, min_occupancy, min_resolution, min_window, flatness, offsets
        As in :func:`generalized_dimension`.  The minimal-cover grid origin and
        the informative-scale mask are chosen once and shared across every ``q``,
        so the spectrum is read from one consistent, alignment-debiased partition
        per scale — which is what makes comparing orders legitimate.
    strict : bool, default False
        ``True`` raises :class:`~tsdynamics.errors.ConvergenceError` if the
        computed spectrum increases with ``q``.  The default warns
        (:class:`NonMonotoneSpectrumWarning`) and returns the numbers, because the
        rising values *are* the diagnosis — see the note above on why this differs
        from the single-value estimators.

    Returns
    -------
    dict[float, DimensionResult]
        ``{q: DimensionResult}`` in the order of ``qs``.

    Raises
    ------
    ConvergenceError
        If the computed spectrum increases with ``q`` (``strict=True``).
    """
    points = _as_points(data, analysis="dimension_spectrum")
    if qs is None:
        qs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    qs = [_coerce_q(q, analysis="dimension_spectrum", param="qs") for q in np.atleast_1d(qs)]
    bad = [q for q in qs if q < 0.0]
    if bad:
        raise invalid_value("q", bad[0], rule=_NEGATIVE_Q_RULE, hint=_NEGATIVE_Q_HINT)
    out = _spectrum_core(
        points,
        qs,
        scales=scales,
        n_scales=n_scales,
        min_occupancy=min_occupancy,
        min_resolution=min_resolution,
        min_window=min_window,
        flatness=flatness,
        offsets=offsets,
        analysis="dimension_spectrum",
    )
    _report_monotonicity(out, strict=strict, what="dimension_spectrum")
    return out


def dimension_spectrum_plot_spec(
    spectrum: dict[float, DimensionResult], kind: str | None = None
) -> Any:
    r"""Describe a :math:`D_q` spectrum as a backend-agnostic :class:`PlotSpec`.

    Renders the Rényi dimension spectrum returned by :func:`dimension_spectrum`
    — the estimated dimension :math:`D_q` against its Rényi order :math:`q` —
    rather than the per-order log--log scaling curves a single
    :class:`DimensionResult` plots.  The spec carries two layers:

    - a ``LINE`` of :math:`D_q` against :math:`q` (the spectrum profile: flat for
      a monofractal, monotonically decreasing for a multifractal);
    - an ``ERRORBAR`` over the same points whose ``"err"`` channel holds each
      :math:`D_q`'s standard error (:attr:`DimensionResult.stderr`).

    A backend draws the line and overlays the error bars at each :math:`q`.  The
    :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
    plotting library; this is a pure viz adapter and does not touch the
    estimator.

    Parameters
    ----------
    spectrum : dict of float to DimensionResult
        The ``{q: DimensionResult}`` mapping returned by
        :func:`dimension_spectrum`.  Iterated in ascending :math:`q`.
    kind : str, optional
        Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
        value).  ``None`` uses ``DIMENSION_SPECTRUM``.

    Returns
    -------
    PlotSpec
        A ``DIMENSION_SPECTRUM`` spec with a ``LINE`` and an ``ERRORBAR`` layer.

    Raises
    ------
    ValueError
        If ``spectrum`` is empty (no orders to plot).
    """
    from .. import _plotbuilder as pb

    if not spectrum:
        raise ValueError("dimension spectrum is empty: nothing to plot.")
    qs = np.array(sorted(spectrum), dtype=float)
    dq = np.array([float(spectrum[q].dimension) for q in qs], dtype=float)
    err = np.array([float(spectrum[q].stderr) for q in qs], dtype=float)
    return pb.spec(
        kind,
        "dimension_spectrum",
        layers=[
            pb.line(qs, dq, label=r"$D_q$"),
            pb.errorbar(qs, dq, err, label="std. error"),
        ],
        xlabel=r"$q$",
        ylabel=r"$D_q$",
        title=r"Rényi dimension spectrum $D_q$",
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
