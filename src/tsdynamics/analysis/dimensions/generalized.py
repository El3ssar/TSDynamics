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

from typing import Any

import numpy as np

from ...errors import invalid_value
from ._common import DimensionResult, _as_points, _default_scales
from ._scaling import ScalingFit, fit_scaling_region

#: Rationale shared by the ``q < 0`` guards: a box-counting :math:`D_q` for
#: negative order is dominated by the rarely-visited, under-sampled boxes (small
#: :math:`p_i` raised to a negative power blows up), so the partition-function
#: estimate is unreliable and divergent in practice — the regime the fixed-mass
#: estimators were designed for instead (Badii & Politi 1985).
_NEGATIVE_Q_RULE = "must be >= 0 for the box-counting estimator"
_NEGATIVE_Q_HINT = (
    "Negative-order Renyi dimensions are dominated by rarely-visited boxes, where "
    "the box-counting partition function is unreliable; restrict to q >= 0."
)

__all__ = [
    "box_counting_dimension",
    "dimension_spectrum",
    "dimension_spectrum_plot_spec",
    "generalized_dimension",
    "information_dimension",
]


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


def _informative_mask(occ: list[np.ndarray], n: int, sat_frac: float) -> np.ndarray:
    r"""Scales where the box partition carries dimension information.

    Drops the degenerate ends of the box-counting curve: large boxes that have
    collapsed to one or two cells (no slope), and small boxes that have
    saturated to (nearly) one point each — a perfectly straight *flat* plateau
    of slope :math:`\approx 0` that would otherwise capture the scaling-region
    fit and return a spurious near-zero dimension.  A box partition is
    informative while ``2 <= n_boxes <= sat_frac * N``.
    """
    n_boxes = np.array([c.size for c in occ])
    return (n_boxes >= 2) & (n_boxes <= sat_frac * n)


def _partition_ordinate(counts: np.ndarray, n: int, q: float) -> float:
    r"""Ordinate whose slope vs :math:`\log\epsilon` is :math:`D_q`.

    Returns :math:`\sum_i p_i \log p_i` for :math:`q = 1` and
    :math:`\log(\sum_i p_i^q)/(q-1)` otherwise.
    """
    p = counts / n
    if abs(q - 1.0) < 1e-12:
        return float(np.sum(p * np.log(p)))
    return float(np.log(np.sum(p**q)) / (q - 1.0))


def _fit_masked(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, min_window: int, tol: float, what: str
) -> ScalingFit:
    """Fit a scaling region on the informative (masked) sub-range of a curve."""
    if int(mask.sum()) < min_window:
        raise ValueError(
            f"only {int(mask.sum())} informative box sizes for {what} (need >= {min_window}); the "
            "scales are saturated or too coarse. Pass a wider/denser `scales` or more data."
        )
    return fit_scaling_region(x[mask], y[mask], min_window=min_window, tol=tol)


def generalized_dimension(
    data: Any,
    q: float = 2.0,
    *,
    scales: np.ndarray | None = None,
    n_scales: int = 18,
    sat_frac: float = 0.85,
    min_window: int = 5,
    tol: float = 1.5,
    offsets: tuple[float, ...] = _DEFAULT_OFFSETS,
) -> DimensionResult:
    r"""Generalized (Rényi) dimension :math:`D_q` by box counting.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    q : float, default 2.0
        Rényi order.  ``q=0`` is box-counting, ``q=1`` information, ``q=2``
        correlation; non-integer ``q`` is allowed.  Must be ``>= 0`` — negative
        orders are rejected (the box-counting estimator is unreliable there).
    scales : ndarray, optional
        Box sizes :math:`\epsilon`.  Default: a log-spaced grid spanning the
        attractor diameter
        (:func:`~tsdynamics.analysis.dimensions._common._default_scales`).
    n_scales : int, default 18
        Number of box sizes when ``scales`` is not given.
    sat_frac : float, default 0.85
        Drop scales whose occupied-box count exceeds ``sat_frac * N`` — the
        saturated small-box regime where each box holds about one point and the
        curve flattens spuriously.
    min_window : int, default 5
        Minimum number of box sizes in the fitted scaling region.
    tol : float, default 1.5
        Scaling-region residual tolerance.
    offsets : tuple of float, default ``(0.0, 0.2, 0.4, 0.6, 0.8)``
        Grid-origin offsets (as fractions of each box side) swept per scale; the
        offset with the fewest occupied boxes (the minimal cover) is kept, which
        removes the alignment bias of a single fixed grid origin.  Pass
        ``(0.0,)`` to recover the naive origin-at-minimum partition.

    Returns
    -------
    DimensionResult
        ``float(result)`` is :math:`D_q`.

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
    if q < 0.0:
        raise invalid_value("q", q, rule=_NEGATIVE_Q_RULE, hint=_NEGATIVE_Q_HINT)
    points = _as_points(data)
    n = points.shape[0]
    mins = points.min(axis=0)
    if scales is None:
        scales = _default_scales(points, n_scales=n_scales)
    scales = np.asarray(scales, dtype=float)
    if np.any(scales <= 0.0):
        raise ValueError("box sizes (scales) must be positive.")

    order = np.argsort(scales)
    scales = scales[order]
    occ = [_min_cover_occupancy(points, e, mins, offsets) for e in scales]
    x = np.log(scales)
    y = np.array([_partition_ordinate(c, n, q) for c in occ])
    mask = _informative_mask(occ, n, sat_frac)
    fit = _fit_masked(x, y, mask, min_window=min_window, tol=tol, what=f"D_{q:g}")
    where = np.nonzero(mask)[0]
    lo, hi = int(where[fit.lo]), int(where[fit.hi])
    return DimensionResult(
        estimate=fit.slope,
        stderr=fit.stderr,
        kind="generalized",
        abscissa=x,
        ordinate=y,
        fit_region=(lo, hi),
        intercept=fit.intercept,
        q=float(q),
        meta={"analysis": "generalized_dimension", "kind": "generalized", "q": float(q)},
    )


def box_counting_dimension(data: Any, **kwargs: Any) -> DimensionResult:
    r"""Box-counting (capacity) dimension :math:`D_0`.

    Thin wrapper over :func:`generalized_dimension` with ``q=0``.  Keyword
    arguments are forwarded.
    """
    return generalized_dimension(data, 0.0, **kwargs)


def information_dimension(data: Any, **kwargs: Any) -> DimensionResult:
    r"""Information dimension :math:`D_1`.

    Thin wrapper over :func:`generalized_dimension` with ``q=1``.  Keyword
    arguments are forwarded.
    """
    return generalized_dimension(data, 1.0, **kwargs)


def dimension_spectrum(
    data: Any,
    qs: Any = None,
    *,
    scales: np.ndarray | None = None,
    n_scales: int = 18,
    sat_frac: float = 0.85,
    min_window: int = 5,
    tol: float = 1.5,
    offsets: tuple[float, ...] = _DEFAULT_OFFSETS,
) -> dict[float, DimensionResult]:
    r"""Compute the :math:`D_q` spectrum over several Rényi orders.

    Computes box occupancies once per scale and reuses them across every ``q``,
    so the whole spectrum costs barely more than a single :math:`D_q`.  For a
    monofractal the spectrum is flat; a decreasing :math:`D_q` signals
    multifractality.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The point set.
    qs : array-like, optional
        Rényi orders.  Default: ``[0, 1, 2, 3, 4, 5]``.
    scales : ndarray, optional
        Box sizes; default as in :func:`generalized_dimension`.
    n_scales, sat_frac, min_window, tol, offsets
        As in :func:`generalized_dimension`.  The minimal-cover grid origin is
        chosen once per scale and shared across every ``q``, so the spectrum is
        read from one consistent, alignment-debiased partition per scale.

    Returns
    -------
    dict[float, DimensionResult]
        ``{q: DimensionResult}`` in the order of ``qs``.
    """
    points = _as_points(data)
    n = points.shape[0]
    mins = points.min(axis=0)
    if qs is None:
        qs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    qs = [float(q) for q in np.atleast_1d(qs)]
    bad = [q for q in qs if q < 0.0]
    if bad:
        raise invalid_value("q", bad[0], rule=_NEGATIVE_Q_RULE, hint=_NEGATIVE_Q_HINT)
    if scales is None:
        scales = _default_scales(points, n_scales=n_scales)
    scales = np.asarray(scales, dtype=float)
    if np.any(scales <= 0.0):
        raise ValueError("box sizes (scales) must be positive.")

    order = np.argsort(scales)
    scales = scales[order]
    occ = [_min_cover_occupancy(points, e, mins, offsets) for e in scales]
    x = np.log(scales)
    mask = _informative_mask(occ, n, sat_frac)
    where = np.nonzero(mask)[0]

    out: dict[float, DimensionResult] = {}
    for q in qs:
        y = np.array([_partition_ordinate(c, n, q) for c in occ])
        fit = _fit_masked(x, y, mask, min_window=min_window, tol=tol, what=f"D_{q:g}")
        out[q] = DimensionResult(
            estimate=fit.slope,
            stderr=fit.stderr,
            kind="generalized",
            abscissa=x,
            ordinate=y,
            fit_region=(int(where[fit.lo]), int(where[fit.hi])),
            intercept=fit.intercept,
            q=q,
            meta={"analysis": "dimension_spectrum", "kind": "generalized", "q": float(q)},
        )
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
    from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

    if not spectrum:
        raise ValueError("dimension spectrum is empty: nothing to plot.")
    spec_kind = PlotKind(kind) if kind is not None else PlotKind.DIMENSION_SPECTRUM
    qs = np.array(sorted(spectrum), dtype=float)
    dq = np.array([float(spectrum[q].dimension) for q in qs], dtype=float)
    err = np.array([float(spectrum[q].stderr) for q in qs], dtype=float)
    layers = [
        Layer(PlotKind.LINE, {"x": qs, "y": dq}, label=r"$D_q$"),
        Layer(PlotKind.ERRORBAR, {"x": qs, "y": dq, "err": err}, label="std. error"),
    ]
    return PlotSpec(
        kind=spec_kind,
        ndim=2,
        title=r"Rényi dimension spectrum $D_q$",
        x=Axis(label=r"$q$"),
        y=Axis(label=r"$D_q$"),
        layers=layers,
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
