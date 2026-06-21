r"""
Grid-origin debiasing of the box-counting / generalized dimensions
(ticket **FIX-BOXCOUNT-ORIGIN**).

A box count taken at a single, fixed grid origin is biased by where the box
boundaries fall relative to the point set: a cluster straddling a boundary is
split across two boxes, inflating :math:`N(\epsilon)`.  In the worst case the
count is doubled.  The fix sweeps the grid origin over several offsets per scale
and keeps the **minimal cover** (the offset with the fewest occupied boxes),
which is the cover closest to the true covering number and removes the alignment
bias.

These tests pin three things:

* the minimal-cover invariant — the debiased occupancy is never larger than any
  single-origin occupancy at the same scale (so debiasing only ever *reduces*
  the count);
* the headline acceptance — the middle-thirds Cantor set has a **flat**
  :math:`D_q` spectrum at :math:`\log 2/\log 3 \approx 0.6309` across :math:`q`
  on a structure-resolving scale grid;
* the failing-first regression — on a grid deliberately mis-aligned to the
  Cantor structure (where the alignment bias is scale-dependent) the debiased
  :math:`D_0` is closer to the known value than the naive single-origin
  estimate that the pre-fix code computed.

The Cantor set is generated from a finite refinement so the tests stay in the
fast tier and exercise only the estimators.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis import dimensions as dim
from tsdynamics.analysis.dimensions.generalized import (
    _DEFAULT_OFFSETS,
    _min_cover_occupancy,
    _occupancy,
)

_CANTOR_D0 = np.log(2.0) / np.log(3.0)  # ≈ 0.6309


def _cantor_points(n: int = 40000, depth: int = 16, seed: int = 3) -> np.ndarray:
    """Middle-thirds Cantor set as a column; uniform measure, D_q = log2/log3."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    scale = 1.0
    for _ in range(depth):
        scale /= 3.0
        x += (2.0 * scale) * rng.integers(0, 2, size=n)
    return x[:, None]


def _ternary_scales(points: np.ndarray, k_lo: int = 2, k_hi: int = 8) -> np.ndarray:
    """Box sizes aligned to the self-similar ternary structure of the set."""
    diam = float(points.max() - points.min())
    return diam * 3.0 ** (-np.arange(k_lo, k_hi + 1).astype(float))


# ── the minimal-cover invariant ─────────────────────────────────────────────────


def test_minimal_cover_never_exceeds_swept_single_origins():
    """The debiased occupancy has <= the boxes of every swept single origin.

    The minimal cover is the minimum over the *swept* offsets, so it is <= each
    of them — in particular <= the naive origin-at-minimum cover (``offset=0.0``,
    always swept), which is exactly the partition the pre-fix code used.
    """
    pts = _cantor_points()
    mins = pts.min(axis=0)
    diam = float(pts.max() - pts.min())
    for eps in diam * np.logspace(-3.0, -0.5, 10):
        mc = _min_cover_occupancy(pts, eps, mins).size
        assert mc <= _occupancy(pts, eps, mins, 0.0).size  # <= naive (pre-fix)
        for off in _DEFAULT_OFFSETS:
            assert mc <= _occupancy(pts, eps, mins, off).size


def test_default_offsets_include_naive_origin():
    """0.0 (origin at the data minimum) is one of the swept offsets."""
    assert 0.0 in _DEFAULT_OFFSETS
    assert len(_DEFAULT_OFFSETS) >= 2  # at least one shifted origin to debias against


def test_single_offset_recovers_naive_partition():
    """offsets=(0.0,) reproduces the pre-fix single-fixed-origin occupancy."""
    pts = _cantor_points()
    mins = pts.min(axis=0)
    eps = float(pts.max() - pts.min()) / 27.0
    naive = _occupancy(pts, eps, mins, 0.0)
    via_helper = _min_cover_occupancy(pts, eps, mins, (0.0,))
    assert np.array_equal(np.sort(naive), np.sort(via_helper))


# ── headline acceptance: flat Cantor spectrum at log2/log3 ───────────────────────


def test_cantor_spectrum_flat_across_q():
    """Cantor D_q is flat at log2/log3 across q with the debiased cover."""
    pts = _cantor_points()
    scales = _ternary_scales(pts)
    qs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    spec = dim.dimension_spectrum(pts, qs=qs, scales=scales, min_window=4)
    vals = np.array([float(spec[q]) for q in qs])
    spread = float(vals.max() - vals.min())
    assert spread < 0.02, f"Cantor D_q not flat: spread={spread:.4f}, D_q={vals}"
    assert np.all(np.abs(vals - _CANTOR_D0) < 0.02), (
        f"Cantor D_q off log2/log3={_CANTOR_D0:.4f}: {vals}"
    )


@pytest.mark.parametrize("q", [0.0, 2.0, 5.0])
def test_cantor_generalized_matches_spectrum(q):
    """Single-q generalized_dimension agrees with the spectrum entry (shared cover)."""
    pts = _cantor_points()
    scales = _ternary_scales(pts)
    one = float(dim.generalized_dimension(pts, q=q, scales=scales, min_window=4))
    assert abs(one - _CANTOR_D0) < 0.02, f"D_{q} = {one:.4f}"


# ── failing-first regression: debiasing reduces the alignment bias ───────────────


def test_debiasing_reduces_d0_bias_on_misaligned_grid():
    r"""On a grid mis-aligned to the Cantor structure the debiased :math:`D_0`
    beats the naive single-origin estimate that the pre-fix code returned.

    Fails on the old code, whose box count used the one fixed origin at the data
    minimum (equivalent to ``offsets=(0.0,)`` here).
    """
    pts = _cantor_points()
    diam = float(pts.max() - pts.min())
    # binary-spaced scales: each scale lands at a different phase vs the ternary
    # gaps, so the alignment over-count is scale-dependent and bends the slope.
    scales = diam * 2.0 ** (-np.arange(2, 11).astype(float))

    naive = float(dim.box_counting_dimension(pts, scales=scales, offsets=(0.0,), min_window=4))
    debiased = float(dim.box_counting_dimension(pts, scales=scales, min_window=4))

    err_naive = abs(naive - _CANTOR_D0)
    err_debiased = abs(debiased - _CANTOR_D0)
    assert err_debiased < err_naive, (
        f"debiased D0={debiased:.4f} (err {err_debiased:.4f}) not closer to "
        f"{_CANTOR_D0:.4f} than naive D0={naive:.4f} (err {err_naive:.4f})"
    )


# ── existing-behaviour guards: uniform sets keep integer dimensions ──────────────


@pytest.mark.parametrize("dim_d,expected", [(1, 1.0), (2, 2.0)])
def test_uniform_sets_unchanged_by_debiasing(dim_d, expected):
    """Debiasing does not disturb the integer dimension of a uniform set."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, (5000, dim_d))
    d = float(dim.box_counting_dimension(pts))
    assert abs(d - expected) < 0.12, f"D0 = {d:.3f}, expected {expected}"
