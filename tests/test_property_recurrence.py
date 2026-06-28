"""
Property and known-value tests for the recurrence / RQA layer (stream I-QA).

Targets ``ts.recurrence_matrix`` / ``ts.rqa`` / ``ts.windowed_rqa`` and the
``RecurrenceMatrix`` / ``RQAResult`` / ``WindowedRQA`` result types.  The
recurrence matrix is the symmetric binary relation
:math:`R_{ij} = \\Theta(\\varepsilon - \\lVert x_i - x_j\\rVert)` (Eckmann,
Kamphorst & Ruelle 1987); RQA reduces its diagonal / vertical line structure to
scalar measures (Marwan et al. 2007).  The invariants asserted here are exact
structural facts (symmetry, Theiler exclusion, density monotonicity, target-rate
calibration, the ``[0, 1]`` bounds of the ratios) plus the qualitative
known-value separation a periodic orbit's near-perfect determinism gives over
white noise at a matched recurrence rate.

Matrix-building tests are point-cloud heavy, so they are capped tightly with
``@settings`` and kept to modest ``N``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _strategies import seeds, sinusoid, white_noise
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tsdynamics as ts

# Embedding parameters for a smooth point cloud out of a scalar series — enough
# delay to unfold a sinusoid into a clean loop, small enough to stay fast.
_EMB_DIM = 3
_EMB_TAU = 8


def _coords(rm) -> tuple[np.ndarray, np.ndarray]:
    """Stored (row, col) recurrence coordinates of a RecurrenceMatrix."""
    coo = rm.matrix.tocoo()
    return coo.row, coo.col


# ---------------------------------------------------------------------------
# recurrence_matrix — structural invariants
# ---------------------------------------------------------------------------


@settings(max_examples=10)
@given(
    seed=seeds,
    n=st.integers(min_value=150, max_value=350),
    dim=st.integers(min_value=2, max_value=5),
    eps=st.floats(min_value=0.5, max_value=4.0),
)
def test_matrix_is_square_symmetric_and_density_bounded(seed, n, dim, eps):
    """The matrix is (N, N), exactly symmetric, and RR lies in [0, 1]."""
    cloud = np.random.default_rng(seed).standard_normal((n, dim))
    rm = ts.recurrence_matrix(cloud, threshold=eps)
    assert rm.matrix.shape == (n, n)
    assert rm.size == n
    # Symmetric: R == R.T as a sparse predicate (no stored disagreements).
    assert (rm.matrix != rm.matrix.T).nnz == 0
    # Density is a genuine fraction.
    assert 0.0 <= rm.recurrence_rate <= 1.0


@settings(max_examples=10)
@given(
    seed=seeds,
    n=st.integers(min_value=150, max_value=350),
    target=st.sampled_from([0.05, 0.1, 0.2]),
)
def test_target_recurrence_rate_is_calibrated(seed, n, target):
    """recurrence_rate=target yields a realised RR within 0.05 of the target.

    The threshold is read off the empirical pair-distance quantile, so the
    realised density should track the request closely; allow 0.05 absolute for
    the discreteness of the distance distribution.
    """
    cloud = np.random.default_rng(seed).standard_normal((n, 3))
    rm = ts.recurrence_matrix(cloud, recurrence_rate=target)
    assert abs(rm.recurrence_rate - target) < 0.05


@settings(max_examples=10)
@given(seed=seeds, n=st.integers(min_value=120, max_value=300))
def test_recurrence_rate_monotone_in_threshold(seed, n):
    """A larger threshold can only add recurrence points: RR is non-decreasing."""
    cloud = np.random.default_rng(seed).standard_normal((n, 3))
    thresholds = [0.1, 0.3, 0.6, 1.0, 1.6, 2.5]
    rates = [ts.recurrence_matrix(cloud, threshold=t).recurrence_rate for t in thresholds]
    # Each successive (larger) threshold has >= the previous density.
    for lo, hi in zip(rates, rates[1:], strict=False):
        assert hi >= lo - 1e-12


@settings(max_examples=10)
@given(
    seed=seeds,
    n=st.integers(min_value=150, max_value=300),
    w=st.integers(min_value=1, max_value=12),
)
def test_theiler_window_excludes_near_diagonal(seed, n, w):
    """With theiler_window=w, no stored recurrence has |i - j| <= w.

    The line of identity and the |i-j| <= w band must carry no recurrence
    points (Theiler 1986) — otherwise line statistics are biased by trivially
    close, temporally adjacent samples.
    """
    # A smooth cloud so that without exclusion there *would* be near-diagonal
    # recurrences (adjacent samples of a flow are spuriously close).
    x = sinusoid(n, freq=0.02)
    pts = ts.embed(x, _EMB_DIM, _EMB_TAU)
    rm = ts.recurrence_matrix(pts, recurrence_rate=0.15, theiler=w)
    rows, cols = _coords(rm)
    if rows.size:
        assert int(np.abs(rows - cols).min()) > w
    assert rm.theiler_window == w


def test_theiler_window_zero_still_drops_line_of_identity():
    """theiler_window=0 keeps off-diagonal recurrences but never the diagonal."""
    x = sinusoid(300, freq=0.03)
    pts = ts.embed(x, _EMB_DIM, _EMB_TAU)
    rm = ts.recurrence_matrix(pts, recurrence_rate=0.2, theiler=0)
    rows, cols = _coords(rm)
    # The diagonal i == j is never stored.
    assert not np.any(rows == cols)


def test_matrix_requires_exactly_one_of_threshold_or_rate():
    """Passing neither / both threshold and recurrence_rate is rejected."""
    cloud = np.random.default_rng(0).standard_normal((100, 3))
    with pytest.raises(ValueError):
        ts.recurrence_matrix(cloud)
    with pytest.raises(ValueError):
        ts.recurrence_matrix(cloud, threshold=0.5, recurrence_rate=0.1)


# ---------------------------------------------------------------------------
# rqa — measure bounds
# ---------------------------------------------------------------------------


@settings(max_examples=10)
@given(
    seed=seeds,
    n=st.integers(min_value=150, max_value=350),
    target=st.sampled_from([0.05, 0.1, 0.2]),
)
def test_rqa_measures_are_bounded(seed, n, target):
    """DET, LAM, RR in [0, 1]; ENTR >= 0; and L_max >= 1 when DET > 0."""
    cloud = np.random.default_rng(seed).standard_normal((n, 3))
    res = ts.rqa(cloud, recurrence_rate=target)
    assert 0.0 <= res.recurrence_rate <= 1.0
    assert 0.0 <= res.determinism <= 1.0
    assert 0.0 <= res.laminarity <= 1.0
    assert res.diagonal_entropy >= 0.0
    # A non-empty determinism implies at least one counted diagonal line.
    if res.determinism > 0.0:
        assert res.max_diagonal_length >= 1
    # L_max / V_max are the longest *unfiltered* lines (Marwan et al. 2007): any
    # stored recurrence point lies on a diagonal AND a vertical line of length
    # >= 1, so the maxima must be >= 1 and DIV = 1/L_max must be finite — even
    # when every line is shorter than min_diagonal (DET == 0).
    if res.recurrence_rate > 0.0:
        assert res.max_diagonal_length >= 1
        assert res.max_vertical_length >= 1
        assert np.isfinite(res.divergence)
    else:
        assert res.max_diagonal_length == 0
        assert res.divergence == float("inf")


def test_rqa_accepts_prebuilt_matrix_unchanged():
    """rqa(RM) reuses the matrix and reproduces its recurrence rate."""
    x = sinusoid(300, freq=0.03)
    pts = ts.embed(x, _EMB_DIM, _EMB_TAU)
    rm = ts.recurrence_matrix(pts, recurrence_rate=0.15)
    res = ts.rqa(rm)
    # Same matrix => identical density and book-keeping.
    assert res.recurrence_rate == pytest.approx(rm.recurrence_rate)
    assert res.size == rm.size
    assert res.epsilon == pytest.approx(rm.epsilon)
    # Passing build args alongside a RecurrenceMatrix is an error.
    with pytest.raises(ValueError):
        ts.rqa(rm, recurrence_rate=0.1)


# ---------------------------------------------------------------------------
# Known qualitative separation: periodic determinism >> noise determinism
# ---------------------------------------------------------------------------


@settings(max_examples=8)
@given(
    seed=seeds,
    # Keep tau*freq well below 0.5 so the delay embedding unfolds the orbit into
    # a clean loop; a delay near half a period folds the reconstruction and
    # shortens the diagonals (an embedding artefact, not a DET regression).
    freq=st.floats(min_value=0.02, max_value=0.05),
)
def test_periodic_determinism_exceeds_noise(seed, freq):
    """A periodic orbit is far more deterministic than white noise.

    At a matched recurrence rate the diagonal-line fraction (DET) of an embedded
    sinusoid is near 1 (parallel-evolving segments), while embedded white noise
    has only short, accidental diagonals.  Assert a wide, robust margin.
    """
    n = 400
    periodic = ts.embed(sinusoid(n, freq=freq), _EMB_DIM, _EMB_TAU)
    noise = ts.embed(white_noise(n, seed=seed), _EMB_DIM, _EMB_TAU)
    rate = 0.1  # fixed so the comparison is at equal density
    det_p = ts.rqa(periodic, recurrence_rate=rate).determinism
    det_n = ts.rqa(noise, recurrence_rate=rate).determinism
    assert det_p > 0.9  # clean periodic signal
    # The separation is the primary invariant: noise DET stays ~0.2, so a 0.4
    # margin is robust to the seed and frequency while still meaningful.
    assert det_p > det_n + 0.4


# ---------------------------------------------------------------------------
# windowed_rqa — shape and bounds
# ---------------------------------------------------------------------------


@settings(max_examples=10)
@given(
    seed=seeds,
    window=st.integers(min_value=60, max_value=120),
    step=st.integers(min_value=20, max_value=80),
)
def test_windowed_rqa_window_count_and_bounds(seed, window, step):
    """n_windows = (N - window)//step + 1, and every DET/LAM/RR in [0, 1]."""
    n = 360
    # A regime-mixed series keeps each window non-degenerate.
    x = sinusoid(n, freq=0.03) + 0.4 * white_noise(n, seed=seed)
    pts = ts.embed(x, _EMB_DIM, _EMB_TAU)
    npts = pts.shape[0]
    assume(window <= npts)
    w = ts.windowed_rqa(pts, window=window, step=step, recurrence_rate=0.1)

    expected = (npts - window) // step + 1
    assert len(w) == expected
    assert w.centers.shape == (expected,)
    assert w.determinism.shape == (expected,)

    det = w.determinism
    lam = w.laminarity
    rr = w.recurrence_rate
    assert np.all((det >= 0.0) & (det <= 1.0))
    assert np.all((lam >= 0.0) & (lam <= 1.0))
    assert np.all((rr >= 0.0) & (rr <= 1.0))


def test_windowed_centers_are_increasing():
    """Window centres advance by exactly `step` and stay within the series."""
    x = sinusoid(400, freq=0.03)
    pts = ts.embed(x, _EMB_DIM, _EMB_TAU)
    w = ts.windowed_rqa(pts, window=100, step=40, recurrence_rate=0.1)
    centers = w.centers
    diffs = np.diff(centers)
    # Consecutive windows are `step` apart.
    assert np.allclose(diffs, 40.0)
    # Centres lie inside the valid range [(window-1)/2, N - (window+1)/2].
    assert centers[0] == pytest.approx((100 - 1) / 2.0)
    assert centers[-1] <= pts.shape[0] - 1
