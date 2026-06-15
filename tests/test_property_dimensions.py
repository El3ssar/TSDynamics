"""
Property and known-value tests for the fractal-dimension API (stream I-QA).

These exercise ``tsdynamics``'s A-DIM estimators
(:func:`~tsdynamics.correlation_sum`, :func:`~tsdynamics.correlation_dimension`,
the generalized/Rényi :func:`~tsdynamics.generalized_dimension` with its
:func:`~tsdynamics.box_counting_dimension` / :func:`~tsdynamics.information_dimension`
wrappers, :func:`~tsdynamics.dimension_spectrum`, and
:func:`~tsdynamics.fixed_mass_dimension`) against *mathematical* invariants rather
than golden numbers:

- the correlation sum :math:`C(r)` is a CDF — non-decreasing in ``r`` and bounded
  in ``[0, 1]``;
- a uniform cloud in a ``d``-dimensional cube has correlation dimension ``~= d``
  (a real known value, not a tautology);
- the ``q=0`` / ``q=1`` wrappers *delegate* exactly to the generalized dimension;
- the generalized spectrum :math:`D_q` is non-increasing in ``q``;
- the fixed-mass estimator recovers the topological dimension of a uniform line
  and square.

Random clouds are seeded (``np.random.default_rng(seed)``) so any failing
Hypothesis example reproduces deterministically.  The point-cloud / k-d-tree tests
are expensive, so ``@settings(max_examples=...)`` caps the example count.
"""

from __future__ import annotations

import numpy as np
from _strategies import henon_series, seeds
from hypothesis import given, settings
from hypothesis import strategies as st

import tsdynamics as ts

# Tolerances are sized empirically (see commit message / brief): finite-N edge
# effects bias the correlation dimension of a unit cube *downward*, so the bands
# are generous-but-meaningful — they would still catch an off-by-one in the
# slope or a broken scaling-region fit.
_D2_TOL = 0.30  # uniform cube, d in {1, 2}: worst observed err ~0.15
# d = 3: the |D2 - 3| gap is a *systematic* finite-N downward bias (edge
# effects), not noise, and it shrinks with N.  We pin N = 3000 (below) so the
# bias is small and its seed-to-seed variance is tight; the empirical worst case
# there is ~0.30, well inside this 0.45 band.  The band is still meaningful: a
# broken slope fit lands the estimate near 2 or 4 (err ~1), which this catches.
_D2_TOL_3D = 0.45
_DELEGATE_TOL = 1e-9  # wrappers must be bit-for-bit the same call
_DQ_ORDER_TOL = 0.30  # D_q non-increasing, up to estimator noise on a finite orbit
_FIXED_MASS_TOL = 0.40  # topological dim of a uniform line/square


def _uniform_cube(d: int, n: int, seed: int) -> np.ndarray:
    """``n`` i.i.d. uniform points in the ``d``-dimensional unit cube."""
    return np.random.default_rng(int(seed)).uniform(0.0, 1.0, size=(int(n), int(d)))


# ---------------------------------------------------------------------------
# correlation_sum: C(r) is a bounded, non-decreasing CDF of pair distances
# ---------------------------------------------------------------------------


@settings(max_examples=12)
@given(seed=seeds, d=st.integers(min_value=1, max_value=3))
def test_correlation_sum_is_a_bounded_cdf(seed: int, d: int) -> None:
    """``correlation_sum`` returns ascending radii and a monotone ``C`` in [0, 1]."""
    pts = _uniform_cube(d, 1500, seed)
    radii, c = ts.correlation_sum(pts)

    # radii come back in the caller's order; the default grid is strictly ascending.
    assert np.all(np.diff(radii) > 0.0)

    # C is a fraction of pairs within r: it must live in [0, 1] everywhere.
    assert float(c.min()) >= 0.0
    assert float(c.max()) <= 1.0

    # C is the empirical CDF of pair distances -> non-decreasing as r grows.
    # (sort by radius first in case a caller-supplied grid were unsorted).
    order = np.argsort(radii)
    c_sorted = c[order]
    assert np.all(np.diff(c_sorted) >= -1e-12)


@settings(max_examples=10)
@given(seed=seeds)
def test_correlation_sum_monotone_in_explicit_radii(seed: int) -> None:
    """A coarser explicit radius grid still yields a CDF in [0, 1], non-decreasing."""
    pts = _uniform_cube(2, 1500, seed)
    radii = np.linspace(0.02, 0.5, 12)
    out_radii, c = ts.correlation_sum(pts, radii)
    assert np.allclose(out_radii, radii)
    assert float(c.min()) >= 0.0
    assert float(c.max()) <= 1.0
    assert np.all(np.diff(c) >= -1e-12)


# ---------------------------------------------------------------------------
# correlation_dimension: known value on a uniform d-cube
# ---------------------------------------------------------------------------


@settings(max_examples=8)
@given(
    seed=seeds,
    d=st.integers(min_value=1, max_value=2),
    n=st.integers(min_value=1500, max_value=3000),
)
def test_correlation_dimension_matches_cube_dimension(seed: int, d: int, n: int) -> None:
    """N uniform points in a d-cube (d in {1, 2}) give D2 ~= d within tolerance."""
    pts = _uniform_cube(d, n, seed)
    d2 = float(ts.correlation_dimension(pts))
    # This is the load-bearing known-value check: a broken slope fit would miss d.
    assert abs(d2 - d) <= _D2_TOL


@settings(max_examples=6)
@given(seed=seeds)
def test_correlation_dimension_3d_cube(seed: int) -> None:
    """A uniform 3-cube gives D2 ~= 3 (wider band: finite-N edge effects bite more).

    ``N`` is pinned to 3000 (the large end of the brief's range) rather than
    drawn: the d=3 underestimate is a *systematic* finite-N bias worst at small
    N, so a variable N would let Hypothesis keep drawing the most-biased,
    edge-of-tolerance cases across CI runs.  At a fixed large N the bias is small
    and the seed-to-seed spread is tight.
    """
    pts = _uniform_cube(3, 3000, seed)
    d2 = float(ts.correlation_dimension(pts))
    assert abs(d2 - 3.0) <= _D2_TOL_3D


@settings(max_examples=8)
@given(seed=seeds, d=st.integers(min_value=1, max_value=3))
def test_correlation_dimension_finite_and_nonnegative(seed: int, d: int) -> None:
    """For any non-degenerate cloud, D2 is finite and >= 0."""
    pts = _uniform_cube(d, 1800, seed)
    d2 = float(ts.correlation_dimension(pts))
    assert np.isfinite(d2)
    assert d2 >= 0.0


# ---------------------------------------------------------------------------
# generalized_dimension wrappers delegate exactly
# ---------------------------------------------------------------------------


@settings(max_examples=8)
@given(seed=seeds)
def test_box_counting_delegates_to_q0(seed: int) -> None:
    """box_counting_dimension(data) == generalized_dimension(data, q=0)."""
    pts = _uniform_cube(2, 2000, seed)
    bc = float(ts.box_counting_dimension(pts))
    g0 = float(ts.generalized_dimension(pts, q=0.0))
    assert abs(bc - g0) <= _DELEGATE_TOL


@settings(max_examples=8)
@given(seed=seeds)
def test_information_delegates_to_q1(seed: int) -> None:
    """information_dimension(data) == generalized_dimension(data, q=1)."""
    pts = _uniform_cube(2, 2000, seed)
    inf = float(ts.information_dimension(pts))
    g1 = float(ts.generalized_dimension(pts, q=1.0))
    assert abs(inf - g1) <= _DELEGATE_TOL


# ---------------------------------------------------------------------------
# dimension_spectrum: one result per requested q, q-tagged, finite
# ---------------------------------------------------------------------------


@settings(max_examples=8)
@given(seed=seeds)
def test_dimension_spectrum_keys_and_finiteness(seed: int) -> None:
    """dimension_spectrum(qs=[...]) returns one finite, correctly q-tagged result per q."""
    pts = _uniform_cube(2, 2000, seed)
    qs = [0.0, 1.0, 2.0, 3.0]
    spectrum = ts.dimension_spectrum(pts, qs=qs)

    # The spectrum maps each requested q -> its DimensionResult (in request order).
    assert list(spectrum.keys()) == qs
    for q in qs:
        result = spectrum[q]
        # The result is tagged with the q it was computed at...
        assert result.q == q
        # ...and every estimate is a finite number.
        assert np.isfinite(float(result))


# ---------------------------------------------------------------------------
# D_q is non-increasing in q on a (multifractal) chaotic attractor
# ---------------------------------------------------------------------------


@settings(max_examples=10)
@given(seed=st.integers(min_value=0, max_value=2_000))
def test_generalized_dimension_nonincreasing_in_q(seed: int) -> None:
    """On a Henon attractor (x reconstructed by delay embedding) D_q does not increase with q."""
    # A slightly perturbed initial condition per seed keeps the orbit on the
    # attractor while giving Hypothesis distinct (reproducible) draws.
    x0 = 0.05 + 0.001 * (seed % 50)
    series = henon_series(2500, x0=x0)
    cloud = ts.embed(series, 2, 1)  # 2-D Takens reconstruction of the attractor

    q_grid = [0.0, 2.0, 4.0]
    spectrum = ts.dimension_spectrum(cloud, qs=q_grid)
    dims = [float(spectrum[q]) for q in q_grid]

    # D_q is theoretically non-increasing in q; allow estimator noise via tol.
    for lo, hi in zip(dims[:-1], dims[1:], strict=True):
        assert lo >= hi - _DQ_ORDER_TOL


# ---------------------------------------------------------------------------
# fixed_mass_dimension recovers the topological dimension of a uniform set
# ---------------------------------------------------------------------------


@settings(max_examples=6)
@given(seed=seeds)
def test_fixed_mass_dimension_uniform_line(seed: int) -> None:
    """A uniform 1-D point set has fixed-mass dimension ~= 1."""
    pts = _uniform_cube(1, 3000, seed)
    d = float(ts.fixed_mass_dimension(pts))
    assert abs(d - 1.0) <= _FIXED_MASS_TOL


@settings(max_examples=6)
@given(seed=seeds)
def test_fixed_mass_dimension_uniform_square(seed: int) -> None:
    """A uniform 2-D point set has fixed-mass dimension ~= 2."""
    pts = _uniform_cube(2, 3000, seed)
    d = float(ts.fixed_mass_dimension(pts))
    assert abs(d - 2.0) <= _FIXED_MASS_TOL
