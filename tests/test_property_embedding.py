"""Property + known-value tests for the delay-embedding API (stream I-QA).

The delay-embedding toolkit (``ts.embed`` / ``ts.optimal_delay`` /
``ts.mutual_information`` / ``ts.autocorrelation`` / ``ts.cao_dimension`` /
``ts.false_nearest_neighbors`` / ``ts.embedding_dimension``) is the named I-QA
acceptance target.  These tests pin its *mathematical* contract rather than any
particular numeric output:

- :func:`embed` realises the Takens delay-coordinate map exactly — the right
  shape *and* the right values (row ``i`` is ``[x[i], x[i+tau], ...]``), and is a
  pure function of its input.
- :func:`autocorrelation` is a genuine normalised ACF (unit at lag 0, bounded by
  1 in magnitude) and :func:`mutual_information` is non-negative — both of the
  documented length.
- :func:`optimal_delay` returns a usable delay ``1 <= tau <= max_lag``.
- the dimension estimators return an integer dimension in ``[1, max_dim]`` with a
  false-neighbour fraction in ``[0, 1]``, and on a deterministic chaotic series
  the FNN fraction has fallen by the largest tested dimension (the geometric
  signature of an unfolded attractor).

Hypothesis drives the cheap shape/value/bound invariants; the slower
neighbour-search estimators run on a couple of fixed long chaotic series.
"""

from __future__ import annotations

import numpy as np
from _strategies import (
    finite_signals,
    henon_series,
    logistic_series,
    seeds,
    sinusoid,
)
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tsdynamics as ts

# ---------------------------------------------------------------------------
# embed — shape, value preservation, purity
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(
    n=st.integers(min_value=8, max_value=400),
    m=st.integers(min_value=1, max_value=8),
    tau=st.integers(min_value=1, max_value=12),
    seed=seeds,
)
def test_embed_shape_matches_takens_window(n: int, m: int, tau: int, seed: int) -> None:
    """``embed(x, m, tau)`` has exactly ``(N-(m-1)*tau, m)`` rows/cols.

    The window must leave at least two rows for a meaningful reconstruction;
    drop draws that don't (the estimator legitimately rejects them).
    """
    span = (m - 1) * tau
    assume(n - span >= 2)
    x = np.random.default_rng(seed).standard_normal(n)

    out = ts.embed(x, m, tau)

    assert out.shape == (n - span, m)


@settings(max_examples=20)
@given(
    n=st.integers(min_value=8, max_value=400),
    m=st.integers(min_value=1, max_value=8),
    tau=st.integers(min_value=1, max_value=12),
    seed=seeds,
)
def test_embed_preserves_values_exactly(n: int, m: int, tau: int, seed: int) -> None:
    """Row ``i`` equals ``[x[i], x[i+tau], ..., x[i+(m-1)*tau]]`` bit-for-bit.

    This is the defining property of the Takens map — verified against an
    independent reference construction over the whole matrix (no copies, no
    interpolation, no reordering).
    """
    span = (m - 1) * tau
    assume(n - span >= 2)
    x = np.random.default_rng(seed).standard_normal(n)
    rows = n - span

    out = ts.embed(x, m, tau)

    # Independent reference: stack the tau-shifted views column by column.
    reference = np.column_stack([x[j * tau : j * tau + rows] for j in range(m)])
    assert np.array_equal(out, reference)


@settings(max_examples=15)
@given(
    n=st.integers(min_value=16, max_value=300),
    m=st.integers(min_value=2, max_value=6),
    tau=st.integers(min_value=1, max_value=6),
    seed=seeds,
)
def test_embed_is_pure(n: int, m: int, tau: int, seed: int) -> None:
    """Same input → identical output (no hidden state, no in-place mutation)."""
    span = (m - 1) * tau
    assume(n - span >= 2)
    x = np.random.default_rng(seed).standard_normal(n)
    x_guard = x.copy()

    first = ts.embed(x, m, tau)
    second = ts.embed(x, m, tau)

    assert np.array_equal(first, second)
    # The input array itself must be untouched.
    assert np.array_equal(x, x_guard)


def test_embed_first_column_is_the_series_for_dimension_one() -> None:
    """``m=1`` is the trivial embedding: a single column equal to the series."""
    x = sinusoid(256, freq=0.03)
    out = ts.embed(x, 1, 5)
    assert out.shape == (x.size, 1)
    assert np.array_equal(out[:, 0], x)


# ---------------------------------------------------------------------------
# autocorrelation — normalised ACF contract
# ---------------------------------------------------------------------------


@settings(max_examples=30)
@given(x=finite_signals(min_size=64, max_size=400), max_lag=st.integers(min_value=1, max_value=40))
def test_autocorrelation_is_normalised_and_bounded(x: np.ndarray, max_lag: int) -> None:
    """``acf[0]==1``, ``|acf[k]| <= 1``, and the curve has length ``max_lag+1``.

    These are the defining properties of a *normalised* autocorrelation; a
    regression that forgot to divide by the zero-lag variance would break the
    first two, a windowing bug the third.  ``max_lag`` is clamped to ``N-1``.
    """
    acf = ts.autocorrelation(x, max_lag=max_lag)

    expected_len = min(max_lag, x.size - 1) + 1
    assert acf.shape == (expected_len,)
    # acf[0] is exactly the normalising variance / variance == 1.
    assert abs(acf[0] - 1.0) <= 1e-9
    # No autocorrelation can exceed 1 in magnitude (Cauchy–Schwarz).
    assert np.all(np.abs(acf) <= 1.0 + 1e-9)


def test_autocorrelation_known_sinusoid_is_periodic() -> None:
    """A cosine's ACF returns to ~1 one period later (a real known value).

    For ``cos(2*pi*f*t)`` the autocorrelation is itself ``~cos(2*pi*f*k)``, so at
    lag ``k = 1/f`` it comes back near +1.  Tolerance is generous because of the
    finite-length linear (non-circular) estimator.
    """
    freq = 0.05
    period = int(round(1.0 / freq))  # 20 samples
    x = sinusoid(2048, freq=freq)

    acf = ts.autocorrelation(x, max_lag=2 * period)

    # One full period later the ACF is close to its lag-0 value of 1.
    assert acf[period] > 0.9
    # A quarter period (near a zero crossing) it is far from 1.
    assert abs(acf[period // 4]) < 0.3


# ---------------------------------------------------------------------------
# mutual_information — non-negativity and length
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(x=finite_signals(min_size=80, max_size=400), max_lag=st.integers(min_value=1, max_value=30))
def test_mutual_information_is_nonnegative(x: np.ndarray, max_lag: int) -> None:
    """``I(tau) >= 0`` for every lag, and the curve has length ``max_lag+1``.

    Mutual information is non-negative by construction; the histogram estimator
    can only dip a hair below zero through float round-off, hence the ``-1e-12``
    floor.  ``max_lag`` is clamped to ``N-2``.
    """
    mi = ts.mutual_information(x, max_lag=max_lag)

    expected_len = min(max_lag, x.size - 2) + 1
    assert mi.shape == (expected_len,)
    assert np.all(mi >= -1e-12)


def test_mutual_information_self_lag_dominates() -> None:
    """``I(0)`` (the binned self-information) is the largest value of the curve.

    At lag 0 the pair ``(x_i, x_i)`` is perfectly dependent, so the lag-0 mutual
    information upper-bounds every lagged value — a real ordering invariant.
    """
    x = logistic_series(2000, r=4.0)
    mi = ts.mutual_information(x, max_lag=40)
    assert np.argmax(mi) == 0
    assert np.all(mi[1:] <= mi[0] + 1e-9)


# ---------------------------------------------------------------------------
# optimal_delay — usable delay in range, both methods
# ---------------------------------------------------------------------------


@given(method=st.sampled_from(["mi", "acf"]))
def test_optimal_delay_in_range(method: str) -> None:
    """``optimal_delay`` returns an int with ``1 <= tau <= max_lag``.

    Run on a long deterministic-chaotic series (``N >> max_lag``) so that
    ``max_lag`` is not the clamping bound and the documented upper limit holds.
    """
    x = henon_series(2000)
    max_lag = 50
    tau = ts.optimal_delay(x, method=method, max_lag=max_lag)

    assert isinstance(tau, int)
    assert 1 <= tau <= max_lag


def test_optimal_delay_mi_lands_in_the_first_minimum_valley() -> None:
    """For a cosine the MI delay sits in the first-minimum valley (known value).

    The time-delayed MI of ``cos(2*pi*f*t)`` falls from its lag-0 peak into a
    broad first minimum and rises back to a *peak* at the half period — there
    ``x_{i+T/2} = -x_i`` is perfectly (anti-)dependent.  Fraser--Swinney's first
    local minimum therefore lands at or past the quarter period and strictly
    before the half period; that whole valley is the correct delay region, so we
    assert membership of it rather than a single brittle sample.
    """
    freq = 0.02
    period = 1.0 / freq  # 50 samples
    quarter = period / 4.0  # 12.5 samples
    half = period / 2.0  # 25 samples
    x = sinusoid(4096, freq=freq)

    tau = ts.optimal_delay(x, method="mi", max_lag=int(period))

    # Inside the descending first-minimum valley: past the redundant short lags,
    # before the half-period MI peak where the signal becomes anti-correlated.
    assert quarter - 1.0 <= tau < half


# ---------------------------------------------------------------------------
# cao_dimension / false_nearest_neighbors / embedding_dimension — bounds
# ---------------------------------------------------------------------------

# A couple of fixed, reasonably long deterministic-chaotic series.  These keep
# the (slow) neighbour search off the Hypothesis hot path while still exercising
# the estimators on realistic data.
_CHAOTIC_SERIES = {
    "henon": henon_series(2500),
    "logistic": logistic_series(2500, r=4.0),
}


def _series(name: str) -> np.ndarray:
    return _CHAOTIC_SERIES[name]


@given(name=st.sampled_from(sorted(_CHAOTIC_SERIES)))
@settings(max_examples=2, deadline=None)
def test_cao_dimension_in_bounds(name: str) -> None:
    """Cao's estimate is an int in ``[1, max_dim]`` with finite ``E1`` / ``E2``."""
    max_dim = 8
    result = ts.cao_dimension(_series(name), delay=1, max_dim=max_dim)

    assert isinstance(result.dimension, int)
    assert 1 <= result.dimension <= max_dim
    # int(result) drops straight into embed → must agree with .dimension.
    assert int(result) == result.dimension
    assert result.afn_e1 is not None and result.afn_e1.shape == (max_dim,)
    assert np.all(np.isfinite(result.afn_e1))
    assert np.all(np.isfinite(result.afn_e2))


@given(name=st.sampled_from(sorted(_CHAOTIC_SERIES)))
@settings(max_examples=2, deadline=None)
def test_fnn_dimension_in_bounds_and_fraction_in_unit_interval(name: str) -> None:
    """FNN: integer dim in ``[1, max_dim]`` and every fraction in ``[0, 1]``."""
    max_dim = 8
    result = ts.false_nearest_neighbors(_series(name), delay=1, max_dim=max_dim)

    assert isinstance(result.dimension, int)
    assert 1 <= result.dimension <= max_dim
    assert result.fnn_fraction is not None and result.fnn_fraction.shape == (max_dim,)
    # A fraction must live in the unit interval.
    assert np.all(result.fnn_fraction >= 0.0)
    assert np.all(result.fnn_fraction <= 1.0)


def test_fnn_fraction_decays_on_deterministic_chaos() -> None:
    """On the Hénon attractor the FNN fraction *collapses* as the embedding unfolds.

    Kennel's construction: false neighbours (projection artefacts) are abundant
    in low dimensions and vanish once the attractor unfolds.  Hénon is a genuine
    2-D attractor, so a 1-D (scalar) reconstruction *is* a projection with many
    false neighbours, and adding coordinates resolves them — this is the case
    that actually exercises the decay.  (A 1-D map such as the logistic already
    has a zero fraction at dimension 1, so it cannot test the *drop* and is
    deliberately excluded here.)

    The assertion demands a real collapse, not mere non-increase: the
    dimension-1 fraction is large (a sizeable chunk of neighbours are false) and
    by the largest tested dimension it has fallen to under a tenth of that.  An
    all-zero or non-decaying curve — the exact regression an all-zeros FNN bug
    would produce — fails both halves.
    """
    max_dim = 8
    result = ts.false_nearest_neighbors(henon_series(2500), delay=1, max_dim=max_dim)
    frac = result.fnn_fraction

    assert 1 <= result.dimension <= max_dim
    # A scalar view of the 2-D Hénon attractor has abundant false neighbours...
    assert frac[0] > 0.1
    # ...that collapse by an order of magnitude once the attractor is unfolded.
    assert frac[-1] < 0.1 * frac[0]
    # The curve is non-increasing overall (generous tol absorbs finite-sample
    # neighbour-search jitter), the geometric signature of an unfolding embedding.
    assert frac[-1] <= frac[0] + 1e-9


def test_embedding_dimension_dispatches_to_both_methods() -> None:
    """``embedding_dimension`` honours ``method=`` and stays in bounds.

    Both backends must return an integer dimension in ``[1, max_dim]`` and tag
    the result with the method that produced it.
    """
    x = henon_series(2500)
    max_dim = 8

    cao = ts.embedding_dimension(x, method="cao", delay=1, max_dim=max_dim)
    fnn = ts.embedding_dimension(x, method="fnn", delay=1, max_dim=max_dim)

    assert cao.method == "cao" and 1 <= cao.dimension <= max_dim
    assert fnn.method == "fnn" and 1 <= fnn.dimension <= max_dim
