"""
Property + known-value tests for the entropy / complexity API (stream I-QA).

Every test asserts a *mathematical* invariant of the entropy estimators in
``tsdynamics.analysis.entropy`` — bounds, monotone-transform invariance, a
known limiting value, or a qualitative ordering between signals of differing
regularity.  Structured signals come from the deterministic builders in
``_strategies`` (Hénon / logistic / sinusoid / white noise); the pure
bound/shape invariants are driven with ``finite_signals``.

The correlation-integral statistics (sample/approximate entropy) are O(N^2),
so they run on short series with a tight Hypothesis budget.
"""

from __future__ import annotations

import numpy as np
from _strategies import (
    finite_signals,
    henon_series,
    logistic_series,
    seeds,
    sinusoid,
    white_noise,
)
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tsdynamics as ts

# ---------------------------------------------------------------------------
# Permutation entropy: bounds, ordinal invariance, known limits
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=64, max_size=384))
@settings(max_examples=40)
def test_permutation_entropy_in_unit_interval(x: np.ndarray) -> None:
    """Normalised permutation entropy is a probability-weighted Shannon ratio."""
    # A genuine constant draw is filtered out by finite_signals; guard anyway.
    assume(float(np.std(x)) > 0.0)
    h = ts.permutation_entropy(x, normalize=True)
    # Shannon entropy / log(m!) lies in [0, 1] for any input. Tiny float slack.
    assert -1e-12 <= h <= 1.0 + 1e-9


@given(
    seed=seeds,
    a=st.floats(min_value=0.1, max_value=50.0),
    b=st.floats(min_value=-100.0, max_value=100.0),
)
@settings(max_examples=30)
def test_permutation_entropy_affine_invariance(seed: int, a: float, b: float) -> None:
    """Strictly increasing affine maps (a>0) preserve ordinal patterns exactly."""
    x = henon_series(600, x0=0.1, y0=0.1, burn=50)
    base = ts.permutation_entropy(x)
    mapped = ts.permutation_entropy(a * x + b)
    # Order-preserving => identical ordinal-pattern distribution => identical PE.
    assert abs(base - mapped) <= 1e-9


@given(seed=seeds)
@settings(max_examples=20)
def test_permutation_entropy_exp_invariance(seed: int) -> None:
    """np.exp is strictly increasing, so it must leave permutation entropy fixed."""
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal(800)
    assume(float(np.std(x)) > 1e-6)
    base = ts.permutation_entropy(x)
    mapped = ts.permutation_entropy(np.exp(x))  # strictly monotone on all of R
    assert abs(base - mapped) <= 1e-9


def test_permutation_entropy_ramp_is_zero() -> None:
    """A strictly increasing ramp has a single ordinal pattern => PE ~= 0."""
    h = ts.permutation_entropy(np.arange(2000.0))
    # Only the all-ascending pattern occurs, so the entropy collapses to 0.
    assert abs(h) <= 1e-9


@given(seed=seeds)
@settings(max_examples=15)
def test_permutation_entropy_white_noise_is_high(seed: int) -> None:
    """Long white noise visits every ordinal pattern near-uniformly => PE > 0.9."""
    x = white_noise(4000, seed=int(seed))
    h = ts.permutation_entropy(x, normalize=True)
    # All m! patterns are near-equiprobable for iid noise; 0.9 is a loose floor.
    assert h > 0.9


@given(x=finite_signals(min_size=64, max_size=384))
@settings(max_examples=30)
def test_weighted_permutation_entropy_in_unit_interval(x: np.ndarray) -> None:
    """Weighted PE is a (variance-weighted) Shannon ratio => still in [0, 1]."""
    assume(float(np.std(x)) > 0.0)
    h = ts.weighted_permutation_entropy(x, normalize=True)
    assert -1e-12 <= h <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Dispersion entropy: bounds
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=64, max_size=384))
@settings(max_examples=30)
def test_dispersion_entropy_in_unit_interval(x: np.ndarray) -> None:
    """Normalised dispersion entropy is a Shannon ratio over c**m classes."""
    assume(float(np.std(x)) > 0.0)
    h = ts.dispersion_entropy(x, normalize=True)
    assert -1e-12 <= h <= 1.0 + 1e-9


@given(seed=seeds)
@settings(max_examples=15)
def test_dispersion_entropy_white_noise_is_high(seed: int) -> None:
    """White noise spreads across dispersion classes => normalised DispEn high."""
    x = white_noise(4000, seed=int(seed))
    h = ts.dispersion_entropy(x, normalize=True)
    # iid Gaussian fills the c amplitude classes near-uniformly.
    assert h > 0.8


# ---------------------------------------------------------------------------
# Sample / approximate entropy: non-negativity (O(N^2) => short series)
# ---------------------------------------------------------------------------


@given(seed=seeds)
@settings(max_examples=10)
def test_sample_entropy_nonnegative(seed: int) -> None:
    """SampEn = -ln(A/B) with A <= B (the m+1 match set is a subset) => >= 0."""
    x = white_noise(400, seed=int(seed))
    se = ts.sample_entropy(x)
    assert se >= 0.0
    assert np.isfinite(se)  # 400 noise samples have m+1 matches => finite


@given(seed=seeds)
@settings(max_examples=10)
def test_approximate_entropy_nonnegative(seed: int) -> None:
    """ApEn = Phi^m - Phi^{m+1} >= 0 (correlation sums shrink with template length)."""
    x = white_noise(400, seed=int(seed))
    ae = ts.approximate_entropy(x)
    assert ae >= -1e-9  # non-negative up to float noise
    assert np.isfinite(ae)


# ---------------------------------------------------------------------------
# Multiscale entropy: shape + non-negativity of finite entries
# ---------------------------------------------------------------------------


@given(seed=seeds, k=st.integers(min_value=2, max_value=6))
@settings(max_examples=10)
def test_multiscale_entropy_shape_and_nonneg(seed: int, k: int) -> None:
    """MSE over scales [1..k] returns a length-k vector of non-negative entropies."""
    x = white_noise(1500, seed=int(seed))
    profile = ts.multiscale_entropy(x, list(range(1, k + 1)))
    assert profile.shape == (k,)
    finite = profile[np.isfinite(profile)]
    # SampEn at every coarse-grained scale is non-negative where it is finite.
    assert np.all(finite >= -1e-9)


# ---------------------------------------------------------------------------
# LZ76 complexity / entropy: known limits + structure-sensitivity
# ---------------------------------------------------------------------------


def test_lz76_complexity_constant_series() -> None:
    """A constant series parses as exactly two LZ76 factors (a | aaa...)."""
    const = np.full(256, 3.0)
    c = ts.lz76_complexity(const)
    # Kaspar-Schuster: a constant string "aaaa..." factorises as a + aaa.. = 2.
    assert c == 2.0


@given(seed=seeds)
@settings(max_examples=20)
def test_lz76_complexity_is_integral_and_at_least_one(seed: int) -> None:
    """The native LZ76 parse returns an integer factor count >= 1."""
    x = white_noise(512, seed=int(seed))
    c = ts.lz76_complexity(x)
    assert c == round(c)  # the count is integer-valued even when typed float
    assert c >= 1


@given(seed=seeds)
@settings(max_examples=20)
def test_lz76_random_more_complex_than_constant(seed: int) -> None:
    """A random binary string needs many more factors than an equal-length constant."""
    rng = np.random.default_rng(int(seed))
    n = 512
    rand = rng.integers(0, 2, size=n)
    const = np.zeros(n, dtype=np.intp)
    c_rand = ts.lz76_complexity(rand, symbolize=None)
    c_const = ts.lz76_complexity(const, symbolize=None)
    # Incompressible noise has far more LZ76 factors than a single repeated symbol.
    assert c_rand > c_const


@given(seed=seeds)
@settings(max_examples=15)
def test_lz76_entropy_nonnegative(seed: int) -> None:
    """The normalised LZ76 density c*log_k(n)/n is non-negative."""
    x = white_noise(512, seed=int(seed))
    h = ts.lz76_entropy(x)
    assert h >= 0.0


def test_lz76_random_entropy_exceeds_periodic() -> None:
    """A chaotic/noisy series carries more LZ76 entropy than a clean periodic one."""
    noise = white_noise(2048, seed=7)
    periodic = sinusoid(2048, freq=0.02)
    # Median-binarised noise is near-incompressible; a sinusoid binarises to a
    # short repeating block, so its LZ76 entropy density is strictly lower.
    assert ts.lz76_entropy(noise) > ts.lz76_entropy(periodic)


# ---------------------------------------------------------------------------
# Qualitative ordering: irregular signals are more "entropic" than periodic ones
# ---------------------------------------------------------------------------


def test_permutation_entropy_noise_exceeds_periodic(
    noise_signal: np.ndarray, periodic_signal: np.ndarray
) -> None:
    """White noise uses ordinal patterns more uniformly than a periodic signal."""
    pe_noise = ts.permutation_entropy(noise_signal)
    pe_periodic = ts.permutation_entropy(periodic_signal)
    # A clean periodic signal cycles through few ordinal patterns.
    assert pe_noise > pe_periodic


def test_sample_entropy_noise_exceeds_periodic() -> None:
    """White noise is far less self-similar than a clean sinusoid => higher SampEn."""
    noise = white_noise(400, seed=8)
    periodic = sinusoid(400, freq=0.05)
    se_noise = ts.sample_entropy(noise)
    se_periodic = ts.sample_entropy(periodic)
    # Periodic templates recur predictably, so their sample entropy is much lower.
    assert se_noise > se_periodic


def test_permutation_entropy_noise_exceeds_chaotic_exceeds_periodic() -> None:
    """Regularity ordering noise > deterministic-chaos > periodic for PE."""
    noise = white_noise(2048, seed=9)
    chaos = logistic_series(2048, r=4.0)
    periodic = sinusoid(2048, freq=0.02)
    pe_noise = ts.permutation_entropy(noise)
    pe_chaos = ts.permutation_entropy(chaos)
    pe_periodic = ts.permutation_entropy(periodic)
    # Deterministic chaos sits between iid noise and a periodic orbit.
    assert pe_noise > pe_chaos > pe_periodic
