"""
Cross-quantifier known-value harness (stream I-QA).

Two kinds of check live here, and neither duplicates the per-stream literature
values that already sit in ``test_chaos.py`` / ``test_dimensions.py`` / ...:

* **Analytic identities** — clean limits an estimator must hit on a signal whose
  answer is known exactly (a monotone ramp has zero permutation entropy, a pure
  tone has near-zero spectral entropy, points on a circle have correlation
  dimension one, ...).

* **Cross-quantifier agreement** — five *independent* complexity quantifiers
  (permutation entropy, sample entropy, spectral entropy, LZ76 complexity and
  RQA determinism) must agree on the qualitative ordering of a periodic vs a
  random signal, with a deterministic-chaotic series falling in between.  No
  single estimator is trusted; the test is that they *concur*.

All randomness is seeded so a failing example reproduces; tolerances are
generous-but-meaningful (documented inline) because these are noisy estimators.
"""

from __future__ import annotations

import numpy as np
import pytest
from _strategies import (
    henon_series,
    logistic_series,
    seeds,
    sinusoid,
    white_noise,
)
from hypothesis import given, settings
from hypothesis import strategies as st

import tsdynamics as ts
import tsdynamics.transforms as tx

# ---------------------------------------------------------------------------
# Analytic identities
# ---------------------------------------------------------------------------


def test_permutation_entropy_monotone_is_zero():
    """A strictly increasing ramp uses a single ordinal pattern → entropy 0."""
    pe = ts.permutation_entropy(np.arange(1000, dtype=float))
    # Exactly one pattern occurs, so the distribution is a point mass → H == 0.
    assert pe == pytest.approx(0.0, abs=1e-12)


def test_permutation_entropy_white_noise_is_near_one():
    """White noise visits all ordinal patterns ~uniformly → normalised PE ≈ 1."""
    pe = ts.permutation_entropy(white_noise(4000, seed=11), normalize=True)
    # Finite-sample undershoot is small for n=4000, m=3; 0.95 is a safe floor.
    assert pe > 0.95
    assert pe <= 1.0 + 1e-9


def test_lz76_constant_sequence_is_two():
    """A single-symbol sequence parses to exactly two LZ76 factors (Kaspar–Schuster).

    The first symbol is factor 1; the remaining identical run is the second,
    closing factor — so a constant series has complexity 2, never 1 (cf. the
    ``lz76_complexity("aaaaaaaa") == 2`` docstring example).
    """
    const = np.full(2000, 3.14159)
    assert ts.lz76_complexity(const) == pytest.approx(2.0)


def test_lz76_white_noise_far_exceeds_constant():
    """Median-binarised white noise has vastly more factors than a constant run."""
    const = np.full(2000, -7.0)
    noise = white_noise(2000, seed=21)
    c_const = ts.lz76_complexity(const)
    c_noise = ts.lz76_complexity(noise)
    # The constant is 2 factors; noise binarises to a balanced random bit string
    # whose factor count grows ~ n/log2(n) — orders of magnitude larger.
    assert c_noise > 20.0 * c_const


def test_spectral_entropy_pure_tone_is_low():
    """A pure sinusoid concentrates power in one bin → spectral entropy ≈ 0."""
    se = tx.spectral_entropy(sinusoid(2048, freq=0.05), normalize=True)
    # A single dominant line gives a near-point-mass spectrum; < 0.2 with leakage.
    assert se < 0.2


def test_spectral_entropy_white_noise_is_high():
    """White noise has a (near-)flat spectrum → spectral entropy near 1."""
    se = tx.spectral_entropy(white_noise(2048, seed=31), normalize=True)
    # A flat spectrum approaches the maximum; > 0.6 is a comfortable floor.
    assert se > 0.6


def test_correlation_dimension_circle_is_one():
    """Points on a unit circle lie on a 1-D manifold → D2 ≈ 1."""
    rng = np.random.default_rng(101)
    theta = rng.uniform(0.0, 2.0 * np.pi, 2500)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    d2 = float(ts.correlation_dimension(circle))
    # A smooth 1-D manifold embedded in R^2; ~0.25 covers the GP slope scatter.
    assert d2 == pytest.approx(1.0, abs=0.25)


def test_correlation_dimension_filled_square_is_two():
    """A uniformly filled square is 2-D → D2 ≈ 2 (edge effects bias it down)."""
    rng = np.random.default_rng(202)
    square = rng.uniform(0.0, 1.0, (2500, 2))
    d2 = float(ts.correlation_dimension(square))
    # Finite-N edge effects bias a filled square slightly below 2; ~0.3 covers it.
    assert d2 == pytest.approx(2.0, abs=0.3)


def test_zero_one_test_periodic_is_near_zero():
    """A periodic logistic orbit (r=3.5, period-4) is regular → K ≈ 0."""
    x = logistic_series(4000, r=3.5)
    k = ts.zero_one_test(x, seed=0)
    # Regular dynamics keep the (p, q) translation bounded → K close to 0.
    assert k < 0.3


def test_zero_one_test_chaotic_is_near_one():
    """The fully chaotic logistic orbit (r=4) diffuses → K ≈ 1."""
    x = logistic_series(4000, r=4.0)
    k = ts.zero_one_test(x, seed=0)
    # Chaos makes (p, q) random-walk → linear MSD growth → K close to 1.
    assert k > 0.5


def test_zero_one_test_separates_regular_from_chaotic():
    """The 0–1 test ranks the chaotic orbit strictly above the periodic one."""
    k_reg = ts.zero_one_test(logistic_series(4000, r=3.5), seed=0)
    k_cha = ts.zero_one_test(logistic_series(4000, r=4.0), seed=0)
    # A wide, unambiguous separation between the two regimes.
    assert k_cha - k_reg > 0.5


# ---------------------------------------------------------------------------
# Cross-quantifier agreement: regular (periodic) vs random (noise)
#
# Five independent complexity quantifiers must concur that a clean periodic
# signal is *simpler* than seeded white noise.  Sample entropy is subsampled to
# 1024 points to keep its O(n^2) template matching fast.
# ---------------------------------------------------------------------------

_RR = 0.05  # fixed target recurrence rate for the RQA leg
_EMB_M = 3  # embedding dimension for the recurrence reconstruction
_EMB_TAU = 5  # embedding delay (samples)
_THEILER = 5  # Theiler band, drops temporally-correlated near-diagonal pairs


def _periodic_signal() -> np.ndarray:
    """A clean two-tone periodic signal (mirrors the ``periodic_signal`` fixture)."""
    return sinusoid(2048, freq=0.02) + 0.5 * sinusoid(2048, freq=0.04, phase=0.7)


def _det(signal: np.ndarray) -> float:
    """RQA determinism of a delay-embedding of ``signal`` at fixed recurrence rate."""
    emb = ts.embed(signal, dimension=_EMB_M, delay=_EMB_TAU)
    return ts.rqa(emb, recurrence_rate=_RR, theiler_window=_THEILER).determinism


def _entropy_quantifiers(signal: np.ndarray) -> dict[str, float]:
    """The four 'higher = more complex' quantifiers for one signal."""
    return {
        "permutation_entropy": ts.permutation_entropy(signal),
        "sample_entropy": ts.sample_entropy(signal[:1024]),
        "spectral_entropy": tx.spectral_entropy(signal, normalize=True),
        "lz76_complexity": ts.lz76_complexity(signal),
    }


@pytest.mark.parametrize("noise_seed", [12345, 7, 2024, 99999])
def test_all_quantifiers_agree_periodic_simpler_than_noise(noise_seed: int):
    """Five independent quantifiers concur: periodic ≺ noise in complexity."""
    regular = _periodic_signal()
    random = white_noise(2048, seed=noise_seed)

    reg = _entropy_quantifiers(regular)
    ran = _entropy_quantifiers(random)
    # Higher-is-more-complex quantifiers: regular strictly below random.
    for name in reg:
        assert reg[name] < ran[name], f"{name}: regular {reg[name]} !< random {ran[name]}"

    # RQA determinism runs the other way (regular is *more* deterministic).
    assert _det(regular) > _det(random)


def test_chaotic_falls_between_periodic_and_noise():
    """Permutation entropy ranks periodic ≺ deterministic-chaotic ≺ white noise.

    Deterministic chaos sits between perfect order and pure randomness: more
    irregular than a periodic orbit, but more structured (lower-entropy) than
    noise — a sanity check that the quantifier is not saturating at either end.
    """
    pe_periodic = ts.permutation_entropy(_periodic_signal())
    pe_chaotic = ts.permutation_entropy(henon_series(2048))
    pe_noise = ts.permutation_entropy(white_noise(2048, seed=555))
    assert pe_periodic < pe_chaotic < pe_noise


@settings(max_examples=10)
@given(noise_seed=seeds, channel_seed=st.integers(min_value=0, max_value=2**16))
def test_quantifier_agreement_is_robust_over_noise_seeds(noise_seed, channel_seed):
    """The periodic-vs-noise ordering holds for *any* seeded noise realisation.

    A small Hypothesis sweep over independent noise seeds shows the five-way
    agreement is a property of the signals' character, not of one lucky draw.
    ``channel_seed`` perturbs the noise scale a touch so the draws really differ.
    """
    regular = _periodic_signal()
    scale = 0.5 + (channel_seed % 1000) / 1000.0  # in [0.5, 1.5), still broadband
    random = white_noise(2048, seed=int(noise_seed), scale=scale)

    reg = _entropy_quantifiers(regular)
    ran = _entropy_quantifiers(random)
    for name in reg:
        assert reg[name] < ran[name], f"{name}: regular {reg[name]} !< random {ran[name]}"
    assert _det(regular) > _det(random)
