"""
Cross-quantifier known-value harness (stream I-QA).

Two kinds of check live here, and neither duplicates the per-stream literature
values that already sit in ``test_chaos.py`` / ``test_dimensions.py`` / ...:

* **Analytic identities** — clean limits an estimator must hit on a signal whose
  answer is known exactly (a periodic orbit is perfectly deterministic under RQA,
  the logistic map at ``r = 4`` has maximal exponent ``ln 2``, points on a circle
  have correlation dimension one, ...).

* **Cross-quantifier agreement** — *independent* dynamical quantifiers must agree
  on the qualitative ordering of regular vs irregular dynamics.  Two panels do
  this.  On a **measured series**: correlation dimension, the 0--1 test, the
  data-driven maximal Lyapunov exponent and RQA determinism must all rank a
  periodic signal below seeded white noise, with a deterministic-chaotic series
  falling in between.  On a **system**: the maximal Lyapunov exponent, GALI₂, the
  0--1 test and the attractor's correlation dimension must all separate a chaotic
  map from a regular one.  No single estimator is trusted; the test is that they
  *concur*.

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

# Shared reconstruction parameters for the data-driven quantifiers.  One
# embedding (m=3, tau=5) feeds every leg, so the panel compares estimators, not
# reconstructions; the Theiler band drops temporally-correlated near-diagonal
# pairs.
_EMB_M = 3  # embedding dimension for the delay reconstruction
_EMB_TAU = 5  # embedding delay (samples)
_RR = 0.05  # fixed target recurrence rate for the RQA leg
_THEILER = 5  # Theiler band

# ---------------------------------------------------------------------------
# Analytic identities
# ---------------------------------------------------------------------------


def test_rqa_determinism_of_a_periodic_signal_is_one():
    """A periodic orbit revisits itself exactly → every recurrence point is on a line.

    Determinism is the fraction of recurrence points lying on diagonal lines of
    length ``>= min_diagonal``.  A periodic signal recurs on parallel diagonals
    only, so DET saturates at 1.
    """
    emb = ts.analysis.embed(sinusoid(2048, freq=0.02), dimension=_EMB_M, delay=_EMB_TAU)
    det = ts.analysis.rqa(emb, recurrence_rate=_RR, theiler=_THEILER).determinism
    # A handful of line-edge points keeps it a hair under 1; 1e-3 covers them.
    assert det == pytest.approx(1.0, abs=1.0e-3)


def test_rqa_determinism_of_white_noise_is_near_zero():
    """Uncorrelated noise recurs in isolated points, not on diagonals → DET ≈ 0."""
    emb = ts.analysis.embed(white_noise(2048, seed=11), dimension=_EMB_M, delay=_EMB_TAU)
    det = ts.analysis.rqa(emb, recurrence_rate=_RR, theiler=_THEILER).determinism
    # Chance alignments give a small positive floor at n=2048; 0.25 is a safe cap.
    assert det < 0.25


def test_lyapunov_from_data_logistic_r4_is_ln2():
    """The fully chaotic logistic map has the exact exponent ``ln 2`` (Ulam–von Neumann).

    ``x_{n+1} = 4 x_n (1 - x_n)`` is conjugate to the doubling map via
    ``x = sin^2(pi theta / 2)``, so its maximal Lyapunov exponent is exactly
    ``ln 2`` — a rare closed-form target for a data-driven estimator.
    """
    lam = float(
        ts.analysis.lyapunov_from_data(logistic_series(4000, r=4.0), dimension=_EMB_M, delay=1)
    )
    # Kantz on 4000 noiseless points lands a few percent low (finite-sample
    # neighbourhood bias); 0.12 covers it while still excluding 0 and 2*ln2.
    assert lam == pytest.approx(np.log(2.0), abs=0.12)


def test_lyapunov_from_data_periodic_is_near_zero():
    """A periodic signal neither diverges nor contracts → maximal exponent ≈ 0."""
    lam = float(
        ts.analysis.lyapunov_from_data(_periodic_signal(), dimension=_EMB_M, delay=_EMB_TAU)
    )
    # Neighbouring cycles stay neighbours; only numerical drift is left.
    assert abs(lam) < 0.05


def test_correlation_dimension_circle_is_one():
    """Points on a unit circle lie on a 1-D manifold → D2 ≈ 1."""
    rng = np.random.default_rng(101)
    theta = rng.uniform(0.0, 2.0 * np.pi, 2500)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    d2 = float(ts.analysis.correlation_dimension(circle))
    # A smooth 1-D manifold embedded in R^2; ~0.25 covers the GP slope scatter.
    assert d2 == pytest.approx(1.0, abs=0.25)


def test_correlation_dimension_filled_square_is_two():
    """A uniformly filled square is 2-D → D2 ≈ 2 (edge effects bias it down)."""
    rng = np.random.default_rng(202)
    square = rng.uniform(0.0, 1.0, (2500, 2))
    d2 = float(ts.analysis.correlation_dimension(square))
    # Finite-N edge effects bias a filled square slightly below 2; ~0.3 covers it.
    assert d2 == pytest.approx(2.0, abs=0.3)


def test_zero_one_test_periodic_is_near_zero():
    """A periodic logistic orbit (r=3.5, period-4) is regular → K ≈ 0."""
    x = logistic_series(4000, r=3.5)
    k = ts.analysis.zero_one_test(x, seed=0)
    # Regular dynamics keep the (p, q) translation bounded → K close to 0.
    assert k < 0.3


def test_zero_one_test_chaotic_is_near_one():
    """The fully chaotic logistic orbit (r=4) diffuses → K ≈ 1."""
    x = logistic_series(4000, r=4.0)
    k = ts.analysis.zero_one_test(x, seed=0)
    # Chaos makes (p, q) random-walk → linear MSD growth → K close to 1.
    assert k > 0.5


def test_zero_one_test_separates_regular_from_chaotic():
    """The 0–1 test ranks the chaotic orbit strictly above the periodic one."""
    k_reg = ts.analysis.zero_one_test(logistic_series(4000, r=3.5), seed=0)
    k_cha = ts.analysis.zero_one_test(logistic_series(4000, r=4.0), seed=0)
    # A wide, unambiguous separation between the two regimes.
    assert k_cha - k_reg > 0.5


# ---------------------------------------------------------------------------
# Cross-quantifier agreement I — a measured series: regular vs random
#
# Four independent dynamical quantifiers must concur that a clean periodic signal
# is *simpler* than seeded white noise.  They read the same reconstruction but
# measure genuinely different things: a fractal dimension (geometry of the point
# cloud), the 0--1 test (diffusion of a driven translation variable), the
# data-driven maximal exponent (neighbour divergence) and RQA determinism
# (recurrence-plot line structure).
# ---------------------------------------------------------------------------


def _periodic_signal() -> np.ndarray:
    """A clean two-tone periodic signal (mirrors the ``periodic_signal`` fixture)."""
    return sinusoid(2048, freq=0.02) + 0.5 * sinusoid(2048, freq=0.04, phase=0.7)


def _det(signal: np.ndarray) -> float:
    """RQA determinism of a delay-embedding of ``signal`` at fixed recurrence rate."""
    emb = ts.analysis.embed(signal, dimension=_EMB_M, delay=_EMB_TAU)
    return float(ts.analysis.rqa(emb, recurrence_rate=_RR, theiler=_THEILER).determinism)


def _complexity_quantifiers(signal: np.ndarray) -> dict[str, float]:
    """The three 'higher = more complex' data quantifiers for one signal."""
    embedded = ts.analysis.embed(signal, dimension=_EMB_M, delay=_EMB_TAU)
    return {
        "correlation_dimension": float(
            ts.analysis.correlation_dimension(embedded, theiler=_THEILER)
        ),
        "zero_one_test": float(ts.analysis.zero_one_test(signal, seed=0)),
        "lyapunov_from_data": float(
            ts.analysis.lyapunov_from_data(signal, dimension=_EMB_M, delay=_EMB_TAU)
        ),
    }


@pytest.mark.parametrize("noise_seed", [12345, 7, 2024, 99999])
def test_all_quantifiers_agree_periodic_simpler_than_noise(noise_seed: int):
    """Four independent quantifiers concur: periodic ≺ noise in complexity."""
    regular = _periodic_signal()
    random = white_noise(2048, seed=noise_seed)

    reg = _complexity_quantifiers(regular)
    ran = _complexity_quantifiers(random)
    # Higher-is-more-complex quantifiers: regular strictly below random.
    for name in reg:
        assert reg[name] < ran[name], f"{name}: regular {reg[name]} !< random {ran[name]}"

    # RQA determinism runs the other way (regular is *more* deterministic).
    assert _det(regular) > _det(random)


def test_chaotic_falls_between_periodic_and_noise():
    """Geometry and recurrence structure rank periodic ≺ chaotic ≺ white noise.

    Deterministic chaos sits between perfect order and pure randomness: its
    attractor is fractal (so a higher correlation dimension than a closed orbit,
    but lower than noise filling the embedding space), and its recurrence plot
    keeps diagonal structure noise has lost.  A sanity check that neither
    quantifier is saturating at either end.
    """
    periodic, chaotic, noise = (
        _periodic_signal(),
        henon_series(2048),
        white_noise(2048, seed=555),
    )
    d2 = [_complexity_quantifiers(s)["correlation_dimension"] for s in (periodic, chaotic, noise)]
    assert d2[0] < d2[1] < d2[2]

    det = [_det(s) for s in (periodic, chaotic, noise)]
    assert det[0] > det[1] > det[2]


@settings(max_examples=10)
@given(noise_seed=seeds, channel_seed=st.integers(min_value=0, max_value=2**16))
def test_quantifier_agreement_is_robust_over_noise_seeds(noise_seed, channel_seed):
    """The periodic-vs-noise ordering holds for *any* seeded noise realisation.

    A small Hypothesis sweep over independent noise seeds shows the four-way
    agreement is a property of the signals' character, not of one lucky draw.
    ``channel_seed`` perturbs the noise scale a touch so the draws really differ.
    """
    regular = _periodic_signal()
    scale = 0.5 + (channel_seed % 1000) / 1000.0  # in [0.5, 1.5), still broadband
    random = white_noise(2048, seed=int(noise_seed), scale=scale)

    reg = _complexity_quantifiers(regular)
    ran = _complexity_quantifiers(random)
    for name in reg:
        assert reg[name] < ran[name], f"{name}: regular {reg[name]} !< random {ran[name]}"
    assert _det(regular) > _det(random)


# ---------------------------------------------------------------------------
# Cross-quantifier agreement II — a system: chaotic vs regular
#
# The same idea one level up, where the quantifiers get the *model* rather than a
# measured series, so the tangent-space indicators join in: the Benettin maximal
# exponent, GALI_2 (Skokos), the 0--1 test on an observable, and the correlation
# dimension of the iterated orbit must all separate the chaotic Hénon map from a
# near-integrable standard (Chirikov) map on an invariant curve.
# ---------------------------------------------------------------------------

#: Deterministic ICs on each map's attractor / invariant curve.
_CHAOTIC_IC = [0.1, 0.1]
_REGULAR_IC = [0.1, 0.3]


def _chaotic_map():
    """The Hénon map at its canonical chaotic parameters (lambda_1 ≈ 0.42)."""
    return ts.systems.Henon()


def _regular_map():
    """The standard map at a near-integrable kick strength (orbits stay on curves)."""
    return ts.systems.Chirikov().with_params(k=0.05)


def test_system_quantifiers_agree_chaotic_above_regular():
    """Four independent system-level indicators concur: Hénon is chaotic, weak-kick
    Chirikov is not.

    Each reads different structure — Benettin rescaling (tangent growth), the GALI
    volume of two deviation vectors (their alignment), the 0--1 test (diffusion of
    a driven observable) and the correlation dimension (attractor geometry) — so
    agreement across all four is a genuine cross-check, not one estimator repeated.
    """
    chaotic, regular = _chaotic_map(), _regular_map()

    lam_c = float(ts.analysis.max_lyapunov(chaotic, n=3000, ic=_CHAOTIC_IC))
    lam_r = float(ts.analysis.max_lyapunov(regular, n=3000, ic=_REGULAR_IC))
    # A positive exponent vs one indistinguishable from zero.
    assert lam_c > 0.3
    assert abs(lam_r) < 0.01
    assert lam_c > lam_r

    # GALI_2 collapses exponentially on a chaotic orbit and only power-law-decays
    # on a regular one, so after the same number of iterations it is orders of
    # magnitude smaller for the chaotic map.
    gali_c = float(ts.analysis.gali(chaotic, k=2, n=500, ic=_CHAOTIC_IC))
    gali_r = float(ts.analysis.gali(regular, k=2, n=500, ic=_REGULAR_IC))
    assert gali_c < 1.0e-8
    assert gali_c < gali_r

    # The 0--1 test on the first component: K ≈ 1 (chaotic) vs K ≈ 0 (regular).
    k_c = float(ts.analysis.zero_one_test(chaotic, n=3000, ic=_CHAOTIC_IC, component=0, seed=0))
    k_r = float(ts.analysis.zero_one_test(regular, n=3000, ic=_REGULAR_IC, component=0, seed=0))
    assert k_c - k_r > 0.5

    # A fractal attractor (D2 ≈ 1.22 for Hénon) vs a smooth invariant curve (≈ 1).
    d2_c = float(ts.analysis.correlation_dimension(chaotic.run(steps=3000, ic=_CHAOTIC_IC)))
    d2_r = float(ts.analysis.correlation_dimension(regular.run(steps=3000, ic=_REGULAR_IC)))
    assert d2_c > d2_r
