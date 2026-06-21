"""
Property + known-value tests for the surrogate-data toolkit (stream **I-QA**).

The surrogate generators (:func:`tsdynamics.random_shuffle`,
:func:`~tsdynamics.fourier_surrogate`, :func:`~tsdynamics.aaft_surrogate`,
:func:`~tsdynamics.iaaft_surrogate` and the :func:`~tsdynamics.surrogates`
dispatcher) are *constrained-realisation* nulls: each preserves a specific linear
property of the data and randomises the rest.  These tests assert those
constraints as hard mathematical invariants —

- length / shape preservation (one surrogate is ``(N,)`` after squeezing the
  ``(1, N)`` ensemble; ``n`` surrogates are ``(n, N)``),
- ``random_shuffle`` preserves the exact multiset of samples,
- ``fourier_surrogate`` preserves the magnitude spectrum ``|rfft|`` exactly,
- ``aaft_surrogate`` preserves the amplitude distribution exactly, and ``iaaft``
  closely (and its spectrum to good accuracy),
- seed reproducibility / seed sensitivity,

plus known-value checks on the discriminating statistics
(:func:`~tsdynamics.time_reversal_asymmetry` vanishes on a time-symmetric cosine)
and the :func:`~tsdynamics.surrogate_test` p-value bounds.
"""

from __future__ import annotations

import numpy as np
from _strategies import ar1, finite_signals, seeds, sinusoid, white_noise
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tsdynamics as ts

# Every generator returns an (n, N) ensemble; n=1 still yields a 2-D (1, N).
_GENERATORS = (
    ts.random_shuffle,
    ts.fourier_surrogate,
    ts.aaft_surrogate,
    ts.iaaft_surrogate,
)
_METHOD_NAMES = ("shuffle", "ft", "aaft", "iaaft")


# ---------------------------------------------------------------------------
# Shape / length preservation — holds for every generator and any finite signal.
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=32, max_size=256), seed=seeds)
def test_single_surrogate_is_ensemble_of_one(x: np.ndarray, seed: int) -> None:
    """``n=1`` returns shape ``(1, N)`` with N preserved, for every generator."""
    for gen in _GENERATORS:
        out = gen(x, 1, seed=seed)
        assert out.shape == (1, x.size)
        assert np.all(np.isfinite(out))


@given(
    x=finite_signals(min_size=32, max_size=200),
    n=st.integers(min_value=2, max_value=6),
    seed=seeds,
)
@settings(max_examples=20)
def test_many_surrogates_shape(x: np.ndarray, n: int, seed: int) -> None:
    """``n`` surrogates form an ``(n, N)`` block, length preserved, all finite."""
    for gen in _GENERATORS:
        out = gen(x, n, seed=seed)
        assert out.shape == (n, x.size)
        assert np.all(np.isfinite(out))


@given(
    x=finite_signals(min_size=16, max_size=128),
    method=st.sampled_from(_METHOD_NAMES),
    n=st.integers(min_value=1, max_value=4),
    seed=seeds,
)
@settings(max_examples=20)
def test_dispatcher_matches_direct_call(x: np.ndarray, method: str, n: int, seed: int) -> None:
    """``surrogates(x, method, ...)`` is bit-identical to the named generator."""
    direct = {
        "shuffle": ts.random_shuffle,
        "ft": ts.fourier_surrogate,
        "aaft": ts.aaft_surrogate,
        "iaaft": ts.iaaft_surrogate,
    }[method]
    via_dispatch = ts.surrogates(x, method, n, seed=seed)
    via_direct = direct(x, n, seed=seed)
    assert via_dispatch.shape == (n, x.size)
    # Same method + same seed must reproduce exactly through either entry point.
    assert np.array_equal(via_dispatch, via_direct)


# ---------------------------------------------------------------------------
# random_shuffle — preserves the EXACT multiset of samples (and nothing else).
# ---------------------------------------------------------------------------


@given(
    x=finite_signals(min_size=16, max_size=256),
    n=st.integers(min_value=1, max_value=5),
    seed=seeds,
)
def test_random_shuffle_preserves_multiset(x: np.ndarray, n: int, seed: int) -> None:
    """Each shuffle surrogate is a permutation: sorted values match exactly."""
    sorted_x = np.sort(x)
    out = ts.random_shuffle(x, n, seed=seed)
    for surrogate in out:
        # A permutation has the identical sorted value vector (bit-for-bit).
        assert np.array_equal(np.sort(surrogate), sorted_x)


@given(seed=seeds, n_pts=st.integers(min_value=64, max_value=256))
def test_random_shuffle_actually_permutes(seed: int, n_pts: int) -> None:
    """A signal of all-distinct values is genuinely reordered, not the identity."""
    # Use strictly-increasing distinct values so the identity permutation is the
    # ONLY ordering equal to the original (a constant-heavy array would be fixed
    # by many permutations).  Draw several surrogates: the chance that all five
    # independent shuffles of 64+ distinct points leave every element in place is
    # ~ (1/n!)^5 — negligible, so "at least one moves" is a robust invariant.
    x = np.linspace(-1.0, 1.0, n_pts) ** 3  # distinct, non-uniformly spaced
    out = ts.random_shuffle(x, 5, seed=seed)
    sorted_x = np.sort(x)
    moved = False
    for surrogate in out:
        assert np.array_equal(np.sort(surrogate), sorted_x)  # multiset preserved
        moved = moved or not np.array_equal(surrogate, x)
    assert moved


# ---------------------------------------------------------------------------
# fourier_surrogate — preserves the magnitude (power) spectrum EXACTLY.
# ---------------------------------------------------------------------------


@given(
    x=finite_signals(min_size=64, max_size=256),
    n=st.integers(min_value=1, max_value=4),
    seed=seeds,
)
def test_fourier_preserves_power_spectrum(x: np.ndarray, n: int, seed: int) -> None:
    """Phase randomisation keeps ``|rfft|`` (hence the power spectrum) intact."""
    mag_x = np.abs(np.fft.rfft(x))
    scale = float(np.max(mag_x))
    assume(scale > 0.0)
    out = ts.fourier_surrogate(x, n, seed=seed)
    for surrogate in out:
        mag_s = np.abs(np.fft.rfft(surrogate))
        # FT surrogate reconstructs the spectrum exactly up to FFT round-off; a
        # tight relative tolerance (scaled by the peak magnitude) is meaningful.
        assert np.allclose(mag_s, mag_x, rtol=1e-8, atol=1e-8 * scale)


@given(x=finite_signals(min_size=64, max_size=256), seed=seeds)
def test_fourier_changes_the_waveform(x: np.ndarray, seed: int) -> None:
    """Randomised phases produce a genuinely different time series."""
    # A pure-DC / single-frequency edge case could be phase-invariant; require
    # real spectral content beyond DC so phase randomisation has an effect.
    spectrum = np.abs(np.fft.rfft(x))
    assume(np.count_nonzero(spectrum[1:] > 1e-6 * spectrum.max()) >= 2)
    out = ts.fourier_surrogate(x, 1, seed=seed)[0]
    # Same |spectrum|, different waveform (phases moved).
    assert not np.allclose(out, x, atol=1e-9)


# ---------------------------------------------------------------------------
# aaft / iaaft — preserve the amplitude distribution (sorted values).
# ---------------------------------------------------------------------------


@given(
    x=finite_signals(min_size=32, max_size=256),
    n=st.integers(min_value=1, max_value=4),
    seed=seeds,
)
def test_aaft_preserves_amplitude_distribution_exactly(x: np.ndarray, n: int, seed: int) -> None:
    """AAFT re-imposes ``x`` by rank, so the sorted values match bit-for-bit."""
    sorted_x = np.sort(x)
    out = ts.aaft_surrogate(x, n, seed=seed)
    for surrogate in out:
        assert np.array_equal(np.sort(surrogate), sorted_x)


@given(
    x=finite_signals(min_size=32, max_size=256),
    n=st.integers(min_value=1, max_value=3),
    seed=seeds,
)
def test_iaaft_preserves_amplitude_distribution(x: np.ndarray, n: int, seed: int) -> None:
    """IAAFT ends on the amplitude step → exact sorted values (allowing FFT noise)."""
    sorted_x = np.sort(x)
    scale = float(np.max(np.abs(x))) or 1.0
    out = ts.iaaft_surrogate(x, n, seed=seed)
    for surrogate in out:
        # The final projection is the exact rank-remap onto sorted(x); the only
        # slack is float round-off in the preceding irfft, so a tight scaled tol.
        assert np.allclose(np.sort(surrogate), sorted_x, rtol=1e-9, atol=1e-9 * scale)


@given(seed=seeds, n_pts=st.integers(min_value=128, max_value=512))
@settings(max_examples=15)
def test_iaaft_approximately_preserves_spectrum(seed: int, n_pts: int) -> None:
    """IAAFT matches the magnitude spectrum to good (looser-than-FT) accuracy."""
    # Use a structured AR(1) series: a smooth power spectrum where the IAAFT
    # amplitude/spectrum trade-off converges cleanly (a meaningful, not trivial,
    # closeness target).
    x = ar1(n_pts, phi=0.6, seed=seed % 1000)
    mag_x = np.abs(np.fft.rfft(x))
    power = float(np.sum(mag_x**2))
    assume(power > 0.0)
    surrogate = ts.iaaft_surrogate(x, 1, seed=seed)[0]
    mag_s = np.abs(np.fft.rfft(surrogate))
    # Relative spectral mismatch (energy norm).  IAAFT trades a tiny amount of
    # spectral fidelity for an exact histogram; <5% energy error is the standard
    # quality bar and is FAR tighter than e.g. a shuffle would achieve (~O(1)).
    rel_err = float(np.linalg.norm(mag_s - mag_x) / np.linalg.norm(mag_x))
    assert rel_err < 0.05


# ---------------------------------------------------------------------------
# Seed reproducibility / sensitivity — every generator + the dispatcher.
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=32, max_size=256), seed=seeds)
def test_same_seed_reproduces(x: np.ndarray, seed: int) -> None:
    """A fixed seed yields a bit-identical surrogate on every generator."""
    for gen in _GENERATORS:
        a = gen(x, 2, seed=seed)
        b = gen(x, 2, seed=seed)
        assert np.array_equal(a, b)


@given(
    seed_a=seeds,
    seed_b=seeds,
    n_pts=st.integers(min_value=64, max_value=256),
)
def test_different_seeds_differ(seed_a: int, seed_b: int, n_pts: int) -> None:
    """Distinct seeds give distinct surrogates (for every randomising generator)."""
    assume(seed_a != seed_b)
    # An all-distinct, smoothly-varying signal (no extreme-multiplicity ties).
    # A sparse/near-constant array admits only a handful of distinct surrogate
    # arrangements, so two seeds could legitimately collide; with 64+ distinct
    # values the chance two independent draws coincide is negligible, making
    # "different seed ⇒ different surrogate" a robust invariant.
    t = np.linspace(0.0, 1.0, n_pts)
    x = np.sin(2.0 * np.pi * 3.0 * t) + 0.3 * t**2  # distinct values, real spectrum
    for gen in _GENERATORS:
        a = gen(x, 1, seed=seed_a)[0]
        b = gen(x, 1, seed=seed_b)[0]
        # Two independent draws of a randomising procedure must not coincide.
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# time_reversal_asymmetry — known values & basic invariants.
# ---------------------------------------------------------------------------


@given(
    cycles=st.integers(min_value=3, max_value=100),
    amplitude=st.floats(min_value=0.5, max_value=5.0),
)
@settings(max_examples=30)
def test_time_reversal_vanishes_on_cosine(cycles: int, amplitude: float) -> None:
    """A pure cosine is time-reversible → the asymmetry statistic is ~0."""
    n = 1024
    # A phase-0 cosine with an INTEGER number of cycles over the window is exactly
    # even-symmetric about the origin and periodic-on-the-window, so the increment
    # distribution is symmetric and the third-moment statistic is zero up to float
    # round-off — a genuinely tight bound (vs the > 0.1 an irreversible sawtooth
    # produces in the contrast test below).  A non-zero phase would start the
    # window mid-cycle and reintroduce a finite-window edge bias; that is a
    # windowing artefact, not a property of the statistic, so it is excluded here.
    x = sinusoid(n, freq=cycles / n, amplitude=amplitude)
    trev = ts.time_reversal_asymmetry(x)
    assert np.isfinite(trev)
    # Residual ~1e-9 (few cycles) growing to ~1e-4 (many cycles, larger
    # increments amplify the ratio's round-off); still > 1000x below the
    # irreversible-sawtooth value, so the contrast stays unambiguous.
    assert abs(trev) < 1e-4


@given(
    freq=st.floats(min_value=0.02, max_value=0.2),
    amplitude=st.floats(min_value=0.5, max_value=5.0),
    offset=st.floats(min_value=-3.0, max_value=3.0),
)
@settings(max_examples=20)
def test_time_reversal_is_scale_and_offset_invariant(
    freq: float, amplitude: float, offset: float
) -> None:
    """The statistic is dimensionless: invariant to affine ``a*x + b`` rescaling."""
    base = sinusoid(1024, freq=freq)
    trev_base = ts.time_reversal_asymmetry(base)
    # Increments cancel the offset and the amplitude factors out of the
    # third/second-moment ratio (3rd power over (2nd power)^1.5) → exact invariance.
    trev_scaled = ts.time_reversal_asymmetry(amplitude * base + offset)
    scale = max(abs(trev_base), 1.0)
    assert abs(trev_base - trev_scaled) < 1e-9 * scale


@given(x=finite_signals(min_size=16, max_size=256), lag=st.integers(min_value=1, max_value=4))
def test_time_reversal_is_finite_and_sign_flips(x: np.ndarray, lag: int) -> None:
    """The statistic is finite and antisymmetric under time reversal."""
    assume(x.size > lag + 1)
    trev = ts.time_reversal_asymmetry(x, delay=lag)
    trev_rev = ts.time_reversal_asymmetry(x[::-1], delay=lag)
    assert np.isfinite(trev)
    assert np.isfinite(trev_rev)
    # Reversing time negates the increments' odd (third) moment → sign flip.
    # The denominator (even second moment) is reversal-invariant, so the ratio
    # negates exactly up to float round-off.
    scale = max(abs(trev), abs(trev_rev), 1.0)
    assert abs(trev + trev_rev) < 1e-9 * scale


def test_time_reversal_nonzero_on_irreversible_series() -> None:
    """A dissipative-flow proxy (a sawtooth) is irreversible → |T_rev| > 0."""
    # An asymmetric ramp-then-drop wave: slow rise, fast fall, so the increment
    # distribution is skewed and the statistic must be clearly non-zero.
    period = 32
    ramp = np.linspace(0.0, 1.0, period, endpoint=False)
    x = np.tile(ramp, 40)
    trev = ts.time_reversal_asymmetry(x)
    assert np.isfinite(trev)
    # Strongly skewed increments → the statistic is well away from zero.
    assert abs(trev) > 0.1


# ---------------------------------------------------------------------------
# nonlinear_prediction_error — bounds + deterministic-vs-noise contrast.
# ---------------------------------------------------------------------------


@given(seed=seeds)
@settings(max_examples=15)
def test_prediction_error_finite_and_nonnegative(seed: int) -> None:
    """The normalised RMS prediction error is a finite, non-negative float."""
    x = ar1(512, phi=0.5, seed=seed % 10_000)
    err = ts.nonlinear_prediction_error(x)
    assert np.isfinite(err)
    assert err >= 0.0


@settings(max_examples=10)
@given(seed=seeds)
def test_prediction_error_smaller_for_deterministic(seed: int) -> None:
    """A deterministic chaotic orbit is more predictable than white noise."""
    # Logistic r=4 is deterministic (low prediction error); seeded white noise is
    # the maximally-unpredictable reference (error ~1).  The ORDERING is the
    # invariant, with a comfortable margin so the test is not knife-edge.
    from _strategies import logistic_series

    det = logistic_series(1024, r=4.0, x0=0.3 + 1e-3 * (seed % 7))
    rand = white_noise(1024, seed=seed % 9999)
    err_det = ts.nonlinear_prediction_error(det, dimension=3, delay=1)
    err_rand = ts.nonlinear_prediction_error(rand, dimension=3, delay=1)
    assert err_det < err_rand
    # Determinism should buy a real margin, not a coin-flip difference.
    assert err_det < 0.5 * err_rand


# ---------------------------------------------------------------------------
# surrogate_test — p-value bounds, attainable extremes, and known rejections.
# ---------------------------------------------------------------------------


@given(
    x=finite_signals(min_size=64, max_size=200),
    method=st.sampled_from(_METHOD_NAMES),
    seed=seeds,
)
@settings(max_examples=15)
def test_surrogate_test_pvalue_bounds(x: np.ndarray, method: str, seed: int) -> None:
    """The reported p-value lies in the attainable rank range ``(0, 1]``."""
    n_surr = 19  # M = 2/alpha - 1 with alpha = 0.1 → two-sided floor 2/20 = 0.1
    result = ts.surrogate_test(x, method=method, n=n_surr, seed=seed, alpha=0.1)
    assert result.surrogate_statistics.shape == (n_surr,)
    # Rank p-value floor is 1/(M+1) one-sided, 2/(M+1) two-sided; ceiling is 1.
    assert 1.0 / (n_surr + 1) <= result.p_value <= 1.0
    # rejected is exactly the alpha decision the dataclass documents.
    assert result.rejected == (result.p_value <= result.alpha)
    assert np.isfinite(result.z_score)


def test_surrogate_test_rejects_lorenz_like_irreversible() -> None:
    """A strongly irreversible deterministic series rejects the linear null."""
    # A skewed sawtooth (slow rise / fast fall) is time-irreversible, which the
    # FT/AAFT/IAAFT surrogates cannot reproduce → time-reversal asymmetry rejects.
    period = 40
    ramp = np.linspace(0.0, 1.0, period, endpoint=False) ** 1.5
    x = np.tile(ramp, 30) + 1e-6 * np.arange(period * 30)  # tiny tilt breaks exact ties
    result = ts.surrogate_test(x, statistic="time_reversal", method="ft", n=39, seed=7, alpha=0.05)
    assert result.rejected
    assert result.p_value <= 0.05
    # The data statistic should sit far in the tail of the surrogate ensemble.
    assert abs(result.z_score) > 2.0


def test_surrogate_test_does_not_reject_linear_ar1() -> None:
    """A genuine linear AR(1) null is (almost always) NOT rejected at α=0.05."""
    # AR(1) is exactly the linear-Gaussian process the FT surrogates realise, so
    # time-reversal asymmetry should be indistinguishable from its surrogates.
    # Average the decision over several seeds: a correct test rejects a true null
    # at ~alpha, so a small majority-of-seeds bound is the honest invariant.
    rejections = 0
    n_trials = 12
    for s in range(n_trials):
        x = ar1(1024, phi=0.7, seed=1000 + s)
        result = ts.surrogate_test(
            x, statistic="time_reversal", method="ft", n=39, seed=500 + s, alpha=0.05
        )
        rejections += int(result.rejected)
    # A valid α=0.05 test must not reject the true linear null wholesale; far
    # fewer than half the trials should reject (generous bound vs the ~0.05 rate).
    assert rejections <= n_trials // 3
