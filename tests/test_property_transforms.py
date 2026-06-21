"""
Property and known-value tests for :mod:`tsdynamics.transforms` (stream I-QA).

These assert mathematical invariants of the transform layer that would fail on a
regression — shape preservation, the detrend/normalize contracts, non-negativity
and monotonicity of the PSD grid, the dominant-frequency known value, the
spectral-entropy ordering (sine low, noise high), the zero-phase low-pass tone
removal, and the Hjorth/feature-bag contracts.  Generic bound/shape invariants
draw from :func:`finite_signals`; the semantic ones use the deterministic
sinusoid / white-noise builders so a failing example reproduces exactly.
"""

from __future__ import annotations

import numpy as np
from _strategies import finite_signals, sinusoid, white_noise
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tsdynamics.transforms as tx

# Tolerances are deliberately generous-but-meaningful: estimators on finite,
# noisy data do not hit exact values, but each bound below is one a broken
# implementation would blow through.
_ATOL = 1.0e-6  # exact algebraic identities (detrend/normalize closed forms)


# ---------------------------------------------------------------------------
# Shape preservation — every shape-preserving transform is length/shape neutral
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=64, max_size=512))
def test_detrend_preserves_shape(x: np.ndarray) -> None:
    """detrend (both kinds) returns an array of the identical shape."""
    for kind in ("linear", "constant"):
        out = tx.detrend(x, method=kind)
        assert out.shape == x.shape


@given(x=finite_signals(min_size=64, max_size=512))
def test_normalize_preserves_shape(x: np.ndarray) -> None:
    """normalize (all methods) returns an array of the identical shape."""
    for method in ("zscore", "minmax", "l2", "demean"):
        out = tx.normalize(x, method=method)
        assert out.shape == x.shape


@given(x=finite_signals(min_size=64, max_size=512))
def test_filters_preserve_shape(x: np.ndarray) -> None:
    """Zero-phase Butterworth filters preserve input length (fs=1, sub-Nyquist)."""
    # Cutoffs strictly inside (0, 0.5) for fs=1; the signal is well longer than
    # the filter's padding (>=64 samples), so sosfiltfilt does not raise.
    assert tx.lowpass(x, 0.2, fs=1.0).shape == x.shape
    assert tx.highpass(x, 0.2, fs=1.0).shape == x.shape
    assert tx.bandpass(x, 0.1, 0.3, fs=1.0).shape == x.shape
    assert tx.bandstop(x, 0.1, 0.3, fs=1.0).shape == x.shape


# ---------------------------------------------------------------------------
# detrend / normalize closed-form invariants
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=64, max_size=512))
def test_detrend_linear_kills_mean_and_slope(x: np.ndarray) -> None:
    """detrend(kind='linear') leaves a zero-mean, zero-slope residual."""
    out = tx.detrend(x, method="linear")
    # Mean removed.
    assert abs(float(out.mean())) <= _ATOL * (1.0 + abs(float(x.mean())))
    # Least-squares slope of the residual against sample index is ~0: refit a
    # line and check the slope coefficient is negligible relative to the data.
    n = out.size
    idx = np.arange(n, dtype=float)
    idx -= idx.mean()
    slope = float((idx @ out) / (idx @ idx))
    # Scale-free bound: slope * (index span) must be tiny vs the data range.
    span = n - 1
    data_range = float(np.ptp(x)) + _ATOL
    assert abs(slope) * span <= 1.0e-6 * data_range


@given(x=finite_signals(min_size=64, max_size=512))
def test_detrend_constant_kills_mean(x: np.ndarray) -> None:
    """detrend(kind='constant') leaves a zero-mean residual."""
    out = tx.detrend(x, method="constant")
    assert abs(float(out.mean())) <= _ATOL * (1.0 + abs(float(x.mean())))


@given(x=finite_signals(min_size=64, max_size=512))
def test_normalize_zscore_unit_moments(x: np.ndarray) -> None:
    """normalize(method='zscore') yields mean ~ 0 and std ~ 1."""
    # A near-constant column would divide by a tiny std (passed through unscaled
    # by the implementation), so require a real spread before asserting std==1.
    assume(float(np.std(x)) > 1.0e-2)
    out = tx.normalize(x, method="zscore")
    assert abs(float(out.mean())) <= 1.0e-6
    assert abs(float(out.std()) - 1.0) <= 1.0e-6


@given(x=finite_signals(min_size=64, max_size=512))
def test_normalize_minmax_unit_interval(x: np.ndarray) -> None:
    """normalize(method='minmax') maps the channel onto [0, 1]."""
    assume(float(np.ptp(x)) > 1.0e-6)
    out = tx.normalize(x, method="minmax")
    assert abs(float(out.min()) - 0.0) <= 1.0e-6
    assert abs(float(out.max()) - 1.0) <= 1.0e-6


@given(x=finite_signals(min_size=64, max_size=512))
def test_normalize_l2_unit_norm(x: np.ndarray) -> None:
    """normalize(method='l2') yields a unit Euclidean norm."""
    # finite_signals guarantees non-constant ⇒ non-zero L2 norm.
    out = tx.normalize(x, method="l2")
    assert abs(float(np.linalg.norm(out)) - 1.0) <= 1.0e-6


@given(x=finite_signals(min_size=64, max_size=512))
def test_normalize_demean_kills_mean(x: np.ndarray) -> None:
    """normalize(method='demean') removes the mean (and nothing else)."""
    out = tx.normalize(x, method="demean")
    assert abs(float(out.mean())) <= _ATOL * (1.0 + abs(float(x.mean())))


# ---------------------------------------------------------------------------
# Power spectral density — grid invariants
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=128, max_size=512), seed=st.integers(0, 2**31 - 1))
def test_psd_nonnegative_and_freqs_increasing(x: np.ndarray, seed: int) -> None:
    """PSD is non-negative everywhere; freqs are >= 0 and strictly increasing."""
    del seed  # parameter unused; present so Hypothesis varies the array more
    freqs, psd = tx.power_spectral_density(x, fs=1.0)
    assert np.all(np.isfinite(psd))
    assert np.all(psd >= 0.0)  # PSD is a power → never negative
    assert freqs[0] >= 0.0
    assert np.all(np.diff(freqs) > 0.0)  # one-sided grid is strictly increasing


# ---------------------------------------------------------------------------
# Dominant frequency — known value on a pure sinusoid
# ---------------------------------------------------------------------------


@given(f0=st.floats(min_value=0.02, max_value=0.4))
@settings(max_examples=30)
def test_dominant_frequency_known_value(f0: float) -> None:
    """A pure tone at f0 (fs=1) has dominant_frequency within one FFT bin."""
    n = 1024
    x = sinusoid(n, freq=f0)
    # Periodogram gives full FFT resolution (bin = fs/n); Welch's coarse
    # nperseg=256 grid would only locate the peak to fs/256.
    peak = tx.dominant_frequency(x, fs=1.0, method="periodogram")
    bin_width = 1.0 / n
    # Within ~1.5 bins covers the case where f0 falls between two bins.
    assert abs(float(peak) - f0) <= 1.5 * bin_width


# ---------------------------------------------------------------------------
# Spectral entropy — bounds and the sine-vs-noise ordering
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=128, max_size=512))
def test_spectral_entropy_in_unit_interval(x: np.ndarray) -> None:
    """Normalised spectral entropy lands in [0, 1]."""
    h = float(tx.spectral_entropy(x, fs=1.0, normalize=True))
    assert -1.0e-9 <= h <= 1.0 + 1.0e-9


@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20)
def test_spectral_entropy_sine_low_noise_high(seed: int) -> None:
    """A pure sine has near-zero spectral entropy; white noise has high entropy."""
    n = 1024
    sine = sinusoid(n, freq=0.1)
    noise = white_noise(n, seed=seed)
    h_sine = float(tx.spectral_entropy(sine, fs=1.0, normalize=True))
    h_noise = float(tx.spectral_entropy(noise, fs=1.0, normalize=True))
    # Generous separation bounds: a single tone concentrates power (low H), a
    # flat spectrum spreads it (high H), and the gap is large in practice.
    assert h_sine < 0.3
    assert h_noise > 0.6
    assert h_noise > h_sine


# ---------------------------------------------------------------------------
# Spectral centroid / dominant frequency live in [0, Nyquist]
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=128, max_size=512))
def test_spectral_summaries_within_band(x: np.ndarray) -> None:
    """Centroid and dominant frequency lie in [0, Nyquist] for fs=1."""
    nyquist = 0.5  # fs/2 with fs=1
    centroid = float(tx.spectral_centroid(x, fs=1.0))
    dom = float(tx.dominant_frequency(x, fs=1.0))
    assert 0.0 <= centroid <= nyquist + 1.0e-9
    assert 0.0 <= dom <= nyquist + 1.0e-9


# ---------------------------------------------------------------------------
# Low-pass attenuates a high tone (zero-phase, length-preserving)
# ---------------------------------------------------------------------------


@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=15)
def test_lowpass_removes_high_tone(seed: int) -> None:
    """lowpass with cutoff between f_lo and f_hi crushes the f_hi power."""
    del seed  # deterministic two-tone; seed only diversifies Hypothesis runs
    n = 2048
    f_lo, f_hi = 0.03, 0.20
    x = sinusoid(n, freq=f_lo) + sinusoid(n, freq=f_hi)
    out = tx.lowpass(x, 0.08, fs=1.0)  # cutoff between the two tones
    assert out.shape == x.shape  # zero-phase filter preserves length

    # Compare the PSD power in a narrow band around f_hi before/after.
    f0, p0 = tx.power_spectral_density(x, fs=1.0, method="periodogram")
    f1, p1 = tx.power_spectral_density(out, fs=1.0, method="periodogram")
    band = (f0 >= f_hi - 0.01) & (f0 <= f_hi + 0.01)
    power_before = float(p0[band].sum())
    power_after = float(p1[band].sum())
    # The stop-band tone must be attenuated by a large factor.
    assert power_after < power_before / 50.0


# ---------------------------------------------------------------------------
# Zero-crossing rate — bounds and the constant-signal corner
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=64, max_size=512))
def test_zero_crossing_rate_in_unit_interval(x: np.ndarray) -> None:
    """ZCR is a fraction of adjacent sign changes ⇒ in [0, 1]."""
    z = float(tx.zero_crossing_rate(x))
    assert 0.0 <= z <= 1.0


@given(c=st.floats(min_value=-1.0e3, max_value=1.0e3), n=st.integers(8, 256))
def test_zero_crossing_rate_constant_is_zero(c: float, n: int) -> None:
    """A constant signal never changes sign ⇒ ZCR == 0."""
    z = float(tx.zero_crossing_rate(np.full(n, c, dtype=float)))
    assert z == 0.0


# ---------------------------------------------------------------------------
# Hjorth parameters — activity is the variance, complexity ~ 1 for a sine
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=64, max_size=512))
def test_hjorth_activity_is_variance(x: np.ndarray) -> None:
    """Hjorth activity equals the signal variance; mobility is non-negative."""
    h = tx.hjorth_parameters(x)
    assert abs(float(h["activity"]) - float(np.var(x))) <= 1.0e-6 * (1.0 + float(np.var(x)))
    assert float(h["mobility"]) >= 0.0


@given(
    freq=st.floats(min_value=0.02, max_value=0.2),
    phase=st.floats(min_value=0.0, max_value=2.0 * np.pi),
)
@settings(max_examples=30)
def test_hjorth_complexity_of_sine_is_one(freq: float, phase: float) -> None:
    """For a pure sinusoid the Hjorth complexity is ~1 (its defining property)."""
    n = 4096
    x = sinusoid(n, freq=freq, phase=phase)
    h = tx.hjorth_parameters(x)
    # Complexity == mobility(x')/mobility(x) → exactly 1 for a sine; finite
    # sampling + discrete differences leave a small bias, so ~0.1 tolerance.
    assert abs(float(h["complexity"]) - 1.0) <= 0.1


# ---------------------------------------------------------------------------
# extract_features — keys match feature_names(); values are finite scalars
# ---------------------------------------------------------------------------


@given(x=finite_signals(min_size=128, max_size=512))
def test_extract_features_full_catalogue(x: np.ndarray) -> None:
    """The full feature bag is keyed by feature_names() with finite scalar values."""
    feats = tx.extract_features(x, fs=1.0)
    assert set(feats) == set(tx.feature_names())
    for name, value in feats.items():
        v = np.asarray(value)
        assert v.ndim == 0, f"1-D input → scalar feature, got shape {v.shape} for {name!r}"
        assert np.isfinite(v), f"feature {name!r} is not finite"


@given(
    x=finite_signals(min_size=128, max_size=512),
    subset=st.lists(st.sampled_from(tx.feature_names()), min_size=1, max_size=5, unique=True),
)
def test_extract_features_subset(x: np.ndarray, subset: list[str]) -> None:
    """A requested subset returns exactly those keys (and only those)."""
    feats = tx.extract_features(x, fs=1.0, features=subset)
    assert set(feats) == set(subset)


@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=15)
def test_extract_features_multichannel_shape(seed: int) -> None:
    """For a (T, C) input every feature value has shape (C,)."""
    n, channels = 1024, 3
    rng = np.random.default_rng(seed)
    # Three distinct, non-constant channels (sinusoids of different frequency).
    cols = [sinusoid(n, freq=f) + 1.0e-3 * rng.standard_normal(n) for f in (0.05, 0.1, 0.2)]
    x = np.column_stack(cols)
    feats = tx.extract_features(x, fs=1.0)
    assert set(feats) == set(tx.feature_names())
    for name, value in feats.items():
        v = np.asarray(value)
        assert v.shape == (channels,), f"feature {name!r} has shape {v.shape}, want ({channels},)"
        assert np.all(np.isfinite(v)), f"feature {name!r} has non-finite values"
