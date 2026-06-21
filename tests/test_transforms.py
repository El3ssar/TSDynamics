"""
Tests for the signal/feature transform layer (stream T-XFORM).

Three groups — spectral estimators, shape-preserving preprocessing, and generic
feature extraction — plus the registration/discovery wiring and the
``Trajectory``/array input contract.  Reference values are chosen to be
analytically obvious (a pure tone, a linear ramp, white noise) so the assertions
double as documentation of what each transform means.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics import registry
from tsdynamics import transforms as tx
from tsdynamics.data import Trajectory

FS = 200.0
F_LOW = 5.0
F_HIGH = 40.0


@pytest.fixture
def time() -> np.ndarray:
    """20 s at 200 Hz (4000 samples) — long enough for stable spectra/filters."""
    return np.linspace(0.0, 20.0, int(20 * FS), endpoint=False)


@pytest.fixture
def two_tone(time: np.ndarray) -> np.ndarray:
    """A 5 Hz tone plus a half-amplitude 40 Hz tone."""
    return np.sin(2 * np.pi * F_LOW * time) + 0.5 * np.sin(2 * np.pi * F_HIGH * time)


@pytest.fixture
def white(time: np.ndarray) -> np.ndarray:
    """Reproducible standard-normal white noise of matching length."""
    return np.random.default_rng(1234).standard_normal(time.size)


# ---------------------------------------------------------------------------
# Registration & public surface
# ---------------------------------------------------------------------------


def test_intree_transforms_registered() -> None:
    """Every in-tree transform self-registers under its name with metadata."""
    expected = {
        "power_spectral_density",
        "spectral_entropy",
        "spectral_centroid",
        "dominant_frequency",
        "detrend",
        "normalize",
        "lowpass",
        "highpass",
        "bandpass",
        "bandstop",
        "extract_features",
    }
    assert expected <= set(registry.transforms.names())
    entry = registry.transforms.entry("power_spectral_density")
    assert entry.obj is tx.power_spectral_density
    assert entry.metadata["kind"] == "spectral"


def test_public_all_is_importable() -> None:
    """Everything in ``__all__`` resolves to a real attribute."""
    for name in tx.__all__:
        assert hasattr(tx, name), name


def test_registration_is_idempotent() -> None:
    """Re-running in-tree registration does not raise or duplicate."""
    before = list(registry.transforms.names())
    tx._register_intree()
    assert registry.transforms.names() == before


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------


def test_psd_peaks_at_tone(two_tone: np.ndarray) -> None:
    """The periodogram peak sits on the dominant 5 Hz line."""
    freqs, psd = tx.power_spectral_density(two_tone, fs=FS, method="periodogram")
    assert psd.shape == freqs.shape
    assert abs(float(freqs[np.argmax(psd)]) - F_LOW) < 0.2


def test_dominant_frequency(two_tone: np.ndarray) -> None:
    """``dominant_frequency`` recovers the strongest component."""
    assert abs(tx.dominant_frequency(two_tone, fs=FS, method="periodogram") - F_LOW) < 0.2


def test_spectral_entropy_orders_sine_below_noise(two_tone: np.ndarray, white: np.ndarray) -> None:
    """A near-periodic signal has far lower spectral entropy than white noise."""
    h_sine = tx.spectral_entropy(two_tone, fs=FS)
    h_noise = tx.spectral_entropy(white, fs=FS)
    assert 0.0 <= h_sine < 0.6
    assert h_noise > 0.9
    assert h_noise <= 1.0 + 1e-9


def test_spectral_entropy_unnormalized_is_larger(two_tone: np.ndarray) -> None:
    """Dropping the log(N) normalisation yields the raw (bits) entropy."""
    norm = tx.spectral_entropy(two_tone, fs=FS, normalize=True)
    raw = tx.spectral_entropy(two_tone, fs=FS, normalize=False)
    assert raw > norm


def test_spectral_centroid_between_tones(two_tone: np.ndarray) -> None:
    """The power-weighted mean frequency lies between the two tones."""
    centroid = tx.spectral_centroid(two_tone, fs=FS)
    assert F_LOW < centroid < F_HIGH


def test_spectral_multichannel_returns_per_channel(two_tone: np.ndarray, white: np.ndarray) -> None:
    """A 2-D signal yields one entropy per column, sine below noise."""
    sig = np.column_stack([two_tone, white])
    ent = tx.spectral_entropy(sig, fs=FS)
    assert ent.shape == (2,)
    assert ent[0] < ent[1]


def test_unknown_psd_method_raises(two_tone: np.ndarray) -> None:
    with pytest.raises(ValueError, match="unknown PSD method"):
        tx.power_spectral_density(two_tone, fs=FS, method="bogus")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_detrend_linear_removes_ramp(time: np.ndarray, two_tone: np.ndarray) -> None:
    """A linear ramp is removed; the residual has ~zero slope and mean."""
    ramped = two_tone + 0.3 * time
    out = tx.detrend(ramped, method="linear")
    slope = np.polyfit(time, out, 1)[0]
    assert abs(slope) < 1e-9
    assert abs(float(out.mean())) < 1e-9


def test_detrend_constant_removes_mean(two_tone: np.ndarray) -> None:
    out = tx.detrend(two_tone + 7.0, method="constant")
    assert abs(float(out.mean())) < 1e-9


def test_detrend_bad_kind_raises(two_tone: np.ndarray) -> None:
    with pytest.raises(ValueError, match="detrend method"):
        tx.detrend(two_tone, method="quadratic")


def test_normalize_zscore(two_tone: np.ndarray) -> None:
    out = tx.normalize(two_tone, method="zscore")
    assert abs(float(out.mean())) < 1e-12
    assert abs(float(out.std()) - 1.0) < 1e-12


def test_normalize_minmax(two_tone: np.ndarray) -> None:
    out = tx.normalize(two_tone, method="minmax")
    assert float(out.min()) == pytest.approx(0.0, abs=1e-12)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-12)


def test_normalize_constant_channel_is_finite() -> None:
    """A constant signal must not divide by zero."""
    out = tx.normalize(np.full(100, 3.0), method="zscore")
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_normalize_bad_method_raises(two_tone: np.ndarray) -> None:
    with pytest.raises(ValueError, match="unknown normalize method"):
        tx.normalize(two_tone, method="whiten")


def test_lowpass_attenuates_high_tone(two_tone: np.ndarray) -> None:
    """A 10 Hz low-pass leaves the 5 Hz tone and removes the 40 Hz one."""
    out = tx.lowpass(two_tone, 10.0, fs=FS)
    assert out.shape == two_tone.shape  # zero-phase keeps length
    assert abs(tx.dominant_frequency(out, fs=FS, method="periodogram") - F_LOW) < 0.5


def test_bandpass_isolates_high_tone(two_tone: np.ndarray) -> None:
    """A band around 40 Hz isolates the high tone."""
    out = tx.bandpass(two_tone, 30.0, 50.0, fs=FS)
    assert abs(tx.dominant_frequency(out, fs=FS, method="periodogram") - F_HIGH) < 1.0


def test_highpass_and_bandstop_run(two_tone: np.ndarray) -> None:
    hp = tx.highpass(two_tone, 20.0, fs=FS)
    bs = tx.bandstop(two_tone, 30.0, 50.0, fs=FS)
    assert hp.shape == two_tone.shape
    assert bs.shape == two_tone.shape
    # band-stop around 40 Hz leaves the 5 Hz tone dominant.
    assert abs(tx.dominant_frequency(bs, fs=FS, method="periodogram") - F_LOW) < 0.5


def test_filter_cutoff_above_nyquist_raises(two_tone: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        tx.lowpass(two_tone, 150.0, fs=FS)  # Nyquist is 100 Hz


def test_bandpass_needs_pair() -> None:
    with pytest.raises(ValueError, match="pair"):
        tx.butter_filter(np.zeros(500), 10.0, btype="bandpass", fs=FS)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_extract_features_all_names(two_tone: np.ndarray) -> None:
    feats = tx.extract_features(two_tone, fs=FS)
    assert set(feats) == set(tx.feature_names())
    assert feats["rms"] == pytest.approx(float(np.sqrt(np.mean(two_tone**2))))


def test_extract_features_subset(two_tone: np.ndarray) -> None:
    feats = tx.extract_features(two_tone, fs=FS, features=["mean", "std"])
    assert set(feats) == {"mean", "std"}


def test_extract_features_unknown_raises(two_tone: np.ndarray) -> None:
    with pytest.raises(KeyError, match="unknown feature"):
        tx.extract_features(two_tone, fs=FS, features=["not_a_feature"])


def test_extract_features_multichannel(two_tone: np.ndarray, white: np.ndarray) -> None:
    sig = np.column_stack([two_tone, white])
    feats = tx.extract_features(sig, fs=FS)
    assert feats["mean"].shape == (2,)


def test_moments_match_numpy(white: np.ndarray) -> None:
    feats = tx.extract_features(white, features=["mean", "std", "var"])
    assert feats["mean"] == pytest.approx(float(white.mean()))
    assert feats["std"] == pytest.approx(float(white.std()))
    assert feats["var"] == pytest.approx(float(white.var()))


def test_skewness_kurtosis_of_constant_are_zero() -> None:
    """A constant channel has no shape — moments degrade to 0, not NaN."""
    feats = tx.extract_features(np.full(256, 2.0), features=["skewness", "kurtosis"])
    assert feats["skewness"] == 0.0
    assert feats["kurtosis"] == 0.0


def test_zero_crossing_rate_of_sine(time: np.ndarray) -> None:
    """A pure sine crosses zero ~2f times per second → rate 2f/fs per sample."""
    sine = np.sin(2 * np.pi * F_LOW * time)
    assert tx.zero_crossing_rate(sine) == pytest.approx(2 * F_LOW / FS, rel=0.05)


def test_hjorth_sine_complexity_near_one(time: np.ndarray) -> None:
    """Hjorth complexity is 1 for a pure sinusoid (the reference shape)."""
    hj = tx.hjorth_parameters(np.sin(2 * np.pi * 3.0 * time))
    assert hj["complexity"] == pytest.approx(1.0, abs=1e-3)
    assert hj["activity"] == pytest.approx(0.5, abs=1e-2)  # var of unit sine


def test_hjorth_multichannel(two_tone: np.ndarray, white: np.ndarray) -> None:
    hj = tx.hjorth_parameters(np.column_stack([two_tone, white]))
    assert hj["activity"].shape == (2,)


# ---------------------------------------------------------------------------
# Input contract: Trajectory ↔ array, fs/dt resolution
# ---------------------------------------------------------------------------


def _traj(time: np.ndarray, y: np.ndarray) -> Trajectory:
    return Trajectory(time, np.atleast_2d(y).T if y.ndim == 1 else y, system=None)


def test_trajectory_infers_fs_from_time(time: np.ndarray, two_tone: np.ndarray) -> None:
    """A Trajectory carries its sampling rate in ``t`` — no explicit fs needed."""
    traj = _traj(time, two_tone)
    df_traj = tx.dominant_frequency(traj, method="periodogram")
    df_arr = tx.dominant_frequency(two_tone, fs=FS, method="periodogram")
    assert df_traj == pytest.approx(df_arr)


def test_shape_preserving_transform_returns_trajectory(
    time: np.ndarray, two_tone: np.ndarray
) -> None:
    """detrend/normalize/filter on a Trajectory return a Trajectory with provenance."""
    traj = _traj(time, two_tone)
    out = tx.normalize(traj, method="zscore")
    assert isinstance(out, Trajectory)
    assert np.array_equal(out.t, traj.t)
    assert out.meta["normalized"] == "zscore"
    filt = tx.lowpass(traj, 10.0)
    assert isinstance(filt, Trajectory)
    assert filt.meta["filtered"]["btype"] == "lowpass"


def test_array_in_array_out(two_tone: np.ndarray) -> None:
    assert isinstance(tx.detrend(two_tone), np.ndarray)


def test_non_uniform_time_raises(two_tone: np.ndarray) -> None:
    """Spectral methods assume uniform sampling — a jittered grid is rejected."""
    t = np.cumsum(np.concatenate([[0.0], np.full(two_tone.size - 1, 1 / FS)]))
    t[100] += 0.05  # inject a gap
    traj = _traj(t, two_tone)
    with pytest.raises(ValueError, match="uniform"):
        tx.power_spectral_density(traj)


def test_fs_and_dt_together_raise(two_tone: np.ndarray) -> None:
    with pytest.raises(ValueError, match="at most one"):
        tx.power_spectral_density(two_tone, fs=FS, dt=1 / FS)


def test_scalar_and_highdim_inputs_raise() -> None:
    with pytest.raises(ValueError, match="scalar"):
        tx.detrend(3.0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        tx.detrend(np.zeros((4, 4, 4)))
