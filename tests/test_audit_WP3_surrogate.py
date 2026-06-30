"""Regression test for WP3_surrogate (finding A6:A6-1).

The Fourier (phase-randomisation) surrogate must carry the DC term — and, for
even ``N``, the Nyquist term — through at its original *signed* real value rather
than its magnitude.  Forcing those bins positive (the pre-fix behaviour) sign-flips
the mean of any negative-mean series, so the constrained realisation no longer
preserves the series mean (Theiler et al., 1992).

These tests fail on the pre-fix logic (which set ``phases[:, 0] = 0`` over the
non-negative ``magnitude``, yielding a positive DC term) and pass after.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.analysis.surrogate.generators import fourier_surrogate


def _random_walk(n: int, mean_offset: float, seed: int) -> np.ndarray:
    """A negative-mean series: a random walk shifted to the requested mean."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n))
    return walk - walk.mean() + mean_offset


def test_fourier_surrogate_preserves_negative_mean_odd_n() -> None:
    """Surrogate of a negative-mean odd-length series keeps the (signed) mean."""
    data = _random_walk(65, mean_offset=-5.0, seed=0)
    assert data.mean() < 0.0
    surr = fourier_surrogate(data, n=8, seed=1)
    # Every surrogate's mean equals the data mean to floating-point tolerance.
    np.testing.assert_allclose(surr.mean(axis=1), data.mean(), atol=1e-9)


def test_fourier_surrogate_preserves_signed_dc_and_nyquist_even_n() -> None:
    """Even N: both DC and Nyquist rfft bins are preserved with sign."""
    data = _random_walk(64, mean_offset=-3.0, seed=2)
    assert data.mean() < 0.0
    surr = fourier_surrogate(data, n=8, seed=3)

    data_spec = np.fft.rfft(data)
    surr_spec = np.fft.rfft(surr, axis=1)
    # DC bin (mean * N) preserved with sign on every surrogate.
    np.testing.assert_allclose(surr_spec[:, 0].real, data_spec[0].real, atol=1e-7)
    # Nyquist bin (real for even N) preserved with sign — the pre-fix code forced
    # it to +|X_Nyq|, flipping it whenever the original was negative.
    np.testing.assert_allclose(surr_spec[:, -1].real, data_spec[-1].real, atol=1e-7)


def test_fourier_surrogate_still_preserves_magnitude_spectrum() -> None:
    """The fix is magnitude-preserving: |X_k| is unchanged (answer-preserving)."""
    data = _random_walk(64, mean_offset=-3.0, seed=4)
    surr = fourier_surrogate(data, n=4, seed=5)
    mag_data = np.abs(np.fft.rfft(data))
    mag_surr = np.abs(np.fft.rfft(surr, axis=1))
    np.testing.assert_allclose(mag_surr, np.broadcast_to(mag_data, mag_surr.shape), atol=1e-7)
    # And the inverse transform stays real-valued.
    assert np.isrealobj(surr)
