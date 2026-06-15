r"""
Tests for the surrogate-data / nonlinearity-test toolkit (stream **A-SURR**).

The headline acceptance is literature-validated and self-contained (no engine, so
the suite stays in the fast tier):

- **Rejects linearity for Lorenz.**  A dissipative chaotic flow is strongly
  time-irreversible, so its time-reversal asymmetry is a gross outlier of the
  IAAFT surrogate ensemble that shares its spectrum and amplitude distribution —
  the linear-Gaussian null is rejected (Theiler et al., *Physica D* **58**, 77,
  1992; Schreiber & Schmitz, *Phys. Rev. Lett.* **77**, 635, 1996).  A genuinely
  linear AR(1) process is *not* rejected.
- **Determinism is detected by prediction error.**  A locally-constant predictor
  is far more accurate on a deterministic map than on its linear surrogates.

Plus the structural contracts each generator must satisfy: the constraints it is
designed to preserve (amplitude distribution and/or power spectrum) hold to
tolerance, and a fixed seed makes the whole pipeline reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis import surrogate as sur
from tsdynamics.data import Trajectory

# ── self-contained data generators (no engine) ───────────────────────────────────


def _lorenz(
    n: int = 4000,
    dt: float = 0.01,
    skip: int = 5,
    transient: int = 2000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    ic=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """An RK4-integrated Lorenz orbit, shape ``(n, 3)`` (sampling dt = dt*skip)."""

    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    s = np.asarray(ic, dtype=float)

    def step(s):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    for _ in range(transient):
        s = step(s)
    out = np.empty((n, 3))
    for i in range(n):
        for _ in range(skip):
            s = step(s)
        out[i] = s
    return out


def _ar1(n: int = 4000, phi: float = 0.5, seed: int = 0) -> np.ndarray:
    """A linear Gaussian AR(1) process — time-reversible, the linear null itself."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 500)
    x = np.empty(n + 500)
    x[0] = e[0]
    for t in range(1, n + 500):
        x[t] = phi * x[t - 1] + e[t]
    return x[500:]


def _logistic(r: float = 4.0, n: int = 3000, transient: int = 500, x0: float = 0.123) -> np.ndarray:
    x = x0
    for _ in range(transient):
        x = r * x * (1.0 - x)
    out = np.empty(n)
    for i in range(n):
        x = r * x * (1.0 - x)
        out[i] = x
    return out


@pytest.fixture(scope="module")
def lorenz_z():
    # The z-component is strongly time-irreversible (it is bounded below and
    # asymmetric); the x/y components inherit the system's (x,y)->(-x,-y) symmetry
    # and are nearly time-symmetric, so the third-moment statistic barely separates
    # them from their surrogates.  Use z for the time-reversal headline.
    return _lorenz()[:, 2]


@pytest.fixture(scope="module")
def lorenz_x():
    return _lorenz()[:, 0]


@pytest.fixture(scope="module")
def ar1():
    return _ar1()


# ── generator structural contracts ───────────────────────────────────────────────


def test_random_shuffle_preserves_distribution_and_is_a_permutation():
    x = _ar1(n=500)
    s = sur.random_shuffle(x, n=3, seed=1)
    assert s.shape == (3, 500)
    for row in s:
        # a permutation: same multiset of values
        assert np.array_equal(np.sort(row), np.sort(x))


def test_fourier_surrogate_preserves_power_spectrum(ar1):
    s = sur.fourier_surrogate(ar1, n=4, seed=2)
    assert s.shape == (4, ar1.size)
    assert np.isrealobj(s)
    target = np.abs(np.fft.rfft(ar1))
    for row in s:
        assert np.allclose(np.abs(np.fft.rfft(row)), target, atol=1e-8, rtol=1e-6)


def test_aaft_surrogate_preserves_amplitude_distribution(ar1):
    s = sur.aaft_surrogate(ar1, n=3, seed=3)
    assert s.shape == (3, ar1.size)
    for row in s:
        # AAFT restores the exact amplitude distribution by rank remap
        assert np.allclose(np.sort(row), np.sort(ar1))


def test_iaaft_surrogate_matches_distribution_exactly_and_spectrum_closely(ar1):
    s = sur.iaaft_surrogate(ar1, n=3, seed=4)
    assert s.shape == (3, ar1.size)
    target_mag = np.abs(np.fft.rfft(ar1))
    spectral_power = float(np.sum(target_mag**2))
    for row in s:
        # exact amplitude distribution (ends on the amplitude-matching step)
        assert np.allclose(np.sort(row), np.sort(ar1))
        # power spectrum matched to high accuracy
        mag = np.abs(np.fft.rfft(row))
        rel_spec_err = float(np.sum((mag - target_mag) ** 2) / spectral_power)
        assert rel_spec_err < 1e-3


def test_generators_are_reproducible_with_seed(ar1):
    for method in ("shuffle", "ft", "aaft", "iaaft"):
        a = sur.surrogates(ar1, method, n=2, seed=7)
        b = sur.surrogates(ar1, method, n=2, seed=7)
        assert np.array_equal(a, b)


def test_surrogates_dispatch_aliases_and_errors(ar1):
    assert np.array_equal(
        sur.surrogates(ar1, "random", n=1, seed=9),
        sur.surrogates(ar1, "shuffle", n=1, seed=9),
    )
    assert np.array_equal(
        sur.surrogates(ar1, "fourier", n=1, seed=9),
        sur.surrogates(ar1, "ft", n=1, seed=9),
    )
    with pytest.raises(ValueError, match="unknown surrogate method"):
        sur.surrogates(ar1, "nope")


# ── p-value arithmetic ───────────────────────────────────────────────────────────


def test_empirical_pvalue_extremes_and_centre():
    surr = np.arange(39, dtype=float)
    # data above every surrogate → minimal one/two-sided p
    assert sur._common.empirical_pvalue(100.0, surr, "greater") == pytest.approx(1 / 40)
    assert sur._common.empirical_pvalue(100.0, surr, "two") == pytest.approx(2 / 40)
    assert sur._common.empirical_pvalue(-100.0, surr, "less") == pytest.approx(1 / 40)
    # data in the centre → far from rejection
    assert sur._common.empirical_pvalue(19.0, surr, "two") > 0.9


def test_empirical_pvalue_bad_tail():
    with pytest.raises(ValueError, match="tail must be"):
        sur._common.empirical_pvalue(0.0, np.arange(5.0), "sideways")


# ── discriminating statistics ─────────────────────────────────────────────────────


def test_time_reversal_asymmetry_zero_for_symmetric_large_for_lorenz(lorenz_z):
    t = np.linspace(0.0, 200.0, 4000)
    sine = np.sin(2.0 * np.pi * 0.5 * t)
    # a time-symmetric signal has (near) zero asymmetry
    assert abs(sur.time_reversal_asymmetry(sine)) < 1e-2
    # the Lorenz flow is strongly irreversible
    assert abs(sur.time_reversal_asymmetry(lorenz_z)) > 0.2


def test_time_reversal_asymmetry_constant_series_is_zero():
    assert sur.time_reversal_asymmetry(np.full(100, 3.0)) == 0.0


def test_prediction_error_low_for_deterministic_high_for_noise():
    logistic = _logistic()
    noise = np.random.default_rng(0).standard_normal(3000)
    det_err = sur.nonlinear_prediction_error(logistic, m=2, tau=1)
    noise_err = sur.nonlinear_prediction_error(noise, m=2, tau=1)
    assert det_err < 0.1  # near-perfect local prediction of a deterministic map
    assert noise_err > 0.8  # white noise is essentially unpredictable
    assert det_err < noise_err


# ── headline acceptance: reject linearity for Lorenz ──────────────────────────────


def test_surrogate_test_rejects_linearity_for_lorenz(lorenz_z, ar1):
    res = sur.surrogate_test(lorenz_z, "time_reversal", method="iaaft", n=39, seed=0)
    assert res.rejected
    assert res.p_value <= 0.05
    assert abs(res.z_score) > 5.0  # a gross outlier of the surrogate ensemble

    # the genuinely linear AR(1) process is not flagged …
    linear = sur.surrogate_test(ar1, "time_reversal", method="iaaft", n=39, seed=0)
    assert not linear.rejected
    # … and is far less significant than Lorenz
    assert abs(linear.z_score) < abs(res.z_score)


def test_surrogate_test_prediction_error_detects_determinism_in_lorenz(lorenz_x):
    # determinism makes the flow far more predictable than its linear surrogates
    res = sur.surrogate_test(
        lorenz_x, "prediction_error", method="iaaft", n=19, seed=0, statistic_kwargs={"m": 3}
    )
    assert res.rejected
    assert res.z_score < -5.0


def test_prediction_error_validates_parameters():
    x = np.random.default_rng(0).standard_normal(500)
    with pytest.raises(ValueError, match="theiler must be"):
        sur.nonlinear_prediction_error(x, theiler=-1)
    with pytest.raises(ValueError, match="n_neighbors must be"):
        sur.nonlinear_prediction_error(x, n_neighbors=0)
    with pytest.raises(ValueError, match=">= 1"):
        sur.nonlinear_prediction_error(x, m=0)


def test_surrogate_test_prediction_error_detects_determinism():
    logistic = _logistic()
    res = sur.surrogate_test(
        logistic, "prediction_error", method="iaaft", n=19, seed=1, statistic_kwargs={"m": 2}
    )
    assert res.tail == "less"  # auto-resolved for prediction error
    assert res.rejected
    assert res.p_value <= 0.05


def test_surrogate_test_accepts_callable_statistic(ar1):
    res = sur.surrogate_test(ar1, statistic=np.std, method="ft", n=19, seed=2)
    assert res.statistic == "std"
    assert res.n_surrogates == 19
    assert res.surrogate_statistics.shape == (19,)


def test_surrogate_test_unknown_statistic(ar1):
    with pytest.raises(ValueError, match="unknown statistic"):
        sur.surrogate_test(ar1, "bogus")


# ── Trajectory / component plumbing ───────────────────────────────────────────────


def test_component_selection_from_array_and_trajectory(lorenz_z):
    two_col = np.column_stack([lorenz_z, np.zeros_like(lorenz_z)])
    with pytest.raises(ValueError, match="multiple columns"):
        sur.surrogate_test(two_col, "time_reversal", n=39, seed=0)
    res = sur.surrogate_test(two_col, "time_reversal", n=39, seed=0, component=0)
    assert res.rejected

    traj = Trajectory(t=np.arange(lorenz_z.size, dtype=float), y=lorenz_z[:, None], system=None)
    res_traj = sur.surrogate_test(traj, "time_reversal", n=39, seed=0)
    assert res_traj.rejected


# ── public API & registry ─────────────────────────────────────────────────────────


def test_top_level_exports():
    assert ts.surrogate_test is sur.surrogate_test
    assert ts.surrogates is sur.surrogates
    assert ts.SurrogateTest is sur.SurrogateTest


def test_self_registration():
    assert registry.analyses.get("surrogate_test") is sur.surrogate_test
    assert registry.analyses.get("surrogates") is sur.surrogates
    assert registry.analyses.entry("surrogate_test").metadata["family"] == "surrogate"


def test_repr_is_informative(ar1):
    res = sur.surrogate_test(ar1, "time_reversal", n=9, seed=0)
    text = repr(res)
    assert "SurrogateTest" in text
    assert "time_reversal" in text
    assert "p=" in text
