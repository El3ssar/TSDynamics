"""
Lyapunov-spectrum correctness tests.

These verify the canonical values for a small set of well-studied systems:

- ODE: Lorenz, Rossler, hyperchaotic HyperBao
- Map: Henon, Logistic (both chaotic and stable regimes)
- DDE: MackeyGlass

All tests are marked ``slow`` (JiTCODE/JiTCDDE compilation + long integration).
We compare to ranges, not exact scalars, because convergence depends on
tolerances and integration window.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Lorenz: ~ [0.91, 0, -14.57]
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLorenzLyapunov:
    @pytest.fixture(scope="class")
    def spectrum(self) -> np.ndarray:
        import tsdynamics as ts

        lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
        return lor.lyapunov_spectrum(
            dt=0.1,
            burn_in=50.0,
            final_time=200.0,
            method="dop853",
            rtol=1e-7,
            atol=1e-10,
        )

    def test_shape(self, spectrum: np.ndarray) -> None:
        assert spectrum.shape == (3,)

    def test_largest_positive(self, spectrum: np.ndarray) -> None:
        assert spectrum[0] > 0.5, f"Lorenz LE1 ≈ 0.91, got {spectrum[0]:.4f}"

    def test_middle_near_zero(self, spectrum: np.ndarray) -> None:
        assert abs(spectrum[1]) < 0.2, f"Lorenz LE2 ≈ 0, got {spectrum[1]:.4f}"

    def test_smallest_strongly_negative(self, spectrum: np.ndarray) -> None:
        assert spectrum[2] < -10.0, f"Lorenz LE3 ≈ -14.57, got {spectrum[2]:.4f}"

    def test_sorted_descending(self, spectrum: np.ndarray) -> None:
        assert spectrum[0] >= spectrum[1] >= spectrum[2]

    def test_sum_dissipative(self, spectrum: np.ndarray) -> None:
        # divergence of Lorenz = -(sigma + 1 + beta) ≈ -13.67
        s = spectrum.sum()
        assert -20.0 < s < -5.0, f"Lorenz LE sum {s:.4f} far from -13.67"

    def test_partial_spectrum_n_exp_2(self) -> None:
        import tsdynamics as ts

        lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
        exps = lor.lyapunov_spectrum(dt=0.1, burn_in=30.0, final_time=100.0, n_exp=2)
        assert exps.shape == (2,)
        assert exps[0] > 0.0


# ---------------------------------------------------------------------------
# Rossler — single positive LE
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rossler_lyapunov() -> None:
    import tsdynamics as ts

    r = ts.Rossler(ic=[1.0, 0.0, 0.0])
    exps = r.lyapunov_spectrum(dt=0.1, burn_in=50.0, final_time=200.0)
    assert exps.shape == (3,)
    assert exps[0] > 0.0
    assert exps.sum() < 0.0


# ---------------------------------------------------------------------------
# HyperBao — 4D hyperchaotic: two positive LEs
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_hyperbao_two_positive_les() -> None:
    import tsdynamics as ts

    hb = ts.HyperBao()
    exps = hb.lyapunov_spectrum(
        n_exp=2,
        dt=0.05,
        burn_in=50.0,
        final_time=200.0,
        method="dop853",
        rtol=1e-6,
        atol=1e-8,
    )
    assert exps.shape == (2,)
    assert exps[0] > 0.0
    assert exps[1] > 0.0


# ---------------------------------------------------------------------------
# Henon: ~ [0.42, -1.62]
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHenonLyapunov:
    @pytest.fixture(scope="class")
    def spectrum(self) -> np.ndarray:
        import tsdynamics as ts

        return ts.Henon().lyapunov_spectrum(steps=10_000)

    def test_shape(self, spectrum: np.ndarray) -> None:
        assert spectrum.shape == (2,)

    def test_largest_positive(self, spectrum: np.ndarray) -> None:
        assert 0.2 < spectrum[0] < 0.6, f"Henon LE1 ≈ 0.42, got {spectrum[0]:.4f}"

    def test_smallest_negative(self, spectrum: np.ndarray) -> None:
        assert -2.0 < spectrum[1] < -1.0, f"Henon LE2 ≈ -1.62, got {spectrum[1]:.4f}"


# ---------------------------------------------------------------------------
# Logistic map
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_logistic_chaotic_r4_near_ln2() -> None:
    """Logistic at r=4 is fully chaotic with LE = ln(2) ≈ 0.693."""
    import tsdynamics as ts

    m = ts.Logistic(params={"r": 4.0})
    exps = m.lyapunov_spectrum(steps=10_000)
    assert exps.shape == (1,)
    assert 0.5 < exps[0] < 0.9


@pytest.mark.slow
def test_logistic_stable_r2_negative() -> None:
    """Logistic at r=2 is at a stable fixed point: LE < 0."""
    import tsdynamics as ts

    m = ts.Logistic(params={"r": 2.0})
    exps = m.lyapunov_spectrum(steps=5_000)
    assert exps[0] < 0.0


# ---------------------------------------------------------------------------
# MackeyGlass DDE — chaotic at tau=17
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mackeyglass_lyapunov_positive() -> None:
    """
    Mackey-Glass with the default tau=17 is in the chaotic regime; LE1 > 0.

    We first integrate from a non-trivial history to seed the trajectory on
    the attractor, then call ``lyapunov_spectrum`` with ``ic=traj.y[-1]``.
    """
    import tsdynamics as ts

    mg = ts.MackeyGlass()
    hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]  # noqa: E731
    traj = mg.integrate(final_time=200.0, dt=0.5, history=hist, rtol=1e-4, atol=1e-4)
    exps = mg.lyapunov_spectrum(
        n_exp=1,
        dt=0.5,
        burn_in=100.0,
        final_time=1000.0,
        ic=traj.y[-1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (1,)
    assert exps[0] > 0.0, f"MackeyGlass LE1 should be > 0, got {exps[0]:.5f}"


@pytest.mark.slow
def test_mackeyglass_lyapunov_n_exp_2() -> None:
    """Requesting two exponents returns two finite values."""
    import tsdynamics as ts

    mg = ts.MackeyGlass()
    traj = mg.integrate(
        final_time=200.0,
        dt=0.5,
        history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)],
        rtol=1e-4,
        atol=1e-4,
    )
    exps = mg.lyapunov_spectrum(
        n_exp=2,
        dt=0.5,
        burn_in=50.0,
        final_time=300.0,
        ic=traj.y[-1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (2,)
    assert np.all(np.isfinite(exps))
