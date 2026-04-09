"""
Lyapunov spectrum tests.

These tests verify:
  1. Return shapes are correct.
  2. For well-known chaotic systems, the largest LE is positive.
  3. For dissipative systems, the sum of exponents is negative.
  4. For stable/periodic systems, all LEs are non-positive.

All tests are marked slow (JiTCODE / JiTCDDE compilation + long integration).
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ODE — Lorenz
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLorenzLyapunov:
    """Lorenz system canonical spectrum ≈ [0.91, ~0, -14.57]."""

    @pytest.fixture(scope="class")
    def lorenz_spectrum(self):
        from tsdynamics.systems.continuous.chaotic_attractors import Lorenz

        lor = Lorenz(initial_conds=[1.0, 1.0, 1.0])
        return lor.lyapunov_spectrum(
            dt=0.1,
            burn_in=50.0,
            final_time=200.0,
            method="dopri5",
            rtol=1e-6,
            atol=1e-9,
        )

    def test_shape(self, lorenz_spectrum):
        assert lorenz_spectrum.shape == (3,)

    def test_largest_is_positive(self, lorenz_spectrum):
        assert lorenz_spectrum[0] > 0.0, (
            f"Lorenz LE1 should be positive, got {lorenz_spectrum[0]:.4f}"
        )

    def test_smallest_is_negative(self, lorenz_spectrum):
        assert lorenz_spectrum[-1] < 0.0, (
            f"Lorenz LE3 should be strongly negative, got {lorenz_spectrum[-1]:.4f}"
        )

    def test_sorted_descending(self, lorenz_spectrum):
        assert lorenz_spectrum[0] >= lorenz_spectrum[1] >= lorenz_spectrum[2], (
            f"LEs must be in descending order: {lorenz_spectrum}"
        )

    def test_sum_is_negative(self, lorenz_spectrum):
        """Lorenz is dissipative: LE1+LE2+LE3 = -(sigma+1+beta) ≈ -13.67."""
        s = lorenz_spectrum.sum()
        assert s < 0.0, f"Sum of Lorenz LEs must be negative (dissipative), got {s:.4f}"
        # Loose range: should be near -13.67
        assert -20.0 < s < -5.0, f"Lorenz LE sum {s:.4f} is far from expected ~-13.67"

    def test_partial_spectrum(self):
        """n_lyap < n_dim should return correct-sized array."""
        from tsdynamics.systems.continuous.chaotic_attractors import Lorenz

        lor = Lorenz(initial_conds=[1.0, 1.0, 1.0])
        exps = lor.lyapunov_spectrum(dt=0.1, burn_in=30.0, final_time=100.0, n_lyap=2)
        assert exps.shape == (2,)
        assert exps[0] > 0.0  # largest must still be positive


# ---------------------------------------------------------------------------
# ODE — Rossler
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rossler_largest_le_positive():
    """Rossler is chaotic: LE1 > 0."""
    from tsdynamics.systems.continuous.chaotic_attractors import Rossler

    r = Rossler(initial_conds=[1.0, 0.0, 0.0])
    exps = r.lyapunov_spectrum(dt=0.1, burn_in=30.0, final_time=150.0)
    assert exps[0] > 0.0, f"Rossler LE1 should be positive, got {exps[0]:.4f}"
    assert exps.shape == (3,)


@pytest.mark.slow
def test_rossler_sum_negative():
    """Rossler is dissipative."""
    from tsdynamics.systems.continuous.chaotic_attractors import Rossler

    r = Rossler(initial_conds=[1.0, 0.0, 0.0])
    exps = r.lyapunov_spectrum(dt=0.1, burn_in=30.0, final_time=150.0)
    assert exps.sum() < 0.0


# ---------------------------------------------------------------------------
# ODE — hyperchaotic system (HyperRossler): two positive LEs
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_hyperbao_two_positive_les():
    """HyperBao is a 4D hyperchaotic system: should have two positive LEs."""
    from tsdynamics.systems.continuous.exotic_systems import HyperBao

    hb = HyperBao()
    exps = hb.lyapunov_spectrum(
        n_lyap=2,
        dt=0.05,
        burn_in=50.0,
        final_time=200.0,
        method="dop853",
        rtol=1e-5,
        atol=1e-8,
    )
    assert exps.shape == (2,)
    assert exps[0] > 0.0, f"HyperBao LE1 should be positive, got {exps}"
    assert exps[1] > 0.0, f"HyperBao LE2 should be positive (hyperchaotic), got {exps}"


# ---------------------------------------------------------------------------
# Discrete map — Henon
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHenonLyapunov:
    """Henon map canonical: LE1 ≈ 0.42, LE2 ≈ -1.62."""

    @pytest.fixture(scope="class")
    def henon_spectrum(self):
        from tsdynamics.systems.discrete.chaotic_maps import Henon

        h = Henon()
        return h.lyapunov_spectrum(steps=5000)

    def test_shape(self, henon_spectrum):
        assert henon_spectrum.shape == (2,)

    def test_largest_positive(self, henon_spectrum):
        assert henon_spectrum[0] > 0.0, f"Henon LE1 should be positive, got {henon_spectrum[0]:.4f}"

    def test_smallest_negative(self, henon_spectrum):
        assert henon_spectrum[1] < 0.0, f"Henon LE2 should be negative, got {henon_spectrum[1]:.4f}"

    def test_sorted_descending(self, henon_spectrum):
        assert henon_spectrum[0] > henon_spectrum[1]


# ---------------------------------------------------------------------------
# Discrete map — Logistic (r=4, fully chaotic)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_logistic_lyapunov_positive():
    """Logistic map at r=4: LE ≈ ln(2) ≈ 0.693."""
    from tsdynamics.systems.discrete.population_maps import Logistic

    m = Logistic(params={"r": 4.0})
    exps = m.lyapunov_spectrum(steps=5000)
    assert exps.shape == (1,)
    assert exps[0] > 0.0, f"Logistic(r=4) LE should be positive, got {exps[0]:.4f}"
    # Should be near ln(2)
    assert 0.3 < exps[0] < 1.2, f"Logistic LE {exps[0]:.4f} far from ln(2)≈0.693"


@pytest.mark.slow
def test_logistic_stable_regime():
    """Logistic map at r=2 (stable fixed point): LE should be negative."""
    from tsdynamics.systems.discrete.population_maps import Logistic

    m = Logistic(params={"r": 2.0})
    exps = m.lyapunov_spectrum(steps=5000)
    assert exps[0] < 0.0, f"Logistic(r=2) LE should be negative, got {exps[0]:.4f}"


# ---------------------------------------------------------------------------
# Discrete map — partial spectrum
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_map_partial_spectrum():
    """num_exponents < n_dim returns the right shape."""
    from tsdynamics.systems.discrete.chaotic_maps import FoldedTowel

    m = FoldedTowel()
    exps = m.lyapunov_spectrum(steps=3000, num_exponents=2)
    assert exps.shape == (2,)


# ---------------------------------------------------------------------------
# DDE — MackeyGlass (chaotic: LE1 > 0)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mackeyglass_lyapunov_positive():
    """MackeyGlass in chaotic regime: largest LE should be positive."""
    from tsdynamics.systems.continuous.delayed_systems import MackeyGlass

    mg = MackeyGlass()
    history = lambda s: [1.5 + 0.05 * np.sin(0.1 * s)]  # noqa: E731
    exps = mg.lyapunov_spectrum(
        n_lyap=1,
        dt=0.2,
        burn_in=100.0,
        final_time=1000.0,
        history=history,
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (1,)
    # MackeyGlass with tau=17 is in the chaotic regime
    assert exps[0] > 0.0, f"MackeyGlass LE1 should be positive, got {exps[0]:.4f}"


@pytest.mark.slow
def test_mackeyglass_lyapunov_n_lyap_2():
    """Requesting n_lyap=2 returns two exponents."""
    from tsdynamics.systems.continuous.delayed_systems import MackeyGlass

    mg = MackeyGlass()
    history = lambda s: [1.5 + 0.05 * np.sin(0.1 * s)]  # noqa: E731
    exps = mg.lyapunov_spectrum(
        n_lyap=2,
        dt=0.2,
        burn_in=30.0,
        final_time=100.0,
        history=history,
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (2,)
