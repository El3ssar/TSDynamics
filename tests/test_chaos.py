r"""
Tests for the chaos indicators (stream **A-CHAOS**).

Each indicator is pinned to a literature value where one is exact or
well-established:

- **GALI** (Skokos et al. 2007/2008): the GALI\ :sub:`k` of a chaotic orbit
  decays like ``exp(-[(λ₁-λ₂)+…+(λ₁-λ_k)] t)``.  For the Lorenz flow the GALI₂
  decay rate must reproduce ``λ₁-λ₂ ≈ 0.906`` (Lorenz max exponent, λ₂=0); for a
  harmonic oscillator (regular) GALI₂ stays at 1.
- **0–1 test** (Gottwald & Melbourne 2004/2009): the logistic map gives
  ``K ≈ 1`` at ``r=4`` (chaos) and ``K ≈ 0`` at ``r=3.5`` (a period-4 cycle).
- **Expansion entropy** (Hunt & Ott 2015): the unit-height tent map has
  ``|f'| ≡ 2`` so ``H = ln 2`` exactly; the Hénon map reproduces its topological
  entropy ``≈ 0.465`` (Newhouse–Pignataro); a quasi-periodic circle map gives
  ``H ≈ 0``.

The map indicators run on Numba-compiled steps; the flow indicators use a
self-contained RK4 variational integrator over the SymEngine-lambdified RHS and
Jacobian, so none of these tests need the compile backends or the Rust engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import (
    Box,
    ContinuousSystem,
    ExpansionEntropyResult,
    GALIResult,
    expansion_entropy,
    gali,
    registry,
    zero_one_test,
)

LN2 = float(np.log(2.0))
LORENZ_GAP = 0.9056  # λ₁ - λ₂ for the classic Lorenz attractor (λ₂ = 0)
HENON_HTOP = 0.4651  # topological entropy of the Hénon map (Newhouse–Pignataro)


class _Harmonic(ContinuousSystem):
    """Undamped harmonic oscillator ``x' = v, v' = -w² x`` — a regular (λ=0) flow."""

    params = {"w": 1.0}
    dim = 2
    variables = ("x", "v")

    @staticmethod
    def _equations(y, t, w):
        return [y(1), -w * w * y(0)]


# ── GALI ─────────────────────────────────────────────────────────────────────


def test_gali_henon_chaotic_collapses():
    """Hénon GALI₂ collapses exponentially; the rate is near λ₁-λ₂ ≈ 2.04."""
    g = gali(ts.Henon(), k=2, ic=[0.1, 0.1], steps=70, seed=0)
    assert isinstance(g, GALIResult)
    assert g.is_chaotic()
    assert g.final < 1e-8
    # λ₁-λ₂ = 0.419 - (-1.623) ≈ 2.04 (Sprott 2003); the finite-time estimate is
    # a touch steeper from the initial alignment transient.
    assert 1.6 < g.decay_rate() < 2.7


def test_gali_lorenz_decay_rate_matches_lyapunov_gap():
    """Lorenz GALI₂ decay rate reproduces λ₁-λ₂ ≈ 0.906 (Skokos law)."""
    g = gali(ts.Lorenz(), k=2, ic=[1.0, 1.0, 1.0], final_time=22.0, dt=0.05, transient=15.0, seed=0)
    rate = g.decay_rate(floor=1e-10, t_min=5.0)
    assert rate == pytest.approx(LORENZ_GAP, abs=0.15)
    assert g.is_chaotic()


def test_gali_lorenz_k3_collapses_faster():
    """GALI₃ adds the (large) λ₁-λ₃ gap, so it dies almost immediately."""
    g = gali(ts.Lorenz(), k=3, ic=[1.0, 1.0, 1.0], final_time=6.0, dt=0.02, transient=15.0, seed=0)
    assert g.final < 1e-10


def test_gali_regular_orbit_stays_unity():
    """A harmonic oscillator is regular: GALI₂ neither aligns nor decays."""
    g = gali(_Harmonic(), k=2, ic=[1.0, 0.0], final_time=40.0, dt=0.1, transient=0.0, seed=0)
    assert g.values.min() > 0.9
    assert float(g) == pytest.approx(1.0, abs=0.05)
    assert not g.is_chaotic()


def test_gali_input_validation():
    with pytest.raises(ValueError, match="k must satisfy"):
        gali(ts.Henon(), k=1)
    with pytest.raises(ValueError, match="k must satisfy"):
        gali(ts.Henon(), k=3)  # dim is 2
    with pytest.raises(ValueError, match="dt has no meaning"):
        gali(ts.Henon(), k=2, dt=0.1)
    with pytest.raises(ValueError, match="steps applies to maps"):
        gali(ts.Lorenz(), k=2, steps=100)
    with pytest.raises(NotImplementedError):
        gali(ts.MackeyGlass(), k=2)  # DDE: no finite tangent space here


# ── 0–1 test ─────────────────────────────────────────────────────────────────


def test_zero_one_logistic_chaotic():
    """Logistic r=4 is fully chaotic → K ≈ 1."""
    x = ts.Logistic(params={"r": 4.0}).iterate(steps=4000, ic=[0.2]).component("x")
    assert zero_one_test(x, n_c=50, seed=1) > 0.9


def test_zero_one_logistic_regular():
    """Logistic r=3.5 settles on a period-4 cycle → K ≈ 0."""
    x = ts.Logistic(params={"r": 3.5}).iterate(steps=4000, ic=[0.2]).component("x")
    assert zero_one_test(x, n_c=50, seed=1) < 0.1


def test_zero_one_quasiperiodic_is_regular():
    """A quasi-periodic (two-tone) signal is non-chaotic → K ≈ 0."""
    j = np.arange(3000, dtype=float)
    x = np.sin(0.4 * j) + np.sin(np.sqrt(2.0) * 0.4 * j)
    assert zero_one_test(x, n_c=50, seed=2) < 0.1


def test_zero_one_distribution_and_errors():
    x = ts.Logistic(params={"r": 4.0}).iterate(steps=2500, ic=[0.3]).component("x")
    k, k_c = zero_one_test(x, n_c=20, seed=0, return_distribution=True)
    assert k_c.shape == (20,)
    assert k == pytest.approx(float(np.median(k_c)))
    with pytest.raises(ValueError, match="long series"):
        zero_one_test(np.zeros(50))
    with pytest.raises(TypeError, match="live system"):
        zero_one_test(ts.Lorenz())


# ── expansion entropy ────────────────────────────────────────────────────────


def test_expansion_entropy_tent_is_ln2():
    """Unit-height tent map: |f'| ≡ 2 ⇒ E(t)=2ᵗ ⇒ H = ln 2 (exact)."""
    h = expansion_entropy(ts.Tent(params={"mu": 1.0}), Box([0.0], [1.0]), n_samples=200, steps=18)
    assert isinstance(h, ExpansionEntropyResult)
    assert float(h) == pytest.approx(LN2, abs=0.02)
    assert h.n_survivors == h.n_samples  # nothing leaves [0, 1]


def test_expansion_entropy_henon_topological():
    """Hénon expansion entropy reproduces its topological entropy ≈ 0.465."""
    box = Box([-1.6, -0.5], [1.6, 0.5])
    h = expansion_entropy(ts.Henon(), box, n_samples=400, steps=12, seed=0)
    assert float(h) == pytest.approx(HENON_HTOP, abs=0.1)


def test_expansion_entropy_circle_quasiperiodic_is_zero():
    """A sub-critical circle map is quasi-periodic (λ=0) → H ≈ 0."""
    h = expansion_entropy(
        ts.Circle(params={"omega": 0.3333, "k": 0.5}), Box([0.0], [1.0]), n_samples=200, steps=18
    )
    assert abs(float(h)) < 0.05


def test_expansion_entropy_lorenz_flow_positive():
    """The Lorenz flow expands on its attractor → H clearly positive."""
    box = Box([-20.0, -25.0, 0.0], [20.0, 25.0, 50.0])
    h = expansion_entropy(ts.Lorenz(), box, n_samples=80, final_time=2.5, dt=0.25, seed=0)
    assert float(h) > 0.5


def test_expansion_entropy_input_validation():
    with pytest.raises(ValueError, match="dt has no meaning"):
        expansion_entropy(ts.Henon(), Box([-2, -2], [2, 2]), dt=0.1)
    with pytest.raises(ValueError, match="region dimension"):
        expansion_entropy(ts.Henon(), Box([0.0], [1.0]))  # 1-D box, 2-D map
    with pytest.raises(NotImplementedError):
        expansion_entropy(ts.MackeyGlass(), Box([0.0], [1.0]))


# ── result objects & registry ────────────────────────────────────────────────


def test_result_repr_and_float():
    g = gali(ts.Henon(), k=2, ic=[0.1, 0.1], steps=30, seed=0)
    assert "GALIResult" in repr(g)
    assert float(g) == g.final
    h = expansion_entropy(ts.Tent(params={"mu": 1.0}), Box([0.0], [1.0]), n_samples=50, steps=10)
    assert "ExpansionEntropyResult" in repr(h)
    assert float(h) == h.entropy


def test_indicators_self_register():
    for name in ("gali", "zero_one_test", "expansion_entropy"):
        assert name in registry.analyses


# ── robustness: degenerate / diverging frames must not crash (regression) ─────


def test_gali_random_ic_henon_never_crashes():
    """Random-IC GALI on the Hénon map must never raise (regression).

    ``Henon`` declares no ``default_ic``, so every call rolls a random IC; many
    land outside the attractor's basin and escape to infinity, which used to make
    the deviation frame non-finite and crash ``np.linalg.svd`` with
    ``LinAlgError`` (run-to-run non-deterministically, and at every long horizon).
    ``gali`` must retry onto the attractor and return a chaotic result every time.
    """
    for _ in range(20):
        g = gali(ts.Henon(), k=2, steps=1500)
        assert isinstance(g, GALIResult)
        assert np.all(np.isfinite(g.values))
        assert g.is_chaotic()  # Hénon is chaotic → GALI₂ collapses to ~0


def test_gali_offbasin_ic_retries_onto_attractor():
    """An explicit IC that escapes the basin is recovered by the random-IC retry."""
    g = gali(ts.Henon(), k=2, ic=[10.0, 10.0], steps=80, seed=0)
    assert isinstance(g, GALIResult)
    assert np.all(np.isfinite(g.values))
    assert g.is_chaotic()


def test_gali_volume_degenerate_returns_zero():
    """A collapsed or non-finite deviation frame spans zero volume, never raises."""
    from tsdynamics.analysis.chaos._common import gali_volume

    # two perfectly aligned unit columns → zero parallelepiped volume
    assert gali_volume(np.array([[1.0, 1.0], [0.0, 0.0]])) == pytest.approx(0.0)
    # a non-finite frame (diverged orbit) is treated as collapsed, not a crash
    assert gali_volume(np.array([[np.inf, 0.0], [np.nan, 1.0]])) == 0.0


def test_expansion_volume_overflow_returns_inf():
    """A non-finite fundamental matrix (overflowed growth) reports +inf, never raises."""
    from tsdynamics.analysis.chaos._common import expansion_volume

    assert expansion_volume(np.array([[np.inf, 0.0], [0.0, 1.0]])) == np.inf
    assert expansion_volume(np.array([[np.nan, 0.0], [0.0, 1.0]])) == np.inf
    assert expansion_volume(np.eye(2)) == pytest.approx(1.0)  # non-expanding


def test_expansion_entropy_long_horizon_no_crash():
    """A long un-renormalised horizon overflows the tangent product but must not crash."""
    box = Box([-1.6, -0.5], [1.6, 0.5])
    h = expansion_entropy(ts.Henon(), box, n_samples=60, steps=300, seed=0)
    assert isinstance(h, ExpansionEntropyResult)
    assert np.isfinite(float(h))
