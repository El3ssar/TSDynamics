"""Tests for the diagonal-Itô SDE family (stream E-SDE).

The Rust engine (``tsdyn-solvers``/``tsdyn-engine``) carries the rigorous,
large-ensemble moment and strong-order checks (run by ``cargo test``); these
Python tests cover the family surface: that ``_drift``/``_diffusion`` lower
correctly, the pure-Python reference integrator reproduces the canonical SDE
moments, seeding is reproducible and per-index, the seeded RNG faithfully ports
the engine's substrate, and the protocol behaves.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics import StochasticSystem
from tsdynamics.families.stochastic import _sde_step, _seed_for, _SplitMix64

# ---------------------------------------------------------------------------
# Example systems (test fixtures, not catalogue systems)
# ---------------------------------------------------------------------------


class GeometricBrownianMotion(StochasticSystem):
    """dX = μX dt + σX dW — analytic moments, multiplicative noise (g' ≠ 0)."""

    params = {"mu": 0.15, "sigma": 0.3}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, mu, sigma):
        return [mu * y(0)]

    @staticmethod
    def _diffusion(y, t, mu, sigma):
        return [sigma * y(0)]


class OrnsteinUhlenbeck(StochasticSystem):
    """dX = θ(μ − X) dt + σ dW — additive noise (g constant ⇒ g' = 0)."""

    params = {"theta": 1.0, "mu": 2.0, "sigma": 0.5}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma):
        return [sigma]


# ---------------------------------------------------------------------------
# Lowering correctness (tolerance-tight — no chaotic-divergence caveat)
# ---------------------------------------------------------------------------


def test_drift_and_diffusion_lower_to_the_symbolic_values():
    from tsdynamics.engine.compile import eval_tape, eval_tape_jac
    from tsdynamics.engine.problem import sde_problem

    gbm = GeometricBrownianMotion()
    prob = sde_problem(gbm, ic=[1.0], with_diffusion_jacobian=True)
    p = prob.params_vec()
    x0 = 1.7
    # f(x) = μx, g(x) = σx, ∂g/∂x = σ.
    assert np.allclose(eval_tape(prob.drift, [x0], p), [0.15 * x0])
    g, gjac = eval_tape_jac(prob.diffusion, [x0], p)
    assert np.allclose(g, [0.3 * x0])
    assert np.allclose(gjac, [[0.3]])


def test_milstein_problem_carries_the_diffusion_jacobian_euler_does_not():
    gbm = GeometricBrownianMotion()
    # The family asks for the diffusion Jacobian only for Milstein.
    em_prob = gbm._problem(ic=np.array([1.0]), t0=0.0, method="euler_maruyama")
    mil_prob = gbm._problem(ic=np.array([1.0]), t0=0.0, method="milstein")
    assert not em_prob.diffusion.has_jacobian
    assert mil_prob.diffusion.has_jacobian


# ---------------------------------------------------------------------------
# Kernel-formula correctness (exact, hand-chosen dW — the Python twin of the
# Rust SdeKernel unit tests)
# ---------------------------------------------------------------------------


def test_euler_maruyama_step_applies_the_exact_formula():
    gbm = GeometricBrownianMotion()
    prob = gbm._problem(ic=np.array([2.0]), t0=0.0, method="euler_maruyama")
    p = prob.params_vec()
    u, dw, h = np.array([2.0]), np.array([-0.2]), 0.05
    got = _sde_step("euler_maruyama", prob.drift, prob.diffusion, u, p, 0.0, dw, h)
    want = 2.0 + 0.15 * 2.0 * h + 0.3 * 2.0 * dw[0]  # f=μx, g=σx
    assert np.allclose(got, [want], atol=1e-14)


def test_milstein_step_applies_the_exact_correction():
    gbm = GeometricBrownianMotion()
    prob = gbm._problem(ic=np.array([2.0]), t0=0.0, method="milstein")
    p = prob.params_vec()
    u, dw, h = np.array([2.0]), np.array([0.25]), 0.05
    got = _sde_step("milstein", prob.drift, prob.diffusion, u, p, 0.0, dw, h)
    g, gp = 0.3 * 2.0, 0.3  # g=σx, g'=σ
    want = 2.0 + 0.15 * 2.0 * h + g * dw[0] + 0.5 * g * gp * (dw[0] ** 2 - h)
    assert np.allclose(got, [want], atol=1e-14)


def test_milstein_correction_vanishes_for_additive_noise():
    # OU has constant diffusion ⇒ g' = 0 ⇒ a Milstein step equals an EM step.
    ou = OrnsteinUhlenbeck()
    em_p = ou._problem(ic=np.array([0.3]), t0=0.0, method="euler_maruyama")
    mil_p = ou._problem(ic=np.array([0.3]), t0=0.0, method="milstein")
    p = em_p.params_vec()
    u, dw, h = np.array([0.3]), np.array([0.31]), 0.07
    em = _sde_step("euler_maruyama", em_p.drift, em_p.diffusion, u, p, 0.0, dw, h)
    mil = _sde_step("milstein", mil_p.drift, mil_p.diffusion, u, p, 0.0, dw, h)
    assert np.array_equal(em, mil)


# ---------------------------------------------------------------------------
# Seeded RNG / Wiener substrate (the port of the engine's rng.rs)
# ---------------------------------------------------------------------------


def test_splitmix64_matches_the_rust_golden_stream_for_seed_zero():
    # The same reference vectors the Rust ``splitmix64_seed_zero_golden_stream``
    # test pins; if the Python port drifts from the engine substrate this fails.
    rng = _SplitMix64(0)
    assert rng.next_u64() == 16294208416658607535
    assert rng.next_u64() == 7960286522194355700
    assert rng.next_u64() == 487617019471545679


def test_seed_for_is_pure_and_separates_adjacent_indices():
    assert _seed_for(42, 7) == _seed_for(42, 7)
    seeds = [_seed_for(0xDEADBEEF, i) for i in range(1000)]
    assert len(set(seeds)) == len(seeds)  # no collisions on 0..1000


def test_next_normal_has_standard_moments():
    rng = _SplitMix64(2024)
    xs = np.array([rng.next_normal() for _ in range(200_000)])
    assert abs(xs.mean()) < 0.02
    assert abs(xs.var() - 1.0) < 0.02


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_the_trajectory():
    gbm = GeometricBrownianMotion()
    a = gbm.integrate(final_time=1.0, dt=0.01, ic=[1.0], seed=12345)
    b = gbm.integrate(final_time=1.0, dt=0.01, ic=[1.0], seed=12345)
    assert np.array_equal(a.y, b.y)
    # The resolved seed is recorded for reproducibility.
    assert a.meta["seed"] == 12345


def test_ensemble_is_reproducible_and_decorrelates_indices():
    gbm = GeometricBrownianMotion()
    ics = np.ones((64, 1))
    a = gbm.ensemble(ics, final_time=1.0, dt=0.02, seed=7)
    b = gbm.ensemble(ics, final_time=1.0, dt=0.02, seed=7)
    assert np.array_equal(a, b)  # same base seed ⇒ identical batch
    # Distinct indices draw distinct noise streams ⇒ distinct finals.
    assert a[0, 0] != a[1, 0]


def test_additive_noise_makes_milstein_equal_euler_maruyama():
    # For OU the diffusion is constant (g' = 0), so the Milstein correction
    # vanishes and — given the same seed/step sequence — the two schemes trace
    # bit-for-bit the same path.
    ou = OrnsteinUhlenbeck()
    em = ou.integrate(final_time=2.0, dt=0.01, ic=[0.5], seed=99, method="euler_maruyama")
    mil = ou.integrate(final_time=2.0, dt=0.01, ic=[0.5], seed=99, method="milstein")
    assert np.array_equal(em.y, mil.y)


def test_seed_omitted_gives_a_fresh_realisation_each_call():
    gbm = GeometricBrownianMotion()
    a = gbm.integrate(final_time=1.0, dt=0.02, ic=[1.0])
    b = gbm.integrate(final_time=1.0, dt=0.02, ic=[1.0])
    assert not np.array_equal(a.y, b.y)


# ---------------------------------------------------------------------------
# Divergence handling (loud, never silent)
# ---------------------------------------------------------------------------


class ExplodingDrift(StochasticSystem):
    """dX = X² dt (no noise) — deterministic finite-time blow-up."""

    params: dict = {}
    dim = 1

    @staticmethod
    def _drift(y, t):
        return [y(0) * y(0)]

    @staticmethod
    def _diffusion(y, t):
        return [0.0]


def test_single_integration_raises_on_divergence():
    sys = ExplodingDrift()
    with pytest.raises(RuntimeError, match="diverged"):
        sys.integrate(final_time=3.0, dt=0.01, ic=[1.0], seed=0)


def test_ensemble_isolates_a_diverged_trajectory_as_nan():
    sys = ExplodingDrift()
    # x0 = 1 blows up before t = 3; x0 = -1 decays and stays finite.
    finals = sys.ensemble(np.array([[1.0], [-1.0]]), final_time=3.0, dt=0.01, seed=0)
    assert np.isnan(finals[0, 0])
    assert np.isfinite(finals[1, 0])


# ---------------------------------------------------------------------------
# Method resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "canon"),
    [("euler_maruyama", "euler_maruyama"), ("EM", "euler_maruyama"), ("Milstein", "milstein")],
)
def test_method_aliases_resolve(alias, canon):
    gbm = GeometricBrownianMotion()
    traj = gbm.integrate(final_time=0.1, dt=0.05, ic=[1.0], seed=0, method=alias)
    assert traj.meta["method"] == canon


def test_unknown_method_raises():
    gbm = GeometricBrownianMotion()
    with pytest.raises(ValueError, match="unknown SDE method"):
        gbm.integrate(final_time=0.1, dt=0.05, ic=[1.0], method="heun")


# ---------------------------------------------------------------------------
# System protocol
# ---------------------------------------------------------------------------


def test_protocol_step_state_setstate_time_reinit():
    gbm = GeometricBrownianMotion()
    gbm.reinit([1.0], seed=0)
    assert gbm.is_discrete is False
    assert gbm.time() == 0.0
    assert np.array_equal(gbm.state(), np.array([1.0]))

    u1 = gbm.step(0.01)
    assert u1.shape == (1,)
    assert gbm.time() == pytest.approx(0.01)
    assert np.array_equal(gbm.state(), u1)

    gbm.set_state([2.0])
    assert np.array_equal(gbm.state(), np.array([2.0]))
    assert gbm.time() == pytest.approx(0.01)  # set_state does not move time


def test_stepping_is_reproducible_given_a_seed():
    a = GeometricBrownianMotion()
    a.reinit([1.0], seed=2024)
    path_a = [a.step(0.01)[0] for _ in range(20)]
    b = GeometricBrownianMotion()
    b.reinit([1.0], seed=2024)
    path_b = [b.step(0.01)[0] for _ in range(20)]
    assert path_a == path_b


def test_trajectory_drops_transient():
    gbm = GeometricBrownianMotion()
    traj = gbm.trajectory(1.0, dt=0.05, transient=0.5, ic=[1.0], seed=0)
    assert traj.t[0] >= 0.5
    assert traj["x"].ndim == 1


def test_implicit_reinit_on_cold_step():
    gbm = GeometricBrownianMotion(ic=[1.0])
    # No explicit reinit — first step must initialise from self.ic.
    u = gbm.step(0.01)
    assert np.isfinite(u).all()


# ---------------------------------------------------------------------------
# Moment convergence (slow — larger ensembles in the pure-Python reference)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gbm_reproduces_analytic_mean_with_both_schemes():
    # E[X_T] = X0 e^{μT}; both Euler–Maruyama and Milstein are weakly consistent.
    # The Rust suite proves this at 40k trajectories; here a modest ensemble (the
    # pure-Python reference) with a loose, non-flaky bound is enough.
    gbm = GeometricBrownianMotion()
    ics = np.ones((2500, 1))
    want = np.exp(0.15)  # X0=1, μ=0.15, T=1
    for method in ("euler_maruyama", "milstein"):
        finals = gbm.ensemble(ics, final_time=1.0, dt=0.02, seed=1, method=method)
        assert np.isfinite(finals).all()
        # MC std error of the mean ≈ 0.009 here; 0.05 is a safe, non-flaky band.
        assert abs(finals.mean() - want) < 0.05, method


@pytest.mark.slow
def test_ou_converges_to_its_stationary_mean_and_variance():
    # Stationary law N(μ, σ²/(2θ)); start at μ and integrate ≫ 1/θ.
    ou = OrnsteinUhlenbeck()
    ics = np.full((2500, 1), 2.0)
    finals = ou.ensemble(ics, final_time=6.0, dt=0.02, seed=3)
    assert abs(finals.mean() - 2.0) < 0.05
    assert abs(finals.var() - 0.5**2 / (2 * 1.0)) < 0.04
