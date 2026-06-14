"""End-to-end tests for the engine wiring that stream E-WIRE adds.

These exercise the parts of the compiled engine that were unreachable from
Python before E-WIRE: the **SDE FFI** (the two-tape diagonal-Itô call and its
seeded ensemble), the **map-ensemble** binding, and the **Cranelift JIT** bridge
(``backend="jit"`` now runs instead of raising ``NotImplementedError``).

The whole module is skipped when the extension is not built (the default
``ci.yml`` Python job runs without it); the dedicated ``engine-bindings.yml`` job
builds ``tsdynamics._rust`` and runs these for real.

The correctness bar:

- **SDE interp == the pure-Python reference** under a fixed seed (the two share
  the SplitMix64 stream and Box–Muller transform, differing only at the libm
  ``sin``/``cos`` vs ``sin_cos`` ULP), for both Euler–Maruyama and Milstein.
- **SDE/ODE jit == interp bit-for-bit** (the JIT evaluator is bit-identical to
  the interpreter and the noise is seed-driven).
- **GBM/OU ensemble moments** reproduce their analytic values.
- **the map-ensemble binding == a serial single-map loop** bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

# The whole module is meaningless without the compiled engine.
_rust = pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts  # noqa: E402
from tsdynamics import StochasticSystem  # noqa: E402
from tsdynamics.engine import run  # noqa: E402
from tsdynamics.engine.problem import map_problem, sde_problem  # noqa: E402

# ---------------------------------------------------------------------------
# SDE test fixtures (not catalogue systems; unique names to avoid registry
# collisions with the reference-only fixtures in test_sde.py)
# ---------------------------------------------------------------------------


class WireGBM(StochasticSystem):
    """dX = μX dt + σX dW — multiplicative noise (g' ≠ 0), analytic moments."""

    params = {"mu": 0.12, "sigma": 0.3}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, mu, sigma):
        return [mu * y(0)]

    @staticmethod
    def _diffusion(y, t, mu, sigma):
        return [sigma * y(0)]


class WireOU(StochasticSystem):
    """dX = θ(μ − X) dt + σ dW — additive noise, stationary N(μ, σ²/2θ)."""

    params = {"theta": 1.0, "mu": 1.0, "sigma": 0.5}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma):
        return [sigma]


# ---------------------------------------------------------------------------
# SDE: the compiled engine reproduces the pure-Python reference (fixed seed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_sde_interp_matches_reference_fixed_seed(method):
    gbm = WireGBM()
    kw = dict(final_time=1.0, dt=0.01, ic=[1.0], method=method, seed=20240614)
    ref = gbm.integrate(backend="reference", **kw)
    eng = gbm.integrate(backend="interp", **kw)
    assert eng.y.shape == ref.y.shape
    np.testing.assert_allclose(eng.y[0], [1.0])  # first row is the IC
    # The two draw the same integer stream; only libm sin/cos vs sin_cos differs,
    # so a short window matches very tightly (not bit-exact across libm/std).
    np.testing.assert_allclose(eng.y, ref.y, rtol=1e-6, atol=1e-9)
    assert eng.meta["engine"] == "rust"
    assert eng.meta["seed"] == ref.meta["seed"]


@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_sde_ensemble_interp_matches_reference_per_row(method):
    gbm = WireGBM()
    rng = np.random.default_rng(1)
    ics = 1.0 + 0.1 * rng.standard_normal((12, 1))
    kw = dict(final_time=1.0, dt=0.01, method=method, seed=99)
    ref = gbm.ensemble(ics, backend="reference", **kw)
    eng = gbm.ensemble(ics, backend="interp", **kw)
    assert eng.shape == (12, 1)
    # Each trajectory i is seeded by seed_for(seed, i) on both paths.
    np.testing.assert_allclose(eng, ref, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# SDE: jit == interp bit-for-bit (seed-driven noise + bit-identical evaluators)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_sde_jit_matches_interp_bit_for_bit(method):
    gbm = WireGBM()
    kw = dict(final_time=1.0, dt=0.005, ic=[1.0], method=method, seed=7)
    interp = gbm.integrate(backend="interp", **kw)
    jit = gbm.integrate(backend="jit", **kw)
    np.testing.assert_array_equal(interp.y, jit.y)


def test_sde_ensemble_jit_matches_interp_bit_for_bit():
    gbm = WireGBM()
    ics = np.linspace(0.8, 1.2, 10).reshape(-1, 1)
    kw = dict(final_time=1.0, dt=0.01, method="milstein", seed=3)
    interp = gbm.ensemble(ics, backend="interp", **kw)
    jit = gbm.ensemble(ics, backend="jit", **kw)
    np.testing.assert_array_equal(interp, jit)


# ---------------------------------------------------------------------------
# SDE: ensemble moments reproduce the analytic values (GBM mean, OU stationary)
# ---------------------------------------------------------------------------


def test_gbm_ensemble_mean_matches_analytic():
    # E[X_T] = X0 e^{μT} for geometric Brownian motion.
    gbm = WireGBM()
    n, tf = 8000, 1.0
    ics = np.ones((n, 1))
    finals = gbm.ensemble(
        ics, final_time=tf, dt=0.005, method="milstein", seed=2024, backend="interp"
    )
    assert np.all(np.isfinite(finals))
    want = np.exp(gbm.params["mu"] * tf)
    assert abs(finals.mean() - want) < 0.02, f"GBM mean {finals.mean()} vs {want}"


def test_ou_ensemble_matches_stationary_law():
    # OU started at μ: stationary mean μ, variance σ²/(2θ), reached after ≫ 1/θ.
    ou = WireOU()
    theta, mu, sigma = (ou.params[k] for k in ("theta", "mu", "sigma"))
    n = 12000
    ics = np.full((n, 1), mu)
    finals = ou.ensemble(
        ics, final_time=10.0, dt=0.005, method="euler_maruyama", seed=11, backend="interp"
    )
    assert np.all(np.isfinite(finals))
    want_var = sigma * sigma / (2.0 * theta)
    assert abs(finals.mean() - mu) < 0.02, f"OU mean {finals.mean()} vs {mu}"
    assert abs(finals.var() - want_var) < 0.02, f"OU var {finals.var()} vs {want_var}"


# ---------------------------------------------------------------------------
# SDE: the engine path is reachable through run.sde_* too, and divergence raises
# ---------------------------------------------------------------------------


def test_run_sde_helpers_dispatch_through_the_engine():
    gbm = WireGBM()
    prob = sde_problem(gbm, ic=[1.0], with_diffusion_jacobian=True)
    t_eval = np.linspace(0.0, 1.0, 101)
    y = run.sde_integrate_dense(prob, t_eval, dt=0.01, method="milstein", seed=5, backend="interp")
    assert y.shape == (101, 1)
    assert y[0, 0] == 1.0
    ics = np.ones((5, 1))
    finals = run.sde_ensemble_final(
        prob, ics, t0=0.0, t1=1.0, dt=0.01, method="milstein", seed=5, backend="interp"
    )
    assert finals.shape == (5, 1)


def test_run_integrate_and_ensemble_reject_sde_problems():
    # The generic ODE seam cannot carry the SDE seed/step, and the drift tape is
    # ODE-shaped, so it must refuse an SDE rather than integrate the drift alone.
    gbm = WireGBM()
    with pytest.raises(NotImplementedError, match="SDE"):
        run.integrate(gbm, final_time=1.0, dt=0.01, backend="interp")
    with pytest.raises(NotImplementedError, match="SDE"):
        run.ensemble(gbm, np.ones((3, 1)), final_time=1.0, backend="interp")


# ---------------------------------------------------------------------------
# Map ensemble: the binding == a serial single-map loop, bit-for-bit
# ---------------------------------------------------------------------------


def test_map_ensemble_binding_matches_serial_iterate():
    henon = ts.Henon()
    prob = map_problem(henon)
    arrays = prob.tape.to_arrays()
    rng = np.random.default_rng(0)
    # On-attractor ICs so the orbits stay finite over the horizon.
    ics = np.array([0.1, 0.1]) + 0.05 * rng.standard_normal((8, 2))
    steps = 40
    batch = np.asarray(_rust.iterate_ensemble_final(*arrays, np.ascontiguousarray(ics), steps))
    assert batch.shape == (8, 2)
    for i, ic in enumerate(ics):
        dense = np.asarray(_rust.iterate_map(*arrays, np.ascontiguousarray(ic), steps))
        np.testing.assert_array_equal(batch[i], dense[-1])  # final iterate, bit-for-bit


def test_map_ensemble_through_run_matches_single_integrate():
    henon = ts.Henon()
    rng = np.random.default_rng(2)
    ics = np.array([0.1, 0.1]) + 0.05 * rng.standard_normal((6, 2))
    steps = 50
    batch = run.ensemble(henon, ics, final_time=steps, backend="interp")
    assert batch.shape == (6, 2)
    for i, ic in enumerate(ics):
        single = run.integrate(henon, final_time=steps, ic=ic, backend="interp")
        np.testing.assert_array_equal(batch[i], single.y[-1])


def test_map_ensemble_reference_matches_engine():
    henon = ts.Henon()
    rng = np.random.default_rng(4)
    ics = np.array([0.1, 0.1]) + 0.05 * rng.standard_normal((5, 2))
    eng = run.ensemble(henon, ics, final_time=30, backend="interp")
    ref = run.ensemble(henon, ics, final_time=30, backend="reference")
    np.testing.assert_array_equal(eng, ref)


# ---------------------------------------------------------------------------
# ODE/DDE jit == interp (the JIT bridge now runs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Lorenz", "Rossler"])
def test_ode_jit_matches_interp_bit_for_bit(name):
    system = getattr(ts, name)()
    ic = system.resolve_ic(None)
    kw = dict(final_time=3.0, dt=0.05, ic=ic, method="dop853", rtol=1e-10, atol=1e-12)
    interp = run.integrate(system, backend="interp", **kw)
    jit = run.integrate(system, backend="jit", **kw)
    assert jit.meta["engine"] == "rust"
    np.testing.assert_array_equal(interp.y, jit.y)


def test_dde_jit_matches_interp():
    mg = ts.MackeyGlass()
    kw = dict(final_time=40.0, dt=0.5, rtol=1e-7, atol=1e-9, method="rk45")
    interp = mg.integrate(backend="interp", **kw)
    jit = mg.integrate(backend="jit", **kw)
    np.testing.assert_array_equal(interp.y, jit.y)
