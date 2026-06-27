"""SDE coverage gate — multi-dimensional & analytic-moment correctness (sde-coverage).

The existing SDE tests (``test_sde.py`` / ``test_sde_seed_parity.py``) exercise
only **scalar** (``dim == 1``) systems and the bit-for-bit landing/seed parity.
The registry SDE sweep is empty (no built-in SDE catalogue systems), so nothing
exercises:

- a **multi-dimensional** (``dim >= 2``) diagonal-Itô SDE whose diffusion is
  **state-dependent** in some components and constant in others — the per-component
  diagonal Wiener substrate and the per-component Milstein correction across a
  vector state;
- the **analytic moments** of the canonical Ornstein–Uhlenbeck / geometric
  Brownian motion laws — the check that actually validates the SDE *math* (drift
  and diffusion lowering + the integrator), not just self-consistency of one code
  path against itself;
- a genuine **strong-order** separation between Euler–Maruyama (order 0.5) and
  Milstein (order 1.0) measured against the *exact* GBM solution over **shared
  Brownian paths** — the test that proves the Milstein ``½ g g' (dW² − h)``
  correction is real and correct, not merely "present".

This module closes that hole. Every assertion is against an **independent oracle**
(an analytic moment, the exact GBM solution, or a different backend), never a value
compared to itself through the same code path.

The engine-backed parity tests ``importorskip`` the compiled extension (auto-tagged
``engine`` via ``tests/_engine_marker.py``), so they skip cleanly on a wheel-free
machine; the pure-Python statistical and strong-order checks always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics import StochasticSystem
from tsdynamics.families.stochastic import _sde_step, _seed_for

try:
    import tsdynamics._rust  # noqa: F401

    _HAS_RUST = True
except ImportError:  # pragma: no cover - wheel-free machines only
    _HAS_RUST = False

requires_engine = pytest.mark.skipif(
    not _HAS_RUST, reason="compiled tsdynamics._rust extension is not built"
)


# ---------------------------------------------------------------------------
# Test-local SDE systems (fixtures, NOT catalogue systems — adding a built-in
# SDE would trigger the SDE_SAMPLES registry guard and release obligations).
# Unique names so they do not collide in the registry with the other SDE tests.
# ---------------------------------------------------------------------------


class _CovGBM(StochasticSystem):
    """1-D geometric Brownian motion ``dX = μX dt + σX dW``.

    Multiplicative noise (``g' = σ ≠ 0``), with the exact closed-form solution
    ``X_T = X_0 exp((μ − ½σ²)T + σ W_T)`` — the strong-order oracle below.
    """

    params = {"mu": 0.5, "sigma": 0.6}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, mu, sigma):
        return [mu * y(0)]

    @staticmethod
    def _diffusion(y, t, mu, sigma):
        return [sigma * y(0)]


class _CovMixed2D(StochasticSystem):
    """A 2-D diagonal-Itô SDE mixing additive and state-dependent diffusion.

    - component 0 is an Ornstein–Uhlenbeck process ``dX₀ = θ(μ − X₀) dt + σ₀ dW₀``
      with **constant** diffusion (``g₀' = 0``);
    - component 1 is geometric Brownian motion ``dX₁ = a X₁ dt + σ₁ X₁ dW₁`` with
      **state-dependent** diffusion (``g₁' = σ₁ ≠ 0``).

    The two driving Wiener increments are independent, so the components stay
    statistically decoupled and each has a known analytic moment — the OU
    transient mean ``μ + (x₀ − μ) e^{−θT}`` and the GBM mean ``x₀ e^{aT}``.
    """

    params = {"theta": 1.0, "mu": 1.5, "sigma0": 0.4, "a": 0.1, "sigma1": 0.3}
    dim = 2
    variables = ("x", "y")

    @staticmethod
    def _drift(y, t, theta, mu, sigma0, a, sigma1):
        return [theta * (mu - y(0)), a * y(1)]

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma0, a, sigma1):
        return [sigma0, sigma1 * y(1)]


# ===========================================================================
# 1. Reference vs engine PARITY on a multi-dimensional system
# ===========================================================================


@requires_engine
@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_2d_reference_matches_engine_to_documented_tolerance(method):
    """``backend="reference"`` ≈ ``backend="interp"`` to the documented float band.

    The contract: both walk the *same* SplitMix64 integer stream in the same draw
    order over the same clean ``dt`` steps, so the only residual difference is the
    Box–Muller normal (Python libm vs Rust ``sin_cos``, ≤1 ULP/draw). A desync of
    the noise stream — or a wrong per-component diffusion in the 2-D vector path —
    would surface as an O(1) divergence, not a ULP one. This is a real cross-check:
    two independent integrator implementations of the same vector SDE.
    """
    sys = _CovMixed2D()
    kw = dict(final_time=1.0, dt=0.01, ic=[0.0, 1.0], seed=20240615, method=method)
    ref = sys.integrate(backend="reference", **kw)
    eng = sys.integrate(backend="interp", **kw)
    assert ref.y.shape == eng.y.shape == (ref.t.size, 2)
    assert eng.meta["engine"] == "rust"
    np.testing.assert_allclose(eng.y, ref.y, rtol=1e-9, atol=1e-11)


@requires_engine
@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_2d_engine_interp_equals_jit_bit_for_bit(method):
    """The interpreter and the Cranelift JIT land identically (bit-for-bit) in 2-D.

    Same seed, same tolerant landing, bit-identical evaluators ⇒ the dense vector
    trajectory agrees exactly across both compiled backends.
    """
    sys = _CovMixed2D()
    kw = dict(final_time=1.0, dt=0.01, ic=[0.0, 1.0], seed=7, method=method)
    interp = sys.integrate(backend="interp", **kw)
    jit = sys.integrate(backend="jit", **kw)
    np.testing.assert_array_equal(interp.y, jit.y)


# ===========================================================================
# 2. Ensemble determinism — seed_for(seed, i), batch-order independence, NaN row
# ===========================================================================


def test_ensemble_row_is_independent_of_batch_size_and_order():
    """Row ``i`` depends only on ``(base_seed, i)``, not the batch it rides in.

    The parallel-equals-serial contract: trajectory ``i``'s noise stream is seeded
    by ``seed_for(base_seed, i)`` alone. So the same IC at index ``i`` must give the
    same final state whether it sits in a 3-row batch or a 9-row batch, and a
    re-run reproduces the batch exactly. A bug that seeded by RNG-draw-consumption
    order (rather than by index) would make the row depend on its neighbours.
    """
    sys = _CovMixed2D()
    base = np.array([[0.0, 1.0], [0.2, 1.1], [-0.3, 0.9]])
    small = sys.ensemble(base, final_time=0.8, dt=0.02, seed=314, backend="reference")
    # Pad the batch with extra ICs *before* and ensure the original rows still
    # land identically once re-indexed at the same positions.
    padded_ics = np.vstack([base, base + 0.05, base - 0.05])
    padded = sys.ensemble(padded_ics, final_time=0.8, dt=0.02, seed=314, backend="reference")
    np.testing.assert_array_equal(small, padded[: base.shape[0]])

    # Same base seed ⇒ byte-identical batch on a re-run.
    again = sys.ensemble(base, final_time=0.8, dt=0.02, seed=314, backend="reference")
    np.testing.assert_array_equal(small, again)

    # Distinct indices draw distinct streams ⇒ distinct finals (no accidental aliasing).
    assert not np.array_equal(small[0], small[1])


def test_ensemble_row_equals_lone_trajectory_seeded_by_index():
    """Each ensemble row == a single ``integrate`` seeded by ``seed_for(seed, i)``.

    This pins the index-seeding *formula* against an independent reconstruction:
    a lone trajectory whose seed is computed by hand from ``_seed_for`` must trace
    the same path as the corresponding batch row. If the family changed how it
    derives per-index seeds, this diverges.
    """
    sys = _CovMixed2D()
    ics = np.array([[0.0, 1.0], [0.5, 0.8], [-0.2, 1.3], [0.1, 1.05]])
    base_seed, tf, dt = 9091, 0.6, 0.02
    batch = sys.ensemble(ics, final_time=tf, dt=dt, seed=base_seed, backend="reference")
    for i, ic in enumerate(ics):
        lone = sys.integrate(
            final_time=tf, dt=dt, ic=ic, seed=_seed_for(base_seed, i), backend="reference"
        )
        np.testing.assert_array_equal(batch[i], lone.y[-1])


class _CovExploder(StochasticSystem):
    """``dX = X² dt`` (no noise) — deterministic finite-time blow-up for one IC."""

    params: dict = {}
    dim = 1

    @staticmethod
    def _drift(y, t):
        return [y(0) * y(0)]

    @staticmethod
    def _diffusion(y, t):
        return [0.0]


def test_diverged_trajectory_becomes_a_nan_row_others_survive():
    """A diverging trajectory yields a NaN row without aborting the batch.

    ``x₀ = 1`` reaches ``+∞`` before ``t = 3`` (``1/(1−t)``); ``x₀ = −1`` decays to
    ``0`` and stays finite. The batch must isolate the blow-up as ``NaN`` and keep
    the finite row finite — the engine must not abort the whole ensemble.
    """
    sys = _CovExploder()
    finals = sys.ensemble(
        np.array([[1.0], [-1.0]]), final_time=3.0, dt=0.01, seed=0, backend="reference"
    )
    assert np.isnan(finals[0, 0])
    assert np.isfinite(finals[1, 0])


# ===========================================================================
# 3. STATISTICAL correctness — sample moments match the analytic law
#    (this is what validates the SDE *math*, against an independent oracle)
# ===========================================================================


@pytest.mark.slow
def test_ou_component_sample_mean_and_variance_match_analytic_law():
    """OU component: sample mean → transient mean, variance → ``σ₀²(1−e^{−2θT})/(2θ)``.

    The Ornstein–Uhlenbeck transition law from ``X₀ = x₀`` is exactly
    Gaussian with mean ``μ + (x₀ − μ) e^{−θT}`` and variance
    ``σ₀² (1 − e^{−2θT}) / (2θ)``. Comparing a large-ensemble sample mean/variance
    to these closed forms validates the additive-diffusion path end-to-end —
    nothing here is a tautology, the targets come from the SDE's analytic solution.
    """
    sys = _CovMixed2D()
    theta, mu, sigma0 = 1.0, 1.5, 0.4
    x0, tf = 0.0, 2.0
    n = 12000

    ics = np.tile([x0, 1.0], (n, 1)).astype(float)
    finals = sys.ensemble(ics, final_time=tf, dt=0.01, seed=2718, backend="reference")
    ou = finals[:, 0]

    want_mean = mu + (x0 - mu) * np.exp(-theta * tf)
    want_var = sigma0**2 * (1.0 - np.exp(-2.0 * theta * tf)) / (2.0 * theta)
    # MC std error of the mean ≈ sqrt(want_var/n) ≈ 0.0026; variance SE ≈ 0.0021.
    # Generous-but-meaningful bands (≈ 15-20 SE) keep this non-flaky yet falsifying.
    assert abs(ou.mean() - want_mean) < 0.04, (ou.mean(), want_mean)
    assert abs(ou.var() - want_var) < 0.04, (ou.var(), want_var)


@pytest.mark.slow
def test_gbm_component_sample_mean_matches_analytic_growth():
    """GBM component: sample mean → ``x₀ e^{aT}`` (the multiplicative-noise drift).

    ``E[X_T] = x₀ e^{aT}`` for ``dX = aX dt + σX dW`` regardless of ``σ`` (the
    Itô drift is unbiased). Matching the large-ensemble mean to this analytic
    growth validates the state-dependent diffusion path's *mean* behaviour.
    """
    sys = _CovMixed2D()
    a, tf = 0.1, 2.0
    y0, n = 1.0, 12000
    ics = np.tile([0.0, y0], (n, 1)).astype(float)
    finals = sys.ensemble(ics, final_time=tf, dt=0.01, seed=1618, backend="reference")
    gbm = finals[:, 1]
    want = y0 * np.exp(a * tf)
    # Lognormal mean → larger MC error; a 5% band is safe at this ensemble size.
    assert abs(gbm.mean() - want) < 0.05 * want, (gbm.mean(), want)


@requires_engine
@pytest.mark.slow
def test_engine_ensemble_reproduces_analytic_ou_moments():
    """The compiled engine's ensemble matches the analytic OU moments too.

    A second, independent path to the same statistical oracle: the Rust engine's
    parallel ensemble (rayon pool) must recover the same OU law, confirming the
    statistical correctness is a property of the *math*, not of the pure-Python
    reference integrator alone.
    """
    sys = _CovMixed2D()
    theta, mu, sigma0 = 1.0, 1.5, 0.4
    x0, tf, n = 0.0, 2.0, 12000
    ics = np.tile([x0, 1.0], (n, 1)).astype(float)
    finals = sys.ensemble(ics, final_time=tf, dt=0.01, seed=2718, backend="interp")
    ou = finals[:, 0]
    want_mean = mu + (x0 - mu) * np.exp(-theta * tf)
    want_var = sigma0**2 * (1.0 - np.exp(-2.0 * theta * tf)) / (2.0 * theta)
    assert abs(ou.mean() - want_mean) < 0.04
    assert abs(ou.var() - want_var) < 0.04


# ===========================================================================
# 4. Strong-order: Milstein (1.0) converges faster than Euler–Maruyama (0.5)
#    measured against the EXACT GBM solution over SHARED Brownian paths.
# ===========================================================================


def _strong_error(method: str, prob, dt: float, n_traj: int, seed: int) -> float:
    """Mean terminal strong error ``E|X_num(T) − X_exact(T)|`` for GBM.

    Drives the kernel ``_sde_step`` directly with an explicitly drawn Brownian
    path ``{dW_k}`` and compares the numerical terminal value to the closed-form
    GBM solution ``X_0 exp((μ − ½σ²)T + σ W_T)`` over the *same* accumulated
    ``W_T``. Sharing the Brownian path between the scheme and the exact solution
    is what makes this a genuine **strong** (pathwise) error — the only honest way
    to separate strong orders 0.5 (Euler–Maruyama) and 1.0 (Milstein).
    """
    mu, sigma, x0, tf = 0.5, 0.6, 1.0, 1.0
    p = prob.params_vec()
    n_steps = int(round(tf / dt))
    rng = np.random.default_rng(seed)
    errs = np.empty(n_traj)
    for j in range(n_traj):
        u = np.array([x0])
        t = 0.0
        w_total = 0.0
        for _ in range(n_steps):
            dw = np.sqrt(dt) * rng.standard_normal(1)
            w_total += dw[0]
            u = _sde_step(method, prob.drift, prob.diffusion, u, p, t, dw, dt)
            t += dt
        exact = x0 * np.exp((mu - 0.5 * sigma**2) * tf + sigma * w_total)
        errs[j] = abs(u[0] - exact)
    return float(errs.mean())


@pytest.mark.slow
def test_milstein_strong_order_beats_euler_maruyama_on_multiplicative_noise():
    """Fitted strong order: EM ≈ 0.5, Milstein ≈ 1.0 — and Milstein is more accurate.

    Over shared Brownian paths and shrinking ``dt``, the strong error of
    Euler–Maruyama decays like ``dt^0.5`` and Milstein like ``dt^1.0`` for
    multiplicative GBM noise. We fit the log–log slope and assert:

    * Milstein's fitted order clearly exceeds Euler–Maruyama's (the convergence
      *rate* is genuinely higher — the headline);
    * each order sits near its textbook value (EM ≳ 0.4, Milstein ≳ 0.85);
    * Milstein is at least as accurate as EM at the finest ``dt`` (never worse).

    If the Milstein ``½ g g' (dW² − h)`` correction were dropped or wrong, Milstein
    would collapse to EM's order-0.5 behaviour and the slope gap would vanish — so
    this fails loudly on that regression.
    """
    gbm = _CovGBM()
    em_prob = gbm._problem(ic=np.array([1.0]), t0=0.0, method="euler_maruyama")
    mil_prob = gbm._problem(ic=np.array([1.0]), t0=0.0, method="milstein")

    dts = np.array([0.08, 0.04, 0.02, 0.01])
    n_traj = 600
    em_err = np.array([_strong_error("euler_maruyama", em_prob, dt, n_traj, 0) for dt in dts])
    mil_err = np.array([_strong_error("milstein", mil_prob, dt, n_traj, 0) for dt in dts])

    em_order = float(np.polyfit(np.log(dts), np.log(em_err), 1)[0])
    mil_order = float(np.polyfit(np.log(dts), np.log(mil_err), 1)[0])

    # Headline: Milstein's convergence rate is strictly, clearly higher.
    assert mil_order > em_order + 0.25, (em_order, mil_order)
    # Each near its textbook strong order.
    assert em_order == pytest.approx(0.5, abs=0.2), em_order
    assert mil_order == pytest.approx(1.0, abs=0.2), mil_order
    # And Milstein is not worse than EM at the finest step.
    assert mil_err[-1] <= em_err[-1]


def test_milstein_correction_term_is_actually_applied_for_multiplicative_noise():
    """A single Milstein step differs from Euler–Maruyama by exactly ``½ g g' (dW²−h)``.

    Anchors the strong-order result to the *formula*: for GBM (``g = σx``,
    ``g' = σ``) one Milstein step must exceed the EM step by precisely the
    analytic Itô correction. This is asserted against a hand-computed value (an
    independent oracle), not against the kernel's own output, so it fails if the
    correction were dropped, mis-signed, or used the wrong ``g'``.
    """
    gbm = _CovGBM()
    em_prob = gbm._problem(ic=np.array([2.0]), t0=0.0, method="euler_maruyama")
    mil_prob = gbm._problem(ic=np.array([2.0]), t0=0.0, method="milstein")
    p = em_prob.params_vec()
    u, dw, h = np.array([2.0]), np.array([0.35]), 0.05

    em = _sde_step("euler_maruyama", em_prob.drift, em_prob.diffusion, u, p, 0.0, dw, h)
    mil = _sde_step("milstein", mil_prob.drift, mil_prob.diffusion, u, p, 0.0, dw, h)

    sigma = 0.6
    g, gprime = sigma * 2.0, sigma  # g = σx at x=2, g' = σ
    want_correction = 0.5 * g * gprime * (dw[0] ** 2 - h)
    assert want_correction != 0.0  # the test is meaningful only if the term is nonzero
    np.testing.assert_allclose(mil - em, [want_correction], atol=1e-14)
