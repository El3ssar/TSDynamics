r"""
Tests for fixed points & periodic orbits (stream **A-FP**).

Each routine is pinned to an analytic or well-established literature value:

- **Map fixed points** — the Hénon saddles (analytic) and the logistic
  fixed points ``{0, 1−1/r}``, found by Newton and by the Schmelcher--Diakonos
  (1997) / Davidchack--Lai (1999) stabilising transformations.
- **Flow equilibria** — the Lorenz equilibria (origin + ``C±`` at
  ``(±√(β(ρ−1)), …, ρ−1)``) and the two Rössler equilibria
  ``x = (c ± √(c²−4ab))/2``, with their (un)stable classification.
- **Map periodic orbits** — the logistic period-2 orbit
  ``x = (r+1 ± √((r+1)(r−3)))/(2r)`` (stable at ``r=3.2``) and the
  period-3 window at ``r=3.83`` (a stable node + an unstable saddle, both born at
  the tangent bifurcation ``r = 1+√8``).
- **Flow periodic orbit (shooting)** — the autonomous Van der Pol limit cycle
  (``T ≈ 6.6633`` at ``μ=1``, ``T ≈ 6.3807`` at ``μ=0.5``), with one trivial
  Floquet multiplier ``≈ 1``; a harmonic-oscillator centre is correctly rejected.
- **estimate_period** — a sinusoid of known period, by autocorrelation and FFT.

All flow routines use the self-contained RK4 (variational) integrator over the
SymEngine-lambdified RHS/Jacobian, so none of these tests compile a backend.

References
----------
Schmelcher & Diakonos (1997), *Phys. Rev. Lett.* 78, 4733.
Davidchack & Lai (1999), *Phys. Rev. E* 60, 6172.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import (
    ContinuousSystem,
    FixedPoint,
    PeriodicOrbit,
    Trajectory,
    estimate_period,
    fixed_points,
    periodic_orbit,
    periodic_orbits,
    registry,
)


class _VanDerPol(ContinuousSystem):
    """Autonomous Van der Pol ``x' = v, v' = μ(1−x²)v − x`` (a stable limit cycle)."""

    params = {"mu": 1.0}
    dim = 2
    variables = ("x", "v")

    @staticmethod
    def _equations(y, t, mu):
        return [y(1), mu * (1 - y(0) * y(0)) * y(1) - y(0)]


class _Harmonic(ContinuousSystem):
    """Undamped harmonic oscillator — a centre (non-isolated periodic orbits)."""

    params = {"w": 1.0}
    dim = 2
    variables = ("x", "v")

    @staticmethod
    def _equations(y, t, w):
        return [y(1), -w * w * y(0)]


def _match(found: list, expected: np.ndarray, atol: float = 1e-4) -> np.ndarray:
    """Return the found point closest to ``expected`` (asserts one is within ``atol``)."""
    pts = np.array([np.asarray(f.x if hasattr(f, "x") else f) for f in found])
    d = np.linalg.norm(pts - expected, axis=1)
    assert d.min() < atol, (
        f"no match for {expected} (closest {pts[d.argmin()]}, dist {d.min():.2e})"
    )
    return pts[d.argmin()]


# ── map fixed points ──────────────────────────────────────────────────────────


class TestMapFixedPoints:
    def test_henon_analytic(self) -> None:
        fps = fixed_points(ts.Henon(), seed=0)
        a, b = 1.4, 0.3
        disc = np.sqrt((1 - b) ** 2 + 4 * a)
        expected = sorted([(-(1 - b) + disc) / (2 * a), (-(1 - b) - disc) / (2 * a)])
        np.testing.assert_allclose(sorted(fp.x[0] for fp in fps), expected, rtol=1e-8)
        assert all(not fp.stable for fp in fps)  # both saddles
        for fp in fps:
            assert fp.x[1] == pytest.approx(b * fp.x[0], rel=1e-8)
            assert not fp.continuous

    @pytest.mark.parametrize("method", ["newton", "sd", "dl"])
    def test_henon_all_methods_find_both_saddles(self, method: str) -> None:
        fps = fixed_points(ts.Henon(), method=method, seed=1)
        a, b = 1.4, 0.3
        disc = np.sqrt((1 - b) ** 2 + 4 * a)
        for xstar in ((-(1 - b) + disc) / (2 * a), (-(1 - b) - disc) / (2 * a)):
            _match(fps, np.array([xstar, b * xstar]), atol=1e-6)

    def test_logistic_fixed_points(self) -> None:
        m = ts.Logistic(params={"r": 2.5})
        fps = fixed_points(m, region=([-0.5], [1.5]), seed=0)
        xs = sorted(fp.x[0] for fp in fps)
        np.testing.assert_allclose(xs, [0.0, 1 - 1 / 2.5], atol=1e-9)
        stable = {round(fp.x[0], 6): fp.stable for fp in fps}
        assert stable[0.0] is False
        assert stable[round(1 - 1 / 2.5, 6)] is True

    def test_logistic_r4_unstable_fixed_points(self) -> None:
        # at r=4 both fixed points {0, 0.75} are unstable; DL must still find them
        fps = fixed_points(
            ts.Logistic(params={"r": 4.0}), region=([-0.2], [1.2]), method="dl", seed=2
        )
        xs = sorted(fp.x[0] for fp in fps)
        np.testing.assert_allclose(xs, [0.0, 0.75], atol=1e-8)
        assert all(not fp.stable for fp in fps)

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            fixed_points(ts.Henon(), method="bogus")


# ── flow equilibria ─────────────────────────────────────────────────────────


class TestFlowEquilibria:
    def test_lorenz_equilibria(self) -> None:
        fps = fixed_points(ts.Lorenz(), region=([-30, -30, -5], [30, 30, 55]), n_seeds=300, seed=1)
        c = np.sqrt(8 / 3 * (28 - 1))  # ±√72 = 8.48528…
        assert len(fps) == 3
        for fp in fps:
            assert fp.continuous and not fp.stable  # all three unstable at ρ=28
        _match(fps, np.array([0.0, 0.0, 0.0]))
        _match(fps, np.array([c, c, 27.0]))
        _match(fps, np.array([-c, -c, 27.0]))
        # the origin is a real saddle (one strongly unstable real eigenvalue)
        origin = next(fp for fp in fps if np.linalg.norm(fp.x) < 1e-5)
        assert origin.eigenvalues.real.max() == pytest.approx(11.8277, abs=1e-3)

    def test_rossler_equilibria(self) -> None:
        fps = fixed_points(ts.Rossler(), region=([-1, -30, -1], [8, 1, 30]), n_seeds=500, seed=2)
        a, c = 0.2, 5.7
        d = np.sqrt(c * c - 4 * a * a)
        for x in ((c + d) / 2, (c - d) / 2):
            _match(fps, np.array([x, -x / a, x / a]), atol=1e-4)
        assert all(not fp.stable for fp in fps)

    def test_flow_rejects_sd_dl(self) -> None:
        with pytest.raises(ValueError, match="method='newton'"):
            fixed_points(ts.Lorenz(), method="dl")

    def test_unsupported_family_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            fixed_points(ts.MackeyGlass())


# ── map periodic orbits ──────────────────────────────────────────────────────


class TestMapPeriodicOrbits:
    def test_logistic_period2(self) -> None:
        orbs = periodic_orbits(ts.Logistic(params={"r": 3.2}), 2, seed=3)
        assert len(orbs) == 1
        o = orbs[0]
        r = 3.2
        disc = np.sqrt((r + 1) * (r - 3))
        expected = sorted([(r + 1 + disc) / (2 * r), (r + 1 - disc) / (2 * r)])
        np.testing.assert_allclose(sorted(o.points.ravel()), expected, atol=1e-9)
        assert o.period == 2 and o.stable and not o.continuous
        # multiplier of the 2-cycle is 4 + 2r − r² = 0.16 at r=3.2
        assert float(np.prod(o.multipliers).real) == pytest.approx(0.16, abs=1e-6)
        assert o.residual < 1e-9

    def test_logistic_period3_window(self) -> None:
        orbs = periodic_orbits(ts.Logistic(params={"r": 3.83}), 3, seed=4)
        assert len(orbs) == 2  # the saddle-node pair born at r = 1+√8
        assert all(o.period == 3 for o in orbs)
        mults = sorted(float(np.abs(o.multipliers).max()) for o in orbs)
        assert mults[0] == pytest.approx(0.33, abs=0.05)  # stable node
        assert mults[1] == pytest.approx(1.65, abs=0.05)  # unstable saddle
        assert {o.stable for o in orbs} == {True, False}

    def test_period1_returns_fixed_points(self) -> None:
        orbs = periodic_orbits(ts.Logistic(params={"r": 2.5}), 1, region=([-0.5], [1.5]), seed=5)
        xs = sorted(float(o.points[0, 0]) for o in orbs)
        np.testing.assert_allclose(xs, [0.0, 0.6], atol=1e-8)
        assert all(o.period == 1 for o in orbs)

    def test_prime_filter_excludes_lower_period(self) -> None:
        # the period-2 orbit is a fixed point of f⁴; with prime=True it must NOT
        # appear in a period-4 search.
        r = 3.2
        orbs4 = periodic_orbits(ts.Logistic(params={"r": r}), 4, prime=True, seed=6)
        disc = np.sqrt((r + 1) * (r - 3))
        p2 = {round((r + 1 + disc) / (2 * r), 6), round((r + 1 - disc) / (2 * r), 6)}
        for o in orbs4:
            assert o.period == 4
            assert not (set(np.round(o.points.ravel(), 6)) & p2)


# ── flow periodic orbit (single shooting) ────────────────────────────────────


class TestFlowShooting:
    def test_vanderpol_mu1(self) -> None:
        orb = periodic_orbit(
            _VanDerPol(params={"mu": 1.0}), ic=[2.0, 0.0], period_guess=6.0, transient=20.0
        )
        assert orb.period == pytest.approx(6.6632869, abs=2e-3)
        assert orb.residual < 1e-8 and orb.continuous and orb.stable
        # one trivial Floquet multiplier ≈ 1 (the flow direction)
        assert np.abs(np.abs(orb.multipliers) - 1.0).min() < 1e-2

    def test_vanderpol_mu05(self) -> None:
        orb = periodic_orbit(
            _VanDerPol(params={"mu": 0.5}), ic=[2.0, 0.0], period_guess=6.3, transient=30.0
        )
        assert orb.period == pytest.approx(6.3806758, abs=2e-3)
        assert orb.stable

    def test_auto_period_guess(self) -> None:
        orb = periodic_orbit(_VanDerPol(params={"mu": 1.0}), ic=[0.5, 0.5], transient=20.0)
        assert orb.period == pytest.approx(6.6632869, abs=5e-2)

    def test_center_is_rejected(self) -> None:
        with pytest.raises(RuntimeError):
            periodic_orbit(_Harmonic(), ic=[1.0, 0.0], period_guess=6.0)

    def test_map_rejected(self) -> None:
        with pytest.raises(NotImplementedError):
            periodic_orbit(ts.Henon())


# ── period estimation ────────────────────────────────────────────────────────


class TestEstimatePeriod:
    def test_sine_autocorrelation(self) -> None:
        t = np.linspace(0, 40 * np.pi, 8000)
        y = np.sin(3.0 * t)
        assert estimate_period(y, dt=t[1] - t[0]) == pytest.approx(2 * np.pi / 3, rel=1e-2)

    def test_sine_fft(self) -> None:
        t = np.linspace(0, 40 * np.pi, 8000)
        y = np.sin(3.0 * t)
        assert estimate_period(y, dt=t[1] - t[0], method="fft") == pytest.approx(
            2 * np.pi / 3, rel=1e-2
        )

    def test_trajectory_component_autopick(self) -> None:
        t = np.linspace(0, 20 * np.pi, 6000)
        y = np.column_stack([0.01 * np.sin(5 * t), 2.0 * np.sin(t)])  # col 1 dominates
        traj = Trajectory(t, y, None)
        assert estimate_period(traj) == pytest.approx(2 * np.pi, rel=1e-2)

    def test_constant_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_period(np.ones(200), dt=0.1)

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 8"):
            estimate_period(np.array([1.0, 2.0, 3.0]))


# ── registration ─────────────────────────────────────────────────────────────


def test_self_registered_in_analyses() -> None:
    for name in ("fixed_points", "periodic_orbits", "periodic_orbit", "estimate_period"):
        assert registry.analyses.get(name) is not None


def test_result_types() -> None:
    assert isinstance(fixed_points(ts.Henon(), seed=0)[0], FixedPoint)
    assert isinstance(periodic_orbits(ts.Logistic(params={"r": 3.2}), 2, seed=0)[0], PeriodicOrbit)


# ── stabilising-matrix generation (A-FP _common) ──────────────────────────────


class TestSignedPermutationMatrices:
    """Lazy generation, truncation consistency, and the arithmetic count."""

    def test_count_helper_matches_full_set(self) -> None:
        from tsdynamics.analysis.fixedpoints import _common as _c

        for dim in (1, 2, 3, 4):
            full = _c.signed_permutation_matrices(dim)
            assert len(full) == _c._signed_permutation_count(dim)
            assert _c._signed_permutation_count(dim) == 2**dim * math.factorial(dim)

    def test_truncation_returns_exactly_max_count(self) -> None:
        from tsdynamics.analysis.fixedpoints import _common as _c

        # dim=4 has 384 matrices; a cap must return exactly that many, identity-first.
        mats = _c.signed_permutation_matrices(4, max_count=10)
        assert len(mats) == 10
        np.testing.assert_array_equal(mats[0], np.eye(4))

    def test_lazy_does_not_materialise_full_set(self) -> None:
        # A large dim would be intractable if all 2^d·d! matrices were built first;
        # with lazy generation a small cap returns immediately.
        from tsdynamics.analysis.fixedpoints import _common as _c

        mats = _c.signed_permutation_matrices(8, max_count=5)  # full set is 10_321_920
        assert len(mats) == 5
        np.testing.assert_array_equal(mats[0], np.eye(8))

    def test_truncated_prefix_is_a_prefix_of_full(self) -> None:
        # The capped list is exactly the first ``max_count`` of the full ordering,
        # so SD and DL (both fed this list) sweep the identical subset.
        from tsdynamics.analysis.fixedpoints import _common as _c

        full = _c.signed_permutation_matrices(3)
        capped = _c.signed_permutation_matrices(3, max_count=7)
        assert len(capped) == 7
        for a, b in zip(capped, full[:7], strict=False):
            np.testing.assert_array_equal(a, b)


def test_dl_sweeps_same_matrices_as_sd_under_truncation() -> None:
    """DL must use ``C`` (not ``Cᵀ``), so a truncated DL/SD sweep is identical.

    Before the fix DL solved ``(β‖g‖Cᵀ − G)δ = g`` while SD used ``C``; under a
    ``max_c`` truncation the two transposed lists differ, so DL and SD explored
    different stabilisation subsets.  After the fix both reference the same ``C``,
    so a capped DL search reaches an unstable fixed point an SD search reaches.
    """
    from tsdynamics.analysis.fixedpoints import _common as _c

    # The transformation matrix DL multiplies by must be C itself: solve a tiny
    # DL step by hand and check the assembled matrix uses C, not C.T, for a
    # genuinely asymmetric C.
    c = np.array([[0.0, 1.0], [-1.0, 0.0]])  # C != C.T
    g = np.array([1.0, 0.0])
    jac = np.zeros((2, 2))
    beta = 1.0
    ng = float(np.linalg.norm(g))

    captured: dict[str, np.ndarray] = {}
    orig = np.linalg.solve

    def spy_solve(a, b):  # noqa: ANN001, ANN202
        captured["a"] = np.asarray(a).copy()
        return orig(a, b)

    np.linalg.solve = spy_solve  # type: ignore[assignment]
    try:
        _c.converge_root(
            lambda x: g,
            lambda x: jac,
            np.zeros(2),
            method="dl",
            c_mat=c,
            lam=0.05,
            beta=beta,
            tol=1e-12,
            max_iter=1,
        )
    finally:
        np.linalg.solve = orig  # type: ignore[assignment]

    # The assembled matrix is β‖g‖·C − G == β·1·C (G = 0): it must equal C, not Cᵀ.
    np.testing.assert_allclose(captured["a"], beta * ng * c)
    assert not np.allclose(captured["a"], beta * ng * c.T)


def test_periodic_orbits_shared_monodromy_is_numerically_identical() -> None:
    """Sharing the p-fold monodromy between residual & Jacobian must not move the answer.

    The residual ``f^p(x) − x`` and its Jacobian ``Df^p − I`` are now read from a
    single cached ``map_orbit_monodromy`` sweep per iterate (instead of two
    independent sweeps).  The cache is bit-for-bit equivalent to recomputing, so
    the converged 2-cycle (and its multipliers/closure residual) must match the
    analytic values exactly — this guards the refactor against any divergence
    from the un-cached path.
    """
    orbs = periodic_orbits(ts.Logistic(params={"r": 3.2}), 2, method="newton", seed=3)
    assert len(orbs) == 1
    o = orbs[0]
    r = 3.2
    disc = np.sqrt((r + 1) * (r - 3))
    expected = sorted([(r + 1 + disc) / (2 * r), (r + 1 - disc) / (2 * r)])
    np.testing.assert_allclose(sorted(o.points.ravel()), expected, atol=1e-9)
    # multiplier of the 2-cycle is 4 + 2r − r² = 0.16 at r=3.2
    assert float(np.prod(o.multipliers).real) == pytest.approx(0.16, abs=1e-6)
    assert o.residual < 1e-9


def test_periodic_orbits_monodromy_shared_per_iterate() -> None:
    """One ``map_orbit_monodromy`` sweep per iterate, not two (residual + Jacobian).

    DL evaluates both the residual and the Jacobian at the same iterate inside one
    Newton step; before the fix each rebuilt the full p-fold orbit + monodromy, so
    a step cost two sweeps.  We monkeypatch the sweep to count distinct iterates
    vs total calls and assert no iterate is swept twice in a row (the cache hit).
    """
    from tsdynamics.analysis.fixedpoints import _common as _c

    seen: list[bytes] = []
    orig = _c.map_orbit_monodromy

    def counting(step: Any, jac: Any, x: np.ndarray, period: int, dim: int) -> Any:
        seen.append(np.asarray(x, dtype=float).ravel().tobytes())
        return orig(step, jac, x, period, dim)

    _c.map_orbit_monodromy = counting  # type: ignore[assignment]
    try:
        periodic_orbits(ts.Logistic(params={"r": 3.2}), 2, method="dl", seed=3, max_iter=50)
    finally:
        _c.map_orbit_monodromy = orig  # type: ignore[assignment]

    # No iterate triggers two *consecutive* sweeps (residual then Jacobian on the
    # same x): the second is served from the per-iterate cache.  Pre-fix, every
    # such pair produced two identical consecutive keys.
    consecutive_dupes = sum(1 for a, b in zip(seen, seen[1:], strict=False) if a == b)
    assert consecutive_dupes == 0


class TestEstimatePeriodFFTRefinement:
    """The FFT path must parabolically refine the peak bin (sub-bin resolution)."""

    def test_fft_beats_coarse_bin_resolution(self) -> None:
        # Coarse FFT bins: with a modest n the bin grid near this period is wide,
        # so the true frequency falls clearly *between* bins and the raw n/k
        # estimate is visibly off — parabolic refinement must close the gap.
        n = 200
        dt = 1.0
        true_period = 13.7
        t = np.arange(n) * dt
        # superpose harmonics so the periodogram peak is sharp and the parabola fit
        # is well-conditioned, but the fundamental still sets the period
        y = np.sin(2 * np.pi * t / true_period) + 0.3 * np.sin(4 * np.pi * t / true_period)
        est = float(estimate_period(y, dt=dt, method="fft"))
        k = round(n / true_period)
        coarse = n / k  # the un-refined single-bin estimate
        assert abs(coarse - true_period) > 0.05  # the grid is genuinely coarse here
        assert abs(est - true_period) < abs(coarse - true_period)  # refinement helps
        # parabolic interpolation on a power periodogram closes most (not all) of
        # the coarse-bin gap; ~0.25-sample residual bias is expected at n=200.
        assert est == pytest.approx(true_period, abs=0.3)


class TestSeedDeterminism:
    """``seed=`` must fully determine the multi-start sampling (issue #487).

    The seeding (random box seeds + the burn-in orbit's starting state) must be
    drawn from a *local* :class:`numpy.random.Generator`, never the process-global
    ``numpy.random`` state — otherwise the result depends on import/test order.
    """

    def test_fixed_points_seed_independent_of_global_rng(self) -> None:
        # The issue #487 repro: a continuous flow with no ``default_ic`` (so the
        # burn-in orbit's start IC falls through to the random branch) and an
        # explicit ``region`` (so ``resolve_box`` does not sample, but
        # ``_build_seeds`` still calls ``sample_orbit_box``).
        from tsdynamics.data import Box

        region = Box([-12.0] * 3, [12.0] * 3)

        np.random.seed(123)
        a = fixed_points(ts.Thomas(), region=region, n_seeds=200, seed=0)
        np.random.seed(999)
        b = fixed_points(ts.Thomas(), region=region, n_seeds=200, seed=0)

        xa = sorted(tuple(np.asarray(fp.x)) for fp in a)
        xb = sorted(tuple(np.asarray(fp.x)) for fp in b)
        assert len(a) == len(b)
        assert xa == xb

    def test_fixed_points_auto_region_independent_of_global_rng(self) -> None:
        # ``region=None`` routes randomness through ``resolve_box`` → the burn-in
        # orbit as well, so cover that path too.
        np.random.seed(1)
        c = fixed_points(ts.Lorenz(), n_seeds=100, seed=7)
        np.random.seed(2)
        d = fixed_points(ts.Lorenz(), n_seeds=100, seed=7)

        xc = sorted(tuple(np.asarray(fp.x)) for fp in c)
        xd = sorted(tuple(np.asarray(fp.x)) for fp in d)
        assert len(c) == len(d)
        assert xc == xd

    def test_periodic_orbits_seed_independent_of_global_rng(self) -> None:
        np.random.seed(5)
        a = periodic_orbits(ts.Henon(), 2, seed=0)
        np.random.seed(50)
        b = periodic_orbits(ts.Henon(), 2, seed=0)

        xa = sorted(tuple(np.asarray(o.points[0])) for o in a)
        xb = sorted(tuple(np.asarray(o.points[0])) for o in b)
        assert len(a) == len(b)
        assert xa == xb

    def test_burn_in_start_ic_uses_local_rng(self) -> None:
        # The unit-level guarantee: ``_orbit_start_ic`` draws its random fallback
        # from the supplied Generator and preserves the ``ic`` / ``default_ic``
        # priority unchanged.
        from tsdynamics.analysis.fixedpoints._common import _orbit_start_ic

        sys = ts.Thomas()  # default_ic is None → random fallback
        np.random.seed(123)
        x1 = _orbit_start_ic(sys, 3, np.random.default_rng(0))
        np.random.seed(999)
        x2 = _orbit_start_ic(sys, 3, np.random.default_rng(0))
        assert np.array_equal(x1, x2)

        # An explicit ``system.ic`` is honored verbatim (priority preserved).
        seeded = ts.Thomas(ic=[1.0, 2.0, 3.0])
        y = _orbit_start_ic(seeded, 3, np.random.default_rng(0))
        assert np.array_equal(y, [1.0, 2.0, 3.0])
