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
        fps = fixed_points(m, box=([-0.5], [1.5]), seed=0)
        xs = sorted(fp.x[0] for fp in fps)
        np.testing.assert_allclose(xs, [0.0, 1 - 1 / 2.5], atol=1e-9)
        stable = {round(fp.x[0], 6): fp.stable for fp in fps}
        assert stable[0.0] is False
        assert stable[round(1 - 1 / 2.5, 6)] is True

    def test_logistic_r4_unstable_fixed_points(self) -> None:
        # at r=4 both fixed points {0, 0.75} are unstable; DL must still find them
        fps = fixed_points(ts.Logistic(params={"r": 4.0}), box=([-0.2], [1.2]), method="dl", seed=2)
        xs = sorted(fp.x[0] for fp in fps)
        np.testing.assert_allclose(xs, [0.0, 0.75], atol=1e-8)
        assert all(not fp.stable for fp in fps)

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            fixed_points(ts.Henon(), method="bogus")


# ── flow equilibria ─────────────────────────────────────────────────────────


class TestFlowEquilibria:
    def test_lorenz_equilibria(self) -> None:
        fps = fixed_points(ts.Lorenz(), box=([-30, -30, -5], [30, 30, 55]), n_seeds=300, seed=1)
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
        fps = fixed_points(ts.Rossler(), box=([-1, -30, -1], [8, 1, 30]), n_seeds=500, seed=2)
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
        orbs = periodic_orbits(ts.Logistic(params={"r": 2.5}), 1, box=([-0.5], [1.5]), seed=5)
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
            _VanDerPol(params={"mu": 1.0}), ic=[2.0, 0.0], period_guess=6.0, burn_in=20.0
        )
        assert orb.period == pytest.approx(6.6632869, abs=2e-3)
        assert orb.residual < 1e-8 and orb.continuous and orb.stable
        # one trivial Floquet multiplier ≈ 1 (the flow direction)
        assert np.abs(np.abs(orb.multipliers) - 1.0).min() < 1e-2

    def test_vanderpol_mu05(self) -> None:
        orb = periodic_orbit(
            _VanDerPol(params={"mu": 0.5}), ic=[2.0, 0.0], period_guess=6.3, burn_in=30.0
        )
        assert orb.period == pytest.approx(6.3806758, abs=2e-3)
        assert orb.stable

    def test_auto_period_guess(self) -> None:
        orb = periodic_orbit(_VanDerPol(params={"mu": 1.0}), ic=[0.5, 0.5], burn_in=20.0)
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
