"""A-ORBIT: return maps, bifurcation quantification, and registry wiring.

The orbit-diagram and Poincaré-section *paths* are exercised in
``test_analysis.py``; this module covers the A-ORBIT additions — the
first-return map, ``OrbitDiagram`` period/bifurcation quantifiers, and the
``registry.analyses`` self-registration.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import ReturnMap, registry, return_map


def _functionality(cur: np.ndarray, suc: np.ndarray) -> float:
    """How single-valued ``suc = F(cur)`` is: median successor jump between
    current-adjacent points, as a fraction of the successor range.  Near 0 for
    a tight 1-D map; ~0.3+ for a 2-D cloud."""
    order = np.argsort(cur)
    s = suc[order]
    rng = s.max() - s.min()
    return float(np.median(np.abs(np.diff(s))) / rng)


# ---------------------------------------------------------------------------
# return_map — extremum mode on synthetic series (fast, no integration)
# ---------------------------------------------------------------------------


class TestReturnMapSeries:
    def test_known_maxima(self) -> None:
        s = np.array([0, 1, 0, 2, 0, 3, 0, 2.5, 0], dtype=float)
        rm = return_map(s, method="max")
        np.testing.assert_allclose(rm.values, [1.0, 2.0, 3.0, 2.5])
        np.testing.assert_allclose(rm.current, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(rm.successor, [2.0, 3.0, 2.5])
        assert isinstance(rm, ReturnMap)
        assert len(rm) == 3

    def test_minima(self) -> None:
        s = np.array([0, -1, 0, -2, 0, -3, 0], dtype=float)
        rm = return_map(s, method="min")
        np.testing.assert_allclose(rm.values, [-1.0, -2.0, -3.0])

    def test_parabolic_refinement_sharpens_peak(self) -> None:
        # Sample a cosine peak off-grid: the true maximum (1.0) lies between
        # samples, so the parabolic-refined value beats the raw sample max.
        t = np.linspace(-0.37, 2 * np.pi - 0.37, 64)  # peak of cos at t=0 is off-grid
        s = np.cos(t)
        rm = return_map(s, method="max")
        assert rm.values.size == 1
        assert s.max() < rm.values[0] <= 1.0 + 1e-9

    def test_flat_and_iter(self) -> None:
        s = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        rm = return_map(s, method="max")
        cur, suc = rm.flat()
        assert cur.shape == suc.shape == (2,)
        pairs = list(rm)
        assert pairs[0] == (1.0, 2.0)

    def test_constant_amplitude_collapses_to_diagonal(self) -> None:
        # A pure sine: every maximum is equal → the return map is one point on
        # the diagonal (current == successor).
        t = np.linspace(0, 60, 6000)
        rm = return_map(np.sin(2 * np.pi * t), method="max")
        assert rm.values.size > 5
        np.testing.assert_allclose(rm.current, rm.successor, atol=1e-6)
        np.testing.assert_allclose(rm.values, rm.values[0], atol=1e-6)

    def test_too_short_series_is_empty(self) -> None:
        rm = return_map(np.array([1.0, 2.0]), method="max")
        assert rm.values.size == 0
        assert len(rm) == 0


# ---------------------------------------------------------------------------
# return_map — input validation
# ---------------------------------------------------------------------------


class TestReturnMapValidation:
    def test_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            return_map(np.zeros(10), method="bogus")

    def test_2d_raw_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            return_map(np.zeros((10, 2)), method="max")

    def test_poincare_needs_plane(self) -> None:
        with pytest.raises(ValueError, match="plane"):
            return_map(ts.Rossler(), method="poincare")

    def test_poincare_rejects_raw_series(self) -> None:
        with pytest.raises(TypeError, match="System or Trajectory"):
            return_map(np.zeros(10), method="poincare", plane=(0, 0.0))

    def test_discrete_map_rejected_for_extrema(self) -> None:
        with pytest.raises(TypeError, match="continuous flow"):
            return_map(ts.Henon(), method="max")

    def test_unknown_named_observable(self) -> None:
        traj = ts.Lorenz().integrate(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="unknown component"):
            return_map(traj, "nope", method="max")


# ---------------------------------------------------------------------------
# return_map — Poincaré mode on trajectory data (fast, synthetic)
# ---------------------------------------------------------------------------


class TestReturnMapPoincareData:
    @staticmethod
    def _circle_traj() -> ts.Trajectory:
        # a clean limit cycle: (sin, cos) crosses the x=0 plane once per period
        t = np.linspace(0.0, 10.0 * np.pi, 4000)
        y = np.column_stack([np.sin(t), np.cos(t)])
        return ts.Trajectory(t, y, None)

    def test_crossings_from_data(self) -> None:
        rm = return_map(self._circle_traj(), 1, method="poincare", plane=(0, 0.0), direction=1)
        assert rm.kind == "poincare"
        assert rm.values.size > 2
        # y = cos at the up-crossings of sin is ≈ +1 each period → a fixed point
        np.testing.assert_allclose(rm.values, 1.0, atol=1e-3)

    def test_transient_drops_leading_crossings(self) -> None:
        traj = self._circle_traj()
        full = return_map(traj, 1, method="poincare", plane=(0, 0.0), direction=1)
        skipped = return_map(
            traj, 1, method="poincare", plane=(0, 0.0), direction=1, skip_crossings=2
        )
        assert skipped.values.size == full.values.size - 2
        np.testing.assert_array_equal(skipped.values, full.values[2:])


# ---------------------------------------------------------------------------
# OrbitDiagram.periods / bifurcation_points (fast — logistic map)
# ---------------------------------------------------------------------------


class TestOrbitDiagramQuantifiers:
    def test_period_doubling_sequence(self) -> None:
        od = ts.orbit_diagram(
            ts.Logistic(),
            "r",
            [2.8, 3.2, 3.5, 3.56],
            n=120,
            transient=2000,
            carry_state=False,
            ic=[0.5],
        )
        p = od.periods()
        assert p[0] == 1  # fixed point
        assert p[1] == 2  # 2-cycle
        assert p[2] == 4  # 4-cycle
        assert p[3] == 8  # 8-cycle

    def test_chaotic_band_is_aperiodic(self) -> None:
        od = ts.orbit_diagram(ts.Logistic(), "r", [3.9], n=200, transient=500, ic=[0.5])
        assert od.periods()[0] == 0  # too many branches → reported aperiodic

    def test_empty_value_is_minus_one(self) -> None:
        # r > 4 escapes [0, 1]: the sweep records an empty set (diverges).
        with pytest.warns(RuntimeWarning, match="diverged"):
            od = ts.orbit_diagram(ts.Logistic(), "r", [4.5], n=50, transient=50, ic=[0.5])
        assert od.periods()[0] == -1

    def test_bifurcation_points_match_literature(self) -> None:
        # Logistic period-doubling onsets: r1 = 3, r2 = 1 + sqrt(6) ≈ 3.449.
        od = ts.orbit_diagram(
            ts.Logistic(), "r", np.linspace(2.9, 3.6, 400), n=64, transient=2000, ic=[0.5]
        )
        bp = od.bifurcation_points()
        assert np.min(np.abs(bp - 3.0)) < 0.03
        assert np.min(np.abs(bp - (1.0 + np.sqrt(6.0)))) < 0.02


# ---------------------------------------------------------------------------
# registry self-registration
# ---------------------------------------------------------------------------


def test_orbit_analyses_self_register() -> None:
    names = registry.analyses.names()
    for n in ("orbit_diagram", "poincare_section", "return_map"):
        assert n in names
        assert registry.analyses.get(n) is getattr(ts, n)


# ---------------------------------------------------------------------------
# Slow: flows — the literature-validated return maps
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lorenz_z_maxima_cusp_map() -> None:
    """Lorenz (1963): successive maxima of z form a near-1-D cusp map."""
    rm = ts.return_map(
        ts.Lorenz(ic=[1.0, 1.0, 1.0]), "z", method="max", final_time=400.0, dt=0.01, transient=40.0
    )
    cur, suc = rm.flat()
    assert len(rm) > 100
    # the classic z-maxima live in a tight band around the cusp
    assert rm.values.min() > 28.0
    assert rm.values.max() < 50.0
    # and the map is effectively single-valued (a 1-D function)
    assert _functionality(cur, suc) < 0.05


@pytest.mark.slow
def test_rossler_poincare_return_map_is_1d() -> None:
    """y at successive x=0 crossings of Rössler is a tight 1-D return map."""
    rm = ts.return_map(
        ts.Rossler(ic=[1.0, 1.0, 0.0]),
        "y",
        method="poincare",
        plane=(0, 0.0),
        n=400,
        skip_crossings=20,
        dt=0.03,
    )
    assert rm.kind == "poincare"
    assert len(rm) > 100
    assert _functionality(*rm.flat()) < 0.05


@pytest.mark.slow
def test_periods_on_flow_bifurcation_diagram() -> None:
    """`periods()` reads the Rössler period-doubling route on a Poincaré section.

    A periodic-orbit branch recorded from a flow differs only by integration
    noise, so this exercises the scale-relative negligible-spread guard that
    keeps `_count_branches` honest for flows (period-1 must not shatter).
    """
    found = {}
    for c in (2.6, 3.5, 5.7):
        pmap = ts.PoincareMap(ts.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.03)
        od = ts.orbit_diagram(pmap, "c", [c], n=80, transient=100, component=1, ic=[3.0, 3.0, 0.0])
        found[c] = int(od.periods()[0])
    assert found[2.6] == 1  # period-1 limit cycle
    assert found[3.5] == 2  # period-2
    assert found[5.7] == 0  # chaotic band → aperiodic


@pytest.mark.slow
def test_system_and_trajectory_paths_agree() -> None:
    """The same integration, read as a System or a Trajectory, gives the same map."""
    ic = [1.0, 1.0, 1.0]
    transient = 30.0
    rm_sys = ts.return_map(
        ts.Lorenz(ic=ic), "z", method="max", final_time=200.0, dt=0.01, transient=transient
    )
    traj = ts.Lorenz(ic=ic).integrate(final_time=200.0, dt=0.01, ic=ic)
    rm_traj = ts.return_map(traj.after(transient), "z", method="max")
    np.testing.assert_allclose(rm_sys.values, rm_traj.values)
