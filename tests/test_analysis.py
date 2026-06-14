"""Analysis pack: orbit diagrams, Poincaré sections, Lyapunov tools, fixed points."""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts

# ---------------------------------------------------------------------------
# kaplan_yorke_dimension (pure function)
# ---------------------------------------------------------------------------


class TestKaplanYorke:
    def test_lorenz_literature_value(self) -> None:
        d = ts.kaplan_yorke_dimension([0.906, 0.0, -14.57])
        assert d == pytest.approx(2.062, abs=0.01)

    def test_all_negative_is_zero(self) -> None:
        assert ts.kaplan_yorke_dimension([-0.1, -1.0]) == 0.0

    def test_never_closing_saturates(self) -> None:
        assert ts.kaplan_yorke_dimension([0.2, 0.1]) == 2.0

    def test_order_independent(self) -> None:
        a = ts.kaplan_yorke_dimension([0.906, 0.0, -14.57])
        b = ts.kaplan_yorke_dimension([-14.57, 0.906, 0.0])
        assert a == b


# ---------------------------------------------------------------------------
# orbit_diagram on maps (fast)
# ---------------------------------------------------------------------------


class TestOrbitDiagram:
    @staticmethod
    def _branches(points: np.ndarray, decimals: int = 4) -> int:
        return len(np.unique(np.round(points[:, 0], decimals)))

    def test_logistic_period_doubling(self) -> None:
        od = ts.orbit_diagram(
            ts.Logistic(),
            "r",
            [3.2, 3.5],
            n=120,
            transient=600,
            carry_state=False,
            ic=[0.5],
        )
        assert self._branches(od.points[0]) == 2  # 2-cycle at r=3.2
        assert self._branches(od.points[1]) == 4  # 4-cycle at r=3.5

    def test_logistic_chaotic_band_dense(self) -> None:
        od = ts.orbit_diagram(ts.Logistic(), "r", [3.9], n=200, transient=500, ic=[0.5])
        assert self._branches(od.points[0]) > 50

    def test_flat_output(self) -> None:
        od = ts.orbit_diagram(ts.Logistic(), "r", [3.2, 3.5], n=50, transient=200, ic=[0.5])
        x, y = od.flat()
        assert x.shape == y.shape == (100,)
        assert set(np.unique(x)) == {3.2, 3.5}

    def test_carry_state_follows_branch(self) -> None:
        od = ts.orbit_diagram(
            ts.Logistic(), "r", np.linspace(2.8, 3.4, 7), n=40, transient=300, ic=[0.5]
        )
        assert len(od) == 7
        assert all(np.all(np.isfinite(p)) for p in od.points)

    def test_continuous_system_rejected(self) -> None:
        with pytest.raises(TypeError, match="discrete"):
            ts.orbit_diagram(ts.Lorenz(), "rho", [28.0])

    def test_original_system_not_mutated(self) -> None:
        m = ts.Logistic()
        ts.orbit_diagram(m, "r", [3.0], n=10, transient=10, ic=[0.5])
        assert m.params["r"] == 3.9


# ---------------------------------------------------------------------------
# fixed_points on maps (fast)
# ---------------------------------------------------------------------------


class TestFixedPoints:
    def test_henon_analytic_fixed_points(self) -> None:
        fps = ts.fixed_points(ts.Henon(), seed=0)
        a, b = 1.4, 0.3
        disc = np.sqrt((1 - b) ** 2 + 4 * a)
        expected_x = sorted([(-(1 - b) + disc) / (2 * a), (-(1 - b) - disc) / (2 * a)])
        found_x = sorted(fp.x[0] for fp in fps)
        np.testing.assert_allclose(found_x, expected_x, rtol=1e-8)
        # classic Hénon fixed points are both unstable (saddles)
        assert all(not fp.stable for fp in fps)
        # y* = b x*
        for fp in fps:
            assert fp.x[1] == pytest.approx(b * fp.x[0], rel=1e-8)

    def test_logistic_fixed_points(self) -> None:
        m = ts.Logistic(params={"r": 2.5})
        fps = ts.fixed_points(m, box=([-0.5], [1.5]), seed=0)
        xs = sorted(fp.x[0] for fp in fps)
        np.testing.assert_allclose(xs, [0.0, 1 - 1 / 2.5], atol=1e-9)
        stable = {round(fp.x[0], 6): fp.stable for fp in fps}
        assert stable[0.0] is False
        assert stable[round(1 - 1 / 2.5, 6)] is True


# ---------------------------------------------------------------------------
# max_lyapunov (fast on maps)
# ---------------------------------------------------------------------------


class TestMaxLyapunov:
    def test_logistic_r4_ln2(self) -> None:
        m = ts.Logistic(params={"r": 4.0})
        lam = ts.max_lyapunov(m, ic=[0.3], n_rescale=600, steps_per=3, seed=1)
        assert lam == pytest.approx(np.log(2), abs=0.1)

    def test_henon(self) -> None:
        lam = ts.max_lyapunov(ts.Henon(), ic=[0.1, 0.1], n_rescale=600, steps_per=3, seed=1)
        assert lam == pytest.approx(0.42, abs=0.12)

    def test_dde_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="set_state"):
            ts.max_lyapunov(ts.MackeyGlass())


# ---------------------------------------------------------------------------
# Slow: flows
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_max_lyapunov_lorenz() -> None:
    lam = ts.max_lyapunov(
        ts.Lorenz(),
        ic=[1.0, 1.0, 1.0],
        dt=0.05,
        n_rescale=600,
        steps_per=4,
        transient=1000,
        seed=2,
    )
    assert lam == pytest.approx(0.906, abs=0.2)


@pytest.mark.slow
def test_poincare_section_from_system_thin_set() -> None:
    section = ts.poincare_section(
        ts.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), steps=100, transient=10, dt=0.05
    )
    assert section.y.shape == (100, 3)
    assert np.max(np.abs(section.y[:, 0])) < 1e-6


@pytest.mark.slow
def test_poincare_section_from_trajectory_data() -> None:
    traj = ts.Lorenz().integrate(final_time=50.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    section = ts.poincare_section(traj, plane=(2, 27.0), direction=0)
    assert section.y.shape[0] > 10
    # linear interpolation puts the crossing near the plane (dt-limited accuracy)
    assert np.max(np.abs(section.y[:, 2] - 27.0)) < 1e-8


@pytest.mark.slow
def test_lorenz_kaplan_yorke_from_spectrum() -> None:
    spec = ts.Lorenz(ic=[1.0, 1.0, 1.0]).lyapunov_spectrum(
        dt=0.1, burn_in=50.0, final_time=300.0, method="dop853", rtol=1e-7, atol=1e-10
    )
    d = ts.kaplan_yorke_dimension(spec)
    assert d == pytest.approx(2.06, abs=0.1)


@pytest.mark.slow
def test_bifurcation_diagram_of_flow_via_poincare() -> None:
    """The composition acceptance test: orbit diagram over a PoincareMap."""
    pmap = ts.PoincareMap(ts.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.05)
    od = ts.orbit_diagram(
        pmap, "c", [4.0, 5.7], n=15, transient=10, components=1, ic=[1.0, 1.0, 0.0]
    )
    assert len(od) == 2
    for _, pts in od:
        assert pts.shape == (15, 1)
        assert np.all(np.isfinite(pts))


@pytest.mark.slow
def test_orbit_diagram_named_component_over_poincare() -> None:
    """Regression: a NAMED component over a derived wrapper must resolve via the
    instance, not ``type(sys).variables`` — which leaks the property descriptor
    and raised ``AttributeError: 'property' object has no attribute 'index'``."""
    pmap = ts.PoincareMap(ts.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.05)
    od = ts.orbit_diagram(pmap, "c", [5.7], n=10, transient=10, components="y", ic=[1.0, 1.0, 0.0])
    assert len(od) == 1
    ((_, pts),) = list(od)
    assert pts.shape == (10, 1)
