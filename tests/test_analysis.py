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
        d = ts.analysis.kaplan_yorke_dimension([0.906, 0.0, -14.57])
        assert d == pytest.approx(2.062, abs=0.01)

    def test_all_negative_is_zero(self) -> None:
        assert ts.analysis.kaplan_yorke_dimension([-0.1, -1.0]) == 0.0

    def test_never_closing_saturates(self) -> None:
        assert ts.analysis.kaplan_yorke_dimension([0.2, 0.1]) == 2.0

    def test_order_independent(self) -> None:
        a = ts.analysis.kaplan_yorke_dimension([0.906, 0.0, -14.57])
        b = ts.analysis.kaplan_yorke_dimension([-14.57, 0.906, 0.0])
        assert a == b


# ---------------------------------------------------------------------------
# orbit_diagram on maps (fast)
# ---------------------------------------------------------------------------


class TestOrbitDiagram:
    @staticmethod
    def _branches(points: np.ndarray, decimals: int = 4) -> int:
        return len(np.unique(np.round(points[:, 0], decimals)))

    def test_logistic_period_doubling(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(),
            "r",
            [3.2, 3.5],
            points_per_value=120,
            transient=600,
            carry_state=False,
            ic=[0.5],
        )
        assert self._branches(od.points[0]) == 2  # 2-cycle at r=3.2
        assert self._branches(od.points[1]) == 4  # 4-cycle at r=3.5

    def test_logistic_chaotic_band_dense(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(), "r", [3.9], points_per_value=200, transient=500, ic=[0.5]
        )
        assert self._branches(od.points[0]) > 50

    def test_flat_output(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(), "r", [3.2, 3.5], points_per_value=50, transient=200, ic=[0.5]
        )
        x, y = od.flat()
        assert x.shape == y.shape == (100,)
        assert set(np.unique(x)) == {3.2, 3.5}

    def test_carry_state_follows_branch(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(),
            "r",
            np.linspace(2.8, 3.4, 7),
            points_per_value=40,
            transient=300,
            ic=[0.5],
        )
        assert len(od) == 7
        assert all(np.all(np.isfinite(p)) for p in od.points)

    def test_continuous_system_is_reduced_to_its_peak_map(self) -> None:
        """A raw flow is accepted since v6 (see tests/test_orbits.py for the contract)."""
        od = ts.analysis.orbit_diagram(
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]), "rho", [28.0], points_per_value=10, transient=15
        )
        assert od.meta["section"] == "successive maxima of x"

    def test_original_system_not_mutated(self) -> None:
        m = ts.systems.Logistic()
        ts.analysis.orbit_diagram(m, "r", [3.0], points_per_value=10, transient=10, ic=[0.5])
        assert m.params["r"] == 3.9


# ---------------------------------------------------------------------------
# fixed_points on maps (fast)
# ---------------------------------------------------------------------------


class TestFixedPoints:
    def test_henon_analytic_fixed_points(self) -> None:
        fps = ts.analysis.fixed_points(ts.systems.Henon(), seed=0)
        a, b = 1.4, 0.3
        disc = np.sqrt((1 - b) ** 2 + 4 * a)
        expected_x = sorted([(-(1 - b) + disc) / (2 * a), (-(1 - b) - disc) / (2 * a)])
        found_x = sorted(fp[0] for fp in fps)
        np.testing.assert_allclose(found_x, expected_x, rtol=1e-8)
        # classic Hénon fixed points are both unstable (saddles)
        assert not any(fps.stable)
        # y* = b x*
        # v6 D1: indexing/iterating a result set yields NUMBERS, so ``fp`` is
        # already the ``(dim,)`` point; the record is at ``fps.details[i]``.
        for fp in fps:
            assert fp[1] == pytest.approx(b * fp[0], rel=1e-8)

    def test_logistic_fixed_points(self) -> None:
        m = ts.systems.Logistic(params={"r": 2.5})
        fps = ts.analysis.fixed_points(m, region=[(-0.5, 1.5)], seed=0)
        xs = sorted(fp[0] for fp in fps)
        np.testing.assert_allclose(xs, [0.0, 1 - 1 / 2.5], atol=1e-9)
        stable = {round(fp.x[0], 6): fp.stable for fp in fps.details}
        assert stable[0.0] is False
        assert stable[round(1 - 1 / 2.5, 6)] is True


# ---------------------------------------------------------------------------
# the maximal exponent — one door since v6 (``max_lyapunov`` was retired)
# ---------------------------------------------------------------------------


class TestTheMaximalExponentHasOneDoor:
    """``lyapunov_spectrum(system, k=1)`` is the only way to ask for lambda_1.

    Two public functions answering one question with two different numbers is a
    trap: measured at HEAD before this change, ``max_lyapunov(henon, ic=[0.1,
    0.1])`` returned 0.4232673 and ``lyapunov_spectrum(henon, k=1, n=20000,
    ic=[0.1, 0.1])`` returned 0.4159989 — one nominal question, two answers,
    and nothing told the reader which to use.
    """

    def test_asking_for_the_old_name_hands_back_the_new_line(self) -> None:
        """A guess at ``max_lyapunov`` names the spelling that replaced it."""
        with pytest.raises(ImportError, match=r"lyapunov_spectrum\(system, k=1\)"):
            ts.analysis.max_lyapunov  # noqa: B018 - the attribute access IS the test

    def test_logistic_r4_is_ln2(self) -> None:
        m = ts.systems.Logistic(params={"r": 4.0})
        lam = float(np.asarray(ts.analysis.lyapunov_spectrum(m, k=1, n=1800, ic=[0.3]))[0])
        assert lam == pytest.approx(np.log(2), abs=0.1)

    def test_henon_matches_the_literature(self) -> None:
        lam = float(
            np.asarray(
                ts.analysis.lyapunov_spectrum(ts.systems.Henon(), k=1, n=1800, ic=[0.1, 0.1])
            )[0]
        )
        assert lam == pytest.approx(0.41922, abs=0.05)

    def test_a_map_burns_in_before_it_measures(self) -> None:
        """The burn-in the retired door had is now this one's default.

        A Lyapunov exponent is a property of the ATTRACTOR, so the iterates
        spent falling onto it are not part of it.  This door used to *refuse*
        ``transient`` on a map — "its QR iteration reorthonormalises from the
        initial condition, so there is nothing to discard" — which confuses the
        tangent frame with the base orbit, and was the whole of the numeric
        disagreement above.
        """
        hen = ts.systems.Henon()
        cold = float(
            np.asarray(
                ts.analysis.lyapunov_spectrum(hen, k=1, n=20_000, ic=[0.1, 0.1], transient=0)
            )[0]
        )
        default = float(
            np.asarray(ts.analysis.lyapunov_spectrum(hen, k=1, n=20_000, ic=[0.1, 0.1]))[0]
        )
        assert cold != default, "transient= is being ignored on a map"
        # and the burnt-in one is the number the retired door used to give
        assert default == pytest.approx(0.4232673343379148, abs=1e-9)

    def test_a_system_with_no_jacobian_is_still_answered(self) -> None:
        """A ``WrappedSystem`` has no RHS to differentiate — and no second door.

        ``max_lyapunov`` was the only way to reach the Jacobian-free
        two-trajectory machine; retiring it without folding that in would have
        deleted a capability, so ``lyapunov_spectrum`` now owns it.
        """

        def step(u, n):
            x = u[0]
            for _ in range(int(n)):
                x = 3.9 * x * (1 - x)
            return [x]

        w = ts.WrappedSystem(step, dim=1, family="map", ic=[0.5])
        spec = ts.analysis.lyapunov_spectrum(w, k=1, ic=[0.3])
        assert float(np.asarray(spec)[0]) == pytest.approx(0.494, abs=0.02)
        assert spec.meta["estimator"] == "two-trajectory"

    def test_two_trajectories_resolve_one_exponent_and_say_so(self) -> None:
        """``k > 1`` off a frame-less machine is refused by name, not truncated."""

        def step(u, n):
            return [0.5 * u[0] + 0.1 * u[1], 0.3 * u[1]]

        w = ts.WrappedSystem(step, dim=2, family="map", ic=[0.5, 0.5])
        with pytest.raises(ValueError, match="k=2 frame"):
            ts.analysis.lyapunov_spectrum(w, k=2)

    def test_dde_raises(self) -> None:
        with pytest.raises((NotImplementedError, TypeError, ValueError)):
            ts.analysis.lyapunov_spectrum(ts.systems.MackeyGlass(), k=1, n=10)


# ---------------------------------------------------------------------------
# Slow: flows
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lorenz_maximal_exponent_matches_the_literature() -> None:
    lam = float(
        np.asarray(
            ts.analysis.lyapunov_spectrum(
                ts.systems.Lorenz(), k=1, ic=[1.0, 1.0, 1.0], final_time=2000.0
            )
        )[0]
    )
    assert lam == pytest.approx(0.9056, abs=0.02)


@pytest.mark.slow
def test_poincare_section_from_system_thin_set() -> None:
    section = ts.analysis.poincare_section(
        ts.systems.Rossler(ic=[1.0, 1.0, 0.0]),
        plane=(0, 0.0),
        crossings=100,
        skip_crossings=10,
        dt=0.05,
    )
    assert section.y.shape == (100, 3)
    assert np.max(np.abs(section.y[:, 0])) < 1e-6


@pytest.mark.slow
def test_poincare_section_from_trajectory_data() -> None:
    traj = ts.systems.Lorenz().run(final_time=50.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    section = ts.analysis.poincare_section(traj, plane=(2, 27.0), direction=0)
    assert section.y.shape[0] > 10
    # linear interpolation puts the crossing near the plane (dt-limited accuracy)
    assert np.max(np.abs(section.y[:, 2] - 27.0)) < 1e-8


@pytest.mark.slow
def test_lorenz_kaplan_yorke_from_spectrum() -> None:
    spec = ts.analysis.lyapunov_spectrum(
        ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]),
        dt=0.1,
        transient=50.0,
        final_time=300.0,
        solver="dop853",  # v6: solver= is the kernel; method= is an estimator
        rtol=1e-7,
        atol=1e-10,
    )
    d = ts.analysis.kaplan_yorke_dimension(spec)
    assert d == pytest.approx(2.06, abs=0.1)


@pytest.mark.slow
def test_bifurcation_diagram_of_flow_via_poincare() -> None:
    """The composition acceptance test: orbit diagram over a PoincareMap."""
    pmap = ts.derived.PoincareMap(ts.systems.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.05)
    od = ts.analysis.orbit_diagram(
        pmap, "c", [4.0, 5.7], points_per_value=15, transient=10, components=1, ic=[1.0, 1.0, 0.0]
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
    pmap = ts.derived.PoincareMap(ts.systems.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.05)
    od = ts.analysis.orbit_diagram(
        pmap, "c", [5.7], points_per_value=10, transient=10, components="y", ic=[1.0, 1.0, 0.0]
    )
    assert len(od) == 1
    ((_, pts),) = list(od)
    assert pts.shape == (10, 1)
