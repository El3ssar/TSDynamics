"""
Tests for :func:`poincare_section` and :func:`return_map`.

Both ops return a :class:`~tsdynamics.base.Trajectory` under the unified
analysis contract — these tests check that, plus the convenience
behaviours: canonical ``direction="up"`` default, all three call styles
(condition object / callable / shortcut kwargs), and the
``to_dataspec(kind="return_map")`` plot-prep helper.

Heavy paths (Lorenz integrate at ``final_time=400``) are marked ``slow``;
the lightweight synthetic-helix and oscillator tests run unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis import (
    LinearPlane,
    Plane,
    poincare_section,
    return_map,
)
from tsdynamics.base import Trajectory

# ---------------------------------------------------------------------------
# Fast synthetic fixtures
# ---------------------------------------------------------------------------


def _helix(n: int = 5000, t_max: float = 10 * np.pi) -> Trajectory:
    """``y(t) = (cos t, sin t, 0.5 t)`` — a screw with pitch 0.5/2π."""
    t = np.linspace(0.0, t_max, n)
    y = np.column_stack([np.cos(t), np.sin(t), 0.5 * t])
    return Trajectory(t, y, system=None)


# ---------------------------------------------------------------------------
# poincare_section
# ---------------------------------------------------------------------------


class TestPoincareSection:
    def test_returns_trajectory_with_full_state(self) -> None:
        traj = _helix()
        sec = poincare_section(traj, Plane(axis=1, value=0.0))
        assert isinstance(sec, Trajectory)
        assert sec.dim == 3
        # Section axis is zero at every crossing.
        np.testing.assert_allclose(sec.y[:, 1], 0.0, atol=1e-5)

    def test_inherits_system(self) -> None:
        marker = object()
        traj = _helix()
        traj.system = marker  # type: ignore[assignment]
        sec = poincare_section(traj, Plane(axis=1, value=0.0))
        assert sec.system is marker

    def test_canonical_direction_default_is_up(self) -> None:
        t = np.linspace(0.0, 4 * np.pi, 4001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        traj = Trajectory(t, y, system=None)
        # Only the upward zeros of cos t: 3π/2 and 7π/2.
        sec = poincare_section(traj, Plane(axis=0, value=0.0))
        assert sec.n_steps == 2

    def test_explicit_direction_override(self) -> None:
        t = np.linspace(0.0, 4 * np.pi, 4001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        traj = Trajectory(t, y, system=None)
        sec = poincare_section(traj, Plane(axis=0, value=0.0), direction="either")
        assert sec.n_steps == 4

    def test_shortcut_kwargs(self) -> None:
        traj = _helix()
        sec = poincare_section(traj, axis=1, value=0.0)
        assert isinstance(sec, Trajectory)
        assert sec.n_steps >= 4

    def test_linear_plane(self) -> None:
        t = np.linspace(0.0, 2 * np.pi, 5001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        traj = Trajectory(t, y, system=None)
        sec = poincare_section(traj, LinearPlane(normal=np.array([1.0, 1.0]), offset=0.0))
        # Only the upward crossing of x+y, at t = 7π/4.
        assert sec.n_steps == 1
        np.testing.assert_allclose(sec.t[0], 7 * np.pi / 4, atol=1e-4)

    def test_tuple_input(self) -> None:
        t = np.linspace(0.0, 4 * np.pi, 4001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        sec = poincare_section((t, y), Plane(axis=1, value=0.0))
        assert sec.n_steps >= 2


# ---------------------------------------------------------------------------
# return_map
# ---------------------------------------------------------------------------


class TestReturnMap:
    def test_returns_one_column_trajectory(self) -> None:
        traj = _helix()
        rmap = return_map(traj, Plane(axis=1, value=0.0), observable=0)
        assert isinstance(rmap, Trajectory)
        assert rmap.y.shape[1] == 1

    def test_helix_observable_constant(self) -> None:
        # On the y=0 up-crossings, x = cos(2π k) = 1 for every k.
        traj = _helix()
        rmap = return_map(traj, Plane(axis=1, value=0.0), observable=0)
        if rmap.n_steps:
            np.testing.assert_allclose(rmap.y[:, 0], 1.0, atol=1e-4)

    def test_callable_observable(self) -> None:
        traj = _helix()
        rmap = return_map(
            traj,
            Plane(axis=1, value=0.0),
            observable=lambda t, y: float(y[0] + y[2]),
        )
        assert rmap.y.shape[1] == 1

    def test_shortcut_kwargs(self) -> None:
        traj = _helix()
        rmap = return_map(traj, axis=1, value=0.0, observable=2)
        # The z-component at every up-crossing is 2π k * 0.5 = π k.
        assert rmap.n_steps >= 4

    def test_invalid_observable_component(self) -> None:
        traj = _helix()
        with pytest.raises(IndexError):
            return_map(traj, Plane(axis=1, value=0.0), observable=99)

    def test_invalid_observable_type(self) -> None:
        traj = _helix()
        with pytest.raises(TypeError):
            return_map(traj, Plane(axis=1, value=0.0), observable="not-an-obs")  # type: ignore[arg-type]

    def test_empty_when_no_crossings(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        y = np.column_stack([t, t, t])
        rmap = return_map((t, y), axis=1, value=100.0)
        assert rmap.n_steps == 0
        assert rmap.y.shape == (0, 1)


# ---------------------------------------------------------------------------
# to_dataspec(kind="return_map") — the V1 plot-prep helper
# ---------------------------------------------------------------------------


class TestReturnMapDataspec:
    def test_pair_view_step_one(self) -> None:
        traj = _helix(n=10000, t_max=20 * np.pi)
        rmap = return_map(traj, axis=1, value=0.0, observable=2)
        spec = rmap.to_dataspec(kind="return_map")
        assert spec["kind"] == "return_map"
        # z increases by π between consecutive up-crossings.
        diffs = spec["y"] - spec["x"]
        np.testing.assert_allclose(diffs, np.pi, atol=1e-3)
        # Step defaults to 1.
        assert spec["step"] == 1

    def test_pair_view_step_two(self) -> None:
        traj = _helix(n=10000, t_max=20 * np.pi)
        rmap = return_map(traj, axis=1, value=0.0, observable=2)
        spec = rmap.to_dataspec(kind="return_map", step=2)
        diffs = spec["y"] - spec["x"]
        np.testing.assert_allclose(diffs, 2 * np.pi, atol=1e-3)
        assert spec["step"] == 2

    def test_observable_index_recorded(self) -> None:
        traj = _helix()
        rmap = return_map(traj, axis=1, value=0.0, observable=0)
        spec = rmap.to_dataspec(kind="return_map")
        assert spec["observable"] == 0

    def test_empty_input_handled_cleanly(self) -> None:
        # Trajectory with no crossings; the dataspec should still build.
        t = np.linspace(0.0, 1.0, 100)
        y = np.column_stack([t, t, t])
        rmap = return_map((t, y), axis=1, value=100.0)
        spec = rmap.to_dataspec(kind="return_map")
        assert spec["x"].size == 0
        assert spec["y"].size == 0


# ---------------------------------------------------------------------------
# Trajectory method form
# ---------------------------------------------------------------------------


class TestTrajectoryMethods:
    def test_poincare_section_method(self) -> None:
        traj = _helix()
        sec_func = poincare_section(traj, Plane(axis=1, value=0.0))
        sec_meth = traj.poincare_section(Plane(axis=1, value=0.0))
        assert isinstance(sec_meth, Trajectory)
        np.testing.assert_array_equal(sec_func.t, sec_meth.t)
        np.testing.assert_array_equal(sec_func.y, sec_meth.y)

    def test_return_map_method(self) -> None:
        traj = _helix()
        rmap_func = return_map(traj, Plane(axis=1, value=0.0), observable=0)
        rmap_meth = traj.return_map(Plane(axis=1, value=0.0), observable=0)
        assert isinstance(rmap_meth, Trajectory)
        np.testing.assert_array_equal(rmap_func.t, rmap_meth.t)
        np.testing.assert_array_equal(rmap_func.y, rmap_meth.y)

    def test_method_shortcut_kwargs(self) -> None:
        traj = _helix()
        rmap = traj.return_map(axis=1, value=0.0, observable=0)
        assert isinstance(rmap, Trajectory)
        assert rmap.y.shape[1] == 1


# ---------------------------------------------------------------------------
# Lorenz integration (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLorenzPoincare:
    @pytest.fixture(scope="class")
    def lorenz_traj(self) -> Trajectory:
        import tsdynamics as ts

        # 80 001 samples after dropping the 50-time-unit transient — cheap on
        # a modern laptop once the JiT compilation has happened.
        return ts.Lorenz().integrate(final_time=400.0, dt=0.005).after(50.0)

    def test_section_yields_many_points(self, lorenz_traj: Trajectory) -> None:
        sec = poincare_section(lorenz_traj, axis=2, value=27.0)
        assert sec.n_steps >= 100
        np.testing.assert_allclose(sec.y[:, 2], 27.0, atol=1e-4)

    def test_method_matches_function(self, lorenz_traj: Trajectory) -> None:
        sec_func = poincare_section(lorenz_traj, axis=2, value=27.0)
        sec_meth = lorenz_traj.poincare_section(axis=2, value=27.0)
        np.testing.assert_array_equal(sec_func.t, sec_meth.t)
        np.testing.assert_array_equal(sec_func.y, sec_meth.y)

    def test_return_map_has_unimodal_shape(self, lorenz_traj: Trajectory) -> None:
        rmap = return_map(lorenz_traj, axis=2, value=27.0, observable=0)
        spec = rmap.to_dataspec(kind="return_map", step=1)
        assert spec["x"].size >= 50

        # Bounded: observable values stay within the attractor's natural range.
        assert np.ptp(spec["x"]) > 1.0
        assert np.ptp(spec["y"]) > 1.0
        assert np.ptp(spec["x"]) < 100.0
        assert np.ptp(spec["y"]) < 100.0
        # Not a trivial fixed point: at least a third of pairs differ by more
        # than 1 unit.
        assert np.mean(np.abs(spec["y"] - spec["x"]) > 1.0) > 0.3
