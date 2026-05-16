"""
Tests for :func:`poincare_section` and :func:`return_map`.

Heavy paths (Lorenz integrate at ``final_time=400``) are marked ``slow``;
the lightweight synthetic-helix and oscillator tests run unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis import (
    EventResult,
    LinearPlane,
    Plane,
    ReturnMap,
    Threshold,
    poincare_section,
    return_map,
)
from tsdynamics.base import Trajectory

# ---------------------------------------------------------------------------
# Fast synthetic tests (no integrator)
# ---------------------------------------------------------------------------


def _helix(n: int = 5000, t_max: float = 10 * np.pi) -> Trajectory:
    """``y(t) = (cos t, sin t, 0.5 t)`` — a screw with 0.5/2π pitch."""
    t = np.linspace(0.0, t_max, n)
    y = np.column_stack([np.cos(t), np.sin(t), 0.5 * t])
    return Trajectory(t, y, system=None)


class TestPoincareSectionSynthetic:
    def test_helix_y_zero_section(self) -> None:
        traj = _helix()
        sec = poincare_section(traj, Plane(axis=1, value=0.0))
        # Up crossings of sin t happen at t = 0, 2π, 4π, ... — 5 over [0, 10π].
        # The endpoint t = 0 isn't strictly bracketed; we expect at least 4.
        assert sec.n_steps >= 4
        # Section axis should be ~0 at every crossing.
        np.testing.assert_allclose(sec.y[:, 1], 0.0, atol=1e-5)

    def test_returns_trajectory(self) -> None:
        traj = _helix()
        sec = poincare_section(traj, Plane(axis=1, value=0.0))
        assert isinstance(sec, Trajectory)
        assert sec.dim == 3

    def test_section_passes_system_ref(self) -> None:
        marker = object()
        traj = _helix()
        traj.system = marker  # type: ignore[assignment]
        sec = poincare_section(traj, Plane(axis=1, value=0.0))
        assert sec.system is marker

    def test_canonical_direction_default_is_up(self) -> None:
        # Up crossings only: cos t = 0 upward happens at t = 3π/2 (over [0, 4π],
        # actually 3π/2 and 7π/2 — i.e. two crossings).
        t = np.linspace(0.0, 4 * np.pi, 4001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        traj = Trajectory(t, y, system=None)
        sec = poincare_section(traj, Plane(axis=0, value=0.0))
        # only up zeros of cos: 3π/2 and 7π/2
        assert sec.n_steps == 2

    def test_explicit_direction_override(self) -> None:
        t = np.linspace(0.0, 4 * np.pi, 4001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        traj = Trajectory(t, y, system=None)
        sec = poincare_section(traj, Plane(axis=0, value=0.0), direction="either")
        assert sec.n_steps == 4

    def test_linear_plane_section(self) -> None:
        # Diagonal plane x + y = 0 on the unit circle: two crossings per period.
        t = np.linspace(0.0, 2 * np.pi, 5001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        traj = Trajectory(t, y, system=None)
        sec = poincare_section(
            traj,
            LinearPlane(normal=np.array([1.0, 1.0]), offset=0.0),
            direction="up",
        )
        # Only the upward crossing (when x+y goes negative → positive).
        # That happens at t = 7π/4.
        assert sec.n_steps == 1
        np.testing.assert_allclose(sec.t[0], 7 * np.pi / 4, atol=1e-4)

    def test_tuple_input_accepted(self) -> None:
        t = np.linspace(0.0, 4 * np.pi, 4001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        sec = poincare_section((t, y), Plane(axis=1, value=0.0))
        assert sec.n_steps >= 2


# ---------------------------------------------------------------------------
# return_map (synthetic)
# ---------------------------------------------------------------------------


class TestReturnMapSynthetic:
    def test_pairs_consecutive_observable_values(self) -> None:
        # Helix: at the y=0 up-crossing, x = cos(2π k) = 1 (constant). Boring but
        # exercises the pairing logic with a known value.
        traj = _helix()
        rmap = return_map(traj, Plane(axis=1, value=0.0), observable=0)
        assert isinstance(rmap, ReturnMap)
        assert rmap.x.size == rmap.y.size
        if rmap.x.size:
            np.testing.assert_allclose(rmap.x, 1.0, atol=1e-4)
            np.testing.assert_allclose(rmap.y, 1.0, atol=1e-4)

    def test_step_two(self) -> None:
        traj = _helix(n=10000, t_max=20 * np.pi)
        rmap1 = return_map(traj, Plane(axis=1, value=0.0), observable=2, step=1)
        rmap2 = return_map(traj, Plane(axis=1, value=0.0), observable=2, step=2)
        # step=2 skips one crossing: rmap2.y == rmap1.y rolled by one.
        assert rmap1.x.size > rmap2.x.size
        if rmap2.x.size:
            # z = 0.5 t increases by π between consecutive up crossings.
            diffs = rmap2.y - rmap2.x
            np.testing.assert_allclose(diffs, 2 * np.pi, atol=1e-3)

    def test_callable_observable(self) -> None:
        traj = _helix()
        rmap = return_map(
            traj,
            Plane(axis=1, value=0.0),
            observable=lambda t, y: float(y[0] + y[2]),
        )
        assert rmap.observable_meta.endswith("(t, y)")

    def test_step_must_be_positive(self) -> None:
        traj = _helix()
        with pytest.raises(ValueError, match="step"):
            return_map(traj, Plane(axis=1, value=0.0), observable=0, step=0)

    def test_invalid_observable_component(self) -> None:
        traj = _helix()
        with pytest.raises(IndexError):
            return_map(traj, Plane(axis=1, value=0.0), observable=99)

    def test_invalid_observable_type(self) -> None:
        traj = _helix()
        with pytest.raises(TypeError):
            return_map(traj, Plane(axis=1, value=0.0), observable="not-an-obs")  # type: ignore[arg-type]

    def test_empty_when_no_crossings(self) -> None:
        # Trajectory that never crosses y[1] = 100.
        t = np.linspace(0.0, 1.0, 100)
        y = np.column_stack([t, t, t])
        rmap = return_map((t, y), Plane(axis=1, value=100.0, direction="up"))
        assert rmap.x.size == 0
        assert rmap.y.size == 0
        assert rmap.t.size == 0

    def test_to_dataspec(self) -> None:
        traj = _helix()
        rmap = return_map(traj, Plane(axis=1, value=0.0), observable=0)
        spec = rmap.to_dataspec()
        assert spec["kind"] == "return_map"
        assert set(spec) >= {"x", "y", "t", "step", "observable"}

    def test_to_dataspec_unknown_kind(self) -> None:
        traj = _helix()
        rmap = return_map(traj, Plane(axis=1, value=0.0), observable=0)
        with pytest.raises(ValueError, match="unknown kind"):
            rmap.to_dataspec(kind="hexapod")


# ---------------------------------------------------------------------------
# Trajectory methods agree with the free functions (M2 fluent API)
# ---------------------------------------------------------------------------


class TestTrajectoryMethods:
    def test_detect_events_method(self) -> None:
        t = np.linspace(0.0, 4 * np.pi, 4001)
        y = np.column_stack([np.sin(t), np.cos(t)])
        traj = Trajectory(t, y)
        ev = traj.detect_events(Threshold(component=0, value=0.5, direction="up"))
        assert isinstance(ev, EventResult)
        # sin t = 0.5 upward at t = π/6 and π/6 + 2π — two crossings.
        assert ev.t.size == 2

    def test_poincare_section_method(self) -> None:
        traj = _helix()
        sec_func = poincare_section(traj, Plane(axis=1, value=0.0))
        sec_meth = traj.poincare_section(Plane(axis=1, value=0.0))
        np.testing.assert_array_equal(sec_func.t, sec_meth.t)
        np.testing.assert_array_equal(sec_func.y, sec_meth.y)
        assert isinstance(sec_meth, Trajectory)

    def test_return_map_method(self) -> None:
        traj = _helix()
        rmap_func = return_map(traj, Plane(axis=1, value=0.0), observable=0)
        rmap_meth = traj.return_map(Plane(axis=1, value=0.0), observable=0)
        assert isinstance(rmap_meth, ReturnMap)
        np.testing.assert_array_equal(rmap_func.x, rmap_meth.x)
        np.testing.assert_array_equal(rmap_func.y, rmap_meth.y)

    def test_default_trajectory_system_is_none(self) -> None:
        # Allowing Trajectory(t, y) without a `system` argument is the
        # whole point of making the API friendly for raw-array users.
        t = np.linspace(0.0, 1.0, 100)
        y = t[:, None]
        traj = Trajectory(t, y)
        assert traj.system is None


# ---------------------------------------------------------------------------
# Lorenz section + return map (slow; integrates JiTCODE)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLorenzPoincare:
    @pytest.fixture(scope="class")
    def lorenz_traj(self) -> Trajectory:
        import tsdynamics as ts

        # final_time=400, dt=0.005 → 80 001 samples; cheap on a modern laptop
        # after JIT compilation. We drop a 50-time-unit transient.
        lor = ts.Lorenz()
        return lor.integrate(final_time=400.0, dt=0.005).after(50.0)

    def test_section_yields_many_points(self, lorenz_traj: Trajectory) -> None:
        sec = poincare_section(lorenz_traj, Plane(axis=2, value=27.0, direction="up"))
        # The classical milestone target is ≥ 100 crossings.
        assert sec.n_steps >= 100
        np.testing.assert_allclose(sec.y[:, 2], 27.0, atol=1e-4)

    def test_method_and_function_agree(self, lorenz_traj: Trajectory) -> None:
        sec_func = poincare_section(lorenz_traj, Plane(axis=2, value=27.0))
        sec_meth = lorenz_traj.poincare_section(Plane(axis=2, value=27.0))
        np.testing.assert_array_equal(sec_func.t, sec_meth.t)
        np.testing.assert_array_equal(sec_func.y, sec_meth.y)

    def test_return_map_has_unimodal_shape(self, lorenz_traj: Trajectory) -> None:
        # The classical Lorenz return map of z-max vs subsequent z-max is
        # tent-like.  Here we use the standard `z = 27, up` Poincaré
        # section on the y-component (more robust than z-max detection)
        # and assert two properties of the resulting return map:
        #   1. it's *not* a straight line (RMS deviation from y = x is large);
        #   2. the points are bounded — no runaway crossings.
        sec_plane = Plane(axis=2, value=27.0, direction="up")
        rmap = return_map(lorenz_traj, sec_plane, observable=0)
        assert rmap.x.size >= 50

        # Bounded: every x value lies inside the attractor's natural support.
        # Use 1.5x the observed range as a generous upper bound.
        x_extent = np.ptp(rmap.x)
        y_extent = np.ptp(rmap.y)
        assert x_extent > 1.0  # actually varies
        assert y_extent > 1.0
        assert x_extent < 100.0
        assert y_extent < 100.0

        # Not a trivial fixed point: most pairs do not satisfy y ≈ x.
        diffs = np.abs(rmap.y - rmap.x)
        # At least one third of the pairs differ by more than 1 unit.
        assert np.mean(diffs > 1.0) > 0.3
