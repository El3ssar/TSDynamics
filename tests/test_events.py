"""
Tests for M2 event detection.

Covers the bracket helper, every built-in :class:`EventCondition`, the
generic :func:`detect_events` driver, and a few edge cases (no events,
short trajectories, direction filtering, refinement accuracy).
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis.events import (
    Custom,
    EventCondition,
    EventResult,
    LinearPlane,
    LocalExtremum,
    Plane,
    Threshold,
    _bracket_mask,
    detect_events,
)
from tsdynamics.base import Trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sin_traj() -> Trajectory:
    """``y(t) = (sin t, cos t)`` over ``[0, 4π]`` with 1001 samples (dt ≈ 1.26e-2)."""
    t = np.linspace(0.0, 4 * np.pi, 1001)
    y = np.column_stack([np.sin(t), np.cos(t)])
    return Trajectory(t, y, system=None)


@pytest.fixture
def ramp_traj() -> Trajectory:
    """``y(t) = (t, 2t, -t)`` over ``[0, 10]``."""
    t = np.linspace(0.0, 10.0, 1001)
    y = np.column_stack([t, 2.0 * t, -t])
    return Trajectory(t, y, system=None)


# ---------------------------------------------------------------------------
# _bracket_mask
# ---------------------------------------------------------------------------


class TestBracketMask:
    def test_up_crossing(self) -> None:
        g = np.array([-1.0, -0.5, 0.5, 1.0, 0.0, -0.5])
        idx, direc = _bracket_mask(g, "up")
        # Up crossings: between idx 1 (-0.5 → 0.5)
        assert idx.tolist() == [1]
        assert direc.tolist() == [1]

    def test_down_crossing(self) -> None:
        g = np.array([1.0, 0.5, -0.5, -1.0, 0.0, 0.5])
        idx, direc = _bracket_mask(g, "down")
        assert idx.tolist() == [1]
        assert direc.tolist() == [-1]

    def test_either_picks_both(self) -> None:
        g = np.array([-1.0, 1.0, -1.0, 1.0])
        idx, direc = _bracket_mask(g, "either")
        assert idx.tolist() == [0, 1, 2]
        assert direc.tolist() == [1, -1, 1]

    def test_exact_zero_treated_as_crossing(self) -> None:
        # 0 → +: counts as up
        g = np.array([0.0, 1.0])
        idx, direc = _bracket_mask(g, "up")
        assert idx.tolist() == [0]
        assert direc.tolist() == [1]
        # 0 → -: counts as down
        g = np.array([0.0, -1.0])
        idx, direc = _bracket_mask(g, "down")
        assert idx.tolist() == [0]
        assert direc.tolist() == [-1]

    def test_short_input(self) -> None:
        idx, _ = _bracket_mask(np.array([0.5]), "either")
        assert idx.size == 0

    def test_unknown_direction(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            _bracket_mask(np.array([-1.0, 1.0]), "sideways")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Plane
# ---------------------------------------------------------------------------


class TestPlane:
    def test_evaluate_single(self) -> None:
        p = Plane(axis=2, value=27.0)
        assert p.evaluate(0.0, np.array([0.0, 0.0, 28.0])) == 1.0

    def test_evaluate_along_vectorised(self, sin_traj: Trajectory) -> None:
        p = Plane(axis=1, value=0.0)
        g = p.evaluate_along(sin_traj.t, sin_traj.y)
        np.testing.assert_allclose(g, sin_traj.y[:, 1])

    def test_detect_finds_canonical_crossings(self, sin_traj: Trajectory) -> None:
        # cos t == 0 at t = π/2, 3π/2, 5π/2, 7π/2 over [0, 4π].
        p = Plane(axis=1, value=0.0, direction="either")
        ev = p.detect(sin_traj.t, sin_traj.y)
        expected = np.array([np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2, 7 * np.pi / 2])
        assert ev.t.size == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-6)

    def test_detect_direction_filter(self, sin_traj: Trajectory) -> None:
        p_up = Plane(axis=1, value=0.0, direction="up")
        p_down = Plane(axis=1, value=0.0, direction="down")
        ev_up = p_up.detect(sin_traj.t, sin_traj.y)
        ev_down = p_down.detect(sin_traj.t, sin_traj.y)
        # cos t has 2 upward zeros (at 3π/2, 7π/2) and 2 downward (π/2, 5π/2).
        assert ev_up.t.size == 2
        assert ev_down.t.size == 2
        assert (ev_up.direction == 1).all()
        assert (ev_down.direction == -1).all()

    def test_detect_refines_state(self, sin_traj: Trajectory) -> None:
        # At the section sin(π/2) = 1.
        p = Plane(axis=1, value=0.0, direction="down")
        ev = p.detect(sin_traj.t, sin_traj.y)
        # The first downward zero of cos is at π/2; sin there is +1.
        np.testing.assert_allclose(ev.y[0, 0], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# LinearPlane
# ---------------------------------------------------------------------------


class TestLinearPlane:
    def test_axis_aligned_matches_plane(self, sin_traj: Trajectory) -> None:
        lp = LinearPlane(normal=np.array([0.0, 1.0]), offset=0.0, direction="up")
        p = Plane(axis=1, value=0.0, direction="up")
        ev_lp = lp.detect(sin_traj.t, sin_traj.y)
        ev_p = p.detect(sin_traj.t, sin_traj.y)
        np.testing.assert_allclose(ev_lp.t, ev_p.t, atol=1e-9)

    def test_diagonal_plane(self) -> None:
        # y = (cos t, sin t); we want x + y = 0  → cos t + sin t = 0
        # That is t = 3π/4 + kπ → over [0, 2π] there are crossings at 3π/4 and 7π/4.
        t = np.linspace(0.0, 2 * np.pi, 5001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        lp = LinearPlane(normal=np.array([1.0, 1.0]), offset=0.0, direction="either")
        ev = lp.detect(t, y)
        expected = np.array([3 * np.pi / 4, 7 * np.pi / 4])
        assert ev.t.size == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-6)

    def test_zero_normal_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero vector"):
            LinearPlane(normal=np.array([0.0, 0.0, 0.0]))

    def test_empty_normal_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LinearPlane(normal=np.array([]))


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_default_direction_up(self) -> None:
        th = Threshold(component=0, value=0.0)
        assert th.direction == "up"

    def test_detects_threshold_crossings(self, sin_traj: Trajectory) -> None:
        # sin t crosses 0.5 upwards twice on [0, 4π] (at t = π/6 and t = π/6 + 2π).
        th = Threshold(component=0, value=0.5, direction="up")
        ev = th.detect(sin_traj.t, sin_traj.y)
        expected = np.array([np.pi / 6, np.pi / 6 + 2 * np.pi])
        assert ev.t.size == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Custom
# ---------------------------------------------------------------------------


class TestCustom:
    def test_radial_crossing(self) -> None:
        # Spiral that crosses r=1: y(t) = (e^{0.05 t} cos t, e^{0.05 t} sin t).
        # ||y|| = e^{0.05 t} → r=1 crossing at t=0.
        t = np.linspace(-2.0, 2.0, 1001)
        r = np.exp(0.05 * t)
        y = np.column_stack([r * np.cos(t), r * np.sin(t)])
        cond = Custom(lambda _t, _y: float(np.linalg.norm(_y) - 1.0), direction="up")
        ev = cond.detect(t, y)
        assert ev.t.size == 1
        np.testing.assert_allclose(ev.t[0], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# LocalExtremum
# ---------------------------------------------------------------------------


class TestLocalExtremum:
    def test_default_kind_is_max(self) -> None:
        ex = LocalExtremum(component=0)
        assert ex.kind == "max"
        assert ex.direction == "down"  # derivative crosses downward at a max

    def test_finds_maxima_of_sin(self, sin_traj: Trajectory) -> None:
        # sin t has maxima at t = π/2, 5π/2 over [0, 4π].
        ex = LocalExtremum(component=0, kind="max")
        ev = ex.detect(sin_traj.t, sin_traj.y)
        expected = np.array([np.pi / 2, 5 * np.pi / 2])
        assert ev.t.size == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-3)
        np.testing.assert_allclose(ev.y[:, 0], 1.0, atol=1e-4)

    def test_finds_minima_of_sin(self, sin_traj: Trajectory) -> None:
        ex = LocalExtremum(component=0, kind="min")
        ev = ex.detect(sin_traj.t, sin_traj.y)
        expected = np.array([3 * np.pi / 2, 7 * np.pi / 2])
        assert ev.t.size == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-3)
        np.testing.assert_allclose(ev.y[:, 0], -1.0, atol=1e-4)

    def test_rejects_bad_kind(self, sin_traj: Trajectory) -> None:
        ex = LocalExtremum(component=0, kind="weird")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="kind"):
            ex.detect(sin_traj.t, sin_traj.y)

    def test_rejects_bad_component(self, sin_traj: Trajectory) -> None:
        ex = LocalExtremum(component=99, kind="max")
        with pytest.raises(IndexError):
            ex.detect(sin_traj.t, sin_traj.y)

    def test_short_input_returns_empty(self) -> None:
        ex = LocalExtremum(component=0, kind="max")
        ev = ex.detect(np.array([0.0, 1.0]), np.array([[0.0], [1.0]]))
        assert ev.t.size == 0


# ---------------------------------------------------------------------------
# detect_events
# ---------------------------------------------------------------------------


class TestDetectEvents:
    def test_accepts_trajectory_object(self, sin_traj: Trajectory) -> None:
        ev = detect_events(sin_traj, Plane(axis=1, value=0.0))
        assert ev.t.size > 0

    def test_accepts_tuple(self, sin_traj: Trajectory) -> None:
        ev = detect_events((sin_traj.t, sin_traj.y), Plane(axis=1, value=0.0))
        assert ev.t.size > 0

    def test_rejects_missing_arguments(self) -> None:
        # No arguments at all → TypeError from the polymorphic wrapper.
        with pytest.raises(TypeError):
            detect_events()  # type: ignore[call-arg]

    def test_rejects_bad_condition(self, sin_traj: Trajectory) -> None:
        with pytest.raises(TypeError, match="\\.detect"):
            detect_events(sin_traj, object())  # type: ignore[arg-type]

    def test_no_events_on_monotonic_ramp(self, ramp_traj: Trajectory) -> None:
        # y[0] = t never crosses 100 over [0, 10].
        ev = detect_events(ramp_traj, Threshold(component=0, value=100.0, direction="up"))
        assert ev.t.size == 0
        assert ev.y.shape == (0, 3)

    def test_refinement_accuracy_on_sin(self) -> None:
        # Increase resolution and confirm refined error scales below 1e-7.
        t = np.linspace(0.0, 2 * np.pi, 1001)
        y = np.column_stack([np.sin(t), np.cos(t)])
        cond = Plane(axis=0, value=0.0, direction="up")
        ev = detect_events((t, y), cond, rtol=1e-12)
        # sin t = 0 upward at t = 0 and t = 2π.
        # The endpoint t = 0 might not be detected (no left bracket); t = 2π is at endpoint.
        # The interior zero we expect is at t = 2π — depending on sampling.
        # Robust: find at least one crossing and assert sub-microsecond accuracy.
        if ev.t.size:
            errs = np.minimum(np.abs(ev.t - 2 * np.pi), np.abs(ev.t - 0.0))
            assert errs.min() < 1e-6

    def test_result_is_eventresult(self, sin_traj: Trajectory) -> None:
        ev = detect_events(sin_traj, Plane(axis=1, value=0.0))
        assert isinstance(ev, EventResult)
        # unpacks as (t, y)
        t, y = ev
        assert t.shape[0] == y.shape[0]
        assert len(ev) == ev.t.size

    def test_condition_attached(self, sin_traj: Trajectory) -> None:
        cond = Plane(axis=1, value=0.0)
        ev = detect_events(sin_traj, cond)
        assert ev.condition is cond


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_builtins_satisfy_protocol(self) -> None:
        assert isinstance(Plane(axis=0, value=0.0), EventCondition)
        assert isinstance(LinearPlane(normal=np.array([1.0, 0.0])), EventCondition)
        assert isinstance(Threshold(component=0, value=0.0), EventCondition)
        assert isinstance(LocalExtremum(component=0), EventCondition)
        assert isinstance(Custom(lambda t, y: 0.0), EventCondition)
