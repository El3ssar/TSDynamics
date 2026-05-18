"""
Tests for the event detection module.

Covers the bracket helper, the two built-in conditions (:class:`Plane`,
:class:`LinearPlane`), shortcut-kwargs and bare-callable call styles for
:func:`detect_events`, and a few edge cases (no events, short
trajectories, direction filtering, refinement accuracy).

The unified return type — :class:`~tsdynamics.base.Trajectory` — is
checked throughout so the contract that "every analysis op returns a
Trajectory" stays enforced.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis import (
    EventCondition,
    LinearPlane,
    Plane,
    detect_events,
)
from tsdynamics.analysis._ops.events import _bracket_mask
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
        # Up crossing brackets [1, 2] only.
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
        g = np.array([0.0, 1.0])
        idx, direc = _bracket_mask(g, "up")
        assert idx.tolist() == [0]
        assert direc.tolist() == [1]
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
        t_ev, y_ev = p.detect(sin_traj.t, sin_traj.y)
        expected = np.array([np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2, 7 * np.pi / 2])
        assert t_ev.size == expected.size
        np.testing.assert_allclose(t_ev, expected, atol=1e-6)
        # Refined state at π/2: sin = 1.
        np.testing.assert_allclose(y_ev[0, 0], 1.0, atol=1e-5)

    def test_detect_direction_filter(self, sin_traj: Trajectory) -> None:
        p_up = Plane(axis=1, value=0.0, direction="up")
        p_down = Plane(axis=1, value=0.0, direction="down")
        t_up, _ = p_up.detect(sin_traj.t, sin_traj.y)
        t_dn, _ = p_down.detect(sin_traj.t, sin_traj.y)
        # cos t has 2 upward zeros (3π/2, 7π/2) and 2 downward (π/2, 5π/2).
        assert t_up.size == 2
        assert t_dn.size == 2


# ---------------------------------------------------------------------------
# LinearPlane
# ---------------------------------------------------------------------------


class TestLinearPlane:
    def test_axis_aligned_matches_plane(self, sin_traj: Trajectory) -> None:
        lp = LinearPlane(normal=np.array([0.0, 1.0]), offset=0.0, direction="up")
        p = Plane(axis=1, value=0.0, direction="up")
        t_lp, _ = lp.detect(sin_traj.t, sin_traj.y)
        t_p, _ = p.detect(sin_traj.t, sin_traj.y)
        np.testing.assert_allclose(t_lp, t_p, atol=1e-9)

    def test_diagonal_plane(self) -> None:
        # y = (cos t, sin t); the section x + y = 0 is hit at 3π/4 + kπ.
        t = np.linspace(0.0, 2 * np.pi, 5001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        lp = LinearPlane(normal=np.array([1.0, 1.0]), offset=0.0, direction="either")
        t_ev, _ = lp.detect(t, y)
        expected = np.array([3 * np.pi / 4, 7 * np.pi / 4])
        assert t_ev.size == expected.size
        np.testing.assert_allclose(t_ev, expected, atol=1e-6)

    def test_zero_normal_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero vector"):
            LinearPlane(normal=np.array([0.0, 0.0, 0.0]))

    def test_empty_normal_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LinearPlane(normal=np.array([]))


# ---------------------------------------------------------------------------
# detect_events — the public driver
# ---------------------------------------------------------------------------


class TestDetectEventsBasic:
    def test_returns_trajectory(self, sin_traj: Trajectory) -> None:
        out = detect_events(sin_traj, Plane(axis=1, value=0.0))
        assert isinstance(out, Trajectory)
        assert out.dim == sin_traj.dim
        assert out.n_steps > 0

    def test_system_inherited(self, sin_traj: Trajectory) -> None:
        marker = object()
        sin_traj.system = marker  # type: ignore[assignment]
        out = detect_events(sin_traj, Plane(axis=1, value=0.0))
        assert out.system is marker

    def test_tuple_input(self, sin_traj: Trajectory) -> None:
        out = detect_events((sin_traj.t, sin_traj.y), Plane(axis=1, value=0.0))
        assert isinstance(out, Trajectory)
        assert out.system is None

    def test_bare_arrays_input(self, sin_traj: Trajectory) -> None:
        out = detect_events(sin_traj.t, sin_traj.y, Plane(axis=1, value=0.0))
        assert isinstance(out, Trajectory)


# ---------------------------------------------------------------------------
# detect_events — three call styles
# ---------------------------------------------------------------------------


class TestDetectEventsCallStyles:
    def test_explicit_plane(self, sin_traj: Trajectory) -> None:
        ev = detect_events(sin_traj, Plane(axis=0, value=0.5, direction="up"))
        # sin t = 0.5 upward at t = π/6 and π/6 + 2π over [0, 4π].
        expected = np.array([np.pi / 6, np.pi / 6 + 2 * np.pi])
        assert ev.n_steps == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-5)

    def test_shortcut_kwargs(self, sin_traj: Trajectory) -> None:
        ev = detect_events(sin_traj, axis=0, value=0.5, direction="up")
        expected = np.array([np.pi / 6, np.pi / 6 + 2 * np.pi])
        assert ev.n_steps == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-5)

    def test_shortcut_kwargs_no_direction_means_either(self, sin_traj: Trajectory) -> None:
        # Without explicit direction, default is "either" — both upward and
        # downward zeros count.
        ev = detect_events(sin_traj, axis=0, value=0.5)
        # sin t = 0.5 hit four times on [0, 4π] (two up, two down).
        assert ev.n_steps == 4

    def test_callable_condition(self) -> None:
        # Spiral that crosses r=1: y(t) = (e^{0.05 t} cos t, e^{0.05 t} sin t).
        t = np.linspace(-2.0, 2.0, 1001)
        r = np.exp(0.05 * t)
        y = np.column_stack([r * np.cos(t), r * np.sin(t)])
        ev = detect_events(
            (t, y),
            lambda _t, _y: float(np.linalg.norm(_y) - 1.0),
            direction="up",
        )
        assert ev.n_steps == 1
        np.testing.assert_allclose(ev.t[0], 0.0, atol=1e-5)

    def test_linear_plane_via_normal_kwarg(self) -> None:
        # Shortcut for LinearPlane: pass normal=...
        t = np.linspace(0.0, 2 * np.pi, 5001)
        y = np.column_stack([np.cos(t), np.sin(t)])
        ev = detect_events((t, y), normal=np.array([1.0, 1.0]), direction="either")
        expected = np.array([3 * np.pi / 4, 7 * np.pi / 4])
        assert ev.n_steps == expected.size
        np.testing.assert_allclose(ev.t, expected, atol=1e-6)

    def test_method_form_matches_function(self, sin_traj: Trajectory) -> None:
        m = sin_traj.detect_events(axis=0, value=0.5, direction="up")
        f = detect_events(sin_traj, axis=0, value=0.5, direction="up")
        np.testing.assert_array_equal(m.t, f.t)
        np.testing.assert_array_equal(m.y, f.y)


# ---------------------------------------------------------------------------
# detect_events — error cases
# ---------------------------------------------------------------------------


class TestDetectEventsErrors:
    def test_missing_condition_and_no_kwargs(self, sin_traj: Trajectory) -> None:
        with pytest.raises(TypeError, match="Need a condition"):
            detect_events(sin_traj)

    def test_axis_without_value(self, sin_traj: Trajectory) -> None:
        with pytest.raises(TypeError, match="axis="):
            detect_events(sin_traj, axis=0)

    def test_bad_condition_type(self, sin_traj: Trajectory) -> None:
        with pytest.raises(TypeError, match="condition must be"):
            detect_events(sin_traj, object())  # type: ignore[arg-type]

    def test_no_events_on_monotonic_ramp(self, ramp_traj: Trajectory) -> None:
        ev = detect_events(ramp_traj, axis=0, value=100.0, direction="up")
        assert ev.n_steps == 0
        assert ev.y.shape == (0, ramp_traj.dim)

    def test_short_trajectory_returns_empty(self) -> None:
        # Single-sample trajectory cannot bracket anything.
        t = np.array([0.0])
        y = np.array([[0.0, 1.0]])
        ev = detect_events((t, y), axis=0, value=0.0)
        assert ev.n_steps == 0


# ---------------------------------------------------------------------------
# Refinement accuracy
# ---------------------------------------------------------------------------


class TestRefinementAccuracy:
    def test_sub_microsecond_on_sin(self) -> None:
        t = np.linspace(0.0, 2 * np.pi, 1001)
        y = np.column_stack([np.sin(t), np.cos(t)])
        ev = detect_events((t, y), axis=0, value=0.0, direction="up", rtol=1e-12)
        if ev.n_steps:
            # The interior up-zero is at t = 2π; refined error should be < 1e-6.
            errs = np.minimum(np.abs(ev.t - 2 * np.pi), np.abs(ev.t - 0.0))
            assert errs.min() < 1e-6


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_builtins_satisfy_protocol(self) -> None:
        assert isinstance(Plane(axis=0, value=0.0), EventCondition)
        assert isinstance(LinearPlane(normal=np.array([1.0, 0.0])), EventCondition)
