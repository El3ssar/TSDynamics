"""
Tests for Trajectory enrichment (M1).

The Trajectory methods are thin wrappers over ``tsdynamics.analysis``, so
every property here implicitly covers both the method and the underlying
algorithm.  A few module-level fixtures stand in for "trajectories" without
spinning up an actual integrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis import trajectory_ops as ops
from tsdynamics.base import Trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ramp_traj() -> Trajectory:
    """A 3-D linear ramp: y_i(t) = (i + 1) * t."""
    t = np.linspace(0.0, 10.0, 1001)
    y = np.column_stack([t, 2.0 * t, 3.0 * t])
    return Trajectory(t, y, system=None)


_SIN_OMEGA = 2.0


@pytest.fixture
def sin_traj() -> Trajectory:
    """A single-component sin with known angular frequency ``_SIN_OMEGA``."""
    t = np.linspace(0.0, 10.0, 10_001)
    y = np.sin(_SIN_OMEGA * t)[:, None]
    return Trajectory(t, y, system=None)


# ---------------------------------------------------------------------------
# decimate
# ---------------------------------------------------------------------------


class TestDecimate:
    def test_strides_correctly(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.decimate(every=10)
        assert out.n_steps == (ramp_traj.n_steps + 9) // 10
        np.testing.assert_allclose(out.t, ramp_traj.t[::10])
        np.testing.assert_allclose(out.y, ramp_traj.y[::10])

    def test_every_one_is_full_copy(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.decimate(every=1)
        assert out.n_steps == ramp_traj.n_steps
        assert out.y is not ramp_traj.y  # new array

    def test_rejects_zero_or_negative(self, ramp_traj: Trajectory) -> None:
        for bad in (0, -1, -10):
            with pytest.raises(ValueError):
                ramp_traj.decimate(every=bad)

    def test_rejects_non_integer(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(ValueError):
            ramp_traj.decimate(every=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_cubic_recovers_smooth_signal(self) -> None:
        t = np.linspace(0.0, 1.0, 101)
        y = np.sin(2 * np.pi * t)[:, None]
        traj = Trajectory(t, y, system=None)
        out = traj.resample(dt_new=0.005, kind="cubic")
        expected = np.sin(2 * np.pi * out.t)[:, None]
        np.testing.assert_allclose(out.y, expected, atol=1e-5)

    def test_linear_kind(self) -> None:
        t = np.linspace(0.0, 1.0, 11)
        y = (2.0 * t + 3.0)[:, None]
        traj = Trajectory(t, y, system=None)
        out = traj.resample(dt_new=0.02, kind="linear")
        np.testing.assert_allclose(out.y[:, 0], 2.0 * out.t + 3.0, atol=1e-12)

    def test_uniform_grid(self) -> None:
        t = np.linspace(0.0, 1.0, 11)
        y = t[:, None]
        traj = Trajectory(t, y, system=None)
        out = traj.resample(dt_new=0.05)
        assert np.allclose(np.diff(out.t), 0.05)

    def test_rejects_non_monotonic(self) -> None:
        t = np.array([0.0, 1.0, 0.5, 2.0])
        y = t[:, None]
        traj = Trajectory(t, y, system=None)
        with pytest.raises(ValueError):
            traj.resample(dt_new=0.1)

    def test_rejects_non_positive_dt(self, ramp_traj: Trajectory) -> None:
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                ramp_traj.resample(dt_new=bad)

    def test_rejects_unknown_kind(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(ValueError):
            ramp_traj.resample(dt_new=0.1, kind="quintic")


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


class TestProject:
    def test_picks_components(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.project(dims=(0, 2))
        assert out.dim == 2
        np.testing.assert_allclose(out.y[:, 0], ramp_traj.y[:, 0])
        np.testing.assert_allclose(out.y[:, 1], ramp_traj.y[:, 2])

    def test_preserves_time(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.project(dims=(1,))
        np.testing.assert_allclose(out.t, ramp_traj.t)

    def test_out_of_range_dim_raises(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(IndexError):
            ramp_traj.project(dims=(0, 5))

    def test_empty_dims_raises(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(ValueError):
            ramp_traj.project(dims=())


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------


class TestWindow:
    def test_inclusive_bounds(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.window(t0=2.0, t1=5.0)
        assert out.t[0] >= 2.0 and out.t[-1] <= 5.0
        assert 2.0 in out.t and 5.0 in out.t

    def test_open_low(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.window(t1=3.0)
        np.testing.assert_allclose(out.t[0], 0.0)
        assert out.t[-1] <= 3.0

    def test_open_high(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.window(t0=7.0)
        assert out.t[0] >= 7.0
        np.testing.assert_allclose(out.t[-1], 10.0)

    def test_clips_outside_range(self, ramp_traj: Trajectory) -> None:
        """t0 < t.min and t1 > t.max should round to full trajectory."""
        out = ramp_traj.window(t0=-100.0, t1=1000.0)
        assert out.n_steps == ramp_traj.n_steps

    def test_inverted_bounds_raise(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(ValueError):
            ramp_traj.window(t0=5.0, t1=2.0)


# ---------------------------------------------------------------------------
# derivative
# ---------------------------------------------------------------------------


class TestDerivative:
    def test_first_order_against_analytic(self, sin_traj: Trajectory) -> None:
        d = sin_traj.derivative(order=1)
        expected = _SIN_OMEGA * np.cos(_SIN_OMEGA * sin_traj.t)
        # Edges are first-order, so compare the interior.
        np.testing.assert_allclose(d.y[10:-10, 0], expected[10:-10], atol=1e-4)

    def test_second_order_against_analytic(self, sin_traj: Trajectory) -> None:
        d2 = sin_traj.derivative(order=2)
        expected = -(_SIN_OMEGA**2) * np.sin(_SIN_OMEGA * sin_traj.t)
        # Recursive np.gradient is noisier at the edges; only check the interior.
        np.testing.assert_allclose(d2.y[20:-20, 0], expected[20:-20], atol=5e-3)

    def test_preserves_time_grid(self, sin_traj: Trajectory) -> None:
        d = sin_traj.derivative(order=1)
        np.testing.assert_allclose(d.t, sin_traj.t)
        assert d.t is not sin_traj.t  # but is a copy

    def test_rejects_zero_or_negative_order(self, sin_traj: Trajectory) -> None:
        for bad in (0, -1):
            with pytest.raises(ValueError):
                sin_traj.derivative(order=bad)


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


class TestNorm:
    def test_returns_ndarray(self, ramp_traj: Trajectory) -> None:
        r = ramp_traj.norm()
        assert isinstance(r, np.ndarray)
        assert r.shape == (ramp_traj.n_steps,)

    def test_values_match(self, ramp_traj: Trajectory) -> None:
        r = ramp_traj.norm()
        # y = [t, 2t, 3t] → ||y|| = sqrt(14) * |t|
        np.testing.assert_allclose(r, np.sqrt(14) * ramp_traj.t)


# ---------------------------------------------------------------------------
# local_maxima / local_minima / return_times
# ---------------------------------------------------------------------------


class TestPeaks:
    def test_local_maxima_on_sin(self, sin_traj: Trajectory) -> None:
        tp, yp = sin_traj.local_maxima(component=0)
        # sin has maxima at t = (π/2 + 2kπ) / ω; over [0, 10] with ω=2 that's k=0..2.
        assert tp.size == 3
        np.testing.assert_allclose(yp, 1.0, atol=1e-3)

    def test_local_minima_on_sin(self, sin_traj: Trajectory) -> None:
        tm, ym = sin_traj.local_minima(component=0)
        assert tm.size == 3
        np.testing.assert_allclose(ym, -1.0, atol=1e-3)

    def test_return_times_reproduce_period(self, sin_traj: Trajectory) -> None:
        isi = sin_traj.return_times(component=0)
        assert isi.size >= 1
        np.testing.assert_allclose(isi, 2 * np.pi / _SIN_OMEGA, rtol=1e-3)

    def test_return_times_empty_when_no_peaks(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        y = t[:, None]  # monotonic, no peaks at all
        traj = Trajectory(t, y, system=None)
        isi = traj.return_times(component=0)
        assert isi.size == 0

    def test_find_peaks_kwargs_forwarded(self, sin_traj: Trajectory) -> None:
        # A unit-amplitude sin has prominence ≈ 2 between consecutive maxima;
        # ask for more and find_peaks returns nothing — proving the kwarg
        # made it through.
        tp_loose, _ = sin_traj.local_maxima(component=0)
        tp_strict, _ = sin_traj.local_maxima(component=0, prominence=2.5)
        assert tp_strict.size <= tp_loose.size
        assert tp_strict.size == 0

    def test_invalid_component_raises(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(IndexError):
            ramp_traj.local_maxima(component=5)


# ---------------------------------------------------------------------------
# to_dataspec (placeholder dict shape — V1 will replace it)
# ---------------------------------------------------------------------------


class TestToDataspec:
    def test_timeseries_default_dims(self, ramp_traj: Trajectory) -> None:
        spec = ramp_traj.to_dataspec(kind="timeseries")
        assert spec["kind"] == "timeseries"
        assert spec["dims"] == (0, 1, 2)
        assert spec["t"] is ramp_traj.t
        assert spec["y"] is ramp_traj.y

    def test_phase_portrait_3d(self, ramp_traj: Trajectory) -> None:
        spec = ramp_traj.to_dataspec(kind="phase_portrait_3d", dims=(0, 1, 2))
        assert spec["kind"] == "phase_portrait_3d"
        assert spec["dims"] == (0, 1, 2)

    def test_phase_portrait_requires_dims(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(ValueError):
            ramp_traj.to_dataspec(kind="phase_portrait_2d")

    def test_wrong_dim_cardinality_raises(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(ValueError):
            ramp_traj.to_dataspec(kind="phase_portrait_2d", dims=(0, 1, 2))

    def test_unknown_kind_raises(self, ramp_traj: Trajectory) -> None:
        with pytest.raises(ValueError):
            ramp_traj.to_dataspec(kind="hexapod")


# ---------------------------------------------------------------------------
# Immutability: every transformation returns a new instance
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_decimate_does_not_share_buffers(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.decimate(every=2)
        assert out.y is not ramp_traj.y
        assert out.t is not ramp_traj.t

    def test_resample_does_not_share_buffers(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.resample(dt_new=0.5)
        assert out.y is not ramp_traj.y
        assert out.t is not ramp_traj.t

    def test_project_does_not_share_buffers(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.project(dims=(0,))
        assert out.y is not ramp_traj.y
        assert out.t is not ramp_traj.t

    def test_window_does_not_share_buffers(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.window(t0=1.0, t1=5.0)
        assert out.y is not ramp_traj.y
        assert out.t is not ramp_traj.t

    def test_derivative_does_not_share_buffers(self, ramp_traj: Trajectory) -> None:
        out = ramp_traj.derivative(order=1)
        assert out.y is not ramp_traj.y
        assert out.t is not ramp_traj.t

    def test_chain_does_not_mutate_source(self, sin_traj: Trajectory) -> None:
        original_t = sin_traj.t.copy()
        original_y = sin_traj.y.copy()
        sin_traj.decimate(every=2).window(t0=1.0, t1=9.0).derivative(order=1)
        np.testing.assert_array_equal(sin_traj.t, original_t)
        np.testing.assert_array_equal(sin_traj.y, original_y)


# ---------------------------------------------------------------------------
# The functional API in tsdynamics.analysis is the same surface
# ---------------------------------------------------------------------------


class TestAnalysisModule:
    def test_functional_and_method_agree(self, ramp_traj: Trajectory) -> None:
        tn1, yn1 = ops.decimate(ramp_traj.t, ramp_traj.y, every=5)
        out2 = ramp_traj.decimate(every=5)
        np.testing.assert_array_equal(tn1, out2.t)
        np.testing.assert_array_equal(yn1, out2.y)

    def test_submodule_reexported_at_top_level(self) -> None:
        import tsdynamics as ts

        assert hasattr(ts, "analysis")
        assert hasattr(ts.analysis, "decimate")
        assert "analysis" in ts.__all__
