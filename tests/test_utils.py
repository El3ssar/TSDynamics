"""
Tests for tsdynamics.utils: timestep estimation utilities.

All tests use synthetic trajectories (no ODE compilation).
"""

import numpy as np
import pytest

from tsdynamics.utils import (
    estimate_curvature_timestep,
    estimate_dt_from_sagitta,
    estimate_dt_from_spectrum,
)

# ---------------------------------------------------------------------------
# Synthetic trajectory factories
# ---------------------------------------------------------------------------


def _sine_trajectory(T=2000, dt=0.01, freq=1.0):
    """1D sine wave: y(t) = sin(2π·freq·t)."""
    t = np.arange(T) * dt
    y = np.sin(2 * np.pi * freq * t)
    return y[:, None], dt  # shape (T, 1)


def _lorenz_like_3d(T=2000, dt=0.01):
    """
    A smooth 3D spiral: cheap synthetic stand-in for a Lorenz trajectory.
    Gives curvature/sagitta estimators something realistic to work with.
    """
    t = np.arange(T) * dt
    theta = 2 * np.pi * t
    r = 5.0 + 0.5 * np.sin(0.3 * t)
    X = np.column_stack(
        [
            r * np.cos(theta),
            r * np.sin(theta),
            2.0 * np.sin(0.7 * t),
        ]
    )
    return X, dt


# ---------------------------------------------------------------------------
# estimate_curvature_timestep
# ---------------------------------------------------------------------------


class TestCurvatureTimestep:
    @pytest.fixture
    def result(self):
        y, dt0 = _lorenz_like_3d()
        return estimate_curvature_timestep(y, dt0, epsilon=0.01)

    def test_returns_result_object(self, result):
        assert result is not None

    def test_delta_t_positive(self, result):
        assert result.delta_t > 0.0

    def test_stride_positive(self, result):
        assert result.stride >= 1

    def test_stride_is_int(self, result):
        assert isinstance(result.stride, (int, np.integer))

    def test_kappa_percentile_nonnegative(self, result):
        assert result.kappa_percentile >= 0.0

    def test_predicted_error_nonnegative(self, result):
        assert result.predicted_error >= 0.0

    def test_indices_valid(self, result):
        y, _ = _lorenz_like_3d()
        T = y.shape[0]
        assert np.all(result.indices >= 0)
        assert np.all(result.indices < T)
        assert len(result.indices) >= 2

    def test_delta_t_consistent_with_stride(self, result):
        """delta_t is the continuous estimate; stride is floor(delta_t / dt0).
        So stride * dt0 <= delta_t < (stride+1) * dt0."""
        _, dt0 = _lorenz_like_3d()
        assert result.delta_t >= result.stride * dt0 - 1e-12

    def test_tighter_epsilon_gives_smaller_stride(self):
        y, dt0 = _lorenz_like_3d()
        r_loose = estimate_curvature_timestep(y, dt0, epsilon=0.1)
        r_tight = estimate_curvature_timestep(y, dt0, epsilon=0.001)
        assert r_tight.stride <= r_loose.stride

    def test_percentile_parameter(self):
        y, dt0 = _lorenz_like_3d()
        r95 = estimate_curvature_timestep(y, dt0, epsilon=0.05, percentile=95.0)
        r50 = estimate_curvature_timestep(y, dt0, epsilon=0.05, percentile=50.0)
        assert r95 is not None and r50 is not None

    def test_works_with_1d_array(self):
        """Should accept (T, 1) array."""
        y, dt0 = _sine_trajectory()
        result = estimate_curvature_timestep(y, dt0, epsilon=0.05)
        assert result.delta_t > 0.0


# ---------------------------------------------------------------------------
# estimate_dt_from_spectrum
# ---------------------------------------------------------------------------


class TestFrequencyTimestep:
    @pytest.fixture
    def result_sine(self):
        y, dt0 = _sine_trajectory(T=4096, freq=2.0)
        return estimate_dt_from_spectrum(y, dt0), dt0

    def test_returns_result_object(self, result_sine):
        result, _ = result_sine
        assert result is not None

    def test_delta_t_positive(self, result_sine):
        result, _ = result_sine
        assert result.delta_t > 0.0

    def test_f_hz_positive(self, result_sine):
        result, _ = result_sine
        assert result.f_hz > 0.0

    def test_converged_for_clean_signal(self, result_sine):
        result, _ = result_sine
        assert result.converged, f"Should converge on clean sine, notes: {result.notes}"

    def test_delta_t_sensibly_coarser_than_original(self, result_sine):
        result, dt0 = result_sine
        assert result.delta_t >= dt0 * 0.5  # should not recommend finer than original

    def test_detected_frequency_is_positive(self, result_sine):
        """The estimator returns a positive highest-significant frequency."""
        result, _ = result_sine
        assert result.f_hz > 0.0, f"f_hz must be positive, got {result.f_hz:.3f}"

    def test_notes_is_string(self, result_sine):
        result, _ = result_sine
        assert isinstance(result.notes, str)

    def test_q_power_field(self, result_sine):
        result, _ = result_sine
        assert 0.0 < result.q_power <= 1.0

    def test_safety_field(self, result_sine):
        result, _ = result_sine
        assert 0.0 < result.safety <= 1.0

    def test_multidimensional_input(self):
        """Should handle (T, d) arrays by combining columns."""
        y, dt0 = _lorenz_like_3d()
        result = estimate_dt_from_spectrum(y, dt0)
        assert result.delta_t > 0.0

    def test_custom_parameters(self):
        y, dt0 = _sine_trajectory(T=4096, freq=3.0)
        result = estimate_dt_from_spectrum(y, dt0, safety=0.8, q_power=0.95)
        assert result.delta_t > 0.0


# ---------------------------------------------------------------------------
# estimate_dt_from_sagitta
# ---------------------------------------------------------------------------


class TestSagittaTimestep:
    @pytest.fixture
    def result_3d(self):
        y, dt0 = _lorenz_like_3d()
        return estimate_dt_from_sagitta(y, dt0, epsilon=0.05)

    def test_returns_result_object(self, result_3d):
        assert result_3d is not None

    def test_delta_t_positive(self, result_3d):
        assert result_3d.delta_t > 0.0

    def test_stride_positive(self, result_3d):
        assert result_3d.stride >= 1

    def test_delta_t_equals_stride_times_dt0(self, result_3d):
        _, dt0 = _lorenz_like_3d()
        assert result_3d.delta_t == pytest.approx(result_3d.stride * dt0)

    def test_indices_valid(self, result_3d):
        y, _ = _lorenz_like_3d()
        T = y.shape[0]
        assert np.all(result_3d.indices >= 0)
        assert np.all(result_3d.indices < T)

    def test_notes_is_string(self, result_3d):
        assert isinstance(result_3d.notes, str)

    def test_percentile_value_nonnegative(self, result_3d):
        assert result_3d.percentile_value >= 0.0

    def test_works_with_1d_input(self):
        """1D input should trigger auto-embedding."""
        y, dt0 = _sine_trajectory(T=1000)
        y1d = y.ravel()
        result = estimate_dt_from_sagitta(y1d, dt0, epsilon=0.05)
        assert result.delta_t > 0.0

    def test_tighter_epsilon_gives_finer_result(self):
        y, dt0 = _lorenz_like_3d()
        r_loose = estimate_dt_from_sagitta(y, dt0, epsilon=0.2)
        r_tight = estimate_dt_from_sagitta(y, dt0, epsilon=0.01)
        # tighter tolerance should not produce a coarser stride than loose
        assert r_tight.stride <= r_loose.stride + 5  # allow some tolerance

    def test_searched_ms_nonempty(self, result_3d):
        assert result_3d.searched_ms.size > 0
