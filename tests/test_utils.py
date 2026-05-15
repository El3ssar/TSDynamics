"""Tests for ``tsdynamics.utils``."""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.utils import SagittaDt, estimate_dt_from_sagitta, staticjit

# ---------------------------------------------------------------------------
# staticjit decorator
# ---------------------------------------------------------------------------


class TestStaticjit:
    def test_callable_returned(self) -> None:
        @staticjit
        def f(x: float, a: float) -> float:
            return a * x

        assert callable(f)

    def test_runs_correctly_when_called_unbound(self) -> None:
        @staticjit
        def f(x: float, a: float) -> float:
            return a * x

        # staticjit returns a staticmethod-wrapped njit; reading it off a
        # class strips the descriptor.
        class Holder:
            g = f

        assert Holder.g(2.0, 3.0) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# estimate_dt_from_sagitta
# ---------------------------------------------------------------------------


def _spiral_3d(T: int = 2000, dt: float = 0.01) -> tuple[np.ndarray, float]:
    """A smooth 3D spiral — analytical Δt is well-defined."""
    t = np.arange(T) * dt
    theta = 2.0 * np.pi * t
    r = 5.0 + 0.5 * np.sin(0.3 * t)
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta), 2.0 * np.sin(0.7 * t)])
    return X, dt


def _sine_1d(T: int = 2000, dt: float = 0.01, freq: float = 1.0) -> tuple[np.ndarray, float]:
    t = np.arange(T) * dt
    return np.sin(2 * np.pi * freq * t), dt


class TestSagitta:
    @pytest.fixture
    def result_3d(self) -> SagittaDt:
        y, dt0 = _spiral_3d()
        return estimate_dt_from_sagitta(y, dt0, epsilon=0.05)

    def test_returns_dataclass(self, result_3d: SagittaDt) -> None:
        assert isinstance(result_3d, SagittaDt)

    def test_delta_t_positive(self, result_3d: SagittaDt) -> None:
        assert result_3d.delta_t > 0.0

    def test_stride_positive_int(self, result_3d: SagittaDt) -> None:
        assert isinstance(result_3d.stride, int)
        assert result_3d.stride >= 1

    def test_delta_t_equals_stride_times_dt0(self, result_3d: SagittaDt) -> None:
        _, dt0 = _spiral_3d()
        assert result_3d.delta_t == pytest.approx(result_3d.stride * dt0)

    def test_indices_within_bounds(self, result_3d: SagittaDt) -> None:
        y, _ = _spiral_3d()
        T = y.shape[0]
        assert result_3d.indices.min() >= 0
        assert result_3d.indices.max() < T

    def test_notes_is_string(self, result_3d: SagittaDt) -> None:
        assert isinstance(result_3d.notes, str)

    def test_percentile_value_nonnegative(self, result_3d: SagittaDt) -> None:
        assert result_3d.percentile_value >= 0.0

    def test_1d_input_triggers_embedding(self) -> None:
        y, dt0 = _sine_1d(T=1000)
        result = estimate_dt_from_sagitta(y, dt0, epsilon=0.05)
        assert result.delta_t > 0.0
        assert "Takens embedding" in result.notes

    def test_tighter_epsilon_does_not_coarsen_stride(self) -> None:
        y, dt0 = _spiral_3d()
        r_loose = estimate_dt_from_sagitta(y, dt0, epsilon=0.2)
        r_tight = estimate_dt_from_sagitta(y, dt0, epsilon=0.01)
        # tighter tolerance ⇒ stride no larger than loose (allow small slack
        # because the search is approximate under relative criterion).
        assert r_tight.stride <= r_loose.stride + 1

    def test_rejects_bad_dt0(self) -> None:
        y, _ = _spiral_3d()
        with pytest.raises(ValueError, match="dt0"):
            estimate_dt_from_sagitta(y, 0.0, epsilon=0.05)

    def test_rejects_bad_epsilon(self) -> None:
        y, dt0 = _spiral_3d()
        with pytest.raises(ValueError, match="epsilon"):
            estimate_dt_from_sagitta(y, dt0, epsilon=-1.0)
