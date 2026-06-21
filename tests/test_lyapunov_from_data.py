"""``lyapunov_from_data`` — maximal Lyapunov exponent from a time series.

Literature targets: Hénon map λ_max ≈ 0.419 and Lorenz λ_max ≈ 0.906, both
recovered from a single scalar coordinate via delay embedding (Kantz 1994;
Rosenstein, Collins & De Luca 1993).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis.lyapunov import LyapunovFromData, lyapunov_from_data
from tsdynamics.analysis.lyapunov.from_data import _delay_embed


@pytest.fixture(scope="module")
def henon_x() -> np.ndarray:
    """A long, transient-free Hénon ``x`` series."""
    return ts.Henon().trajectory(6000, transient=500, ic=[0.1, 0.1]).y[:, 0]


# ---------------------------------------------------------------------------
# Delay embedding helper
# ---------------------------------------------------------------------------


class TestDelayEmbed:
    def test_scalar_shape_and_values(self) -> None:
        x = np.arange(10.0)
        emb = _delay_embed(x, m=3, tau=2)
        assert emb.shape == (10 - 2 * 2, 3)  # rows = n - (m-1)*tau
        # row n is [x[n], x[n+tau], x[n+2 tau]]
        np.testing.assert_array_equal(emb[0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(emb[1], [1.0, 3.0, 5.0])

    def test_multivariate_interleaves_channels(self) -> None:
        x = np.column_stack([np.arange(6.0), 10 + np.arange(6.0)])
        emb = _delay_embed(x, m=2, tau=1)
        assert emb.shape == (5, 4)  # (m * n_channels) columns
        np.testing.assert_array_equal(emb[0], [0.0, 10.0, 1.0, 11.0])

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            _delay_embed(np.arange(5.0), m=4, tau=3)


# ---------------------------------------------------------------------------
# Hénon map — fast tier
# ---------------------------------------------------------------------------


class TestHenon:
    @pytest.mark.parametrize("m", [2, 4])
    def test_kantz_recovers_mlle(self, henon_x: np.ndarray, m: int) -> None:
        res = lyapunov_from_data(
            henon_x, dimension=m, delay=1, k_max=12, method="kantz", fit=(0, 6)
        )
        assert float(res) == pytest.approx(0.419, abs=0.06)

    def test_rosenstein_recovers_mlle(self, henon_x: np.ndarray) -> None:
        res = lyapunov_from_data(
            henon_x, dimension=4, delay=1, k_max=12, method="rosenstein", fit=(0, 8)
        )
        assert float(res) == pytest.approx(0.419, abs=0.08)

    def test_auto_fit_is_sensible(self, henon_x: np.ndarray) -> None:
        # The default (no explicit fit) must land near the true value for a
        # clean, monotone divergence curve like the Hénon map's.
        res = lyapunov_from_data(henon_x, dimension=2, delay=1, k_max=12)
        assert 0.30 < float(res) < 0.55
        lo, hi = res.fit_region
        assert 0 <= lo < hi <= 12

    def test_multivariate_input(self, henon_x: np.ndarray) -> None:
        traj = ts.Henon().trajectory(6000, transient=500, ic=[0.1, 0.1])
        res = lyapunov_from_data(
            traj.y, dimension=2, delay=1, theiler=2, k_max=12, method="kantz", fit=(0, 6)
        )
        assert float(res) == pytest.approx(0.419, abs=0.08)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


class TestResult:
    def test_fields_and_float(self, henon_x: np.ndarray) -> None:
        res = lyapunov_from_data(henon_x, dimension=3, delay=1, k_max=15, dt=1.0, fit=(0, 7))
        assert isinstance(res, LyapunovFromData)
        assert res.embedding_dim == 3
        assert res.delay == 1
        assert res.theiler == 2  # (m-1)*tau
        assert res.method == "kantz"
        assert res.times.shape == res.divergence.shape == (16,)  # k = 0..k_max
        np.testing.assert_allclose(res.times, np.arange(16.0))  # dt = 1
        assert res.fit_region == (0, 7)
        assert float(res) == res.lyapunov
        assert "lyapunov" in repr(res)

    def test_dt_scales_exponent(self, henon_x: np.ndarray) -> None:
        # Per-time exponent halves when each sample spans twice the time.
        a = lyapunov_from_data(henon_x, dimension=3, k_max=12, dt=1.0, fit=(0, 6))
        b = lyapunov_from_data(henon_x, dimension=3, k_max=12, dt=2.0, fit=(0, 6))
        assert float(b) == pytest.approx(0.5 * float(a), rel=1e-12)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dimension": 0}, "embedding dimension"),
            ({"delay": 0}, "embedding delay"),
            ({"k_max": 1}, "k_max"),
            ({"n_neighbors": 0}, "n_neighbors"),
            ({"dt": 0.0}, "dt"),
            ({"method": "nope"}, "method"),
            ({"theiler": -1}, "theiler"),
            ({"fit": (5, 5)}, "fit region"),
            ({"fit": (0, 99)}, "fit region"),
        ],
    )
    def test_bad_params_raise(self, henon_x: np.ndarray, kwargs: dict, match: str) -> None:
        call = {"k_max": 10, **kwargs}  # kwargs overrides the default k_max
        with pytest.raises(ValueError, match=match):
            lyapunov_from_data(henon_x, **call)

    def test_constant_series_raises(self) -> None:
        with pytest.raises(ValueError, match="eps must be positive"):
            lyapunov_from_data(np.ones(500), dimension=2, k_max=10, method="kantz")

    def test_k_max_too_large_for_series(self) -> None:
        # n_rows = 40 - (3-1)*2 = 36; k_max=40 leaves no forward images.
        with pytest.raises(ValueError, match="too large"):
            lyapunov_from_data(
                np.random.default_rng(0).normal(size=40), dimension=3, delay=2, k_max=40
            )


# ---------------------------------------------------------------------------
# Lorenz flow — slow tier (the literature acceptance value)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("method", ["kantz", "rosenstein"])
def test_lorenz_from_x_series(method: str) -> None:
    lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
    traj = lor.integrate(final_time=300.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    xs = traj.y[1000:, 0]  # drop the initial transient
    # Fit the settled linear scaling region (t ≈ 0.8–1.9), past the early
    # overshoot and before saturation.
    res = lyapunov_from_data(
        xs, dt=0.05, dimension=5, delay=3, k_max=60, method=method, fit=(16, 38)
    )
    assert float(res) == pytest.approx(0.906, abs=0.15)
