"""Shared pytest fixtures and configuration for TSDynamics tests."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must come before any pyplot import

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that compile JiTCODE/JiTCDDE C code or run long simulations "
        "(deselect with: pytest -m 'not slow')",
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_3d_trajectory(rng):
    """
    Synthetic 3D trajectory shaped (T, 3) — no ODE compilation needed.
    Resembles a noisy spiral to give realistic spreads to plotters.
    """
    T = 800
    t = np.linspace(0.0, 20.0, T)
    theta = t * 3.5
    r = 5.0 + 0.5 * np.sin(0.3 * t)
    X = np.column_stack(
        [
            r * np.cos(theta),
            r * np.sin(theta),
            t * 0.4 + 0.3 * rng.standard_normal(T),
        ]
    )
    return t, X


@pytest.fixture
def scalar_series(rng):
    """Scalar time series with some structure (800 points)."""
    t = np.linspace(0, 20, 800)
    x = np.sin(2 * np.pi * 0.5 * t) + 0.3 * rng.standard_normal(800)
    return t, x


@pytest.fixture
def synthetic_ks_trajectory(rng):
    """
    KS-like high-dimensional trajectory shaped (T, N_spatial).
    Mimics the output of KuramotoSivashinsky.integrate() with N=32 spatial points.
    """
    T, N = 200, 32
    t = np.linspace(0, 10, T)
    X = rng.standard_normal((T, N))
    return t, X
