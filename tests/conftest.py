"""Shared pytest fixtures and configuration for TSDynamics tests."""

from __future__ import annotations

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker centrally."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that compile JiTCODE/JiTCDDE C code or run long "
        "simulations (deselect with: pytest -m 'not slow').",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """A reproducible NumPy ``Generator`` seeded at 42."""
    return np.random.default_rng(42)
