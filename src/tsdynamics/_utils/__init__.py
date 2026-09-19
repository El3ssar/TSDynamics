"""Utility helpers for TSDynamics."""

from .grids import make_output_grid
from .tolerances import (
    BASIN_ATOL,
    BASIN_RTOL,
    DDE_ATOL,
    DDE_LYAPUNOV_ATOL,
    DDE_LYAPUNOV_RTOL,
    DDE_RTOL,
    DEFAULT_ATOL,
    DEFAULT_RTOL,
)

__all__ = [
    "BASIN_ATOL",
    "BASIN_RTOL",
    "DDE_ATOL",
    "DDE_LYAPUNOV_ATOL",
    "DDE_LYAPUNOV_RTOL",
    "DDE_RTOL",
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "make_output_grid",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
