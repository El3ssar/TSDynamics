"""Utility helpers for TSDynamics."""

from .general import staticjit
from .grids import make_output_grid
from .sagitta_dt import SagittaDt, estimate_dt_from_sagitta

__all__ = [
    "SagittaDt",
    "estimate_dt_from_sagitta",
    "make_output_grid",
    "staticjit",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
