"""Sampling / resampling tools for time-ordered trajectory data.

The sagitta-based output-step selector :func:`estimate_dt_from_sagitta` and the
per-point :func:`sagitta_profile` (the local bow of the trajectory off its chord)
live here.  ``SagittaDt`` (the selector's result container) is intentionally not
part of the public surface — reach it via its return value.
"""

from .sagitta import estimate_dt_from_sagitta, sagitta_profile

__all__ = ["estimate_dt_from_sagitta", "sagitta_profile"]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
