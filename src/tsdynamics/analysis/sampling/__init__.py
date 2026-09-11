"""Sampling / resampling tools for time-ordered trajectory data.

The sagitta-based output-step selector :func:`estimate_dt_from_sagitta` and the
per-point :func:`sagitta_profile` (the local bow of the trajectory off its chord)
live here.  ``SagittaDt`` (the selector's result container) is intentionally not
part of the public surface — reach it via its return value.
"""

from .._discovery import register as _register
from .sagitta import estimate_dt_from_sagitta, sagitta_profile

__all__ = ["estimate_dt_from_sagitta", "sagitta_profile"]

# Self-register: the definition site is the registration site (CONTRACT §7.7).
_DATA = ("trajectory", "array")
_register(
    estimate_dt_from_sagitta,
    subjects=_DATA,
    area="sampling",
    keywords="output step resolution smoothness undersampling",
)
_register(
    sagitta_profile,
    subjects=_DATA,
    area="sampling",
    keywords="bow curvature chord resolution smoothness",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
