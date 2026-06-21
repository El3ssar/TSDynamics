"""Back-compat shim — :class:`WrappedSystem` now lives in :mod:`tsdynamics.families`.

``WrappedSystem`` is a base class users *subclass* (or instantiate) to adapt an
external stepper, so its canonical home is :mod:`tsdynamics.families.wrapped`,
alongside the other family bases.  It is re-exported here so the historical
paths — ``tsdynamics.derived.wrapped.WrappedSystem`` and
``from tsdynamics.derived import WrappedSystem`` — keep resolving unchanged.
"""

from __future__ import annotations

from tsdynamics.families.wrapped import WrappedSystem

__all__ = ["WrappedSystem"]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
