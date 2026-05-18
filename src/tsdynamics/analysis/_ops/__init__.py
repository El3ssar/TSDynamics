"""
Domain-split analysis primitives.

Importing this package triggers all ``@trajectory_op`` decorators, which
registers every op in the global registry.  :func:`~._registry.install_methods`
(called from :mod:`tsdynamics.base.base` after :class:`~tsdynamics.base.Trajectory`
is defined) then drains the registry and installs one method per op on
``Trajectory``.

Adding a new milestone's ops
----------------------------
Drop a new ``<domain>.py`` sibling here, decorate each function with
``@trajectory_op``, and add an import line below.  Nothing else needs
to change — the registry handles method installation automatically.
"""

from __future__ import annotations

from .dataspec import to_dataspec
from .differential import derivative, norm
from .events import (
    Direction,
    EventCondition,
    LinearPlane,
    Plane,
    detect_events,
    poincare_section,
    return_map,
)
from .peaks import local_maxima, local_minima, return_times
from .resampling import decimate, project, resample, window

__all__ = [
    # Resampling / windowing
    "decimate",
    "resample",
    "project",
    "window",
    # Differential / norm
    "derivative",
    "norm",
    # Peak detection
    "local_maxima",
    "local_minima",
    "return_times",
    # Event-driven ops
    "Direction",
    "EventCondition",
    "LinearPlane",
    "Plane",
    "detect_events",
    "poincare_section",
    "return_map",
    # Plot-prep
    "to_dataspec",
]
