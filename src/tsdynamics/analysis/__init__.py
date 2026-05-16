"""
Analysis primitives for trajectories.

Two API layers, one implementation
----------------------------------

Every analysis primitive is defined exactly once — as a function

    fn(t, y, *args, **kwargs) -> result

decorated with :func:`._registry.trajectory_op`.  The decorator gives you
*both* of these for free:

1. A polymorphic free function — call with a :class:`~tsdynamics.Trajectory`,
   a ``(t, y)`` tuple, or bare ``t, y`` arrays::

       decimate(traj, every=10)
       decimate((t, y), every=10)
       decimate(t, y, every=10)
       detect_events(traj, Plane(axis=2, value=27.0))
       poincare_section((t, y), Plane(axis=2, value=27.0))

2. A fluent method on :class:`~tsdynamics.Trajectory` — installed at
   ``Trajectory``-class-init time by :func:`._registry.install_methods`::

       traj.decimate(every=10)
       traj.detect_events(Plane(axis=2, value=27.0))
       traj.poincare_section(Plane(axis=2, value=27.0))

Adding a new analysis primitive is therefore one decorated function plus
one import here — there is no separate method to write, no docstring to
duplicate, and no `_trajectory_ops`-vs-`base.py` mirror to keep in sync.

Re-exported names below cover every registered op.  Implementation files
live under leading-underscore names (``_registry.py``,
``_trajectory_ops.py``) to mark them internal — import from this package
instead.
"""

from __future__ import annotations

from ._trajectory_ops import (
    decimate,
    derivative,
    local_maxima,
    local_minima,
    norm,
    project,
    resample,
    return_times,
    to_dataspec,
    window,
)
from .events import (
    Custom,
    Direction,
    EventCondition,
    EventResult,
    LinearPlane,
    LocalExtremum,
    Plane,
    Threshold,
    detect_events,
)
from .return_map import Observable, ReturnMap, return_map
from .sections import poincare_section

__all__ = [
    # M2 — event detection and section construction.
    "Custom",
    "Direction",
    "EventCondition",
    "EventResult",
    "LinearPlane",
    "LocalExtremum",
    "Observable",
    "Plane",
    "ReturnMap",
    "Threshold",
    "detect_events",
    "poincare_section",
    "return_map",
    # M1 — trajectory enrichment.
    "decimate",
    "derivative",
    "local_maxima",
    "local_minima",
    "norm",
    "project",
    "resample",
    "return_times",
    "to_dataspec",
    "window",
]
