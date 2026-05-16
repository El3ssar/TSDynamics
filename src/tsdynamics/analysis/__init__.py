"""
Analysis primitives for trajectories.

One contract, one return type
-----------------------------

Every analysis primitive in this module is a function
``fn(t, y, *args, **kwargs)`` returning a
:class:`~tsdynamics.base.Trajectory`.  No bare tuples, no ad-hoc result
objects, no special-case shapes.  This is what lets plotters and
downstream pipelines consume *any* analysis result uniformly:

- Trajectory ops (``decimate``, ``resample``, ``project``, ``window``,
  ``derivative``) — output has the trajectory's time + state shape.
- Reductions (``norm``, ``local_maxima``, ``local_minima``,
  ``return_times``) — output is a 1-column Trajectory ``(K, 1)``.
- Event-driven ops (``detect_events``, ``poincare_section``,
  ``return_map``) — output has crossing times + state (or observable)
  at crossings.
- Plot-prep (``to_dataspec``) — the single non-Trajectory exit, returns
  a plain dict that V1 will turn into a real ``DataSpec``.

Two API layers, one implementation
----------------------------------

Each primitive is decorated with :func:`._registry.trajectory_op`.  The
decorator gives both forms for free:

1. A polymorphic free function::

       decimate(traj, every=10)
       decimate((t, y), every=10)
       decimate(t, y, every=10)
       detect_events(traj, axis=2, value=27.0)
       poincare_section((t, y), Plane(axis=2, value=27.0))

2. A fluent method on :class:`~tsdynamics.base.Trajectory`, installed
   at ``Trajectory``-class-init time by
   :func:`._registry.install_methods`::

       traj.decimate(every=10)
       traj.detect_events(axis=2, value=27.0)
       traj.poincare_section(Plane(axis=2, value=27.0))

Adding a new primitive is one decorated function plus one re-export
from this module — no class to touch, no docstring to duplicate.

Event-driven ops accept three call styles
-----------------------------------------

:func:`detect_events`, :func:`poincare_section`, :func:`return_map` all
accept any of:

- An :class:`EventCondition` (``Plane``, ``LinearPlane``, or any
  ``.detect``-bearing object).
- A bare callable ``fn(t, y) -> float`` (sub-sample refinement still
  applied; pass ``direction="up"`` etc. as a kwarg).
- Shortcut kwargs ``axis=`` + ``value=`` (axis-aligned plane), or
  ``normal=`` + ``offset=`` (linear plane).

Pick whichever reads best at the call site.
"""

from __future__ import annotations

from ._events import (
    Direction,
    EventCondition,
    LinearPlane,
    Plane,
    detect_events,
    poincare_section,
    return_map,
)
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

__all__ = [
    # Event conditions (the only public condition classes — Threshold,
    # Custom, LocalExtremum were dropped; use Plane(direction="up"),
    # bare callables, and local_maxima(refined=True) respectively).
    "Direction",
    "EventCondition",
    "LinearPlane",
    "Plane",
    # Event-driven ops.
    "detect_events",
    "poincare_section",
    "return_map",
    # Trajectory enrichment.
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
