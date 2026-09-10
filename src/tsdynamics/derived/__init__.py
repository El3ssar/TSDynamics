"""
Derived systems — wrappers that present an existing system through a new lens.

The composition layer of the library: the single-state wrappers
(:class:`PoincareMap`, :class:`StroboscopicMap`, :class:`TangentSystem`,
:class:`ProjectedSystem`) implement the
:class:`~tsdynamics.families.System` protocol, so anything written for systems
works on wrapped systems too.  A :class:`PoincareMap` of a flow *is* a
discrete map; an orbit diagram over it is a bifurcation diagram of the flow.
:class:`Ensemble` advances many copies in lockstep; since v6 it answers the
``System`` protocol too (``state()`` is the stacked member states) and its
``run`` returns a :class:`TrajectoryBatch`.

- :class:`PoincareMap` — hyperplane crossings of a flow as a discrete map.
- :class:`PoincareSection` — the crossing states it collects: a thin
  :class:`~tsdynamics.data.Trajectory` subclass carrying section (viz) intent.
- :class:`StroboscopicMap` — once-per-period samples of a forced flow.
- :class:`TangentSystem` — state + deviation vectors (Lyapunov engine).
- :class:`Ensemble` — many copies stepped, or run, together.  It is what
  ``system.ensemble(states)`` returns.
- :class:`TrajectoryBatch` — what ``Ensemble.run(...)`` returns: the list of
  trajectories, plus ``.final`` (the ``(n, dim)`` end states).
- :class:`ProjectedSystem` — observation-side component projection.
- :class:`WrappedSystem` — adapt any external stepper into the protocol.  It is a
  base class users subclass, so its canonical home is :mod:`tsdynamics.families`;
  it is re-exported here (where the other wrappers live) for back-compat.
"""

from ._base import DerivedSystem
from .ensemble import Ensemble, EnsembleSystem, TrajectoryBatch
from .poincare import PoincareMap, PoincareSection
from .projected import ProjectedSystem
from .stroboscopic import StroboscopicMap
from .tangent import TangentSystem
from .wrapped import WrappedSystem  # canonical home: tsdynamics.families.wrapped

__all__ = [
    "DerivedSystem",
    "Ensemble",
    "EnsembleSystem",
    "PoincareMap",
    "PoincareSection",
    "ProjectedSystem",
    "StroboscopicMap",
    "TangentSystem",
    "TrajectoryBatch",
    "WrappedSystem",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
