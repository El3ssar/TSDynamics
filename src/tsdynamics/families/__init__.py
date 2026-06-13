"""
System families — the base classes users subclass to define a system.

Public, intended for subclassing:

- :class:`ContinuousSystem` (``families/continuous.py``) — ODE systems.
- :class:`DelaySystem` (``families/delay.py``) — delay differential systems.
- :class:`DiscreteMap` (``families/discrete.py``) — iterated maps.
- :class:`Trajectory` — the result type returned by ``integrate`` / ``iterate``.

Internal but accessible for advanced use:

- :class:`SystemBase` — common base for all three; most users do not need it.
- :class:`ParamSet` — fixed-key parameter container backing ``system.params``.

A future :mod:`~tsdynamics.families.stochastic` adds the SDE family (stream
E-SDE); all families lower to the shared Rust engine under
:mod:`~tsdynamics.engine`.
"""

from .base import MetaStore, ParamSet, SystemBase, Trajectory
from .continuous import ContinuousSystem
from .delay import DelaySystem
from .discrete import DiscreteMap
from .protocol import System

__all__ = [
    # The three classes users subclass.
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    # Return type.
    "Trajectory",
    # The runtime protocol all analysis functions consume.
    "System",
    # Lower-level surface (kept here so it has a real import path, but not
    # in tsdynamics.__all__).
    "SystemBase",
    "ParamSet",
    "MetaStore",
]
