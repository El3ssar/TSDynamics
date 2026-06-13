"""
Base classes for dynamical systems.

Public, intended for subclassing:

- :class:`ContinuousSystem` — ODE systems (compiled via JiTCODE).
- :class:`DelaySystem` — delay differential systems (compiled via JiTCDDE).
- :class:`DiscreteMap` — iterated maps (compiled via Numba).
- :class:`Trajectory` — the result type returned by ``integrate`` / ``iterate``.

Internal but accessible for advanced use:

- :class:`SystemBase` — common base for all three; most users do not need it.
- :class:`ParamSet` — fixed-key parameter container backing ``system.params``.
"""

from .base import MetaStore, ParamSet, SystemBase, Trajectory
from .dde_base import DelaySystem
from .map_base import DiscreteMap
from .ode_base import ContinuousSystem
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
