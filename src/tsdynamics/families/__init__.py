"""
System families — the base classes users subclass to define a system.

Public, intended for subclassing:

- :class:`ContinuousSystem` (``families/continuous.py``) — ODE systems.
- :class:`DelaySystem` (``families/delay.py``) — delay differential systems.
- :class:`DiscreteMap` (``families/discrete.py``) — iterated maps.
- :class:`StochasticSystem` (``families/stochastic.py``) — diagonal-Itô SDEs
  (``_drift`` + ``_diffusion``; Euler–Maruyama / Milstein).
- :class:`Trajectory` — the result type returned by ``integrate`` / ``iterate``.

Internal but accessible for advanced use:

- :class:`SystemBase` — common base for all families; most users do not need it.
- :class:`ParamSet` — fixed-key parameter container backing ``system.params``.

All families lower to the shared Rust engine under :mod:`~tsdynamics.engine`.
"""

from .base import MetaStore, ParamSet, SystemBase, Trajectory
from .continuous import ContinuousSystem
from .delay import DelaySystem
from .discrete import DiscreteMap
from .protocol import System
from .stochastic import StochasticSystem

__all__ = [
    # The classes users subclass.
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "StochasticSystem",
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
