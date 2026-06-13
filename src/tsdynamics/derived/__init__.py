"""
Derived systems — wrappers that present an existing system through a new lens.

The composition layer of the library: every wrapper implements the
:class:`~tsdynamics.base.System` protocol, so anything written for systems
works on wrapped systems too.  A :class:`PoincareMap` of a flow *is* a
discrete map; an orbit diagram over it is a bifurcation diagram of the flow.

- :class:`PoincareMap` — hyperplane crossings of a flow as a discrete map.
- :class:`StroboscopicMap` — once-per-period samples of a forced flow.
- :class:`TangentSystem` — state + deviation vectors (Lyapunov engine).
- :class:`EnsembleSystem` — many copies stepped in lockstep.
- :class:`ProjectedSystem` — observation-side component projection.
- :class:`WrappedSystem` — adapt any external stepper into the protocol.
"""

from ._base import DerivedSystem
from .ensemble import EnsembleSystem
from .poincare import PoincareMap
from .projected import ProjectedSystem
from .stroboscopic import StroboscopicMap
from .tangent import TangentSystem
from .wrapped import WrappedSystem

__all__ = [
    "DerivedSystem",
    "EnsembleSystem",
    "PoincareMap",
    "ProjectedSystem",
    "StroboscopicMap",
    "TangentSystem",
    "WrappedSystem",
]
