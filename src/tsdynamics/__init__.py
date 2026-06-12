"""
TSDynamics — compiled ODE/DDE integration and Lyapunov analysis for dynamical systems.

Quick start
-----------
>>> from tsdynamics import Lorenz, MackeyGlass, Henon
>>> traj = Lorenz().integrate(final_time=100.0, dt=0.01)
>>> traj.t.shape, traj.y.shape
((10001,), (10001, 3))

The top-level namespace re-exports every built-in system, the three base
classes that users subclass to define new systems, and ``Trajectory``.
Internal helpers (``ParamSet``, ``SystemBase``, ``staticjit``) live under
``tsdynamics.base`` / ``tsdynamics.utils`` for users who need them.
"""

from . import base, systems, utils
from .base import ContinuousSystem, DelaySystem, DiscreteMap, Trajectory
from .systems import continuous as _continuous
from .systems import discrete as _discrete

# Single source of truth for the package version; rewritten by python-semantic-release.
__version__ = "1.0.0"

# Re-export every system class at the top level so ``from tsdynamics import Lorenz``
# works without users having to remember which submodule a system lives in.
_continuous_names = list(_continuous.__all__)
_discrete_names = list(_discrete.__all__)
for _name in _continuous_names:
    globals()[_name] = getattr(_continuous, _name)
for _name in _discrete_names:
    globals()[_name] = getattr(_discrete, _name)
del _name

__all__ = [
    "__version__",
    # User-facing base classes (subclass these to define a new system)
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "Trajectory",
    # Sub-namespaces (for ``tsdynamics.systems.continuous.chaotic_attractors`` etc.)
    "base",
    "systems",
    "utils",
    # All built-in systems
    *_continuous_names,
    *_discrete_names,
]
