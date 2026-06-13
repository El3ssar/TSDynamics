"""
TSDynamics — compiled dynamical systems: integration, iteration, and chaos analysis.

Quick start
-----------
>>> from tsdynamics import Lorenz, MackeyGlass, Henon
>>> traj = Lorenz().integrate(final_time=100.0, dt=0.01)
>>> traj.t.shape, traj.y.shape
((10001,), (10001, 3))
>>> traj["x"]                          # named component access
>>> Lorenz().lyapunov_spectrum()       # ≈ [0.91, 0, -14.57]

Beyond integration, the :mod:`~tsdynamics.derived` wrappers re-present any
system through a new lens (Poincaré map, stroboscopic map, tangent dynamics,
ensembles), and :mod:`~tsdynamics.analysis` provides the quantifiers that
consume them (orbit/bifurcation diagrams, Poincaré sections, Lyapunov tools,
fixed points).

The top-level namespace re-exports every built-in system (see
:mod:`tsdynamics.registry` for programmatic access), the three base classes
users subclass to define new systems, the derived-system wrappers, and the
analysis functions.  Internal helpers (``ParamSet``, ``SystemBase``,
``staticjit``) live under ``tsdynamics.base`` / ``tsdynamics.utils``.
"""

from . import analysis, base, derived, registry, sampling, systems, utils
from .analysis import (
    FixedPoint,
    OrbitDiagram,
    fixed_points,
    kaplan_yorke_dimension,
    lyapunov_spectrum,
    max_lyapunov,
    orbit_diagram,
    poincare_section,
)
from .base import ContinuousSystem, DelaySystem, DiscreteMap, Trajectory
from .derived import (
    EnsembleSystem,
    PoincareMap,
    ProjectedSystem,
    StroboscopicMap,
    TangentSystem,
    WrappedSystem,
)
from .sampling import Ball, Box, Grid, grid_points, sampler, set_distance
from .systems import continuous as _continuous
from .systems import discrete as _discrete

# Single source of truth for the package version; rewritten by python-semantic-release.
__version__ = "2.1.1"

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
    # Derived-system wrappers (composition layer)
    "EnsembleSystem",
    "PoincareMap",
    "ProjectedSystem",
    "StroboscopicMap",
    "TangentSystem",
    "WrappedSystem",
    # Analysis toolkit
    "FixedPoint",
    "OrbitDiagram",
    "fixed_points",
    "kaplan_yorke_dimension",
    "lyapunov_spectrum",
    "max_lyapunov",
    "orbit_diagram",
    "poincare_section",
    # State-space geometry (regions, samplers, set distances)
    "Ball",
    "Box",
    "Grid",
    "grid_points",
    "sampler",
    "set_distance",
    # Sub-namespaces (for ``tsdynamics.systems.continuous.chaotic_attractors`` etc.)
    "analysis",
    "base",
    "derived",
    "registry",
    "sampling",
    "systems",
    "utils",
    # All built-in systems
    *_continuous_names,
    *_discrete_names,
]
