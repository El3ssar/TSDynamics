"""
Orbit diagrams, bifurcation sweeps and Poincaré sections.

Owned by stream **A-ORBIT**.  Two cooperating concerns share this subpackage:

- :func:`orbit_diagram` (:class:`OrbitDiagram`) — asymptotic states swept across
  a parameter; over a :class:`~tsdynamics.derived.PoincareMap` /
  :class:`~tsdynamics.derived.StroboscopicMap` it is the bifurcation diagram of
  a flow.
- :func:`poincare_section` — surfaces of section from a system (exact, root-
  refined crossings) or a :class:`~tsdynamics.data.Trajectory` (interpolated).
"""

from .orbit_diagram import OrbitDiagram, orbit_diagram
from .poincare import poincare_section

__all__ = [
    "OrbitDiagram",
    "orbit_diagram",
    "poincare_section",
]
