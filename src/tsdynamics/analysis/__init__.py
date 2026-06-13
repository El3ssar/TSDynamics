"""
Analysis toolkit — quantifiers that consume any :class:`~tsdynamics.base.System`.

- :func:`orbit_diagram` — parameter sweeps of discrete(-ized) systems;
  composed with :class:`~tsdynamics.derived.PoincareMap` /
  :class:`~tsdynamics.derived.StroboscopicMap` it draws bifurcation diagrams
  of flows.
- :func:`poincare_section` — surfaces of section from systems or trajectories.
- :func:`lyapunov_spectrum` / :func:`max_lyapunov` /
  :func:`kaplan_yorke_dimension` — Lyapunov quantifiers.
- :func:`fixed_points` — multi-start Newton fixed-point finding for maps,
  with linear stability.
"""

from .fixed_points import FixedPoint, fixed_points
from .lyapunov import kaplan_yorke_dimension, lyapunov_spectrum, max_lyapunov
from .orbit_diagram import OrbitDiagram, orbit_diagram
from .poincare import poincare_section

__all__ = [
    "FixedPoint",
    "OrbitDiagram",
    "fixed_points",
    "kaplan_yorke_dimension",
    "lyapunov_spectrum",
    "max_lyapunov",
    "orbit_diagram",
    "poincare_section",
]
