"""
Orbit diagrams, bifurcation sweeps, Poincaré sections and return maps.

Owned by stream **A-ORBIT**.  Cooperating concerns share this subpackage:

- :func:`orbit_diagram` (:class:`OrbitDiagram`) — asymptotic states swept across
  a parameter; over a :class:`~tsdynamics.derived.PoincareMap` /
  :class:`~tsdynamics.derived.StroboscopicMap` it is the bifurcation diagram of
  a flow.  :meth:`OrbitDiagram.periods` / :meth:`OrbitDiagram.bifurcation_points`
  quantify the cascade.
- :func:`poincare_section` — surfaces of section from a system (exact, root-
  refined crossings) or a :class:`~tsdynamics.data.Trajectory` (interpolated).
- :func:`return_map` (:class:`ReturnMap`) — the first-return / next-amplitude
  map of a recurring observable (Lorenz, 1963), exposing the one-dimensional
  dynamics inside a flow.

The estimators self-register into :data:`tsdynamics.registry.analyses` so they
are discoverable by name alongside out-of-tree analysis plugins.
"""

from ... import registry as _registry
from .orbit_diagram import OrbitDiagram, orbit_diagram
from .poincare import poincare_section
from .return_map import ReturnMap, return_map

__all__ = [
    "OrbitDiagram",
    "ReturnMap",
    "orbit_diagram",
    "poincare_section",
    "return_map",
]

# Self-register the headline analyses (D4 / §4e: in-tree analyses register from
# their own subpackage).  Idempotent across re-imports.
for _name, _fn in (
    ("orbit_diagram", orbit_diagram),
    ("poincare_section", poincare_section),
    ("return_map", return_map),
):
    _registry.analyses.register(_name, _fn, needs="system", family="orbits")
del _name, _fn


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
