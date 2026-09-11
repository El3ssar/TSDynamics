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

from .._discovery import register as _register
from .orbit_diagram import OrbitDiagram, orbit_diagram
from .poincare import PoincareSection, poincare_section
from .return_map import ReturnMap, return_map

__all__ = [
    "OrbitDiagram",
    "PoincareSection",
    "ReturnMap",
    "orbit_diagram",
    "poincare_section",
    "return_map",
]

# Self-register the analyses: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_register(
    orbit_diagram,
    subjects=("system",),
    area="orbits",
    returns=OrbitDiagram,
    keywords="bifurcation cascade sweep period doubling feigenbaum",
    cite="May (1976), Nature 261, 459",
    doi="10.1038/261459a0",
)
_register(
    poincare_section,
    subjects=("system",),
    area="orbits",
    returns=PoincareSection,
    keywords="surface section crossings plane transversal",
    cite="Poincare (1899), Les methodes nouvelles de la mecanique celeste III",
)
_register(
    return_map,
    subjects=("system",),
    area="orbits",
    returns=ReturnMap,
    keywords="first return next amplitude cusp lorenz",
    cite="Lorenz (1963), J. Atmos. Sci. 20, 130",
    doi="10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
