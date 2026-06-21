"""
The data layer — state-space geometry and the trajectory lingua franca.

Home of the primitives every analysis consumes:

- :class:`Trajectory` — the result of integrating or iterating a system; the
  time/state container the whole analysis layer consumes.  Named-component
  access, transient trimming, point-set ops, and lazy KD-tree neighbour
  queries live here.
- :class:`Box`, :class:`Ball`, :class:`Grid` — regions of state space, each
  with a ``contains`` predicate.
- :func:`sampler` — reproducible Monte-Carlo draws of initial conditions from
  a region.
- :func:`grid_points` — full-grid enumeration of a region.
- :func:`region` — terse ``[(lo, hi, n), ...]`` builder for a grid region.
- :func:`set_distance` — distance between two point sets (the matching
  primitive behind attractor deduplication and continuation).

These are pure NumPy/SciPy and depend on nothing from the compiled engine, so
they work uniformly across every system family.  :class:`Trajectory`
re-exports through :mod:`tsdynamics.families` and the top-level namespace, so
``from tsdynamics import Trajectory`` resolves to the same object defined here.
"""

from .sampling import Ball, Box, Grid, Region, grid_points, region, sampler, set_distance
from .trajectory import Trajectory

__all__ = [
    "Ball",
    "Box",
    "Grid",
    "Region",
    "Trajectory",
    "grid_points",
    "region",
    "sampler",
    "set_distance",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
