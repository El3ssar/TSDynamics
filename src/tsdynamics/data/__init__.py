"""
The data layer — state-space geometry and the trajectory lingua franca.

Home of the primitives every analysis consumes:

- :class:`Box`, :class:`Ball`, :class:`Grid` — regions of state space, each
  with a ``contains`` predicate.
- :func:`sampler` — reproducible Monte-Carlo draws of initial conditions from
  a region.
- :func:`grid_points` — full-grid enumeration of a region.
- :func:`set_distance` — distance between two point sets (the matching
  primitive behind attractor deduplication and continuation).

These are pure NumPy/SciPy and depend on nothing from the compiled engine, so
they work uniformly across every system family.  Stream C-DATA grows this
package further (re-homing :class:`~tsdynamics.families.Trajectory` and the
KD-tree neighbour queries here, to feature-parity with the v2 surface).
"""

from .sampling import Ball, Box, Grid, Region, grid_points, sampler, set_distance

__all__ = [
    "Ball",
    "Box",
    "Grid",
    "Region",
    "grid_points",
    "sampler",
    "set_distance",
]
