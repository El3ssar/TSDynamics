"""
The data layer — state-space geometry and the trajectory lingua franca.

Home of the primitives every analysis consumes:

- :class:`Trajectory` — the result of integrating or iterating a system; the
  time/state container the whole analysis layer consumes.  Named-component
  access, transient trimming, point-set ops, and lazy KD-tree neighbour
  queries live here.
- :class:`Box`, :class:`Ball`, :class:`Grid` — regions of state space, each
  with a ``contains`` predicate.  They are *accepted* everywhere a region is
  taken and *required* nowhere: see :func:`as_region`.
- :func:`as_region` — the one region reading in the library: one ``(lo, hi)``
  bound (or ``(lo, hi, n)`` triple) per state component.
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

Why the region types live here and not on the top level
-------------------------------------------------------
Corollary **C1**, the toll rule: *plain Python reaches every front door.*  Every
``region=`` argument in the library reads one ``(lo, hi)`` bound — or a
``(lo, hi, n)`` triple where a resolution is meaningful — per state component,
through the single reading in :func:`as_region`::

    ts.analysis.basins(hen, [(-2, 2, 200), (-2, 2, 200)])
    ts.analysis.fixed_points(vdp, region=[(-3, 3), (-3, 3)])
    ts.data.sampler([(-3, 3), (-3, 3)], seed=0)()

:class:`Box` / :class:`Ball` / :class:`Grid` are *accepted* at every one of those
doors and *required* at none — measured, the tuple literal is not merely allowed,
it is shorter than constructing the type.  A name exported *because a signature
demands it* would be evidence of a signature bug; these are the opposite case, so
they stay one dot down at ``ts.data.<Name>``, and the address book in
``tsdynamics.__getattr__`` sends anyone who guesses ``ts.Box`` straight here.
:class:`Trajectory` is the one exception, because you annotate it and
``isinstance`` it — a type you *type*, not merely one you receive.
"""

from .sampling import (
    Ball,
    Box,
    Grid,
    Region,
    as_region,
    grid_points,
    region,
    sampler,
    set_distance,
)
from .trajectory import Trajectory

__all__ = [
    "Ball",
    "Box",
    "Grid",
    "Region",
    "Trajectory",
    "as_region",
    "grid_points",
    "region",
    "sampler",
    "set_distance",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
