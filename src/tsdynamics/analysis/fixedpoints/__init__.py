r"""
Fixed points & periodic orbits — stream **A-FP**.

Locating the invariant sets that organise a dynamical system, for maps *and*
flows, each with linear-stability classification:

- :func:`fixed_points` — fixed points of a :class:`~tsdynamics.families.DiscreteMap`
  (:math:`f(x) = x`) or equilibria of a :class:`~tsdynamics.families.ContinuousSystem`
  (:math:`f(x) = 0`) by multi-start Newton, with optional Schmelcher--Diakonos /
  Davidchack--Lai stabilising transformations to reach unstable points (maps).
- :func:`periodic_orbits` — period-``p`` orbits of a map (fixed points of
  :math:`f^{p}`), filtered to minimal period and merged over cyclic shifts.
- :func:`periodic_orbit` — a periodic orbit of a flow by single shooting on
  ``(x0, T)`` with the monodromy matrix and Floquet multipliers.
- :func:`estimate_period` — the dominant period of a sampled signal
  (autocorrelation / spectral peak), used to characterise a cycle or seed
  shooting.

:class:`FixedPoint` and :class:`PeriodicOrbit` carry the point/orbit, its
multipliers, and a stability flag using the right convention for its family.

The estimators self-register into :data:`tsdynamics.registry.analyses` so they
are discoverable by name alongside out-of-tree analysis plugins.

References
----------
Schmelcher & Diakonos (1997), *Phys. Rev. Lett.* 78, 4733.
Davidchack & Lai (1999), *Phys. Rev. E* 60, 6172.
"""

from __future__ import annotations

from ... import registry as _registry
from .fixed import FixedPoint, fixed_points
from .periodic import PeriodicOrbit, estimate_period, periodic_orbit, periodic_orbits

__all__ = [
    "FixedPoint",
    "PeriodicOrbit",
    "estimate_period",
    "fixed_points",
    "periodic_orbit",
    "periodic_orbits",
]

# Self-register the fixed-point / periodic-orbit finders (D4 / §4e: in-tree
# analyses register from their own subpackage).  Idempotent across re-imports.
for _name, _fn, _meta in (
    ("fixed_points", fixed_points, {"needs": "system", "family": "fixedpoints"}),
    ("periodic_orbits", periodic_orbits, {"needs": "system", "family": "fixedpoints"}),
    ("periodic_orbit", periodic_orbit, {"needs": "system", "family": "fixedpoints"}),
    ("estimate_period", estimate_period, {"needs": "series", "family": "fixedpoints"}),
):
    _registry.analyses.register(_name, _fn, **_meta)
del _name, _fn, _meta


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
