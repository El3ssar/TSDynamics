r"""
Fixed points & periodic orbits — stream **A-FP**.

Locating the invariant sets that organise a dynamical system, for maps *and*
flows, each with linear-stability classification:

- :func:`fixed_points` — fixed points of a :class:`~tsdynamics.families.DiscreteMap`
  (:math:`f(x) = x`) or equilibria of a :class:`~tsdynamics.families.ContinuousSystem`
  (:math:`f(x) = 0`) by multi-start Newton, with optional Schmelcher--Diakonos /
  Davidchack--Lai stabilising transformations to reach unstable points (maps).
- :func:`periodic_orbits` — period-``p`` orbits of a map (fixed points of
  :math:`f^{p}`), filtered to minimal period and merged over cyclic shifts; on a
  flow, the limit cycle by single shooting on ``(x0, T)`` with the monodromy
  matrix and Floquet multipliers.  One verb, one return type (:class:`OrbitSet`).
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

from .._discovery import register as _register
from .._result import ScalarResult
from .fixed import FixedPoint, FixedPointSet, fixed_points
from .periodic import (
    OrbitSet,
    PeriodicOrbit,
    estimate_period,
    period_diagnostic,
    periodic_orbits,
)

__all__ = [
    "FixedPoint",
    "FixedPointSet",
    "OrbitSet",
    "PeriodicOrbit",
    "estimate_period",
    "fixed_points",
    "period_diagnostic",
    "periodic_orbits",
]

# Self-register the finders: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_register(
    fixed_points,
    subjects=("system",),
    area="fixedpoints",
    returns=FixedPointSet,
    keywords="equilibria equilibrium roots steady state saddle node",
    cite="Schmelcher & Diakonos (1997), Phys. Rev. Lett. 78, 4733",
    doi="10.1103/PhysRevLett.78.4733",
)
_register(
    periodic_orbits,
    subjects=("system",),
    area="fixedpoints",
    returns=OrbitSet,
    keywords="limit cycle periodic unstable shooting floquet",
    cite="Davidchack & Lai (1999), Phys. Rev. E 60, 6172",
    doi="10.1103/PhysRevE.60.6172",
)
_register(
    estimate_period,
    subjects=("trajectory", "array"),
    area="fixedpoints",
    returns=ScalarResult,
    keywords="period frequency oscillation cycle autocorrelation",
    cite="Box & Jenkins (1970), Time Series Analysis: Forecasting and Control",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
