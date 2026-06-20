r"""
Attractors & basins — stream **A-BASIN** (the parity moat).

The global picture of a multistable system: *which* attractors it has, *where*
each one wins, and how that partition behaves and breaks.

- :func:`find_attractors` / :func:`basins_of_attraction` — locate the attractors
  and paint the basin of attraction of each, by following trajectories through a
  cell tessellation until they recurrently revisit cells (Datseris &
  Wagemakers, 2022).
- :func:`basin_fractions` — basin stability: each attractor's share of a sampled
  region, with a dimension-free Monte-Carlo error (Menck et al., 2013).
- :func:`basin_entropy` — basin entropy :math:`S_b` and boundary basin entropy
  :math:`S_{bb}`, with the :math:`S_{bb}>\log 2` fractal-boundary test (Daza et
  al., 2016).
- :func:`uncertainty_exponent` — the final-state-sensitivity exponent of a basin
  boundary (Grebogi et al., 1983).
- :func:`wada_property` — a grid test for Wada basins (Daza et al., 2015).
- :func:`continuation` / :func:`tipping_points` — track attractors and basin
  fractions across a parameter and read off where a basin annihilates (Datseris,
  Rossi & Wagemakers, 2023).
- :func:`resilience` — the minimal-fatal-shock distance from an attractor to its
  basin boundary (Halekotte & Feudel, 2020).

Every headline function self-registers into
:data:`tsdynamics.registry.analyses`.
"""

from __future__ import annotations

from ... import registry as _registry
from .attractors import Attractor, AttractorSet, find_attractors
from .basins import BasinFractions, BasinsResult, basin_fractions, basins_of_attraction
from .continuation import ContinuationResult, continuation, tipping_points
from .metrics import (
    BasinEntropy,
    UncertaintyExponent,
    WadaResult,
    basin_entropy,
    resilience,
    uncertainty_exponent,
    wada_property,
)

__all__ = [
    "Attractor",
    "AttractorSet",
    "BasinEntropy",
    "BasinFractions",
    "BasinsResult",
    "ContinuationResult",
    "UncertaintyExponent",
    "WadaResult",
    "basin_entropy",
    "basin_fractions",
    "basins_of_attraction",
    "continuation",
    "find_attractors",
    "resilience",
    "tipping_points",
    "uncertainty_exponent",
    "wada_property",
]

# Self-register the headline analyses (D4 / §4e: in-tree analyses register from
# their own subpackage).  Idempotent across re-imports.
for _name, _fn, _needs in (
    ("find_attractors", find_attractors, "system"),
    ("basins_of_attraction", basins_of_attraction, "system"),
    ("basin_fractions", basin_fractions, "system"),
    ("continuation", continuation, "system"),
    ("basin_entropy", basin_entropy, "basins"),
    ("uncertainty_exponent", uncertainty_exponent, "basins"),
    ("wada_property", wada_property, "basins"),
    ("resilience", resilience, "basins"),
    ("tipping_points", tipping_points, "continuation"),
):
    _registry.analyses.register(_name, _fn, needs=_needs, family="basins")
del _name, _fn, _needs


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
