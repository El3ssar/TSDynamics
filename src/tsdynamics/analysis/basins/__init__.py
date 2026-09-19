r"""
Attractors & basins — stream **A-BASIN** (the parity moat).

The global picture of a multistable system: *which* attractors it has, *where*
each one wins, and how that partition behaves and breaks.

- :func:`attractors` / :func:`basins` — locate the attractors
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

from .._discovery import register as _register
from .._result import CollectionResult, ScalarResult
from .attractors import Attractor, AttractorSet, attractors
from .basins import BasinFractions, BasinsResult, basin_fractions, basins
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
    "basins",
    "continuation",
    "attractors",
    "resilience",
    "tipping_points",
    "uncertainty_exponent",
    "wada_property",
]

# Self-register the headline analyses: the definition site is the registration
# site (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_register(
    attractors,
    subjects=("system",),
    area="basins",
    returns=AttractorSet,
    keywords="multistability coexisting attracting sets recurrence",
    cite="Datseris & Wagemakers (2022), Chaos 32, 023104",
    doi="10.1063/5.0076568",
)
_register(
    basins,
    subjects=("system",),
    area="basins",
    returns=BasinsResult,
    keywords="multistability basin attraction final state riddled",
    cite="Datseris & Wagemakers (2022), Chaos 32, 023104",
    doi="10.1063/5.0076568",
)
_register(
    basin_fractions,
    subjects=("system",),
    area="basins",
    returns=BasinFractions,
    keywords="multistability basin stability menck sampling",
    cite="Menck, Heitzig, Marwan & Kurths (2013), Nature Physics 9, 89",
    doi="10.1038/nphys2516",
)
_register(
    continuation,
    subjects=("system",),
    area="basins",
    returns=ContinuationResult,
    keywords="bifurcation tracking parameter sweep multistability",
    cite="Datseris, Rossi & Wagemakers (2023), Int. J. Bifurc. Chaos 33, 2330008",
    doi="10.1142/S0218127423300082",
)
_register(
    basin_entropy,
    subjects=("BasinsResult",),
    area="basins",
    returns=BasinEntropy,
    keywords="fractal boundary uncertainty daza multistability",
    cite="Daza, Wagemakers, Georgeot, Guery-Odelin & Sanjuan (2016), Sci. Rep. 6, 31416",
    doi="10.1038/srep31416",
)
_register(
    uncertainty_exponent,
    subjects=("BasinsResult",),
    area="basins",
    returns=UncertaintyExponent,
    keywords="fractal boundary predictability final state sensitivity",
    cite="Grebogi, McDonald, Ott & Yorke (1983), Phys. Lett. A 99, 415",
    doi="10.1016/0375-9601(83)90945-3",
)
_register(
    wada_property,
    subjects=("BasinsResult",),
    area="basins",
    returns=WadaResult,
    keywords="wada boundary fractal three basins",
    cite="Daza, Wagemakers, Sanjuan & Yorke (2015), Sci. Rep. 5, 16579",
    doi="10.1038/srep16579",
)
_register(
    resilience,
    subjects=("BasinsResult",),
    area="basins",
    returns=ScalarResult,
    keywords="shock perturbation robustness tipping distance",
    cite="Halekotte & Feudel (2020), Sci. Rep. 10, 11783",
    doi="10.1038/s41598-020-68805-6",
)
_register(
    tipping_points,
    subjects=("ContinuationResult",),
    area="basins",
    returns=CollectionResult,
    keywords="bifurcation tipping catastrophe annihilation multistability",
    cite="Datseris, Rossi & Wagemakers (2023), Int. J. Bifurc. Chaos 33, 2330008",
    doi="10.1142/S0218127423300082",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
