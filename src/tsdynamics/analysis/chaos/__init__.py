r"""
Chaos indicators — stream **A-CHAOS**.

Three complementary finite-data diagnostics, with different assumptions and
different meanings:

- :func:`gali` — the Generalized Alignment Index GALI\ :sub:`k`
  (Skokos, Bountis & Antonopoulos 2007): the volume spanned by ``k`` unit
  deviation vectors. Alignment decay measures Lyapunov-exponent gaps; it also
  occurs on regular attracting cycles, so collapse alone does not prove chaos.
- :func:`zero_one_test` — the 0--1 test for chaos (Gottwald & Melbourne 2004,
  2009): a correlation statistic :math:`K \in [-1, 1]` read off a single
  observable, ideally near zero for regular and near one for chaotic dynamics.
  Sampling, noise and finite duration can compromise that distinction.
- :func:`expansion_entropy` — Hunt & Ott's (2015) expansion entropy: the
  exponential growth rate of the surviving, expanding tangent volume in a
  chosen region; the result is a finite-time Monte Carlo estimate of that rate.

:func:`gali` and :func:`expansion_entropy` evolve tangent dynamics, so they take
a live :class:`~tsdynamics.families.DiscreteMap` or
:class:`~tsdynamics.families.ContinuousSystem`; :func:`zero_one_test` consumes a
sampled observable and so works downstream of any family.  :class:`GALIResult`
and :class:`ExpansionEntropyResult` behave as their headline number in
arithmetic while carrying the curve the conclusion was read from.

The estimators self-register into :data:`tsdynamics.registry.analyses` so they
are discoverable by name alongside out-of-tree analysis plugins.
"""

from __future__ import annotations

from tsdynamics.families import ContinuousSystem as _ContinuousSystem
from tsdynamics.families import DiscreteMap as _DiscreteMap

from .._discovery import register as _register
from .expansion import ExpansionEntropyResult, expansion_entropy
from .gali import GALIResult, gali
from .zero_one import OversamplingWarning, ZeroOneResult, zero_one_test

__all__ = [
    "ExpansionEntropyResult",
    "GALIResult",
    "OversamplingWarning",
    "ZeroOneResult",
    "expansion_entropy",
    "gali",
    "zero_one_test",
]

# Self-register the indicators: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_register(
    gali,
    system_types=(_ContinuousSystem, _DiscreteMap),
    families=("ode", "map"),
    min_dim=2,
    subjects=("system",),
    area="chaos",
    returns=GALIResult,
    keywords="chaotic chaos regular ordered skokos alignment",
    cite="Skokos, Bountis & Antonopoulos (2007), Physica D 231, 30",
    doi="10.1016/j.physd.2007.04.004",
)
_register(
    zero_one_test,
    families=("ode", "map", "dde", "sde"),
    # A SYSTEM **or** measured data: its own summary line says so, it works on a
    # bare ndarray, and it was registered system-only — so ``find(my_array)``, the
    # library's best discovery route, hid the one analysis that answers the
    # question a data-first user arrives with.
    subjects=("system", "trajectory", "array"),
    area="chaos",
    returns=ZeroOneResult,
    keywords="chaotic chaos regular gottwald melbourne binary",
    cite="Gottwald & Melbourne (2004), Proc. R. Soc. Lond. A 460, 603",
    doi="10.1098/rspa.2003.1183",
)
_register(
    expansion_entropy,
    system_types=(_ContinuousSystem, _DiscreteMap),
    families=("ode", "map"),
    subjects=("system",),
    area="chaos",
    returns=ExpansionEntropyResult,
    keywords="chaotic chaos volume growth hunt ott",
    cite="Hunt & Ott (2015), Chaos 25, 097618",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
