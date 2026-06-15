r"""
Chaos indicators — stream **A-CHAOS**.

Three complementary, literature-validated answers to "is this orbit chaotic?",
each reproducing the discriminator from its original paper:

- :func:`gali` — the Generalized Alignment Index GALI\ :sub:`k`
  (Skokos, Bountis & Antonopoulos 2007): the volume spanned by ``k`` unit
  deviation vectors.  Exponential decay marks chaos and measures the
  Lyapunov-exponent gaps; a constant/power-law tail marks order.
- :func:`zero_one_test` — the 0--1 test for chaos (Gottwald & Melbourne 2004,
  2009): a scalar :math:`K \in [0, 1]` read off a single observable, ``~0`` for
  regular and ``~1`` for chaotic dynamics.
- :func:`expansion_entropy` — Hunt & Ott's (2015) expansion entropy: the
  exponential growth rate of the in-region tangent-space volume, exact
  (``ln 2``) for a uniformly expanding map.

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

from ... import registry as _registry
from .expansion import ExpansionEntropyResult, expansion_entropy
from .gali import GALIResult, gali
from .zero_one import zero_one_test

__all__ = [
    "ExpansionEntropyResult",
    "GALIResult",
    "expansion_entropy",
    "gali",
    "zero_one_test",
]

# Self-register the indicators (D4 / §4e: in-tree analyses register from their
# own subpackage).  Idempotent across re-imports.
for _name, _fn, _meta in (
    ("gali", gali, {"needs": "system", "family": "chaos"}),
    ("zero_one_test", zero_one_test, {"needs": "series", "family": "chaos"}),
    ("expansion_entropy", expansion_entropy, {"needs": "system", "family": "chaos"}),
):
    _registry.analyses.register(_name, _fn, **_meta)
del _name, _fn, _meta
