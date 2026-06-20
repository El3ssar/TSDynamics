r"""
Surrogate data & nonlinearity tests — stream **A-SURR**.

The surrogate-data method (Theiler, Eubank, Longtin, Galdrikian & Farmer,
*Physica D* **58**, 77, 1992; Schreiber & Schmitz, *Phys. Rev. Lett.* **77**, 635,
1996) tests a measured series for nonlinear structure by comparison with an
ensemble of *constrained-realisation* surrogates that reproduce its linear
properties — amplitude distribution and/or power spectrum — but are otherwise
random.  If a nonlinearity statistic separates the data from its surrogates, the
linear null is rejected.

- :func:`surrogates` — the by-name generator dispatcher over
  :func:`random_shuffle`, :func:`fourier_surrogate`, :func:`aaft_surrogate` and
  :func:`iaaft_surrogate` (progressively constrained nulls).
- :func:`time_reversal_asymmetry` / :func:`nonlinear_prediction_error` — the
  discriminating statistics, both blind to the linear properties the surrogates
  preserve.
- :func:`surrogate_test` — runs the full test and returns a :class:`SurrogateTest`
  (data statistic, surrogate ensemble, rank p-value, significance in sigmas).

Every entry reads a :class:`~tsdynamics.data.Trajectory` (select a component) or a
raw 1-D array interchangeably.  The headline callables self-register into
:data:`tsdynamics.registry.analyses`.
"""

from __future__ import annotations

from ... import registry as _registry
from .generators import (
    aaft_surrogate,
    fourier_surrogate,
    iaaft_surrogate,
    random_shuffle,
    surrogates,
)
from .hypothesis import SurrogateTest, surrogate_test
from .statistics import nonlinear_prediction_error, time_reversal_asymmetry

__all__ = [
    "SurrogateTest",
    "aaft_surrogate",
    "fourier_surrogate",
    "iaaft_surrogate",
    "nonlinear_prediction_error",
    "random_shuffle",
    "surrogate_test",
    "surrogates",
    "time_reversal_asymmetry",
]

# Self-register the headline analyses (D4 / §4e: in-tree analyses register from
# their own subpackage).  ``needs="series"`` flags scalar-series consumers.
# Idempotent across re-imports.
for _name, _fn in (
    ("surrogate_test", surrogate_test),
    ("surrogates", surrogates),
):
    _registry.analyses.register(_name, _fn, needs="series", family="surrogate")
del _name, _fn


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
