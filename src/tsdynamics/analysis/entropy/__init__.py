r"""
Entropy & complexity quantifiers — stream **A-ENT**.

Two layers share this package:

* a **composable core** — :func:`entropy` stacks an :class:`OutcomeSpace`
  (ordinal / dispersion / amplitude / raw symbols) on a
  :class:`ProbabilityEstimator` (:class:`MLE`, :class:`AddConstant`) on an
  :class:`InformationMeasure` (:class:`Shannon`, :class:`Renyi`, :class:`Tsallis`);
* **named measures** built on it or standalone —
  :func:`permutation_entropy` / :func:`weighted_permutation_entropy`,
  :func:`dispersion_entropy`, :func:`sample_entropy` / :func:`approximate_entropy`,
  :func:`multiscale_entropy`, and the Lempel–Ziv estimators
  :func:`lz76_complexity` / :func:`lz76_entropy`.

Every scalar entropy accepts a 1-D array, a 2-D array (with ``component=``), or a
:class:`~tsdynamics.data.Trajectory` (with ``component=`` naming a variable), so
they consume the :class:`~tsdynamics.data.Trajectory` lingua franca directly.

The headline callables self-register into :data:`tsdynamics.registry.analyses`
under stable names so they are discoverable alongside out-of-tree plugins.
"""

from __future__ import annotations

from ... import registry as _registry
from .core import (
    MLE,
    AddConstant,
    AmplitudeBinning,
    Dispersion,
    InformationMeasure,
    OrdinalPatterns,
    OutcomeSpace,
    ProbabilityEstimator,
    Renyi,
    Shannon,
    Tsallis,
    UniqueValues,
    as_series,
    entropy,
    probabilities,
)
from .dispersion import dispersion_entropy
from .lz import binarize, lz76_complexity, lz76_entropy, lz76_factors
from .multiscale import coarse_grain, multiscale_entropy
from .permutation import permutation_entropy, weighted_permutation_entropy
from .sample import approximate_entropy, sample_entropy

__all__ = [
    "MLE",
    "AddConstant",
    "AmplitudeBinning",
    "Dispersion",
    "InformationMeasure",
    "OrdinalPatterns",
    "OutcomeSpace",
    "ProbabilityEstimator",
    "Renyi",
    "Shannon",
    "Tsallis",
    "UniqueValues",
    "approximate_entropy",
    "as_series",
    "binarize",
    "coarse_grain",
    "dispersion_entropy",
    "entropy",
    "lz76_complexity",
    "lz76_entropy",
    "lz76_factors",
    "multiscale_entropy",
    "permutation_entropy",
    "probabilities",
    "sample_entropy",
    "weighted_permutation_entropy",
]

# In-tree self-registration into the generic analyses registry (mirrors the
# A-* stream convention).  ``needs="series"`` flags that these consume a scalar
# time series rather than a live System.  Idempotent across re-imports.
for _name, _fn in {
    "entropy": entropy,
    "permutation_entropy": permutation_entropy,
    "weighted_permutation_entropy": weighted_permutation_entropy,
    "dispersion_entropy": dispersion_entropy,
    "sample_entropy": sample_entropy,
    "approximate_entropy": approximate_entropy,
    "multiscale_entropy": multiscale_entropy,
    "lz76_complexity": lz76_complexity,
    "lz76_entropy": lz76_entropy,
}.items():
    _registry.analyses.register(_name, _fn, needs="series")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
