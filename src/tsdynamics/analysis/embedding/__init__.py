r"""
Delay embeddings — stream **A-EMBED**.

State-space reconstruction from a scalar (or multivariate) measurement, following
Takens' theorem: a single time series is turned into a trajectory in a
delay-coordinate space whose attractor is diffeomorphic to the original, so its
geometric and dynamical invariants are recoverable.

The toolkit covers the three choices a reconstruction needs:

- :func:`embed` — the time-delay (Takens) map itself, univariate or multivariate.
- delay selection — :func:`optimal_delay` (and the underlying
  :func:`mutual_information` / :func:`autocorrelation` curves): the first minimum
  of the time-delayed mutual information (Fraser--Swinney) or an autocorrelation
  rule.
- dimension selection — :func:`cao_dimension` (Cao's averaged false neighbours)
  and :func:`false_nearest_neighbors` (Kennel's FNN), unified behind
  :func:`embedding_dimension`; both return an :class:`EmbeddingDimension` that
  drops straight into :func:`embed` (``int(result)``).

Every estimator reads a :class:`~tsdynamics.data.Trajectory` or a raw array
interchangeably, and :func:`embed`'s output feeds the point-set analyses
directly (e.g. ``correlation_dimension(embed(x, m, tau))``).  The headline
functions self-register into :data:`tsdynamics.registry.analyses`.
"""

from __future__ import annotations

from ... import registry as _registry
from .delay import MutualInformation, autocorrelation, mutual_information, optimal_delay
from .dimension import (
    EmbeddingDimension,
    cao_dimension,
    embedding_dimension,
    false_nearest_neighbors,
)
from .embed import Embedding, embed

__all__ = [
    "Embedding",
    "EmbeddingDimension",
    "MutualInformation",
    "autocorrelation",
    "cao_dimension",
    "embed",
    "embedding_dimension",
    "false_nearest_neighbors",
    "mutual_information",
    "optimal_delay",
]

# Self-register the headline estimators (D4 / §4e: in-tree analyses register from
# their own subpackage).  Idempotent across re-imports.
for _name, _fn in (
    ("embed", embed),
    ("optimal_delay", optimal_delay),
    ("mutual_information", mutual_information),
    ("cao_dimension", cao_dimension),
    ("false_nearest_neighbors", false_nearest_neighbors),
    ("embedding_dimension", embedding_dimension),
):
    _registry.analyses.register(_name, _fn, needs="series", family="embedding")
del _name, _fn


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
