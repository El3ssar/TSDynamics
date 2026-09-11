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

from .._discovery import register as _register
from .._result import CountResult
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

# Self-register the estimators: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_DATA = ("trajectory", "array")
_register(
    embed,
    subjects=_DATA,
    area="embedding",
    returns=Embedding,
    keywords="takens reconstruction delay coordinates state space",
    cite="Takens (1981), Lecture Notes in Mathematics 898, 366",
    doi="10.1007/BFb0091924",
)
_register(
    optimal_delay,
    subjects=_DATA,
    area="embedding",
    returns=CountResult,
    keywords="takens delay tau reconstruction decorrelation",
    cite="Fraser & Swinney (1986), Phys. Rev. A 33, 1134",
    doi="10.1103/PhysRevA.33.1134",
)
_register(
    mutual_information,
    subjects=_DATA,
    area="embedding",
    returns=MutualInformation,
    keywords="takens delay fraser swinney information",
    cite="Fraser & Swinney (1986), Phys. Rev. A 33, 1134",
    doi="10.1103/PhysRevA.33.1134",
)
_register(
    autocorrelation,
    subjects=_DATA,
    area="embedding",
    keywords="takens delay decorrelation correlation time",
)
_register(
    cao_dimension,
    subjects=_DATA,
    area="embedding",
    returns=EmbeddingDimension,
    keywords="takens reconstruction false neighbours cao",
    cite="Cao (1997), Physica D 110, 43",
    doi="10.1016/S0167-2789(97)00118-8",
)
_register(
    false_nearest_neighbors,
    subjects=_DATA,
    area="embedding",
    returns=EmbeddingDimension,
    keywords="takens reconstruction kennel fnn neighbours",
    cite="Kennel, Brown & Abarbanel (1992), Phys. Rev. A 45, 3403",
    doi="10.1103/PhysRevA.45.3403",
)
_register(
    embedding_dimension,
    subjects=_DATA,
    area="embedding",
    returns=EmbeddingDimension,
    keywords="takens reconstruction false neighbours dimension",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
