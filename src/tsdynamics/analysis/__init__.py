"""
Analysis toolkit — quantifiers that consume any :class:`~tsdynamics.families.System`.

Each capability cluster lives in its own subpackage (one per analysis stream),
re-exported here so the public surface is flat:
``from tsdynamics import lyapunov_spectrum`` and
``from tsdynamics.analysis import lyapunov_spectrum`` both work.

- :mod:`~tsdynamics.analysis.orbits` — :func:`orbit_diagram` (parameter sweeps of
  discrete(-ized) systems; over a :class:`~tsdynamics.derived.PoincareMap` /
  :class:`~tsdynamics.derived.StroboscopicMap` it draws bifurcation diagrams of
  flows) and :func:`poincare_section` (surfaces of section).
- :mod:`~tsdynamics.analysis.lyapunov` — :func:`lyapunov_spectrum` /
  :func:`max_lyapunov` / :func:`kaplan_yorke_dimension`.
- :mod:`~tsdynamics.analysis.fixedpoints` — :func:`fixed_points`, multi-start
  Newton fixed-point finding for maps with linear stability.
- :mod:`~tsdynamics.analysis.entropy` — composable :func:`entropy` plus
  :func:`permutation_entropy`, :func:`dispersion_entropy`,
  :func:`sample_entropy` / :func:`approximate_entropy`,
  :func:`multiscale_entropy`, and Lempel–Ziv :func:`lz76_complexity` /
  :func:`lz76_entropy`.
- :mod:`~tsdynamics.analysis.dimensions` — fractal dimensions:
  :func:`correlation_dimension` (Grassberger--Procaccia), the generalized/Rényi
  :func:`generalized_dimension` (with :func:`box_counting_dimension`,
  :func:`information_dimension`, :func:`dimension_spectrum`) and
  :func:`fixed_mass_dimension`.
- :mod:`~tsdynamics.analysis.embedding` — delay embeddings: :func:`embed`
  (Takens reconstruction), delay selection :func:`optimal_delay` (mutual
  information / autocorrelation) and dimension selection :func:`cao_dimension` /
  :func:`false_nearest_neighbors` (unified by :func:`embedding_dimension`).

The remaining subpackages (``chaos``, ``basins``, ``recurrence``,
``surrogate``) are placeholders the analysis streams fill in.

Out-of-tree analyses register through the ``tsdynamics.analyses`` entry-point
group (see :mod:`tsdynamics.plugins`); :func:`discover_plugins` loads them into
:data:`tsdynamics.registry.analyses`.
"""

from .. import registry as _registry
from ..plugins import ANALYSES_GROUP, register_entry_points
from .dimensions import (
    DimensionResult,
    box_counting_dimension,
    correlation_dimension,
    correlation_sum,
    dimension_spectrum,
    fixed_mass_dimension,
    generalized_dimension,
    information_dimension,
)
from .embedding import (
    EmbeddingDimension,
    autocorrelation,
    cao_dimension,
    embed,
    embedding_dimension,
    false_nearest_neighbors,
    mutual_information,
    optimal_delay,
)
from .entropy import (
    approximate_entropy,
    dispersion_entropy,
    entropy,
    lz76_complexity,
    lz76_entropy,
    multiscale_entropy,
    permutation_entropy,
    sample_entropy,
    weighted_permutation_entropy,
)
from .fixedpoints import FixedPoint, fixed_points
from .lyapunov import (
    LyapunovFromData,
    kaplan_yorke_dimension,
    lyapunov_from_data,
    lyapunov_spectrum,
    max_lyapunov,
)
from .orbits import OrbitDiagram, orbit_diagram, poincare_section

__all__ = [
    "DimensionResult",
    "EmbeddingDimension",
    "FixedPoint",
    "LyapunovFromData",
    "OrbitDiagram",
    "approximate_entropy",
    "autocorrelation",
    "box_counting_dimension",
    "cao_dimension",
    "correlation_dimension",
    "correlation_sum",
    "dimension_spectrum",
    "dispersion_entropy",
    "embed",
    "embedding_dimension",
    "entropy",
    "false_nearest_neighbors",
    "fixed_mass_dimension",
    "fixed_points",
    "generalized_dimension",
    "information_dimension",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "lyapunov_spectrum",
    "lz76_complexity",
    "lz76_entropy",
    "max_lyapunov",
    "multiscale_entropy",
    "mutual_information",
    "optimal_delay",
    "orbit_diagram",
    "permutation_entropy",
    "poincare_section",
    "sample_entropy",
    "weighted_permutation_entropy",
]


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree analysis plugins into :data:`tsdynamics.registry.analyses`.

    Walks the ``tsdynamics.analyses`` entry-point group and registers each loaded
    object under its entry-point name (see
    :func:`tsdynamics.plugins.register_entry_points`).  Called once at import;
    safe to re-invoke after installing a plugin.

    Parameters
    ----------
    strict : bool, default False
        Re-raise the first plugin load failure instead of warning and skipping.

    Returns
    -------
    list[str]
        The names newly registered by this call.
    """
    return register_entry_points(_registry.analyses, ANALYSES_GROUP, strict=strict)


# Populate the analyses registry from out-of-tree plugins at import. In-tree
# analyses register themselves from their own subpackages (the analysis streams);
# plugin failures are isolated inside `register_entry_points`.
discover_plugins()
