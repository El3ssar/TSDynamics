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
- :mod:`~tsdynamics.analysis.recurrence` — recurrence plots and RQA:
  :func:`recurrence_matrix` (fixed threshold / target rate, sparse), :func:`rqa`
  (determinism, laminarity, line entropy, trapping time, …) and
  :func:`windowed_rqa` (those measures in a sliding window).
- :mod:`~tsdynamics.analysis.surrogate` — surrogate-data nonlinearity tests:
  :func:`surrogates` (shuffle / FT / AAFT / IAAFT generators), the discriminating
  statistics :func:`time_reversal_asymmetry` / :func:`nonlinear_prediction_error`,
  and :func:`surrogate_test` (rank p-value + significance via a
  :class:`SurrogateTest`).
- :mod:`~tsdynamics.analysis.basins` — attractors & basins: :func:`find_attractors`
  and :func:`basins_of_attraction` (recurrence finder), :func:`basin_fractions`
  (basin stability), :func:`basin_entropy`, :func:`uncertainty_exponent` and
  :func:`wada_property` (boundary structure), :func:`continuation` /
  :func:`tipping_points` (global continuation) and :func:`resilience`.

Out-of-tree analyses register through the ``tsdynamics.analyses`` entry-point
group (see :mod:`tsdynamics.plugins`); :func:`discover_plugins` loads them into
:data:`tsdynamics.registry.analyses`.
"""

from .. import registry as _registry
from ..plugins import ANALYSES_GROUP, register_entry_points
from .basins import (
    Attractor,
    AttractorSet,
    BasinEntropy,
    BasinFractions,
    BasinsResult,
    ContinuationResult,
    UncertaintyExponent,
    WadaResult,
    basin_entropy,
    basin_fractions,
    basins_of_attraction,
    continuation,
    find_attractors,
    resilience,
    tipping_points,
    uncertainty_exponent,
    wada_property,
)
from .chaos import (
    ExpansionEntropyResult,
    GALIResult,
    expansion_entropy,
    gali,
    zero_one_test,
)
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
from .fixedpoints import (
    FixedPoint,
    PeriodicOrbit,
    estimate_period,
    fixed_points,
    periodic_orbit,
    periodic_orbits,
)
from .lyapunov import (
    LyapunovFromData,
    kaplan_yorke_dimension,
    lyapunov_from_data,
    lyapunov_spectrum,
    max_lyapunov,
)
from .orbits import OrbitDiagram, ReturnMap, orbit_diagram, poincare_section, return_map
from .recurrence import (
    RecurrenceMatrix,
    RQAResult,
    WindowedRQA,
    recurrence_matrix,
    rqa,
    windowed_rqa,
)
from .surrogate import (
    SurrogateTest,
    aaft_surrogate,
    fourier_surrogate,
    iaaft_surrogate,
    nonlinear_prediction_error,
    random_shuffle,
    surrogate_test,
    surrogates,
    time_reversal_asymmetry,
)

__all__ = [
    "Attractor",
    "AttractorSet",
    "BasinEntropy",
    "BasinFractions",
    "BasinsResult",
    "ContinuationResult",
    "DimensionResult",
    "EmbeddingDimension",
    "ExpansionEntropyResult",
    "FixedPoint",
    "GALIResult",
    "LyapunovFromData",
    "OrbitDiagram",
    "PeriodicOrbit",
    "RQAResult",
    "RecurrenceMatrix",
    "ReturnMap",
    "SurrogateTest",
    "UncertaintyExponent",
    "WadaResult",
    "WindowedRQA",
    "aaft_surrogate",
    "approximate_entropy",
    "autocorrelation",
    "basin_entropy",
    "basin_fractions",
    "basins_of_attraction",
    "box_counting_dimension",
    "cao_dimension",
    "continuation",
    "correlation_dimension",
    "correlation_sum",
    "dimension_spectrum",
    "dispersion_entropy",
    "embed",
    "embedding_dimension",
    "entropy",
    "estimate_period",
    "expansion_entropy",
    "false_nearest_neighbors",
    "find_attractors",
    "fixed_mass_dimension",
    "fixed_points",
    "fourier_surrogate",
    "gali",
    "generalized_dimension",
    "iaaft_surrogate",
    "information_dimension",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "lyapunov_spectrum",
    "lz76_complexity",
    "lz76_entropy",
    "max_lyapunov",
    "multiscale_entropy",
    "mutual_information",
    "nonlinear_prediction_error",
    "optimal_delay",
    "orbit_diagram",
    "periodic_orbit",
    "periodic_orbits",
    "permutation_entropy",
    "poincare_section",
    "random_shuffle",
    "recurrence_matrix",
    "resilience",
    "return_map",
    "rqa",
    "sample_entropy",
    "surrogate_test",
    "surrogates",
    "time_reversal_asymmetry",
    "tipping_points",
    "uncertainty_exponent",
    "wada_property",
    "weighted_permutation_entropy",
    "windowed_rqa",
    "zero_one_test",
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
