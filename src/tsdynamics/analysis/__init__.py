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

# Bind the capability subpackages as public sub-namespaces so ``ts.analysis.<TAB>``
# surfaces the ~10 categories (scipy-style), each listing its own estimators,
# instead of one flat dump of ~75 functions.  ``entropy`` is intentionally absent
# from this list: that name is shadowed by the :func:`entropy` *function* (a
# documented, griffe-safe collision), so ``ts.analysis.entropy`` is the function,
# not the module.  Reach the entropy estimators by name —
# ``from tsdynamics.analysis.entropy import permutation_entropy`` — or grab the
# module with ``importlib.import_module("tsdynamics.analysis.entropy")``.  The flat
# re-exports above are retained, so ``from tsdynamics.analysis import
# correlation_dimension`` still works.
from . import (
    basins,
    chaos,
    dimensions,
    embedding,
    fixedpoints,
    lyapunov,
    orbits,
    recurrence,
    surrogate,
)

# The shared result-object model (stream WS-RESULT/WS-SCALING/WS-WRAP): every
# analysis returns an :class:`AnalysisResult` subclass, never a bare value.
from ._result import (
    AnalysisResult,
    ArrayResult,
    CollectionResult,
    CountResult,
    ScalarResult,
    ScalingResult,
    VisualizationNotInstalled,
)
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
    Embedding,
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
    FixedPointSet,
    OrbitSet,
    PeriodicOrbit,
    estimate_period,
    fixed_points,
    periodic_orbit,
    periodic_orbits,
)
from .lyapunov import (
    LyapunovFromData,
    LyapunovSpectrum,
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
    SurrogateEnsemble,
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

#: The capability subpackages, in canonical order — the decluttered
#: ``ts.analysis.<TAB>`` surface (see :func:`__dir__`).  ``entropy`` resolves to
#: the function of the same name; reach the entropy estimators via
#: ``from tsdynamics.analysis.entropy import …`` or ``importlib.import_module``.
_CATEGORY_SUBPACKAGES = (
    "lyapunov",
    "dimensions",
    "chaos",
    "recurrence",
    "entropy",
    "embedding",
    "surrogate",
    "orbits",
    "fixedpoints",
    "basins",
)

__all__ = [
    "AnalysisResult",
    "ArrayResult",
    "Attractor",
    "AttractorSet",
    "BasinEntropy",
    "BasinFractions",
    "BasinsResult",
    "CollectionResult",
    "ContinuationResult",
    "CountResult",
    "DimensionResult",
    "Embedding",
    "EmbeddingDimension",
    "ExpansionEntropyResult",
    "FixedPoint",
    "FixedPointSet",
    "GALIResult",
    "LyapunovFromData",
    "LyapunovSpectrum",
    "OrbitDiagram",
    "OrbitSet",
    "PeriodicOrbit",
    "RQAResult",
    "RecurrenceMatrix",
    "ReturnMap",
    "ScalarResult",
    "ScalingResult",
    "SurrogateEnsemble",
    "SurrogateTest",
    "UncertaintyExponent",
    "VisualizationNotInstalled",
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
    "discover_plugins",
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
    # Capability subpackages (the navigable categories; ``entropy`` is the
    # function-shadowed name already listed above).
    "basins",
    "chaos",
    "dimensions",
    "embedding",
    "fixedpoints",
    "lyapunov",
    "orbits",
    "recurrence",
    "surrogate",
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


def __dir__() -> list[str]:
    """Show the ~10 capability categories to ``dir()`` / autocomplete (scipy-style).

    ``ts.analysis.<TAB>`` surfaces the navigable category subpackages
    (``lyapunov``, ``dimensions``, ``chaos``, …) plus :func:`discover_plugins`,
    not the ~75 flat quantifier names.  Those flat names stay fully reachable —
    ``from tsdynamics.analysis import correlation_dimension`` and
    ``ts.analysis.correlation_dimension`` both resolve, and ``__all__`` still
    carries them for ``from tsdynamics.analysis import *`` — they are simply kept
    off the tab surface so the structure, not the dump, is what you see.  Drill in
    with ``ts.analysis.dimensions.<TAB>`` to reach the estimators.
    """
    return sorted((*_CATEGORY_SUBPACKAGES, "discover_plugins"))
