"""
Analysis toolkit — quantifiers that consume any :class:`~tsdynamics.families.System`.

Each capability cluster lives in its own subpackage (one per analysis stream),
re-exported here so the public surface is flat:
``from tsdynamics import lyapunov_spectrum`` and
``from tsdynamics.analysis import lyapunov_spectrum`` both work.

- :mod:`~tsdynamics.analysis.orbits` — :func:`bifurcation_diagram` (parameter
  sweeps; a raw flow is reduced to its successive-maxima map, or to the
  ``section=`` you name, and ``orbit_diagram`` is the same function under its
  map-centric name) and :func:`poincare_section` (surfaces of section).
- :mod:`~tsdynamics.analysis.lyapunov` — :func:`lyapunov_spectrum` /
  :func:`max_lyapunov` / :func:`kaplan_yorke_dimension`.
- :mod:`~tsdynamics.analysis.fixedpoints` — :func:`fixed_points`, multi-start
  Newton fixed-point finding for maps with linear stability.
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
- :mod:`~tsdynamics.analysis.basins` — attractors & basins: :func:`find_attractors`
  and :func:`basins_of_attraction` (recurrence finder), :func:`basin_fractions`
  (basin stability), :func:`basin_entropy`, :func:`uncertainty_exponent` and
  :func:`wada_property` (boundary structure), :func:`continuation` /
  :func:`tipping_points` (global continuation) and :func:`resilience`.
- :mod:`~tsdynamics.analysis.planar` — the planar (2-D slice) toolkit:
  :func:`~tsdynamics.analysis.planar.nullclines`,
  :func:`~tsdynamics.analysis.planar.flow_field` /
  :func:`~tsdynamics.analysis.planar.streamlines`,
  :func:`~tsdynamics.analysis.planar.ftle_field`,
  :func:`~tsdynamics.analysis.planar.escape_time_field` /
  :func:`~tsdynamics.analysis.planar.transient_time_field`,
  :func:`~tsdynamics.analysis.planar.invariant_density` and the
  trace–determinant classification
  (:func:`~tsdynamics.analysis.planar.trace_determinant`,
  :func:`~tsdynamics.analysis.planar.classify_linear`).
- :mod:`~tsdynamics.analysis.sampling` — sagitta-based sampling tools:
  :func:`estimate_dt_from_sagitta` (choose an output ``dt`` for a trajectory) and
  :func:`sagitta_profile` (the per-point bow off the local chord, e.g. a
  ``color_by="sagitta"`` field).

Out-of-tree analyses register through the ``tsdynamics.analyses`` entry-point
group (see :mod:`tsdynamics.plugins`); :func:`discover_plugins` loads them into
:data:`tsdynamics.registry.analyses`.
"""

from .. import registry as _registry
from ..plugins import ANALYSES_GROUP, register_entry_points

# Bind the capability subpackages as public sub-namespaces so ``ts.analysis.<TAB>``
# surfaces the capability categories (scipy-style), each listing its own
# estimators, instead of one flat dump of quantifier names.  The flat re-exports
# below are retained, so ``from tsdynamics.analysis import correlation_dimension``
# still works.
from . import (
    basins,
    chaos,
    dimensions,
    embedding,
    fixedpoints,
    lyapunov,
    orbits,
    planar,
    recurrence,
    sampling,
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
from .orbits import (
    OrbitDiagram,
    PoincareSection,
    ReturnMap,
    bifurcation_diagram,
    orbit_diagram,
    poincare_section,
    return_map,
)
from .recurrence import (
    RecurrenceMatrix,
    RQAResult,
    WindowedRQA,
    recurrence_matrix,
    rqa,
    windowed_rqa,
)
from .sampling import estimate_dt_from_sagitta, sagitta_profile

#: The capability subpackages, in canonical order — half of the decluttered
#: ``ts.analysis.<TAB>`` surface (see :func:`__dir__`).
_CATEGORY_SUBPACKAGES = (
    "lyapunov",
    "dimensions",
    "chaos",
    "recurrence",
    "embedding",
    "orbits",
    "fixedpoints",
    "basins",
    "planar",
    "sampling",
)

#: The headline quantifiers shown alongside the categories, so
#: ``ts.analysis.<TAB>`` answers "what can I *do*?" as well as "what is in here?".
#: One per capability cluster — the function a newcomer reaches for first.  The
#: other ~60 flat re-exports stay importable (``from tsdynamics.analysis import
#: correlation_sum``) and stay in ``__all__``; they are just not on the tab
#: surface.  Drill in with ``ts.analysis.dimensions.<TAB>`` for the rest.
_HEADLINE_ANALYSES = (
    "lyapunov_spectrum",
    "correlation_dimension",
    "gali",
    "recurrence_matrix",
    "embed",
    # The headline spelling is the flow one (``bifurcation_diagram``); the
    # map-centric ``orbit_diagram`` is the SAME function object and stays
    # exported, just off the tab surface — mirroring the top level, so a name is
    # not primary in one namespace and absent from the other.
    "bifurcation_diagram",
    "poincare_section",
    "fixed_points",
    "basins_of_attraction",
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
    "PoincareSection",
    "RQAResult",
    "RecurrenceMatrix",
    "ReturnMap",
    "ScalarResult",
    "ScalingResult",
    "UncertaintyExponent",
    "VisualizationNotInstalled",
    "WadaResult",
    "WindowedRQA",
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
    "embed",
    "embedding_dimension",
    "estimate_dt_from_sagitta",
    "estimate_period",
    "expansion_entropy",
    "false_nearest_neighbors",
    "find_attractors",
    "fixed_mass_dimension",
    "fixed_points",
    "gali",
    "generalized_dimension",
    "information_dimension",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "lyapunov_spectrum",
    "max_lyapunov",
    "mutual_information",
    "optimal_delay",
    "orbit_diagram",
    "bifurcation_diagram",
    "periodic_orbit",
    "periodic_orbits",
    "poincare_section",
    "recurrence_matrix",
    "resilience",
    "return_map",
    "rqa",
    "sagitta_profile",
    "tipping_points",
    "uncertainty_exponent",
    "wada_property",
    "windowed_rqa",
    "zero_one_test",
    # Capability subpackages (the navigable categories).
    "basins",
    "chaos",
    "dimensions",
    "embedding",
    "fixedpoints",
    "lyapunov",
    "orbits",
    "planar",
    "recurrence",
    "sampling",
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
    """Show the capability categories + the headline quantifiers (scipy-style).

    ``ts.analysis.<TAB>`` surfaces the navigable category subpackages
    (``lyapunov``, ``dimensions``, ``chaos``, …) **and** one headline quantifier
    per category (:func:`lyapunov_spectrum`, :func:`correlation_dimension`,
    :func:`recurrence_matrix`, …) — 18 names, not a flat dump of ~75.

    Everything else stays fully reachable: ``from tsdynamics.analysis import
    correlation_sum`` and ``ts.analysis.correlation_sum`` both resolve, and
    ``__all__`` still carries them for ``from tsdynamics.analysis import *``.
    They are simply kept off the tab surface so the structure, not the dump, is
    what you see — drill in with ``ts.analysis.dimensions.<TAB>``.

    :func:`discover_plugins` is deliberately absent: loading entry points is
    packaging machinery, not something a user of the analysis toolkit types.
    """
    return sorted((*_CATEGORY_SUBPACKAGES, *_HEADLINE_ANALYSES))
