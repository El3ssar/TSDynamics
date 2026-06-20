"""
TSDynamics — compiled dynamical systems: integration, iteration, and chaos analysis.

Quick start
-----------
>>> from tsdynamics.systems import Lorenz, MackeyGlass, Henon
>>> traj = Lorenz().integrate(final_time=100.0, dt=0.01)
>>> traj.t.shape, traj.y.shape
((10001,), (10001, 3))
>>> traj["x"]                          # named component access
>>> Lorenz().lyapunov_spectrum()       # ≈ [0.91, 0, -14.57]

Beyond integration, the :mod:`~tsdynamics.derived` wrappers re-present any
system through a new lens (Poincaré map, stroboscopic map, tangent dynamics,
ensembles), and :mod:`~tsdynamics.analysis` provides the quantifiers that
consume them (orbit/bifurcation diagrams, Poincaré sections, Lyapunov tools,
fixed points).

Built-in systems live under :mod:`tsdynamics.systems` (``tsdynamics.systems.Lorenz``)
so the top-level namespace stays focused on the base classes, the derived-system
wrappers, the analysis functions, and the submodules.  For backwards
compatibility ``tsdynamics.Lorenz`` (and ``from tsdynamics import Lorenz``) still
resolve lazily.  See :mod:`tsdynamics.registry` for programmatic access.  Internal
helpers (``ParamSet``, ``SystemBase``, ``staticjit``) live under
``tsdynamics.families`` / ``tsdynamics.utils``.
"""

from . import analysis, data, derived, families, registry, systems, utils
from .analysis import (
    Attractor,
    AttractorSet,
    BasinEntropy,
    BasinFractions,
    BasinsResult,
    ContinuationResult,
    DimensionResult,
    EmbeddingDimension,
    ExpansionEntropyResult,
    FixedPoint,
    GALIResult,
    LyapunovFromData,
    OrbitDiagram,
    PeriodicOrbit,
    RecurrenceMatrix,
    ReturnMap,
    RQAResult,
    SurrogateTest,
    UncertaintyExponent,
    WadaResult,
    WindowedRQA,
    aaft_surrogate,
    approximate_entropy,
    autocorrelation,
    basin_entropy,
    basin_fractions,
    basins_of_attraction,
    box_counting_dimension,
    cao_dimension,
    continuation,
    correlation_dimension,
    correlation_sum,
    dimension_spectrum,
    dispersion_entropy,
    embed,
    embedding_dimension,
    entropy,
    estimate_period,
    expansion_entropy,
    false_nearest_neighbors,
    find_attractors,
    fixed_mass_dimension,
    fixed_points,
    fourier_surrogate,
    gali,
    generalized_dimension,
    iaaft_surrogate,
    information_dimension,
    kaplan_yorke_dimension,
    lyapunov_from_data,
    lyapunov_spectrum,
    lz76_complexity,
    lz76_entropy,
    max_lyapunov,
    multiscale_entropy,
    mutual_information,
    nonlinear_prediction_error,
    optimal_delay,
    orbit_diagram,
    periodic_orbit,
    periodic_orbits,
    permutation_entropy,
    poincare_section,
    random_shuffle,
    recurrence_matrix,
    resilience,
    return_map,
    rqa,
    sample_entropy,
    surrogate_test,
    surrogates,
    time_reversal_asymmetry,
    tipping_points,
    uncertainty_exponent,
    wada_property,
    weighted_permutation_entropy,
    windowed_rqa,
    zero_one_test,
)
from .data import Ball, Box, Grid, grid_points, sampler, set_distance
from .derived import (
    EnsembleSystem,
    PoincareMap,
    ProjectedSystem,
    StroboscopicMap,
    TangentSystem,
    WrappedSystem,
)
from .families import (
    ContinuousSystem,
    DelaySystem,
    DiscreteMap,
    StochasticSystem,
    Trajectory,
)

# Single source of truth for the package version; rewritten by python-semantic-release.
__version__ = "3.0.3"

# Built-in system classes are NOT bound into this namespace — that would bury the
# submodules (``analysis``, ``data``, ``systems``, …) under ~149 model names in
# ``dir()`` / autocomplete. The canonical path is ``tsdynamics.systems.<Name>``
# (e.g. ``tsdynamics.systems.Lorenz``). For backwards compatibility, ``tsd.Lorenz``
# and ``from tsdynamics import Lorenz`` still resolve, lazily, via ``__getattr__``
# below — but the names stay out of ``__all__`` and ``__dir__`` so they don't clutter
# the top-level surface.
_SYSTEM_NAMES = frozenset(systems._SYSTEM_NAMES)

__all__ = [
    "__version__",
    # User-facing base classes (subclass these to define a new system)
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "StochasticSystem",
    "Trajectory",
    # Derived-system wrappers (composition layer)
    "EnsembleSystem",
    "PoincareMap",
    "ProjectedSystem",
    "StroboscopicMap",
    "TangentSystem",
    "WrappedSystem",
    # Analysis toolkit
    "DimensionResult",
    "EmbeddingDimension",
    "FixedPoint",
    "LyapunovFromData",
    "OrbitDiagram",
    "PeriodicOrbit",
    "ReturnMap",
    "box_counting_dimension",
    "correlation_dimension",
    "correlation_sum",
    "dimension_spectrum",
    "estimate_period",
    "fixed_mass_dimension",
    "fixed_points",
    "generalized_dimension",
    "information_dimension",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "lyapunov_spectrum",
    "max_lyapunov",
    "orbit_diagram",
    "periodic_orbit",
    "periodic_orbits",
    "poincare_section",
    "return_map",
    # Delay embeddings
    "autocorrelation",
    "cao_dimension",
    "embed",
    "embedding_dimension",
    "false_nearest_neighbors",
    "mutual_information",
    "optimal_delay",
    # Chaos indicators
    "ExpansionEntropyResult",
    "GALIResult",
    "expansion_entropy",
    "gali",
    "zero_one_test",
    # Recurrence & RQA
    "RQAResult",
    "RecurrenceMatrix",
    "WindowedRQA",
    "recurrence_matrix",
    "rqa",
    "windowed_rqa",
    # Surrogates & nonlinearity tests
    "SurrogateTest",
    "aaft_surrogate",
    "fourier_surrogate",
    "iaaft_surrogate",
    "nonlinear_prediction_error",
    "random_shuffle",
    "surrogate_test",
    "surrogates",
    "time_reversal_asymmetry",
    # Entropy & complexity
    "approximate_entropy",
    "dispersion_entropy",
    "entropy",
    "lz76_complexity",
    "lz76_entropy",
    "multiscale_entropy",
    "permutation_entropy",
    "sample_entropy",
    "weighted_permutation_entropy",
    # Attractors & basins (stream A-BASIN)
    "Attractor",
    "AttractorSet",
    "BasinEntropy",
    "BasinFractions",
    "BasinsResult",
    "ContinuationResult",
    "UncertaintyExponent",
    "WadaResult",
    "basin_entropy",
    "basin_fractions",
    "basins_of_attraction",
    "continuation",
    "find_attractors",
    "resilience",
    "tipping_points",
    "uncertainty_exponent",
    "wada_property",
    # State-space geometry (regions, samplers, set distances)
    "Ball",
    "Box",
    "Grid",
    "grid_points",
    "sampler",
    "set_distance",
    # Sub-namespaces (for ``tsdynamics.systems.continuous.chaotic_attractors`` etc.)
    "analysis",
    "data",
    "derived",
    "families",
    "registry",
    "systems",
    "utils",
]


def __getattr__(name: str):
    """Lazily resolve a built-in system class (``tsdynamics.Lorenz``).

    Kept for backwards compatibility — the canonical path is
    ``tsdynamics.systems.<Name>``. Resolving here (instead of binding all ~149
    classes into the namespace) keeps ``dir()`` / autocomplete focused on the
    submodules and public API.
    """
    if name in _SYSTEM_NAMES:
        return getattr(systems, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Top-level surface = the public API in ``__all__`` (models live under ``systems``)."""
    return sorted(__all__)
