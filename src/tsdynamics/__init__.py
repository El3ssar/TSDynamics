"""
TSDynamics — compiled dynamical systems: integration, iteration, and chaos analysis.

Quick start
-----------
>>> from tsdynamics import Lorenz, MackeyGlass, Henon
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

The top-level namespace re-exports every built-in system (see
:mod:`tsdynamics.registry` for programmatic access), the three base classes
users subclass to define new systems, the derived-system wrappers, and the
analysis functions.  Internal helpers (``ParamSet``, ``SystemBase``,
``staticjit``) live under ``tsdynamics.families`` / ``tsdynamics.utils``.
"""

from . import analysis, data, derived, families, registry, systems, utils
from .analysis import (
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
    WindowedRQA,
    aaft_surrogate,
    approximate_entropy,
    autocorrelation,
    box_counting_dimension,
    cao_dimension,
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
    return_map,
    rqa,
    sample_entropy,
    surrogate_test,
    surrogates,
    time_reversal_asymmetry,
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
from .systems import continuous as _continuous
from .systems import discrete as _discrete

# Single source of truth for the package version; rewritten by python-semantic-release.
__version__ = "2.5.0"

# Re-export every system class at the top level so ``from tsdynamics import Lorenz``
# works without users having to remember which submodule a system lives in.
_continuous_names = list(_continuous.__all__)
_discrete_names = list(_discrete.__all__)
for _name in _continuous_names:
    globals()[_name] = getattr(_continuous, _name)
for _name in _discrete_names:
    globals()[_name] = getattr(_discrete, _name)
del _name

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
    # All built-in systems
    *_continuous_names,
    *_discrete_names,
]
