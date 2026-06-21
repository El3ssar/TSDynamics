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

Curated top level (stream WS-NAMESPACE)
---------------------------------------
The top-level ``__all__`` is **curated** to ~30 headline names — the family
bases, the derived wrappers, :class:`Trajectory`, the six headline analyses, and
the navigable submodules — so ``tsdynamics.<TAB>`` shows the mental model, not a
flat dump of ~100 functions.  Everything else stays **fully reachable** at its
qualified path (``ts.analysis.dimensions.correlation_dimension``) *and* as a flat
re-export (``from tsdynamics import correlation_dimension`` and
``ts.correlation_dimension`` both still resolve) — the flat names simply drop out
of ``__all__`` / autocomplete.  The six promoted analyses are
:func:`lyapunov_spectrum`, :func:`bifurcation_diagram` (the discoverable spelling
of :func:`orbit_diagram`), :func:`poincare_section`, :func:`recurrence_matrix`,
:func:`basins` (short alias of :func:`basins_of_attraction`) and
:func:`fixed_points`.

Reachable submodules (bound on the top-level namespace, so they show up in
``tsdynamics.<TAB>``):

- :mod:`~tsdynamics.transforms` — signal/feature transforms that feed the
  analysis layer (``ts.transforms.power_spectral_density``); a headline user
  capability.
- :mod:`~tsdynamics.errors` — the :class:`~tsdynamics.errors.TSDynamicsError`
  hierarchy that public entry points raise.
- :mod:`~tsdynamics.viz` — the backend-agnostic ``PlotSpec`` IR (no renderer
  ships yet).  Resolved **lazily** (via ``__getattr__``) so a plain
  ``import tsdynamics`` pulls in no plotting machinery.
- :mod:`~tsdynamics.engine`, :mod:`~tsdynamics.solvers` — advanced/internal: the
  Rust-facing compile/run seam and the solver registry.  Reachable for inspection
  but rarely imported directly.

Canonical homes for the data primitives (each has exactly one defining module;
the rest are convenience re-exports): :class:`Trajectory`, :class:`Box`,
:class:`Ball`, :class:`Grid` live in :mod:`tsdynamics.data`; :class:`WrappedSystem`
lives with the family bases in :mod:`tsdynamics.families`.
"""

from typing import Any

from . import (
    analysis,
    data,
    derived,
    engine,
    errors,
    families,
    registry,
    solvers,
    systems,
    transforms,
    utils,
)
from .analysis import (
    Attractor as Attractor,
)
from .analysis import (
    AttractorSet as AttractorSet,
)
from .analysis import (
    BasinEntropy as BasinEntropy,
)
from .analysis import (
    BasinFractions as BasinFractions,
)
from .analysis import (
    BasinsResult as BasinsResult,
)
from .analysis import (
    ContinuationResult as ContinuationResult,
)
from .analysis import (
    DimensionResult as DimensionResult,
)
from .analysis import (
    EmbeddingDimension as EmbeddingDimension,
)
from .analysis import (
    ExpansionEntropyResult as ExpansionEntropyResult,
)
from .analysis import (
    FixedPoint as FixedPoint,
)
from .analysis import (
    GALIResult as GALIResult,
)
from .analysis import (
    LyapunovFromData as LyapunovFromData,
)
from .analysis import (
    OrbitDiagram as OrbitDiagram,
)
from .analysis import (
    PeriodicOrbit as PeriodicOrbit,
)
from .analysis import (
    PoincareSection as PoincareSection,
)
from .analysis import (
    RecurrenceMatrix as RecurrenceMatrix,
)
from .analysis import (
    ReturnMap as ReturnMap,
)
from .analysis import (
    RQAResult as RQAResult,
)
from .analysis import (
    SurrogateTest as SurrogateTest,
)
from .analysis import (
    UncertaintyExponent as UncertaintyExponent,
)
from .analysis import (
    WadaResult as WadaResult,
)
from .analysis import (
    WindowedRQA as WindowedRQA,
)
from .analysis import (
    aaft_surrogate as aaft_surrogate,
)
from .analysis import (
    approximate_entropy as approximate_entropy,
)
from .analysis import (
    autocorrelation as autocorrelation,
)
from .analysis import (
    basin_entropy as basin_entropy,
)
from .analysis import (
    basin_fractions as basin_fractions,
)
from .analysis import (
    basins_of_attraction as basins_of_attraction,
)
from .analysis import (
    box_counting_dimension as box_counting_dimension,
)
from .analysis import (
    cao_dimension as cao_dimension,
)
from .analysis import (
    continuation as continuation,
)
from .analysis import (
    correlation_dimension as correlation_dimension,
)
from .analysis import (
    correlation_sum as correlation_sum,
)
from .analysis import (
    dimension_spectrum as dimension_spectrum,
)
from .analysis import (
    dispersion_entropy as dispersion_entropy,
)
from .analysis import (
    embed as embed,
)
from .analysis import (
    embedding_dimension as embedding_dimension,
)
from .analysis import (
    entropy as entropy,
)
from .analysis import (
    estimate_period as estimate_period,
)
from .analysis import (
    expansion_entropy as expansion_entropy,
)
from .analysis import (
    false_nearest_neighbors as false_nearest_neighbors,
)
from .analysis import (
    find_attractors as find_attractors,
)
from .analysis import (
    fixed_mass_dimension as fixed_mass_dimension,
)
from .analysis import (
    fixed_points as fixed_points,
)
from .analysis import (
    fourier_surrogate as fourier_surrogate,
)
from .analysis import (
    gali as gali,
)
from .analysis import (
    generalized_dimension as generalized_dimension,
)
from .analysis import (
    iaaft_surrogate as iaaft_surrogate,
)
from .analysis import (
    information_dimension as information_dimension,
)
from .analysis import (
    kaplan_yorke_dimension as kaplan_yorke_dimension,
)
from .analysis import (
    lyapunov_from_data as lyapunov_from_data,
)
from .analysis import (
    lyapunov_spectrum as lyapunov_spectrum,
)
from .analysis import (
    lz76_complexity as lz76_complexity,
)
from .analysis import (
    lz76_entropy as lz76_entropy,
)
from .analysis import (
    max_lyapunov as max_lyapunov,
)
from .analysis import (
    multiscale_entropy as multiscale_entropy,
)
from .analysis import (
    mutual_information as mutual_information,
)
from .analysis import (
    nonlinear_prediction_error as nonlinear_prediction_error,
)
from .analysis import (
    optimal_delay as optimal_delay,
)
from .analysis import (
    orbit_diagram as orbit_diagram,
)
from .analysis import (
    periodic_orbit as periodic_orbit,
)
from .analysis import (
    periodic_orbits as periodic_orbits,
)
from .analysis import (
    permutation_entropy as permutation_entropy,
)
from .analysis import (
    poincare_section as poincare_section,
)
from .analysis import (
    random_shuffle as random_shuffle,
)
from .analysis import (
    recurrence_matrix as recurrence_matrix,
)
from .analysis import (
    resilience as resilience,
)
from .analysis import (
    return_map as return_map,
)
from .analysis import (
    rqa as rqa,
)
from .analysis import (
    sample_entropy as sample_entropy,
)
from .analysis import (
    surrogate_test as surrogate_test,
)
from .analysis import (
    surrogates as surrogates,
)
from .analysis import (
    time_reversal_asymmetry as time_reversal_asymmetry,
)
from .analysis import (
    tipping_points as tipping_points,
)
from .analysis import (
    uncertainty_exponent as uncertainty_exponent,
)
from .analysis import (
    wada_property as wada_property,
)
from .analysis import (
    weighted_permutation_entropy as weighted_permutation_entropy,
)
from .analysis import (
    windowed_rqa as windowed_rqa,
)
from .analysis import (
    zero_one_test as zero_one_test,
)
from .data import (
    Ball as Ball,
)
from .data import (
    Box as Box,
)
from .data import (
    Grid as Grid,
)
from .data import (
    grid_points as grid_points,
)
from .data import (
    sampler as sampler,
)
from .data import (
    set_distance as set_distance,
)
from .derived import (
    EnsembleSystem,
    PoincareMap,
    ProjectedSystem,
    StroboscopicMap,
    TangentSystem,
)
from .families import (
    ContinuousSystem,
    DelaySystem,
    DiscreteMap,
    StochasticSystem,
    Trajectory,
    WrappedSystem,
)

# Headline analysis aliases promoted to the curated top level (stream
# WS-NAMESPACE). The canonical implementations keep their original names (still
# flat re-exported and reachable); these are the discoverable headline spellings
# advertised in ``__all__``.
bifurcation_diagram = orbit_diagram  #: discoverable spelling of :func:`orbit_diagram`
basins = basins_of_attraction  #: short alias of :func:`basins_of_attraction`

# Single source of truth for the package version; rewritten by python-semantic-release.
__version__ = "3.1.2"

# Built-in system classes are NOT bound into this namespace — that would bury the
# submodules (``analysis``, ``data``, ``systems``, …) under ~149 model names in
# ``dir()`` / autocomplete. The canonical path is ``tsdynamics.systems.<Name>``
# (e.g. ``tsdynamics.systems.Lorenz``). For backwards compatibility, ``tsd.Lorenz``
# and ``from tsdynamics import Lorenz`` still resolve, lazily, via ``__getattr__``
# below — but the names stay out of ``__all__`` and ``__dir__`` so they don't clutter
# the top-level surface.
_SYSTEM_NAMES = frozenset(systems._SYSTEM_NAMES)

# The curated top-level surface (~30 names). Demoted analysis functions / result
# classes / state-space primitives stay fully reachable (flat re-exported above
# and resolvable as ``ts.<name>``); they are simply no longer advertised in
# ``__all__`` / autocomplete. Reach them at their qualified path —
# ``ts.analysis.dimensions.correlation_dimension`` — or by flat re-export —
# ``from tsdynamics import correlation_dimension``.
__all__ = [
    "__version__",
    # User-facing base classes (subclass these to define a new system)
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "StochasticSystem",
    "WrappedSystem",
    # Trajectory — the lingua franca every family produces
    "Trajectory",
    # Derived-system wrappers (composition layer)
    "EnsembleSystem",
    "PoincareMap",
    "ProjectedSystem",
    "StroboscopicMap",
    "TangentSystem",
    # Headline analyses (the six a newcomer reaches for; the rest live under
    # ``ts.analysis.*`` and stay flat-re-exported for back-compat)
    "lyapunov_spectrum",
    "bifurcation_diagram",
    "poincare_section",
    "recurrence_matrix",
    "basins",
    "fixed_points",
    # Navigable submodules (the depth lives here, scipy-style)
    "analysis",
    "transforms",
    "data",
    "derived",
    "families",
    "registry",
    "systems",
    "utils",
    "errors",
    # Backend-agnostic viz IR — resolved lazily (see ``__getattr__``).
    "viz",
    # Advanced / internal submodules (reachable but rarely imported directly).
    "engine",
    "solvers",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the ``viz`` submodule and built-in system classes.

    Two lazy resolutions live here:

    * ``tsdynamics.viz`` — imported on first access (and cached) so a plain
      ``import tsdynamics`` pulls in no plotting/IR machinery; ``viz`` still shows
      in ``__all__`` / ``dir()`` for discoverability.
    * ``tsdynamics.Lorenz`` and friends — the ~149 built-in system classes,
      resolved from :mod:`tsdynamics.systems` (the canonical path) instead of
      binding all of them into the namespace, which keeps ``dir()`` / autocomplete
      focused on the curated public API.
    """
    if name == "viz":
        import importlib

        # import_module loads the submodule through the import machinery without
        # re-entering this __getattr__ (a plain ``from . import viz`` would recurse).
        _viz = importlib.import_module(f"{__name__}.viz")
        globals()["viz"] = _viz  # cache: subsequent access skips __getattr__
        return _viz
    if name in _SYSTEM_NAMES:
        return getattr(systems, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Top-level surface = the public API in ``__all__`` (models live under ``systems``)."""
    return sorted(__all__)
