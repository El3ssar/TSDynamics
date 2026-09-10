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
helpers (``ParamSet``, ``SystemBase``) live under ``tsdynamics.families``.

Curated top level
-----------------
``tsdynamics.<TAB>`` shows **only what you would use**: the family bases, the
derived wrappers, :class:`Trajectory`, the state-space regions, the six headline
analyses, the plotting front door, and the four submodules worth typing a dot
after (:mod:`~tsdynamics.systems`, :mod:`~tsdynamics.analysis`,
:mod:`~tsdynamics.viz`, :mod:`~tsdynamics.errors`).

Everything else stays **fully reachable** — it is dropped from ``__all__`` /
autocomplete, never removed:

- the demoted analysis functions and result classes —
  ``ts.correlation_dimension``, ``from tsdynamics import correlation_dimension``
  and ``ts.analysis.dimensions.correlation_dimension`` all resolve;
- the **machinery submodules** — :mod:`~tsdynamics.engine` (the Rust-facing
  compile/run seam), :mod:`~tsdynamics.solvers` (the solver registry),
  :mod:`~tsdynamics.registry` (the system/analysis/renderer registries),
  :mod:`~tsdynamics.families` (``SystemBase`` / ``ParamSet`` / the ``System``
  protocol), :mod:`~tsdynamics.utils` (the shared grid + tolerance constants).
  ``ts.engine`` still resolves and ``from tsdynamics.engine import run`` still
  imports; they are simply not what a newcomer should be reading first.
- :mod:`~tsdynamics.data` and :mod:`~tsdynamics.derived` — every name a user
  needs from them is already *on* the top level (:class:`Trajectory`,
  :class:`Box`, :class:`Ball`, :class:`Grid`; :class:`PoincareMap` and the other
  wrappers), so the extra hop earns no tab slot.

The six promoted analyses are :func:`lyapunov_spectrum`,
:func:`bifurcation_diagram` (the discoverable spelling of :func:`orbit_diagram`),
:func:`poincare_section`, :func:`recurrence_matrix`, :func:`basins` (short alias
of :func:`basins_of_attraction`) and :func:`fixed_points`.  :class:`Box` /
:class:`Ball` / :class:`Grid` are promoted alongside them because
``ts.basins(system, region)`` cannot be called without one.

:mod:`~tsdynamics.viz` (and the :func:`plot` / :func:`T` front door) resolves
**lazily** via ``__getattr__``, so a plain ``import tsdynamics`` pulls in no
plotting machinery.

Canonical homes for the data primitives (each has exactly one defining module;
the rest are convenience re-exports): :class:`Trajectory`, :class:`Box`,
:class:`Ball`, :class:`Grid` live in :mod:`tsdynamics.data`; :class:`WrappedSystem`
lives with the family bases in :mod:`tsdynamics.families`.
"""

from typing import Any

from . import (
    analysis,
    errors,
    systems,
)

# Machinery submodules: bound eagerly so ``ts.engine`` / ``ts.registry`` resolve
# and ``from tsdynamics.engine import run`` imports, but kept OFF ``__all__`` /
# ``dir()`` — see ``_INTERNAL_SUBMODULES`` below.  The redundant ``as`` form marks
# them as deliberate re-exports rather than unused imports.
from . import (
    data as data,
)
from . import (
    derived as derived,
)
from . import (
    engine as engine,
)
from . import (
    families as families,
)
from . import (
    registry as registry,
)
from . import (
    solvers as solvers,
)
from . import (
    utils as utils,
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
    FixedPointSet as FixedPointSet,
)
from .analysis import (
    GALIResult as GALIResult,
)
from .analysis import (
    LyapunovFromData as LyapunovFromData,
)
from .analysis import (
    LyapunovSpectrum as LyapunovSpectrum,
)
from .analysis import (
    OrbitDiagram as OrbitDiagram,
)
from .analysis import (
    OrbitSet as OrbitSet,
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
    UncertaintyExponent as UncertaintyExponent,
)
from .analysis import (
    WadaResult as WadaResult,
)
from .analysis import (
    WindowedRQA as WindowedRQA,
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
    embed as embed,
)
from .analysis import (
    embedding_dimension as embedding_dimension,
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
    gali as gali,
)
from .analysis import (
    generalized_dimension as generalized_dimension,
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
    max_lyapunov as max_lyapunov,
)
from .analysis import (
    mutual_information as mutual_information,
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
    poincare_section as poincare_section,
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
    tipping_points as tipping_points,
)
from .analysis import (
    uncertainty_exponent as uncertainty_exponent,
)
from .analysis import (
    wada_property as wada_property,
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
__version__ = "5.4.0"

# Built-in system classes are NOT bound into this namespace — that would bury the
# submodules (``analysis``, ``data``, ``systems``, …) under ~150 model names in
# ``dir()`` / autocomplete. The canonical path is ``tsdynamics.systems.<Name>``
# (e.g. ``tsdynamics.systems.Lorenz``). For backwards compatibility, ``tsd.Lorenz``
# and ``from tsdynamics import Lorenz`` still resolve, lazily, via ``__getattr__``
# below — but the names stay out of ``__all__`` and ``__dir__`` so they don't clutter
# the top-level surface.
_SYSTEM_NAMES = frozenset(systems._SYSTEM_NAMES)

# The two plotting names promoted to the curated top level (30 -> 32). They live
# in ``tsdynamics.viz.transforms`` and resolve lazily through ``__getattr__``, so
# advertising them costs nothing at import time:
#
#   ts.plot(traj)                                  # what ts.viz.plot does
#   ts.plot(traj, "delay_embedding", delay=7)      # ... plus a named transform
#   ts.plot(vdp, "flow_speed", "streamlines", "nullclines")   # ... several, one subject
#   ts.plot(vdp, ts.T("flow_speed", log=True), ts.T("streamlines", color="w"))
#
# Every name in an example here is a REGISTERED transform (``ts.viz.compatibility()``
# lists them).  It reads like a detail; it is not.  These four lines were the first
# thing a new user copied, and they used to name ``"basins"`` / ``"trajectory"`` —
# transforms that do not exist — so the documented one-liner answered with
# ``InvalidParameterError: unknown plot transform 'basins'``.
#
# Everything else in the plotting surface (``spec`` / ``geometry`` / ``draw`` /
# ``compatibility``) stays under ``ts.viz``, so the curated namespace does not
# drift back towards a flat dump.
_VIZ_FRONT_DOOR = frozenset({"plot", "T"})

#: Submodules that are bound eagerly (so ``ts.engine`` resolves and
#: ``from tsdynamics.engine import run`` imports) but kept **off** ``__all__`` /
#: ``dir()``.  Two reasons, both from the same rule — a tab slot is spent only on
#: something a user will reach for:
#:
#: * ``engine`` / ``solvers`` / ``registry`` / ``families`` / ``utils`` are
#:   machinery — the Rust-facing compile/run seam, the solver table, the system
#:   and plugin registries, the ``SystemBase``/``ParamSet`` internals, and the
#:   shared grid/tolerance constants;
#: * ``data`` / ``derived`` are *redundant* here: everything a user needs from
#:   them (``Trajectory``, ``Box``, ``Ball``, ``Grid``; ``PoincareMap`` and the
#:   other wrappers) is already bound on the top level.
#:
#: This tuple is the single source of truth for that decision — the namespace
#: gate (``tests/test_namespace_curation.py``) reads it, so demoting or promoting
#: a submodule is a one-line edit here.
_INTERNAL_SUBMODULES = (
    "data",
    "derived",
    "engine",
    "families",
    "registry",
    "solvers",
    "utils",
)

# The curated top-level surface. Demoted analysis functions / result classes /
# machinery submodules stay fully reachable (bound above, resolvable as
# ``ts.<name>``); they are simply no longer advertised in ``__all__`` /
# autocomplete. Reach them at their qualified path —
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
    # State-space regions — the argument ``basins`` / ``find_attractors`` take,
    # so they belong next to the analyses that require them.
    "Box",
    "Ball",
    "Grid",
    # Headline analyses (the six a newcomer reaches for; the rest live under
    # ``ts.analysis.*`` and stay flat-re-exported for back-compat)
    "lyapunov_spectrum",
    "bifurcation_diagram",
    "poincare_section",
    "recurrence_matrix",
    "basins",
    "fixed_points",
    # Plotting — the headline one-liner and its per-transform option carrier.
    # Both resolve lazily (see ``__getattr__``) so ``import tsdynamics`` still
    # pulls in no plotting machinery.
    "plot",
    "T",
    # The four submodules worth typing a dot after. Everything else
    # (``engine`` / ``solvers`` / ``registry`` / ``families`` / ``utils`` /
    # ``data`` / ``derived``) stays importable but off the tab surface — see
    # ``_INTERNAL_SUBMODULES``.
    "systems",
    "analysis",
    "viz",
    "errors",
]


#: Names the v6 scope surgery removed outright, mapped to the reason.  Curating
#: the namespace hides ~230 reachable names from autocomplete, so this
#: ``AttributeError`` becomes the *only* feedback a user gets when they guess a
#: spelling — it has to teach, not just refuse.  A v5 user typing
#: ``ts.permutation_entropy`` should learn the scope changed, not read "no
#: attribute" and assume the install is broken.
_REMOVED_IN_V6 = {
    "entropy": "entropy estimators",
    "permutation_entropy": "entropy estimators",
    "dispersion_entropy": "entropy estimators",
    "sample_entropy": "entropy estimators",
    "multiscale_entropy": "entropy estimators",
    "lz76_complexity": "entropy estimators",
    "surrogates": "surrogate-data tests",
    "surrogate_test": "surrogate-data tests",
    "SurrogateTest": "surrogate-data tests",
    "time_reversal_asymmetry": "surrogate-data tests",
    "nonlinear_prediction_error": "surrogate-data tests",
    "transforms": "signal transforms (PSD, detrend, filters, feature extraction)",
    "power_spectrum": "signal transforms (PSD, detrend, filters, feature extraction)",
    "detrend": "signal transforms (PSD, detrend, filters, feature extraction)",
    "extract_features": "signal transforms (PSD, detrend, filters, feature extraction)",
}


def _attribute_error(name: str) -> AttributeError:
    """Build an ``AttributeError`` that names the line to type, not the mistake.

    Three cases, in the order a user is likely to hit them: a name the v6 scope
    surgery removed, a near-miss on something still here (a built-in system,
    which autocomplete deliberately hides, or a demoted analysis), and a genuine
    miss — answered with the two listings worth tab-completing.
    """
    import difflib

    if name in _REMOVED_IN_V6:
        return AttributeError(
            f"tsdynamics has no attribute {name!r}: the generic time-series layer "
            f"({_REMOVED_IN_V6[name]}) was removed in v6.\n"
            "TSDynamics is scoped to phase-space methods now. What stayed, and the "
            "closest thing to reach for:\n"
            "    ts.analysis.recurrence   # recurrence plots / RQA\n"
            "    ts.analysis.embedding    # delay embedding (data -> phase space)\n"
            "    ts.analysis.lyapunov     # lyapunov_from_data"
        )

    system_hit = difflib.get_close_matches(name, sorted(_SYSTEM_NAMES), n=1, cutoff=0.6)
    if system_hit:
        return AttributeError(
            f"module 'tsdynamics' has no attribute {name!r}. Built-in systems live "
            f"under ts.systems — you probably want:\n    ts.systems.{system_hit[0]}()"
        )

    reachable = sorted(set(__all__) | set(globals()) | set(getattr(analysis, "__all__", [])))
    hits = difflib.get_close_matches(name, reachable, n=3, cutoff=0.6)
    if hits:
        lines = "\n".join(f"    ts.{h}" for h in hits)
        return AttributeError(
            f"module 'tsdynamics' has no attribute {name!r}. Did you mean:\n{lines}"
        )

    # Counted live: a hardcoded total goes stale every time a system is added
    # (the docs already say 171 where the registry says 177).
    return AttributeError(
        f"module 'tsdynamics' has no attribute {name!r}.\n"
        "Tab-complete the two catalogues to find it:\n"
        f"    ts.systems.<TAB>    # the {len(_SYSTEM_NAMES)} built-in systems\n"
        "    ts.analysis.<TAB>   # the quantifiers, by capability"
    )


def __getattr__(name: str) -> Any:
    """Lazily resolve the ``viz`` submodule and built-in system classes.

    Two lazy resolutions live here:

    * ``tsdynamics.viz`` — imported on first access (and cached) so a plain
      ``import tsdynamics`` pulls in no plotting/IR machinery; ``viz`` still shows
      in ``__all__`` / ``dir()`` for discoverability.
    * ``tsdynamics.plot`` / ``tsdynamics.T`` — the plotting front door, resolved
      from :mod:`tsdynamics.viz.transforms` on first access (and cached), for the
      same reason: naming them in ``__all__`` must not make ``import tsdynamics``
      import a plotting layer.
    * ``tsdynamics.Lorenz`` and friends — the ~150 built-in system classes,
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
    if name in _VIZ_FRONT_DOOR:
        import importlib

        _transforms = importlib.import_module(f"{__name__}.viz.transforms")
        for attr in _VIZ_FRONT_DOOR:
            globals()[attr] = getattr(_transforms, attr)  # cache both at once
        return globals()[name]
    if name in _SYSTEM_NAMES:
        return getattr(systems, name)
    raise _attribute_error(name)


def __dir__() -> list[str]:
    """Top-level surface = the public API in ``__all__`` (models live under ``systems``).

    Machinery submodules (:data:`_INTERNAL_SUBMODULES`) and the ~150 built-in
    system classes stay resolvable but out of this listing, so ``ts.<TAB>`` is the
    mental model rather than a dump.
    """
    return sorted(__all__)
