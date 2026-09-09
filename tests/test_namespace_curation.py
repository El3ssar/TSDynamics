"""Stream WS-NAMESPACE: the curated v4 top-level surface + the analysis tree.

Locks the namespace contract that WS-NAMESPACE establishes:

* the top-level ``tsdynamics.__all__`` is curated to the ~30 headline names (the
  family bases, the derived wrappers, :class:`Trajectory`, the six promoted
  analyses, and the navigable submodules);
* every demoted analysis function / result class / state-space primitive stays
  **fully reachable** (flat re-export + ``ts.<name>``) — only its ``__all__``
  membership is dropped;
* the headline aliases ``bifurcation_diagram`` / ``basins`` resolve to their
  canonical implementations;
* ``ts.errors`` is reachable and ``ts.viz`` resolves lazily (a plain
  ``import tsdynamics`` pulls in neither ``tsdynamics.viz`` nor a plot library);
* ``ts.analysis.<TAB>`` surfaces the ~10 capability subpackages while the flat
  re-exports remain importable.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import tsdynamics as ts
from tsdynamics import analysis

# ── the curated top-level surface ────────────────────────────────────────────────

#: The exact curated ``tsdynamics.__all__``: WS-NAMESPACE's ~30 names plus the two
#: v6 plotting front doors (``plot`` / ``T``).  Plotting is the library's headline
#: feature, so the one-liner has to be reachable without a submodule hop; every
#: other plotting name (``spec`` / ``geometry`` / ``draw`` / ``compatibility``)
#: stays under ``ts.viz`` so the namespace does not drift back to a flat dump.
_CURATED_TOP_LEVEL = {
    "__version__",
    # family bases (subclass these)
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "StochasticSystem",
    "WrappedSystem",
    # the trajectory type
    "Trajectory",
    # derived wrappers
    "EnsembleSystem",
    "PoincareMap",
    "ProjectedSystem",
    "StroboscopicMap",
    "TangentSystem",
    # the two plotting front doors (lazily resolved, like ``viz``)
    "plot",
    "T",
    # the six promoted headline analyses
    "lyapunov_spectrum",
    "bifurcation_diagram",
    "poincare_section",
    "recurrence_matrix",
    "basins",
    "fixed_points",
    # navigable submodules
    "analysis",
    "data",
    "derived",
    "families",
    "registry",
    "systems",
    "utils",
    "errors",
    "viz",
    "engine",
    "solvers",
}


def test_top_level_all_is_curated():
    """``ts.__all__`` is exactly the curated headline set — no flat dump."""
    assert set(ts.__all__) == _CURATED_TOP_LEVEL
    # ``__dir__`` mirrors ``__all__`` (curated autocomplete surface).
    assert set(dir(ts)) >= _CURATED_TOP_LEVEL - {"__version__"}


def test_headline_aliases_resolve_to_canonical():
    """The promoted aliases delegate to the original implementations."""
    assert ts.bifurcation_diagram is ts.orbit_diagram
    assert ts.basins is ts.basins_of_attraction


#: A representative slice of names DEMOTED from the curated top level: they must
#: stay reachable (flat re-export) but no longer appear in ``__all__``.
_DEMOTED_ANALYSIS = [
    "orbit_diagram",
    "basins_of_attraction",
    "max_lyapunov",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "correlation_dimension",
    "generalized_dimension",
    "DimensionResult",
    "gali",
    "zero_one_test",
    "expansion_entropy",
    "rqa",
    "windowed_rqa",
    "RQAResult",
    "embed",
    "optimal_delay",
    "find_attractors",
    "return_map",
    "OrbitDiagram",
    "FixedPoint",
]


@pytest.mark.parametrize("name", _DEMOTED_ANALYSIS)
def test_demoted_analysis_names_stay_reachable(name):
    """A demoted analysis name is dropped from ``__all__`` but still resolves."""
    assert name not in ts.__all__, f"{name} should be demoted from the curated top level"
    assert hasattr(ts, name), f"tsdynamics.{name} must stay reachable (flat re-export)"
    # ``from tsdynamics import <name>`` and the analysis re-export are the same object.
    assert getattr(ts, name) is getattr(analysis, name)


_DEMOTED_DATA = ["Box", "Ball", "Grid", "sampler", "grid_points", "set_distance"]


@pytest.mark.parametrize("name", _DEMOTED_DATA)
def test_demoted_data_primitives_stay_reachable(name):
    """State-space primitives drop from ``__all__`` but stay reachable via ``ts`` and ``ts.data``."""
    assert name not in ts.__all__
    assert hasattr(ts, name)
    assert getattr(ts, name) is getattr(ts.data, name)


def test_models_stay_hidden_but_reachable():
    """Built-in systems remain off the curated surface yet resolve lazily."""
    assert "Lorenz" not in ts.__all__
    assert "Lorenz" not in dir(ts)
    assert ts.Lorenz is ts.systems.Lorenz


# ── errors (eager) + viz (lazy) ──────────────────────────────────────────────────


def test_errors_submodule_reachable():
    assert hasattr(ts, "errors")
    assert ts.errors.TSDynamicsError is ts.errors.TSDynamicsError
    assert issubclass(ts.errors.InvalidParameterError, ValueError)


def test_viz_is_reachable_and_cached():
    """``ts.viz`` resolves (lazily) to the viz package and caches the binding."""
    import tsdynamics.viz as viz_mod

    assert ts.viz is viz_mod
    assert ts.viz is ts.viz  # cached: identical object on repeat access
    assert hasattr(ts.viz, "PlotSpec")


def test_plain_import_pulls_no_viz_or_plot_library():
    """A fresh ``import tsdynamics`` loads neither ``tsdynamics.viz`` nor matplotlib."""
    code = (
        "import sys, tsdynamics\n"
        "assert 'tsdynamics.viz' not in sys.modules, 'viz eagerly imported'\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib eagerly imported'\n"
        "tsdynamics.viz\n"  # touch -> lazy import
        "assert 'tsdynamics.viz' in sys.modules, 'viz did not resolve lazily'\n"
        "assert 'matplotlib' not in sys.modules, 'viz pulled in matplotlib'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_plot_front_door_resolves_lazily_and_is_cached():
    """``ts.plot`` / ``ts.T`` are advertised, but cost nothing until touched.

    Naming them in ``__all__`` must not make ``import tsdynamics`` import a
    plotting layer — that is the whole reason ``viz`` itself is lazy — so both
    resolve through ``__getattr__`` and cache on first access.
    """
    code = (
        "import sys, tsdynamics as ts\n"
        "assert 'plot' in ts.__all__ and 'T' in ts.__all__\n"
        "assert 'tsdynamics.viz' not in sys.modules, 'naming plot imported viz'\n"
        "p = ts.plot\n"
        "assert 'tsdynamics.viz.transforms' in sys.modules\n"
        "assert ts.plot is p and ts.T is ts.T\n"
        "assert 'matplotlib' not in sys.modules, 'the front door pulled in matplotlib'\n"
        "import tsdynamics.viz.transforms as tr\n"
        "assert ts.plot is tr.plot and ts.T is tr.T\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_viz_transforms_subpackage_is_not_shadowed_by_a_function():
    """``ts.viz.transforms`` is the **module**, never a listing function.

    The v4 namespace work removed exactly this defect once already (a function
    shadowing a subpackage of the same name).  The filtered listing is therefore
    ``ts.viz.plot_transforms(...)``, named after the registry it reads.
    """
    import types

    assert isinstance(ts.viz.transforms, types.ModuleType)
    assert ts.viz.transforms.Geometry is not None
    assert callable(ts.viz.plot_transforms)
    assert {t.source for t in ts.viz.plot_transforms()} == {"data", "model"}


# ── the analysis tree ────────────────────────────────────────────────────────────

_CATEGORIES = (
    "lyapunov",
    "dimensions",
    "chaos",
    "recurrence",
    "embedding",
    "orbits",
    "fixedpoints",
    "basins",
    "sampling",
)


def test_analysis_dir_shows_categories():
    """``ts.analysis.<TAB>`` surfaces the categories + ``discover_plugins``, not the flat dump."""
    assert set(dir(analysis)) == {*_CATEGORIES, "discover_plugins"}


@pytest.mark.parametrize("cat", _CATEGORIES)
def test_analysis_category_in_all(cat):
    """Each capability category is advertised in ``analysis.__all__``."""
    assert cat in analysis.__all__


@pytest.mark.parametrize("cat", _CATEGORIES)
def test_analysis_category_importable_with_all(cat):
    """Each category is an importable subpackage that lists its own estimators."""
    import importlib

    mod = importlib.import_module(f"tsdynamics.analysis.{cat}")
    assert isinstance(mod.__all__, list)


_FLAT_ANALYSIS_SAMPLE = [
    "correlation_dimension",
    "lyapunov_spectrum",
    "gali",
    "recurrence_matrix",
    "embed",
    "orbit_diagram",
    "fixed_points",
    "basins_of_attraction",
]


@pytest.mark.parametrize("name", _FLAT_ANALYSIS_SAMPLE)
def test_analysis_flat_reexports_retained(name):
    """Flat re-exports survive the decluttered ``__dir__``: still importable + in ``__all__``."""
    assert name in analysis.__all__
    assert hasattr(analysis, name)
    assert getattr(analysis, name) is getattr(ts, name)


_REMOVED_SUBPACKAGES = [
    "tsdynamics.transforms",
    "tsdynamics.analysis.entropy",
    "tsdynamics.analysis.surrogate",
]


@pytest.mark.parametrize("path", _REMOVED_SUBPACKAGES)
def test_generic_series_statistics_layer_is_gone(path):
    """The v6 scope surgery removed the generic time-series statistics layer.

    TSDynamics is scoped to *dynamical-systems* methods: phase-space quantifiers
    stay, generic series statistics (entropy estimators, surrogate-data tests,
    spectra / filters / feature extraction) left the library.  Their modules must
    be gone outright — no shim, no lazy re-export.
    """
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)


_REMOVED_NAMES = [
    "entropy",
    "permutation_entropy",
    "sample_entropy",
    "lz76_complexity",
    "surrogates",
    "surrogate_test",
    "SurrogateTest",
    "time_reversal_asymmetry",
    "nonlinear_prediction_error",
    "transforms",
]


@pytest.mark.parametrize("name", _REMOVED_NAMES)
def test_removed_names_are_unreachable(name):
    """No removed name survives as a top-level or ``analysis`` attribute."""
    assert not hasattr(ts, name), f"tsdynamics.{name} should have been removed"
    assert name not in ts.__all__
    assert not hasattr(analysis, name), f"tsdynamics.analysis.{name} should have been removed"


def test_surviving_phase_space_quantifiers_are_untouched():
    """The dynamics-side names that share a spelling with the removed layer survive.

    ``expansion_entropy`` (A-CHAOS) and ``basin_entropy`` (A-BASIN) are *not* the
    generic entropy estimators — they are phase-space quantifiers that merely
    carry "entropy" in their names, and the surgery must not have taken them.
    """
    for name in ("expansion_entropy", "basin_entropy", "ExpansionEntropyResult"):
        assert hasattr(ts, name)
        assert getattr(ts, name) is getattr(analysis, name)
