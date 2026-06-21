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

#: The exact curated ``tsdynamics.__all__`` after WS-NAMESPACE (~30 names).
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
    # the six promoted headline analyses
    "lyapunov_spectrum",
    "bifurcation_diagram",
    "poincare_section",
    "recurrence_matrix",
    "basins",
    "fixed_points",
    # navigable submodules
    "analysis",
    "transforms",
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
    "permutation_entropy",
    "sample_entropy",
    "embed",
    "optimal_delay",
    "surrogate_test",
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


# ── the analysis tree ────────────────────────────────────────────────────────────

_CATEGORIES = (
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
    "permutation_entropy",
    "embed",
    "surrogate_test",
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


def test_entropy_collision_and_documented_escape_hatch():
    """``ts.analysis.entropy`` is the function; the subpackage is reached by name / importlib.

    The ``entropy`` subpackage shares a name with the :func:`entropy` function, so
    attribute access resolves to the function (the documented, griffe-safe
    collision).  The estimators stay reachable via the spellings the docstrings
    advertise — a ``from`` import or :func:`importlib.import_module`.
    """
    import importlib

    # attribute access -> the function (not the module)
    assert callable(analysis.entropy)
    # the working escape hatches reach the subpackage's estimators
    from tsdynamics.analysis.entropy import permutation_entropy

    assert callable(permutation_entropy)
    mod = importlib.import_module("tsdynamics.analysis.entropy")
    assert mod.__name__ == "tsdynamics.analysis.entropy"
    assert "permutation_entropy" in mod.__all__
