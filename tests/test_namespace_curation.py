"""The curated namespace: the top level **and every public subpackage**.

The owner's complaint that opened this stream was a literal ``ts.<TAB>``: eleven
submodules — ``engine``, ``solvers``, ``registry``, ``families``, ``utils``,
``data``, ``derived`` among them — sitting next to the things a user actually
reaches for.  "We should only see there what is supposed to be exposed to the
user, no internal use things. Same applies to submodules!!"

So this file locks two contracts:

1. **Per-namespace curation.**  ``ts.__all__`` is the ~27 headline names; the
   machinery submodules are demoted; ``ts.analysis.<TAB>`` shows the capability
   categories plus one headline quantifier each; ``ts.viz`` shows the plotting
   API and not the envelope/registry plumbing.
2. **A generic anti-rot gate** (:func:`test_public_package_listing_is_clean`)
   that sweeps *every* public package — discovered live, so a new subpackage
   joins the sweep with zero test edits — and fails on a private name, a
   re-exported stdlib/third-party module, or anything on the explicit internals
   list.  A namespace cannot quietly re-accumulate junk.

Demotion is never removal: every demoted name stays importable at its qualified
path and, where it was flat-re-exported, as ``ts.<name>``.  That invariant is
asserted for each demotion below.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import types

import pytest

import tsdynamics as ts
from tsdynamics import analysis

# ── the curated top-level surface ────────────────────────────────────────────────

#: The exact curated ``tsdynamics.__all__``.  Read it as the answer to "what does
#: a user need in tab completion?":  the five family bases, the trajectory type,
#: the five derived wrappers, the three state-space regions ``basins`` /
#: ``find_attractors`` take as an argument, the six headline analyses, the
#: plotting front door, and the four submodules worth typing a dot after.
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
    # state-space regions — you cannot call ``ts.basins(system, region)`` without one
    "Box",
    "Ball",
    "Grid",
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
    # the four navigable submodules
    "systems",
    "analysis",
    "viz",
    "errors",
}


def test_top_level_all_is_curated():
    """``ts.__all__`` is exactly the curated headline set — no flat dump."""
    assert set(ts.__all__) == _CURATED_TOP_LEVEL
    # ``__dir__`` mirrors ``__all__`` (curated autocomplete surface).
    assert set(dir(ts)) == _CURATED_TOP_LEVEL


def test_headline_aliases_resolve_to_canonical():
    """The promoted aliases delegate to the original implementations."""
    assert ts.bifurcation_diagram is ts.orbit_diagram
    assert ts.basins is ts.basins_of_attraction


def test_state_space_regions_are_promoted_because_basins_takes_one():
    """``Box`` / ``Ball`` / ``Grid`` earn their tab slot: they are the ``region`` type.

    The original rationale here was "you *cannot* call ``ts.basins`` without a
    ``Grid``".  That is no longer true and the test must not keep asserting it:
    ``basins`` now also accepts a bare ``(lo, hi, n)`` triple per component, and
    its no-region error teaches exactly that spelling.  So the honest claim is
    the weaker one — a region object is a *first-class* argument type of the
    analyses the curated surface advertises, which is what earns three tab slots
    next to them.

    Pinned to the real annotation rather than to a parameter name, so that
    dropping ``Grid`` from the accepted region types fails here instead of
    quietly leaving three promoted names as decoration.
    """
    import inspect

    for fn in (ts.basins, ts.find_attractors):
        params = inspect.signature(fn).parameters
        assert "region" in params, f"{fn.__name__} no longer takes a region"
        annotation = str(params["region"].annotation)
        for cls in ("Box", "Ball", "Grid"):
            assert cls in annotation, (
                f"{fn.__name__}'s region no longer accepts {cls}; "
                "either re-justify or drop the top-level promotion"
            )
    for name in ("Box", "Ball", "Grid"):
        assert name in ts.__all__
        assert getattr(ts, name) is getattr(ts.data, name)


# ── demoted machinery submodules ─────────────────────────────────────────────────


def test_internal_submodules_are_demoted_but_fully_reachable():
    """The machinery submodules leave ``dir()`` and keep every other capability.

    ``engine`` / ``solvers`` / ``registry`` / ``families`` / ``utils`` are
    machinery; ``data`` / ``derived`` are redundant with names already bound on
    the top level.  All seven stay bound attributes and importable modules — only
    the tab slot is reclaimed.
    """
    assert ts._INTERNAL_SUBMODULES, "the demotion list emptied — nothing is being curated"
    for name in ts._INTERNAL_SUBMODULES:
        assert name not in ts.__all__, f"{name} crept back into the curated __all__"
        assert name not in dir(ts), f"{name} crept back into tab completion"
        # ...but every reach still works.
        assert hasattr(ts, name), f"ts.{name} must stay resolvable"
        mod = importlib.import_module(f"tsdynamics.{name}")
        assert getattr(ts, name) is mod


@pytest.mark.parametrize(
    "stmt",
    [
        "from tsdynamics.engine import run",
        "from tsdynamics.engine.run import integrate",
        "from tsdynamics.solvers import recommend",
        "from tsdynamics.registry import all_systems",
        "from tsdynamics.families import SystemBase, ParamSet, System",
        "from tsdynamics.utils import make_output_grid, DEFAULT_RTOL",
        "from tsdynamics.data import Box, Trajectory",
        "from tsdynamics.derived import DerivedSystem",
    ],
)
def test_demoted_submodule_deep_imports_still_work(stmt):
    """Demotion is a *listing* change: every documented import path still resolves."""
    exec(compile(stmt, "<test>", "exec"), {})  # noqa: S102 - the import IS the assertion


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


_DEMOTED_DATA = ["sampler", "grid_points", "set_distance"]


@pytest.mark.parametrize("name", _DEMOTED_DATA)
def test_demoted_data_primitives_stay_reachable(name):
    """Sampling helpers drop from ``__all__`` but stay reachable via ``ts`` and ``ts.data``."""
    assert name not in ts.__all__
    assert hasattr(ts, name)
    assert getattr(ts, name) is getattr(ts.data, name)


def test_models_stay_hidden_but_reachable():
    """Built-in systems remain off the curated surface yet resolve lazily."""
    assert "Lorenz" not in ts.__all__
    assert "Lorenz" not in dir(ts)
    assert ts.Lorenz is ts.systems.Lorenz


# ── the generic anti-rot gate ────────────────────────────────────────────────────

#: The one dunder a public listing may carry: the package version.
_ALLOWED_DUNDERS = frozenset({"__version__"})

#: Names that must never appear in any public listing, whatever the package.
#: ``discover_plugins`` is entry-point machinery; the rest are typing / ``__future__``
#: artefacts that leak when a module has no ``__dir__``.
_NEVER_PUBLIC = frozenset({"annotations", "Any", "TYPE_CHECKING", "discover_plugins"})

#: Renderer *backend* packages, excluded from the sweep because their
#: ``__init__.py`` files are outside this stream's ownership (they are being
#: edited concurrently by the renderer work).  Each still leaks its private
#: constants and, for threejs, the ``json`` / ``os`` stdlib modules into ``dir()``.
#: They are three levels deep (``ts.viz.render.mpl``) and ``render`` itself is not
#: on ``ts.viz``'s listing, so no user tab-completes into them — but the exclusion
#: should be deleted, not grown, once those files get a ``__dir__``.
#:
#: An ordered tuple, not a set: it is a ``parametrize`` argument, and a set's
#: iteration order varies with the per-process string hash seed, which makes
#: xdist workers disagree about what they collected.
_UNCURATED_RENDERER_BACKENDS = (
    "tsdynamics.viz.render.mpl",
    "tsdynamics.viz.render.plotly",
    "tsdynamics.viz.render.threejs",
)


def _public_packages() -> list[str]:
    """Every public package in the tree, discovered live (so new ones join the gate)."""
    found = [
        m.name
        for m in pkgutil.walk_packages(ts.__path__, "tsdynamics.")
        if m.ispkg and "._" not in m.name
    ]
    return ["tsdynamics", *sorted(found)]


def _advertised_public_modules() -> list[str]:
    """Every *plain module* a public package advertises in its own ``dir()``.

    The package sweep filters on ``ispkg``, which silently exempts a public
    **module** sitting on a curated listing.  ``tsdynamics.analysis.planar`` was
    exactly that: ``analysis.__dir__`` advertises ``planar`` as a capability
    category, a user tab-completes into it, and the module — having no
    ``__dir__`` of its own — handed them ``np`` / ``warnings`` / ``dataclass`` /
    ``field`` / ``Any`` / ``Callable`` / ``Sequence`` plus its private grid
    helpers.  A namespace you can *reach* by tab completion is a namespace this
    gate must cover, package or not.
    """
    out: set[str] = set()
    for pkg_name in _public_packages():
        if pkg_name in _UNCURATED_RENDERER_BACKENDS:
            continue
        pkg = importlib.import_module(pkg_name)
        for attr in dir(pkg):
            obj = getattr(pkg, attr, None)
            if isinstance(obj, types.ModuleType) and not hasattr(obj, "__path__"):
                out.add(obj.__name__)
    return sorted(out)


def test_the_package_sweep_actually_finds_packages():
    """Guard the guard: an empty/tiny discovery would make the gate below vacuous."""
    pkgs = _public_packages()
    assert len(pkgs) >= 20, pkgs
    # The headline namespaces are in scope.
    assert {"tsdynamics", "tsdynamics.analysis", "tsdynamics.viz", "tsdynamics.systems"} <= set(
        pkgs
    )


def test_the_module_sweep_actually_finds_the_advertised_modules():
    """Guard the guard: the module sweep must really reach ``analysis.planar``."""
    mods = _advertised_public_modules()
    assert "tsdynamics.analysis.planar" in mods, mods


@pytest.mark.parametrize("mod_name", _advertised_public_modules())
def test_advertised_public_module_listing_is_clean(mod_name):
    """A public *module* on a curated listing shows its API, not its imports.

    Same contract as the package sweep, minus the dunder rule: a plain module
    always carries ``__name__`` / ``__file__`` / ``__builtins__``, and every
    autocompleter hides leading-underscore names until you type one.  The rot
    that actually reaches a user is the non-dunder kind — a re-exported ``numpy``
    or ``warnings``, a ``_HELPER`` constant, a typing artefact.
    """
    mod = importlib.import_module(mod_name)
    listing = dir(mod)

    private = [n for n in listing if n.startswith("_") and not n.startswith("__")]
    assert not private, f"{mod_name} leaks private names into dir(): {private}"

    foreign = [
        n
        for n in listing
        if isinstance(getattr(mod, n, None), types.ModuleType)
        and getattr(mod, n).__name__ != f"{mod_name}.{n}"
    ]
    assert not foreign, f"{mod_name} re-exports foreign modules into dir(): {foreign}"

    internals = sorted(set(listing) & _NEVER_PUBLIC)
    assert not internals, f"{mod_name} lists internals: {internals}"

    for n in listing:
        assert hasattr(mod, n), f"{mod_name}.{n} is advertised but does not resolve"


@pytest.mark.parametrize("name", _UNCURATED_RENDERER_BACKENDS)
def test_renderer_backend_exclusions_are_live(name):
    """The carve-out must not outlive the packages it names."""
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("pkg_name", _public_packages())
def test_public_package_listing_is_clean(pkg_name):
    """``dir()`` of every public package shows only its own public API.

    Three rules, one per way a listing rots:

    * **no private names** — a package with no ``__dir__`` dumps its ``_HELPERS``
      and every module dunder;
    * **no foreign modules** — a bare ``import warnings`` at the top of an
      ``__init__.py`` becomes ``ts.viz.render.warnings`` in autocomplete; only a
      real submodule of *this* package may appear;
    * **nothing on the internals list** — plugin hooks and typing artefacts.
    """
    if pkg_name in _UNCURATED_RENDERER_BACKENDS:
        pytest.skip("renderer backend __init__ is outside this stream's file ownership")
    mod = importlib.import_module(pkg_name)
    listing = dir(mod)

    private = [n for n in listing if n.startswith("_") and n not in _ALLOWED_DUNDERS]
    assert not private, f"{pkg_name} leaks private names into dir(): {private}"

    foreign = [
        n
        for n in listing
        if isinstance(getattr(mod, n, None), types.ModuleType)
        and getattr(mod, n).__name__ != f"{pkg_name}.{n}"
    ]
    assert not foreign, f"{pkg_name} re-exports foreign modules into dir(): {foreign}"

    internals = sorted(set(listing) & _NEVER_PUBLIC)
    assert not internals, f"{pkg_name} lists internals: {internals}"

    # Everything advertised must actually resolve, and no duplicates.
    assert len(listing) == len(set(listing)), f"{pkg_name} lists a name twice"
    for n in listing:
        assert hasattr(mod, n), f"{pkg_name}.{n} is advertised but does not resolve"


@pytest.mark.parametrize("pkg_name", _public_packages())
def test_public_package_declares_all(pkg_name):
    """Every public package declares an ``__all__`` — the curation is explicit, not accidental."""
    if pkg_name in _UNCURATED_RENDERER_BACKENDS:
        pytest.skip("renderer backend __init__ is outside this stream's file ownership")
    mod = importlib.import_module(pkg_name)
    assert isinstance(getattr(mod, "__all__", None), list), f"{pkg_name} has no __all__"


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
    assert isinstance(ts.viz.transforms, types.ModuleType)
    assert ts.viz.transforms.Geometry is not None
    assert callable(ts.viz.plot_transforms)
    assert {t.source for t in ts.viz.plot_transforms()} == {"data", "model"}


# ── the viz namespace ────────────────────────────────────────────────────────────

#: What ``ts.viz.<TAB>`` shows: the front door, the IR, styling, the transform
#: extension point, and the JSON round trip.  Not the envelope plumbing.
_CURATED_VIZ = {
    "plot",
    "T",
    "PlotSpec",
    "PlotKind",
    "Layer",
    "Axis",
    "Layout",
    "Annotation",
    "Animation",
    "STYLE_KEYS",
    "Theme",
    "themes",
    "get_theme",
    "set_theme",
    "register_theme",
    "transforms",
    "PlotTransform",
    "Geometry",
    "plot_transform",
    "plot_transforms",
    "compatibility",
    "geometry",
    "draw",
    # Both halves of the JSON envelope stay listed on purpose — see
    # ``test_viz_internals_are_demoted_but_reachable``.
    "to_json",
    "from_json",
    "to_dict_envelope",
    "from_dict_envelope",
    "SCHEMA_VERSION",
}


def test_viz_listing_is_curated():
    assert set(ts.viz.__all__) == _CURATED_VIZ
    assert set(dir(ts.viz)) == _CURATED_VIZ


def test_viz_internals_are_demoted_but_reachable():
    """The registry / validation / mixin plumbing keeps working, off the listing.

    The JSON envelope is deliberately **not** in this set: a previous stream
    promoted the loader half because ``ts.viz.from_json`` failing to resolve made
    the save/load round trip one-way in practice.  "Curated" means *judged*, not
    *minimal* — pruning a name that a documented workflow needs would trade the
    owner's complaint for a worse one.
    """
    viz = ts.viz
    assert viz._INTERNAL_NAMES, "the viz demotion list emptied"
    for name in viz._INTERNAL_NAMES:
        assert name not in viz.__all__, f"ts.viz.{name} crept back into __all__"
        assert name not in dir(viz), f"ts.viz.{name} crept back into tab completion"
        assert hasattr(viz, name), f"ts.viz.{name} must stay reachable"
    assert callable(viz.normalize_style)
    assert viz.THEMES and viz.Plottable is not None
    # ...while the envelope round trip stays on the surface.
    assert {"to_dict_envelope", "from_dict_envelope", "SCHEMA_VERSION"} <= set(dir(viz))


def test_viz_render_dispatch_listing_is_the_dispatch_api():
    """``ts.viz.render`` had no ``__dir__``: it leaked ``importlib`` / ``warnings`` / ``Any``.

    It also leaked *non-deterministically* — the ``mpl`` / ``plotly`` / ``threejs``
    backend submodules appear as attributes only once something has rendered, so
    the listing depended on test order.
    """
    from tsdynamics.viz import render

    assert set(dir(render)) == set(render.__all__)
    for leaked in ("importlib", "warnings", "Any", "caps", "annotations"):
        assert leaked not in dir(render)
    # ...and the internals stay reachable.
    assert render.caps.__name__ == "tsdynamics.viz.render.caps"


def test_viz_render_resolves_in_a_session_that_has_never_drawn():
    """``ts.viz.render`` must not blink into existence after the first plot.

    Nothing imports the subpackage at ``tsdynamics.viz`` import time, so before
    this was made lazy-but-deterministic ``ts.viz.render`` raised
    ``AttributeError`` in a fresh session and succeeded in one that had already
    rendered (the first render imports it, which binds it on the parent package
    as a side effect).  ``render`` is demoted from the listing, but it is the
    dispatch/plugin surface — ``select_renderer``, ``register_builtin_renderers``
    — so *demotion is never removal* has to hold for it too.

    Asserted in a subprocess, because any earlier test in this process may
    already have drawn something and would hide the defect.
    """
    code = (
        "import sys, tsdynamics as ts\n"
        "r = ts.viz.render\n"  # the access IS the assertion
        "assert r.__name__ == 'tsdynamics.viz.render'\n"
        "assert callable(r.select_renderer) and callable(r.register_builtin_renderers)\n"
        # ...and resolving it must not drag a plotting library in.
        "assert 'matplotlib.pyplot' not in sys.modules, 'touching render imported matplotlib'\n"
        # ...and it stays off the curated listing.
        "assert 'render' not in dir(ts.viz)\n"
        "assert ts.viz.render is r\n"  # cached
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_viz_transforms_internals_are_demoted_but_reachable():
    """The geometry pieces, primitive tables and lowering pipeline stay importable."""
    tr = ts.viz.transforms
    demoted = (
        "Channel",
        "ChannelType",
        "Part",
        "PRIMITIVES",
        "RESERVED_PRIMITIVES",
        "ADMITTED_SERIES_DIAGNOSTICS",
        "EXCLUDED_SERIES_TOOLBOX",
        "TransformCall",
        "build_spec",
        "lower",
    )
    for name in demoted:
        assert name not in tr.__all__, f"{name} crept back into ts.viz.transforms.__all__"
        assert name not in dir(tr), f"{name} crept back into tab completion"
        assert hasattr(tr, name), f"ts.viz.transforms.{name} must stay reachable"


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
    "planar",
    "sampling",
)

_ANALYSIS_HEADLINE = (
    "lyapunov_spectrum",
    "correlation_dimension",
    "gali",
    "recurrence_matrix",
    "embed",
    # The flow spelling is the headline one here as at the top level — the two
    # names are the SAME function object, and a name that is primary in one
    # namespace must not be absent from the other (``ts.bifurcation_diagram``
    # resolved while ``ts.analysis.bifurcation_diagram`` raised AttributeError).
    "bifurcation_diagram",
    "poincare_section",
    "fixed_points",
    "basins_of_attraction",
)


def test_analysis_dir_shows_categories_and_headline_quantifiers():
    """``ts.analysis.<TAB>`` answers both "what is in here?" and "what can I do?"."""
    assert set(dir(analysis)) == {*_CATEGORIES, *_ANALYSIS_HEADLINE}
    # One headline per category, and the flat dump stays off the surface.
    assert len(dir(analysis)) == 19
    assert "correlation_sum" not in dir(analysis)
    assert "discover_plugins" not in dir(analysis), "plugin machinery is not user API"
    assert hasattr(analysis, "discover_plugins"), "...but it must stay reachable"


@pytest.mark.parametrize("name", _ANALYSIS_HEADLINE)
def test_analysis_headline_is_a_real_callable(name):
    """Every advertised headline resolves to the same object the top level exposes."""
    fn = getattr(analysis, name)
    assert callable(fn)
    assert fn is getattr(ts, name)


@pytest.mark.parametrize("cat", _CATEGORIES)
def test_analysis_category_in_all(cat):
    """Each capability category is advertised in ``analysis.__all__``."""
    assert cat in analysis.__all__


@pytest.mark.parametrize("cat", _CATEGORIES)
def test_analysis_category_importable_with_all(cat):
    """Each category is an importable subpackage that lists its own estimators."""
    mod = importlib.import_module(f"tsdynamics.analysis.{cat}")
    assert isinstance(mod.__all__, list)


def test_every_analysis_category_resolves_without_touching_viz():
    """A category must not blink into existence because something else imported it.

    ``tsdynamics.analysis.planar`` used to be absent from a fresh
    ``import tsdynamics`` and then *appear* the moment anything touched the viz
    transforms (``viz/transforms/fields.py`` does ``from tsdynamics.analysis
    import planar``, which binds the attribute on the parent package as a side
    effect).  So ``ts.analysis.planar`` resolved or raised depending on what the
    session had done first — the same order-dependence that used to make
    ``dir(ts.viz.render)`` change after the first render, and worse here because
    the capability was never in ``dir()`` in either state.

    The categories are imported eagerly in ``analysis/__init__.py``, so this
    holds in a subprocess that has touched nothing else.
    """
    code = (
        "import sys, tsdynamics as ts\n"
        "assert 'tsdynamics.viz' not in sys.modules\n"
        f"for cat in {list(_CATEGORIES)!r}:\n"
        "    assert hasattr(ts.analysis, cat), cat\n"
        "    assert cat in dir(ts.analysis), cat\n"
        "assert 'tsdynamics.viz' not in sys.modules, 'analysis pulled in viz'\n"
        # the planar estimators are reachable by drilling in, as documented
        "ts.analysis.planar.nullclines\n"
        "ts.analysis.planar.ftle_field\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


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


def test_dimension_spectrum_plot_spec_is_demoted_but_reachable():
    """A ``*_plot_spec`` builder is viz plumbing, not a dimension estimator."""
    from tsdynamics.analysis import dimensions

    assert "dimension_spectrum_plot_spec" not in dimensions.__all__
    assert "dimension_spectrum_plot_spec" not in dir(dimensions)
    assert callable(dimensions.dimension_spectrum_plot_spec)


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


# ── the miss is the other half of the curation ───────────────────────────────────
#
# Curating ``dir()`` hides ~230 reachable names (every built-in system, ~60
# analysis re-exports).  Autocomplete therefore stops being the channel that
# corrects a wrong guess, and this ``AttributeError`` becomes the only one.  It
# has to name the line to type.


def test_a_mistyped_system_points_at_the_systems_catalogue():
    """Systems are hidden from ``dir(ts)`` on purpose, so the miss must route there."""
    with pytest.raises(AttributeError, match=r"ts\.systems\.Lorenz\(\)"):
        ts.lorenz  # noqa: B018 - the access IS the assertion
    with pytest.raises(AttributeError, match=r"ts\.systems\.Rossler\(\)"):
        ts.Rosler  # noqa: B018


def test_a_mistyped_analysis_suggests_the_reachable_spelling():
    """A demoted-but-reachable name is suggested by its real spelling."""
    with pytest.raises(AttributeError, match=r"ts\.correlation_dimension"):
        ts.corelation_dimension  # noqa: B018


@pytest.mark.parametrize("name", ["permutation_entropy", "surrogate_test", "transforms"])
def test_a_v6_removed_name_explains_the_scope_change(name):
    """A v5 user must learn the scope changed, not read "no attribute" and assume a bad install."""
    with pytest.raises(AttributeError, match="removed in v6") as exc:
        getattr(ts, name)
    # ...and is pointed at what survived.
    assert "ts.analysis.recurrence" in str(exc.value)


def test_an_unrecognisable_name_names_the_two_catalogues():
    with pytest.raises(AttributeError, match=r"ts\.systems\.<TAB>") as exc:
        ts.zzz_definitely_not_a_name  # noqa: B018
    assert "ts.analysis.<TAB>" in str(exc.value)


def test_the_removed_name_map_only_lists_genuinely_removed_names():
    """Guard: a name that came back must leave ``_REMOVED_IN_V6`` or the error lies."""
    for name in ts._REMOVED_IN_V6:
        assert not hasattr(ts, name), f"{name} is reachable again — stop claiming it was removed"


def test_the_catalogue_count_in_the_error_is_live_not_hardcoded():
    """The fallback message counts systems at call time; a hardcoded total goes stale."""
    from tsdynamics import registry

    with pytest.raises(AttributeError) as exc:
        ts.zzz_definitely_not_a_name  # noqa: B018
    assert f"{sum(registry.families().values())} built-in systems" in str(exc.value)
