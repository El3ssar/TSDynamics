"""The v6 public surface, pinned — one file you can read to know the whole API.

`planning/api-v6/CONTRACT.md` §2 says "Every listing in §2 is a **contract**.  A
builder who produces a different listing has failed."  This module is where that
sentence acquires teeth.  It pins six invariants, each one the *inverse* of a way
the surface historically eroded:

===========================  =============================================
invariant                    the erosion it makes impossible
===========================  =============================================
**the listings** (§2)        a name creeping back onto a curated namespace
**one spelling** (C3/C4)     two names for one object; a verb answering
                             ``TypeError: 'module' object is not callable``
**plain Python** (C1)        a signature that *demands* a library type, which
                             is then exported "because the signature needs it"
**legible results** (§4.2)   a repr that prints a class name and no answer
**taught removals** (§5.5)   a removed name whose error names a spelling that
                             does not resolve either
**real signatures** (§3.1)   ``help()`` showing ``(*args, **kwargs)``, or a
                             return annotation naming a deleted type
===========================  =============================================

Three deliberate choices about how this file stays honest:

* **Doors are swept from the live library, never hand-listed.**  The region /
  plane / components sweeps read ``inspect.signature`` over
  ``ts.analysis.__all__``, so a new door joins the gate the moment it is
  registered.  Each sweep carries a call table and a guard test that fails when
  the table stops covering the live door set — over-selection is safe here,
  silence is not.
* **Not-yet-landed contract items are tracked as a DECLARED SET, not a skip.**
  Where a §2 listing is a binary property another slot owns, the row is a
  ``strict`` xfail naming that slot.  Where the defect is a *population* that
  must shrink (submodules shadowed by a same-named function; doors still
  spelling ``components=``), the gate asserts the population **equals** a declared
  table — so a new instance fails, and fixing one also fails until the row is
  deleted.  Neither can quietly become permanent.
* **Nothing here may pass vacuously.**  Every sweep asserts its own subject count
  first, so an empty registry or a renamed attribute fails loudly rather than
  certifying an empty set.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import pkgutil
import re
import types
import typing
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import _redirects, registry
from tsdynamics.errors import MovedInV6

# ---------------------------------------------------------------------------
# §2 — THE FINAL SURFACES.  These tuples are the contract, transcribed.
# ---------------------------------------------------------------------------

#: §2.1 — ``ts.<TAB>``.  Five classes you subclass, one type you receive, one
#: plotting verb, three registries, six names you type inside ``except``, and
#: ``__version__``.
TOP_LEVEL: tuple[str, ...] = (
    "BackendError",
    "ContinuousSystem",
    "ConvergenceError",
    "DelaySystem",
    "DiscreteMap",
    "InvalidInputError",
    "InvalidParameterError",
    "StepBudgetError",
    "StochasticSystem",
    "TSDynamicsError",
    "Trajectory",
    "WrappedSystem",
    "__version__",
    "analysis",
    "plot",
    "systems",
    "viz",
)

#: §2.2 — ``system.<TAB>``, the 19 names shared by every family.  Nothing adds to
#: this list; a family only ever *loses* names (:data:`FAMILY_ABSENCES`).
SYSTEM_CORE: tuple[str, ...] = (
    "copy",
    "dim",
    "ensemble",
    "family",
    "ic",
    "info",
    "jacobian",
    "jacobian_sym",
    "params",
    "plot",
    "poincare",
    "reinit",
    "run",
    "set_state",
    "state",
    "step",
    "time",
    "variables",
    "with_params",
)

#: §2.2 / §3.6 — what each family loses, and the mathematical reason.  A name that
#: cannot work does not exist; its ``AttributeError`` states why.
FAMILY_ABSENCES: dict[str, dict[str, str]] = {
    "ContinuousSystem": {},
    "DelaySystem": {
        "set_state": "a delay state is a whole history function, not a point",
        "jacobian": "the RHS reads the state at several past times, so df/du at "
        "one point is not evaluable from (u, t) alone",
        "jacobian_sym": "ditto — there is no single symbolic df/du to hand back",
    },
    "DiscreteMap": {
        "poincare": "a section crosses a CONTINUOUS trajectory; a map has no in-between",
        "jacobian_sym": "a map's kernel is traced numerically, not held as a symbolic tree",
    },
    "StochasticSystem": {
        "jacobian_sym": "a map's/SDE's kernel is traced, not held as a symbolic tree",
    },
    "WrappedSystem": {
        "jacobian": "it wraps an opaque stepper — there is no RHS to differentiate",
        "jacobian_sym": "ditto",
        # §2.2 renders WrappedSystem at 13; the six further absences are the
        # wrapper's own, measured below rather than asserted here.
    },
}

#: One representative per family, as ``(family class name, factory)``.
FAMILY_SUBJECTS: dict[str, Callable[[], Any]] = {
    "ContinuousSystem": lambda: ts.systems.Lorenz(),
    "DelaySystem": lambda: ts.systems.MackeyGlass(),
    "DiscreteMap": lambda: ts.systems.Henon(),
    "StochasticSystem": lambda: ts.systems.OrnsteinUhlenbeck(),
}

#: §2.3 — ``traj.<TAB>``.  Data (4), properties (4), methods (4).
TRAJECTORY_SURFACE: tuple[str, ...] = (
    "after",
    "dim",
    "dt",
    "meta",
    "plot",
    "sel",
    "shape",
    "system",
    "t",
    "to_frame",
    # v6 round 7: a run that blows up WITHOUT reaching the engine's 1e150 guard
    # used to come back as an ordinary finite trajectory, and every answer taken
    # from it looked exactly like an honest one.  This is the flag that tells
    # the two apart, and the repr prints it — a property that earns its slot by
    # being the difference between a result and a wrong number.
    "unbounded",
    "variables",
    "y",
)

#: §2.4 — the three verbs that sit beside the 50 analyses.
ANALYSIS_VERBS: tuple[str, ...] = ("find", "register", "results")

#: §2.5 — ``ts.viz.<TAB>``.
VIZ_SURFACE: tuple[str, ...] = (
    "Plot",
    # §11.3 T3 — PROMOTED in the visibility ruling (13 -> 14).  34 user-doc
    # mentions and 120 test uses, and its only address was the internal
    # ``ts.viz.render``; ``docs/visualization/styling.md`` worked around that by
    # comparing ``w[0].category.__name__`` to a *string*.  v6 promoted six
    # exception classes on exactly that argument.
    "VisualizationDegraded",
    "compatibility",
    "draw",
    "geometry",
    "grid",
    "load",
    "plot",
    "primitives",
    "renderers",
    "spec",
    "styles",
    "themes",
    "transforms",
)

#: §2.6 — the registry verbs every catalogue namespace answers.
REGISTRY_VERBS: tuple[str, ...] = ("find", "get", "names")

#: C5's declared set of public packages.  ``dir(M) == sorted(M.__all__)`` for
#: each, and no API name resolves on ``M`` unless it is listed.
PUBLIC_PACKAGES: tuple[str, ...] = (
    "tsdynamics",
    "tsdynamics.analysis",
    "tsdynamics.analysis.results",
    "tsdynamics.data",
    "tsdynamics.derived",
    "tsdynamics.engine",
    "tsdynamics.errors",
    "tsdynamics.families",
    "tsdynamics.registry",
    "tsdynamics.solvers",
    "tsdynamics.systems",
    "tsdynamics.utils",
    "tsdynamics.viz",
    "tsdynamics.viz.spec",
    # §11.3 T4 — the gap the ruling closed.  Measured: the fourteen-package gate
    # did not include the one viz namespace carrying 22 names, which is exactly
    # why that listing drifted while every swept one held.
    "tsdynamics.viz.transforms",
)


def public(obj: Any) -> list[str]:
    """The tab surface of *obj*: every non-underscored name ``dir()`` offers."""
    return sorted(n for n in dir(obj) if not n.startswith("_"))


def _module(name: str) -> types.ModuleType:
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# 1. THE LISTINGS (§2)
# ---------------------------------------------------------------------------


def test_the_top_level_is_the_seventeen_names_the_contract_names() -> None:
    """``ts.__all__`` is exactly §2.1, and ``dir()`` mirrors it.

    Both halves matter.  ``__all__`` governs ``from tsdynamics import *``;
    ``dir()`` governs tab-completion, which is what a user actually meets.  A
    curated surface that leaks through ``dir()`` is not curated.
    """
    assert tuple(sorted(ts.__all__)) == TOP_LEVEL
    assert dir(ts) == sorted(TOP_LEVEL)
    assert len(TOP_LEVEL) == 17


@pytest.mark.parametrize("pkg_name", PUBLIC_PACKAGES)
def test_dir_mirrors_all_for_every_public_package(pkg_name: str) -> None:
    """C5 — ``dir()`` is the truth: it equals ``sorted(__all__)``, and all of it resolves."""
    module = _module(pkg_name)
    names = getattr(module, "__all__", None)
    assert isinstance(names, list | tuple) and names, f"{pkg_name} declares no __all__"
    assert all(isinstance(n, str) for n in names)
    assert dir(module) == sorted(names), f"{pkg_name}: dir() and __all__ disagree"
    missing = [n for n in names if not hasattr(module, n)]
    assert not missing, f"{pkg_name}.__all__ advertises names that do not resolve: {missing}"


@pytest.mark.parametrize("family", sorted(FAMILY_SUBJECTS))
def test_the_family_tab_surface_is_the_core_minus_its_declared_absences(family: str) -> None:
    """§2.2 — every family answers ``SYSTEM_CORE`` minus the names it cannot have.

    The listing is *derived*, not four hand-written tuples: a name added to a
    family without a contract edit fails here, and a declared absence that is
    silently still present fails too.
    """
    system = FAMILY_SUBJECTS[family]()
    absent = FAMILY_ABSENCES[family]
    assert public(system) == sorted(set(SYSTEM_CORE) - set(absent))


@pytest.mark.parametrize("family", sorted(FAMILY_SUBJECTS))
def test_a_name_a_family_cannot_have_does_not_exist_at_all(family: str) -> None:
    """§3.6 — *a name that cannot work does not exist*, and ``hasattr`` says so.

    ``DelaySystem.set_state`` used to exist and raise ``NotImplementedError``: a
    member that lies.  The v6 contract is that it is absent, so a capability
    probe is a probe rather than a try/except.
    """
    system = FAMILY_SUBJECTS[family]()
    for name in FAMILY_ABSENCES[family]:
        assert not hasattr(system, name), f"{family}.{name} still exists"
        with pytest.raises(AttributeError) as excinfo:
            getattr(system, name)
        assert name in str(excinfo.value), f"{family}.{name}: the error does not name it"


def test_the_wrapped_system_surface_is_a_subset_of_the_core() -> None:
    """§2.2 — a ``WrappedSystem`` loses names; it never gains one.

    It is rendered at 13 rather than 19, and the exact absences are the
    wrapper's own business.  The invariant worth pinning is the direction: an
    opaque stepper may answer *less* than a system, never something new.
    """
    wrapped = ts.WrappedSystem(lambda u, dt: np.asarray(u), dim=2, family="map")
    extra = set(public(wrapped)) - set(SYSTEM_CORE)
    assert not extra, f"WrappedSystem answers names outside the core: {sorted(extra)}"
    for name in FAMILY_ABSENCES["WrappedSystem"]:
        assert not hasattr(wrapped, name)


@pytest.mark.xfail(
    strict=True,
    reason="C5 · TRAJECTORY — §8.1 item 8 (§2.3): traj.<TAB> is 19 names, not 12. "
    "`to_plot_spec` and the four analysis accessors HAVE landed; the rest "
    "(component/unpack/minmax/standardize/neighbors/n_steps/set_distance) STAY by the "
    "owner's ruling — the criterion is learnability, not smallness, and a useful name "
    "that costs a newcomer nothing is not the problem a second spelling is.",
)
def test_the_trajectory_tab_surface_is_twelve_names() -> None:
    """§2.3 — data (4), properties (4), methods (4), and nothing else.

    A ``Trajectory`` is "a glorified nd-array" in the owner's words, and 20 names
    is not that.  Three of the names that must go are *analysis accessors*
    (``dims`` / ``lyap`` / ``recurrence``) removed by ruling A2, and one
    (``n_steps``) is a second spelling of ``len(traj)``.
    """
    traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    assert public(traj) == sorted(TRAJECTORY_SURFACE)


def test_the_trajectory_surface_never_grows_past_the_contract() -> None:
    """The half of §2.3 that holds TODAY: no name may be ADDED to a ``Trajectory``.

    The removals are C5's (above).  This direction is enforceable now, and it is
    the direction that erodes: a helper lands on the class "just for this", and
    the 12-name listing is dead before it is written.
    """
    traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    known_pre_v6 = {
        "component",
        "dims",
        "lyap",
        "minmax",
        "n_steps",
        "neighbors",
        "recurrence",
        "set_distance",
        "standardize",
        "unpack",
    }
    unexpected = set(public(traj)) - set(TRAJECTORY_SURFACE) - known_pre_v6
    assert not unexpected, f"names added to Trajectory outside §2.3: {sorted(unexpected)}"


def test_the_systems_namespace_answers_the_registry_verbs() -> None:
    """§2.6 — 180 names: the catalogue plus ``names()`` / ``find()`` / ``get()``.

    Without them ``ts.systems`` is the one registry in the library you cannot
    search, which is precisely the namespace with 177 members.
    """
    missing = [verb for verb in REGISTRY_VERBS if not hasattr(ts.systems, verb)]
    assert not missing, f"ts.systems is missing {missing}"
    assert len(ts.systems.__all__) == 180


def test_the_analysis_listing_is_the_analyses_and_three_verbs() -> None:
    """§2.4 — flat, sorted, generated from the registry, 94 % callable analyses.

    The density is the point: today's predecessor was 84 names of which 32 were
    classes you never construct and 10 were modules that answered ``TypeError``.

    §2.4 was written at **50** analyses; round 8 closed ``max_lyapunov``, a
    second door onto ``lyapunov_spectrum``'s question that answered it with a
    different number (Hénon: 0.4233 against 0.4160 at one nominal horizon), so
    the count is **49**.  The number is asserted, not derived from the registry
    it is checking — a count that reads its own subject cannot fail.
    """
    names = list(ts.analysis.__all__)
    assert names == sorted(names), "ts.analysis.__all__ must be flat and sorted"
    assert dir(ts.analysis) == sorted(names)
    verbs = [n for n in names if n in ANALYSIS_VERBS]
    assert sorted(verbs) == sorted(ANALYSIS_VERBS)
    analyses = sorted(set(names) - set(ANALYSIS_VERBS))
    assert analyses == sorted(registry.analyses.names()), (
        "the listing must be generated from registry.analyses, never hand-written"
    )
    assert len(names) == 52, f"the analysis listing is 52 names, measured {len(names)}"
    assert len(analyses) == 49


def test_the_viz_listing_is_the_thirteen_names() -> None:
    """§2.5 — four registries, two drawing doors, one arranger, one received type."""
    assert tuple(sorted(ts.viz.__all__)) == VIZ_SURFACE
    assert dir(ts.viz) == sorted(VIZ_SURFACE)


def test_the_transforms_namespace_is_the_five_verb_shape() -> None:
    """§11.3 T3 — ``ts.viz.transforms`` is ``register`` / ``names`` / ``find`` / ``get`` / ``allow``.

    It used to be 22: the five verbs plus six front doors that are ``is``-identical
    to ``ts.viz.<name>``, five IR nouns that already live at ``ts.viz.spec``, and
    ``plot_transform``, which **is** ``register`` under a second name.  It is also
    the namespace :data:`PUBLIC_PACKAGES` never swept — which is precisely why it
    was the one that drifted, and why the row now exists.
    """
    module = _module("tsdynamics.viz.transforms")
    assert tuple(sorted(module.__all__)) == ("allow", "find", "get", "names", "register")
    assert "tsdynamics.viz.transforms" in PUBLIC_PACKAGES, (
        "the namespace is curated now — it must be inside the C5 gate that keeps it so"
    )


@pytest.mark.parametrize("name", ("primitives", "renderers", "themes", "transforms"))
def test_the_four_viz_registries_share_one_four_verb_shape(name: str) -> None:
    """§2.5 — ``register`` / ``names`` / ``find`` / ``get``, identically, on all four.

    One shape means a user who learned ``ts.viz.transforms.find(...)`` already
    knows the other three, and an extension author has one recipe.
    """
    reg = getattr(ts.viz, name)
    missing = [verb for verb in ("register", "names", "find", "get") if not hasattr(reg, verb)]
    assert not missing, f"ts.viz.{name} is missing {missing}"


def test_the_systems_listing_is_the_catalogue() -> None:
    """§2.6 — every built-in class is listed, and ``dir()`` mirrors ``__all__``."""
    classes = {e.name for e in registry.all_systems(builtin=True)}
    assert len(classes) == 177, f"the catalogue moved: {len(classes)} built-in systems"
    assert classes <= set(ts.systems.__all__)
    assert dir(ts.systems) == sorted(ts.systems.__all__)


@pytest.mark.parametrize("received", ("analysis.results", "viz.spec"))
def test_a_type_you_only_receive_lives_at_exactly_one_address(received: str) -> None:
    """C2 — the 32 result classes and the 19 IR nouns are reachable, and not on the top level.

    They are types you get *back*, never types you type, so they earn one
    importable address and no tab slot above it.
    """
    module = _module(f"tsdynamics.{received}")
    names = list(module.__all__)
    assert names == sorted(names)
    assert dir(module) == names
    leaked = [n for n in names if n in ts.__all__]
    assert not leaked, f"{received} names leaked onto the top level: {leaked}"


# ---------------------------------------------------------------------------
# 2. ONE SPELLING PER CONCEPT (C3, C4)
# ---------------------------------------------------------------------------

#: ``__all__`` entries that ARE modules, and the contract line that puts each
#: there.  C4 forbids a listed name being shadowed by a same-named submodule, so
#: every module-valued entry has to be one a user *means* as a module.
DECLARED_MODULE_EXPORTS: dict[str, dict[str, str]] = {
    "tsdynamics": {
        "analysis": "§2.1 — one of the three registries you tab into",
        "systems": "§2.1 — ditto",
        "viz": "§2.1 — ditto (bound lazily, so `import tsdynamics` pulls in no backend)",
    },
    "tsdynamics.analysis": {
        "results": "§2.7 — the 32 received types behind one dot (C2)",
    },
    "tsdynamics.viz": {
        "spec": "§2.5/§2.7 — the IR behind one dot; F2 records this as the one place "
        "C4 and §2.5 trade off, and §2.5 wins: a user typing ts.viz.spec means the module",
        "transforms": "§2.5 — one of the four registries; it is a module that also "
        "answers register/names/find/get",
    },
    # §2.6 — 180 names = 177 classes + names/find/get.  The two category
    # packages stay importable but are off the listing, like every other
    # internal submodule, so no __all__ entry is module-valued here.
    "tsdynamics.systems": {},
    "tsdynamics.engine": {
        # `engine` is reachable but off the top level's __all__ and flagged
        # internal in its own docstring (§2.1).  Its listing is four submodules
        # *because* that is what it offers: `ts.engine.run` means the module, so
        # the C4 hazard (a verb guess answered by a module) cannot arise.
        "compile": "§2.1 — engine is internal; its listing is the modules it holds",
        "problem": "§2.1 — ditto",
        "run": "§2.1 — ditto",
        "symbols": "§2.1 — ditto",
    },
}


@pytest.mark.parametrize("pkg_name", PUBLIC_PACKAGES)
def test_every_module_valued_export_is_one_the_contract_declares(pkg_name: str) -> None:
    """C4 — a listed name resolves to what it advertises.

    Asserted as set equality, so a submodule that quietly lands in a public
    ``__all__`` fails, and one that is correctly demoted fails until its row is
    deleted.  The failure mode being closed is ``ts.basins``, where the *alias*
    was a function while ``ts.analysis.basins`` was the *subpackage*: one name,
    two meanings, and whichever you got depended on where you typed it.
    """
    module = _module(pkg_name)
    found = {
        name
        for name in getattr(module, "__all__", ())
        if isinstance(getattr(module, name, None), types.ModuleType)
    }
    declared = set(DECLARED_MODULE_EXPORTS.get(pkg_name, {}))
    assert found == declared, (
        f"{pkg_name}: module-valued __all__ entries moved.\n"
        f"  newly listed (demote, or declare): {sorted(found - declared)}\n"
        f"  demoted (delete the row):          {sorted(declared - found)}"
    )


#: Exported names that are the SAME OBJECT as another exported name, each with the
#: contract line that sanctions it.  Two spellings for one concept is the
#: silent-wrong-answer defect, so this table may only shrink — and
#: :func:`test_the_declared_aliases_are_still_aliases` fails when a row goes stale.
#: **It is empty, and that is the target state.**  Its one row —
#: ``ts.derived.EnsembleSystem``, an alias of ``Ensemble`` — was ruled out in
#: §11.3 T2 and the listing edit landed, so the row was deleted here in the same
#: commit.  The teeth are in :func:`test_no_two_exported_names_are_the_same_object`,
#: which sweeps all fifteen public packages and is the test that cannot pass
#: vacuously; this table is only the sanctioned-exception door, and an empty
#: door is the whole point.  Re-opening it costs a written contract line.
DECLARED_ALIASES: dict[tuple[str, str], str] = {}


def _api_objects(module: types.ModuleType) -> dict[str, Any]:
    """``__all__`` entries that are *API objects* — functions, classes, modules.

    Scalars are excluded deliberately: ``utils`` exports ``DEFAULT_RTOL`` and
    ``DDE_LYAPUNOV_ATOL`` as ``1e-9``, and CPython folds equal float literals in
    one module to one object.  Sharing a *value* is not two spellings of one
    concept; sharing a *function* is.
    """
    out: dict[str, Any] = {}
    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name, None)
        if callable(obj) or isinstance(obj, types.ModuleType):
            out[name] = obj
    return out


@pytest.mark.parametrize("pkg_name", PUBLIC_PACKAGES)
def test_no_two_exported_names_are_the_same_object(pkg_name: str) -> None:
    """C3 — one concept, one spelling, measured by object identity.

    An alias is not flexibility: a shared implementation can name only one of its
    spellings in an error message, so half of all callers get answered about a
    function they never typed.  That is exactly why ``bifurcation_diagram`` was
    deleted.
    """
    module = _module(pkg_name)
    by_id: dict[int, list[str]] = {}
    for name, obj in _api_objects(module).items():
        by_id.setdefault(id(obj), []).append(name)
    duplicates = {tuple(sorted(names)): names for names in by_id.values() if len(names) > 1}
    undeclared = [
        names
        for names in duplicates.values()
        if not any((pkg_name, n) in DECLARED_ALIASES for n in names)
    ]
    assert not undeclared, (
        f"{pkg_name} exports one object under several names: {undeclared}. "
        f"Demote all but one and add a row to _redirects.py."
    )


def test_the_declared_aliases_are_still_aliases() -> None:
    """A sanctioned alias that stopped being one must leave the table.

    Self-cleaning, like the doctest exemptions: the escape hatch can only shrink.
    """
    stale: list[str] = []
    for (pkg_name, name), reason in DECLARED_ALIASES.items():
        module = _module(pkg_name)
        obj = getattr(module, name, None)
        siblings = [n for n, o in _api_objects(module).items() if o is obj and n != name]
        if not siblings:
            stale.append(f"{pkg_name}.{name} ({reason.split(':')[0]})")
    assert not stale, f"DECLARED_ALIASES rows that are no longer aliases — delete them: {stale}"


def test_ts_plot_is_the_one_plot_function() -> None:
    """§2.5 — ``ts.plot is ts.viz.plot``, exactly one function object.

    The front door used to shadow a *different* ``compose.plot`` that accepted
    ``cols=`` while ``ts.plot`` refused it: one name, two functions, two
    vocabularies.
    """
    assert ts.plot is ts.viz.plot
    assert callable(ts.plot)


#: Every kind of thing that can be a plot *subject*, with the noun its refusal
#: message uses.  ``ts.plot`` recognises a subject by ONE predicate —
#: ``__plot_spec__`` — so this list is also the list of things that must carry it.
PLOT_SUBJECTS: dict[str, Callable[[], Any]] = {
    "system": lambda: ts.systems.Lorenz(),
    "trajectory": lambda: _traj(),
    "result": lambda: ts.analysis.lyapunov_spectrum(ts.systems.Henon(), steps=2000),
    "array-result": lambda: ts.analysis.fixed_points(ts.systems.Henon()),
    "derived": lambda: ts.systems.Rossler().poincare("y", 0.0),
}


@pytest.mark.parametrize("subject", sorted(PLOT_SUBJECTS))
def test_building_a_plot_has_exactly_one_public_door(subject: str) -> None:
    """§6.2 — ``to_plot_spec`` is retired; ``ts.plot(x)`` already returns the ``Plot``.

    The owner's ruling: *"it has to be easy to make a plot, not several different
    ways to do so."*  Three spellings built a plot without drawing it —
    ``ts.plot(x)``, ``x.plot()`` and ``x.to_plot_spec()`` — and the first two
    cover every case, so the third was a choice a newcomer had to make and could
    not make wrongly enough to learn anything.  The seam survives as the dunder
    ``__plot_spec__``, which is what ``ts.plot`` asks every subject for.
    """
    thing = PLOT_SUBJECTS[subject]()
    assert hasattr(thing, "__plot_spec__"), f"{subject} lost the plot seam"
    assert not hasattr(thing, "to_plot_spec"), f"{subject} still answers to_plot_spec"
    assert isinstance(ts.plot(thing), ts.viz.Plot)
    assert isinstance(thing.plot(), ts.viz.Plot)


@pytest.mark.parametrize("subject", sorted(PLOT_SUBJECTS))
def test_the_retired_plot_name_hands_back_a_spelling_that_resolves(subject: str) -> None:
    """§9.4 — a message offering ``ts.<something>`` must NAME SOMETHING THAT RUNS.

    Every line the refusal prints is executed here, on the very object that was
    held when the guess was made.
    """
    thing = PLOT_SUBJECTS[subject]()
    with pytest.raises(AttributeError) as excinfo:
        _ = thing.to_plot_spec
    message = str(excinfo.value)
    assert "ts.plot(" in message and ".plot()" in message
    assert "__plot_spec__" in message
    # The two offered lines, run.
    assert isinstance(ts.plot(thing), ts.viz.Plot)
    assert isinstance(thing.plot(), ts.viz.Plot)


def test_building_a_plot_renders_nothing() -> None:
    """§6.2 — the capability ``to_plot_spec`` existed for did not move: it is ``ts.plot``.

    ``ts.plot(x)`` describes the picture and stops; drawing happens at ``.fig`` /
    ``.render()`` / ``.save()`` / ``.show()``.  Pinned by counting real
    ``matplotlib.figure.Figure`` constructions, because a plotting front door
    that quietly rasterises is the difference between a notebook that scrolls and
    one that hangs.
    """
    pytest.importorskip("matplotlib")
    figure_module = importlib.import_module("matplotlib.figure")
    pyplot = importlib.import_module("matplotlib.pyplot")

    built: list[int] = []
    original = figure_module.Figure.__init__

    def counting_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        built.append(1)
        return original(self, *args, **kwargs)

    traj = _traj()
    figure_module.Figure.__init__ = counting_init  # type: ignore[method-assign]
    try:
        spec = ts.plot(traj)
        spec = spec.style(lw=2).relabel(title="held")
        assert traj.plot(color="crimson") is not None
        assert ts.viz.grid(spec, ts.plot(traj), cols=2) is not None
        assert not built, f"{len(built)} figure(s) drawn before anyone asked"
        assert spec.fig is not None
        assert built, "touching .fig must actually draw"
    finally:
        figure_module.Figure.__init__ = original  # type: ignore[method-assign]
        pyplot.close("all")


#: Submodules that a same-named object in their parent's namespace shadows, so
#: ``import pkg.mod as alias`` binds the WRONG thing.  §5.4 promises the
#: subpackages stay importable; ``import a.b.c as n`` ends in a ``getattr``
#: chain, so a function named like its module breaks that promise silently.
#: Each row names the slot that owns the fix (§9.5).  The table may only shrink.
SHADOWED_SUBMODULES: dict[str, str] = {
    "tsdynamics.analysis.basins": "S3 — §9.5 row (c): the one collection error left in the suite",
    "tsdynamics.analysis.basins.attractors": "S3 — same defect one level down",
    "tsdynamics.analysis.basins.basins": "S3 — same defect one level down",
    "tsdynamics.analysis.basins.continuation": "S3 — same defect one level down",
    "tsdynamics.analysis.chaos.gali": "S3 — §9.5 asks for this check on every area",
    "tsdynamics.analysis.embedding.embed": "S3 — §9.5 asks for this check on every area",
    "tsdynamics.analysis.orbits.orbit_diagram": "S3 — §9.5 asks for this check on every area",
    "tsdynamics.analysis.orbits.return_map": "S3 — §9.5 asks for this check on every area",
    "tsdynamics.analysis.recurrence.rqa": "S3 — §9.5 asks for this check on every area",
    "tsdynamics.solvers.select": "C6 — solvers/** is C6's for v6 (§9.4 ownership gaps)",
}


def _iter_public_submodules() -> Iterator[str]:
    """Every importable non-underscored module under ``tsdynamics``."""
    for info in pkgutil.walk_packages(ts.__path__, "tsdynamics."):
        if any(part.startswith("_") for part in info.name.split(".")[1:]):
            continue
        yield info.name


def test_no_submodule_is_shadowed_by_a_same_named_object() -> None:
    """C4 — ``import tsdynamics.analysis.recurrence.rqa as m`` must bind the MODULE.

    Measured: it binds the *function* ``rqa``, because ``import a.b.c as n``
    resolves through ``getattr`` and the package ``__init__`` re-exports a
    function of the module's own name.  The same shape that got ``ts.basins``
    deleted, recreated one level down — and it is why
    ``import tsdynamics.analysis.basins.metrics as bas`` raises ``ImportError``.

    Asserted as set equality so the table is self-cleaning in both directions: a
    new shadow fails, and a fixed one fails until its row is deleted.
    """
    found: dict[str, str] = {}
    for name in _iter_public_submodules():
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — an unimportable module is another gate's failure
            continue
        parent_name, _, leaf = name.rpartition(".")
        got = getattr(_module(parent_name), leaf, None)
        if got is not None and not isinstance(got, types.ModuleType):
            found[name] = type(got).__name__
    assert len(found) < 40, "sanity: the sweep found implausibly many shadows"
    assert set(found) == set(SHADOWED_SUBMODULES), (
        "shadowed submodules moved.\n"
        f"  newly shadowed (fix, or add a row): {sorted(set(found) - set(SHADOWED_SUBMODULES))}\n"
        f"  fixed (delete the row):             {sorted(set(SHADOWED_SUBMODULES) - set(found))}"
    )


@pytest.mark.parametrize(
    "area",
    (
        "basins",
        "chaos",
        "dimensions",
        "embedding",
        "fixedpoints",
        "lyapunov",
        "orbits",
        "recurrence",
        "sampling",
    ),
)
def test_no_capability_subpackage_answers_on_ts_analysis(area: str) -> None:
    """§5.4 — the implementation packages are unbound; only free functions answer.

    ``ts.analysis.lyapunov`` used to be a *module*, so a user who guessed the verb
    got ``TypeError: 'module' object is not callable`` — a message about Python,
    not about dynamics.  ``basins`` is the exception: it is a real analysis now,
    and must therefore be the FUNCTION.
    """
    value = getattr(ts.analysis, area, None)
    if area in registry.analyses.names():
        assert callable(value) and not isinstance(value, types.ModuleType)
        return
    assert value is None or not isinstance(value, types.ModuleType), (
        f"ts.analysis.{area} still resolves to the implementation package"
    )
    assert importlib.import_module(f"tsdynamics.analysis.{area}") is not None


# ---------------------------------------------------------------------------
# 3. PLAIN PYTHON REACHES EVERY DOOR (C1)
# ---------------------------------------------------------------------------
#
# The toll rule: tuples, lists, strings, numbers and ndarrays reach every front
# door.  A name exported *because a signature demands it* is evidence of a
# signature bug, not of a needed export — which is why ``Box`` / ``Ball`` /
# ``Grid`` / ``Theme`` / ``Geometry`` are all demoted.  The only way to know the
# toll is gone is to pay plain Python at every door and watch it open, so these
# sweeps CALL the functions.

_REGION = [(-2.0, 2.0), (-2.0, 2.0)]
_GRID_REGION = [(-2.0, 2.0, 8), (-2.0, 2.0, 8)]


def _henon() -> Any:
    return ts.systems.Henon()


def _vdp() -> Any:
    return ts.systems.VanDerPol()


#: door name -> a call using ONLY plain Python for the region.  Kept small on
#: purpose (8x8 grids, 3 parameter values) so the gate is an inner-loop test.
REGION_CALLS: dict[str, Callable[[], Any]] = {
    "attractors": lambda: ts.analysis.attractors(_henon(), _GRID_REGION),
    "basin_fractions": lambda: ts.analysis.basin_fractions(_henon(), _REGION, n_seeds=20, seed=0),
    "basins": lambda: ts.analysis.basins(_henon(), _GRID_REGION),
    "continuation": lambda: ts.analysis.continuation(
        _henon(), "a", [1.2, 1.25], [(-2.0, 2.0, 6), (-2.0, 2.0, 6)]
    ),
    "expansion_entropy": lambda: ts.analysis.expansion_entropy(
        _henon(), region=[(-1.0, 1.0), (-1.0, 1.0)], n_samples=60, n=6, seed=0
    ),
    # The six windowed FIELD analyses read the same region grammar since v6
    # round 7 — and read it POSITIONALLY, so the call is spelled exactly like
    # ``basins(system, region)``.  Before that, ``ts.analysis`` had two grammars
    # for "this box of state space" and crossing them gave a bare
    # ``TypeError: takes 1 positional argument but 2 were given``.
    "escape_time_field": lambda: ts.analysis.escape_time_field(
        _vdp(), [(-2.0, 2.0, 8), (-2.0, 2.0, 8)], final_time=1.0
    ),
    "fixed_points": lambda: ts.analysis.fixed_points(_henon(), region=_REGION, seed=0),
    "flow_field": lambda: ts.analysis.flow_field(_vdp(), [(-2.0, 2.0, 8), (-2.0, 2.0, 8)]),
    "ftle_field": lambda: ts.analysis.ftle_field(
        _vdp(), [(-2.0, 2.0, 8), (-2.0, 2.0, 8)], final_time=1.0
    ),
    "nullclines": lambda: ts.analysis.nullclines(_vdp(), [(-2.0, 2.0, 15), (-2.0, 2.0, 15)]),
    "streamlines": lambda: ts.analysis.streamlines(_vdp(), [(-2.0, 2.0, 4), (-2.0, 2.0, 4)]),
    "transient_time_field": lambda: ts.analysis.transient_time_field(
        _vdp(), [(-2.0, 2.0, 8), (-2.0, 2.0, 8)], final_time=1.0
    ),
    "periodic_orbits": lambda: ts.analysis.periodic_orbits(
        _henon(), period=1, region=_REGION, seed=0
    ),
}

#: door name -> a call naming the plane with plain strings.
PLANE_CALLS: dict[str, Callable[[], Any]] = {
    "escape_time_field": lambda: ts.analysis.escape_time_field(
        _vdp(), plane=("x", "y"), grid=8, final_time=1.0
    ),
    "flow_field": lambda: ts.analysis.flow_field(_vdp(), plane=("x", "y"), grid=8),
    "ftle_field": lambda: ts.analysis.ftle_field(_vdp(), plane=("x", "y"), grid=8, final_time=1.0),
    "nullclines": lambda: ts.analysis.nullclines(_vdp(), plane=("x", "y"), grid=15),
    "poincare_section": lambda: ts.analysis.poincare_section(
        ts.systems.Rossler(), plane=("y", 0.0), crossings=5, seed=0
    ),
    "return_map": lambda: ts.analysis.return_map(_traj(), plane=("z", 25.0)),
    "streamlines": lambda: ts.analysis.streamlines(_vdp(), plane=("x", "y")),
    "trace_determinant": lambda: ts.analysis.trace_determinant(_vdp(), plane=("x", "y")),
    "transient_time_field": lambda: ts.analysis.transient_time_field(
        ts.systems.Lorenz(), plane=("x", "y"), grid=8, final_time=1.0
    ),
}

_traj_cache: Any = None


def _traj() -> Any:
    """One short Lorenz orbit, built once — the data subject for the door sweeps."""
    global _traj_cache
    if _traj_cache is None:
        _traj_cache = ts.systems.Lorenz().run(final_time=8.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    return _traj_cache


def _doors_taking(parameter: str) -> list[str]:
    """Public analyses whose signature binds *parameter*, read live."""
    out: list[str] = []
    for name in ts.analysis.__all__:
        obj = getattr(ts.analysis, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover — defensive
            continue
        if parameter in sig.parameters:
            out.append(name)
    return out


def test_the_region_and_plane_call_tables_cover_every_live_door() -> None:
    """A new ``region=`` / ``plane=`` door joins this gate, or fails it.

    Without this the sweeps below would certify whichever doors somebody
    remembered — the exact way a curated list goes stale.
    """
    for parameter, table in (("region", REGION_CALLS), ("plane", PLANE_CALLS)):
        live = set(_doors_taking(parameter))
        assert live, f"no public analysis takes {parameter}= — the sweep would be vacuous"
        missing = live - set(table)
        assert not missing, (
            f"analyses taking {parameter}= with no plain-Python call: {sorted(missing)}"
        )
        stale = set(table) - live
        assert not stale, f"{parameter} table names doors that are gone: {sorted(stale)}"


@pytest.mark.parametrize("door", sorted(REGION_CALLS))
def test_every_region_door_opens_for_plain_tuples(door: str) -> None:
    """C1 — ``[(lo, hi), (lo, hi)]`` is a region, everywhere.

    One ``(lo, hi[, n])`` pair per state component is *the* reading (§2, the
    demotion of ``Box``/``Ball``/``Grid`` rests on it).  A door that insisted on
    the type would put those three names back on the top level.
    """
    import warnings

    with warnings.catch_warnings():
        if door in _NUMERICALLY_NOISY_DOORS:
            warnings.simplefilter("ignore", RuntimeWarning)
        result = REGION_CALLS[door]()
    assert result is not None


#: Field doors whose *numerics* warn on a short, small-lattice call — the
#: arrival test on ``transient_time_field`` legitimately reports "almost nothing
#: settled" at ``final_time=1.0``.  The gate is about the DOOR opening for plain
#: Python, not about the answer, so the benign RuntimeWarning is allowed here and
#: nowhere else (the suite runs under ``filterwarnings = error``).
_NUMERICALLY_NOISY_DOORS: frozenset[str] = frozenset({"transient_time_field"})


@pytest.mark.parametrize("door", sorted(PLANE_CALLS))
def test_every_plane_door_opens_for_plain_names(door: str) -> None:
    """C1 — a section is ``("y", 0.0)`` or ``("x", "y")``: component names and numbers."""
    import warnings

    with warnings.catch_warnings():
        if door in _NUMERICALLY_NOISY_DOORS:
            warnings.simplefilter("ignore", RuntimeWarning)
        result = PLANE_CALLS[door]()
    assert result is not None


def test_the_section_verb_on_a_system_takes_plain_python() -> None:
    """§3.2 — ``sys.poincare("y", 0.0)`` and the direction word, with no type to build.

    And the absorbed strobe: ``poincare(period=T)`` where ``T`` is a float.
    """
    ros = ts.systems.Rossler()
    assert ros.poincare("y", 0.0) is not None
    assert ros.poincare(("y", 0.0, "up")) is not None
    assert ts.systems.Duffing().poincare(period=4.488) is not None
    with pytest.raises(ts.InvalidParameterError) as excinfo:
        ros.poincare("y", 0.0, period=1.0)
    assert "period" in str(excinfo.value)


#: §5.7 / M38 — ``components=`` is the ONE spelling.  **Empty since round 4**:
#: the nine analysis doors and every plot transform were renamed together, so a
#: user meets one word at every door.  Set equality, so the table is
#: self-cleaning: a new singular door fails this gate.
COMPONENT_SINGULAR_DOORS: frozenset[str] = frozenset()


def test_components_is_the_one_spelling_and_the_singular_is_a_shrinking_set() -> None:
    """C3 / M38 — ``components=`` names *which* component(s), never *how many*.

    Two grammars for one argument is the silent-wrong-answer defect: measured,
    ``estimate_period(components=...)`` sliced a *row* and returned 0.026 where the
    truth was 8.0 — a 311x error with no exception.
    """
    singular = set(_doors_taking("component"))
    plural = set(_doors_taking("components"))
    assert plural, "no door spells it components= — the contract's spelling vanished"
    assert singular == set(COMPONENT_SINGULAR_DOORS), (
        "the components= population moved.\n"
        f"  newly singular (rename it):  {sorted(singular - COMPONENT_SINGULAR_DOORS)}\n"
        f"  renamed (delete the row):    {sorted(COMPONENT_SINGULAR_DOORS - singular)}"
    )


@pytest.mark.parametrize("door", ("estimate_period", "zero_one_test", "return_map"))
def test_a_components_door_takes_a_plain_component_name(door: str) -> None:
    """C1 — you name a component with the string the system declares, not an index."""
    subject = _traj() if door != "zero_one_test" else ts.systems.Henon().run(2000, transient=200)
    result = getattr(ts.analysis, door)(subject, components="x")
    assert result is not None


def test_a_trajectory_selects_components_with_plain_strings() -> None:
    """§4.1 — strings select columns; everything else selects rows."""
    traj = _traj()
    assert traj["x"].shape == (traj.y.shape[0],)
    picked = traj["x", "z"]
    assert getattr(picked, "y", picked).shape[1] == 2


# ---------------------------------------------------------------------------
# 4. EVERY RESULT IS LEGIBLE (§4.2, §4.3)
# ---------------------------------------------------------------------------

#: Words that make a repr a verdict rather than a dump.
_VERDICT = re.compile(
    r"chaotic|regular|hyperchaotic|stable|unstable|deterministic|none found|"
    r"not applicable|indeterminate|sensitive|diverged|wada|untrusted",
    re.IGNORECASE,
)


@functools.cache
def _result_examples() -> dict[str, Any]:
    """One realistic instance of every ``AnalysisResult`` subclass, built once.

    ``_result_fixtures.build()`` constructs all 32 from numbers taken from real
    runs; cached because three parametrized sweeps below ask for it 96 times.
    """
    from _result_fixtures import build

    return build()


def _result_names() -> list[str]:
    return sorted(_result_examples())


@pytest.mark.parametrize("name", _result_names())
def test_every_result_repr_states_the_answer(name: str) -> None:
    """§4.2 r3 — ``summary()`` is deleted and ``__repr__`` became what it printed.

    The good text already existed inside ``summary()``, which nothing advertised;
    a bare ``LyapunovSpectrum(values=array([...]))`` at the REPL made the user do
    the reading.  So the headline must carry the ANSWER — a number, a count, or a
    verdict — and not merely the class name.
    """
    result = _result_examples()[name]
    head = repr(result).splitlines()[0]
    assert head.strip(), f"{name}: empty repr"
    rest = head.replace(type(result).__name__, "", 1).strip()
    if name == "AnalysisResult":
        # The abstract base is returned by no registered analysis; it has no
        # answer to state, only a subject.  Every subclass below is held to it.
        assert rest
        return
    assert rest, f"{name}: the repr is only the class name"
    assert re.search(r"\d", rest) or _VERDICT.search(rest), (
        f"{name}: the headline states no number, count or verdict — {head!r}"
    )


@pytest.mark.parametrize("name", _result_names())
def test_every_result_prints_its_whole_answer(name: str) -> None:
    """§4.2 r12 — ``print(result)`` answers as fully as the REPL does.

    ``str`` was the headline alone, so ``print`` dropped the supporting lines —
    the attractor locations, the fit window, the caveat — on every result that
    has them, and scripts are written with ``print``.  ``result.headline`` is the
    one-line form, and it is what an f-string embeds.

    ``CountResult`` is the one stated exception: its ``__str__`` is
    ``repr(int(self))``, because ``int.__str__ is object.__str__`` makes assigning
    it a measured no-op, and ``f"tau={c}"`` printing ``tau=CountResult(28)`` is
    committed to the repository inside ``docs/assets/figures/analysis/embedding.svg``.
    """
    result = _result_examples()[name]
    if name == "CountResult":
        assert str(result) == repr(int(result))
        assert repr(result).splitlines()[0] != str(result)
        return
    assert str(result) == repr(result)
    assert result.headline == repr(result).splitlines()[0]


@pytest.mark.parametrize("name", _result_names())
def test_no_result_advertises_summary(name: str) -> None:
    """§8.3 — ``summary()`` is DELETED outright, not deprecated.

    Two spellings of one rendering is C3; keeping the loser reachable is how the
    repr stays the second-best thing to look at.
    """
    assert not hasattr(_result_examples()[name], "summary")


def test_every_result_class_is_reachable_at_exactly_one_address() -> None:
    """C2 — ``ts.analysis.results.<Name>``, and nowhere on the top level."""
    examples = _result_examples()
    assert len(examples) == 32, f"§2.7 says 32 result classes, fixtures build {len(examples)}"
    results_mod = _module("tsdynamics.analysis.results")
    for name, result in examples.items():
        assert name in results_mod.__all__, f"{name} is not exported at ts.analysis.results"
        assert getattr(results_mod, name) is type(result)
        assert name not in ts.__all__


def test_a_numeric_result_is_a_complete_number() -> None:
    """§4.2 r4/r5 — arithmetic, comparison and ``np.asarray`` all work on the answer."""
    spectrum = _result_examples()["LyapunovSpectrum"]
    assert np.asarray(spectrum).dtype.kind == "f"
    scalar = _result_examples()["ScalarResult"]
    assert float(scalar * 2) == pytest.approx(2 * float(scalar))
    assert float(scalar + 1) == pytest.approx(float(scalar) + 1)
    assert (scalar > 0) is True


# ---------------------------------------------------------------------------
# 5. EVERY REMOVED NAME TEACHES ITS REPLACEMENT (§5.5, §5.6)
# ---------------------------------------------------------------------------
#
# Curation hides ~260 reachable names from autocomplete, so a wrong guess at
# ``ts.<name>`` is the only feedback a user gets.  The AttributeError IS the
# migration guide — which means a message naming a spelling that does not resolve
# is worse than no message at all (M37).

_RUNNABLE = re.compile(r"\bts(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _resolves(path: str) -> bool:
    """Walk a dotted ``ts.a.b`` path and report whether every hop exists."""
    obj: Any = ts
    for part in path.split(".")[1:]:
        try:
            obj = getattr(obj, part)
        except Exception:  # noqa: BLE001 — a MovedInV6 on the way is still a failure
            return False
    return True


def _remedies(message: str, typed: str = "") -> list[str]:
    """Every ``ts.a.b`` path a message hands back, minus the name that was typed.

    The first line quotes what the user wrote (``ts.PlotSpec``), which is by
    definition the one path that does *not* resolve — that is why they are
    reading this message.
    """
    found = _RUNNABLE.findall(message)
    return [r for r in found if r != f"ts.{typed}"]


def _teaching_message(name: str) -> str:
    with pytest.raises((AttributeError, MovedInV6)) as excinfo:
        getattr(ts, name)
    return str(excinfo.value)


@pytest.mark.parametrize("name", sorted(_redirects.RENAMED_IN_V6))
def test_every_renamed_name_hands_back_a_spelling_that_resolves(name: str) -> None:
    """§5.5 case 1 — a rename answers with the new spelling, and the new spelling works."""
    message = _teaching_message(name)
    assert name in message
    remedies = _remedies(message, typed=name)
    assert remedies, f"ts.{name} teaches nothing runnable: {message!r}"
    unresolvable = [r for r in remedies if not _resolves(r)]
    assert not unresolvable, f"ts.{name} hands back names that do not resolve: {unresolvable}"


@pytest.mark.parametrize("name", sorted(_redirects.REMOVED_IN_V6))
def test_every_removed_name_hands_back_a_spelling_that_resolves(name: str) -> None:
    """§5.5 case 2 — scope surgery answers with what SURVIVED, not with a module path.

    The live defect this closes: ``ts.permutation_entropy`` used to answer with
    ``ts.analysis.recurrence`` / ``.embedding`` / ``.lyapunov`` — module paths,
    which are not runnable lines and which stop resolving under §5.4.
    """
    message = _teaching_message(name)
    assert name in message
    remedies = _remedies(message, typed=name)
    assert remedies, f"ts.{name} teaches nothing runnable: {message!r}"
    unresolvable = [r for r in remedies if not _resolves(r)]
    assert not unresolvable, f"ts.{name} hands back names that do not resolve: {unresolvable}"


#: The public homes the address book searches (§5.5 ``_home_of``).
_HOMES = ("systems", "analysis", "analysis.results", "data", "derived", "viz", "viz.spec")


def _named_object(path: str) -> Any:
    obj: Any = ts
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("home", _HOMES)
def test_every_demoted_name_is_answered_with_the_address_it_lives_at(home: str) -> None:
    """§5.5 case 3 — an EXACT hit in a public ``__all__`` outranks every fuzzy guess.

    Without this ordering a name that merely moved was answered with nonsense:
    ``ts.region`` (real, at ``ts.data.region``) suggested ``ts.systems.Oregonator()``
    and ``ts.Region`` suggested ``ts.engine``.  Swept over the whole address book —
    ~260 names — because one wrong answer is one user who concludes the library
    cannot do it.

    The assertion is on the OBJECT, not on a string: ``set_distance`` is public at
    both ``ts.data`` and ``ts.analysis`` (one object, two addresses, both
    sanctioned), so demanding a particular home would be testing the search order
    rather than the answer.
    """
    module = _module(f"tsdynamics.{home}")
    names = [n for n in module.__all__ if n not in ts.__all__]
    assert names, f"{home} contributes nothing to the address book"
    wrong: list[str] = []
    for name in names:
        wanted = getattr(module, name, None)
        if isinstance(wanted, types.ModuleType):
            continue  # a submodule export, reached as a module, not a demoted name
        message = _teaching_message(name)
        offered = [
            r
            for r in _remedies(message, typed=name)
            if r.rsplit(".", 1)[-1] == name and _resolves(r)
        ]
        if not any(_named_object(r) is wanted for r in offered):
            wrong.append(f"{name}: answered {offered or message.splitlines()[-1].strip()!r}")
    assert not wrong, "names not answered with the address they live at:\n  " + "\n  ".join(
        wrong[:10]
    )


@pytest.mark.parametrize(
    "name", ("Lorenz", "basins_of_attraction", "permutation_entropy", "PlotSpec")
)
def test_a_redirect_survives_the_import_spelling(name: str) -> None:
    """C6 — ``from tsdynamics import X`` keeps the text; an ``AttributeError`` loses it.

    Measured on CPython: a module ``__getattr__`` raising ``AttributeError`` is
    rewritten by the import machinery to ``cannot import name 'X' from 'pkg'``,
    discarding the message.  Raising ``ImportError`` propagates it verbatim — and
    the corpus uses ``from tsdynamics import X`` 111 times, so this is the
    spelling that matters.
    """
    with pytest.raises(ImportError) as excinfo:
        exec(f"from tsdynamics import {name}")  # noqa: S102 — the spelling under test
    message = str(excinfo.value)
    assert name in message
    assert "cannot import name" not in message, "the redirect text was discarded"
    assert _remedies(message), f"the import-spelling answer teaches nothing runnable: {message!r}"


def test_a_guess_is_still_an_attribute_error_so_hasattr_keeps_working() -> None:
    """§5.5 cases 4-5 — only an EXACT table hit is an ``ImportError``.

    If a guess raised ``ImportError`` too, ``hasattr(ts, anything)`` would raise
    for every name in the universe.
    """
    assert not hasattr(ts, "random_typo_that_is_not_a_name")
    with pytest.raises(AttributeError):
        _ = ts.random_typo_that_is_not_a_name
    assert getattr(ts, "another_typo", "default") == "default"


def test_the_error_a_deleted_accessor_raises_teaches_the_free_function() -> None:
    """§5.6 — analyses came off the object; the object door must say so, and point.

    ``sys.chaos.zero_one()`` pre-ran with the family's ``run()`` defaults and
    reported K = -0.026 for Lorenz against the free function's 0.999.  A
    convenience that hides a sampling choice is a silent-wrong-answer generator,
    so the accessors are gone — and the ``AttributeError`` is the whole migration.
    """
    lor = ts.systems.Lorenz()
    for name in ("lyap", "chaos", "dims", "recurrence", "lyapunov_spectrum"):
        with pytest.raises(AttributeError) as excinfo:
            getattr(lor, name)
        message = str(excinfo.value)
        assert name in message
        remedies = _remedies(message)
        assert remedies, f"lor.{name} teaches nothing runnable: {message!r}"
        assert any(r.startswith("ts.analysis") for r in remedies), (
            f"lor.{name} does not point at ts.analysis: {message!r}"
        )


# ---------------------------------------------------------------------------
# 6. help() SHOWS A REAL SIGNATURE (§3.1)
# ---------------------------------------------------------------------------

#: §3.1's why-table, transcribed: the keywords each family's ``run`` binds.
#: ``run`` used to be a ``**kwargs`` passthrough, so the library's most-typed call
#: showed 2 of ~12 keywords to ``help()``; ``DelaySystem.run`` accepted and
#: silently DROPPED ``max_step``, ``t0``, ``events`` and ``nonsense_kw`` alike.
RUN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ContinuousSystem": (
        "final_time",
        "dt",
        "t0",
        "ic",
        "transient",
        "solver",
        "rtol",
        "atol",
        "max_step",
        "backend",
        "seed",
        "events",
    ),
    "DelaySystem": (
        "final_time",
        "dt",
        "ic",
        "history",
        "transient",
        "solver",
        "rtol",
        "atol",
        "backend",
        "seed",
    ),
    "DiscreteMap": ("steps", "ic", "transient", "backend", "seed", "max_retries"),
    "StochasticSystem": (
        "final_time",
        "dt",
        "t0",
        "ic",
        "transient",
        "solver",
        "seed",
        "backend",
    ),
}


@pytest.mark.parametrize("family", sorted(RUN_KEYWORDS))
def test_every_family_run_states_its_own_signature(family: str) -> None:
    """§3.1 — closed signatures, and the horizon word is the family's own.

    ``help(sys.run)`` is the library's most-read tooltip.  The bound set is
    exactly the why-table: a map has no ``final_time`` because a map has no time,
    an SDE has no ``rtol`` because a fixed step has no embedded error estimate,
    and a DDE has no ``t0`` because its clock is pinned to ``[-tau_max, 0]``.
    """
    cls = getattr(ts, family)
    sig = inspect.signature(cls.run)
    bound = tuple(
        p.name
        for p in sig.parameters.values()
        if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    )
    assert bound == RUN_KEYWORDS[family], f"{family}.run binds {bound}"
    first = bound[0]
    assert first == ("steps" if family == "DiscreteMap" else "final_time")
    assert sig.parameters[first].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.parametrize("family", sorted(RUN_KEYWORDS))
def test_a_wrong_family_horizon_word_is_refused_by_name(family: str) -> None:
    """§3.1 — an unknown keyword raises; it is never accepted and dropped."""
    system = FAMILY_SUBJECTS[family]()
    wrong = "steps" if family != "DiscreteMap" else "final_time"
    if wrong in RUN_KEYWORDS[family]:  # pragma: no cover — guards the table itself
        pytest.skip(f"{wrong} is a real {family} keyword")
    with pytest.raises((ts.InvalidParameterError, TypeError)) as excinfo:
        system.run(**{wrong: 5})
    assert wrong in str(excinfo.value)


def test_a_solver_is_spelled_solver_and_a_method_is_an_estimator() -> None:
    """C3 — ``solver=`` selects a numerical kernel; ``method=`` selects an estimator.

    One word cannot mean two things at a library whose users type both in the same
    session, so ``run`` refuses ``method=`` *by name* and points at ``solver=``.
    The estimator half is read live rather than from a hardcoded example: the
    contract's own illustration (``max_lyapunov(method="kantz")``) names a
    function that has no ``method=`` — ``lyapunov_from_data`` is the Kantz /
    Rosenstein door.
    """
    lor = ts.systems.Lorenz()
    assert "solver" in inspect.signature(lor.run).parameters
    with pytest.raises(ts.InvalidParameterError) as excinfo:
        lor.run(final_time=1.0, dt=0.5, method="rk45")
    assert "solver" in str(excinfo.value)
    estimators = _doors_taking("method")
    assert estimators, "no analysis takes method= — the estimator half of C3 vanished"
    for name in estimators:
        assert "solver" not in inspect.signature(getattr(ts.analysis, name)).parameters, (
            f"{name} takes BOTH method= and solver=: the two words must stay disjoint"
        )


#: Public callables that are variadic BY DESIGN, with the vocabulary they accept.
#: Every other public callable must show a real signature: ``help()`` answering
#: ``(*args, **kwargs)`` is a door with no label on it.
DECLARED_VARIADIC: dict[str, str] = {
    "with_params": "the parameter names are the system's own — the whole point",
    "plot": "§6.2: positional transforms, then five keyword vocabularies",
    "sel": "§4.1: the components to select, by name or index — one or many",
}


def _tab_callables(subject: Any) -> Iterator[tuple[str, Any]]:
    for name in public(subject):
        attribute = inspect.getattr_static(type(subject), name, None)
        if isinstance(attribute, property) or not callable(attribute):
            continue
        yield name, getattr(type(subject), name)


def test_no_public_method_hides_behind_varargs() -> None:
    """§3.1 — ``help()`` on a public method shows a real signature.

    The two variadic doors are declared and each carries a docstring naming what
    it takes, so the ``help()`` a user reads is still informative.
    """
    offenders: list[str] = []
    subjects = [factory() for factory in FAMILY_SUBJECTS.values()]
    subjects.append(_traj())
    for subject in subjects:
        for name, func in _tab_callables(subject):
            parameters = [
                p for p in inspect.signature(func).parameters.values() if p.name != "self"
            ]
            variadic = bool(parameters) and all(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters
            )
            if not variadic:
                continue
            if name in DECLARED_VARIADIC:
                assert func.__doc__, f"{type(subject).__name__}.{name} is variadic AND undocumented"
                continue
            offenders.append(f"{type(subject).__name__}.{name}{inspect.signature(func)}")
    assert not offenders, f"public methods whose signature says nothing: {offenders}"


def test_every_public_method_is_documented() -> None:
    """A signature without prose is half a door."""
    undocumented: list[str] = []
    for factory in FAMILY_SUBJECTS.values():
        subject = factory()
        for name, func in _tab_callables(subject):
            if not (func.__doc__ or "").strip():
                undocumented.append(f"{type(subject).__name__}.{name}")
    assert not undocumented, f"public methods with no docstring: {undocumented}"


#: Return annotations that name a type which no longer exists.  ``PlotSpec`` was
#: renamed ``Plot`` in v6 (§6.3) — the *class* moved, the annotations on the
#: plotting seam did not, so ``help(lor.plot)`` promises a type a user cannot
#: import.  Keyed by the DEFINING function, so the row does not multiply across
#: the four family subjects that inherit it.  The table may only shrink.
#:
#: **It is empty.**  Its one row named ``Trajectory.to_plot_spec``, which §2.3
#: removes outright; retiring the name took the annotation with it.
STALE_RETURN_ANNOTATIONS: dict[str, str] = {}


def test_every_return_annotation_names_a_type_that_resolves() -> None:
    """§6.3 — ``help()`` may not promise a type that was renamed away.

    Asserted as set equality so the table is self-cleaning: a new stale
    annotation fails, and a fixed one fails until its row is deleted.
    """
    stale: dict[str, str] = {}
    subjects = [factory() for factory in FAMILY_SUBJECTS.values()]
    subjects.append(_traj())
    for subject in subjects:
        for _name, func in _tab_callables(subject):
            annotation = inspect.signature(func).return_annotation
            if not isinstance(annotation, str):
                continue
            namespace = vars(importlib.import_module(func.__module__))
            try:
                eval(annotation, {**vars(typing), **namespace})  # noqa: S307 — the check itself
            except Exception as exc:  # noqa: BLE001 — that IS the failure
                key = f"{func.__module__}.{func.__qualname__}"
                stale[key] = f"{annotation} ({type(exc).__name__})"
    assert set(stale) == set(STALE_RETURN_ANNOTATIONS), (
        "stale return annotations moved.\n"
        f"  newly stale: {sorted(set(stale) - set(STALE_RETURN_ANNOTATIONS))}\n"
        f"  fixed (delete the row): {sorted(set(STALE_RETURN_ANNOTATIONS) - set(stale))}\n"
        f"  detail: {stale}"
    )


@pytest.mark.parametrize("namespace", ("analysis", "viz"))
def test_no_registry_function_hides_behind_varargs(namespace: str) -> None:
    """§3.1 — the same rule at the free-function doors.

    ``ts.analysis.<TAB>`` and ``ts.viz.<TAB>`` are the whole discovery story since
    ruling A2 took analyses off the object.  A listed name whose ``help()`` reads
    ``(*args, **kwargs)`` puts the user back where the curation found them.
    """
    module = getattr(ts, namespace)
    checked = 0
    offenders: list[str] = []
    for name in module.__all__:
        obj = getattr(module, name)
        if not callable(obj) or isinstance(obj, type | types.ModuleType):
            continue
        checked += 1
        try:
            parameters = list(inspect.signature(obj).parameters.values())
        except (TypeError, ValueError):
            offenders.append(f"{name}: no signature at all")
            continue
        if parameters and all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
            offenders.append(f"{name}{inspect.signature(obj)}")
        if not (obj.__doc__ or "").strip():
            offenders.append(f"{name}: no docstring")
    assert checked >= 5, f"ts.{namespace} exposed only {checked} callables — sweep is vacuous"
    assert not offenders, f"ts.{namespace} names whose help() says nothing: {offenders}"


def test_every_analysis_takes_its_subject_as_the_first_positional_argument() -> None:
    """§5.1 — "an analysis is a free function whose FIRST argument is the thing it is about".

    That sentence is the entire replacement for the bound methods ruling A2
    removed, so it has to be true of all 49 — a door that takes its subject by
    keyword breaks ``ts.analysis.correlation_dimension(traj)`` and with it the
    grammar every error message teaches.
    """
    offenders: list[str] = []
    analyses = [n for n in ts.analysis.__all__ if n not in ANALYSIS_VERBS]
    assert len(analyses) == 49
    for name in analyses:
        obj = getattr(ts.analysis, name)
        parameters = list(inspect.signature(obj).parameters.values())
        if not parameters:
            offenders.append(f"{name}: takes no subject at all")
            continue
        first = parameters[0]
        if first.kind not in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD):
            offenders.append(f"{name}: first parameter {first.name!r} is {first.kind.description}")
        if first.default is not inspect.Parameter.empty:
            offenders.append(f"{name}: the subject {first.name!r} is optional")
    assert not offenders, f"analyses whose subject is not the first positional: {offenders}"
