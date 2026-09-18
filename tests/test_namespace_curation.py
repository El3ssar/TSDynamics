"""The curated namespace: the top level, every public subpackage, and the redirects.

The owner's complaint that opened this stream was a literal ``ts.<TAB>``: eleven
submodules — ``engine``, ``solvers``, ``registry``, ``families``, ``utils``,
``data``, ``derived`` among them — sitting next to the things a user actually
reaches for.  "We should only see there what is supposed to be exposed to the
user, no internal use things. Same applies to submodules!!"

v6 finishes the job.  Curation used to mean *"drop the name from ``__all__`` and
leave it bound"*, which bought a tidy ``dir()`` and nothing else: ``ts.<TAB>``
said seventeen names while ``getattr`` answered to two hundred and sixty, and the
two hand-written re-export blocks that produced those bindings were already
measurably stale (``ts.LyapunovSpectrum`` resolved; ``ts.Embedding`` did not).

So this file locks four contracts:

1. **The listing is the surface.**  ``ts.__all__`` is seventeen names, sorted;
   ``dir()`` mirrors it; and — corollary **C5** — *no API name resolves unless it
   is listed*.  :func:`test_no_api_name_resolves_on_the_top_level` sweeps the
   whole demoted population, name by name.
2. **A wrong guess teaches.**  The ordered ladder in ``tsdynamics.__getattr__``
   answers five ways, and :func:`test_the_ladder_renders` pins every one of them
   on a guess a real user makes.
3. **A redirect survives the import spelling.**  An exact hit raises
   :class:`~tsdynamics.errors.MovedInV6`, an ``ImportError``, because CPython
   throws away a module ``__getattr__``'s message for ``from tsdynamics import
   X`` whenever the exception matches ``AttributeError`` — and that is the
   spelling the corpus uses 111 times.  :func:`test_a_redirect_survives_the_import_spelling`
   is the measurement, run as a gate.
4. **A generic anti-rot sweep** over *every* public package — discovered live, so
   a new subpackage joins with zero test edits.

Reachability is never removed: every demoted name still lives at exactly one
address, and every redirect prints it.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import types

import pytest

import tsdynamics as ts
from tsdynamics import _redirects, analysis
from tsdynamics.errors import MovedInV6

# ── the curated top-level surface ────────────────────────────────────────────────

#: The exact curated ``tsdynamics.__all__`` — **seventeen** names.
#:
#: Five family bases you subclass · one received type you annotate · one plotting
#: verb · three registries · six names you type inside ``except`` · the version.
_CURATED_TOP_LEVEL = {
    # the family bases: what you subclass
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "StochasticSystem",
    "WrappedSystem",
    # the one received type that is also a typed one
    "Trajectory",
    # the plotting front door (lazily resolved, like ``viz``)
    "plot",
    # the three registries worth typing a dot after
    "systems",
    "analysis",
    "viz",
    "__version__",
}


def test_top_level_all_is_curated():
    """``ts.__all__`` is exactly the curated headline set — no flat dump."""
    assert set(ts.__all__) == _CURATED_TOP_LEVEL
    assert len(ts.__all__) == 11
    # ``__dir__`` mirrors ``__all__`` (curated autocomplete surface).
    assert dir(ts) == sorted(ts.__all__)


def test_top_level_all_is_sorted_one_name_per_line():
    """The listing is data, sorted — the one format two builders can both append to.

    A list grouped by comment is the maximally merge-hostile format: two people
    adding a name both append inside the same group and collide on the same line.
    Sortedness is the mechanical rule; the *reasons* live in the module docstring,
    where they read as prose.
    """
    assert ts.__all__ == sorted(ts.__all__), "tsdynamics.__all__ is not sorted"


def test_the_typed_errors_live_at_their_own_address_because_you_never_need_them():
    """The exceptions are NOT on the top level, and the hierarchy is why.

    They were promoted in v6.0 on the argument that catching is ordinary work.
    The measurement says otherwise: every one subclasses the builtin a caller
    would already reach for, so ``except ValueError`` catches a bad ``dt``
    today and NOTHING a user writes requires this library's spelling.  The
    names are a *refinement* — for telling one failure from another — and a
    refinement lives one dot down.

    Both halves are asserted, because only together do they make the demotion
    safe: the additive hierarchy (you can work without the names) and the
    redirect (a guess teaches the address).
    """
    # It is additive, so plain Python catches everything this library raises.
    assert issubclass(ts.errors.InvalidParameterError, ValueError)
    assert issubclass(ts.errors.InvalidInputError, TypeError)
    assert issubclass(ts.errors.ConvergenceError, RuntimeError)
    assert issubclass(ts.errors.BackendError, RuntimeError)
    assert issubclass(ts.errors.StepBudgetError, ts.errors.ConvergenceError)

    for name in (
        "TSDynamicsError",
        "ConvergenceError",
        "StepBudgetError",
        "BackendError",
        "InvalidParameterError",
        "InvalidInputError",
    ):
        assert name not in ts.__all__
        assert hasattr(ts.errors, name), f"{name} must stay reachable at its address"
        # ``MovedInV6`` is an ImportError on purpose: CPython discards a module
        # ``__getattr__``'s message on the ``from tsdynamics import X`` path
        # unless it inherits ImportError, and the corpus uses that spelling.
        with pytest.raises(ImportError, match=rf"ts\.errors\.{name}"):
            getattr(ts, name)


def test_errors_the_module_left_the_listing_but_is_bound_forever():
    """``ts.errors`` is an **ABI**, not a curation choice.

    The Rust bridge builds its typed exceptions by importing the module path by
    name — ``py.import("tsdynamics.errors")`` in ``crates/tsdyn-core/src/lib.rs``
    — so every engine surface (integrate / DDE / SDE / map / events / stepper /
    basin / Lyapunov) depends on it resolving.  It left ``__all__`` only because
    its six classes are on the top level now; it can never leave the package.
    """
    assert "errors" not in ts.__all__
    assert "errors" not in dir(ts)
    assert "errors" in ts._INTERNAL_SUBMODULES
    assert isinstance(ts.errors, types.ModuleType)
    assert importlib.import_module("tsdynamics.errors") is ts.errors


def test_the_rust_bridge_still_names_the_module_it_imports():
    """Guard the guard above: if the bridge stops importing it, say so out loud.

    This is the *reason* ``errors`` is exempt from the demotion rule.  A reader
    who deletes the binding because "nothing in src/ imports it" needs to find
    the Rust call site from here, not from a code comment that has drifted.
    """
    import pathlib

    source = pathlib.Path("crates/tsdyn-core/src/lib.rs").read_text(encoding="utf-8")
    assert '"tsdynamics.errors"' in source, (
        "the Rust bridge no longer imports tsdynamics.errors by name; "
        "if that is deliberate, the ABI exemption above can go"
    )


# ── C5, clause 4: no API name resolves unless it is listed ───────────────────────


def _demoted_population() -> list[str]:
    """Every API name that used to answer to ``ts.<name>`` and must not any more.

    Built from the *live* public homes plus the redirect tables, so it cannot go
    stale: adding a system or an analysis grows this population automatically.
    """
    names: set[str] = set(_redirects.REMOVED_IN_V6) | set(_redirects.RENAMED_IN_V6)
    for home in ts._PUBLIC_HOMES:
        names |= set(ts._public_names(home))
    return sorted(names - set(ts.__all__))


def test_the_demoted_population_is_the_size_the_contract_measured():
    """Guard the guard: an empty population would make the C5 sweep vacuous."""
    population = _demoted_population()
    assert len(population) > 200, f"only {len(population)} demoted names found"
    # the headline members, one per kind
    assert {"Lorenz", "correlation_dimension", "PoincareMap", "Box"} <= set(population)


@pytest.mark.parametrize("name", _demoted_population())
def test_no_api_name_resolves_on_the_top_level(name):
    """C5 clause 4 — ``dir()`` is the truth, and ``getattr`` agrees with it.

    Before v6 this was false for 253 names: two hand-written re-export blocks in
    ``__init__.py`` bound every analysis function, result class, derived wrapper
    and region primitive, and a lazy ``__getattr__`` bound all 177 systems.  The
    listing said seventeen; the namespace answered to two hundred and seventy.

    Every one of them still exists.  It just lives at one address now, and the
    exception says which.
    """
    with pytest.raises((AttributeError, MovedInV6)):
        getattr(ts, name)


def test_every_demoted_name_still_resolves_at_exactly_one_address():
    """Demotion is never removal — the capability is one dot away, and it is the same object."""
    cases = {
        "Lorenz": ts.systems,
        "correlation_dimension": ts.analysis,
        "orbit_diagram": ts.analysis,
        "PoincareMap": ts.derived,
        "TangentSystem": ts.derived,
        "Box": ts.data,
        "Ball": ts.data,
        "Grid": ts.data,
        "sampler": ts.data,
        "grid_points": ts.data,
        "set_distance": ts.data,
        "SystemBase": ts.families,
        "ParamSet": ts.families,
    }
    for name, home in cases.items():
        assert hasattr(home, name), f"{home.__name__}.{name} vanished"
        assert name in home.__all__, f"{home.__name__}.{name} resolves but is not listed"


# ── the redirect mechanism (C6) ──────────────────────────────────────────────────


def test_a_redirect_survives_the_import_spelling():
    """An exact hit is an ``ImportError``, so ``from tsdynamics import X`` keeps the text.

    This is the measurement the design called impossible, run as a gate.  On
    CPython a module ``__getattr__`` that raises anything matching
    ``AttributeError`` has its message **discarded** by the from-import machinery
    and replaced with the generic "cannot import name"; raise ``ImportError`` and
    the text propagates verbatim.  Since the corpus reaches for the from-import
    spelling 111 times, that is the spelling the migration guide has to survive.
    """
    namespace: dict[str, object] = {}
    with pytest.raises(ImportError) as excinfo:
        exec("from tsdynamics import correlation_dimension", namespace)  # noqa: S102
    message = str(excinfo.value)
    assert "ts.analysis.correlation_dimension" in message, message
    assert "cannot import name" not in message, "the teaching text was discarded"


def test_a_guess_stays_an_attribute_error_so_hasattr_keeps_working():
    """Cases 4-5 must NOT be ``ImportError``, or ``hasattr`` breaks for every name.

    ``hasattr`` only swallows ``AttributeError``.  Making every miss an
    ``ImportError`` would turn ``hasattr(ts, "anything")`` into a raise, which is
    the kind of blast radius a teaching message is not worth.  The cost is
    confined to the enumerated dead names — and there it is the *point*: a v5
    script probing ``hasattr(ts, "permutation_entropy")`` must not silently take
    the "not installed" branch when the honest answer is "that moved out".
    """
    assert hasattr(ts, "definitely_not_a_name_xyz") is False
    assert getattr(ts, "definitely_not_a_name_xyz", "sentinel") == "sentinel"
    with pytest.raises(MovedInV6):
        hasattr(ts, "permutation_entropy")


def test_moved_in_v6_cannot_be_both_and_says_so():
    """The two-base class the design wanted is impossible; this is why one was chosen."""
    assert issubclass(MovedInV6, ImportError)
    assert not issubclass(MovedInV6, AttributeError)
    assert issubclass(MovedInV6, ts.errors.TSDynamicsError)
    with pytest.raises(TypeError, match="lay-out conflict"):
        type("Both", (AttributeError, ImportError), {})


def test_star_import_and_introspection_are_untouched():
    """The redirect must not leak into ``import *``, ``dir()`` or ``getmembers``."""
    import inspect

    namespace: dict[str, object] = {}
    exec("from tsdynamics import *", namespace)  # noqa: S102
    assert set(namespace) - {"__builtins__"} == set(ts.__all__)
    assert [name for name, _ in inspect.getmembers(ts)]  # does not raise


def test_private_and_dunder_probes_stay_cheap_and_quiet():
    """A protocol probe is not a user typing a name: no import, no teaching, no cost.

    ``from tsdynamics import _rust`` asks ``hasattr`` *first* and only falls back
    to importing the submodule if that answered ``False`` — so a ``MovedInV6``
    here would break the engine import outright.  And a notebook's
    ``_ipython_canary_method_should_not_exist_`` probe must not pull in the
    plotting layer to be told "no".
    """
    code = (
        "import sys, tsdynamics as ts\n"
        "for probe in ('__wrapped__', '__bases__', '_ipython_canary_x', '_redirects_x'):\n"
        "    assert not hasattr(ts, probe), probe\n"
        "assert 'tsdynamics.viz' not in sys.modules, 'a dunder probe imported viz'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ── the ladder: every case, rendered on a guess a real user makes ────────────────

#: ``(what the user typed, the exception kind, the substrings the answer must carry)``.
#:
#: One row per case of the ladder, plus the guesses that used to be answered with
#: nonsense.  Each expectation is a *line the user can type*, never a description
#: of the mistake.
_LADDER = [
    # case 3 — an exact hit in a public home: the commonest v5 spelling of all
    ("Lorenz", MovedInV6, ["ts.systems.Lorenz()", "moved: the top level"]),
    ("correlation_dimension", MovedInV6, ["ts.analysis.correlation_dimension"]),
    ("orbit_diagram", MovedInV6, ["ts.analysis.orbit_diagram"]),
    ("PoincareMap", MovedInV6, ["ts.derived.PoincareMap"]),
    ("Box", MovedInV6, ["ts.data.Box"]),
    ("sampler", MovedInV6, ["ts.data.sampler"]),
    # ...including the two that used to fall through to a fuzzy match and answer
    # a real question with an unrelated object (``ts.data.region`` -> Oregonator).
    ("region", MovedInV6, ["ts.data.region"]),
    ("Region", MovedInV6, ["ts.data.Region"]),
    # case 1 — renamed: the capability is here, under one spelling
    (
        "bifurcation_diagram",
        MovedInV6,
        ["ts.analysis.orbit_diagram(system,", "was renamed"],
    ),
    ("basins_of_attraction", MovedInV6, ["ts.analysis.basins(system, region)"]),
    ("find_attractors", MovedInV6, ["ts.analysis.attractors(system, region)"]),
    # case 2 — removed by the scope surgery: explain, then name what stayed
    (
        "permutation_entropy",
        MovedInV6,
        ["was removed", "phase-space", "ts.analysis.rqa(traj)", 'ts.analysis.find("recurrence")'],
    ),
    ("surrogate_test", MovedInV6, ["was removed", "ts.analysis.embed(data, dimension, delay)"]),
    # case 4 — a near miss, ranked
    ("Lorentz", AttributeError, ["Did you mean:", "ts.systems.Lorenz()"]),
    ("correlation_dimensio", AttributeError, ["ts.analysis.correlation_dimension"]),
    # case 5 — nothing matches: name the listings, and the search that needs no name
    ("integrate", AttributeError, ["ts.systems.<TAB>", "ts.analysis.find('integrate')"]),
    ("zzzz_not_a_name", AttributeError, ["Tab-complete a registry, or search:"]),
    # ...and the rename that landed with the plotting layer
    ("PlotSpec", MovedInV6, ["ts.viz.Plot", "was renamed"]),
]


@pytest.mark.parametrize(("typed", "kind", "expected"), _LADDER, ids=[c[0] for c in _LADDER])
def test_the_ladder_renders(typed, kind, expected):
    """Every ladder case answers with the line to type, in the right exception kind."""
    with pytest.raises(kind) as excinfo:
        getattr(ts, typed)
    message = str(excinfo.value)
    assert typed in message, f"the answer does not name what was typed:\n{message}"
    for fragment in expected:
        assert fragment in message, f"missing {fragment!r} in:\n{message}"


@pytest.mark.parametrize(("typed", "kind", "expected"), _LADDER, ids=[c[0] for c in _LADDER])
def test_every_ladder_answer_fits_the_terminal(typed, kind, expected):
    """≤ 88 columns **including the class-name prefix the traceback prepends**.

    Counting the prefix is the whole point: without it a message is a tidy 88 in
    the source and wraps in the terminal at the one place it must not — the first
    line, the one carrying the name that was typed.  ``MovedInV6``'s prefix is 29
    characters (it is not a builtin, so Python prints the qualified path), which
    is 13 more than ``AttributeError``'s.
    """
    with pytest.raises(kind) as excinfo:
        getattr(ts, typed)
    cls = type(excinfo.value)
    prefix = f"{cls.__module__}.{cls.__qualname__}: " if cls.__module__ != "builtins" else ""
    lines = str(excinfo.value).splitlines()
    lines[0] = prefix + lines[0]
    over = [(len(line), line) for line in lines if len(line) > 88]
    assert not over, f"{typed}: lines over 88 columns: {over}"


@pytest.mark.parametrize(("typed", "kind", "expected"), _LADDER, ids=[c[0] for c in _LADDER])
def test_every_ladder_answer_carries_a_runnable_line(typed, kind, expected):
    """The message standard: at least one indented line that parses as Python.

    Naming the mistake is necessary and not sufficient.  This is the same rule
    ``tests/test_polish_standards.py`` applies to the library's raise sites,
    applied to the namespace's own answers — which is where it matters most,
    because for a demoted name this exception is the *only* documentation the
    user will see.
    """
    import ast

    with pytest.raises(kind) as excinfo:
        getattr(ts, typed)
    runnable = []
    for line in str(excinfo.value).splitlines():
        if not line.startswith("    "):
            continue
        candidate = line.strip().split("#")[0].strip().replace("<TAB>", "")
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        runnable.append(candidate)
    assert runnable, f"{typed} answered with no runnable line:\n{excinfo.value}"


# ── the suggestion scorer ────────────────────────────────────────────────────────


def test_the_scorer_ranks_verbs_above_types_at_equal_score():
    """A tie goes to the function you call, not the type you only receive.

    ``difflib`` breaks ties by *string order*, which encodes nothing — so whether
    a user guessing ``fixedpoints`` was offered ``fixed_points`` (the function) or
    ``FixedPoint`` (a result class nobody constructs: a grep for a construction of
    any of the 32 across ``docs/`` returns zero) depended on where the ASCII
    codepoints fell.  The tie-break is stated now.
    """
    assert ts._rank("fixed_points") < ts._rank("FixedPoint")
    assert ts._rank("basins") < ts._rank("BasinsResult")
    # ...and it really is consulted: an exact tie resolves verb-first.
    import difflib

    query = "xxfixedpointsxx"
    pairs = [("fixed_points", "FixedPoint")]
    for verb, cls in pairs:
        assert difflib.SequenceMatcher(None, query, verb).ratio() != pytest.approx(
            difflib.SequenceMatcher(None, query, cls).ratio()
        ) or ts._rank(verb) < ts._rank(cls)


def test_the_scorer_offers_lines_you_can_run():
    """A suggestion is a line, not a name: a system is shown *constructed*."""
    assert "ts.systems.Lorenz()" in ts._suggest("Lorentz")
    assert all(line.startswith("ts.") for line in ts._suggest("Lorentz"))


def test_the_scorer_needs_both_similarity_and_coverage():
    """``ts.Event`` must not be answered with ``ts.systems.Tent()``.

    One number cannot do this job, which is why there are two.  A plain 0.6
    similarity floor offers ``Tent`` for ``Event`` (ratio 0.667 — three shared
    letters sold as an answer); raising the floor to reject that also rejects
    ``lyapunov`` → ``lyapunov_spectrum`` (ratio 0.640), the most useful
    suggestion in the library.  *Coverage* — how much of what the user typed the
    candidate accounts for — separates them cleanly: 0.600 against 1.000.
    """
    assert "ts.systems.Tent()" not in ts._suggest("Event")
    assert ts._suggest("Event") == []
    # ...while the abbreviation the floor alone would have killed still works.
    assert "ts.analysis.lyapunov_spectrum" in ts._suggest("lyapunov")
    assert "ts.systems.Lorenz()" in ts._suggest("Lorentz")


# ── the generic anti-rot gate ────────────────────────────────────────────────────

#: The one dunder a public listing may carry: the package version.
_ALLOWED_DUNDERS = frozenset({"__version__"})

#: Names that must never appear in any public listing, whatever the package.
#: ``discover_plugins`` is entry-point machinery; the rest are typing / ``__future__``
#: artefacts that leak when a module has no ``__dir__``.
_NEVER_PUBLIC = frozenset({"annotations", "Any", "TYPE_CHECKING", "discover_plugins"})

#: Renderer *backend* packages, excluded from the sweep because their
#: ``__init__.py`` files are outside this stream's ownership.  They are three
#: levels deep (``ts.viz.render.mpl``) and ``render`` itself is not on
#: ``ts.viz``'s listing, so no user tab-completes into them — but the exclusion
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


#: Packages that failed to import during discovery, name -> exception.  Filled by
#: :func:`_public_packages` and asserted empty by a test of its own, so one broken
#: package fails **one named test** instead of erroring this whole module at
#: collection time and taking the other ninety gates with it.
_IMPORT_FAILURES: dict[str, str] = {}


def _public_packages() -> list[str]:
    """Every public package in the tree, discovered live (so new ones join the gate)."""

    def _note(name: str) -> None:
        _IMPORT_FAILURES[name] = repr(sys.exc_info()[1])

    found = [
        m.name
        for m in pkgutil.walk_packages(ts.__path__, "tsdynamics.", onerror=_note)
        if m.ispkg and "._" not in m.name
    ]
    return ["tsdynamics", *sorted(n for n in found if n not in _IMPORT_FAILURES)]


def test_every_public_package_imports():
    """Discovery must not be silently skipping a package it could not import."""
    _public_packages()
    assert not _IMPORT_FAILURES, f"public packages that fail to import: {_IMPORT_FAILURES}"


def _advertised_public_modules() -> list[str]:
    """Every *plain module* a public package advertises in its own ``dir()``.

    The package sweep filters on ``ispkg``, which silently exempts a public
    **module** sitting on a curated listing.  ``tsdynamics.analysis.planar`` was
    exactly that: a user tab-completes into it, and the module — having no
    ``__dir__`` of its own — handed them ``np`` / ``warnings`` / ``dataclass``
    plus its private grid helpers.  A namespace you can *reach* by tab completion
    is a namespace this gate must cover, package or not.
    """
    out: set[str] = set()
    for pkg_name in _public_packages():
        if pkg_name in _UNCURATED_RENDERER_BACKENDS:
            continue
        pkg = importlib.import_module(pkg_name)
        for attr in dir(pkg):
            try:  # a lazy attribute may import a sibling package that is broken
                obj = getattr(pkg, attr, None)
            except Exception:  # noqa: BLE001 - reported by test_every_public_package_imports
                _IMPORT_FAILURES[f"{pkg_name}.{attr}"] = repr(sys.exc_info()[1])
                continue
            if isinstance(obj, types.ModuleType) and not hasattr(obj, "__path__"):
                out.add(obj.__name__)
    return sorted(out)


def test_the_package_sweep_actually_finds_packages():
    """Guard the guard: an empty/tiny discovery would make the gate below vacuous."""
    pkgs = _public_packages()
    assert len(pkgs) >= 20, pkgs
    assert {"tsdynamics", "tsdynamics.analysis", "tsdynamics.viz", "tsdynamics.systems"} <= set(
        pkgs
    )


@pytest.mark.parametrize("mod_name", _advertised_public_modules())
def test_advertised_public_module_listing_is_clean(mod_name):
    """A public *module* on a curated listing shows its API, not its imports.

    Same contract as the package sweep, minus the dunder rule: a plain module
    always carries ``__name__`` / ``__file__`` / ``__builtins__``, and every
    autocompleter hides leading-underscore names until you type one.
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

    Three rules, one per way a listing rots: no private names, no foreign modules
    (a bare ``import warnings`` becomes ``ts.viz.render.warnings`` in
    autocomplete), nothing on the internals list.
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

    assert len(listing) == len(set(listing)), f"{pkg_name} lists a name twice"
    for n in listing:
        assert hasattr(mod, n), f"{pkg_name}.{n} is advertised but does not resolve"


@pytest.mark.parametrize("pkg_name", _public_packages())
def test_public_package_declares_all(pkg_name):
    """Every public package declares an ``__all__`` — curation is explicit, not accidental."""
    if pkg_name in _UNCURATED_RENDERER_BACKENDS:
        pytest.skip("renderer backend __init__ is outside this stream's file ownership")
    mod = importlib.import_module(pkg_name)
    assert isinstance(getattr(mod, "__all__", None), list), f"{pkg_name} has no __all__"


@pytest.mark.parametrize("pkg_name", _public_packages())
def test_public_package_dir_mirrors_all(pkg_name):
    """One mechanical policy, everywhere: ``dir(M) == sorted(M.__all__)``.

    Not "``dir()`` is a subset" and not "``__all__`` is a subset": *equal*.  A
    package whose ``dir()`` is bigger is leaking; one whose ``dir()`` is smaller
    is advertising a name that does not resolve.  Both have shipped.
    """
    if pkg_name in _UNCURATED_RENDERER_BACKENDS:
        pytest.skip("renderer backend __init__ is outside this stream's file ownership")
    mod = importlib.import_module(pkg_name)
    assert dir(mod) == sorted(mod.__all__), f"{pkg_name}: dir() and __all__ disagree"


#: ``__all__`` entries that legitimately resolve to a **module**, per package.
#: An entry not on this table is expected to be a class or a function; if it
#: resolves to a module, some submodule is shadowing it.
#:
#: ``tsdynamics.viz: ("spec",)`` and ``tsdynamics.analysis: ("results",)`` are the
#: one place C4 and the curated listings genuinely trade off, and the listings
#: win: a user who types ``ts.viz.spec`` means the module, and a user who types
#: ``ts.analysis.results`` means the module.  Neither name is also a verb, so
#: there is nothing for either to shadow.
_DECLARED_SUBMODULE_EXPORTS = {
    "tsdynamics": {"analysis", "systems", "viz"},
    "tsdynamics.analysis": set(getattr(analysis, "_CATEGORY_SUBPACKAGES", ())) | {"results"},
    "tsdynamics.viz": {"transforms", "spec"},
    "tsdynamics._engine": {"compile", "problem", "run", "symbols"},
    "tsdynamics.systems": {"continuous", "discrete"},
}


@pytest.mark.parametrize("pkg_name", _public_packages())
def test_no_all_entry_is_shadowed(pkg_name):
    """Corollary C4: a name must resolve to what its namespace advertises.

    This one gate catches all three instances of the defect the library has
    shipped: ``ts.analysis.basins`` (a function shadowing the basins subpackage),
    ``ts.viz.transforms`` (a listing function over the transforms subpackage, so
    ``ts.viz.transforms()`` answered ``TypeError: 'module' object is not
    callable``), and the v4 ``entropy`` collision.
    """
    if pkg_name in _UNCURATED_RENDERER_BACKENDS:
        pytest.skip("renderer backend __init__ is outside this stream's file ownership")
    mod = importlib.import_module(pkg_name)
    declared = _DECLARED_SUBMODULE_EXPORTS.get(pkg_name, set())
    shadowed = sorted(
        name
        for name in getattr(mod, "__all__", [])
        if name not in declared and isinstance(getattr(mod, name, None), types.ModuleType)
    )
    assert not shadowed, (
        f"{pkg_name}.__all__ advertises {shadowed} as API, but each resolves to a MODULE. "
        "Either declare it a submodule export in _DECLARED_SUBMODULE_EXPORTS, or rename "
        "whichever of the two the user is less likely to have meant."
    )


# ── §11.1's enforcement clause: every public MODULE curates its own listing ──────
#
# The three sweeps above cover packages, and modules a package *advertises*.
# Neither reaches a public module a curated listing does not mention — and that
# is most of them: ``dir(tsdynamics._engine)`` is four names, so
# ``tsdynamics._engine.events`` was invisible to the gate while offering ``math``,
# ``np``, ``dataclass``, ``field``, ``Any``, ``Problem`` and two tolerance
# constants next to its four real ones.  ``import tsdynamics._engine.events`` is a
# line a user can type, so its listing is a listing.
#
# The mechanism is the contract's, stated once: **a module that declares
# ``__all__`` MUST define ``__dir__`` returning ``sorted(__all__)``.**  ``__all__``
# alone governs only ``from X import *``; it has zero effect on ``dir()``, which
# is what tab completion reads.  Measured before this gate: 44 public modules
# leaked 523 names between them, every one of them by that single omission.
#
# Note what is *not* asserted: that a listed name is defined locally.  A module
# may legitimately re-export (``engine.run`` is documented as re-exporting every
# name its five split-out submodules own).  Foreign names are excluded by
# construction instead — ``np`` can only reach ``dir()`` if someone writes
# ``"np"`` into ``__all__``, which is a decision, not a leak.


def _public_modules() -> list[str]:
    """Every module in the tree whose dotted path is public end to end.

    Packages included: a package is a module.  ``_``-prefixed components are
    skipped at any depth, so ``viz.render.mpl._anim`` is out while
    ``viz.render.mpl`` is in.
    """
    found = [
        m.name
        for m in pkgutil.walk_packages(ts.__path__, "tsdynamics.", onerror=_note_import_failure)
        if not any(part.startswith("_") for part in m.name.split("."))
    ]
    return ["tsdynamics", *sorted(n for n in found if n not in _IMPORT_FAILURES)]


def _note_import_failure(name: str) -> None:
    _IMPORT_FAILURES[name] = repr(sys.exc_info()[1])


def _listing_defect(mod: types.ModuleType) -> str | None:
    """Why *mod* does not curate its own listing, or ``None`` when it does."""
    listed = vars(mod).get("__all__")
    if not isinstance(listed, list):
        return "no __all__" if listed is None else f"__all__ is a {type(listed).__name__}"
    if not all(isinstance(n, str) for n in listed):
        return "__all__ holds non-strings"
    if "__dir__" not in vars(mod):
        return "no __dir__"
    if dir(mod) != sorted(listed):
        return "dir() != sorted(__all__)"
    return None


#: The modules that do **not** curate their own listing yet, each with the slot
#: that owns the file.  Two gates read it, and the split is deliberate: the fast
#: tier refuses a **new** offender (:func:`test_the_uncurated_module_backlog_admits_no_new_offender`)
#: and the ``full`` tier refuses a **stale row** (:func:`test_the_uncurated_module_backlog_is_self_cleaning`),
#: exactly like the doctest exemptions.  So the list can only shrink, without a
#: sibling slot landing its half turning the fast tier red for everyone else.
#:
#: The fix is always the same five lines — ``__all__`` after the imports if the
#: module has none, then::
#:
#:     def __dir__() -> list[str]:
#:         """Expose only the curated public API (``__all__``) to ``dir()``."""
#:         return sorted(__all__)
#:
#: The catalogue modules are the highest-harm rows: ``chaotic_attractors``
#: currently offers 58 names for 51 systems, and the seven extras are symengine's
#: ``sin`` / ``cos`` / ``exp`` / ``sign`` / ``np`` / ``ClassVar``, which *look*
#: like library helpers.
_UNCURATED_MODULE_LISTINGS: dict[str, str] = {
    # ── the 16 catalogue modules, owned by the SYSTEMS slot ──────────────────
    # These are the highest-harm rows in the tree and the reason the row count
    # is worth keeping visible: a catalogue module's public names are supposed
    # to be *system classes*, and every one of them also offers the symengine
    # functions its kernels are written with.
    "tsdynamics.systems.continuous.chaotic_attractors": "SYSTEMS · 58 names for 51 systems",
    "tsdynamics.systems.continuous.chem_bio_systems": "SYSTEMS · 27 names for 21 systems",
    "tsdynamics.systems.continuous.climate_geophysics": "SYSTEMS · 20 names for 14 systems",
    "tsdynamics.systems.continuous.coupled_systems": "SYSTEMS · 24 names for 19 systems",
    "tsdynamics.systems.continuous.delayed_systems": "SYSTEMS · 9 names for 7 systems",
    "tsdynamics.systems.continuous.exotic_systems": "SYSTEMS · 23 names for 18 systems",
    "tsdynamics.systems.continuous.oscillatory_systems": "SYSTEMS · 14 names for 10 systems",
    "tsdynamics.systems.continuous.physical_systems": "SYSTEMS · 14 names for 9 systems",
    "tsdynamics.systems.continuous.population_dynamics": "SYSTEMS · 7 names for 6 systems",
    "tsdynamics.systems.continuous.spatial_fields": "SYSTEMS · has __all__, no __dir__",
    "tsdynamics.systems.continuous.stochastic_systems": "SYSTEMS · 4 names for 3 systems",
    "tsdynamics.systems.discrete.chaotic_maps": "SYSTEMS · 11 names for 10 maps",
    "tsdynamics.systems.discrete.exotic_maps": "SYSTEMS · 9 names for 8 maps",
    "tsdynamics.systems.discrete.geometric_maps": "SYSTEMS · 6 names for 5 maps",
    "tsdynamics.systems.discrete.polynomial_maps": "SYSTEMS · 5 names for 4 maps",
    "tsdynamics.systems.discrete.population_maps": "SYSTEMS · 5 names for 4 maps",
}


def test_the_module_sweep_actually_finds_modules():
    """Guard the guard: a discovery that collapsed would certify an empty set."""
    mods = _public_modules()
    # Was >= 100.  Privatising engine/solvers/utils moved 19 modules out of the
    # public sweep, which is the point of this round; the floor tracks it rather
    # than being nudged each time.
    assert len(mods) >= 90, mods
    assert not _IMPORT_FAILURES, f"public modules that fail to import: {_IMPORT_FAILURES}"
    # Sentinels from three different subpackages, all still PUBLIC.  The old
    # trio named engine/solvers/utils modules, which this round made private —
    # so they would now assert the opposite of what the sweep is for.
    assert {
        "tsdynamics.viz.compose",
        "tsdynamics.analysis.lyapunov",
        "tsdynamics.data.trajectory",
    } <= set(mods)


@pytest.mark.parametrize(
    "mod_name", [m for m in _public_modules() if m not in _UNCURATED_MODULE_LISTINGS]
)
def test_every_public_module_curates_its_own_listing(mod_name):
    """``__all__`` **and** ``__dir__``, mirroring, on every public module."""
    mod = importlib.import_module(mod_name)
    defect = _listing_defect(mod)
    assert defect is None, (
        f"{mod_name}: {defect}.  A module that declares __all__ must define\n"
        "    def __dir__() -> list[str]:\n"
        '        """Expose only the curated public API (``__all__``) to ``dir()``."""\n'
        "        return sorted(__all__)\n"
        "or every import it makes is offered to the next person who tab-completes it."
    )
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"{mod_name}.__all__ advertises names that do not resolve: {missing}"
    internals = sorted(set(mod.__all__) & _NEVER_PUBLIC)
    assert not internals, f"{mod_name} lists internals: {internals}"


def _measured_uncurated() -> dict[str, str]:
    """``module -> defect`` for every public module that does not curate itself."""
    return {
        name: defect
        for name in _public_modules()
        if (defect := _listing_defect(importlib.import_module(name))) is not None
    }


def test_the_uncurated_module_backlog_admits_no_new_offender():
    """The anti-rot half: a module may not ship without a curated listing.

    This is the direction that has to hold on every commit, and it is safe under
    a concurrent tree — a sibling slot fixing one of the rows below only makes
    the measured set *smaller*.
    """
    measured = _measured_uncurated()
    new = sorted(f"{n} ({d})" for n, d in measured.items() if n not in _UNCURATED_MODULE_LISTINGS)
    assert not new, (
        f"public modules leaking their imports into dir(): {new}.  Add __all__ and the "
        "three-line __dir__; do not add a row to _UNCURATED_MODULE_LISTINGS, which may "
        "only shrink."
    )
    assert _UNCURATED_MODULE_LISTINGS, "the backlog went empty — delete it and this test"


@pytest.mark.full
def test_the_uncurated_module_backlog_is_self_cleaning():
    """The shrink half: a row that starts passing must be deleted.

    In the ``full`` tier for the reason the doctest exemptions are: the
    "fixed one, delete the row" failure is a *bookkeeping* failure, and firing it
    in the fast tier makes one slot's correct change break every other slot's
    green run.  Nightly is soon enough for bookkeeping; never is not.
    """
    fixed = sorted(set(_UNCURATED_MODULE_LISTINGS) - set(_measured_uncurated()))
    assert not fixed, (
        "these modules now curate their listing — delete their rows from "
        f"_UNCURATED_MODULE_LISTINGS: {fixed}"
    )


#: The listings this ruling created, transcribed.  ``engine.run`` re-exports
#: every one of these names, so the split-out seams stay reachable exactly as
#: before (:func:`test_the_split_out_engine_seams_are_still_reachable_through_run`);
#: what changed is that ``dir()`` on the seam itself no longer offers ``math``,
#: ``np``, ``dataclass`` and the two tolerance constants as if they were its API.
#:
#: The three empty ones are the honest answer, not an oversight: those modules'
#: entry points are all ``_``-prefixed (``engine.reference``, ``engine.run_methods``)
#: or they exist purely for the import side effect that registers their kernels
#: (the three ``solvers`` spec modules).  Declaring ``__all__ = []`` says so;
#: omitting it offered ``SolverSpec`` / ``SolverCaps`` / ``register`` from three
#: more addresses than the one that owns them.
_CURATED_MODULE_LISTINGS: dict[str, list[str]] = {
    "tsdynamics._engine.events": ["Event", "EventSolution", "crossings", "integrate_events"],
    "tsdynamics._engine.reference": [],
    "tsdynamics._engine.run_methods": [],
    "tsdynamics._engine.sde_run": ["sde_ensemble_final", "sde_integrate_dense"],
    "tsdynamics._engine.stepper": ["make_ode_stepper", "step_advance", "step_advance_to_event"],
    "tsdynamics._solvers.explicit": [],
    "tsdynamics._solvers.implicit": [],
    "tsdynamics._solvers.stochastic": [],
    "tsdynamics._utils.escape": [
        "ESCAPE_GROWTH",
        "ESCAPE_SCALE",
        "Unbounded",
        "detect_unbounded",
        "escaped",
    ],
    "tsdynamics._utils.plot_namespace": ["PlotNamespace", "plot_namespace", "plot_seam_error"],
    "tsdynamics._utils.tolerances": [
        "BASIN_ATOL",
        "BASIN_RTOL",
        "DDE_ATOL",
        "DDE_LYAPUNOV_ATOL",
        "DDE_LYAPUNOV_RTOL",
        "DDE_RTOL",
        "DEFAULT_ATOL",
        "DEFAULT_RTOL",
    ],
}


@pytest.mark.parametrize("mod_name", sorted(_CURATED_MODULE_LISTINGS))
def test_the_curated_module_listing_is_exactly_the_contract(mod_name):
    """The eleven listings this ruling wrote, pinned name for name.

    The generic sweep above proves a module *has* a curated listing; this proves
    it is still the one that was reviewed.  A listing that silently regrows — the
    failure this whole stream exists to make impossible — fails here by name.
    """
    mod = importlib.import_module(mod_name)
    assert dir(mod) == _CURATED_MODULE_LISTINGS[mod_name]


def test_the_split_out_engine_seams_are_still_reachable_through_run():
    """Hiding is a DISCOVERY change, never a REACHABILITY one (§11.1).

    ``engine/run.py`` re-exports the five split-out seams' names — including the
    ``_``-prefixed ones — so ``tsdynamics._engine.run.<X>`` keeps working for every
    ``X``.  Curating the seams' own ``dir()`` must not touch that, and the private
    names are the half a ``__all__`` edit could plausibly have broken.
    """
    from tsdynamics._engine import run

    for name in (
        # public, now listed on the seam that owns them
        "Event",
        "EventSolution",
        "crossings",
        "integrate_events",
        "sde_integrate_dense",
        "sde_ensemble_final",
        "make_ode_stepper",
        "step_advance",
        "step_advance_to_event",
        # private, listed nowhere and reachable all the same
        "_reference_ode",
        "_reference_map",
        "_reference_ensemble",
        "_scipy_method",
        "_resolve_method_for",
        "_recommend_method",
        "_resolve_method_and_prepare",
        "_engine_events",
        "_normalize_event_direction",
    ):
        assert hasattr(run, name), f"engine.run lost its re-export of {name}"


# ── the nine listings that were shape-checked but never content-pinned ──────────

#: The package-level and seam listings this slot owns, transcribed name for name.
#:
#: These nine were **shape**-checked and never **content**-checked.
#: ``test_dir_mirrors_all_for_every_public_package`` proves ``dir(M) ==
#: sorted(M.__all__)``, which is a statement about two listings *agreeing* — it
#: stays green when a name is added to both.  So the exact defect this whole
#: stream exists to prevent, a listing silently regrowing, had no gate on the
#: modules a plugin author, a solver author and the engine all reach through.
#: ``_CURATED_MODULE_LISTINGS`` above pins the eleven *seams* round 9 curated;
#: this pins the nine that already had a listing and were simply never reviewed
#: against one.
#:
#: Deliberately **not** here: ``tsdynamics.systems`` (180) and
#: ``tsdynamics.analysis`` (52), whose listings are *generated* — from the
#: catalogue and from ``registry.analyses`` — and are pinned as such in
#: ``test_api_contract.py``.  Transcribing a generated listing would make adding
#: a system a two-file edit, which is the thing ``systems/__init__.py`` is built
#: to avoid.
#:
#: Adding a name below is a review, not an accident: that is the entire point.
_CURATED_PACKAGE_LISTINGS: dict[str, list[str]] = {
    "tsdynamics._engine": ["compile", "problem", "run", "symbols"],
    "tsdynamics._engine.compile": [
        "DelaySlot",
        "LoweredSDE",
        "Tape",
        "TapeCompileError",
        "clear_tape_cache",
        "eval_tape",
        "eval_tape_jac",
        "lower_dde",
        "lower_dde_cached",
        "lower_expressions",
        "lower_map",
        "lower_map_cached",
        "lower_map_sweep",
        "lower_map_sweep_cached",
        "lower_ode",
        "lower_ode_cached",
        "lower_sde",
        "lower_sde_cached",
        "map_jacobian_fn",
        "run_tape",
        "tape_cache_stats",
        "tape_jacobian_is_smooth",
    ],
    "tsdynamics._engine.problem": [
        "DDEProblem",
        "DelaySlot",
        "MapProblem",
        "ODEProblem",
        "Problem",
        "SDEProblem",
        "build_problem",
        "dde_problem",
        "map_problem",
        "ode_problem",
        "sde_problem",
    ],
    "tsdynamics._engine.run": [
        "BACKENDS",
        "EngineNotAvailableError",
        "Event",
        "EventSolution",
        "clear_jit_cache",
        "crossings",
        "ensemble",
        "eval_jac",
        "eval_rhs",
        "integrate",
        "integrate_events",
        "jit_cache_stats",
        "make_ode_stepper",
        "map_lyapunov",
        "resolve_backend",
        "sde_ensemble_final",
        "sde_integrate_dense",
        "step_advance",
        "step_advance_to_event",
    ],
    "tsdynamics.plugins": [
        "ALL_GROUPS",
        "ANALYSES_GROUP",
        "PLOT_PRIMITIVES_GROUP",
        "PLOT_TRANSFORMS_GROUP",
        "RENDERERS_GROUP",
        "SOLVERS_GROUP",
        "SYSTEMS_GROUP",
        "import_submodules",
        "iter_entry_points",
        "load_plugins",
        "register_entry_points",
    ],
    "tsdynamics.registry": [
        "Registry",
        "RegistryEntry",
        "SystemEntry",
        "all_systems",
        "analyses",
        "by_family",
        "categories",
        "families",
        "get",
        "plot_transforms",
        "renderers",
    ],
    "tsdynamics._solvers": [
        "DEFAULT_METHOD",
        "Resolution",
        "STIFF_METHOD",
        "SolverCaps",
        "SolverSpec",
        "all_specs",
        "available",
        "available_for",
        "build_kwargs",
        "default_method",
        "get",
        "is_implicit",
        "is_stiff",
        "needs_jacobian",
        "normalize",
        "recommend",
        "register",
        "resolve",
        "select",
        "unregister",
    ],
    "tsdynamics._solvers.select": [
        "DEFAULT_METHOD",
        "Resolution",
        "STIFF_METHOD",
        "available_for",
        "build_kwargs",
        "default_method",
        "is_implicit",
        "is_stiff",
        "needs_jacobian",
        "normalize",
        "recommend",
        "resolve",
        "select",
    ],
    "tsdynamics._utils": [
        "BASIN_ATOL",
        "BASIN_RTOL",
        "DDE_ATOL",
        "DDE_LYAPUNOV_ATOL",
        "DDE_LYAPUNOV_RTOL",
        "DDE_RTOL",
        "DEFAULT_ATOL",
        "DEFAULT_RTOL",
        "make_output_grid",
    ],
}


@pytest.mark.parametrize("mod_name", sorted(_CURATED_PACKAGE_LISTINGS))
def test_the_package_listing_is_exactly_the_contract(mod_name):
    """A reviewed listing, pinned name for name, so it cannot silently regrow.

    The failure message names both directions, because they are different
    mistakes: a **new** name means something public appeared without review, and
    a **missing** one means a public name was demoted without the table being
    told.
    """
    module = importlib.import_module(mod_name)
    expected = _CURATED_PACKAGE_LISTINGS[mod_name]
    measured = dir(module)
    assert measured == expected, (
        f"{mod_name}'s listing moved:\n"
        f"  appeared (review it, then add the row): {sorted(set(measured) - set(expected))}\n"
        f"  vanished (delete the row): {sorted(set(expected) - set(measured))}"
    )


def test_every_pinned_package_listing_is_sorted_and_resolves():
    """Guard the guard: a table that drifted out of sort order would pin nothing."""
    for mod_name, expected in _CURATED_PACKAGE_LISTINGS.items():
        assert expected == sorted(expected), f"{mod_name}: the pinned listing is not sorted"
        module = importlib.import_module(mod_name)
        missing = [n for n in expected if not hasattr(module, n)]
        assert not missing, f"{mod_name} advertises names that do not resolve: {missing}"
    assert len(_CURATED_PACKAGE_LISTINGS) == 9


# ── a record you RECEIVE shows what was measured, not its container's plumbing ──


def _container_leak(obj: object) -> list[str]:
    """Public names ``dir(obj)`` offers that no ``tsdynamics`` class in its MRO owns.

    For a record built on ``list`` / ``tuple`` / ``dict`` that is every inherited
    member — the plumbing of the container it happens to be implemented with.
    """
    own: set[str] = set()
    for klass in type(obj).__mro__:
        if klass.__module__.split(".")[0] == "tsdynamics":
            own |= {a for a in vars(klass) if not a.startswith("_")}
    own |= set(getattr(type(obj), "_fields", ()) or ())
    return sorted(n for n in dir(obj) if not n.startswith("_") and n not in own)


def _received_records() -> dict[str, object]:
    """One instance of each record a user is handed and then tab-completes."""
    from tsdynamics._engine.compile import DelaySlot
    from tsdynamics._utils.escape import Unbounded

    return {
        # ── this slot ────────────────────────────────────────────────────────
        "ts.systems.find(...)": ts.systems.find("delay"),
        "traj.unbounded": Unbounded(
            peak=1e9, start=1.0, growth=1e9, first_sample=3, non_finite=False
        ),
        "engine.compile.DelaySlot": DelaySlot(input_index=3, component=0, delay=1.0),
        # ── sibling slots (see the backlog below) ────────────────────────────
        "ts.analysis.find(...)": analysis.find("chaos"),
        "ts.viz.transforms.find(...)": ts.viz.transforms.find("spectrum"),
        "ts.viz.renderers.find(...)": ts.viz.renderers.find(),
        "ts.viz.styles.find(...)": ts.viz.styles.find(),
    }


#: The records that still hand back their container's plumbing, each with the
#: slot that owns the file.  Split fast/``full`` exactly like
#: ``_UNCURATED_MODULE_LISTINGS``: the fast tier refuses a **new** offender and
#: the ``full`` tier refuses a **stale row**, so a sibling slot landing its half
#: never turns anyone else's run red.
#:
#: The defect is one defect, found on all five ``find`` doors at once: measured,
#: ``ts.analysis.find("is this chaotic").<TAB>`` offers ``append clear copy count
#: extend index insert pop remove reverse sort`` — eleven ways to edit a search
#: result and not one way to use it — on the verb v6 advertises as *the* route to
#: discovery.  ``ts.systems.find`` is fixed here; the other four are one
#: three-line ``__dir__`` each, in files this slot does not own.
#:
#: The two ``list`` rows are the sharper case and need a decision, not a patch:
#: ``renderers.find`` and ``styles.find`` return a **bare** ``list``, which cannot
#: be curated without giving them the small subclass their two siblings already
#: have.  That the four-verb shape returns three different answer shapes is worth
#: fixing on its own.
_UNCURATED_RECEIVED_RECORDS: dict[str, str] = {
    "ts.analysis.find(...)": "RESULTS · AnalysisList, analysis/_discovery.py",
    "ts.viz.transforms.find(...)": "VIZ · TransformList, viz/transforms/_registry.py",
    "ts.viz.renderers.find(...)": "VIZ · a bare list — needs a subclass, not a __dir__",
    "ts.viz.styles.find(...)": "VIZ · a bare list — needs a subclass, not a __dir__",
}


def test_the_received_record_sweep_actually_builds_its_subjects():
    """Guard the guard: a subject that failed to build would certify an empty set."""
    records = _received_records()
    assert len(records) == 7, sorted(records)
    assert len(ts.systems.find("delay")) >= 5, "the find() subject came back empty"
    assert set(_UNCURATED_RECEIVED_RECORDS) <= set(records), "a backlog row names no subject"


@pytest.mark.parametrize(
    "label", [k for k in _received_records() if k not in _UNCURATED_RECEIVED_RECORDS]
)
def test_a_received_record_shows_what_was_measured(label):
    """A record you are *handed* lists its answer, never ``list``/``tuple`` plumbing.

    C2 — a type you receive is not one you type, and that goes for its members
    too.  ``ts.systems.find("delay")`` is the answer to a question; ``.sort()``
    and ``.append()`` are not things anyone does to an answer.

    Reachability is untouched and
    :func:`test_hiding_a_record_s_plumbing_costs_no_capability` measures it.
    """
    obj = _received_records()[label]
    leak = _container_leak(obj)
    assert not leak, (
        f"{label} offers {len(leak)} inherited container members: {leak}.  "
        "Give the class the three-line __dir__; do not add a row to "
        "_UNCURATED_RECEIVED_RECORDS, which may only shrink."
    )


def test_the_received_record_backlog_admits_no_new_offender():
    """The anti-rot half: a new record may not ship with its plumbing showing."""
    measured = {k for k, v in _received_records().items() if _container_leak(v)}
    new = sorted(measured - set(_UNCURATED_RECEIVED_RECORDS))
    assert not new, f"records leaking their container's members: {new}"


@pytest.mark.full
def test_the_received_record_backlog_is_self_cleaning():
    """The shrink half: a row that starts passing must be deleted.

    In the ``full`` tier, and the emptiness check with it: all four rows are
    owned by *other* slots working concurrently, so "you fixed one, now delete
    the row" is bookkeeping — and this file's own argument is that bookkeeping
    fired in the fast tier makes one slot's correct change redden everyone else.
    """
    measured = {k for k, v in _received_records().items() if _container_leak(v)}
    fixed = sorted(set(_UNCURATED_RECEIVED_RECORDS) - measured)
    assert not fixed, f"these records now curate their listing — delete their rows: {fixed}"
    assert _UNCURATED_RECEIVED_RECORDS, "the backlog went empty — delete it and this test"


def test_hiding_a_record_s_plumbing_costs_no_capability():
    """G1 — every hidden member stays bound, callable and correct.

    This is the measurement the whole ruling rests on: a ``__dir__`` is a
    *discovery* change.  If any of this ever fails, the hide was a removal and
    must be reverted.
    """
    from tsdynamics._engine.compile import DelaySlot
    from tsdynamics._utils.escape import Unbounded

    found = ts.systems.find("delay")
    assert len(found) >= 5 and isinstance(found, list)
    first = found[0]
    assert found.count(first) == 1 and found.index(first) == 0
    grown = found.copy()
    grown.append(int)
    assert len(grown) == len(found) + 1
    assert [c.__name__ for c in found] == [c.__name__ for c in list(found)]
    assert "systems match" in repr(found), "the tabulated repr is the answer and must survive"

    record = Unbounded(peak=1e9, start=1.0, growth=1e9, first_sample=3, non_finite=False)
    assert record.peak == 1e9 and record.first_sample == 3
    assert record.count(1.0) == 1 and record.index(1.0) == 1
    assert tuple(record) == (1e9, 1.0, 1e9, 3, False)
    assert list(record._asdict()) == ["peak", "start", "growth", "first_sample", "non_finite"]
    assert "unbounded" in str(record)

    slot = DelaySlot(input_index=3, component=0, delay=1.0)
    assert (slot.input_index, slot.component, slot.delay) == (3, 0, 1.0)
    assert tuple(slot) == (3, 0, 1.0) and slot.count(0) == 1


def test_the_standard_library_does_not_resolve_on_the_top_level():
    """``ts.importlib`` / ``ts.textwrap`` / ``ts.Any`` were never a decision.

    A module's own imports land in its namespace, so ``import importlib`` at the
    top of ``tsdynamics/__init__.py`` made ``from tsdynamics import importlib``
    work — measured, it did.  ``dir(ts)`` is curated, which is exactly why this
    went unnoticed for so long: the leak is invisible until someone types the
    name.  The bindings are ``_``-prefixed now; the corresponding failure is a
    plain teaching ``AttributeError``, because there is no replacement to name.
    """
    for name in ("importlib", "textwrap", "Any"):
        assert not hasattr(ts, name), f"ts.{name} resolves again"
        with pytest.raises(ImportError):
            exec(f"from tsdynamics import {name}", {})  # noqa: S102
    # ...and the one registry that has its own ``__getattr__`` answers the same.
    assert not hasattr(ts.analysis, "Any"), "ts.analysis.Any resolves again"
    assert ts._INTERNAL_SUBMODULES, "the sanctioned bindings table went empty"
    leaked = sorted(
        n
        for n in vars(ts)
        if not n.startswith("_") and n not in ts.__all__ and n not in ts._INTERNAL_SUBMODULES
    )
    assert not leaked, f"tsdynamics binds undeclared public names: {leaked}"


# ── the internal submodules: demoted, never removed ──────────────────────────────


@pytest.mark.parametrize("name", ts._INTERNAL_SUBMODULES)
def test_internal_submodules_are_demoted_but_fully_reachable(name):
    """Machinery stays reachable and stays off the tab surface.

    C5's one exemption: a submodule bound by an ordinary ``import`` cannot be
    intercepted by a module ``__getattr__``, so it is enumerated instead.
    """
    assert name not in ts.__all__, f"{name} crept back onto the curated listing"
    assert name not in dir(ts), f"{name} crept back into tab completion"
    module = getattr(ts, name)
    assert isinstance(module, types.ModuleType)
    assert module is importlib.import_module(f"tsdynamics.{name}")


@pytest.mark.parametrize(
    "stmt",
    [
        "from tsdynamics._engine import run",
        "from tsdynamics.families import SystemBase, ParamSet",
        "from tsdynamics.data import Box, Trajectory",
        "from tsdynamics.derived import PoincareMap",
        "from tsdynamics.registry import all_systems",
        "from tsdynamics._solvers import recommend",
        "from tsdynamics._utils.grids import make_output_grid",
        "from tsdynamics.errors import ConvergenceError",
        "from tsdynamics.plugins import ALL_GROUPS",
    ],
)
def test_demoted_submodule_deep_imports_still_work(stmt):
    """Demotion touches the *listing*, never the import graph."""
    exec(stmt, {})  # noqa: S102


def test_no_region_argument_requires_a_library_type():
    """C1, the toll rule: plain per-axis bounds reach every region door.

    ``Box`` / ``Ball`` / ``Grid`` are demoted precisely because no call needs
    them — the tuple literal is not merely accepted, it is *easier* than the
    type.  If that ever stops being true the right fix is the signature, not a
    promotion.
    """
    import numpy as np

    hen = ts.systems.Henon()
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    grid = [(-2.0, 2.0, 20), (-2.0, 2.0, 20)]

    assert ts.analysis.fixed_points(hen, region=bounds, seed=0) is not None
    assert ts.analysis.basins(hen, grid) is not None
    assert ts.analysis.attractors(hen, grid) is not None
    assert ts.analysis.basin_fractions(hen, bounds, n_seeds=40, seed=0) is not None
    assert ts.analysis.periodic_orbits(hen, 2, region=bounds, seed=0) is not None
    assert ts.data.sampler(bounds, seed=0)().shape == (2,)
    assert ts.data.grid_points([(-1.0, 1.0, 3), (-1.0, 1.0, 3)]).shape == (9, 2)
    assert isinstance(ts.data.set_distance(np.zeros((3, 2)), np.ones((3, 2))), float)


# ── viz (lazy) ───────────────────────────────────────────────────────────────────


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


def test_plot_front_door_resolves_lazily_and_is_the_viz_one():
    """``ts.plot`` is advertised, costs nothing until touched, and is ``ts.viz.plot``.

    "One function object" is not pedantry: the front door used to shadow a
    *different* ``compose.plot`` that accepted ``cols=`` while ``ts.plot``
    refused it, so the same name took different keywords depending on how you
    reached it.
    """
    code = (
        "import sys, tsdynamics as ts\n"
        "assert 'plot' in ts.__all__ and 'T' not in ts.__all__\n"
        "assert 'tsdynamics.viz' not in sys.modules, 'naming plot imported viz'\n"
        "p = ts.plot\n"
        "assert 'tsdynamics.viz' in sys.modules\n"
        "assert ts.plot is p\n"
        "import tsdynamics.viz as viz\n"
        "assert ts.plot is viz.plot, 'ts.plot is not ts.viz.plot'\n"
        "assert 'matplotlib' not in sys.modules, 'the front door pulled in matplotlib'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_viz_is_reachable_and_cached():
    """``ts.viz`` resolves (lazily) to the viz package and caches the binding."""
    import tsdynamics.viz as viz_mod

    assert ts.viz is viz_mod
    assert ts.viz is ts.viz  # cached: identical object on repeat access


# ── the redirect tables themselves ───────────────────────────────────────────────


@pytest.mark.parametrize("table", ["REMOVED_IN_V6", "RENAMED_IN_V6"])
def test_redirect_tables_are_sorted_by_key(table):
    """Sorted and append-only: the format nine builders can all add a row to."""
    keys = list(getattr(_redirects, table))
    assert keys == sorted(keys), f"_redirects.{table} is not sorted by key"


def test_removed_names_are_genuinely_gone():
    """A row claiming a name was removed must be telling the truth.

    Otherwise the table becomes a place where a live capability is declared dead
    — worse than no message, because the user believes it.

    Checked against the homes the deleted layer *lived in*, not against every
    home: ``transforms`` is a genuine homonym.  The v5 ``ts.viz.transforms`` was the
    signal-transform toolbox and is gone; ``ts.viz.transforms`` is the
    plot-transform package and is very much alive.  Two concepts, one word — and
    the reason the plot registry is named ``plot_transforms``.
    """
    for name in _redirects.REMOVED_IN_V6:
        for home in ("analysis", "analysis.results"):
            assert name not in ts._public_names(home), (
                f"{name} is listed as removed in v6 but is public at ts.{home}"
            )
    assert "transforms" in _redirects.REMOVED_IN_V6
    assert "transforms" in ts.viz.__all__, "the homonym this carve-out is about vanished"


def _dotted_targets() -> list[str]:
    """Every ``ts.a.b`` expression named by a redirect row or the scope remedy."""
    import re

    text = "\n".join(
        [line for line, _ in _redirects.RENAMED_IN_V6.values()],
    ) + "\n".join(_redirects.SCOPE_SURGERY_REMEDY)
    return sorted(set(re.findall(r"\bts(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text)))


#: Redirect targets whose *owner slot* has not landed them yet, with the slot
#: named.  Self-cleaning: :func:`test_pending_redirect_targets_are_still_pending`
#: fails the moment one starts resolving, so the list can only shrink — it held
#: ``ts.analysis.find`` until C7 landed it, and this gate is what said so.
_PENDING_TARGETS: dict[str, str] = {}


def test_every_redirect_target_resolves():
    """A redirect that names a spelling which does not exist is worse than none."""
    for target in _dotted_targets():
        if target in _PENDING_TARGETS:
            continue
        obj: object = ts
        for part in target.split(".")[1:]:
            assert hasattr(obj, part), f"redirect target {target} does not resolve"
            obj = getattr(obj, part)


def test_pending_redirect_targets_are_still_pending():
    """Self-cleaning exemption list: drop a row the moment its owner lands it."""
    for target, owner in _PENDING_TARGETS.items():
        obj: object = ts
        for part in target.split(".")[1:]:
            if not hasattr(obj, part):
                break
            obj = getattr(obj, part)
        else:  # pragma: no cover - fires exactly once, when the owner lands it
            pytest.fail(f"{target} resolves now ({owner}) — delete it from _PENDING_TARGETS")


def test_scope_surgery_remedy_names_only_survivors():
    """What the "that was removed" answer points at must still be here."""
    assert "rqa" in ts.analysis.__all__
    assert "embed" in ts.analysis.__all__
    assert "lyapunov_from_data" in ts.analysis.__all__


# ---------------------------------------------------------------------------
# No reachable module offers a name it merely imported
# ---------------------------------------------------------------------------


def _foreign_names(module: object) -> list[str]:
    """Return the public names on ``module`` that belong to another library.

    ``__all__`` governs ``import *`` and nothing else, so a module that imports
    ``symengine.sin`` or ``numpy as np`` offers them on ``dir()`` — and 175 of
    the catalogue's classes sat in modules doing exactly that, so a page holding
    51 systems tab-completed 58 names.  The fix is ``__dir__``; this is the gate.

    A ``X | Y`` type union reports ``__module__ == "typing"`` and is a false
    positive, so a name the module itself declares in ``__all__`` is exempt.
    """
    import types

    declared = set(getattr(module, "__all__", ()))
    out = []
    for name in dir(module):
        if name.startswith("_") or name in declared:
            continue
        value = getattr(module, name, None)
        owner = getattr(value, "__module__", None)
        if isinstance(value, types.ModuleType):
            if not (value.__name__ or "").startswith("tsdynamics"):
                out.append(name)
        elif owner and not owner.startswith("tsdynamics") and owner != "builtins":
            out.append(name)
    return out


def test_no_reachable_module_offers_a_borrowed_name():
    """Every module a user can tab into lists only names this library owns.

    Measured before the fix: 18 reachable modules leaked 50 names — SymEngine's
    ``sin``/``cos``/``exp``/``sqrt``/``tanh``/``Min``, ``numpy as np``,
    ``ClassVar``.  A module whose own path is ``_``-private is exempt: nobody
    tab-completes into it.
    """
    import importlib
    import pkgutil

    import tsdynamics

    offenders = {}
    for info in pkgutil.walk_packages(tsdynamics.__path__, "tsdynamics."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:  # pragma: no cover - an optional backend, not our business
            continue
        leaked = _foreign_names(module)
        if leaked:
            offenders[info.name] = leaked

    assert not offenders, (
        "these modules offer names they only imported — give each a __dir__ "
        f"returning sorted(__all__):\n{offenders}"
    )


def test_every_catalogue_module_lists_the_classes_it_defines():
    """``__all__`` on a catalogue module is the classes defined there, all of them.

    The ``__dir__`` above is only honest if ``__all__`` is complete: a system
    missing from it would vanish from its own module's tab surface while still
    being registered and importable.
    """
    import importlib
    import inspect
    import pkgutil

    import tsdynamics.systems as systems
    from tsdynamics.families.base import SystemBase

    for info in pkgutil.walk_packages(systems.__path__, "tsdynamics.systems."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        module = importlib.import_module(info.name)
        defined = {
            name
            for name, value in vars(module).items()
            if not name.startswith("_")
            and inspect.isclass(value)
            and issubclass(value, SystemBase)
            and value.__module__ == info.name
        }
        if not defined:  # a category __init__, which re-exports rather than defines
            continue
        listed = set(getattr(module, "__all__", ()))
        assert defined <= listed, f"{info.name}.__all__ omits {sorted(defined - listed)}"
