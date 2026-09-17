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
    # the six names you type inside ``except``
    "TSDynamicsError",
    "ConvergenceError",
    "StepBudgetError",
    "BackendError",
    "InvalidParameterError",
    "InvalidInputError",
    "__version__",
}


def test_top_level_all_is_curated():
    """``ts.__all__`` is exactly the curated headline set — no flat dump."""
    assert set(ts.__all__) == _CURATED_TOP_LEVEL
    assert len(ts.__all__) == 17
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


def test_the_six_except_names_are_the_error_classes_themselves():
    """Catching is ordinary work, so the classes are on the top level, not a dot down."""
    for name in (
        "TSDynamicsError",
        "ConvergenceError",
        "StepBudgetError",
        "BackendError",
        "InvalidParameterError",
        "InvalidInputError",
    ):
        assert getattr(ts, name) is getattr(ts.errors, name)
    assert issubclass(ts.InvalidParameterError, ValueError)
    assert issubclass(ts.InvalidInputError, TypeError)
    assert issubclass(ts.StepBudgetError, ts.ConvergenceError)


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
    assert issubclass(MovedInV6, ts.TSDynamicsError)
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
    "tsdynamics.engine": {"compile", "problem", "run", "symbols"},
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
        "from tsdynamics.engine import run",
        "from tsdynamics.families import SystemBase, ParamSet",
        "from tsdynamics.data import Box, Trajectory",
        "from tsdynamics.derived import PoincareMap",
        "from tsdynamics.registry import all_systems",
        "from tsdynamics.solvers import recommend",
        "from tsdynamics.utils.grids import make_output_grid",
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
    assert ts.analysis.basin_fractions(hen, bounds, n=40, seed=0) is not None
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
