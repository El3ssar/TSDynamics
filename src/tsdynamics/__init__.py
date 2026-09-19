"""
TSDynamics — compiled dynamical systems: integration, iteration, and chaos analysis.

Quick start
-----------
>>> import tsdynamics as ts
>>> traj = ts.systems.Lorenz().run(final_time=100.0, dt=0.01)
>>> traj.t.shape, traj.y.shape
((10001,), (10001, 3))

You define a **system** and you get back a **trajectory**.  Everything else is a
verb on one of those two things, or lives one dot down a named registry.

The whole library in six lines
------------------------------
>>> lor = ts.systems.Lorenz()                      # a system: find it, or write it
>>> traj = lor.run(final_time=100.0, dt=0.01)      # run it  -> a Trajectory
>>> ts.plot(traj)                                  # doctest: +SKIP
>>> spec = ts.analysis.lyapunov_spectrum(lor)      # doctest: +SKIP
>>> ts.analysis.find(traj)                         # doctest: +SKIP
>>> print(lor.info)                                # doctest: +SKIP

Four things to know, and nothing else is required reading:

``system.info``
    prints the equations back at you, with the parameters, the defaults and the
    literature they come from — the fastest way to check you typed your model
    right.
``ts.analysis.find(x)``
    ``x`` is what you are holding — a system, a trajectory, an array, another
    result — or a plain-English question (``find("is this chaotic")``).
``print(ts.analysis.__doc__)``
    the whole quantifier catalogue, grouped by what you have to hold.
``ts.plot(anything)``
    one plotting verb; ``ts.viz.compatibility()`` lists what can be drawn.

Writing your own system is four declarations —
``params`` / ``dim`` / ``variables`` / one kernel; see
``help(ts.ContinuousSystem)``, which also pre-empts the one trap (the state is
an **accessor**: ``y(0)``, not ``y[0]``).

Design notes — the curated top level, and why a name is not here
----------------------------------------------------------------
Everything below this line is the *rationale* for the namespace.  It is worth
reading before you add a name to it, and not otherwise.

The eleven names
----------------
``tsdynamics.<TAB>`` shows **only what you type**:

============================== =================================================
five family bases              :class:`ContinuousSystem` :class:`DelaySystem`
                               :class:`DiscreteMap` :class:`StochasticSystem`
                               :class:`WrappedSystem` — what you *subclass*
one received type              :class:`Trajectory` — you annotate it, you
                               ``isinstance`` it, you construct it from data
one plotting verb              :func:`plot`
three registries               :mod:`~tsdynamics.systems`
                               :mod:`~tsdynamics.analysis` :mod:`~tsdynamics.viz`
``__version__``                the package version
============================== =================================================

THE RULE, and why a name is not here
------------------------------------
*A public name is a verb you call on the thing you are already holding, a fact
about that thing, or a member of exactly one named registry.  Everything else
still exists — importable, reachable, tested — it just stops shouting.*

Four corollaries, applied mechanically, so the next person to add a name argues
against a written rule rather than against a list:

``C1`` **the toll rule.**  Plain Python — tuples, lists, strings, numbers,
arrays, dicts — reaches every front door.  A name exported *because a signature
demands it* is evidence of a signature bug; fix the signature, then demote the
name, never the reverse.  ``ts.analysis.basins(vdp, [(-3, 3), (-3, 3)])`` takes
plain bounds, which is why :class:`~tsdynamics.data.Box` is not here.

``C2`` **received is not typed.**  A type you only ever get *back* lives at one
importable address and appears in no ``__all__``.  Exactly two exceptions —
:class:`Trajectory` and :class:`~tsdynamics.viz.Plot`.

``C3`` **one concept, one spelling.**  Two grammars for one argument is the
silent-wrong-answer defect, not flexibility.

``C4`` **a name resolves to what it advertises.**  No ``__all__`` entry may be
shadowed by a submodule of the same name.

``C5`` **dir() is the truth, for API names.**  ``dir(ts) == sorted(ts.__all__)``,
every listed name resolves, and **no API name resolves unless it is listed** — no
system class, analysis function, result class, derived wrapper, region primitive
or IR noun answers to ``ts.<name>`` any more.  Submodules bound by an ordinary
``import`` are the one exemption and are enumerated in
:data:`_INTERNAL_SUBMODULES`.

Where everything went
---------------------
Nothing was deleted; the addresses are one dot longer and the error tells you
which one::

    ts.correlation_dimension   ->  ts.analysis.correlation_dimension
    ts.Lorenz                  ->  ts.systems.Lorenz
    ts.PoincareMap             ->  ts.derived.PoincareMap   (verb: sys.poincare)
    ts.Box                     ->  ts.data.Box              (or just pass bounds)
    ts.LyapunovSpectrum        ->  ts.analysis.results.LyapunovSpectrum
    ts.T                       ->  ts.viz.spec.T            (or a plain pair)

A wrong guess is answered by the ordered ladder in :func:`__getattr__`.  An exact
hit in a redirect table raises :class:`~tsdynamics.errors.MovedInV6` — an
``ImportError``, so the text survives ``from tsdynamics import X`` as well as
``ts.X``.  A guess stays an ``AttributeError``, so ``hasattr`` keeps working.

Still reachable, off every listing
----------------------------------
:mod:`~tsdynamics.errors` (the module — the Rust bridge imports it by name, so it
is an ABI, not a choice), :mod:`~tsdynamics.data`, :mod:`~tsdynamics.derived`,
:mod:`~tsdynamics._engine`, :mod:`~tsdynamics.families`, :mod:`~tsdynamics.plugins`,
:mod:`~tsdynamics.registry`, :mod:`~tsdynamics._solvers`, :mod:`~tsdynamics._utils`.

:mod:`~tsdynamics.viz` and :func:`plot` resolve **lazily**, so a plain
``import tsdynamics`` pulls in no plotting library.
"""

# Underscored on purpose.  A module's own imports land in its namespace, so a
# plain ``import importlib`` here makes ``ts.importlib`` resolve and
# ``from tsdynamics import importlib`` work — measured, both did.  ``dir(ts)`` is
# curated by ``__dir__`` and never showed them, which is precisely why nobody
# noticed: the leak is invisible until a user types the name.  Nothing in the
# library decided to re-export the standard library, so the binding goes private
# rather than the listing growing a carve-out (§11.3 T4).
import importlib as _importlib
import textwrap as _textwrap
from typing import Any as _Any

from . import (
    analysis as analysis,
)

# Machinery submodules: bound eagerly so ``ts.engine`` / ``ts.registry`` resolve
# and ``from tsdynamics._engine import run`` imports, but kept OFF ``__all__`` /
# ``dir()`` — see ``_INTERNAL_SUBMODULES``.  The redundant ``as`` form marks them
# as deliberate re-exports rather than unused imports.
from . import (
    data as data,
)
from . import (
    derived as derived,
)
from . import (
    errors as errors,
)
from . import (
    families as families,
)
from . import (
    plugins as plugins,
)
from . import (
    registry as registry,
)
from . import (
    systems as systems,
)
from ._redirects import (
    REMOVED_IN_V6 as _REMOVED_IN_V6,
)
from ._redirects import (
    RENAMED_IN_V6 as _RENAMED_IN_V6,
)
from ._redirects import (
    SCOPE_SURGERY_REMEDY as _SCOPE_SURGERY_REMEDY,
)

# The typed exceptions are NOT promoted.  The hierarchy is purely additive —
# ``InvalidParameterError`` IS a ``ValueError``, ``ConvergenceError`` IS a
# ``RuntimeError``, ``InvalidInputError`` IS a ``TypeError`` — so ``except
# ValueError`` already catches a bad ``dt`` and no user needs this library's
# spelling to write working code.  The names are a *refinement* you reach for
# when you want to tell one failure from another, and a refinement lives at its
# own address: ``ts.errors.<Name>``.  ``ts.errors`` stays bound forever (the Rust
# bridge imports the path by name at ``crates/tsdyn-core/src/lib.rs``, an ABI).
# Not exported: it is the exception this module's ``__getattr__`` RAISES, never
# one a user types.  ``ts.errors.MovedInV6`` is its address.
from .errors import (
    MovedInV6 as _MovedInV6,
)
from .families import (
    ContinuousSystem,
    DelaySystem,
    DiscreteMap,
    StochasticSystem,
    Trajectory,
    WrappedSystem,
)

# Single source of truth for the package version; rewritten by python-semantic-release.
__version__ = "5.4.0"

#: The curated top level.  **Sorted, one name per line** — the same mechanical
#: policy every public package in this library follows, because a list grouped by
#: comment is the maximally merge-hostile format (two people adding a name both
#: append inside the same group).  The *reasons* live in the module docstring,
#: where they can be read as prose; this list is data.
__all__ = [
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "StochasticSystem",
    "Trajectory",
    "WrappedSystem",
    "__version__",
    "analysis",
    "plot",
    "systems",
    "viz",
]

#: Submodules bound by an ordinary ``import`` and kept **off** ``__all__`` /
#: ``dir()``.  They stay reachable forever — ``ts.engine`` resolves and
#: ``from tsdynamics._engine import run`` imports — they simply do not earn a tab
#: slot, and the reason is C1: **nothing in them is required to make a call.**
#:
#: ``errors`` is here for a second, harder reason: the Rust bridge does
#: ``py.import("tsdynamics.errors")`` when it builds a typed exception at the FFI
#: boundary, so the module path is an ABI.  It left ``__all__`` only because its
#: six classes are on the top level now.
#:
#: This tuple is the single source of truth for that decision, and the C5 gate in
#: ``tests/test_namespace_curation.py`` reads it: promoting or demoting a
#: submodule is a one-line edit here.
_INTERNAL_SUBMODULES = (
    "data",
    "derived",
    "errors",
    "families",
    "plugins",
    "registry",
)

#: The public homes a curated top level sends people to, searched **in this
#: order** so the address a user is most likely to have meant wins a name that is
#: public in two places.  Each is consulted through its own ``__all__``, so the
#: address book cannot go stale the way a hand-written table does — which is how
#: ``ts.LyapunovSpectrum`` came to resolve while ``ts.Embedding`` did not.
_PUBLIC_HOMES = (
    "systems",
    "analysis",
    "analysis.results",
    "data",
    "derived",
    # The typed exceptions.  They left ``__all__`` in v6.1 (the hierarchy is
    # additive, so ``except ValueError`` already works and nobody NEEDS this
    # library's spelling), which makes a guess at ``ts.ConvergenceError`` the only
    # way a user learns the new address — so ``errors`` must be searched here.
    "errors",
    # The four names a user reaches for when they WRITE against the library
    # rather than call it — ``System`` (the runtime Protocol you annotate and
    # ``isinstance``-check), ``SystemBase`` (the class you subclass to add a
    # family), ``ParamSet`` and ``MetaStore``.  Without this row they were the
    # only public-home names with no forwarding address: ``ts.System`` answered
    # with three classes that are not it (``DerivedSystem`` /
    # ``TangentSystem`` / ``WrappedSystem``) and the other three fell through to
    # the generic "tab-complete a registry" line, which cannot find a type.
    "families",
    "viz",
    "viz.spec",
)

#: Names the plotting front door resolves lazily from :mod:`tsdynamics.viz`, so
#: advertising ``plot`` in ``__all__`` still costs ``import tsdynamics`` no
#: plotting library.  Resolved from the *package* (not ``viz.transforms``) so
#: that ``ts.plot is ts.viz.plot`` holds by construction rather than by luck.
_VIZ_FRONT_DOOR = frozenset({"plot"})

#: Message width, counted **including the traceback's own class-name prefix**.
#: Before the prefix was counted, a message could be a tidy 88 columns in the
#: source and wrap in the terminal at the one place it must not: the first line,
#: which is the one carrying the name that was typed.
_WIDTH = 88


def _wrap(text: str, prefix: str) -> list[str]:
    """Wrap *text* to :data:`_WIDTH`, reserving room for the traceback *prefix*.

    The first physical line shares its row with ``prefix`` (``"AttributeError: "``
    or ``"tsdynamics.errors.MovedInV6: "``), so it gets that much less width.
    """
    pad = " " * len(prefix)
    lines = _textwrap.wrap(text, width=_WIDTH, initial_indent=pad)
    if lines:
        lines[0] = lines[0][len(pad) :]
    return lines


def _prefix_of(cls: type[BaseException]) -> str:
    """Return the text Python itself prepends when it prints an exception of *cls*."""
    if cls.__module__ in ("builtins", "__main__"):
        return f"{cls.__qualname__}: "
    return f"{cls.__module__}.{cls.__qualname__}: "


def _public_names(home: str) -> tuple[str, ...]:
    """Return one public home's ``__all__``, or ``()`` if it cannot be imported.

    A home that fails to import must not turn a typo into *its* traceback: this
    runs only on the error path, where the user's question is "what should I have
    typed", and answering it partially beats answering with an unrelated
    ``TypeError`` from a half-installed plotting backend.
    """
    try:
        module = _importlib.import_module(f"{__name__}.{home}")
    except Exception:  # noqa: BLE001 - see the docstring; this is the error path
        return ()
    return tuple(getattr(module, "__all__", ()))


#: The four verbs every curated registry namespace answers (``ts.systems``,
#: ``ts.analysis``, ``ts.viz.transforms``, …).  They are listed beside the
#: catalogue but they are not *of* it, so :func:`_catalogue_size` discounts them.
_REGISTRY_VERBS = frozenset({"find", "get", "names", "register"})


def _catalogue_size(home: str) -> int:
    """How many CATALOGUE ENTRIES a home offers — systems, or analyses.

    Not ``len(__all__)``: a registry namespace lists its own verbs and its
    result namespace beside the entries, and the sentence this number lands in
    says *"built-in systems"* or *"quantifiers"*, which those are not.  Measured
    before the discount, the same error message advertised **"the 180 built-in
    systems"** (177 + ``names``/``find``/``get``) and **"the 51 quantifiers"**
    (49 + ``find``/``register``), so two of the three counts a lost user was
    shown in one message were wrong, on a surface whose whole pitch is that the
    listing is the contract.

    Submodules are discounted for the same reason (``ts.analysis.results`` is a
    namespace of result classes, not a quantifier).
    """
    import types

    try:
        module = _importlib.import_module(f"{__name__}.{home}")
    except Exception:  # noqa: BLE001 - the error path; see _public_names
        return 0
    return sum(
        1
        for name in getattr(module, "__all__", ())
        if name not in _REGISTRY_VERBS
        and not isinstance(getattr(module, name, None), types.ModuleType)
    )


def _line_for(home: str, name: str) -> str:
    """Render the line a user should type for *name* at *home*.

    A system is shown *constructed* (``ts.systems.Lorenz()``) because a class you
    never instantiate is not a line you can run; everything else is shown as the
    qualified name, which is a complete expression you then call.
    """
    qualified = f"ts.{home}.{name}"
    return f"{qualified}()" if home == "systems" else qualified


def _homes_of(name: str) -> list[str]:
    """Return every line to type for *name*, across :data:`_PUBLIC_HOMES`.

    An **exact** hit in a public ``__all__`` is a certainty and must outrank every
    fuzzy guess below it.  Without this ordering a name that merely moved was
    answered with nonsense — ``ts.region`` (real, at ``ts.data.region``) used to
    suggest ``ts.systems.Oregonator()``.

    A few names are public at **more than one** address — ``find`` is a verb on
    ``ts.analysis`` *and* on ``ts.systems``, and ``set_distance`` lives at both
    ``ts.data`` and ``ts.analysis``.  Naming only the first would answer half the
    people who typed it about a door they did not mean, so all of them are
    listed.
    """
    return [_line_for(home, name) for home in _PUBLIC_HOMES if name in _public_names(home)]


def _home_of(name: str) -> str | None:
    """Return the first address :func:`_homes_of` knows for *name*, or ``None``."""
    found = _homes_of(name)
    return found[0] if found else None


def _rank(name: str) -> int:
    """Tie-break rank for a suggestion: **verbs before types**.

    At equal similarity a user is far likelier to have meant the function they
    call than the class they only ever receive — ``fixed_points`` over
    ``FixedPoint``, ``basins`` over ``BasinsResult``.  ``difflib`` breaks ties by
    string order, which encodes nothing at all, so the tie-break is stated here.
    """
    return 0 if name[:1].islower() else 1


#: A suggestion must clear **both** floors.  One number cannot do this job:
#:
#: * ``_MIN_RATIO`` alone, at 0.6, offers ``ts.systems.Tent()`` for ``ts.Event``
#:   (ratio 0.667) — three shared letters sold as a confident answer;
#: * raising it to reject that also rejects ``lyapunov`` → ``lyapunov_spectrum``
#:   (ratio 0.640), which is the single most useful suggestion in the library.
#:
#: The discriminator is **coverage**: how much of what the *user typed* the
#: candidate accounts for.  A typo or an abbreviation is almost entirely covered
#: (``Lorentz`` → ``Lorenz`` 0.857, ``lyapunov`` → ``lyapunov_spectrum`` 1.000);
#: a coincidence is not (``Event`` → ``Tent`` 0.600).
_MIN_RATIO = 0.6
_MIN_COVERAGE = 0.8


def _suggest(name: str, *, n: int = 3) -> list[str]:
    """Up to *n* lines to type, ranked by similarity then by :func:`_rank`."""
    import difflib

    scored: list[tuple[float, int, str, str]] = []
    seen: set[str] = set()
    for home in _PUBLIC_HOMES:
        for candidate in _public_names(home):
            if candidate in seen:
                continue
            seen.add(candidate)
            matcher = difflib.SequenceMatcher(None, name, candidate)
            ratio = matcher.ratio()
            if ratio < _MIN_RATIO:
                continue
            covered = sum(block.size for block in matcher.get_matching_blocks())
            if covered / max(len(name), 1) < _MIN_COVERAGE:
                continue
            scored.append((-ratio, _rank(candidate), candidate, _line_for(home, candidate)))
    scored.sort()
    return [line for _, _, _, line in scored[:n]]


def _moved_error(name: str) -> ImportError:
    """Case 1-3: an **exact** hit in a redirect table or a public home."""
    prefix = _prefix_of(_MovedInV6)

    if name in _RENAMED_IN_V6:
        line, why = _RENAMED_IN_V6[name]
        body = _wrap(f"ts.{name} was renamed. Same capability, one spelling:", prefix)
        body.append(f"    {line}")
        body.extend(_wrap(f"({why})", ""))
        return _MovedInV6("\n".join(body))

    if name in _REMOVED_IN_V6:
        body = _wrap(
            f"ts.{name} was removed along with {_REMOVED_IN_V6[name]}: TSDynamics is "
            "scoped to phase-space methods, and the generic time-series layer lives "
            "in a companion library now. What stayed:",
            prefix,
        )
        body.extend(f"    {line}" for line in _SCOPE_SURGERY_REMEDY)
        return _MovedInV6("\n".join(body))

    addresses = _homes_of(name)
    assert addresses, name  # only called after _home_of said yes
    where = "its own address" if len(addresses) == 1 else "these addresses"
    body = _wrap(
        f"ts.{name} moved: the top level is {len(__all__)} names now, and "
        f"this one lives at {where}.",
        prefix,
    )
    body.extend(f"    {line}" for line in addresses)
    return _MovedInV6("\n".join(body))


def _attribute_error(name: str) -> AttributeError:
    """Case 4-5: a **guess** — a near miss, or a name nothing matches.

    These stay ``AttributeError`` deliberately.  ``hasattr(ts, anything)`` has to
    keep answering ``False`` for every name in the universe; only the enumerated
    dead names of :func:`_moved_error` are allowed to raise.
    """
    prefix = _prefix_of(AttributeError)
    # CASE first.  ``ts.Systems`` is one shift key away from the registry the
    # user wants, and ``difflib`` scores case as an ordinary character
    # difference, so the fuzzy pass answered it with three unrelated wrapper
    # classes (``DerivedSystem`` / ``TangentSystem`` / ``WrappedSystem``) and
    # never with ``ts.systems``.  A wrong-case name is a certainty, not a guess.
    folded = {n.casefold(): n for n in __all__}
    cased = folded.get(name.casefold())
    if cased is not None and cased != name:
        body = _wrap(
            f"module 'tsdynamics' has no attribute {name!r} — it is spelled "
            f"{cased!r} (case matters).",
            prefix,
        )
        body.append(f"    ts.{cased}")
        return AttributeError("\n".join(body))

    hits = _suggest(name)
    if hits:
        body = _wrap(f"module 'tsdynamics' has no attribute {name!r}. Did you mean:", prefix)
        body.extend(f"    {hit}" for hit in hits)
        return AttributeError("\n".join(body))

    # Nothing is close enough to guess, so answer the question behind the guess:
    # *how do I find it?*  The last line is the one that reaches a user who never
    # guesses a name close to anything — the counts are read live, because a
    # hardcoded total goes stale every time a system is added (the docs already
    # said 171 where the registry said 177).
    body = _wrap(f"module 'tsdynamics' has no attribute {name!r}.", prefix)
    search = f"    ts.analysis.find({name!r})"
    comment = " # ...or search by what it does"
    body += [
        "Tab-complete a registry, or search:",
        f"    ts.systems.<TAB>             # the {_catalogue_size('systems')} built-in systems",
        f"    ts.analysis.<TAB>            # the {_catalogue_size('analysis')} quantifiers",
        # A pathologically long guess must not push the last line past the wrap.
        # The comment is the first thing to go, because the call is the payload.
        search.ljust(32) + comment if len(search) + len(comment) <= _WIDTH else search,
    ]
    return AttributeError("\n".join(body))


def __getattr__(name: str) -> _Any:
    """Resolve ``viz`` / ``plot`` lazily, and teach every other miss.

    The ordered ladder, and why the order is the order:

    ============ ====================================== ========================
    case         hit                                    raises
    ============ ====================================== ========================
    0            a private or dunder probe              ``AttributeError``, and
                                                        imports nothing
    0            ``viz`` / ``plot``                     (resolves, and caches)
    1            exact in ``_RENAMED_IN_V6``            ``MovedInV6``
    2            exact in ``_REMOVED_IN_V6``            ``MovedInV6``
    3            exact in a public home's ``__all__``   ``MovedInV6``
    4            a near miss (ranked, ≥ 0.6)            ``AttributeError``
    5            nothing matches                        ``AttributeError``
    ============ ====================================== ========================

    Case 3 **before** case 4 is load-bearing: an exact hit in a public ``__all__``
    is a certainty and must outrank every guess.  Cases 4-5 **must** stay
    ``AttributeError``, or ``hasattr`` breaks for every name that was never here.

    Case 0's ``_``-prefix short-circuit is not cosmetic either: ``from tsdynamics
    import _rust`` asks ``hasattr`` first and falls back to importing the
    submodule *only* if that said ``False`` — so a ``MovedInV6`` there would break
    the engine import, and any import at all would make a notebook's
    ``_ipython_canary_`` probe pull in the plotting layer.
    """
    if name.startswith("_"):
        raise AttributeError(f"module 'tsdynamics' has no attribute {name!r}")
    if name == "viz":
        # import_module loads the submodule through the import machinery without
        # re-entering this __getattr__ (a plain ``from . import viz`` recurses).
        module = _importlib.import_module(f"{__name__}.viz")
        globals()["viz"] = module  # cache: subsequent access skips __getattr__
        return module
    if name in _VIZ_FRONT_DOOR:
        module = _importlib.import_module(f"{__name__}.viz")
        for attr in _VIZ_FRONT_DOOR:
            globals()[attr] = getattr(module, attr)
        return globals()[name]
    if name in _RENAMED_IN_V6 or name in _REMOVED_IN_V6 or _home_of(name) is not None:
        raise _moved_error(name)
    raise _attribute_error(name)


def __dir__() -> list[str]:
    """Return exactly ``__all__`` — the C5 rule written out in the module docstring."""
    return sorted(__all__)
