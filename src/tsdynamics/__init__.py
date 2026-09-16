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

The curated top level — seventeen names
---------------------------------------
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
six names for ``except``       :class:`TSDynamicsError` :class:`ConvergenceError`
                               :class:`StepBudgetError` :class:`BackendError`
                               :class:`InvalidParameterError`
                               :class:`InvalidInputError`
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
:mod:`~tsdynamics.engine`, :mod:`~tsdynamics.families`, :mod:`~tsdynamics.plugins`,
:mod:`~tsdynamics.registry`, :mod:`~tsdynamics.solvers`, :mod:`~tsdynamics.utils`.

:mod:`~tsdynamics.viz` and :func:`plot` resolve **lazily**, so a plain
``import tsdynamics`` pulls in no plotting library.
"""

import importlib
import textwrap
from typing import Any

from . import (
    analysis as analysis,
)

# Machinery submodules: bound eagerly so ``ts.engine`` / ``ts.registry`` resolve
# and ``from tsdynamics.engine import run`` imports, but kept OFF ``__all__`` /
# ``dir()`` — see ``_INTERNAL_SUBMODULES``.  The redundant ``as`` form marks them
# as deliberate re-exports rather than unused imports.
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
    solvers as solvers,
)
from . import (
    systems as systems,
)
from . import (
    utils as utils,
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

# The six names you type inside ``except``.  They are promoted from
# ``ts.errors.<Name>`` because catching is ordinary work and the module dot was
# pure toll; ``ts.errors`` itself stays bound forever (the Rust bridge imports
# the path by name at ``crates/tsdyn-core/src/lib.rs``, so it is an ABI).
from .errors import (
    BackendError,
    ConvergenceError,
    InvalidInputError,
    InvalidParameterError,
    StepBudgetError,
    TSDynamicsError,
)

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
]

#: Submodules bound by an ordinary ``import`` and kept **off** ``__all__`` /
#: ``dir()``.  They stay reachable forever — ``ts.engine`` resolves and
#: ``from tsdynamics.engine import run`` imports — they simply do not earn a tab
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
    "engine",
    "errors",
    "families",
    "plugins",
    "registry",
    "solvers",
    "utils",
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
    lines = textwrap.wrap(text, width=_WIDTH, initial_indent=pad)
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
        module = importlib.import_module(f"{__name__}.{home}")
    except Exception:  # noqa: BLE001 - see the docstring; this is the error path
        return ()
    return tuple(getattr(module, "__all__", ()))


def _listing_size(home: str) -> int:
    """How many *things* a home's tab listing offers, not counting submodules.

    ``ts.systems.__all__`` carries the two category packages alongside the 177
    system classes, so ``len(__all__)`` would advertise "179 built-in systems"
    and be wrong by exactly the names that are not systems.
    """
    import types

    try:
        module = importlib.import_module(f"{__name__}.{home}")
    except Exception:  # noqa: BLE001 - the error path; see _public_names
        return 0
    return sum(
        1
        for name in getattr(module, "__all__", ())
        if not isinstance(getattr(module, name, None), types.ModuleType)
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
        body = _wrap(f"ts.{name} was renamed in v6. Same capability, one spelling:", prefix)
        body.append(f"    {line}")
        body.extend(_wrap(f"({why})", ""))
        return _MovedInV6("\n".join(body))

    if name in _REMOVED_IN_V6:
        body = _wrap(
            f"ts.{name} was removed in v6 with {_REMOVED_IN_V6[name]}: TSDynamics is "
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
        f"ts.{name} moved in v6: the top level is {len(__all__)} names now, and "
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
        f"    ts.systems.<TAB>             # the {_listing_size('systems')} built-in systems",
        f"    ts.analysis.<TAB>            # the {_listing_size('analysis')} quantifiers",
        # A pathologically long guess must not push the last line past the wrap.
        # The comment is the first thing to go, because the call is the payload.
        search.ljust(32) + comment if len(search) + len(comment) <= _WIDTH else search,
    ]
    return AttributeError("\n".join(body))


def __getattr__(name: str) -> Any:
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
        module = importlib.import_module(f"{__name__}.viz")
        globals()["viz"] = module  # cache: subsequent access skips __getattr__
        return module
    if name in _VIZ_FRONT_DOOR:
        module = importlib.import_module(f"{__name__}.viz")
        for attr in _VIZ_FRONT_DOOR:
            globals()[attr] = getattr(module, attr)
        return globals()[name]
    if name in _RENAMED_IN_V6 or name in _REMOVED_IN_V6 or _home_of(name) is not None:
        raise _moved_error(name)
    raise _attribute_error(name)


def __dir__() -> list[str]:
    """Return exactly ``__all__`` — the C5 rule written out in the module docstring."""
    return sorted(__all__)
