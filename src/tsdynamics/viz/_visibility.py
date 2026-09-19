"""The two helpers that make a curated tab surface cheap — and therefore uniform.

Hiding, in this library, is a **discovery** change and never a reachability one.
Every name these helpers remove from a listing stays bound, importable, callable
and tested; ``__dir__`` takes no part in attribute lookup, and ``__all__``
governs only ``from X import *``.  So nothing here can break a caller — it can
only stop shouting at a newcomer.

Why both halves are needed
--------------------------
``__all__`` alone does **not** curate ``dir()``.  A module that declares one and
stops has a *documented* surface and an *actual* surface that disagree, and it is
the actual one a REPL completes against.  Measured across ``tsdynamics.viz``
before this module existed: 32 modules declared ``__all__``, **none of the 32
also defined** ``__dir__``, and between them they offered **405** public names
that no ``__all__`` claims — ``np``, ``Any``, ``Mapping``, ``field``, every
helper a module happens to import, and every private-by-intent function whose
name simply lacks an underscore.

The fix has to be one line per module or it will not be applied to every module,
which is why it is a factory rather than a recipe::

    __dir__ = listing_dir(__all__)

Two rules this module exists to enforce
---------------------------------------
1. **A module that declares ``__all__`` defines ``__dir__``** (the contract's
   §11.1 enforcement clause).  :func:`listing_dir` is that definition.
2. **A record hides machinery, not declaration.**  :func:`dir_without` takes the
   default listing and removes a named, documented set — so the removal is
   reviewable as a set literal at the class, instead of an allow-list somewhere
   else that drifts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

__all__ = ["INHERITED_DICT_METHODS", "INHERITED_STR_METHODS", "dir_without", "listing_dir"]

#: The 47 public methods ``str`` contributes to **every** ``str`` subclass.
#:
#: Subclassing a builtin is how three values in this library stay drop-in
#: replacements for the plain thing they replace — ``Plot.title`` is a ``str``
#: you can compare and format, and :class:`~tsdynamics.viz.spec.PlotKind` /
#: :class:`~tsdynamics.viz._frames.FrameSpace` are ``StrEnum``\ s whose members
#: serialize as their own value.  The tab surface pays for that in full:
#: ``capitalize`` / ``casefold`` / ``expandtabs`` / ``zfill`` and 43 siblings
#: land on a value nobody reached for in order to manipulate text.
#:
#: Measured across ``src/``, ``tests/``, ``docs/`` and ``hooks/``: **zero**
#: call sites invoke any of the 47 on a kind or a frame space.  Hiding them is
#: a ``__dir__`` edit, so even a future caller keeps working.
INHERITED_STR_METHODS: frozenset[str] = frozenset(n for n in dir(str) if not n.startswith("_"))

#: The 11 public methods ``dict`` contributes to every ``dict`` subclass.
#:
#: The same trade as :data:`INHERITED_STR_METHODS`, one container down.  A
#: ``dict`` subclass keeps ``m[k]``, ``k in m``, ``len(m)``, iteration and
#: ``isinstance(m, dict)`` — all dunders, none of them listed — while donating
#: ``keys`` / ``values`` / ``items`` / ``get`` / ``copy`` and six mutators to a
#: listing that has one verb of its own.
INHERITED_DICT_METHODS: frozenset[str] = frozenset(n for n in dir(dict) if not n.startswith("_"))


def listing_dir(names: Sequence[str]) -> Callable[[], list[str]]:
    """Build the module ``__dir__`` that mirrors a module's ``__all__``.

    The one-line form of the enforcement clause::

        __all__ = ["plot", "draw"]
        __dir__ = listing_dir(__all__)

    The returned callable reads ``names`` **live**, so a module that appends to
    its ``__all__`` after this line (a registration side effect, a lazily
    discovered plugin) still lists the result.

    Parameters
    ----------
    names : sequence of str
        The module's ``__all__``.

    Returns
    -------
    callable
        A zero-argument function returning ``sorted(names)`` — the shape Python
        expects of a module-level ``__dir__``.
    """

    def __dir__() -> list[str]:  # noqa: N807 - it IS the dunder, by name and by role
        """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
        return sorted(names)

    return __dir__


def dir_without(obj: object, hidden: Iterable[str]) -> list[str]:
    """Return the default ``dir(obj)`` minus ``hidden``.

    For a record whose fields split into *what it declares* (a reader's
    question) and *what the library drives it with* (nobody's question).  The
    hidden set is written at the class as a named frozenset, so it is one
    reviewable literal rather than a policy spread over several files.

    Parameters
    ----------
    obj : object
        The instance being listed.
    hidden : iterable of str
        Attribute names to drop from the listing.  A name that is not there
        anyway is silently ignored, so the set can outlive a field.

    Returns
    -------
    list of str
        Sorted, and still containing every dunder — introspection tooling reads
        those, and a listing with no ``__class__`` looks broken rather than tidy.
    """
    drop = frozenset(hidden)
    return sorted(name for name in object.__dir__(obj) if name not in drop)


__dir__ = listing_dir(__all__)
