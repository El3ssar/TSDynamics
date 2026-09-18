"""Built-in system catalogue — and the registry verbs that search it.

Every built-in system class is re-exported flat here, so the canonical path to a
model is ``tsdynamics.systems.<Name>`` (e.g. ``tsdynamics.systems.Lorenz``) — no
need to remember whether it lives under ``continuous`` or ``discrete``.

**Since v6 this namespace answers the same three verbs every other registry in
the library answers** (CONTRACT §2.6): :func:`names`, :func:`find` and
:func:`get`.  Without them ``ts.systems`` was the one registry you could not
search — and it is the one with 177 members::

    ts.systems.names()                 # every catalogue class name
    ts.systems.find("chaotic map")     # free text over name / category / citation
    ts.systems.find(family="dde")      # or by family / dimension
    ts.systems.get("Lorenz")()         # the class, by name

The per-category submodules (``continuous`` / ``discrete``) remain importable for
finer navigation; they are off the listing, like every other internal submodule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from . import continuous, discrete

# Flat re-export of every catalogue class. Adding a system to a category module's
# ``__all__`` automatically surfaces it here (and at ``tsdynamics.<Name>`` via the
# top-level lazy accessor) — no manual edit needed.
for _name in continuous.__all__:
    globals()[_name] = getattr(continuous, _name)
for _name in discrete.__all__:
    globals()[_name] = getattr(discrete, _name)
del _name

_SYSTEM_NAMES: tuple[str, ...] = (*continuous.__all__, *discrete.__all__)

__all__ = ["find", "get", "names", *_SYSTEM_NAMES]


def names() -> list[str]:
    """Return every catalogue system's class name, sorted.

    Returns
    -------
    list of str

    Examples
    --------
    >>> import tsdynamics as ts
    >>> "Lorenz" in ts.systems.names()
    True
    """
    return sorted(_SYSTEM_NAMES)


def get(name: str) -> type:
    """Return the catalogue class called ``name``.

    Parameters
    ----------
    name : str
        The class name, e.g. ``"Lorenz"``.

    Returns
    -------
    type
        The system class — call it to build an instance.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If nothing is registered under that name; the message suggests the
        nearest catalogue names and points at :func:`find`.

    Examples
    --------
    >>> import tsdynamics as ts
    >>> ts.systems.get("Lorenz")().dim
    3
    """
    import difflib

    from tsdynamics.errors import InvalidParameterError

    cls = globals().get(name)
    if isinstance(cls, type):
        return cls
    close = difflib.get_close_matches(name, _SYSTEM_NAMES, n=3, cutoff=0.6)
    hint = "".join(f"\n    ts.systems.{c}()" for c in close)
    raise InvalidParameterError(
        f"no catalogue system named {name!r}."
        + (f"  Did you mean:{hint}" if close else "")
        + f'\n    ts.systems.find("{name}")   # search all {len(_SYSTEM_NAMES)}'
    )


class SystemList(list):  # type: ignore[type-arg]
    """The list :func:`find` returns — the system **classes**, tabulated in its repr.

    Holds the classes, so ``find("lorenz")[0]()`` builds one and
    ``[c.__name__ for c in find(family="dde")]`` gives the names.  Received,
    never constructed.

    It exists because the same verb should answer the same way at every
    registry: ``ts.analysis.find`` prints a grouped, captioned table and this one
    printed ``[<class 'tsdynamics.systems.continuous.chaotic_attractors.Chua'>,
    ...]`` — the module path of a private file, three times the width of the
    answer, carrying none of what a reader is choosing between.
    """

    __slots__ = ("_header",)

    def __init__(self, classes: Sequence[type], header: str) -> None:
        super().__init__(classes)
        self._header = header

    def __dir__(self) -> list[str]:
        """Nothing — this **is** the answer, not a container to edit (§11.1).

        Measured before this: ``ts.systems.find("delay").<TAB>`` offered
        ``append clear copy count extend index insert pop remove reverse sort``
        — eleven ways to edit a search result and not one way to use it, on the
        verb v6 advertises as *the* way to search the one namespace with 177
        members.  Nobody types ``find("delay").sort()``.

        Hiding is a **discovery** change, never a reachability one: every one of
        those eleven stays bound and callable, the object is still a complete
        ``list``, and what you actually do with it — ``[0]``, ``len``, ``for``,
        ``in``, ``print`` — are builtins that never needed a name on the object.
        """
        return []

    def __repr__(self) -> str:  # noqa: D105
        from tsdynamics import registry

        if not self:
            return self._header
        entries = {e.cls: e for e in registry.all_systems()}
        lines = [self._header]
        for family in ("ode", "dde", "map", "sde"):
            rows = [c for c in self if getattr(entries.get(c), "family", None) == family]
            if not rows:
                continue
            lines.append("")
            lines.append(f"  {_FAMILY_WORD[family]}   ({len(rows)})")
            for cls in rows[:_MAX_LISTED]:
                entry = entries[cls]
                where = f"dim {entry.dim}" if entry.dim else "dim varies"
                cite = _short_reference(entry.reference)
                lines.append(f"    ts.systems.{cls.__name__:<22s} {where:<11s} {cite}")
            if len(rows) > _MAX_LISTED:
                lines.append(f"    ... and {len(rows) - _MAX_LISTED} more")
        unknown = [c for c in self if c not in entries]
        for cls in unknown[:_MAX_LISTED]:
            lines.append(f"    {cls.__name__}")
        return "\n".join(lines)


#: How each family word is spelled in the :class:`SystemList` table.
_FAMILY_WORD = {
    "ode": "flows (ODE)",
    "dde": "delay systems (DDE)",
    "map": "maps",
    "sde": "stochastic systems (SDE)",
}

#: Rows shown per family before the table truncates.
_MAX_LISTED = 12


#: ``Author, Author & Author (1984)`` — everything up to and including the year.
#: Splitting on the first comma instead renders "Yalçın, Suykens & Vandewalle
#: (2005), ..." as the single word "Yalçın", which is not a citation.
_CITATION_HEAD = re.compile(r"^(.*?\(\d{4}[a-z]?\))")


def _short_reference(reference: str | None) -> str:
    """Render a catalogue citation down to ``Author(s) (year)``."""
    if not reference:
        return ""
    text = " ".join(str(reference).split())
    match = _CITATION_HEAD.match(text)
    head = match.group(1) if match else text.split(",")[0].strip()
    return head if len(head) <= 44 else head[:41] + "..."


def _find_header(what: str, family: str | None, dim: int | None, n: int) -> str:
    """Render the one line above the table: how many, and what was asked for."""
    asked = []
    if what:
        asked.append(repr(str(what)))
    if family is not None:
        asked.append(f"family={family!r}")
    if dim is not None:
        asked.append(f"dim={dim}")
    query = " · ".join(asked)
    if not n:
        return (
            f"no catalogue system matches {query or 'that'}."
            f"\n    ts.systems.names()                 # all {len(__all__) - 3}"
            f'\n    ts.systems.find(family="map")      # by family'
        )
    plural = "" if n == 1 else "s"
    return f"{n} system{plural} match {query}" if query else f"{n} system{plural}"


def find(
    what: str = "",
    /,
    *,
    family: str | None = None,
    dim: int | None = None,
) -> list[type]:
    """Search the catalogue — free text, and/or by family and dimension.

    One positional argument, like :func:`tsdynamics.analysis.find`: the words you
    would use to describe what you are looking for.  They are matched against the
    class name, the category (the module stem, e.g. ``chaotic_attractors``), the
    family word, and the literature citation.

    Parameters
    ----------
    what : str, optional
        Free text; every whitespace-separated word must match something.  Empty
        (the default) matches everything, so ``find(family="dde")`` works.
    family : {"ode", "dde", "map", "sde"}, optional
        Keep only this family.
    dim : int, optional
        Keep only systems of this state-space dimension.

    Returns
    -------
    SystemList
        A plain ``list`` of the matching **classes**, sorted by name — call one
        to build it.  Its *repr* is the grouped catalogue table (family, state
        dimension, and the literature the model comes from), so the same verb
        answers as legibly here as it does on ``ts.analysis``.

    Examples
    --------
    >>> import tsdynamics as ts
    >>> len(ts.systems.find(family="dde"))
    6
    >>> ts.systems.get("Lorenz") in ts.systems.find("lorenz")
    True
    """
    from tsdynamics import registry

    words = [w for w in str(what).lower().split() if w]
    out: list[type] = []
    for entry in registry.all_systems():
        if family is not None and entry.family != family:
            continue
        if dim is not None and entry.dim != dim:
            continue
        haystack = " ".join(
            str(part).lower()
            for part in (entry.name, entry.category, entry.family, entry.reference or "")
        )
        if all(word in haystack for word in words):
            out.append(entry.cls)
    ordered = sorted(out, key=lambda c: c.__name__)
    return SystemList(ordered, _find_header(what, family, dim, len(ordered)))


def __dir__() -> list[str]:
    """Expose the catalogue surface (``__all__``) to ``dir()`` / autocomplete.

    That is the three registry verbs plus every flat-re-exported model class.
    The two category subpackages stay importable but off the listing (§2.6), the
    same treatment every other internal submodule gets.
    """
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    """Answer a wrong guess by naming the nearest systems and the search verb.

    Deliberately an ``AttributeError`` (not the typed
    :class:`~tsdynamics.errors.InvalidParameterError` :func:`get` raises): an
    attribute miss must keep ``hasattr(ts.systems, x)`` answering ``False``, and
    the import machinery falls back to ``sys.modules`` for a submodule only on
    ``AttributeError`` — which is what keeps ``import tsdynamics.systems.continuous``
    working now that the two category packages are off the listing.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    from tsdynamics.errors import InvalidParameterError

    try:
        return get(name)
    except InvalidParameterError as exc:
        raise AttributeError(str(exc)) from None
