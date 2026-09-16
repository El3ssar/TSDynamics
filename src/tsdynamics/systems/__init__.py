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
    list of type
        The matching **classes**, sorted by name — call one to build it.

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
    return sorted(out, key=lambda c: c.__name__)


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
