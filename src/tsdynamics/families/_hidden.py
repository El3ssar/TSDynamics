"""One mechanism for taking a name off ``dir()`` **without unbinding it**.

The v6 visibility ruling (``CONTRACT.md`` §11) turns on a single sentence:

    Hiding is a DISCOVERY change, never a REACHABILITY change.

A name that a user never *types* still has internal callers, shipped doc pages,
and tests that assert on it — and, more importantly, a user who has already
found it is entitled to keep using it.  So hiding here means exactly one thing:
the name stops being offered by ``dir()`` (and therefore by ``<TAB>`` and by
``help(instance)``), and stays bound, importable and callable.  Measured
capability loss is zero, which is what makes the owner's two rulings — *make it
less overwhelming* and *do not hinder customizability* — satisfiable at once.

The counter-proposal, renaming to ``_name``, fails both halves: it breaks the
callers (``viz.transforms.spectra`` drives ``TangentSystem.convergence``;
``docs/analysis/lyapunov.md`` runs ``tang.growths()`` under the doctest gate),
and it converts a discoverability question into a capability cut.

Usage — one decorator at the class definition site, naming what drops out::

    @hide("plane_auto")
    class PoincareMap(DerivedSystem):
        ...

What it does **not** do, deliberately:

- It does not touch ``dir(TheClass)`` (only instances).  The class is where the
  reference documentation lives, and ``help(PoincareMap)`` must keep describing
  every member it really has — pydoc walks the MRO ``__dict__``, not ``dir()``.
- It does not make ``hasattr`` lie.  That is :class:`~tsdynamics.families.base.Absent`'s
  job, and it is reserved for a name that genuinely *cannot work* on a family.
  A hidden name works; it is merely not what you came for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["HIDDEN_NAMES", "hide"]

_C = TypeVar("_C", bound=type)

#: ``"module.QualName" -> frozenset of names hidden from ``dir(instance)``.
#: Populated at import by every :func:`hide` call, and read by the surface gate
#: (``tests/test_families_v6_surface.py``) so a hidden name cannot quietly become
#: an unbound one: the gate asserts each entry still resolves.
HIDDEN_NAMES: dict[str, frozenset[str]] = {}


def _inherited_dir(owner: type, obj: Any) -> list[str]:
    """Return the listing ``obj`` would have had, had *owner* not been decorated.

    The explicit spelling of ``super(owner, obj).__dir__()``, which a dynamic
    ``owner`` makes untypable.  Delegating to the MRO — rather than jumping
    straight to ``object.__dir__`` — is what lets a decorated class sit *under*
    one that already curates its own listing: the only such base today is
    :class:`~tsdynamics.families.base.SystemBase`, whose ``__dir__`` drops the
    ``Absent`` slots, and short-circuiting it would silently re-advertise names
    that cannot work on that family.
    """
    mro = type(obj).__mro__
    for base in mro[mro.index(owner) + 1 :]:
        inherited = base.__dict__.get("__dir__")
        if inherited is not None:
            return list(inherited(obj))
    return list(object.__dir__(obj))


def hide(*names: str) -> Callable[[_C], _C]:
    """Drop *names* from ``dir(instance)`` for the decorated class.

    Parameters
    ----------
    *names : str
        Attribute names to withhold from ``dir()``.  They remain bound: reading,
        calling and ``hasattr`` are all unchanged.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    >>> @hide("scratch")
    ... class Counter:
    ...     scratch = 0
    ...     def tick(self):
    ...         return 1
    >>> "scratch" in dir(Counter())          # off the tab surface...
    False
    >>> Counter().scratch                    # ...and still perfectly readable
    0
    """
    asked = frozenset(names)

    def decorate(cls: _C) -> _C:
        hidden: frozenset[str] = getattr(cls, "_HIDDEN_FROM_DIR", frozenset()) | asked
        setattr(cls, "_HIDDEN_FROM_DIR", hidden)  # noqa: B010 - the point is dynamism
        HIDDEN_NAMES[f"{cls.__module__}.{cls.__qualname__}"] = hidden
        owner: type = cls

        def __dir__(self: Any) -> list[str]:  # noqa: N807 - it IS the dunder being bound
            """List what this object offers — internals withheld, never unbound."""
            drop: frozenset[str] = getattr(type(self), "_HIDDEN_FROM_DIR", frozenset())
            return sorted(n for n in _inherited_dir(owner, self) if n not in drop)

        setattr(cls, "__dir__", __dir__)  # noqa: B010 - ditto
        return cls

    return decorate


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
