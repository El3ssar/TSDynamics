"""
Runtime registry of dynamical-system classes.

Every concrete subclass of :class:`~tsdynamics.families.base.SystemBase` is
registered automatically at class-definition time (via
``SystemBase.__init_subclass__``) — built-in systems and user-defined ones
alike.  Built-in systems (those defined under ``tsdynamics.systems``) are
what the bulk test-suite and the documentation generator iterate over;
user-defined classes are registered too but excluded from iteration by
default.

Alongside the system registry, this module hosts three *reserved* generic name
registries — :data:`analyses`, :data:`transforms` and :data:`renderers` —
:class:`Registry` containers (name → object + metadata) for the analysis,
transform and visualization-backend streams to register into.  :data:`renderers`
ships **empty** (visualization is deferred, decision D6); it is the seam a future
multi-backend plotting suite registers into.  Solvers are **not** registered
here: they live in the richer
:mod:`tsdynamics.solvers` registry (a ``name → SolverSpec`` table carrying
capability flags, populated by that package's directory scan + the
:mod:`~tsdynamics.plugins` entry-point loader).

This module must stay import-light: it is imported while the ``tsdynamics``
package itself is still initialising, so it may only depend on the standard
library.

Examples
--------
>>> from tsdynamics import registry
>>> registry.families()
{'ode': 120, 'dde': 5, 'map': 26}
>>> lorenz = registry.get("Lorenz")
>>> lorenz.family, lorenz.category
('ode', 'chaotic_attractors')
>>> [e.name for e in registry.all_systems(family="dde")]
['MackeyGlass', 'IkedaDelay', 'SprottDelay', 'ScrollDelay', 'PiecewiseCircuit']
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "Registry",
    "RegistryEntry",
    "SystemEntry",
    "all_systems",
    "analyses",
    "by_family",
    "categories",
    "families",
    "get",
    "renderers",
    "transforms",
]

Family = Literal["ode", "dde", "map", "sde", "other"]

#: Maps the *name* of the family base class to the family tag.  Matched
#: against the MRO, nearest ancestor first, so a ``DelaySystem`` subclass is
#: tagged ``"dde"`` even if ``DelaySystem`` itself derives from
#: ``ContinuousSystem``.
_FAMILY_BASES: dict[str, Family] = {
    "DiscreteMap": "map",
    "DelaySystem": "dde",
    "StochasticSystem": "sde",
    "ContinuousSystem": "ode",
}

_BUILTIN_PREFIX = "tsdynamics.systems"
_BASE_PREFIX = "tsdynamics.families"


@dataclass(frozen=True, eq=False)
class SystemEntry:
    """One registered system class plus the metadata the tooling needs."""

    name: str
    cls: type
    family: Family
    category: str
    module: str
    dim: int | None
    params: Mapping[str, Any]
    is_builtin: bool
    reference: str | None = None
    known_lyapunov: Mapping[str, Any] | None = None

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SystemEntry({self.name!r}, family={self.family!r}, "
            f"category={self.category!r}, dim={self.dim})"
        )


#: name → list of entries (a user class may shadow a builtin name).
_BY_NAME: dict[str, list[SystemEntry]] = {}


def _family_of(cls: type) -> Family:
    for ancestor in cls.__mro__:
        tag = _FAMILY_BASES.get(ancestor.__name__)
        if tag is not None and ancestor.__module__.startswith(_BASE_PREFIX):
            return tag
    return "other"


#: The method names that mark a *concrete* system: an ODE/DDE ``_equations``,
#: a map ``_step``, or an SDE ``_drift`` (the diagonal-Itô drift, mirroring how
#: ``_equations`` marks the deterministic families).  A class that defines one
#: of these outside the framework bases is registrable; an abstract intermediate
#: base that only inherits them is not.
_RHS_METHODS: frozenset[str] = frozenset({"_equations", "_step", "_drift"})


def _has_concrete_rhs(cls: type) -> bool:
    """Check that a concrete right-hand side is defined outside the framework bases."""
    for ancestor in cls.__mro__:
        if ancestor.__module__.startswith(_BASE_PREFIX):
            return False  # reached the framework bases without finding a rhs
        if any(name in ancestor.__dict__ for name in _RHS_METHODS):
            return True
    return False


def register_class(cls: type) -> None:
    """
    Register a system class.  Called from ``SystemBase.__init_subclass__``.

    Classes without a concrete ``_equations``/``_step`` (abstract intermediate
    bases) are ignored.  Two *built-in* classes sharing a name is a bug and
    raises immediately; user classes may freely shadow builtin names.
    """
    if not _has_concrete_rhs(cls):
        return

    is_builtin = cls.__module__.startswith(_BUILTIN_PREFIX)
    entry = SystemEntry(
        name=cls.__name__,
        cls=cls,
        family=_family_of(cls),
        category=cls.__module__.rsplit(".", 1)[-1],
        module=cls.__module__,
        dim=getattr(cls, "dim", None),
        params=MappingProxyType(dict(getattr(cls, "params", {}))),
        is_builtin=is_builtin,
        reference=getattr(cls, "reference", None),
        known_lyapunov=getattr(cls, "known_lyapunov", None),
    )

    bucket = _BY_NAME.setdefault(entry.name, [])
    # Re-importing/reloading the SAME builtin module is fine (the entry below
    # replaces the stale one); only a different module is a duplicate bug.
    if is_builtin and any(e.is_builtin and e.module != entry.module for e in bucket):
        other = next(e for e in bucket if e.is_builtin and e.module != entry.module)
        raise TypeError(
            f"Duplicate built-in system name {entry.name!r}: "
            f"already defined in {other.module}, redefined in {entry.module}."
        )
    # Re-definition of the *same* user class (e.g. re-running a notebook cell
    # or pytest re-importing a module) replaces the stale entry.
    bucket[:] = [e for e in bucket if e.module != entry.module] + [entry]


def all_systems(
    *,
    family: str | None = None,
    category: str | None = None,
    builtin: bool | None = True,
) -> list[SystemEntry]:
    """
    Return registered systems, in registration (= import) order.

    Parameters
    ----------
    family : {"ode", "dde", "map", "sde", "other"}, optional
        Keep only one family.
    category : str, optional
        Keep only one category (module stem, e.g. ``"chaotic_attractors"``).
    builtin : bool or None, default True
        ``True`` → only built-in systems (the default for tests/docs);
        ``False`` → only user-defined classes; ``None`` → everything.
    """
    out: list[SystemEntry] = []
    for bucket in _BY_NAME.values():
        for e in bucket:
            if builtin is not None and e.is_builtin is not builtin:
                continue
            if family is not None and e.family != family:
                continue
            if category is not None and e.category != category:
                continue
            out.append(e)
    return out


def by_family(family: str, **kwargs: Any) -> list[SystemEntry]:
    """Shorthand for :func:`all_systems` filtered to one family."""
    return all_systems(family=family, **kwargs)


def get(name: str, *, builtin: bool = True) -> SystemEntry:
    """
    Look up a single system by class name.

    Prefers the built-in entry when ``builtin`` is True (the default);
    raises ``KeyError`` with name suggestions otherwise.
    """
    bucket = _BY_NAME.get(name, [])
    for e in bucket:
        if e.is_builtin == builtin:
            return e
    if bucket:  # exists, but with the other builtin-ness
        return bucket[-1]
    close = [n for n in _BY_NAME if n.lower() == name.lower()]
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise KeyError(f"No registered system named {name!r}.{hint}")


def families(*, builtin: bool | None = True) -> dict[str, int]:
    """Return ``{family: count}`` for the registered systems."""
    counts: dict[str, int] = {}
    for e in all_systems(builtin=builtin):
        counts[e.family] = counts.get(e.family, 0) + 1
    return counts


def categories(family: str | None = None, *, builtin: bool | None = True) -> dict[str, int]:
    """Return ``{category: count}``, optionally restricted to one family."""
    counts: dict[str, int] = {}
    for e in all_systems(family=family, builtin=builtin):
        counts[e.category] = counts.get(e.category, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Generic name registries: analyses / transforms / renderers
#
# The system registry above is deliberately specialised (family detection,
# builtin shadowing, ``__init_subclass__`` hooks).  The analysis, transform and
# renderer kinds need only a name → object map with metadata, so they share one
# small generic container.  Discovery (directory scans, entry-point plugins)
# lives elsewhere and merely calls ``register``.
#
# Solvers do *not* use this generic container: they have their own richer
# ``name → SolverSpec`` registry in :mod:`tsdynamics.solvers` (capability flags,
# kernel names, directory + entry-point discovery).  Keep solver registration
# there — do not re-add a ``solvers`` registry here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    """One named, registered object plus its metadata."""

    name: str
    obj: Any
    metadata: Mapping[str, Any]

    def __repr__(self) -> str:  # noqa: D105
        return f"RegistryEntry({self.name!r})"


class Registry:
    """
    A minimal, generic name → object registry.

    Backs the :data:`analyses` and :data:`transforms` registries.  It only
    stores and looks up; the *discovery* that fills it (directory scans,
    :mod:`~tsdynamics.plugins` entry points) is built on top of it elsewhere.

    Registration is usable directly or as a decorator::

        analyses.register("lyapunov", lyapunov_spectrum)        # direct

        @analyses.register("corr_dim", needs="trajectory")      # decorator
        def correlation_dimension(traj): ...

    Re-registering the *same* object under a name is idempotent (safe across
    module re-imports).  A clash between two *different* objects on one name
    raises unless ``replace=True``.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, RegistryEntry] = {}

    @property
    def kind(self) -> str:
        """What this registry holds (``"solver"`` / ``"analysis"`` / ``"transform"``)."""
        return self._kind

    def _insert(self, name: str, obj: Any, *, replace: bool, metadata: Mapping[str, Any]) -> None:
        existing = self._entries.get(name)
        if existing is not None and existing.obj is not obj and not replace:
            raise ValueError(
                f"{self._kind} {name!r} is already registered to a different object; "
                f"pass replace=True to override."
            )
        self._entries[name] = RegistryEntry(
            name=name, obj=obj, metadata=MappingProxyType(dict(metadata))
        )

    def register(
        self,
        name: str,
        obj: Any = None,
        *,
        replace: bool = False,
        **metadata: Any,
    ) -> Any:
        """
        Register ``obj`` under ``name``.

        Called with ``obj`` it registers and returns it; called without it
        returns a decorator (so it can wrap a class/function definition).
        Extra keyword arguments are stored as the entry's ``metadata``.
        """
        if obj is None:

            def _decorator(target: Any) -> Any:
                self._insert(name, target, replace=replace, metadata=metadata)
                return target

            return _decorator
        self._insert(name, obj, replace=replace, metadata=metadata)
        return obj

    def get(self, name: str) -> Any:
        """Return the object registered under ``name`` (raises ``KeyError`` with hints)."""
        entry = self._entries.get(name)
        if entry is not None:
            return entry.obj
        close = [n for n in self._entries if n.lower() == name.lower()]
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise KeyError(f"No {self._kind} registered as {name!r}.{hint}")

    def entry(self, name: str) -> RegistryEntry:
        """Return the full :class:`RegistryEntry` (object + metadata) for ``name``."""
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"No {self._kind} registered as {name!r}.") from None

    def names(self) -> list[str]:
        """Return registered names, in registration order."""
        return list(self._entries)

    def all(self) -> list[RegistryEntry]:
        """All entries, in registration order."""
        return list(self._entries.values())

    def unregister(self, name: str) -> None:
        """Remove ``name`` (raises ``KeyError`` if absent)."""
        del self._entries[name]

    def clear(self) -> None:
        """Drop every entry (mainly for tests)."""
        self._entries.clear()

    def __contains__(self, name: object) -> bool:  # noqa: D105
        return name in self._entries

    def __iter__(self) -> Iterator[RegistryEntry]:  # noqa: D105
        return iter(self._entries.values())

    def __len__(self) -> int:  # noqa: D105
        return len(self._entries)

    def __repr__(self) -> str:  # noqa: D105
        return f"Registry(kind={self._kind!r}, {len(self._entries)} registered)"


#: Registered analysis functions (Lyapunov, dimensions, recurrence, …).
analyses = Registry("analysis")
#: Registered data/signal transforms (spectral, filters, feature extractors).
transforms = Registry("transform")
#: Registered visualization renderers (backend name → a callable that consumes a
#: :class:`~tsdynamics.viz.spec.PlotSpec` and draws it).  Ships **empty**:
#: visualization is deferred (decision D6), so no backend is registered yet — the
#: seam exists so a future matplotlib / Plotly / web renderer plugs in by
#: construction.  :meth:`~tsdynamics.viz.spec.PlotSpec.render` resolves a backend
#: through this registry; out-of-tree backends are discovered via the
#: ``tsdynamics.renderers`` entry-point group (see :mod:`tsdynamics.viz`).
renderers = Registry("renderer")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete.

    Re-exported stdlib helpers (``dataclass``, ``Mapping``, …) and the internal
    ``register_class`` hook stay importable but off the tab-completion surface.
    """
    return sorted(__all__)
