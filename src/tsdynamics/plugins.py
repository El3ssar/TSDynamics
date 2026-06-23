"""Entry-point plugin discovery — the external-plugin half of D4 (ROADMAP §4).

Third-party packages extend TSDynamics *without forking* by declaring Python
packaging **entry points**.  On import, TSDynamics walks the relevant groups and
loads whatever it finds, so an installed plugin's systems, solvers, analyses or
transforms register themselves automatically.

The five group names below are the frozen contract a plugin author declares
against in their own ``pyproject.toml``::

    [project.entry-points."tsdynamics.solvers"]
    my_method = "my_pkg.solvers:MY_SPEC"

This module is deliberately *generic*: it discovers and loads entry points and
imports submodules, but knows nothing about what a system/solver/analysis/
transform *is*.  Each consuming subpackage interprets the loaded objects in its
own terms (e.g. :mod:`tsdynamics.solvers` turns them into solver specs).

Stream **F2** owns this mechanism; it is consumed by ``tsdynamics.solvers``
(also F2) and, later, by the analyses/transforms registries.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from collections.abc import Iterator
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import Any, Protocol

# ── Entry-point group names (the frozen plugin contract) ───────────────────────
SYSTEMS_GROUP = "tsdynamics.systems"
SOLVERS_GROUP = "tsdynamics.solvers"
ANALYSES_GROUP = "tsdynamics.analyses"
TRANSFORMS_GROUP = "tsdynamics.transforms"
RENDERERS_GROUP = "tsdynamics.renderers"

#: Every plugin group TSDynamics recognises.
ALL_GROUPS: tuple[str, ...] = (
    SYSTEMS_GROUP,
    SOLVERS_GROUP,
    ANALYSES_GROUP,
    TRANSFORMS_GROUP,
    RENDERERS_GROUP,
)


def iter_entry_points(group: str) -> Iterator[EntryPoint]:
    """Yield the installed entry points declared under *group*.

    Thin wrapper over :func:`importlib.metadata.entry_points` so callers (and
    tests) have one place to go through.

    Parameters
    ----------
    group : str
        The entry-point group name, e.g. :data:`SOLVERS_GROUP`.

    Yields
    ------
    importlib.metadata.EntryPoint
        Each entry point declared by an installed distribution under *group*.
    """
    yield from entry_points(group=group)


def load_plugins(group: str, *, strict: bool = False) -> dict[str, Any]:
    """Load every entry point in *group* into a ``name -> object`` mapping.

    Each entry point is resolved with :meth:`~importlib.metadata.EntryPoint.load`
    (importing its target module and reading the named attribute).  By default a
    plugin that fails to load is **skipped with a warning**, so one broken
    third-party package can never break ``import tsdynamics``.

    Parameters
    ----------
    group : str
        The entry-point group to load (one of :data:`ALL_GROUPS`).
    strict : bool, default False
        If True, re-raise the first load failure instead of warning and
        continuing.  Useful in tests.

    Returns
    -------
    dict[str, Any]
        Maps each entry point's ``name`` to the object it resolves to.  Later
        duplicate names within the group overwrite earlier ones.
    """
    loaded: dict[str, Any] = {}
    for ep in iter_entry_points(group):
        try:
            loaded[ep.name] = ep.load()
        except Exception as exc:  # noqa: BLE001 — isolate third-party failures
            if strict:
                raise
            warnings.warn(
                f"failed to load plugin {ep.name!r} from group {group!r}: {exc}",
                stacklevel=2,
            )
    return loaded


def import_submodules(package: ModuleType) -> dict[str, ModuleType]:
    """Import every public submodule of *package*, returning ``name -> module``.

    This is the "directory scan at import" primitive (ROADMAP §4d): importing a
    submodule runs its top-level code, so a module that registers something on
    import (a solver spec, an analysis, …) becomes active simply by existing in
    the package.  Submodules whose names start with ``_`` are skipped, leaving
    room for private helpers that should *not* auto-run.

    Parameters
    ----------
    package : module
        An imported package (must expose ``__path__`` and ``__name__``).

    Returns
    -------
    dict[str, ModuleType]
        Maps each imported submodule's short name to the module object.
    """
    imported: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        full_name = f"{package.__name__}.{info.name}"
        imported[info.name] = importlib.import_module(full_name)
    return imported


class _RegistryLike(Protocol):
    """The minimal surface :func:`register_entry_points` needs of a registry.

    Both :class:`tsdynamics.registry.Registry` instances satisfy this; spelling
    it as a Protocol keeps :mod:`plugins` decoupled from the registry module
    (which imports nothing from here, so a hard import would risk a cycle).
    """

    def __contains__(self, name: object) -> bool: ...  # noqa: D105

    def register(self, name: str, obj: Any) -> Any: ...  # noqa: D105


def register_entry_points(
    registry: _RegistryLike, group: str, *, strict: bool = False
) -> list[str]:
    """Load the plugins in *group* and register each into *registry* by name.

    The generic-registry counterpart of
    :func:`tsdynamics.solvers.discover_plugins`: it wires the
    ``tsdynamics.analyses`` and ``tsdynamics.transforms`` plugin kinds into their
    :class:`~tsdynamics.registry.Registry` consumers (the two of the four D4
    plugin kinds that otherwise have no consumer).

    Each entry point resolves to the object to register **verbatim** under the
    entry point's own name — an analysis function, a transform callable, … —
    unlike :mod:`~tsdynamics.solvers`, whose plugins resolve to ``SolverSpec``
    metadata.  Names already present are left untouched, so this is safe to call
    repeatedly (e.g. after installing a new plugin).  Plugin load failures are
    isolated by :func:`load_plugins` (warn-and-skip unless *strict*).

    Parameters
    ----------
    registry : Registry
        The generic registry to populate (``registry.analyses`` /
        ``registry.transforms``).
    group : str
        The entry-point group to load (:data:`ANALYSES_GROUP` /
        :data:`TRANSFORMS_GROUP`).
    strict : bool, default False
        Forwarded to :func:`load_plugins`: re-raise the first load failure
        instead of warning and continuing.

    Returns
    -------
    list[str]
        The names newly registered by this call, in load order.
    """
    newly: list[str] = []
    for name, obj in load_plugins(group, strict=strict).items():
        if name not in registry:
            registry.register(name, obj)
            newly.append(name)
    return newly


__all__ = [
    # The frozen entry-point group contract (what a plugin author declares against).
    "SYSTEMS_GROUP",
    "SOLVERS_GROUP",
    "ANALYSES_GROUP",
    "TRANSFORMS_GROUP",
    "RENDERERS_GROUP",
    "ALL_GROUPS",
    # The generic discovery / loading primitives the consuming subpackages use.
    "iter_entry_points",
    "load_plugins",
    "import_submodules",
    "register_entry_points",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete.

    Hides the re-exported stdlib imports (``importlib``, ``warnings``, ``Any``, …)
    and the private ``_RegistryLike`` protocol from the tab-completion surface.
    """
    return sorted(__all__)
