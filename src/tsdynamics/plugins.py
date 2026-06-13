"""Entry-point plugin discovery — the external-plugin half of D4 (ROADMAP §4).

Third-party packages extend TSDynamics *without forking* by declaring Python
packaging **entry points**.  On import, TSDynamics walks the relevant groups and
loads whatever it finds, so an installed plugin's systems, solvers, analyses or
transforms register themselves automatically.

The four group names below are the frozen contract a plugin author declares
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
from typing import Any

# ── Entry-point group names (the frozen plugin contract) ───────────────────────
SYSTEMS_GROUP = "tsdynamics.systems"
SOLVERS_GROUP = "tsdynamics.solvers"
ANALYSES_GROUP = "tsdynamics.analyses"
TRANSFORMS_GROUP = "tsdynamics.transforms"

#: Every plugin group TSDynamics recognises.
ALL_GROUPS: tuple[str, ...] = (
    SYSTEMS_GROUP,
    SOLVERS_GROUP,
    ANALYSES_GROUP,
    TRANSFORMS_GROUP,
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
