"""The Python solver registry — the discovery half of the pluggable engine (D4).

This package is the Python-side mirror of the Rust link-time solver registry in
the ``tsdyn-solvers`` crate.  It holds a ``name -> SolverSpec`` table of solver
*metadata* (capabilities + the name of the Rust kernel that does the work) and
auto-populates it from two sources at import time:

1. **a directory scan** of this package's own modules — dropping a module that
   calls :func:`register` makes its solver available with no central table to
   edit (ROADMAP §4d); and
2. **out-of-tree plugins** declared under the ``tsdynamics.solvers`` entry-point
   group (see :mod:`tsdynamics.plugins`).

Stream **F2** owns this *mechanism* only.  It ships no built-in solvers — those
arrive with the Rust kernels (streams E3/E4/E-SDE) and the selection layer
(``method=`` resolution + auto-stiffness) is stream **C-SOLV**, which builds on
top of this registry.  A :class:`SolverSpec` therefore carries no Python
implementation; its :attr:`SolverSpec.kernel` names the Rust kernel to dispatch
to once the engine bindings (stream E7) land.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..plugins import SOLVERS_GROUP, import_submodules, load_plugins

# The problem families a solver may declare support for — the Python spelling of
# the Rust `ProblemKind` set (`tsdyn_solvers::ProblemKind`).
_VALID_FAMILIES = frozenset({"ode", "dde", "sde", "map"})
_VALID_KINDS = frozenset({"explicit", "implicit"})


@dataclass(frozen=True)
class SolverCaps:
    """A solver's capabilities — the Python mirror of ``tsdyn_solvers::Caps``.

    Drives ``method=`` resolution and the auto-stiffness layer (stream C-SOLV).

    Parameters
    ----------
    kind : {"explicit", "implicit"}
        Whether the kernel is explicit or implicit (the primary auto-stiffness
        axis).
    adaptive : bool, default False
        Whether the kernel does its own embedded error estimate + step adaption.
    needs_jacobian : bool, default False
        Whether the kernel requires the analytic Jacobian.
    supports : frozenset[str], default {"ode"}
        The problem families the kernel can integrate; a subset of
        ``{"ode", "dde", "sde", "map"}``.
    """

    kind: str
    adaptive: bool = False
    needs_jacobian: bool = False
    supports: frozenset[str] = field(default_factory=lambda: frozenset({"ode"}))

    def __post_init__(self) -> None:
        """Validate the enum-like fields and normalise ``supports``."""
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}, got {self.kind!r}")
        supports = frozenset(self.supports)
        unknown = supports - _VALID_FAMILIES
        if unknown:
            raise ValueError(
                f"unknown problem families {sorted(unknown)}; "
                f"valid families are {sorted(_VALID_FAMILIES)}"
            )
        # Normalise to a frozenset even if a list/set was passed (frozen → setattr).
        object.__setattr__(self, "supports", supports)

    def supports_family(self, family: str) -> bool:
        """Return whether this kernel can integrate problems of *family*."""
        return family in self.supports


@dataclass(frozen=True)
class SolverSpec:
    """Registry metadata for one solver — the Python mirror of a Rust kernel.

    Parameters
    ----------
    name : str
        The unique ``method=`` name users select the solver by.
    caps : SolverCaps
        The solver's capabilities.
    kernel : str, optional
        The name of the Rust kernel (``tsdyn_solvers`` registry) that performs
        the integration.  Defaults to :attr:`name` when omitted.
    description : str, default ""
        A one-line human description (shown in listings / docs).
    origin : {"builtin", "plugin"}, default "builtin"
        Where the spec came from — set to ``"plugin"`` for entry-point plugins.
    """

    name: str
    caps: SolverCaps
    kernel: str = ""
    description: str = ""
    origin: str = "builtin"

    def __post_init__(self) -> None:
        """Validate the name and default ``kernel`` to ``name``."""
        if not self.name:
            raise ValueError("SolverSpec.name must be a non-empty string")
        if not self.kernel:
            object.__setattr__(self, "kernel", self.name)


# ── The registry ──────────────────────────────────────────────────────────────
_REGISTRY: dict[str, SolverSpec] = {}


def register(spec: SolverSpec, *, override: bool = False) -> None:
    """Register *spec* under its :attr:`~SolverSpec.name`.

    Parameters
    ----------
    spec : SolverSpec
        The solver metadata to register.
    override : bool, default False
        If False (the default), registering a name that already exists raises
        :class:`ValueError` — this catches accidental in-tree clashes the way the
        system registry does for duplicate class names.  Pass True to replace.

    Raises
    ------
    ValueError
        If *spec*'s name is already registered and ``override`` is False.
    """
    if spec.name in _REGISTRY and not override:
        raise ValueError(
            f"a solver named {spec.name!r} is already registered (pass override=True to replace it)"
        )
    _REGISTRY[spec.name] = spec


def get(name: str) -> SolverSpec:
    """Return the :class:`SolverSpec` registered under *name*.

    Parameters
    ----------
    name : str
        The solver's registry name.

    Returns
    -------
    SolverSpec
        The registered spec.

    Raises
    ------
    KeyError
        If no solver is registered under *name*; the message lists the available
        names.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown solver {name!r}; available solvers: {available()}") from None


def available() -> list[str]:
    """Return every registered solver name, sorted."""
    return sorted(_REGISTRY)


def all_specs() -> dict[str, SolverSpec]:
    """Return a copy of the full ``name -> SolverSpec`` registry."""
    return dict(_REGISTRY)


def unregister(name: str) -> bool:
    """Remove the solver registered under *name*; return whether one was removed.

    Mostly for tests and dynamic reconfiguration; normal use is append-only.
    """
    return _REGISTRY.pop(name, None) is not None


def _coerce_to_specs(name: str, obj: Any) -> list[SolverSpec]:
    """Interpret a loaded plugin object as zero or more :class:`SolverSpec`.

    A ``tsdynamics.solvers`` entry point may resolve to a :class:`SolverSpec`, an
    iterable of them, or a zero-argument callable returning either (a callable
    that registers solvers itself returns ``None`` and contributes nothing here).
    Anything else is ignored with a warning.
    """
    if isinstance(obj, SolverSpec):
        return [obj]
    if isinstance(obj, str):  # str is Iterable; reject before the Iterable branch
        warnings.warn(
            f"solver plugin {name!r} resolved to a string, not a SolverSpec; ignoring",
            stacklevel=2,
        )
        return []
    if callable(obj):
        return _coerce_to_specs(name, obj())
    if obj is None:
        return []
    if isinstance(obj, Iterable):
        specs: list[SolverSpec] = []
        for item in obj:
            specs.extend(_coerce_to_specs(name, item))
        return specs
    warnings.warn(
        f"solver plugin {name!r} resolved to {type(obj).__name__}, not a SolverSpec; ignoring",
        stacklevel=2,
    )
    return []


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree solver plugins and register any they contribute.

    Walks the ``tsdynamics.solvers`` entry-point group (see
    :mod:`tsdynamics.plugins`), coerces each loaded object to specs, and
    registers those not already present (so it is safe to call repeatedly —
    e.g. after installing a plugin).

    Parameters
    ----------
    strict : bool, default False
        Forwarded to :func:`tsdynamics.plugins.load_plugins`: if True, a plugin
        that fails to import raises instead of being skipped with a warning.

    Returns
    -------
    list[str]
        The names newly registered by this call.
    """
    newly_registered: list[str] = []
    for ep_name, obj in load_plugins(SOLVERS_GROUP, strict=strict).items():
        for spec in _coerce_to_specs(ep_name, obj):
            if spec.name not in _REGISTRY:
                register(spec)
                newly_registered.append(spec.name)
    return newly_registered


def _scan() -> None:
    """Import this package's own solver modules so they self-register.

    A no-op until streams E3/E4/E-SDE/C-SOLV add kernel-spec modules; failures
    here are real in-tree bugs and propagate (unlike plugin failures).
    """
    import_submodules(sys.modules[__name__])


# Populate the registry at import: in-tree modules first, then out-of-tree
# plugins. Plugin failures are isolated inside `discover_plugins`.
_scan()
discover_plugins()

__all__ = [
    "SolverCaps",
    "SolverSpec",
    "all_specs",
    "available",
    "discover_plugins",
    "get",
    "register",
    "unregister",
]

# ── Selection layer (stream C-SOLV) ────────────────────────────────────────────
# Built on top of the F2 registry above: ``method=`` resolution + aliases, the
# auto-stiffness detector/policy, and the ``with_jacobian`` auto-set hook the
# family engine-dispatch seam (stream C-FAM) consumes.  Imported after the
# registry is populated (``_scan()`` already imported ``select`` as a submodule;
# this binds its public surface onto the package).
from .select import (  # noqa: E402
    DEFAULT_METHOD,
    STIFF_METHOD,
    Resolution,
    available_for,
    build_kwargs,
    default_method,
    is_implicit,
    is_stiff,
    needs_jacobian,
    normalize,
    recommend,
    resolve,
    select,
)

__all__ += [
    "DEFAULT_METHOD",
    "STIFF_METHOD",
    "Resolution",
    "available_for",
    "build_kwargs",
    "default_method",
    "is_implicit",
    "is_stiff",
    "needs_jacobian",
    "normalize",
    "recommend",
    "resolve",
    "select",
]
