"""Backend dispatch for the visualization seam (stream VIZ-DISPATCH).

:meth:`tsdynamics.viz.spec.PlotSpec.render` delegates here.  This module turns a
backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into a figure (or export
payload) by choosing a registered renderer and **falling back** to the
matplotlib reference renderer when the chosen backend cannot draw a spec's kind:

- :func:`register_builtin_renderers` — lazily import and register the in-tree
  backends that are *installed* (matplotlib, plotly, json, three.js).  Called on
  first render, **never at import**, so ``import tsdynamics`` stays plot-free.
- :func:`select_renderer` — resolve a backend by name (or pick a default) and,
  via the :class:`~tsdynamics.viz.render.caps.RendererCapabilities`, fall back to
  a capable backend (emitting :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`)
  when the requested one declines the spec.
- :func:`render_spec` — the one entry point :meth:`PlotSpec.render` calls.
- :func:`normalize_kind` / :data:`_KIND_ALIAS` — canonicalise friendly / legacy
  kind spellings (the ``result.plot.phase()`` / ``.image()`` accessor strings) to
  real :class:`~tsdynamics.viz.spec.PlotKind` members, so a backend's
  kind-keyed preset table never trips over an alias.

An in-tree backend module (``tsdynamics.viz.render.<name>``) self-registers by
exposing a module-level ``register(registry)`` that adds its renderer callable
(carrying a ``.capabilities`` descriptor) to ``registry`` iff its plotting
library imports — so a backend whose dependency is absent simply does not
register and dispatch falls back.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from ..spec import PlotKind, PlotSpec
from .caps import Renderer, RendererCapabilities, RenderResult, VisualizationDegraded

__all__ = [
    "Renderer",
    "RendererCapabilities",
    "RenderResult",
    "VisualizationDegraded",
    "normalize_kind",
    "register_builtin_renderers",
    "render_spec",
    "select_renderer",
]

#: The in-tree backend submodules dispatch tries to register, in preference
#: order (matplotlib is the universal reference renderer / fallback).  Each is a
#: ``tsdynamics.viz.render.<name>`` module exposing ``register(registry)``; a
#: module that is absent (its stream has not landed) or whose plotting library is
#: not installed is skipped.
_BUILTIN_BACKENDS: tuple[str, ...] = ("mpl", "plotly", "json", "threejs")

#: Friendly / legacy kind spellings → canonical :class:`~tsdynamics.viz.spec.PlotKind`.
#: The ``result.plot`` accessor's short names (``"phase"``, ``"image"``, …) and a
#: few aliases route through here so a renderer's preset table sees only real
#: semantic kinds.
_KIND_ALIAS: dict[str, PlotKind] = {
    "phase": PlotKind.PHASE_PORTRAIT_2D,
    "phase2d": PlotKind.PHASE_PORTRAIT_2D,
    "phase_portrait": PlotKind.PHASE_PORTRAIT_2D,
    "phase3d": PlotKind.PHASE_PORTRAIT_3D,
    "phase_portrait_field": PlotKind.PHASE_PORTRAIT_FIELD,
    "image": PlotKind.IMAGE,
    "spectrum": PlotKind.POWER_SPECTRUM,
    "psd": PlotKind.POWER_SPECTRUM,
    "histogram": PlotKind.HISTOGRAM_NULL,
    "section": PlotKind.POINCARE_SECTION,
    "bifurcation_diagram": PlotKind.BIFURCATION,
    "diagnostic": PlotKind.DIAGNOSTIC_CURVE,
    "scaling": PlotKind.SCALING_FIT,
}


def normalize_kind(kind: PlotKind | str) -> PlotKind:
    """Resolve a kind spelling to a canonical :class:`~tsdynamics.viz.spec.PlotKind`.

    A :class:`~tsdynamics.viz.spec.PlotKind` passes through unchanged; a string is
    looked up in :data:`_KIND_ALIAS` (friendly / legacy spellings such as the
    ``result.plot.phase()`` / ``.image()`` accessor names) and otherwise coerced
    directly.  Raises :class:`ValueError` for a string that is neither an alias
    nor a real kind.
    """
    if isinstance(kind, PlotKind):
        return kind
    key = str(kind).lower()
    aliased = _KIND_ALIAS.get(key)
    if aliased is not None:
        return aliased
    return PlotKind(key)


def register_builtin_renderers(*, strict: bool = False) -> list[str]:
    """Register the installed in-tree rendering backends; return the new names.

    Imports each ``tsdynamics.viz.render.<name>`` backend module in
    :data:`_BUILTIN_BACKENDS` and calls its ``register(registry)`` hook.  A
    backend whose module does not exist yet, whose plotting library is not
    installed, or that otherwise fails to import is **skipped** (warn unless
    ``strict``), so dispatch degrades to whatever backends are present.

    Idempotent — a backend already in the registry is left untouched — and only
    ever called from :func:`render_spec` (never at import), so importing
    TSDynamics never pulls in a plotting library.
    """
    from tsdynamics import registry

    before = set(registry.renderers.names())
    for name in _BUILTIN_BACKENDS:
        try:
            module = importlib.import_module(f"tsdynamics.viz.render.{name}")
        except ImportError:
            if strict:
                raise
            continue
        register = getattr(module, "register", None)
        if callable(register):
            try:
                register(registry.renderers)
            except Exception as exc:  # noqa: BLE001 — isolate a backend's failure
                if strict:
                    raise
                warnings.warn(f"failed to register viz backend {name!r}: {exc}", stacklevel=2)
    return [n for n in registry.renderers.names() if n not in before]


def _capabilities_of(renderer: Any) -> RendererCapabilities | None:
    """Return a renderer's declared capabilities, or ``None`` if it carries none.

    A plain callable (no ``.capabilities``) is treated by the dispatch as a
    universal fallback that draws anything.
    """
    caps = getattr(renderer, "capabilities", None)
    return caps if isinstance(caps, RendererCapabilities) else None


def _can_render(renderer: Any, spec: PlotSpec) -> bool:
    """Whether ``renderer`` can draw ``spec`` (a capability-less one draws all)."""
    caps = _capabilities_of(renderer)
    return True if caps is None else caps.can_render_spec(spec)


def select_renderer(spec: PlotSpec, backend: str | None = None) -> tuple[str, Any]:
    """Choose the ``(name, renderer)`` to draw ``spec``, with capability fallback.

    Parameters
    ----------
    spec : PlotSpec
        The spec to render (its kind / 3-D-ness drive the capability check).
    backend : str, optional
        A requested backend name.  ``None`` picks the first registered backend
        that can draw the spec (the registration order in
        :data:`_BUILTIN_BACKENDS` puts matplotlib first, so it is the default
        when present).

    Returns
    -------
    (str, callable)
        The chosen backend name and its renderer callable.

    Raises
    ------
    VisualizationNotInstalled
        If no rendering backend is registered at all.
    KeyError
        If ``backend`` names a backend that is not registered.

    Warns
    -----
    VisualizationDegraded
        If the requested ``backend`` cannot draw the spec and a capable backend
        is used instead.
    """
    from tsdynamics import registry

    renderers = registry.renderers
    if len(renderers) == 0:
        raise _visualization_not_installed()

    if backend is not None:
        renderer = renderers.get(backend)  # KeyError (naming) if unknown
        if _can_render(renderer, spec):
            return backend, renderer
        fallback = _first_capable(spec, renderers, exclude=backend)
        if fallback is None:
            return backend, renderer  # nothing better; let the backend try / error clearly
        warnings.warn(
            f"backend {backend!r} cannot draw a {spec.kind.value!r} spec; "
            f"falling back to {fallback[0]!r}.",
            VisualizationDegraded,
            stacklevel=2,
        )
        return fallback

    chosen = _first_capable(spec, renderers)
    if chosen is not None:
        return chosen
    # No backend declares it can draw this; use the first registered and let it
    # try (a capability-less universal renderer would have matched above).
    name = renderers.names()[0]
    return name, renderers.get(name)


def _first_capable(
    spec: PlotSpec, renderers: Any, *, exclude: str | None = None
) -> tuple[str, Any] | None:
    """Return the first ``(name, renderer)`` that can draw ``spec``, else ``None``."""
    for name in renderers.names():
        if name == exclude:
            continue
        renderer = renderers.get(name)
        if _can_render(renderer, spec):
            return name, renderer
    return None


def render_spec(spec: PlotSpec, backend: str | None = None, **backend_kw: Any) -> Any:
    """Render ``spec`` through a registered backend (the :meth:`PlotSpec.render` seam).

    Registers the installed in-tree backends (lazily, once), selects one by name
    or by capability (falling back when the requested backend declines the spec),
    and calls it, forwarding ``backend_kw``.

    Parameters
    ----------
    spec : PlotSpec
        The spec to render.
    backend : str, optional
        Backend name; ``None`` selects the default capable backend.
    **backend_kw
        Forwarded to the chosen renderer callable.

    Returns
    -------
    Any
        Whatever the backend returns (a figure handle, a
        :class:`~tsdynamics.viz.render.caps.RenderResult`, or an export payload).

    Raises
    ------
    VisualizationNotInstalled
        If no rendering backend is registered.
    KeyError
        If ``backend`` names an unregistered backend.
    """
    register_builtin_renderers()
    _name, renderer = select_renderer(spec, backend)
    return renderer(spec, **backend_kw)


def _visualization_not_installed() -> Exception:
    """Build the canonical no-backend error (reused from the analysis layer)."""
    msg = (
        "No visualization backend is registered. Install one with "
        "`pip install tsdynamics[viz]` (matplotlib) or `tsdynamics[interactive]` "
        "(plotly), or export the spec with .to_dict() and render it yourself."
    )
    try:
        from tsdynamics.analysis._result import VisualizationNotInstalled
    except Exception:  # pragma: no cover - analysis layer unavailable
        return ImportError(msg)
    return VisualizationNotInstalled(msg)
