"""Backend dispatch for the visualization seam (stream VIZ-DISPATCH).

:meth:`tsdynamics.viz.spec.PlotSpec.render` delegates here.  This module turns a
backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into a figure (or export
payload) by choosing a registered renderer and **falling back** to the matplotlib
reference renderer when the chosen backend cannot draw a spec's kind:

- :func:`register_builtin_renderers` — lazily import and register the in-tree
  backends that are *installed* (matplotlib, plotly, json, three.js).  Called on
  first render, **never at import**, so ``import tsdynamics`` stays plot-free.
- :func:`select_renderer` — resolve a backend by name (or pick a default) and,
  via the :class:`~tsdynamics.viz.render.caps.RendererCapabilities`, fall back to
  a capable backend (emitting :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`)
  when the requested one declines the spec.  When no backend is named,
  **matplotlib is the default** — it is the universal reference renderer and is
  always preferred over partial backends (plotly, json, threejs).
- :func:`render_spec` — the one entry point :meth:`PlotSpec.render` calls.
  Before calling the renderer it emits **one consolidated**
  :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` warning (via
  :func:`~tsdynamics.viz.render.caps.style_honoring_gaps`) naming every style
  key, animation knob, or theme/axis field that the chosen backend will ignore.
  The renderer is then called with ``warn=False`` so it suppresses duplicate
  per-key warnings.
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
from .caps import (
    Renderer,
    RendererCapabilities,
    RenderResult,
    VisualizationDegraded,
    _normalize_backend_name,
    style_honoring_gaps,
)

__all__ = [
    "Renderer",
    "RendererCapabilities",
    "RenderResult",
    "VisualizationDegraded",
    "normalize_kind",
    "register_builtin_renderers",
    "render_spec",
    "select_renderer",
    "style_honoring_gaps",
]

#: The in-tree backend submodules dispatch tries to register, in preference
#: order.  Each is a ``tsdynamics.viz.render.<name>`` module exposing
#: ``register(registry)``; a module that is absent (its stream has not landed)
#: or whose plotting library is not installed is skipped.
#:
#: **Note on ordering:** matplotlib is listed first to make it the *default*
#: drawing backend when no ``backend=`` is given — it is the universal reference
#: renderer that draws every kind.  Plotly comes second so ``backend="plotly"``
#: always works when available.  The data-export backends (json / threejs) are
#: listed last; ``select_renderer`` skips them when no backend is named (they
#: return a payload, not a figure).
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

#: The canonical name that ``select_renderer`` prefers as the default drawing
#: backend (when no ``backend=`` is given and no capability-based selection
#: already picked a winner).  Must be a name in ``_BUILTIN_BACKENDS``'s
#: ``register(registry)`` hook.  Matplotlib is preferred because it is the
#: universal reference renderer that draws every kind.
_PREFERRED_DEFAULT_BACKEND = "matplotlib"


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

    After registration the registry order is normalised so that the preferred
    default backend (matplotlib) is **first** in iteration order when present,
    giving :func:`select_renderer` a deterministic default independent of which
    backends are installed or how many times this function is called.
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
    # Normalise iteration order: ensure the preferred default backend is first
    # so that select_renderer(no-backend) always picks it when available.
    _seat_preferred_first(registry.renderers)
    return [n for n in registry.renderers.names() if n not in before]


def _seat_preferred_first(renderers: Any) -> None:
    """Move the preferred default backend to the *front* of iteration order.

    Idempotent — if the preferred backend is already first (or not registered at
    all) this is a no-op.  Calling this after every ``register_builtin_renderers``
    makes the default selection deterministic regardless of registration order.
    """
    names = renderers.names()
    if not names or names[0] == _PREFERRED_DEFAULT_BACKEND:
        return
    if _PREFERRED_DEFAULT_BACKEND not in names:
        return
    renderer = renderers.get(_PREFERRED_DEFAULT_BACKEND)
    renderers.unregister(_PREFERRED_DEFAULT_BACKEND)
    # Re-insert at the front: unregister removes it; re-registering now puts it
    # at the end, so we unregister everything after it and re-add them after.
    # Simpler: collect all current names after removal, prepend preferred, rebuild.
    remaining_names = renderers.names()
    remaining = [(n, renderers.get(n)) for n in remaining_names]
    for n in remaining_names:
        renderers.unregister(n)
    renderers.register(_PREFERRED_DEFAULT_BACKEND, renderer)
    for n, r in remaining:
        renderers.register(n, r)


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

    When ``backend`` is ``None`` (the default), **matplotlib is preferred** — it
    is the universal reference renderer that draws every kind and is always
    registered first after :func:`register_builtin_renderers`.  If matplotlib is
    not installed, the first registered drawing backend that can handle the spec
    is chosen (data-export backends such as ``json`` / ``threejs`` are skipped in
    the default selection).

    Parameters
    ----------
    spec : PlotSpec
        The spec to render (its kind / 3-D-ness drive the capability check).
    backend : str, optional
        A requested backend name.  ``None`` picks **matplotlib** when available,
        otherwise the first registered drawing backend that can draw the spec.

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
        # Accept the same friendly aliases caps uses (``"mpl"`` → ``"matplotlib"``)
        # so ``.render(backend="mpl")`` resolves the registered renderer instead of
        # raising KeyError on the alias.
        backend = _normalize_backend_name(backend)
        renderer = renderers.get(backend)  # KeyError (naming) if unknown
        if _can_render(renderer, spec):
            return backend, renderer
        # The named backend declines: prefer a *drawing* fallback over a
        # data-export one (the caller asked to draw, not to serialize).
        fallback = _first_capable(spec, renderers, exclude=backend, skip_data_export=True)
        if fallback is None:
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

    # Default (no-backend) selection:
    # 1. Prefer the canonical default backend (matplotlib) if it is registered
    #    and can draw the spec — deterministic, independent of registry order.
    if _PREFERRED_DEFAULT_BACKEND in renderers:
        preferred = renderers.get(_PREFERRED_DEFAULT_BACKEND)
        if _can_render(preferred, spec):
            return _PREFERRED_DEFAULT_BACKEND, preferred

    # 2. Fall through to capability-based selection among drawing backends
    #    (data-export backends return payloads, not figures — skip them).
    chosen = _first_capable(
        spec, renderers, skip_data_export=True, exclude=_PREFERRED_DEFAULT_BACKEND
    )
    if chosen is None:
        chosen = _first_capable(spec, renderers, exclude=_PREFERRED_DEFAULT_BACKEND)
    if chosen is not None:
        return chosen

    # 3. Last resort: use the first registered backend and let it try.
    name = renderers.names()[0]
    return name, renderers.get(name)


def _first_capable(
    spec: PlotSpec,
    renderers: Any,
    *,
    exclude: str | None = None,
    skip_data_export: bool = False,
) -> tuple[str, Any] | None:
    """Return the first ``(name, renderer)`` that can draw ``spec``, else ``None``.

    When ``skip_data_export`` is set, a backend that declares
    ``data_export=True`` in its :class:`~tsdynamics.viz.render.caps.RendererCapabilities`
    (a serializer such as ``json`` / ``threejs``, which returns a payload rather
    than a figure) is skipped — so default selection prefers a real drawing
    backend.  A capability-less renderer (no descriptor) is never skipped.
    """
    for name in renderers.names():
        if name == exclude:
            continue
        renderer = renderers.get(name)
        if skip_data_export:
            caps = _capabilities_of(renderer)
            if caps is not None and caps.data_export:
                continue
        if _can_render(renderer, spec):
            return name, renderer
    return None


def render_spec(spec: PlotSpec, backend: str | None = None, **backend_kw: Any) -> Any:
    """Render ``spec`` through a registered backend (the :meth:`PlotSpec.render` seam).

    Registers the installed in-tree backends (lazily, once), selects one by name
    or by capability (falling back when the requested backend declines the spec),
    emits **one consolidated** :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`
    warning for any style keys / animation knobs / theme fields the chosen backend
    will ignore (via :func:`~tsdynamics.viz.render.caps.style_honoring_gaps`), then
    calls the renderer with ``warn=False`` so it suppresses duplicate per-key
    warnings.

    Parameters
    ----------
    spec : PlotSpec
        The spec to render.
    backend : str, optional
        Backend name; ``None`` selects **matplotlib** (the default), or the first
        capable drawing backend when matplotlib is not installed.
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
    chosen_name, renderer = select_renderer(spec, backend)

    # Emit ONE consolidated degradation warning for all the knobs the chosen
    # backend will silently ignore (style keys, animation, theme/axis fields).
    gaps = style_honoring_gaps(spec, chosen_name)
    if gaps:
        warnings.warn(
            f"{chosen_name}: ignoring {', '.join(gaps)}",
            VisualizationDegraded,
            stacklevel=2,
        )

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
