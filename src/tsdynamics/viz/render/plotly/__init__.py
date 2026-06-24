"""The plotly interactive 2-D rendering backend (stream PLOTLY-RENDER).

This package is the in-tree ``plotly`` rendering backend the dispatch
(:mod:`tsdynamics.viz.render`) registers on first render.  It turns a
backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into an **interactive**
:class:`plotly.graph_objects.Figure` (pan / zoom / hover) suitable for notebooks
and the web.

It draws the 2-D layer marks and 2-D semantic kinds **and** the 3-D marks
(``LINE3D`` / ``SURFACE3D``, plus the ``PHASE_PORTRAIT_3D`` semantic kind) via the
:mod:`._threed` core — an orbitable ``go.Scatter3d`` / ``go.Surface`` ``scene``.
It still **declines** the animation kinds (deferred everywhere) — its
:class:`~tsdynamics.viz.render.caps.RendererCapabilities` list exactly what it
supports, so dispatch falls back to the matplotlib reference renderer (emitting
:class:`~tsdynamics.viz.render.caps.VisualizationDegraded`) for anything it lacks.
Its capabilities advertise ``supports_3d=True``, ``interactive=True`` and
``web_export=True`` (a plotly figure serialises to a self-contained HTML fragment).

The package exposes one hook, :func:`register`, called by
:func:`tsdynamics.viz.render.register_builtin_renderers`.  Importing this package
imports plotly only **lazily** — on first render, via the drawing core in
:mod:`._core` — never at ``import tsdynamics``; if plotly is absent the hook is a
no-op so dispatch degrades to whatever backends are present.  The renderer uses
plotly's ``graph_objects`` API only (no ``plotly.express``, no ``kaleido``).
Self-contained HTML export (stream PLOTLY-HTML) rides on the same hook: the
renderer's :func:`._core.render` accepts ``html=True`` (return the embeddable
interactive HTML fragment) or ``path=`` (write a standalone HTML file), built in
:mod:`._html` (``include_plotlyjs="cdn"`` so the artifact needs no Python kernel
and stays small).  So ``result.plot(backend="plotly", html=True)`` yields HTML
ready for an mkdocs page / web frontend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...spec import PlotKind
from ..caps import RendererCapabilities

if TYPE_CHECKING:
    from tsdynamics.registry import Registry

__all__ = ["register"]

#: The registry name the plotly backend registers under.
_BACKEND_NAME = "plotly"

#: The 2-D layer **marks** the core renderer draws; the 3-D marks live in
#: :data:`_KINDS_3D` and route through :mod:`._threed`.
_MARKS: frozenset[PlotKind] = frozenset(
    {
        PlotKind.LINE,
        PlotKind.SCATTER,
        PlotKind.MARKERS,
        PlotKind.IMAGE,
        PlotKind.HISTOGRAM,
        PlotKind.BAR,
        PlotKind.AREA,
        PlotKind.ERRORBAR,
        PlotKind.QUIVER,
    }
)

#: The **3-D** semantic kind + the 3-D layer marks plotly draws through the
#: :mod:`._threed` core (``go.Scatter3d`` / ``go.Surface`` on an orbitable scene).
_KINDS_3D: frozenset[PlotKind] = frozenset(
    {
        PlotKind.PHASE_PORTRAIT_3D,
        PlotKind.LINE3D,
        PlotKind.SURFACE3D,
    }
)

#: The semantic kinds plotly declines, so a spec of these kinds falls back to
#: matplotlib: the animation kinds (deferred everywhere) and ``COMPOSITE`` (the
#: multi-panel figure — its subplot tiling is the matplotlib reference renderer's
#: job for now; an overlay single-panel spec still renders here natively).
_DECLINED_KINDS: frozenset[PlotKind] = frozenset(
    {
        PlotKind.TRAJECTORY_ANIMATION,
        PlotKind.ENSEMBLE_ANIMATION,
        PlotKind.COMPOSITE,
    }
)

#: The 2-D **semantic kinds** plotly supports: every semantic kind that is not a
#: 3-D / animation kind.  A semantic-kind spec whose layer marks are all in
#: :data:`_MARKS` renders here; otherwise the capability check declines it.
_SEMANTIC_KINDS: frozenset[PlotKind] = frozenset(
    PlotKind.semantic_kinds() - _DECLINED_KINDS - _KINDS_3D
)

#: The 2-D kind set — the 2-D semantic kinds and the 2-D marks.  Kept distinct
#: from :data:`_REGISTERED_KINDS` so the 2-D-path contract stays explicit.
_SUPPORTED_KINDS: frozenset[PlotKind] = _SEMANTIC_KINDS | _MARKS

#: The full kind set the backend advertises — its 2-D kinds plus the 3-D kind /
#: marks the :mod:`._threed` core draws.  Only the animation kinds are declined.
_REGISTERED_KINDS: frozenset[PlotKind] = _SUPPORTED_KINDS | _KINDS_3D


def _plotly_available() -> bool:
    """Whether plotly can be imported (the backend only registers if so)."""
    import importlib.util

    return importlib.util.find_spec("plotly") is not None


def register(registry: Registry) -> bool:
    """Register the plotly renderer into ``registry`` (idempotent, guarded).

    Builds the renderer callable from :func:`tsdynamics.viz.render.plotly._core.render`,
    attaches a :class:`~tsdynamics.viz.render.caps.RendererCapabilities` declaring
    the 2-D **and** 3-D kinds it supports (``supports_3d=True`` so 3-D specs render
    here on an orbitable scene), ``interactive=True`` and ``web_export=True``, and
    adds it under ``"plotly"``.

    The hook is a **no-op** when plotly is not installed (so dispatch falls back)
    and when the backend is already registered (so a second
    :func:`~tsdynamics.viz.render.register_builtin_renderers` call does not
    re-register).  This is the contract
    :func:`tsdynamics.viz.render.register_builtin_renderers` relies on.

    Parameters
    ----------
    registry : Registry
        The :data:`tsdynamics.registry.renderers` container to add the backend to.

    Returns
    -------
    bool
        ``True`` if the backend was newly registered, ``False`` if it was skipped
        (already present, or plotly unavailable).
    """
    if not _plotly_available():
        return False

    if _BACKEND_NAME in registry:
        return False

    capabilities = RendererCapabilities.of_kinds(
        _BACKEND_NAME,
        _REGISTERED_KINDS,
        supports_3d=True,
        interactive=True,
        web_export=True,
    )

    def _render(spec: Any, /, **kw: Any) -> Any:
        # Import the drawing core lazily so registration pulls plotly in only when
        # a render actually happens (the no-plot-at-import guarantee).
        from ._core import render as _render_spec

        return _render_spec(spec, **kw)

    _render.capabilities = capabilities  # type: ignore[attr-defined]
    registry.register(_BACKEND_NAME, _render)
    return True
