"""The plotly interactive 2-D rendering backend (stream PLOTLY-RENDER).

This package is the in-tree ``plotly`` rendering backend the dispatch
(:mod:`tsdynamics.viz.render`) registers on first render.  It turns a
backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into an **interactive**
:class:`plotly.graph_objects.Figure` (pan / zoom / hover) suitable for notebooks
and the web.

It is a *partial* backend: it draws the 2-D layer marks and 2-D semantic kinds
and **declines** the 3-D marks (``LINE3D`` / ``SURFACE3D``) — its
:class:`~tsdynamics.viz.render.caps.RendererCapabilities` list exactly what it
supports, so dispatch falls back to the matplotlib reference renderer (emitting
:class:`~tsdynamics.viz.render.caps.VisualizationDegraded`) for anything it lacks.
Its capabilities advertise ``interactive=True`` and ``web_export=True`` (a plotly
figure serialises to a self-contained HTML fragment).

The package exposes one hook, :func:`register`, called by
:func:`tsdynamics.viz.render.register_builtin_renderers`.  Importing this package
imports plotly only **lazily** — on first render, via the drawing core in
:mod:`._core` — never at ``import tsdynamics``; if plotly is absent the hook is a
no-op so dispatch degrades to whatever backends are present.  The renderer uses
plotly's ``graph_objects`` API only (no ``plotly.express``, no ``kaleido``).
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

#: The 2-D layer **marks** plotly draws (the 3-D ``LINE3D`` / ``SURFACE3D`` marks
#: are deliberately excluded so dispatch falls back to matplotlib for them).
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

#: The **3-D** semantic kinds + the animation kinds plotly's 2-D renderer
#: declines (so a spec of these kinds falls back to matplotlib).
_DECLINED_KINDS: frozenset[PlotKind] = frozenset(
    {
        PlotKind.PHASE_PORTRAIT_3D,
        PlotKind.TRAJECTORY_ANIMATION,
        PlotKind.ENSEMBLE_ANIMATION,
    }
)

#: The 2-D **semantic kinds** plotly supports: every semantic kind that is not a
#: 3-D / animation kind.  A semantic-kind spec whose layer marks are all in
#: :data:`_MARKS` renders here; otherwise the capability check declines it.
_SEMANTIC_KINDS: frozenset[PlotKind] = frozenset(PlotKind.semantic_kinds() - _DECLINED_KINDS)

#: The full kind set the backend advertises — its 2-D semantic kinds and the 2-D
#: marks.  Anything outside this (the 3-D marks, the 3-D / animation semantic
#: kinds) is declined.
_SUPPORTED_KINDS: frozenset[PlotKind] = _SEMANTIC_KINDS | _MARKS


def _plotly_available() -> bool:
    """Whether plotly can be imported (the backend only registers if so)."""
    import importlib.util

    return importlib.util.find_spec("plotly") is not None


def register(registry: Registry) -> bool:
    """Register the plotly renderer into ``registry`` (idempotent, guarded).

    Builds the renderer callable from :func:`tsdynamics.viz.render.plotly._core.render`,
    attaches a :class:`~tsdynamics.viz.render.caps.RendererCapabilities` declaring
    the 2-D kinds it supports (``supports_3d=False`` so 3-D specs fall back),
    ``interactive=True`` and ``web_export=True``, and adds it under ``"plotly"``.

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
        _SUPPORTED_KINDS,
        supports_3d=False,
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
