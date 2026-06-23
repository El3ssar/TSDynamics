"""The matplotlib reference renderer backend (stream VIZ-MPL-CORE).

This package is the in-tree ``matplotlib`` rendering backend the dispatch
(:mod:`tsdynamics.viz.render`) registers on first render.  It is the
*conformance oracle* of the visualization seam: its
:class:`~tsdynamics.viz.render.caps.RendererCapabilities` declare **all** kinds,
so dispatch falls back to it whenever a partial backend (plotly / json /
three.js) declines a spec.

The package exposes one hook, :func:`register`, called by
:func:`tsdynamics.viz.render.register_builtin_renderers`.  Importing this package
imports matplotlib (lazily — only on first render, never at ``import
tsdynamics``); if matplotlib is absent the hook is a no-op so dispatch degrades to
whatever backends are present.

The drawing lives in :mod:`._core` (the :data:`~._core.KIND_PRESETS` /
:data:`~._core.MARK_DISPATCH` tables + :func:`~._core.render`); this module is
just the registry wiring.  It uses matplotlib's **object-oriented** API only —
no ``matplotlib.pyplot`` anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..caps import RendererCapabilities

if TYPE_CHECKING:
    from tsdynamics.registry import Registry

__all__ = ["register"]

#: The registry name the matplotlib backend registers under.
_BACKEND_NAME = "matplotlib"


def _matplotlib_available() -> bool:
    """Whether matplotlib can be imported (the backend only registers if so)."""
    import importlib.util

    return importlib.util.find_spec("matplotlib") is not None


def register(registry: Registry) -> bool:
    """Register the matplotlib renderer into ``registry`` (idempotent, guarded).

    Builds the renderer callable from :func:`tsdynamics.viz.render.mpl._core.render`,
    attaches an all-kinds :class:`~tsdynamics.viz.render.caps.RendererCapabilities`
    (``supports_3d=True`` so 3-D specs route here; the 3-D *drawing* is the
    VIZ-MPL-3D follow-up), and adds it under ``"matplotlib"``.

    The hook is a **no-op** when matplotlib is not installed (so dispatch falls
    back) and when the backend is already registered (so a second
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
        (already present, or matplotlib unavailable).

    Notes
    -----
    matplotlib is the *universal fallback* (the conformance oracle): it draws
    every kind, so dispatch routes to it whenever the chosen backend declines a
    spec.  To keep it a true last-resort rather than shadowing a more specific
    backend (or a renderer a caller registered explicitly), it is kept **last**
    in the registry's iteration order — when this hook runs again and another
    renderer has since been registered ahead of it, matplotlib is re-seated at
    the end.  Selecting it by name (``render("matplotlib")``) is unaffected.
    """
    if not _matplotlib_available():
        return False

    if _BACKEND_NAME in registry:
        _reseat_last(registry)
        return False

    capabilities = RendererCapabilities.all_kinds(_BACKEND_NAME, supports_3d=True)

    def _render(spec: Any, /, **kw: Any) -> Any:
        # Import the drawing core lazily so registration pulls matplotlib in only
        # when a render actually happens (the no-plot-at-import guarantee).
        from ._core import render as _render_spec

        return _render_spec(spec, **kw)

    _render.capabilities = capabilities  # type: ignore[attr-defined]
    registry.register(_BACKEND_NAME, _render)
    return True


def _reseat_last(registry: Registry) -> None:
    """Move an already-registered matplotlib entry to the end of iteration order.

    Keeps matplotlib the *last-resort* fallback so a more specific backend — or a
    renderer the caller registered explicitly — is preferred by capability-based
    default selection.  A no-op when matplotlib is already last.
    """
    names = registry.names()
    if not names or names[-1] == _BACKEND_NAME or _BACKEND_NAME not in names:
        return
    renderer = registry.get(_BACKEND_NAME)
    registry.unregister(_BACKEND_NAME)
    registry.register(_BACKEND_NAME, renderer)
