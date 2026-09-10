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

#: Extensions ``Figure.savefig`` writes — **measured**, not assumed.  ``.pgf`` was
#: missing (it works) and ``.webp`` was declared but unreachable, because a single
#: undivided ``writes`` set mixed still formats with movie formats.
_WRITES_STATIC: frozenset[str] = frozenset(
    {
        ".png",
        ".pdf",
        ".svg",
        ".svgz",
        ".jpg",
        ".jpeg",
        ".eps",
        ".ps",
        ".pgf",
        ".tif",
        ".tiff",
        ".webp",
        ".gif",
    }
)

#: Extensions :class:`~matplotlib.animation.FuncAnimation`'s own ``save`` writes
#: (ffmpeg / pillow) — the **only** way this library writes a movie, since plotly's
#: animation core is single-panel HTML, so an animated composite must land here.
#: Measured: the four formats ``savefig`` refuses outright (``.apng`` ``.m4v``
#: ``.mov`` ``.webm``) **are** writable here, and ``.mp4`` — the headline movie
#: format — is animated-only.  Declaring both halves in one undivided set is what
#: made ``p.save("x.mp4")`` on a static plot raise matplotlib's raw
#: ``ValueError: Format 'mp4' is not supported``.
_WRITES_ANIMATED: frozenset[str] = frozenset({".mp4", ".gif", ".webm", ".mov", ".m4v", ".apng"})


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
    spec.  Iteration order is owned by the dispatch layer
    (:func:`tsdynamics.viz.render.register_builtin_renderers`), which seats
    matplotlib **first** via ``_seat_preferred_first`` after every registration
    pass — so this hook just registers (or no-ops if already present) and never
    re-seats itself.  Selecting it by name (``render("matplotlib")``) is
    unaffected.
    """
    if not _matplotlib_available():
        return False

    if _BACKEND_NAME in registry:
        return False

    capabilities = RendererCapabilities.all_kinds(
        _BACKEND_NAME,
        supports_3d=True,
        writes=_WRITES_STATIC,
        writes_animated=_WRITES_ANIMATED,
    )

    def _render(spec: Any, /, **kw: Any) -> Any:
        # Import the drawing core lazily so registration pulls matplotlib in only
        # when a render actually happens (the no-plot-at-import guarantee).
        from ._core import render as _render_spec

        return _render_spec(spec, **kw)

    _render.capabilities = capabilities  # type: ignore[attr-defined]
    registry.register(_BACKEND_NAME, _render)
    return True
