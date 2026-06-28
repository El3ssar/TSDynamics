"""
Visualization — the backend-agnostic plot seam (decision D6).

``import tsdynamics`` (and ``import tsdynamics.viz``) **imports no plot library
at import time** — there is no matplotlib / Plotly / web import here.  Four
renderers (matplotlib, plotly, json, threejs) self-register lazily on first
render.

What it ships:

- :mod:`~tsdynamics.viz.spec` — the backend-agnostic, JSON-serializable
  :class:`~tsdynamics.viz.spec.PlotSpec` intermediate representation (plus
  :class:`~tsdynamics.viz.spec.PlotKind`, :class:`~tsdynamics.viz.spec.Layer`,
  :class:`~tsdynamics.viz.spec.Axis`, :class:`~tsdynamics.viz.spec.Annotation`
  and the :class:`~tsdynamics.viz.spec.Plottable` mixin).  A result describes
  itself with ``to_plot_spec()``; the spec carries data + semantic intent and no
  rendering state.
- :mod:`~tsdynamics.viz.style` — the **canonical per-layer style vocabulary**
  (:data:`STYLE_KEYS`, :func:`normalize_style`) and the **figure-level theme
  system** (:class:`Theme`, :func:`get_theme`, :func:`set_theme`, :func:`themes`,
  :func:`register_theme`).  :data:`STYLE_KEYS` is the public introspection mapping
  (``name → StyleKey``); :func:`normalize_style` is the single choke point that
  canonicalizes aliases, validates values, and drops unknown keys.
- :func:`plot` (:mod:`~tsdynamics.viz.compose`) — the **composition front door**:
  arranges one or more plottables into a single- or multi-panel
  :class:`~tsdynamics.viz.spec.PlotSpec`.
- the **renderers registry** (:data:`tsdynamics.registry.renderers`) — a backend
  name → renderer-callable map.  A backend self-registers on first use;
  :meth:`~tsdynamics.viz.spec.PlotSpec.render` looks it up by name.

Out-of-tree renderers register through the ``tsdynamics.renderers`` entry-point
group; :func:`discover_plugins` loads them into
:data:`tsdynamics.registry.renderers` at import.
"""

from .. import registry as _registry
from ..plugins import register_entry_points
from .compose import plot
from .spec import (
    Animation,
    Annotation,
    Axis,
    Layer,
    Layout,
    PlotKind,
    PlotSpec,
    Plottable,
)
from .style import (
    STYLE_KEYS,
    THEMES,
    Theme,
    get_theme,
    normalize_style,
    register_theme,
    set_theme,
    themes,
)

#: The entry-point group out-of-tree visualization backends declare against
#: (the renderer analogue of :data:`tsdynamics.plugins.ANALYSES_GROUP` /
#: :data:`~tsdynamics.plugins.TRANSFORMS_GROUP`).  A backend package wires itself
#: in with, in its own ``pyproject.toml``::
#:
#:     [project.entry-points."tsdynamics.renderers"]
#:     matplotlib = "my_pkg.backends:render_matplotlib"
RENDERERS_GROUP = "tsdynamics.renderers"

__all__ = [
    "Animation",
    "Annotation",
    "Axis",
    "Layer",
    "Layout",
    "PlotKind",
    "PlotSpec",
    "Plottable",
    "STYLE_KEYS",
    "THEMES",
    "Theme",
    "discover_plugins",
    "get_theme",
    "normalize_style",
    "plot",
    "register_theme",
    "set_theme",
    "themes",
]


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree renderer plugins into :data:`tsdynamics.registry.renderers`.

    Walks the ``tsdynamics.renderers`` entry-point group and registers each
    loaded object (a renderer callable) under its entry-point name (see
    :func:`tsdynamics.plugins.register_entry_points`).  Called once at import;
    safe to re-invoke after installing a backend.  Names already taken are left
    untouched.

    Parameters
    ----------
    strict : bool, default False
        Re-raise the first plugin load failure instead of warning and skipping.

    Returns
    -------
    list[str]
        The names newly registered by this call.
    """
    return register_entry_points(_registry.renderers, RENDERERS_GROUP, strict=strict)


# Populate the renderers registry from out-of-tree plugins at import. No backend
# ships in-tree yet (visualization is deferred), so this is the only path that can
# fill the registry today; plugin failures are isolated inside
# `register_entry_points` (warn-and-skip), so a broken backend never breaks import.
discover_plugins()


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
