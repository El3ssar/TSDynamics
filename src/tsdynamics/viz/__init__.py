"""
Visualization — the deferred-but-designed-for plot seam (decision D6).

This package ships **no rendering backend**: visualization is deferred, and
``import tsdynamics`` (and ``import tsdynamics.viz``) pulls in **no plot library**
— there is no matplotlib / Plotly / web import anywhere here.  What it does ship
is the *contract* a future multi-backend suite plugs into by construction:

- :mod:`~tsdynamics.viz.spec` — the backend-agnostic, JSON-serializable
  :class:`~tsdynamics.viz.spec.PlotSpec` intermediate representation (plus
  :class:`~tsdynamics.viz.spec.PlotKind`, :class:`~tsdynamics.viz.spec.Layer`,
  :class:`~tsdynamics.viz.spec.Axis`, :class:`~tsdynamics.viz.spec.Annotation`
  and the :class:`~tsdynamics.viz.spec.Plottable` mixin).  A result describes
  itself with ``to_plot_spec()``; the spec carries data + semantic intent and no
  rendering state.
- the **renderers registry** (:data:`tsdynamics.registry.renderers`) — a backend
  name → renderer-callable map mirroring :data:`tsdynamics.registry.analyses` /
  :data:`~tsdynamics.registry.transforms` exactly.  A backend self-registers a
  callable that consumes a :class:`~tsdynamics.viz.spec.PlotSpec`;
  :meth:`~tsdynamics.viz.spec.PlotSpec.render` looks the backend up by name.
  Until a backend lands the registry stays empty, so
  :meth:`~tsdynamics.viz.spec.PlotSpec.render` raises a helpful
  ``VisualizationNotInstalled``.

Out-of-tree renderers register through the ``tsdynamics.renderers`` entry-point
group (the visualization analogue of ``tsdynamics.analyses`` /
``tsdynamics.transforms``; see :mod:`tsdynamics.plugins`); :func:`discover_plugins`
loads them into :data:`tsdynamics.registry.renderers` at import.  In-tree backends
(when a viz stream lands) self-register from their own modules.

Do not add rendering code to an engine/analysis stream — only the spec IR and
this registry seam live here today.
"""

from .. import registry as _registry
from ..plugins import register_entry_points
from .spec import (
    Annotation,
    Axis,
    Layer,
    PlotKind,
    PlotSpec,
    Plottable,
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
    "Annotation",
    "Axis",
    "Layer",
    "PlotKind",
    "PlotSpec",
    "Plottable",
    "discover_plugins",
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
