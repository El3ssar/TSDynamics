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
- :mod:`~tsdynamics.viz.transforms` — the **plot-transform registry**: what to
  plot (a transform turns a subject into :class:`~tsdynamics.viz.transforms.Geometry`)
  separated from how to draw it (a *primitive* turns geometry into layers), with
  a **declared** compatibility matrix connecting them.  :func:`compatibility`
  prints it; :func:`geometry` / :func:`draw` are the raw-array escape hatches;
  :func:`~tsdynamics.viz.transforms.plot_transform` registers a new one in one
  call.
- :func:`plot` (:mod:`~tsdynamics.viz.compose`) — the **composition front door**:
  arranges one or more plottables into a single- or multi-panel
  :class:`~tsdynamics.viz.spec.PlotSpec`.
- the **renderers registry** (:data:`tsdynamics.registry.renderers`) — a backend
  name → renderer-callable map.  A backend self-registers on first use;
  :meth:`~tsdynamics.viz.spec.PlotSpec.render` looks it up by name.
- :func:`to_json` / :func:`from_json` (and the mapping-level
  :func:`to_dict_envelope` / :func:`from_dict_envelope`, plus
  :data:`SCHEMA_VERSION`) — the **versioned JSON envelope** for a
  :class:`~tsdynamics.viz.spec.PlotSpec`.  ``spec.save("fig.json")`` and
  ``spec.render("json")`` both write this envelope; these are the matching
  *readers*, so a spec computed on a cluster can be shipped, cached, and replotted
  elsewhere without re-running the analysis or installing a plotting library.
  (The read half existed but was unreachable — ``ts.viz.from_json`` did not
  resolve — which made the round trip one-way in practice.  Serialization is half
  the "embed plots in web" story, so it is promoted rather than deleted.)

Out-of-tree renderers register through the ``tsdynamics.renderers`` entry-point
group and out-of-tree plot transforms through ``tsdynamics.plot_transforms``;
:func:`discover_plugins` loads both into :data:`tsdynamics.registry.renderers`
and :data:`tsdynamics.registry.plot_transforms` at import.
"""

from .. import registry as _registry
from ..plugins import PLOT_TRANSFORMS_GROUP, register_entry_points
from .compose import plot
from .export import SCHEMA_VERSION, from_dict_envelope, from_json, to_dict_envelope, to_json
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
from .transforms import (
    Geometry,
    PlotTransform,
    T,
    compatibility,
    draw,
    geometry,
    plot_transform,
)
from .transforms import transforms as plot_transforms

#: The entry-point group out-of-tree visualization backends declare against
#: (the renderer analogue of :data:`tsdynamics.plugins.ANALYSES_GROUP`).
#: A backend package wires itself in with, in its own ``pyproject.toml``::
#:
#:     [project.entry-points."tsdynamics.renderers"]
#:     matplotlib = "my_pkg.backends:render_matplotlib"
RENDERERS_GROUP = "tsdynamics.renderers"

#: The entry-point group out-of-tree **plot transforms** declare against::
#:
#:     [project.entry-points."tsdynamics.plot_transforms"]
#:     my_plot = "my_pkg.transforms:MY_TRANSFORM"
#:
#: The target must be a :class:`~tsdynamics.viz.transforms.PlotTransform` record
#: (what :func:`~tsdynamics.viz.transforms.plot_transform` builds), so a
#: third-party plot arrives with its own declared compatibility row and is
#: indistinguishable from an in-tree one.
TRANSFORMS_GROUP = PLOT_TRANSFORMS_GROUP

__all__ = [
    "Animation",
    "Annotation",
    "Axis",
    "Geometry",
    "Layer",
    "Layout",
    "PlotKind",
    "PlotSpec",
    "PlotTransform",
    "Plottable",
    "SCHEMA_VERSION",
    "STYLE_KEYS",
    "T",
    "THEMES",
    "Theme",
    "compatibility",
    "discover_plugins",
    "draw",
    "from_dict_envelope",
    "from_json",
    "geometry",
    "get_theme",
    "normalize_style",
    "plot",
    "plot_transform",
    "plot_transforms",
    "register_theme",
    "set_theme",
    "themes",
    "to_dict_envelope",
    "to_json",
    "transforms",
]

# NOTE: the *listing* function is exported as ``plot_transforms``, not
# ``transforms``: this package has a ``transforms`` **subpackage**, and binding a
# function of that name over it is exactly the shadowing defect the v4 namespace
# work removed elsewhere (a function hiding a subpackage of the same name).
# ``ts.viz.transforms`` is therefore always the module — navigable, holding
# ``Geometry`` / ``plot_transform`` / ``PRIMITIVES`` — and
# ``ts.viz.plot_transforms(source="model")`` is the filtered listing, named after
# the registry (:data:`tsdynamics.registry.plot_transforms`) it reads.


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree renderer **and plot-transform** plugins.

    Walks the ``tsdynamics.renderers`` and ``tsdynamics.plot_transforms``
    entry-point groups and registers each loaded object — a renderer callable, a
    :class:`~tsdynamics.viz.transforms.PlotTransform` record — under its
    entry-point name (see :func:`tsdynamics.plugins.register_entry_points`).
    Called once at import; safe to re-invoke after installing a plugin.  Names
    already taken are left untouched.

    Parameters
    ----------
    strict : bool, default False
        Re-raise the first plugin load failure instead of warning and skipping.

    Returns
    -------
    list[str]
        The names newly registered by this call (renderers first, then
        transforms).
    """
    found = register_entry_points(_registry.renderers, RENDERERS_GROUP, strict=strict)
    found += register_entry_points(_registry.plot_transforms, TRANSFORMS_GROUP, strict=strict)
    return found


# Populate the renderer and plot-transform registries from out-of-tree plugins at
# import.  The in-tree renderers self-register on first render and the in-tree
# transforms register when ``.transforms`` is imported above; plugin failures are
# isolated inside `register_entry_points` (warn-and-skip), so a broken third-party
# package never breaks import.
discover_plugins()


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
