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

from typing import Any

from .. import registry as _registry
from ..plugins import PLOT_TRANSFORMS_GROUP, register_entry_points
from .export import (
    SCHEMA_VERSION,
    from_dict_envelope,
    from_json,
    to_dict_envelope,
    to_json,
)

# Kept bound (and importable) but off the curated tab surface — see
# ``_INTERNAL_NAMES``.  The redundant ``as`` form marks them as deliberate
# re-exports rather than unused imports.
from .spec import (
    Animation,
    Annotation,
    Axis,
    Layer,
    Layout,
    PlotKind,
    PlotSpec,
)
from .spec import (
    Plottable as Plottable,
)
from .style import (
    STYLE_KEYS,
    Theme,
    get_theme,
    register_theme,
    set_theme,
    themes,
)
from .style import (
    THEMES as THEMES,
)
from .style import (
    normalize_style as normalize_style,
)
from .transforms import (
    FrameSpace,
    Geometry,
    Part,
    PlotTransform,
    Presentation,
    T,
    compatibility,
    draw,
    geometry,
    make_frame,
    plot,  # the transform-aware front door IS `viz.plot`
    plot_transform,
)
from .transforms import transforms as list_transforms

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

#: Names bound on this package but kept **off** ``__all__`` / ``dir()``: plumbing
#: a user of the plotting layer never types.  Each stays reachable
#: (``ts.viz.SCHEMA_VERSION``, ``from tsdynamics.viz import normalize_style``) —
#: it is only the tab surface that is curated.
#:
#: * ``discover_plugins`` — entry-point loading, packaging machinery.
#: * ``THEMES`` — the mutable theme registry behind :func:`themes` /
#:   :func:`get_theme` / :func:`register_theme`, which are the accessors.
#: * ``normalize_style`` — the internal validation choke point every renderer
#:   funnels through; users pass style keys to ``.style(...)``.
#: * ``Plottable`` — the mixin a *result class* implements, not something a
#:   plotting caller instantiates.
#:
#: The **JSON envelope stays listed** (``SCHEMA_VERSION`` / ``to_dict_envelope`` /
#: ``from_dict_envelope`` alongside :func:`to_json` / :func:`from_json`).  It
#: reads like plumbing, but a previous stream promoted the loader half precisely
#: because ``ts.viz.from_json`` not resolving made the save/load round trip
#: one-way in practice — being *listed*, not merely importable, is the point, and
#: ``tests/test_plotspec.py::test_export_names_are_in_the_curated_viz_all`` pins it.
#: * ``render`` — the backend-dispatch subpackage.  Demoted from the listing, but
#:   it must still *resolve*: it is the plugin/dispatch surface
#:   (``ts.viz.render.select_renderer`` / ``register_builtin_renderers``).  It is
#:   bound lazily by :func:`__getattr__` below rather than eagerly, so touching
#:   ``ts.viz`` still costs no renderer import.
_INTERNAL_NAMES = (
    "THEMES",
    "Plottable",
    "discover_plugins",
    "normalize_style",
    "render",
)

#: Subpackages of :mod:`tsdynamics.viz` resolved on demand by :func:`__getattr__`.
#:
#: ``render`` used to resolve **or not depending on session history**: nothing
#: imports it at ``tsdynamics.viz`` import time, so ``ts.viz.render`` raised
#: ``AttributeError`` in a fresh session and succeeded in one that had already
#: drawn something (the first render imports the subpackage, which binds it on
#: this package as a side effect).  That is the same order-dependence that made
#: ``dir(ts.viz.render)`` change after the first render, and it broke the rule
#: the rest of the curation keeps: *demotion is never removal*.  Resolving it
#: here makes it deterministic without importing it eagerly.
_LAZY_SUBMODULES = frozenset({"render"})

__all__ = [
    # The front door.
    "plot",
    "T",
    # The IR you build, inspect and render.
    "PlotSpec",
    "PlotKind",
    "Layer",
    "Axis",
    "Layout",
    "Annotation",
    "Animation",
    # Styling & themes.
    "STYLE_KEYS",
    "Theme",
    "themes",
    "get_theme",
    "set_theme",
    "register_theme",
    # Plot transforms: what to plot, how to draw it, and what pairs with what.
    "transforms",
    "list_transforms",
    "compatibility",
    "geometry",
    "draw",
    # Writing one: the decorator plus the four substrate names its body needs.
    # Before v6 an author had to reach into two PRIVATE modules
    # (``viz.transforms._base``, ``viz._frames``) for ``Part`` / ``FrameSpace`` /
    # ``make_frame`` / ``Presentation``, which made "one decorator call and
    # nothing else" true of the registry and false of the author.
    "plot_transform",
    "PlotTransform",
    "Geometry",
    "Part",
    "FrameSpace",
    "make_frame",
    "Presentation",
    # Serialization round trip — both halves, listed on purpose (see
    # ``_INTERNAL_NAMES``).
    "to_json",
    "from_json",
    "to_dict_envelope",
    "from_dict_envelope",
    "SCHEMA_VERSION",
]

# NOTE: the *listing* function is exported as ``list_transforms``, not
# ``transforms``: this package has a ``transforms`` **subpackage**, and binding a
# function of that name over it is exactly the shadowing defect the v4 namespace
# work removed elsewhere (a function hiding a subpackage of the same name — the
# reason ``ts.viz.transforms()``, the obvious spelling of "what can this draw?",
# used to answer ``TypeError: 'module' object is not callable``).
# ``ts.viz.transforms`` is therefore always the module — navigable, holding
# ``Geometry`` / ``plot_transform`` / ``PRIMITIVES`` — and
# ``ts.viz.list_transforms(source="model")`` is the filtered listing.  The name
# is a *verb* rather than the registry's noun (``registry.plot_transforms``), so
# the function and the table it reads are never the same word either.


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


def __getattr__(name: str) -> Any:
    """Resolve the demoted :mod:`~tsdynamics.viz.render` subpackage on demand.

    See :data:`_LAZY_SUBMODULES`: this exists so ``ts.viz.render`` resolves the
    same way in every session instead of depending on whether something has
    already drawn, while still keeping the renderer import off the ``ts.viz``
    import path.
    """
    if name in _LAZY_SUBMODULES:
        import importlib

        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod  # cache: subsequent access skips __getattr__
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete.

    The envelope/registry/validation plumbing (:data:`_INTERNAL_NAMES`) stays
    bound and importable, just off the tab surface.
    """
    return sorted(__all__)
