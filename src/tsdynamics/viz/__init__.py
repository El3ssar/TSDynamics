"""Visualization — one type, one verb, and four registries.

``import tsdynamics`` (and ``import tsdynamics.viz``) **imports no plot library**
— there is no matplotlib / Plotly / web import on this path.  Four renderers
(matplotlib, plotly, json, threejs) self-register on first use.

The whole surface, in one screen
--------------------------------

.. code-block:: python

    ts.plot(traj)                                # a Plot; ts.plot IS ts.viz.plot
    ts.plot(traj, "psd", yscale="log")           # a named transform, styled at the door
    ts.viz.draw({"x": r, "y": C}, "line")        # arrays straight to a primitive
    ts.viz.grid(p1, p2, p3, cols=2)              # arrange finished plots
    ts.viz.load("f.json")                        # the round trip's read half

    ts.viz.transforms.names()                    # what can be drawn FROM something
    ts.viz.primitives.names()                    # ...and HOW it can be drawn
    ts.viz.transforms.allow("psd", "stem")       # ...admit YOUR primitive to a row
    ts.viz.renderers.find(writes=".svg")         # ...and by whom
    ts.viz.themes.use("publication")             # ...in what look
    ts.viz.styles                                # the style vocabulary, printed
    print(ts.viz.compatibility())                # the declared transform x primitive matrix

One name per picture
--------------------
**A picture is named by its transform, positionally** — ``ts.plot(traj,
"time_series")``, ``ts.plot(rossler, "poincare_section", plane=("y", 0.0))``,
``ts.plot(traj, "phase_portrait", ndim=2)``.  :class:`~tsdynamics.viz.spec.PlotKind`
is the IR's own vocabulary and is not a word you type at a plotting door; a
retired ``kind=`` spelling is answered with the transform line that works.

Composing is three operators and one arranger
---------------------------------------------
``a + b`` overlays, ``a | b`` puts them side by side, ``a / b`` stacks them, and
:func:`grid` arranges any number into a 2-D panel grid.  Every one of them takes
and returns a :class:`~tsdynamics.viz.spec.Plot`, so they nest::

    (ts.plot(traj, "phase_portrait") | ts.plot(traj, "psd")) / ts.plot(traj)
    ts.viz.grid(p1, p2, p3, cols=2, share_color=True).save("figure.pdf")

A one-row grid **is** a row: ``ts.viz.grid(a, b, cols=2)`` and ``a | b`` build
the same arrangement.

Fourteen names, and the shape repeats
-------------------------------------
Four registries (``transforms`` / ``primitives`` / ``renderers`` / ``themes``)
answer to the **same four verbs** — ``register`` / ``names`` / ``find`` / ``get``
— so learning one teaches the rest: each is callable (``ts.viz.transforms()`` is
its listing), each ``names()`` is sorted, and each ``find(text, /, **filters)``
takes free text and returns **names** (``get(name)`` is how you reach a record).
``transforms`` carries a fifth verb that lives nowhere else —
:func:`~tsdynamics.viz.transforms.allow`, which admits a primitive *you*
registered into a shipped transform's row.  :data:`styles` answers the same
listing verbs and deliberately refuses ``register``: it is a contract with the
backends, not an extension point.

Two drawing doors (:func:`plot`, :func:`draw`), one panel arranger
(:func:`grid`), one received type (:class:`~tsdynamics.viz.spec.Plot`), the
arrays escape hatch (:func:`geometry`), the matrix (:func:`compatibility`), the
style table (:data:`styles`), the loader (:func:`load`), the warning you catch
(:class:`~tsdynamics.viz.render.caps.VisualizationDegraded`), and the IR one dot
away (:mod:`~tsdynamics.viz.spec`).

Everything else still exists — importable, reachable, tested.  It just stops
shouting: the 18 IR nouns live at :mod:`ts.viz.spec <tsdynamics.viz.spec>`, and a
name that moved says where it went instead of raising a bare ``AttributeError``.

Out-of-tree renderers register through the ``tsdynamics.renderers`` entry-point
group, out-of-tree plot transforms through ``tsdynamics.plot_transforms`` and
out-of-tree plot primitives through ``tsdynamics.plot_primitives``;
:func:`discover_plugins` loads all three at import.
"""

import os as _os
from collections.abc import Sequence as _Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from .. import registry as _registry
from ..plugins import PLOT_PRIMITIVES_GROUP as _PLOT_PRIMITIVES_GROUP
from ..plugins import PLOT_TRANSFORMS_GROUP as _PLOT_TRANSFORMS_GROUP
from ..plugins import load_plugins as _load_plugins
from ..plugins import register_entry_points as _register_entry_points
from ._visibility import listing_dir as _listing_dir

# Bound but off the tab surface (see ``_INTERNAL_NAMES``).  The redundant ``as``
# form marks these as deliberate re-exports rather than unused imports.
from .export import (
    SCHEMA_VERSION as SCHEMA_VERSION,
)
from .export import (
    from_dict_envelope as from_dict_envelope,
)
from .export import (
    from_json as from_json,
)
from .export import (
    to_dict_envelope as to_dict_envelope,
)
from .export import (
    to_json as to_json,
)
from .spec import (
    Animation as Animation,
)
from .spec import (
    Annotation as Annotation,
)
from .spec import (
    Axis as Axis,
)
from .spec import (
    Layer as Layer,
)
from .spec import (
    Layout as Layout,
)
from .spec import (
    Plot,
)
from .spec import (
    PlotKind as PlotKind,
)
from .spec import (
    Plottable as Plottable,
)
from .style import (
    STYLE_KEYS as STYLE_KEYS,
)
from .style import (
    THEMES as THEMES,
)
from .style import (
    Theme as Theme,
)
from .style import (
    get_theme as get_theme,
)
from .style import (
    normalize_style as normalize_style,
)
from .style import (
    register_theme as register_theme,
)
from .style import (
    set_theme as set_theme,
)
from .style import (
    styles,
    themes,
)
from .transforms import (
    FrameSpace as FrameSpace,
)
from .transforms import (
    Geometry as Geometry,
)
from .transforms import (
    Part as Part,
)
from .transforms import (
    PlotTransform as PlotTransform,
)
from .transforms import (
    Presentation as Presentation,
)
from .transforms import (
    T as T,
)
from .transforms import (
    compatibility,
    draw,
    geometry,
    plot,  # the transform-aware front door IS `viz.plot`
)
from .transforms import (
    make_frame as make_frame,
)
from .transforms import (
    plot_transform as plot_transform,
)

if _TYPE_CHECKING:  # pragma: no cover - typing only
    from .render import renderers as renderers


class _PrimitiveRegistry:
    """``ts.viz.primitives`` — how geometry is drawn, in the shared four-verb shape.

    A *primitive* is the drawing half of the plot layer: a transform turns a
    subject into channels, a primitive turns channels into layers.  ``line`` /
    ``points`` / ``image`` / ``density`` / ``contour`` / ``surface3d`` / ``quiver``
    / ``bars`` / ``band`` / … — "primitive" does not mean "simple"; a basin image
    and a 3-D surface are primitives.

    ::

        ts.viz.primitives.names()
        ts.viz.primitives.get("line").requires
        ts.viz.primitives.find(requires="z")

    .. note::
        ``register`` is the extension door for a **new** primitive and is owned by
        :mod:`tsdynamics.viz.transforms._primitives`; until it lands this facade
        forwards to whatever that module exposes, so the verb starts working the
        moment it exists rather than needing an edit here.
    """

    __slots__ = ()

    def __call__(self) -> list[str]:
        """Return the registered primitive names (the sorted listing)."""
        return self.names()

    def names(self) -> list[str]:
        """Return the sorted names of every registered primitive."""
        from .transforms import primitive_names

        return sorted(primitive_names())

    def get(self, name: str) -> _Any:
        """Return one primitive record (its ``requires`` / ``marks`` declaration)."""
        from .transforms import get_primitive

        return get_primitive(name)

    def find(
        self, what: str = "", /, *, requires: str | None = None, mark: str | None = None
    ) -> list[str]:
        """Return the **names** of the primitives matching a query and filters.

        The fourth shared registry verb, in the shape all four now have::

            ts.viz.primitives.find("line")        # free text over name + summary
            ts.viz.primitives.find(requires="z")  # what can draw a height channel
            ts.viz.primitives.find(mark="image")

        .. versionchanged:: 6.0
            Took no positional argument, so ``primitives.find("line")`` raised
            ``TypeError: find() takes 1 positional argument but 2 were given``.
        """
        text = what.lower()
        out = []
        for name in self.names():
            record = self.get(name)
            if requires is not None and requires not in getattr(record, "requires", ()):
                continue
            if mark is not None and mark not in [str(m) for m in getattr(record, "marks", ())]:
                continue
            if text and text not in name.lower() and text not in getattr(record, "doc", "").lower():
                continue
            out.append(name)
        return out

    def register(
        self,
        name: str | None = None,
        /,
        *,
        requires: _Sequence[str] = (),
        marks: _Sequence[_Any] = (),
        frames: _Sequence[_Any] | None = None,
        options: _Sequence[str] = (),
        emits_frame: _Any = None,
        doc: str = "",
        replace: bool = False,
    ) -> _Any:
        """Register a **new way of drawing** — one decorator, zero private imports.

        ::

            @ts.viz.primitives.register()          # the name is the function's
            def stem(part, **options):
                '''A vertical drop to the baseline plus a marker at each point.'''
                ...

        ``name`` is optional — omitted, it is ``fn.__name__``, exactly as a
        transform's is.  (This facade re-declared it as *required*, so the
        zero-argument spelling the underlying function had already grown raised
        ``TypeError: register() missing 1 required positional argument: 'name'``
        at the public address while working at the private one.)

        See :func:`tsdynamics.viz.transforms.register_primitive` — this facade is
        the same function under the registry's shared verb name, with the same
        signature.  It used to forward through ``(*args, **kwargs)``, so
        ``help(ts.viz.primitives.register)`` showed nothing at all for the door
        the library most wants people to walk through.

        To draw a **shipped** transform with your new primitive, add it to that
        transform's declared row with :func:`tsdynamics.viz.transforms.allow`.
        """
        from .transforms import register_primitive

        return register_primitive(
            name,
            requires=requires,
            marks=marks,
            frames=frames,
            options=options,
            emits_frame=emits_frame,
            doc=doc,
            replace=replace,
        )

    def __contains__(self, name: object) -> bool:
        """Whether a primitive of that name is registered."""
        return name in self.names()

    def __repr__(self) -> str:
        """List each primitive with the channels it requires."""
        rows = ["ts.viz.primitives"]
        for name in self.names():
            requires = " ".join(getattr(self.get(name), "requires", ()) or ())
            rows.append(f"  {name:<12} requires: {requires or '(nothing)'}")
        return "\n".join(rows)


#: ``ts.viz.primitives`` — the primitive registry (also callable, returning names).
primitives = _PrimitiveRegistry()

#: The entry-point group out-of-tree visualization backends declare against
#: (the renderer analogue of :data:`tsdynamics.plugins.ANALYSES_GROUP`)::
#:
#:     [project.entry-points."tsdynamics.renderers"]
#:     matplotlib = "my_pkg.backends:render_matplotlib"
RENDERERS_GROUP = "tsdynamics.renderers"

#: The entry-point group out-of-tree **plot transforms** declare against::
#:
#:     [project.entry-points."tsdynamics.plot_transforms"]
#:     my_plot = "my_pkg.transforms:MY_TRANSFORM"
TRANSFORMS_GROUP = _PLOT_TRANSFORMS_GROUP

#: The entry-point group out-of-tree **plot primitives** declare against::
#:
#:     [project.entry-points."tsdynamics.plot_primitives"]
#:     stem = "my_pkg.primitives:stem"
PRIMITIVES_GROUP = _PLOT_PRIMITIVES_GROUP

#: Names bound here but kept **off** ``__all__`` / ``dir()``.  Each stays
#: reachable — ``ts.viz.SCHEMA_VERSION``, ``from tsdynamics.viz import
#: normalize_style`` — it is only the tab surface that is curated.  The IR nouns
#: (``Animation`` / ``Annotation`` / ``Axis`` / ``Layer`` / ``Layout`` /
#: ``PlotKind`` / ``Geometry`` / ``Part`` / ``FrameSpace`` / ``Presentation`` /
#: ``PlotTransform`` / ``T`` / ``make_frame``) are listed one dot away, at
#: :mod:`ts.viz.spec <tsdynamics.viz.spec>`, which is where a renderer author
#: reads them and where nobody else has to look.
#:
#: **This table is checked against reality**, by
#: ``tests/test_viz_visibility.py::test_every_public_name_bound_on_ts_viz_is_declared``:
#: a public name bound on this module and named by neither :data:`__all__` nor
#: this tuple fails the build.  It used to be decorative, and drifted — eleven
#: names (``os``, ``Any``, ``Sequence``, ``TYPE_CHECKING``,
#: ``register_entry_points``, the ``compose`` / ``export`` / ``style``
#: submodules, and the three entry-point group constants) were bound here and
#: declared nowhere, so ``from tsdynamics.viz import os`` worked and nobody had
#: decided that.  The imports are underscored now; the rest are named below.
_INTERNAL_NAMES: tuple[str, ...] = (
    "Animation",
    "Annotation",
    "Axis",
    "FrameSpace",
    "Geometry",
    "Layer",
    "Layout",
    "Part",
    "PlotKind",
    "PlotTransform",
    "Plottable",
    "Presentation",
    # The three entry-point group constants a plugin author declares against.
    # They are strings in a ``pyproject.toml``, not names anyone types in Python.
    "PRIMITIVES_GROUP",
    "RENDERERS_GROUP",
    "SCHEMA_VERSION",
    "STYLE_KEYS",
    "T",
    "THEMES",
    "TRANSFORMS_GROUP",
    "Theme",
    "discover_plugins",
    "from_dict_envelope",
    "from_json",
    "get_theme",
    "make_frame",
    "normalize_style",
    "plot_transform",
    "register_theme",
    # ``render`` and ``spec`` are submodules resolved by ``__getattr__`` on first
    # touch (``_LAZY_SUBMODULES``); ``spec`` is listed in ``__all__``, ``render``
    # is not.  Listed here so the table covers the name once it is cached.
    "render",
    "set_theme",
    "to_dict_envelope",
    "to_json",
    # Submodules bound as a side effect of the ``from .X import ...`` lines above
    # (Python binds the submodule on the parent package).  Nothing imports them
    # through ``ts.viz``; they are reachable at their own dotted path.
    "compose",
    "export",
    "producers",
    "style",
    "transforms",
)

#: Submodules of :mod:`tsdynamics.viz` resolved on demand by :func:`__getattr__`,
#: so touching ``ts.viz`` still costs no renderer import.  ``render`` used to
#: resolve **or not depending on session history**; resolving it here makes it
#: deterministic without importing it eagerly.
_LAZY_SUBMODULES = frozenset({"render", "spec"})

#: Names resolved lazily out of a submodule: ``{name: (module, attribute)}``.
#: ``renderers`` lives in the render subpackage (it is that layer's registry) and
#: must not be imported eagerly, but it is a *listed* name — so it resolves
#: through :func:`__getattr__` while appearing in ``dir()`` like any other.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "renderers": (".render", "renderers"),
    "VisualizationDegraded": (".render", "VisualizationDegraded"),
}

#: ``old name -> the sentence naming the working spelling``.  Only names that
#: genuinely stop resolving belong here; a *demoted* name (``STYLE_KEYS``,
#: ``to_json``, ``set_theme``, …) still resolves and is listed in
#: :data:`_INTERNAL_NAMES` instead.  A rename's error message **is** its
#: migration guide (see :func:`__getattr__`).
_MOVED: dict[str, str] = {
    "PlotSpec": "PlotSpec is now Plot: `ts.viz.Plot`. Same class, shorter name.",
    "list_transforms": (
        "Use ts.viz.transforms.names() — one registry, the same four verbs "
        "(register / names / find / get) as primitives, renderers and themes."
    ),
}

#: ``ts.viz`` — fourteen names (contract §2, §11.3 T3).
#:
#: .. versionchanged:: 6.0
#:    :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` was **promoted**
#:    (13 → 14).  It is the warning this layer emits when a backend cannot fully
#:    honor a plot — 34 mentions across the documentation, 120 uses across the
#:    suite — and its only address was the internal ``ts.viz.render``, so
#:    ``docs/visualization/styling.md`` caught it by comparing
#:    ``w[0].category.__name__`` to the *string* ``"VisualizationDegraded"``.
#:    A name users are told to catch is a name they must be able to type;
#:    v6 promoted six exception classes to the top level on that same argument.
__all__ = [
    "Plot",
    "VisualizationDegraded",
    "compatibility",
    "draw",
    "geometry",
    "grid",
    "load",
    "plot",
    "primitives",
    "renderers",
    "spec",
    "styles",
    "themes",
    "transforms",
]


def grid(*plots: _Any, rows: int | None = None, cols: int | None = None, **options: _Any) -> Plot:
    """Arrange finished plots into a panel grid, and return the composite :class:`Plot`.

    The named spelling of ``plot(..., layout="grid")``, for the case the owner
    asked for by name — *a grid of different plots*::

        ts.viz.grid(
            ts.plot(tr, "phase_portrait", title="orbit"),
            ts.plot(tr, "time_series", components="x"),
            ts.plot(tr, "psd", xscale="log", yscale="log"),
            cols=2, theme="publication",
        ).save("three-views.png")

    Because the result is itself a :class:`Plot`, a grid nests, animates and
    composes with everything else — that is closure, and it is why this is a
    four-line front rather than a subsystem.

    Parameters
    ----------
    *plots
        Anything :func:`plot` accepts — finished ``Plot`` objects, trajectories,
        systems, results, arrays.
    A **single row or column is a row or a column**, not a degenerate grid:
    ``ts.viz.grid(a, b, cols=2)`` and ``a | b`` build the *same* arrangement, and
    ``rows=1`` / ``cols=1`` resolve to ``"row"`` / ``"stack"``.  They used to
    produce two different ``Layout.mode`` values for one visible 1x2 figure —
    two representations of one picture, which is the defect, not a detail.

    Parameters
    ----------
    *plots
        Anything :func:`plot` accepts — finished ``Plot`` objects, trajectories,
        systems, results, arrays.  The panels of the returned grid **are** the
        plots you passed (not copies), so ``g[0].style(...)`` restyles the panel
        in place, which is what makes a grid inspectable.
    rows, cols : int, optional
        The grid shape; give one and the other is filled in, give neither and the
        grid is made near-square.
    **options
        Forwarded to :func:`plot` (``share_x`` / ``share_y`` / ``share_color`` /
        ``title`` / ``theme`` / any style keyword).

    Returns
    -------
    Plot

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``layout=`` is passed — this verb always arranges a grid, and the
        leak it used to produce named a private module path
        (``tsdynamics.viz.transforms._frontdoor.plot() got multiple values``).
    """
    if "layout" in options:
        from tsdynamics.errors import InvalidParameterError, remedy

        wanted = options.pop("layout")
        raise InvalidParameterError(
            f"ts.viz.grid() always arranges a grid, so it takes no layout= "
            f"(you passed {wanted!r})." + remedy("a | b   # one row", "a / b   # one column")
        )
    mode, rows, cols = _grid_mode(len(plots), rows, cols)
    built: Plot = plot(*plots, layout=mode, rows=rows, cols=cols, **options)
    return built


def _grid_mode(
    n_panels: int, rows: int | None, cols: int | None
) -> tuple[str, int | None, int | None]:
    """Resolve a requested grid shape to the arrangement it actually is.

    A 1xN grid *is* a row and an Nx1 grid *is* a stack, and the operators
    (``a | b``, ``a / b``) build exactly those — so resolving here is what makes
    ``ts.viz.grid(a, b, cols=2)`` and ``a | b`` the same ``Layout`` instead of
    two descriptions of one figure.  ``rows``/``cols`` are dropped once the mode
    fixes the shape.
    """
    from .spec import Layout

    if n_panels <= 0:
        return "grid", rows, cols
    shape_rows, shape_cols = Layout(mode="grid", rows=rows, cols=cols).grid(n_panels)
    if shape_rows == 1:
        return "row", None, None
    if shape_cols == 1:
        return "stack", None, None
    return "grid", rows, cols


def load(source: str | _os.PathLike[str]) -> Plot:
    """Read a :class:`~tsdynamics.viz.spec.Plot` back from a ``.json`` file or JSON text.

    The read half of the round trip whose write half is
    :meth:`Plot.to_json <tsdynamics.viz.spec.Plot.to_json>` /
    ``p.save("f.json")`` — so a plot computed on a cluster can be shipped,
    cached, and drawn elsewhere without re-running the analysis or installing a
    plotting library::

        p.save("run4.json")
        ts.viz.load("run4.json").theme("dark").save("run4.png")

    Parameters
    ----------
    source : str
        A path to a ``.json`` file, or the JSON document itself.

    Returns
    -------
    Plot
    """
    # ``.save`` accepts a ``pathlib.Path``, so ``load`` must too — the sniff used
    # to call ``.lstrip()`` on the argument unconditionally and answered a Path
    # with ``AttributeError: 'PosixPath' object has no attribute 'lstrip'``.
    text = source if isinstance(source, str) else _os.fspath(source)
    if not text.lstrip().startswith(("{", "[")) and _os.path.exists(text):
        with open(text, encoding="utf-8") as fh:
            text = fh.read()
    return from_json(text)


def _register_primitive_entry_points(*, strict: bool = False) -> list[str]:
    """Load the ``tsdynamics.plot_primitives`` group into the primitive registry.

    Primitives cannot go through :func:`tsdynamics.plugins.register_entry_points`
    the way renderers and transforms do: a primitive is not registered *verbatim*
    under its name, it is registered together with the channels it ``requires``
    and the marks it emits.  Two shapes are accepted, and the first is the one a
    plugin author writes:

    1. the entry point names a function whose module already applied
       ``@ts.viz.primitives.register(...)`` — loading it imports that module, so
       the primitive is simply *there* afterwards and nothing more is done;
    2. the entry point names a bare ``build`` callable — it is registered under
       the entry point's own name, reading its declaration off the attributes
       ``requires`` / ``marks`` / ``frames`` / ``options`` / ``emits_frame``.

    A name already registered is left untouched, exactly as for the other groups.
    """
    from .transforms import primitive_names, register_primitive

    before = set(primitive_names())
    for name, obj in _load_plugins(PRIMITIVES_GROUP, strict=strict).items():
        if name in set(primitive_names()):
            continue
        try:
            register_primitive(
                name,
                requires=getattr(obj, "requires", ()),
                marks=getattr(obj, "marks", ()),
                frames=getattr(obj, "frames", None),
                options=getattr(obj, "options", ()),
                emits_frame=getattr(obj, "emits_frame", None),
            )(obj)
        except Exception as exc:  # noqa: BLE001 — isolate third-party failures
            if strict:
                raise
            import warnings

            warnings.warn(
                f"failed to register plot primitive {name!r} from group "
                f"{PRIMITIVES_GROUP!r}: {exc}",
                stacklevel=2,
            )
    return sorted(set(primitive_names()) - before)


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree renderer, plot-transform **and plot-primitive** plugins.

    Walks the ``tsdynamics.renderers``, ``tsdynamics.plot_transforms`` and
    ``tsdynamics.plot_primitives`` entry-point groups and registers each loaded
    object under its entry-point name (see
    :func:`tsdynamics.plugins.register_entry_points`).  Called once at import;
    safe to re-invoke after installing a plugin.  Names already taken are left
    untouched.

    .. versionchanged:: 6.0.1
        ``tsdynamics.plot_primitives`` is loaded.  It had been advertised in
        :data:`tsdynamics.plugins.ALL_GROUPS` with no consumer, so a third-party
        primitive published under the documented group was silently dropped —
        and with it the *only* route a custom primitive has into a shipped
        transform, :func:`ts.viz.transforms.allow`.

    Parameters
    ----------
    strict : bool, default False
        Re-raise the first plugin load failure instead of warning and skipping.

    Returns
    -------
    list[str]
        The names newly registered by this call (renderers, then transforms,
        then primitives).
    """
    found = _register_entry_points(_registry.renderers, RENDERERS_GROUP, strict=strict)
    found += _register_entry_points(_registry.plot_transforms, TRANSFORMS_GROUP, strict=strict)
    found += _register_primitive_entry_points(strict=strict)
    return found


# Populate the renderer and plot-transform registries from out-of-tree plugins at
# import.  The in-tree renderers self-register on first render and the in-tree
# transforms register when ``.transforms`` is imported above; plugin failures are
# isolated inside ``register_entry_points`` (warn-and-skip), so a broken
# third-party package never breaks import.
discover_plugins()


def __getattr__(name: str) -> _Any:
    """Resolve the lazy names, then teach a name that moved.

    Three ordered cases, mirroring the top-level package's:

    1. a demoted **submodule** (``render`` / ``spec``) — imported and cached;
    2. a listed name that lives in a submodule (``renderers`` / ``primitives``) —
       resolved without importing a plotting library at ``ts.viz`` import time;
    3. an **exact hit** in :data:`_MOVED` — answered with the spelling that works.

    Anything else is an ordinary ``AttributeError``, so ``hasattr`` still works
    for every name in the universe.
    """
    import importlib

    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod  # cache: subsequent access skips __getattr__
        return mod
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module, attr = target
        value = getattr(importlib.import_module(module, __name__), attr)
        globals()[name] = value
        return value
    moved = _MOVED.get(name)
    if moved is not None:
        raise AttributeError(f"ts.viz has no {name!r}. {moved}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Expose only the curated public API (:data:`__all__`) to ``dir()`` /
#: autocomplete.  The IR nouns, the envelope helpers and the validation plumbing
#: (:data:`_INTERNAL_NAMES`) stay bound and importable, just off the tab surface.
__dir__ = _listing_dir(__all__)
