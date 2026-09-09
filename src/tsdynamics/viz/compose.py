"""Compose specs into one figure — the ``tsdynamics.viz.plot`` front door.

This is the *composition* seam.  Where :meth:`tsdynamics.data.Trajectory.to_plot_spec`
describes **one panel**, :func:`plot` arranges one or more things into a figure:

- :func:`plot` takes any mix of plottables (a :class:`~tsdynamics.data.Trajectory`,
  a system, an analysis result) and already-built
  :class:`~tsdynamics.viz.spec.PlotSpec` objects,
  converts each to a spec, and returns a **spec** — a single-panel spec for
  ``layout="overlay"`` (everything drawn on one set of axes) or a
  :data:`~tsdynamics.viz.spec.PlotKind.COMPOSITE` spec for ``layout="stack"`` /
  ``"row"`` / ``"grid"`` (one panel each).
- Because the input type and the return type are **the same** (a ``PlotSpec``), a
  ``plot(...)`` result feeds straight back into ``plot(...)``: build each panel
  with one flat call, then arrange the panels with another::

      px = ts.viz.plot(lor1, lor2, components="x")   # overlay → one panel
      py = ts.viz.plot(lor1, lor2, components="y")
      ts.viz.plot(px, py, layout="stack")            # two stacked panels

**What may share one set of axes (v6).**  Overlay legality is *frame*
compatibility — the same coordinate space, the same dimension, the same axes
(:class:`tsdynamics.viz._frames.Frame`) — replacing the hard-coded three-kind
whitelist that used to decide it.  So::

      ts.viz.plot(basins, attractors, traj, fixed_points)   # one plane, one axes
      ts.viz.plot(traj_xy, fixed_points_xz)                 # raises: different planes

and the draw order is fixed **by role** (fields under curves under markers), not
by argument order, so the call is order-free.  Pass ``on="force"`` to overlay a
deliberate mismatch with a warning.  Grow a figure incrementally with
:meth:`~tsdynamics.viz.spec.PlotSpec.add`, which goes through the same merge.

The returned spec renders itself (notebook display, ``.plot()``, ``.save(...)``,
``.render(...)``); see :class:`tsdynamics.viz.spec.PlotSpec`.

This module imports **no plotting library** — it only builds the backend-agnostic
IR — so ``import tsdynamics`` stays plot-free (``tsdynamics.viz`` itself is lazy).
"""

from __future__ import annotations

from typing import Any

from ._frames import check_overlay, force_requested, role_of
from .spec import Animation, Annotation, Layer, Layout, Legend, PlotKind, PlotSpec

__all__ = ["plot"]

#: Panel arrangements (``layout=``) that build a :data:`PlotKind.COMPOSITE`.
_COMPOSITE_MODES: frozenset[str] = frozenset({"stack", "row", "grid"})


def plot(
    *things: Any,
    layout: str = "overlay",
    animate: bool | dict[str, Any] | Animation = False,
    on: str | None = None,
    rows: int | None = None,
    cols: int | None = None,
    share_x: bool | None = None,
    share_y: bool | None = None,
    share_color: bool | None = None,
    **build_kw: Any,
) -> PlotSpec:
    """Compose one or more things into a single (possibly multi-panel) spec.

    Parameters
    ----------
    *things
        The things to plot — any mix of plottables (a
        :class:`~tsdynamics.data.Trajectory`, a system, an analysis result) and
        already-built :class:`~tsdynamics.viz.spec.PlotSpec` objects (including
        specs returned by an earlier ``plot`` call).  A single list/tuple argument is
        unwrapped, so ``plot([a, b])`` and ``plot(a, b)`` are equivalent.
    layout : {"overlay", "stack", "row", "grid"}, optional
        ``"overlay"`` (default) draws everything on one set of axes (a single
        panel); ``"stack"`` / ``"row"`` / ``"grid"`` give each thing its own panel
        in a :data:`~tsdynamics.viz.spec.PlotKind.COMPOSITE` figure.

        An overlay is legal when every thing draws in a **compatible frame** —
        the same coordinate space, the same dimension, the same axes (see
        :class:`tsdynamics.viz._frames.Frame`).  That is what lets a basin image,
        its attractors, a trajectory and the equilibria share one axes, and what
        refuses an ``(x, y)`` portrait under an ``(x, z)`` overlay.
    on : {"force"}, optional
        ``"force"`` overlays a deliberate frame mismatch anyway, warning once
        (:class:`~tsdynamics.viz.render.caps.VisualizationDegraded`) instead of
        raising.  Only meaningful for ``layout="overlay"``.

        .. versionadded:: 6.0
    animate : bool or dict or Animation, optional
        Animate the **whole figure**.  A composite plays every panel in lockstep on
        one shared clock (each panel keeps its own per-kind head default); an
        overlay animates the merged panel.  ``True`` uses defaults, a dict / an
        :class:`~tsdynamics.viz.spec.Animation` configures it.  Tweak further with
        the chainable ``.animate()`` / ``.trail()`` / … methods on the result.
    rows, cols : int, optional
        Explicit grid shape for ``layout="grid"``.  ``None`` (both) derives a
        near-square grid.  Ignored by ``"overlay"`` (one panel) and by
        ``"stack"`` / ``"row"`` (whose shape is fixed by the mode).

        .. versionadded:: 6.0
           :class:`~tsdynamics.viz.spec.Layout` always had these fields, but
           ``plot()`` had no way to set them, so a 4-panel grid was stuck on the
           auto-derived 2x2 and a 2x3 could not be asked for at all.
    share_x, share_y : bool, optional
        Force shared x / y axes across the panels.  ``None`` keeps the
        conservative auto-default (a *stack* of time-series panels naming the
        same x axis shares x; nothing else does).
    share_color : bool, optional
        Draw **one** figure-level colorbar rather than one per panel — the right
        presentation for a row of basin images across a parameter, where the
        per-panel colorbars repeat the same scale.  Default ``False``.
    **build_kw
        Forwarded to each non-spec thing's ``to_plot_spec`` (``components`` /
        ``kind`` / the per-kind options), so ``plot(a, b, components="x")``
        composes the same view of each.  Cannot be combined with an already-built
        ``PlotSpec`` argument.

    Returns
    -------
    PlotSpec
        A single-panel spec (overlay) or a ``COMPOSITE`` spec (panelled).  The
        result renders itself — ``.plot()`` / ``.save(...)`` / ``.render(...)``.

    Notes
    -----
    **Theme resolution order** for a composite figure (``layout="stack"`` /
    ``"row"`` / ``"grid"``): renderers resolve the theme per panel as
    ``panel.theme or composite.theme or get_theme(None)`` — the panel's own theme
    wins, then the composite-level theme (set via
    :meth:`~tsdynamics.viz.spec.PlotSpec.theme` on the returned spec), then
    the active global default.  An overlay (single panel) uses
    ``spec.theme or get_theme(None)`` directly.  To give every panel the same
    theme, call ``result.theme("dark")`` on the composite result; to style
    one panel differently, call ``.theme(...)`` on that panel before passing
    it to ``plot``.
    """
    from tsdynamics.errors import InvalidParameterError

    items = (
        list(things[0])
        if len(things) == 1 and isinstance(things[0], (list, tuple))
        else list(things)
    )
    if not items:
        raise InvalidParameterError("plot() needs at least one thing to plot.")

    specs = [_to_spec(item, build_kw) for item in items]
    layout_kw = {
        "rows": rows,
        "cols": cols,
        "share_x": share_x,
        "share_y": share_y,
        "share_color": share_color,
    }

    if layout == "overlay":
        _reject_layout_kw_for_overlay(layout_kw)
        result = _overlay(specs, on=on)
    elif layout in _COMPOSITE_MODES:
        if on is not None:
            raise InvalidParameterError(
                f"on={on!r} applies to layout='overlay' (one set of axes); panelled "
                "layouts draw each thing in its own frame, so there is nothing to force."
            )
        result = _composite(specs, layout, layout_kw)
    else:
        raise InvalidParameterError(
            f"unknown layout {layout!r}; use 'overlay', 'stack', 'row', or 'grid'."
        )
    if animate is not False and animate is not None:
        _apply_figure_animation(result, animate)
    return result


def _reject_layout_kw_for_overlay(layout_kw: dict[str, Any]) -> None:
    """Raise if a panel-arrangement keyword was passed to ``layout="overlay"``.

    An overlay is *one* set of axes, so a grid shape or a shared axis has no
    meaning there.  Accepting it silently would be the same class of defect this
    phase is closing everywhere else: the caller sees a plot and believes the
    keyword landed.
    """
    from tsdynamics.errors import InvalidParameterError

    given = sorted(k for k, v in layout_kw.items() if v is not None)
    if given:
        raise InvalidParameterError(
            f"{given} apply to a panelled figure, not to layout='overlay' (one set "
            "of axes); pass layout='stack' / 'row' / 'grid'."
        )


def _apply_figure_animation(result: PlotSpec, animate: bool | dict[str, Any] | Animation) -> None:
    """Stamp a figure-level animation: lockstep master on a composite, else the panel.

    A composite gets the master clock on itself and a per-panel animation (the
    master's timeline, with the head default following each panel's kind) on every
    panel that is not already animated.  A single-panel (overlay) result is
    animated directly with a head default following its kind.
    """
    from dataclasses import replace

    def _make(kind: PlotKind) -> Animation:
        head_default = kind != PlotKind.TIME_SERIES
        if isinstance(animate, Animation):
            # An explicit Animation wins wholesale — its own ``head`` (and every
            # other knob the user set) is honored verbatim; the per-kind head
            # default applies only to the bare-``True`` / dict spellings below.
            return animate
        if isinstance(animate, dict):
            return Animation(**{"head": head_default, **animate})
        return Animation(head=head_default)

    if result.is_composite:
        master = _make(PlotKind.COMPOSITE)
        result.animation = master
        for panel in result.panels:
            if panel.animation is None:
                panel.animation = replace(master, head=panel.kind != PlotKind.TIME_SERIES)
    else:
        result.animation = _make(result.kind)


def _to_spec(thing: Any, build_kw: dict[str, Any]) -> PlotSpec:
    """Convert one ``thing`` to a :class:`PlotSpec` (forwarding ``build_kw``)."""
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    if isinstance(thing, PlotSpec):
        if build_kw:
            raise InvalidParameterError(
                "build keywords (components=, kind=, …) cannot apply to an "
                "already-built PlotSpec; pass them when you first build it."
            )
        return thing
    to_plot_spec = getattr(thing, "to_plot_spec", None)
    if not callable(to_plot_spec):
        raise InvalidInputError(
            f"cannot plot a {type(thing).__name__}: it is not a Trajectory / system / "
            f"result / PlotSpec (no to_plot_spec())."
        )
    spec = to_plot_spec(**build_kw)
    if not isinstance(spec, PlotSpec):  # pragma: no cover - defensive
        raise InvalidInputError(
            f"{type(thing).__name__}.to_plot_spec() returned {type(spec).__name__}, not a PlotSpec."
        )
    return spec


# ---------------------------------------------------------------------------
# Overlay — many specs onto one set of axes (one panel)
# ---------------------------------------------------------------------------


def _overlay(specs: list[PlotSpec], *, on: str | None = None) -> PlotSpec:
    """Merge frame-compatible specs into one single-panel spec.

    Two policy decisions live here, and they are the whole of the v6 composability
    work:

    **Legality is frame identity, not kind identity.**  The old rule was a
    hard-coded three-member whitelist of :class:`PlotKind` values, which refused
    the flagship overlay (a basin image + its attractors + a trajectory + the
    equilibria are four *kinds* of one *plane*) while happily accepting an
    ``(x, y)`` portrait under an ``(x, z)`` fixed-point overlay — markers in the
    wrong place with nothing to say so.  Both are decided correctly by comparing
    :attr:`~tsdynamics.viz.spec.PlotSpec.resolved_frame`.

    **Z-order is by role, not by argument order.**  A field (image / quiver)
    draws under a curve, which draws under markers, because that is what those
    things *are* — so ``plot(basins, traj)`` and ``plot(traj, basins)`` produce
    the same picture.  The sort is stable, so specs of equal role keep their
    argument order (and an overlay of same-kind specs, the only kind that was
    legal before v6, is byte-identical to what it produced then).

    The **axis base** (axes, aspect, colorbar, clim, semantic kind) is the
    first spec in *role* order — the field owns the frame it is a picture of, so
    a basin image keeps its categorical colorbar and its ``basins_image`` kind
    when a trajectory is drawn over it.  The **figure context** (theme,
    animation) comes from the first spec in *argument* order: presentation is the
    caller's, z-order is the data's.
    """
    from dataclasses import replace

    if len(specs) == 1:
        return specs[0]

    frame = check_overlay(specs, force=force_requested(on))

    # Stable sort by role: field (0) < base (1) < overlay (2).
    order = sorted(range(len(specs)), key=lambda i: (int(role_of(specs[i])), i))
    base = specs[order[0]]
    context = specs[0]

    tags = _source_tags(specs)
    multi = len(specs) > 1
    layers: list[Layer] = []
    annotations: list[Annotation] = []
    composed: list[str] = []
    for i in order:
        # A spec that is *itself* an overlay already carries per-source labels
        # (and its own source list).  Re-tagging them would double-prefix every
        # legend entry, which is what made an incremental ``.add()`` chain
        # disagree with the equivalent one-shot ``plot(...)`` call.
        done = list(specs[i].meta.get("composed", ()))
        for layer in specs[i].layers:
            layers.append(
                _copy_layer(layer) if done else _relabel_for_overlay(layer, tags[i], multi=multi)
            )
        annotations.extend(replace(a) for a in specs[i].annotations)
        composed.extend(done or [tags[i]])

    # Deep-copy the carried-over presentation objects (Axis / Colorbar / Legend
    # are mutable dataclasses, and PlotSpec's in-place tweaks — relabel / rescale /
    # limits / ticks / grid — would otherwise rewrite the *first* input spec's
    # axes through the shared reference (spooky action at a distance).  The layers
    # are already freshly copied by ``_relabel_for_overlay``.
    return PlotSpec(
        kind=base.kind,
        ndim=base.ndim,
        aspect=base.aspect,
        x=replace(base.x) if base.x is not None else None,
        y=replace(base.y) if base.y is not None else None,
        z=replace(base.z) if base.z is not None else None,
        clim=base.clim,
        colorbar=replace(base.colorbar) if base.colorbar is not None else None,
        legend=Legend() if len(layers) > 1 else _copy_legend(base.legend),
        title=_common_title(specs),
        layers=layers,
        annotations=annotations,
        meta={**dict(base.meta), "composed": composed},
        animation=context.animation,
        _theme=context._theme,
        frame=frame,
    )


#: The :class:`PlotSpec` fields :func:`_merge_into` copies from a freshly merged
#: overlay onto an existing spec.  Deliberately **not** ``animation`` / ``_theme``
#: (the target already owns the figure context — it is the first argument of its
#: own ``add`` call) and not ``panels`` / ``layout`` (an overlay has neither).
_MERGE_FIELDS: tuple[str, ...] = (
    "kind",
    "layers",
    "x",
    "y",
    "z",
    "clim",
    "colorbar",
    "legend",
    "title",
    "ndim",
    "aspect",
    "annotations",
    "meta",
    "frame",
)


def _merge_into(target: PlotSpec, merged: PlotSpec) -> PlotSpec:
    """Write a merged overlay's fields back onto ``target`` in place.

    The engine behind :meth:`tsdynamics.viz.spec.PlotSpec.add`: ``add`` must
    mutate-and-return-``self`` like every other tweak (so it chains and so a
    reference held elsewhere sees the addition), while the merge itself is a
    pure function that builds a fresh spec.  This is the one place the two meet.
    """
    for name in _MERGE_FIELDS:
        setattr(target, name, getattr(merged, name))
    return target


def _copy_legend(legend: Legend | None) -> Legend | None:
    """Return a fresh copy of ``legend`` (``None`` passes through)."""
    from dataclasses import replace

    return replace(legend) if legend is not None else None


def _copy_layer(layer: Layer, *, label: str | None = None) -> Layer:
    """Shallow-copy a layer (channel arrays are shared, the containers are not).

    A merged spec must never alias an input's mutable ``Layer``: a later
    ``.style()`` / ``.recolor()`` on the composition would otherwise reach back
    and rewrite the spec the caller passed in.
    """
    return Layer(
        layer.kind,
        dict(layer.data),
        label=layer.label if label is None else label,
        style=dict(layer.style),
        transform=layer.transform,
    )


def _relabel_for_overlay(layer: Layer, tag: str, *, multi: bool) -> Layer:
    """Copy ``layer``, disambiguating its legend label by source ``tag``.

    A label that already *is* the tag is left alone — prefixing it would produce
    ``"streamlines: streamlines"``, which disambiguates nothing.
    """
    if not multi:
        return layer
    if not layer.label:
        return _copy_layer(layer, label=tag)
    if layer.label == tag:
        return _copy_layer(layer)
    return _copy_layer(layer, label=f"{tag}: {layer.label}")


def _source_tags(specs: list[PlotSpec]) -> list[str]:
    """Return a unique, human-readable tag per source spec.

    Preference order: the spec's own title, then — for a spec built by one plot
    transform — that transform's name, then a positional fallback.  The middle
    rung matters: overlaying a direction field, its nullclines and an orbit used
    to legend the nullclines as ``"series 2: v' = 0"``, where ``"nullclines:
    v' = 0"`` says the same thing and is true.
    """
    titles = [s.title or _transform_tag(s) or f"series {i + 1}" for i, s in enumerate(specs)]
    counts: dict[str, int] = {}
    tags: list[str] = []
    for title in titles:
        if titles.count(title) > 1:
            counts[title] = counts.get(title, 0) + 1
            tags.append(f"{title} ({counts[title]})")
        else:
            tags.append(title)
    return tags


def _transform_tag(spec: PlotSpec) -> str:
    """Return the producing transform's name, when every layer of ``spec`` agrees on one."""
    names = {layer.transform for layer in spec.layers if layer.transform}
    return names.pop() if len(names) == 1 else ""


def _common_title(specs: list[PlotSpec]) -> str:
    """Return the shared title if every source agrees, else empty.

    An untitled source is skipped — *unless* it is itself an overlay
    (``meta["composed"]``), whose empty title is the considered result of this
    same rule rather than an absence.  Counting it keeps an incremental
    ``.add()`` chain titled exactly like the one-shot ``plot(...)`` call.
    """
    titles = {s.title for s in specs if s.title or s.meta.get("composed")}
    return next(iter(titles)) if len(titles) == 1 else ""


# ---------------------------------------------------------------------------
# Composite — many specs into panels (one figure)
# ---------------------------------------------------------------------------


def _composite(specs: list[PlotSpec], mode: str, layout_kw: dict[str, Any]) -> PlotSpec:
    """Arrange specs into a ``COMPOSITE`` figure (one panel each; composites flattened).

    A child composite is flattened one level.  Flattening used to **discard** the
    child's ``_theme`` and ``animation``, so ``plot(plot(a, b).theme("dark"), c,
    layout="stack")`` lost the dark theme without a word; the child's context is
    now pushed down onto its own panels first (via
    :meth:`~tsdynamics.viz.spec.PlotSpec.resolved_panels`, the same inheritance
    the renderers apply).  Its ``layout`` genuinely cannot survive — a flat panel
    list has one arrangement — so the dropped modes are recorded in
    ``meta["flattened_layouts"]`` instead of vanishing.
    """
    panels: list[PlotSpec] = []
    dropped_layouts: list[str] = []
    for spec in specs:
        if spec.is_composite:
            # Push the child's figure-level context (theme / animation) onto its
            # panels before they are absorbed, then note the arrangement we lose.
            panels.extend(spec.resolved_panels())
            if spec.layout is not None and spec.layout.mode != mode:
                dropped_layouts.append(spec.layout.mode)
        else:
            panels.append(spec)
    if not panels:  # pragma: no cover - defensive (every spec had empty panels)
        from tsdynamics.errors import InvalidParameterError

        raise InvalidParameterError("nothing to arrange: no panels were produced.")

    # Auto-share the x axis only for a *stack* of time-series panels that name the
    # same x axis (the canonical "x1 & x2 over t, then y1 & y2 over t" case) — a
    # conservative default; arbitrary kinds / a row / grid keep independent axes.
    # An explicit ``share_x=`` / ``share_y=`` / ``share_color=`` overrides it.
    auto_share_x = (
        mode == "stack"
        and all(p.kind == PlotKind.TIME_SERIES for p in panels)
        and len({p.x.label for p in panels}) == 1
    )
    share_x = layout_kw["share_x"] if layout_kw["share_x"] is not None else auto_share_x
    layout = Layout(
        mode=mode,  # type: ignore[arg-type]
        rows=layout_kw["rows"],
        cols=layout_kw["cols"],
        share_x=bool(share_x),
        share_y=bool(layout_kw["share_y"]),
        share_color=bool(layout_kw["share_color"]),
    )
    meta: dict[str, Any] = {"n_panels": len(panels)}
    if dropped_layouts:
        meta["flattened_layouts"] = dropped_layouts
    return PlotSpec(
        kind=PlotKind.COMPOSITE,
        ndim=2,
        title=_common_title(panels),
        panels=panels,
        layout=layout,
        meta=meta,
    )
