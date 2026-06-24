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

The returned spec renders itself (notebook display, ``.plot()``, ``.save(...)``,
``.render(...)``); see :class:`tsdynamics.viz.spec.PlotSpec`.

This module imports **no plotting library** — it only builds the backend-agnostic
IR — so ``import tsdynamics`` stays plot-free (``tsdynamics.viz`` itself is lazy).
"""

from __future__ import annotations

from typing import Any

from .spec import Layer, Layout, Legend, PlotKind, PlotSpec

__all__ = ["plot"]

#: The semantic kinds that can share **one** set of axes (an overlay).  Image /
#: section / composite kinds each need their own axes + colorbar, so they are
#: arranged into panels (``layout="stack"`` / …) rather than overlaid.
_OVERLAYABLE: frozenset[PlotKind] = frozenset(
    {PlotKind.TIME_SERIES, PlotKind.PHASE_PORTRAIT_2D, PlotKind.PHASE_PORTRAIT_3D}
)

#: Panel arrangements (``layout=``) that build a :data:`PlotKind.COMPOSITE`.
_COMPOSITE_MODES: frozenset[str] = frozenset({"stack", "row", "grid"})


def plot(*things: Any, layout: str = "overlay", **build_kw: Any) -> PlotSpec:
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

    if layout == "overlay":
        return _overlay(specs)
    if layout in _COMPOSITE_MODES:
        return _composite(specs, layout)
    raise InvalidParameterError(
        f"unknown layout {layout!r}; use 'overlay', 'stack', 'row', or 'grid'."
    )


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


def _overlay(specs: list[PlotSpec]) -> PlotSpec:
    """Merge overlay-compatible specs into one single-panel spec."""
    from tsdynamics.errors import InvalidParameterError

    if len(specs) == 1:
        return specs[0]

    kinds = {s.kind for s in specs}
    if not kinds <= _OVERLAYABLE or len(kinds) != 1:
        raise InvalidParameterError(
            f"cannot overlay specs of kinds {sorted(k.value for k in kinds)} on one "
            "set of axes; use layout='stack' (or 'row' / 'grid') to give each its "
            "own panel."
        )

    base = specs[0]
    tags = _source_tags(specs)
    multi = len(specs) > 1
    layers: list[Layer] = []
    for tag, spec in zip(tags, specs, strict=True):
        for layer in spec.layers:
            layers.append(_relabel_for_overlay(layer, tag, multi=multi))

    return PlotSpec(
        kind=base.kind,
        ndim=base.ndim,
        aspect=base.aspect,
        x=base.x,
        y=base.y,
        z=base.z,
        clim=base.clim,
        colorbar=base.colorbar,
        legend=Legend() if len(layers) > 1 else base.legend,
        title=_common_title(specs),
        layers=layers,
        meta={**dict(base.meta), "composed": list(tags)},
    )


def _relabel_for_overlay(layer: Layer, tag: str, *, multi: bool) -> Layer:
    """Copy ``layer``, disambiguating its legend label by source ``tag``."""
    if not multi:
        return layer
    label = f"{tag}: {layer.label}" if layer.label else tag
    return Layer(layer.kind, dict(layer.data), label=label, style=dict(layer.style))


def _source_tags(specs: list[PlotSpec]) -> list[str]:
    """Return a unique, human-readable tag per source spec (title, made unique)."""
    titles = [s.title or f"series {i + 1}" for i, s in enumerate(specs)]
    counts: dict[str, int] = {}
    tags: list[str] = []
    for title in titles:
        if titles.count(title) > 1:
            counts[title] = counts.get(title, 0) + 1
            tags.append(f"{title} ({counts[title]})")
        else:
            tags.append(title)
    return tags


def _common_title(specs: list[PlotSpec]) -> str:
    """Return the shared title if every source agrees, else empty."""
    titles = {s.title for s in specs if s.title}
    return next(iter(titles)) if len(titles) == 1 else ""


# ---------------------------------------------------------------------------
# Composite — many specs into panels (one figure)
# ---------------------------------------------------------------------------


def _composite(specs: list[PlotSpec], mode: str) -> PlotSpec:
    """Arrange specs into a ``COMPOSITE`` figure (one panel each; composites flattened)."""
    panels: list[PlotSpec] = []
    for spec in specs:
        if spec.is_composite:
            panels.extend(spec.panels)  # flatten one level
        else:
            panels.append(spec)
    if not panels:  # pragma: no cover - defensive (every spec had empty panels)
        from tsdynamics.errors import InvalidParameterError

        raise InvalidParameterError("nothing to arrange: no panels were produced.")

    # Auto-share the x axis only for a *stack* of time-series panels that name the
    # same x axis (the canonical "x1 & x2 over t, then y1 & y2 over t" case) — a
    # conservative default; arbitrary kinds / a row / grid keep independent axes
    # (build a Layout by hand to override).
    share_x = (
        mode == "stack"
        and all(p.kind == PlotKind.TIME_SERIES for p in panels)
        and len({p.x.label for p in panels}) == 1
    )
    layout = Layout(mode=mode, share_x=share_x)  # type: ignore[arg-type]
    return PlotSpec(
        kind=PlotKind.COMPOSITE,
        ndim=2,
        title=_common_title(panels),
        panels=panels,
        layout=layout,
        meta={"n_panels": len(panels)},
    )
