"""The primitives — how geometry becomes layers.

A **primitive** consumes a :class:`~tsdynamics.viz.transforms._base.Part`'s
channels and emits :class:`~tsdynamics.viz.spec.Layer` objects.  It knows nothing
about which transform produced the numbers, which is what makes
``basins`` drawable as an ``image`` *or* a ``contour`` without ``basins``
knowing either exists.

**Zero new** :class:`~tsdynamics.viz.spec.PlotKind` **members.**  Every primitive
here lowers to the frozen 11-mark vocabulary — including ``contour`` and
``steps``, which lower to plain ``LINE`` layers (marching squares via
``contourpy``, a matplotlib hard dependency, and an explicit staircase
polyline).  That is the whole reason the compatibility matrix can grow without
touching a single renderer.

Registering one is a dict entry: build the record, add it to :data:`PRIMITIVES`.
Whether a *transform* may use it is a separate, declared decision — see
:class:`~tsdynamics.viz.transforms._base.PlotTransform.primitives`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._frames import FrameSpace
from ..spec import Layer, PlotKind
from ._base import Geometry, Part, Primitive

__all__ = ["PRIMITIVES", "RESERVED_PRIMITIVES", "get_primitive", "primitive_names"]


# ---------------------------------------------------------------------------
# Shared construction helpers
# ---------------------------------------------------------------------------


def _layer(
    geometry: Geometry,
    part: Part,
    mark: PlotKind,
    data: dict[str, np.ndarray],
    *,
    label: str | None = None,
    style: Mapping[str, Any] | None = None,
) -> Layer:
    """Build one :class:`Layer`, stamped with the producing transform's name.

    The ``transform`` stamp is the provenance that makes per-source restyling
    inside an overlay (``spec.style("basins", cmap=…)``) and legend grouping by
    source possible at all; every primitive goes through here so no layer can
    forget it.
    """
    return Layer(
        mark,
        data,
        label=part.label if label is None else label,
        style=dict(part.style if style is None else style),
        transform=geometry.transform,
    )


def _passthrough(
    mark: PlotKind,
    names: Sequence[str],
    *,
    requires: Sequence[str],
    frames: frozenset[FrameSpace] | None = None,
    doc: str = "",
) -> Primitive:
    """Build a primitive that copies the named channels straight into one layer.

    The common case: the transform already computed exactly the numbers the mark
    draws, so the "data-shaping rule" is the identity.  Channels in ``names``
    that the part does not carry are skipped (a ``LINE`` with no ``c`` channel is
    simply not colour-mapped), which is what lets one primitive serve a plain and
    a colour-by-time curve.
    """

    def build(geometry: Geometry, part: Part, options: Mapping[str, Any]) -> list[Layer]:
        data = {name: arr for name in names if (arr := part.array(name)) is not None}
        return [_layer(geometry, part, mark, data)]

    return Primitive(
        name="",  # filled in by the registry table below
        build=build,
        marks=frozenset({mark}),
        requires=frozenset(requires),
        frames=frames,
        doc=doc,
    )


def _named(name: str, primitive: Primitive) -> Primitive:
    """Return ``primitive`` with its registry ``name`` filled in."""
    from dataclasses import replace

    return replace(primitive, name=name)


def _xy(part: Part) -> tuple[np.ndarray, np.ndarray]:
    """Return the part's ``x`` / ``y`` channels as float arrays."""
    x = part.array("x")
    y = part.array("y")
    assert x is not None and y is not None  # noqa: S101 - guarded by `requires`
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# Reshaping primitives
# ---------------------------------------------------------------------------


def _build_steps(geometry: Geometry, part: Part, options: Mapping[str, Any]) -> list[Layer]:
    """Lower ``(x, y)`` to an explicit **post-step** staircase polyline (a ``LINE``).

    The staircase is materialised in the *data*, not delegated to a backend
    draw-style, so every backend — including the two that have no such option —
    draws the identical curve.  The natural primitive for a devil's staircase, a
    period ladder, or a piecewise-constant control parameter.
    """
    x, y = _xy(part)
    if x.size < 2:
        return [_layer(geometry, part, PlotKind.LINE, {"x": x, "y": y})]
    xs = np.repeat(x, 2)[1:]
    ys = np.repeat(y, 2)[:-1]
    return [_layer(geometry, part, PlotKind.LINE, {"x": xs, "y": ys})]


def _build_density(geometry: Geometry, part: Part, options: Mapping[str, Any]) -> list[Layer]:
    """Bin an ``(x, y)`` point cloud into an ``IMAGE`` — the anti-overdraw primitive.

    A 3.2-million-point orbit diagram drawn as markers is a black rectangle; the
    same points binned onto a grid are a bifurcation diagram.  Bins default to
    ``(400, 400)``; pass ``bins=(nx, ny)`` (or one int) to change it.
    """
    x, y = _xy(part)
    bins = options.get("bins", 400)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    image = counts.T  # (ny, nx) — rows are y, as an IMAGE expects
    centres_x = 0.5 * (xedges[:-1] + xedges[1:])
    centres_y = 0.5 * (yedges[:-1] + yedges[1:])
    return [
        _layer(
            geometry,
            part,
            PlotKind.IMAGE,
            {"x": centres_x, "y": centres_y, "z": image, "c": image.ravel()},
        )
    ]


def _build_contour(geometry: Geometry, part: Part, options: Mapping[str, Any]) -> list[Layer]:
    """Lower a 2-D ``z`` field to level sets — **N plain ``LINE`` layers**.

    Marching squares via ``contourpy`` (a matplotlib hard dependency, so no new
    requirement), then each polyline becomes an ordinary ``LINE``.  That is why a
    contour draws on **all four** backends on day one — including three.js, which
    a new ``CONTOUR`` mark never would.

    Every polyline is coloured **by its level**, through the transform's declared
    colormap, because the alternative is worse than it sounds: N unstyled layers
    take N successive colours from the theme palette, so eight level sets of one
    scalar field arrive as a rainbow that reads as eight unrelated series.  The
    colour is baked into the layer style (a resolved hex), so it survives JSON and
    three.js export, and an explicit ``color=`` from the caller still wins — the
    style merge is applied after lowering.

    Options: ``levels`` — an int (that many evenly-spaced levels, the default 8)
    or an explicit sequence of level values.
    """
    from contourpy import contour_generator

    z = np.asarray(part.array("z"), dtype=float)
    x = part.array("x")
    y = part.array("y")
    xs = np.arange(z.shape[1], dtype=float) if x is None else np.asarray(x, dtype=float)
    ys = np.arange(z.shape[0], dtype=float) if y is None else np.asarray(y, dtype=float)
    gx, gy = np.meshgrid(xs, ys)

    levels = options.get("levels", 8)
    finite = z[np.isfinite(z)]
    if isinstance(levels, (int, np.integer)):
        if finite.size == 0 or float(finite.min()) == float(finite.max()):
            values: list[float] = [float(finite[0])] if finite.size else [0.0]
        else:
            values = list(
                np.linspace(float(finite.min()), float(finite.max()), int(levels) + 2)[1:-1]
            )
    else:
        values = [float(v) for v in levels]

    gen = contour_generator(gx, gy, z)
    colours = _level_colours(geometry, values)
    layers: list[Layer] = []
    for level, colour in zip(values, colours, strict=True):
        style = dict(part.style)
        if colour is not None:
            style.setdefault("color", colour)
        for line in gen.lines(level):
            pts = np.asarray(line, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            # Only the first polyline carries the label: N level sets of one field
            # are one legend entry, not N.
            layers.append(
                _layer(
                    geometry,
                    part,
                    PlotKind.LINE,
                    {"x": pts[:, 0], "y": pts[:, 1]},
                    label=part.label if not layers else None,
                    style=style,
                )
            )
    return layers


def _level_colours(geometry: Geometry, values: Sequence[float]) -> list[str | None]:
    """One colour per contour level, sampled from the transform's colormap.

    Falls back to ``[None] * n`` — i.e. leave the layers unstyled, the previous
    behaviour — when matplotlib is absent, since resolving a named colormap to
    hex needs it and ``viz`` must not acquire a hard plotting dependency.  A
    single level is drawn in the map's midpoint rather than its darkest end.
    """
    n = len(values)
    try:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex
    except ImportError:  # pragma: no cover - matplotlib is a viz extra
        return [None] * n
    from ._registry import get

    try:
        name = get(geometry.transform).presentation.cmap or "viridis"
        cmap = colormaps[name]
    except Exception:  # noqa: BLE001 # pragma: no cover - unknown transform/colormap
        cmap = colormaps["viridis"]
    positions = [0.5] if n == 1 else [i / (n - 1) for i in range(n)]
    # Trim the extremes: the ends of a sequential map are near-white or near-black,
    # which on a transparent page is either invisible or indistinguishable.
    return [str(to_hex(cmap(0.12 + 0.76 * p))) for p in positions]


def _build_boundary(geometry: Geometry, part: Part, options: Mapping[str, Any]) -> list[Layer]:
    """Draw only the *boundaries* between labelled cells of an integer field.

    The one contour that is not a level set: category labels have no meaningful
    intermediate value, so the boundary is where a cell differs from its
    neighbour, not where an interpolant crosses a threshold.  Exclusive to
    ``basins`` (a boundary between what, otherwise?).
    """
    z = np.asarray(part.array("z"), dtype=float)
    x = part.array("x")
    y = part.array("y")
    xs = np.arange(z.shape[1], dtype=float) if x is None else np.asarray(x, dtype=float)
    ys = np.arange(z.shape[0], dtype=float) if y is None else np.asarray(y, dtype=float)
    edge = np.zeros(z.shape, dtype=bool)
    edge[:, :-1] |= z[:, :-1] != z[:, 1:]
    edge[:-1, :] |= z[:-1, :] != z[1:, :]
    rows, cols = np.nonzero(edge)
    return [
        _layer(
            geometry,
            part,
            PlotKind.SCATTER,
            {"x": xs[cols], "y": ys[rows]},
            style={**dict(part.style), "markersize": part.style.get("markersize", 1.0)},
        )
    ]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

#: name → :class:`~tsdynamics.viz.transforms._base.Primitive`.  Nineteen ways of
#: drawing, all lowering to the frozen 11-mark vocabulary.
PRIMITIVES: dict[str, Primitive] = {}


def _register(name: str, primitive: Primitive) -> None:
    """Insert ``primitive`` into :data:`PRIMITIVES` under ``name``."""
    PRIMITIVES[name] = _named(name, primitive)


_register(
    "line",
    _passthrough(
        PlotKind.LINE,
        ("x", "y", "c", "frames"),
        requires=("x", "y"),
        doc="A connected curve; a 'c' channel colours it per-vertex.",
    ),
)
_register(
    "line3d",
    _passthrough(
        PlotKind.LINE3D,
        ("x", "y", "z", "c"),
        requires=("x", "y", "z"),
        doc="A connected space curve.",
    ),
)
_register(
    "points",
    _passthrough(
        PlotKind.SCATTER,
        ("x", "y", "c", "size"),
        requires=("x", "y"),
        doc="An unconnected point cloud — a map orbit, a section, a sample.",
    ),
)
_register(
    "points3d",
    _passthrough(
        PlotKind.SCATTER,
        ("x", "y", "z", "c", "size"),
        requires=("x", "y", "z"),
        doc="A 3-D point cloud (a SCATTER carrying z; the renderers dispatch on ndim).",
    ),
)
_register(
    "markers",
    _passthrough(
        PlotKind.MARKERS,
        ("x", "y", "z", "size"),
        requires=("x", "y"),
        doc="Annotation glyphs — equilibria, tipping points, marked values.",
    ),
)
_register(
    "image",
    _passthrough(
        PlotKind.IMAGE,
        ("x", "y", "z", "c", "frames"),
        requires=("z",),
        doc="A colour-mapped lattice; 1-D x/y set the pixel-edge extent.",
    ),
)
_register(
    "surface3d",
    _passthrough(
        PlotKind.SURFACE3D,
        ("x", "y", "z"),
        requires=("x", "y", "z"),
        doc="A 2-D field drawn as a 3-D surface.",
    ),
)
_register(
    "quiver",
    _passthrough(
        PlotKind.QUIVER,
        ("x", "y", "u", "v", "c"),
        requires=("x", "y", "u", "v"),
        frames=frozenset({FrameSpace.STATE2}),
        doc="Arrows (u, v) at positions (x, y) — a vector / direction field.",
    ),
)
_register(
    "bars",
    _passthrough(
        PlotKind.BAR,
        ("x", "y", "cat"),
        requires=("y",),
        doc="One bar per category — a Lyapunov spectrum, basin fractions, RQA measures.",
    ),
)
_register(
    "histogram",
    _passthrough(
        PlotKind.HISTOGRAM,
        ("x", "y"),
        requires=("x",),
        doc="A distribution — pre-binned (x = centres, y = counts) or raw samples (x only).",
    ),
)
_register(
    "errorbars",
    _passthrough(
        PlotKind.ERRORBAR,
        ("x", "y", "err"),
        requires=("x", "y", "err"),
        doc="A curve with symmetric uncertainties — D(q), a scaling fit.",
    ),
)
_register(
    "band",
    _passthrough(
        PlotKind.AREA,
        ("x", "y", "lo", "hi"),
        requires=("x",),
        doc="A shaded lo <= hi envelope — an ensemble fan, a continuation band.",
    ),
)
_register(
    "steps",
    Primitive(
        name="",
        build=_build_steps,
        marks=frozenset({PlotKind.LINE}),
        requires=frozenset({"x", "y"}),
        doc="A piecewise-constant staircase, materialised as a polyline.",
    ),
)
_register(
    "density",
    Primitive(
        name="",
        build=_build_density,
        marks=frozenset({PlotKind.IMAGE}),
        requires=frozenset({"x", "y"}),
        options=frozenset({"bins"}),
        doc="A binned point cloud drawn as an image — the cure for a million-point overdraw.",
    ),
)
_register(
    "contour",
    Primitive(
        name="",
        build=_build_contour,
        marks=frozenset({PlotKind.LINE}),
        requires=frozenset({"z"}),
        options=frozenset({"levels"}),
        doc="Level sets of a 2-D field, lowered to plain lines (marching squares).",
    ),
)
_register(
    "boundary",
    Primitive(
        name="",
        build=_build_boundary,
        marks=frozenset({PlotKind.SCATTER}),
        requires=frozenset({"z"}),
        doc="The cells where a categorical field changes label — a basin boundary.",
    ),
)


#: Primitives that exist as building blocks but are claimed by **no** transform
#: row yet, with the reason.  The governance gate asserts that
#: ``PRIMITIVES == (every declared row) | RESERVED_PRIMITIVES``, so a primitive
#: can never sit in the library unaccounted for — either a transform advertises
#: it (and the compatibility gate renders it), or it is listed here on purpose
#: and smoke-tested directly by ``tests/test_viz_transforms.py``.
#:
#: This is deliberately *not* the same thing as advertising a row that cannot
#: draw: nothing here appears in any transform's declared row, so no caller is
#: ever told these are available for a plot they cannot get.
RESERVED_PRIMITIVES: dict[str, str] = {
    "bars": "claimed by the result-class transforms (lyapunov_spectrum, basin fractions, RQA)",
    "errorbars": "claimed by the scaling-fit and D(q) transforms",
    "band": "claimed by the ensemble-fan and continuation transforms",
    "markers": "claimed by the fixed-point / tipping-point overlays",
    "boundary": "exclusive to the basins transform",
}


def primitive_names() -> tuple[str, ...]:
    """Return every registered primitive name, sorted."""
    return tuple(sorted(PRIMITIVES))


def get_primitive(name: str) -> Primitive:
    """Return the :class:`Primitive` registered as ``name``.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If no primitive of that name exists — with the full list, because a typo
        here is otherwise indistinguishable from a primitive that is valid
        somewhere else.
    """
    from tsdynamics.errors import InvalidParameterError

    try:
        return PRIMITIVES[name]
    except KeyError:
        raise InvalidParameterError(
            f"unknown primitive {name!r}; the registered primitives are {list(primitive_names())}."
        ) from None
