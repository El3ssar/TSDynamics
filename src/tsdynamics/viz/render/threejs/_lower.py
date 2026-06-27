"""Geometry lowering for the ``threejs`` data-export backend (stream VIZ-THREEJS-EXPORT).

This module turns a backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into a
**three.js BufferGeometry-ready** JSON-able payload — a pure-Python dict of plain
floats / ints / strings, no optional dependency and no plotting library import.
A web frontend reads the payload and builds ``THREE.BufferGeometry`` objects
directly: the flat :data:`positions` list is a ``Float32Array`` source
(``x, y, z`` interleaved), the optional flat :data:`colors` list a per-vertex RGB
``Float32Array``, and the :data:`indices` list a draw-index buffer.

Schema
------
The payload is a JSON object::

    {
        "schema_version": <int>,            # tsdynamics.viz.export.SCHEMA_VERSION
        "kind": "<PlotSpec.kind value>",    # the semantic spec kind
        "title": "<plot title>",
        "geometries": [ <geometry>, ... ],  # one per drawable layer
        "metadata": {
            "schema_version": <int>,
            "labels": {"x": "<str>", "y": "<str>", "z": "<str>"},
            "units": {"x": "<str>", "y": "<str>", "z": "<str>"},
            "bounds": {
                "x": [<min>, <max>], "y": [<min>, <max>], "z": [<min>, <max>]
            },
            "camera": {
                "position": [<x>, <y>, <z>],
                "target":   [<x>, <y>, <z>],
                "up":       [<x>, <y>, <z>]
            },
            "theme": {                           # ALWAYS present; resolved Theme
                "background": "<str | null>",    # scene background color
                "foreground": "<str | null>",    # default ink color
                "palette": ["<str>", ...]        # color cycle for auto-colored layers
            },
            "animation": {                       # ONLY when spec.animation is set
                "fps": <float>,
                "duration": <float | null>,
                "n_frames": <int | null>,
                "loop": <bool>,
                "pingpong": <bool>,
                "trail_length_samples": <int | null>,   # null ⇒ persistent trail
                "head": <bool>,
                "head_size": <float>,
                "head_color": [<r>, <g>, <b>] | null,
                "n_samples": <int>               # vertices on the longest animated line
            }
        }
    }

A static (non-animated) spec carries **no** ``animation`` key in ``metadata`` —
the export is byte-for-byte the pre-animation payload.  When ``spec.animation`` is
present the geometry buffers are unchanged: the loader animates by **draw-range**
(``geometry.setDrawRange``) over a faint full-curve backdrop, so no positions /
colors are re-uploaded and no per-vertex time attribute is needed (the line
vertices are already the natural reveal order).

Each geometry now carries a ``material`` block with the layer's style vocabulary
keys that the three.js backend honors::

    "material": {
        "color":       "<CSS string | null>",   # explicit layer color
        "linewidth":   <float | null>,          # line width (pt)
        "markersize":  <float | null>,          # point size (pt)
        "alpha":       <float | null>,          # 0..1 opacity
        "cmap":        "<str | null>",          # colormap name for per-vertex colors
        "zorder":      <int | null>             # maps to THREE renderOrder
    }

Keys are ``null`` when not set by the user (the loader applies its own default).
``linestyle`` and marker *shape* are **not** in the material block — they are
excluded from the threejs backend's ``honored_by`` set (see
:data:`~tsdynamics.viz.style.STYLE_KEYS`) and the loader ignores them.

Each ``geometry`` is::

    {
        "type": "line" | "points" | "surface",
        "label": "<layer label or null>",
        "positions": [x0, y0, z0, x1, y1, z1, ...],   # FLAT, plain floats
        "indices":   [...],                            # FLAT, plain ints (or [])
        "colors":    [r0, g0, b0, r1, g1, b1, ...]     # FLAT floats, optional
    }

- A 3-D ``LINE3D`` / a 2-D ``LINE`` (lifted to ``z = 0``) → a ``"line"`` geometry
  whose ``indices`` are consecutive segment endpoints ``0, 1, 1, 2, 2, 3, ...``
  (``THREE.LineSegments`` order).
- A ``SCATTER`` / ``MARKERS`` (2-D lifted to ``z = 0``, or 3-D) → a ``"points"``
  geometry (no ``indices``).
- A ``SURFACE3D`` → a ``"surface"`` geometry whose ``indices`` triangulate the
  grid (two triangles per quad).

The optional ``colors`` come from the layer's ``"c"`` channel mapped through a
small built-in colormap (so no matplotlib import), or from an explicit per-vertex
RGB ``style["color"]``.  ``bounds`` are the per-axis ``[min, max]`` over every
geometry's vertices; the ``camera`` is derived from those bounds (a corner view
looking at the centre) unless ``spec.meta["camera"]`` overrides it.

Composite (multi-panel) payloads
--------------------------------
A :data:`~tsdynamics.viz.spec.PlotKind.COMPOSITE` spec carries its drawable
content in :attr:`~tsdynamics.viz.spec.PlotSpec.panels` (one sub-spec per panel)
plus a :class:`~tsdynamics.viz.spec.Layout`, not in its top-level ``layers``.
:func:`lower_spec` detects a composite and lowers **each panel recursively**,
emitting a ``"panels"`` list instead of a single ``"geometries"`` block::

    {
        "schema_version": <int>,
        "kind": "composite",
        "title": "<plot title>",
        "geometries": [],                   # always empty for a composite
        "panels": [ <panel>, ... ],         # one per child panel
        "metadata": {
            "schema_version": <int>,
            "layout": {                     # the Layout, plus the resolved grid
                "mode": "stack" | "row" | "grid",
                "rows": <int>, "cols": <int>,
                "share_x": <bool>, "share_y": <bool>
            },
            "bounds": { ... },              # union over every (placed) panel
            "camera": { ... }               # framing that whole placed scene
        }
    }

Each ``panel`` is a single-panel payload (the same ``geometries`` / per-panel
``metadata`` a non-composite spec produces) plus its **identity and placement**::

    {
        "index": <int>,                     # panel order (0-based)
        "title": "<panel title>",
        "kind": "<panel PlotSpec.kind value>",
        "grid": {"row": <int>, "col": <int>},   # cell in the layout grid
        "offset": [<x>, <y>, <z>],          # local-origin translation (see below)
        "geometries": [ <geometry>, ... ],
        "metadata": { ... }                 # the panel's own labels/units/bounds/camera
    }

The panel's geometry ``positions`` stay in the panel's **own** local coordinates
(unshifted), so a frontend can render each panel into its own viewport
untouched.  The separate ``offset`` is a convenience translation — each panel's
local-bounds centre laid out on the resolved ``rows`` × ``cols`` grid with unit
cell spacing (column → +x, row → −y, so row 0 is at the top) — for a frontend
that prefers to drop every panel into **one** shared scene rather than tile
viewports.  Either reading is valid: the panel ``grid`` cell and ``offset`` are
redundant placement hints, and the geometry itself is never mutated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...export import SCHEMA_VERSION
from ...spec import PlotKind
from ...style import normalize_style

if TYPE_CHECKING:
    from ...spec import Animation, Axis, Layer, Layout, PlotSpec
    from ...style import Theme

__all__ = ["lower_spec"]

#: The geometry mark a layer's :class:`~tsdynamics.viz.spec.PlotKind` lowers to.
#: 2-D ``LINE`` / ``SCATTER`` / ``MARKERS`` are lifted to ``z = 0`` and keep their
#: line / points semantics; the 3-D marks map to their direct three.js type.
_GEOMETRY_TYPE: dict[PlotKind, str] = {
    PlotKind.LINE: "line",
    PlotKind.LINE3D: "line",
    PlotKind.SCATTER: "points",
    PlotKind.MARKERS: "points",
    PlotKind.SURFACE3D: "surface",
}

#: Layer marks the reveal animation drives — the **line** marks whose vertices
#: are a natural sweep order and whose index buffer (``0,1,1,2,...``) the loader
#: reveals by ``setDrawRange``.  ``SCATTER`` / ``MARKERS`` (a ``points`` geometry
#: with no index buffer) and ``SURFACE3D`` (a mesh) have **no** comet reveal in
#: the reference loader, so they are excluded: a points-only / surface-only spec
#: emits no animation block (and :func:`_animation_metadata` warns) rather than a
#: block the loader cannot play (which would leave the export static *and* freeze
#: the camera).  Mirror this set in the loader's ``geom.type === "line"`` guard.
_ANIMATED_MARKS: frozenset[PlotKind] = frozenset({PlotKind.LINE, PlotKind.LINE3D})


def lower_spec(spec: PlotSpec) -> dict[str, Any]:
    """Lower ``spec`` to a three.js BufferGeometry-ready JSON-able payload.

    For a **single-panel** spec, walks the spec's drawable layers, lowering each
    one whose mark is a line / points / surface (other marks — images, bars,
    quivers — have no BufferGeometry analogue and are skipped) into a
    flat-positions geometry, then derives the top-level ``metadata`` (labels /
    units / bounds / camera) from the axes and the lowered vertices.  When
    ``spec.animation`` is set, an ``animation`` block is added to ``metadata``
    (the reveal directive — fps / duration / trail length in samples / head) so
    the reference loader plays a comet reveal via ``geometry.setDrawRange``; the
    geometry buffers are unchanged, and a static spec's payload is byte-for-byte
    the pre-animation one.

    For a **composite** spec (``PlotKind.COMPOSITE`` — its drawable content lives
    in :attr:`~tsdynamics.viz.spec.PlotSpec.panels`, not the top-level layers),
    lowers each panel recursively and emits a ``"panels"`` list of per-panel
    payloads (each carrying its identity, grid placement, and a layout-offset)
    plus a top-level ``layout`` block — see the module docstring for the schema.

    Parameters
    ----------
    spec : PlotSpec
        The spec to lower.

    Returns
    -------
    dict
        A JSON-serializable payload conforming to the module-docstring schema
        (every value is a plain ``str`` / ``int`` / ``float`` / ``list``).
    """
    if spec.is_composite:
        return _lower_composite(spec)
    return _lower_single(spec)


def _lower_single(spec: PlotSpec) -> dict[str, Any]:
    """Lower a single-panel spec to the ``geometries`` + ``metadata`` payload.

    The non-composite lowering: walk ``spec.layers``, lower each drawable mark,
    and derive the per-spec ``metadata`` (labels / units / bounds / camera /
    theme).  Kept as a stable helper so the composite path can reuse it per
    panel.
    """
    # Resolve the theme once (either the spec's own or the global default).

    theme = spec.resolved_theme

    geometries: list[dict[str, Any]] = []
    for i, layer in enumerate(spec.layers):
        palette_color = theme.palette[i % len(theme.palette)] if theme.palette else None
        geom = _lower_layer(layer, palette_color=palette_color)
        if geom is not None:
            geometries.append(geom)

    bounds = _bounds(geometries)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "labels": _axis_labels(spec),
        "units": _axis_units(spec),
        "bounds": bounds,
        "camera": _camera(spec, bounds),
        "theme": _theme_metadata(theme),
    }
    animation = _animation_metadata(spec)
    if animation is not None:
        metadata["animation"] = animation
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": spec.kind.value,
        "title": spec.title,
        "geometries": geometries,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# composite (multi-panel) → panel groups
# ---------------------------------------------------------------------------


def _lower_composite(spec: PlotSpec) -> dict[str, Any]:
    """Lower a ``COMPOSITE`` spec to a panelled payload.

    Lowers each child panel via :func:`_lower_single`, places it on the resolved
    ``rows`` × ``cols`` grid (per the :class:`~tsdynamics.viz.spec.Layout`), tags
    it with its identity (index / title / kind) + grid cell + a layout-offset, and
    aggregates the per-panel bounds (each shifted by its offset) into a top-level
    ``bounds`` / ``camera`` that frames the whole laid-out scene.
    """
    panels_in = spec.panels
    rows, cols = _composite_grid(spec.layout, len(panels_in))

    panels_out: list[dict[str, Any]] = []
    placed_geometries: list[dict[str, Any]] = []
    for i, panel in enumerate(panels_in):
        sub = _lower_single(panel)
        row, col = divmod(i, cols) if cols else (i, 0)
        offset = _panel_offset(sub["metadata"]["bounds"], row, col)
        panels_out.append(
            {
                "index": i,
                "title": panel.title,
                "kind": panel.kind.value,
                "grid": {"row": row, "col": col},
                "offset": offset,
                "geometries": sub["geometries"],
                "metadata": sub["metadata"],
            }
        )
        # A bounds-only copy of each geometry, shifted by the panel's offset, so
        # the aggregate camera frames the laid-out scene (positions stay local).
        for geom in sub["geometries"]:
            placed_geometries.append({"positions": _shift_positions(geom["positions"], offset)})

    bounds = _bounds(placed_geometries)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "layout": _layout_dict(spec.layout, rows, cols),
        "bounds": bounds,
        "camera": _camera(spec, bounds),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": spec.kind.value,
        "title": spec.title,
        "geometries": [],
        "panels": panels_out,
        "metadata": metadata,
    }


def _composite_grid(layout: Layout | None, n: int) -> tuple[int, int]:
    """Return the ``(rows, cols)`` grid for a composite's :class:`Layout`.

    Mirrors the matplotlib renderer's tiling: ``"row"`` → one row, ``"grid"`` →
    the explicit ``rows`` × ``cols`` (or a near-square fit when unset), and
    ``"stack"`` (the default) → one column.  ``n == 0`` yields ``(0, 0)``.
    """
    if n <= 0:
        return 0, 0
    mode = getattr(layout, "mode", "stack")
    if mode == "row":
        return 1, n
    if mode == "grid":
        rows = getattr(layout, "rows", None)
        cols = getattr(layout, "cols", None)
        if rows and cols:
            return int(rows), int(cols)
        c = int(np.ceil(np.sqrt(n)))
        r = int(np.ceil(n / c))
        return r, c
    return n, 1  # "stack" (default): one column


def _panel_offset(bounds: dict[str, list[float]], row: int, col: int) -> list[float]:
    """Local-origin translation placing a panel at grid cell ``(row, col)``.

    Each panel is centred on its own bounds, then translated so that adjacent
    cells sit one (max panel span) apart: column → +x, row → −y (row 0 on top),
    z untouched.  The span scale keeps panels from overlapping regardless of their
    individual extents.
    """
    span_x = bounds["x"][1] - bounds["x"][0]
    span_y = bounds["y"][1] - bounds["y"][0]
    span = max(span_x, span_y, 1.0)
    pitch = span * 1.2  # a small gutter between cells
    cx = (bounds["x"][0] + bounds["x"][1]) / 2.0
    cy = (bounds["y"][0] + bounds["y"][1]) / 2.0
    return [col * pitch - cx, -row * pitch - cy, 0.0]


def _layout_dict(layout: Layout | None, rows: int, cols: int) -> dict[str, Any]:
    """Serialize the composite layout (its mode/share flags + the resolved grid)."""
    return {
        "mode": getattr(layout, "mode", "stack"),
        "rows": int(rows),
        "cols": int(cols),
        "share_x": bool(getattr(layout, "share_x", False)),
        "share_y": bool(getattr(layout, "share_y", False)),
    }


def _shift_positions(positions: list[float], offset: list[float]) -> list[float]:
    """Translate a flat ``[x0, y0, z0, ...]`` list by ``offset`` ``[dx, dy, dz]``."""
    ox, oy, oz = offset
    shifted = list(positions)
    shifted[0::3] = [v + ox for v in positions[0::3]]
    shifted[1::3] = [v + oy for v in positions[1::3]]
    shifted[2::3] = [v + oz for v in positions[2::3]]
    return shifted


# ---------------------------------------------------------------------------
# layer → geometry
# ---------------------------------------------------------------------------


def _lower_layer(layer: Layer, *, palette_color: str | None = None) -> dict[str, Any] | None:
    """Lower one :class:`~tsdynamics.viz.spec.Layer` to a geometry, or ``None``.

    Returns ``None`` for a mark with no BufferGeometry analogue (an image, bar,
    histogram, quiver, …) so the caller drops it from the payload.

    Parameters
    ----------
    layer : Layer
        The layer to lower.
    palette_color : str, optional
        The auto-color from the theme palette for this layer's position in the
        spec's layer list.  Used as the fallback color when the layer carries no
        explicit ``style["color"]`` and no per-vertex ``"c"`` channel.
    """
    geom_type = _GEOMETRY_TYPE.get(layer.kind)
    if geom_type is None:
        return None

    if geom_type == "surface":
        return _lower_surface(layer, palette_color=palette_color)
    return _lower_line_or_points(layer, geom_type, palette_color=palette_color)


def _lower_line_or_points(
    layer: Layer, geom_type: str, *, palette_color: str | None = None
) -> dict[str, Any] | None:
    """Lower a line / points layer to a flat-positions geometry.

    A 2-D layer (no ``"z"`` channel) is lifted to ``z = 0``.  A ``"line"`` gets a
    consecutive-segment-endpoint ``indices`` list; ``"points"`` gets an empty one.
    The geometry carries a ``material`` block (the three.js-honored style keys) and
    the auto-palette color is used when the layer has no explicit color or per-vertex
    ``"c"`` channel.
    """
    x = _flat(layer.data.get("x"))
    y = _flat(layer.data.get("y"))
    if x is None or y is None:
        return None
    n = min(x.size, y.size)
    if n == 0:
        return None
    x = x[:n]
    y = y[:n]
    z_arr = _flat(layer.data.get("z"))
    z = z_arr[:n] if z_arr is not None and z_arr.size >= n else np.zeros(n, dtype=float)

    positions = _interleave(x, y, z)
    indices = _line_indices(n) if geom_type == "line" else []
    geometry: dict[str, Any] = {
        "type": geom_type,
        "label": layer.label,
        "positions": positions,
        "indices": indices,
        "material": _material_style(layer, palette_color=palette_color),
    }
    colors = _colors(layer, n)
    if colors is not None:
        geometry["colors"] = colors
    return geometry


def _lower_surface(layer: Layer, *, palette_color: str | None = None) -> dict[str, Any] | None:
    """Lower a ``SURFACE3D`` layer to a triangulated-grid geometry.

    Expects the ``"x"`` / ``"y"`` / ``"z"`` channels as 2-D grids of identical
    shape (rows × cols).  Emits row-major interleaved vertex positions and an
    index list of two triangles per grid quad (``THREE.Mesh`` / ``BufferGeometry``
    order).  The geometry carries a ``material`` block with the three.js-honored
    style keys.
    """
    x = _grid(layer.data.get("x"))
    y = _grid(layer.data.get("y"))
    z = _grid(layer.data.get("z"))
    if x is None or y is None or z is None:
        return None
    if not (x.shape == y.shape == z.shape) or x.ndim != 2:
        return None
    rows, cols = x.shape
    if rows < 2 or cols < 2:
        return None

    positions = _interleave(x.reshape(-1), y.reshape(-1), z.reshape(-1))
    indices = _surface_indices(rows, cols)
    geometry: dict[str, Any] = {
        "type": "surface",
        "label": layer.label,
        "positions": positions,
        "indices": indices,
        "material": _material_style(layer, palette_color=palette_color),
    }
    c = _grid(layer.data.get("c"))
    cflat = c.reshape(-1) if c is not None and c.shape == z.shape else z.reshape(-1)
    colors = _scalar_colors(cflat)
    if colors is not None:
        geometry["colors"] = colors
    return geometry


# ---------------------------------------------------------------------------
# index buffers
# ---------------------------------------------------------------------------


def _line_indices(n: int) -> list[int]:
    """Segment-endpoint indices ``0, 1, 1, 2, ..., n-2, n-1`` for ``n`` vertices."""
    if n < 2:
        return []
    indices: list[int] = []
    for i in range(n - 1):
        indices.append(i)
        indices.append(i + 1)
    return indices


def _surface_indices(rows: int, cols: int) -> list[int]:
    """Two-triangles-per-quad index list over a ``rows`` × ``cols`` vertex grid.

    Vertices are addressed row-major (``r * cols + c``).  Each quad
    ``(r, c)``–``(r+1, c+1)`` becomes triangles ``(v00, v10, v11)`` and
    ``(v00, v11, v01)`` (consistent winding).
    """
    indices: list[int] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = r * cols + c
            v01 = v00 + 1
            v10 = v00 + cols
            v11 = v10 + 1
            indices.extend((v00, v10, v11, v00, v11, v01))
    return indices


# ---------------------------------------------------------------------------
# colors
# ---------------------------------------------------------------------------


def _colors(layer: Layer, n: int) -> list[float] | None:
    """Per-vertex flat RGB for a line / points layer's scalar ``"c"`` channel.

    Returns the per-vertex buffer **only** for a genuine ``"c"`` gradient (mapped
    through the built-in colormap).  A *solid* color — an explicit
    ``style["color"]`` or the theme-palette auto-color — is carried once by the
    layer ``material.color`` (see :func:`_material_style`); baking it into a
    redundant per-vertex array would just repeat the same RGB for every vertex,
    so the function returns ``None`` and lets the flat ``material.color`` stand.
    """
    c = _flat(layer.data.get("c"))
    if c is not None and c.size >= n:
        return _scalar_colors(c[:n])
    return None


def _scalar_colors(values: np.ndarray) -> list[float] | None:
    """Map a scalar field to a flat per-vertex RGB list via the built-in colormap.

    Normalizes ``values`` to ``[0, 1]`` over its finite extent (a constant field
    maps to the colormap midpoint) and looks each up in :func:`_colormap`.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    lo = float(finite.min())
    hi = float(finite.max())
    span = hi - lo
    flat: list[float] = []
    for v in arr:
        if not np.isfinite(v):
            t = 0.0
        elif span == 0.0:
            t = 0.5
        else:
            t = (float(v) - lo) / span
        r, g, b = _colormap(t)
        flat.extend((r, g, b))
    return flat


#: A small built-in perceptual-ish colormap (a coarse viridis-like ramp), as
#: ``(r, g, b)`` control points in ``[0, 1]``.  Kept tiny and dependency-free so
#: the threejs exporter never imports matplotlib for a colormap.
_COLORMAP_STOPS: tuple[tuple[float, float, float], ...] = (
    (0.267, 0.005, 0.329),
    (0.283, 0.141, 0.458),
    (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553),
    (0.164, 0.471, 0.558),
    (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518),
    (0.267, 0.749, 0.441),
    (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150),
    (0.993, 0.906, 0.144),
)


def _colormap(t: float) -> tuple[float, float, float]:
    """Map ``t`` in ``[0, 1]`` to an ``(r, g, b)`` triple via linear interpolation.

    Clamps ``t`` to ``[0, 1]`` and linearly interpolates the built-in
    :data:`_COLORMAP_STOPS` ramp.
    """
    t = min(1.0, max(0.0, float(t)))
    last = len(_COLORMAP_STOPS) - 1
    pos = t * last
    i = int(pos)
    if i >= last:
        return _COLORMAP_STOPS[last]
    frac = pos - i
    r0, g0, b0 = _COLORMAP_STOPS[i]
    r1, g1, b1 = _COLORMAP_STOPS[i + 1]
    return (
        r0 + (r1 - r0) * frac,
        g0 + (g1 - g0) * frac,
        b0 + (b1 - b0) * frac,
    )


def _parse_rgb(color: Any) -> tuple[float, float, float] | None:
    """Coerce a style color to an ``(r, g, b)`` triple in ``[0, 1]``, or ``None``.

    Accepts a 3- or 4-sequence of floats (an RGB / RGBA tuple); anything else
    (a named color string, ``None``) returns ``None`` — the exporter leaves such a
    layer uncolored (the frontend applies its default material color).
    """
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
        except (TypeError, ValueError):
            return None
        return (r, g, b)
    return None


# ---------------------------------------------------------------------------
# material style (the honored per-layer style keys for threejs)
# ---------------------------------------------------------------------------


def _material_style(layer: Layer, *, palette_color: str | None = None) -> dict[str, Any]:
    """Extract the three.js-honored per-layer style keys into a ``material`` dict.

    The three.js backend honors: ``color``, ``linewidth``, ``markersize``,
    ``alpha``, ``cmap``, and ``zorder`` (mapped to ``renderOrder`` by the loader).
    ``linestyle`` and marker *shape* are excluded from the backend's
    ``honored_by`` set and are **not** serialized here.

    The layer's ``style`` dict is first canonicalized via :func:`normalize_style`
    (aliases → canonical names, values validated); then the six honored keys are
    extracted, with ``None`` for any key not set.  When the layer carries no
    explicit ``color`` the ``palette_color`` fallback is used so the loader sees a
    deterministic per-layer color from the theme palette.

    Parameters
    ----------
    layer : Layer
        The layer whose ``style`` dict to extract.
    palette_color : str, optional
        Theme-palette auto-color for this layer (used when ``style["color"]`` is
        absent).

    Returns
    -------
    dict
        A JSON-friendly ``material`` block; every value is a plain Python scalar
        (str / float / int / None).
    """
    canon = normalize_style(layer.style, warn=False)
    # Resolve the color: explicit style > palette auto-color.
    color: str | None = canon.get("color")
    if color is None and palette_color is not None:
        color = palette_color
    lw = canon.get("linewidth")
    ms = canon.get("markersize")
    alpha = canon.get("alpha")
    cmap = canon.get("cmap")
    zorder = canon.get("zorder")
    return {
        "color": str(color) if color is not None else None,
        "linewidth": float(lw) if lw is not None else None,
        "markersize": float(ms) if ms is not None else None,
        "alpha": float(alpha) if alpha is not None else None,
        "cmap": str(cmap) if cmap is not None else None,
        "zorder": int(zorder) if zorder is not None else None,
    }


def _theme_metadata(theme: Theme) -> dict[str, Any]:
    """Serialize the resolved theme into a compact ``metadata.theme`` block.

    The loader reads three fields to apply scene-level presentation:

    - ``background``: the scene background color (a CSS string or ``null``).
    - ``foreground``: the default ink / axis color (a CSS string or ``null``).
    - ``palette``: the ordered color cycle for auto-colored layers (a list of
      CSS strings); the loader assigns ``palette[i % len(palette)]`` to geometry
      ``i`` when it carries no per-vertex colors and its ``material.color`` is
      also ``null``.

    Only these three fields travel to the loader — the other Theme fields
    (font, grid, line/marker sizes) are not meaningful in a three.js scene.

    Parameters
    ----------
    theme : Theme
        The resolved (never ``None``) theme for the spec.

    Returns
    -------
    dict
        A JSON-serializable mapping with ``"background"``, ``"foreground"``,
        and ``"palette"`` keys.
    """
    return {
        "background": theme.background,
        "foreground": theme.foreground,
        "palette": list(theme.palette),
    }


def _axis_labels(spec: PlotSpec) -> dict[str, str]:
    """Return the per-axis label strings (``""`` when an axis carries none)."""
    return {
        "x": spec.x.label,
        "y": spec.y.label,
        "z": spec.z.label if spec.z is not None else "",
    }


def _axis_units(spec: PlotSpec) -> dict[str, str]:
    """Return the per-axis unit strings, read from each axis's ``tickformat``.

    The spec IR has no dedicated unit field; the backend-neutral ``tickformat``
    (when set) is the closest carrier, so it is surfaced here as the axis unit
    hint (``""`` when absent).
    """
    return {
        "x": _unit(spec.x),
        "y": _unit(spec.y),
        "z": _unit(spec.z) if spec.z is not None else "",
    }


def _unit(axis: Axis) -> str:
    """Return the unit hint for one axis (its ``tickformat`` or ``""``)."""
    return axis.tickformat or ""


def _bounds(geometries: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Compute the per-axis ``[min, max]`` over every geometry's interleaved positions.

    Falls back to ``[0.0, 0.0]`` per axis when there are no vertices, so the
    payload always carries a well-formed ``bounds`` block.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for geom in geometries:
        pos = geom["positions"]
        xs.extend(pos[0::3])
        ys.extend(pos[1::3])
        zs.extend(pos[2::3])
    return {
        "x": _minmax(xs),
        "y": _minmax(ys),
        "z": _minmax(zs),
    }


def _minmax(values: list[float]) -> list[float]:
    """``[min, max]`` of ``values``, or ``[0.0, 0.0]`` when empty."""
    if not values:
        return [0.0, 0.0]
    return [float(min(values)), float(max(values))]


def _camera(spec: PlotSpec, bounds: dict[str, list[float]]) -> dict[str, list[float]]:
    """Build the camera block — ``spec.meta["camera"]`` if present, else from ``bounds``.

    A caller-supplied ``spec.meta["camera"]`` (a mapping with ``position`` /
    ``target`` / ``up`` sequences) is normalized and passed through.  Otherwise a
    default view is derived: the target is the bounds centre, the camera sits one
    bounding-box diagonal away on a ``(1, 1, 1)`` corner direction, and ``up`` is
    ``+z``.
    """
    override = spec.meta.get("camera") if isinstance(spec.meta, dict) else None
    if isinstance(override, dict):
        return {
            "position": _vec3(override.get("position"), default=(1.0, 1.0, 1.0)),
            "target": _vec3(override.get("target"), default=(0.0, 0.0, 0.0)),
            "up": _vec3(override.get("up"), default=(0.0, 0.0, 1.0)),
        }

    cx = (bounds["x"][0] + bounds["x"][1]) / 2.0
    cy = (bounds["y"][0] + bounds["y"][1]) / 2.0
    cz = (bounds["z"][0] + bounds["z"][1]) / 2.0
    dx = bounds["x"][1] - bounds["x"][0]
    dy = bounds["y"][1] - bounds["y"][0]
    dz = bounds["z"][1] - bounds["z"][0]
    diagonal = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if diagonal == 0.0:
        diagonal = 1.0
    offset = diagonal  # one diagonal back along the (1, 1, 1) corner direction
    step = offset / float(np.sqrt(3.0))
    return {
        "position": [cx + step, cy + step, cz + step],
        "target": [cx, cy, cz],
        "up": [0.0, 0.0, 1.0],
    }


def _vec3(value: Any, *, default: tuple[float, float, float]) -> list[float]:
    """Coerce a 3-sequence to ``[x, y, z]`` floats, falling back to ``default``."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return list(default)
    return list(default)


# ---------------------------------------------------------------------------
# animation (the reveal directive — draw-range driven on the frontend)
# ---------------------------------------------------------------------------


def _animation_metadata(spec: PlotSpec) -> dict[str, Any] | None:
    """Build the ``metadata["animation"]`` block, or ``None`` for a static spec.

    Mirrors the matplotlib / plotly reveal model so the three.js loader plays the
    same comet (a windowed trail + a head marker sweeping the curve over a faint
    full-curve backdrop).  The geometry buffers are **not** touched — the loader
    advances ``geometry.setDrawRange`` over the line vertices, so all the block
    carries is the directive plus ``n_samples`` (the longest animated line's
    vertex count) for the loop to size its window / stride against.

    Returns ``None`` when ``spec.animation`` is absent (so a static export is
    byte-identical to the pre-animation payload) or when the spec has no
    animatable **line** layer to reveal.  In the latter case — an animation *was*
    requested but the reference loader has no comet to play (a ``points``-only or
    ``surface``-only spec) — a :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`
    warning is emitted so the animation is never *silently* dropped: the export is
    a valid static payload that the loader renders (auto-rotating) as usual.
    """
    anim = spec.animation
    if anim is None:
        return None
    n_samples = _animated_sample_count(spec)
    if n_samples < 2:
        _warn_unrevealable(spec)
        return None
    trail = _trail_length_samples(spec, anim)
    return {
        "fps": float(anim.fps),
        "duration": None if anim.duration is None else float(anim.duration),
        "n_frames": None if anim.n_frames is None else int(anim.n_frames),
        "loop": bool(anim.loop),
        "pingpong": bool(anim.pingpong),
        "trail_length_samples": trail,
        "head": bool(anim.head),
        "head_size": float(anim.head_size),
        "head_color": _head_color(anim.head_color),
        "n_samples": int(n_samples),
    }


def _warn_unrevealable(spec: PlotSpec) -> None:
    """Warn that an animated spec has no line geometry the threejs loader can reveal.

    Honors the issue's "animates, **or warns** — never silently drops" contract:
    the threejs reveal comet is a ``setDrawRange`` sweep over a **line** index
    buffer, so a ``points``-only / ``surface``-only animated spec has nothing to
    reveal.  Rather than emit a block the loader cannot play (which would freeze
    the export *and* the camera), the exporter drops the animation to a static
    payload and warns here.
    """
    import warnings

    from ..caps import VisualizationDegraded

    kind = spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind)
    warnings.warn(
        f"threejs export: the animation on this {kind!r} spec was dropped — its "
        "reveal comet needs a line (LINE / LINE3D) geometry, but the spec has only "
        "points / surface layers. Exporting a static payload instead.",
        VisualizationDegraded,
        stacklevel=2,
    )


def _head_color(color: Any) -> list[float] | None:
    """Coerce the head color to a plain ``[r, g, b]`` list, or ``None``.

    Reuses :func:`_parse_rgb` (an RGB / RGBA sequence → a triple; a named-color
    string / ``None`` → ``None``) but returns a JSON-friendly ``list`` so the
    animation block stays plain-list / plain-float like the rest of the payload.
    """
    rgb = _parse_rgb(color)
    return None if rgb is None else [rgb[0], rgb[1], rgb[2]]


def _animated_sample_count(spec: PlotSpec) -> int:
    """Vertices on the longest animatable (line) layer — the reveal length (0 if none)."""
    n = 0
    for layer in spec.layers:
        if PlotKind(layer.kind) not in _ANIMATED_MARKS:
            continue
        arr = layer.data.get("x", layer.data.get("y"))
        if arr is not None:
            n = max(n, int(np.asarray(arr).reshape(-1).shape[0]))
    return n


def _trail_length_samples(spec: PlotSpec, anim: Animation) -> int | None:
    """Resolve the comet tail length to a vertex count (``None`` ⇒ persistent).

    Reuses :meth:`~tsdynamics.viz.spec.Animation.tail_samples` (the same
    ``"time"`` ÷ ``dt`` / ``"steps"`` rule the other backends use), reading the
    sample spacing from ``spec.meta["dt"]`` for a time-unit trail.
    """
    dt = spec.meta.get("dt") if isinstance(spec.meta, dict) else None
    try:
        dt_f = float(dt) if dt is not None and float(dt) > 0 else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        dt_f = None
    return anim.tail_samples(dt_f)


# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------


def _flat(value: Any) -> np.ndarray | None:
    """Coerce a channel to a 1-D float array, or ``None`` if absent."""
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def _grid(value: Any) -> np.ndarray | None:
    """Coerce a surface channel to a 2-D float array, or ``None`` if absent."""
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _interleave(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float]:
    """Interleave ``x``, ``y``, ``z`` into a flat ``[x0, y0, z0, x1, ...]`` list.

    Returns plain Python floats (never nested arrays), so the result is directly
    JSON-serializable and a ``Float32Array`` source on the frontend.
    """
    stacked = np.stack((x, y, z), axis=1).reshape(-1)
    return [float(v) for v in stacked]
