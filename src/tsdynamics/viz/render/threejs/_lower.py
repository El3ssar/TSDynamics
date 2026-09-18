"""Geometry lowering for the ``threejs`` data-export backend (stream VIZ-THREEJS-EXPORT).

This module turns a backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` into a
**three.js BufferGeometry-ready** JSON-able payload — a pure-Python dict of plain
floats / ints / strings, no optional dependency and no plotting library import.
A web frontend reads the payload and builds ``THREE.BufferGeometry`` objects
directly: the flat :data:`positions` list is a ``Float32Array`` source
(``x, y, z`` interleaved) and the optional flat :data:`c` list a per-vertex
*scalar* field the loader maps through its own colour ramp.

Payload weight is a first-class concern here
-------------------------------------------
This payload is the library's **web-embedding** artifact — it is inlined into an
HTML page and parsed by a browser — so its size is a correctness property, not a
micro-optimisation.  Three decisions keep it small, each measured:

- **A vertex cap** (:data:`DEFAULT_MAX_POINTS`), applied by **arc-length**
  resampling (:mod:`tsdynamics.viz._resample`), never by a stride.  Uncapped, a
  1e6-sample attractor lowers to ~26 MB of JSON (~60 MB with ``decimals=None``, and
  ~137 MB under the pre-v6 exporter, which also shipped an index buffer and
  pre-expanded per-vertex RGB); capped it is ~1 MB.  A stride
  would hit the same byte count with 52x the geometric error on a fast attractor
  (see :mod:`tsdynamics.viz._resample` for the measurement).
- **No index buffer for lines.**  A polyline's vertices are already in draw order,
  so a contiguous ``THREE.Line`` needs no index; the old ``0,1,1,2,2,3,...``
  ``LineSegments`` buffer was 12.9% of the payload for the same picture at twice
  the GPU index work.  ``"surface"`` keeps its indices — that is a real
  triangulation, not a restatement of vertex order.
- **A scalar ``c`` channel instead of pre-expanded per-vertex RGB.**  Emitting
  three floats per vertex to encode one scalar was 38.6% of the payload; the
  loader owns the colour ramp (and, for the brand comet, ignores it entirely).

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
            "bounds": {
                "x": [<min>, <max>], "y": [<min>, <max>], "z": [<min>, <max>]
            },
            "camera": {
                "position": [<x>, <y>, <z>],
                "target":   [<x>, <y>, <z>],
                "up":       [<x>, <y>, <z>]
            },
            "resample": {                        # ALWAYS present; the vertex budget
                "max_points": <int | null>,      # null ⇒ uncapped (max_points=None)
                "original_vertices": <int>,      # before capping
                "vertices": <int>,               # after capping
                "capped": <bool>
            },
            "theme": {                           # ALWAYS present; resolved Theme
                "background": "<str | null>",    # scene background color
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

A static (non-animated) spec carries **no** ``animation`` key in ``metadata``.
When ``spec.animation`` is present the geometry buffers are unchanged: the loader
animates by rewriting a fixed-length trail window over the full-curve backdrop, so
no positions are re-uploaded and no per-vertex time attribute is needed (the line
vertices are already the natural reveal order).  ``n_samples`` and
``trail_length_samples`` are reported **in capped vertices** — after the cap the
curve genuinely has that many vertices, and a trail measured against the uncapped
count would sweep the wrong fraction of the attractor.

Each geometry carries a ``material`` block with the layer's style vocabulary keys
that the three.js backend honors::

    "material": {
        "color":       "<CSS string | null>",   # explicit layer color
        "linewidth":   <float | null>,          # line width (pt)
        "markersize":  <float | null>,          # point size (pt)
        "alpha":       <float | null>,          # 0..1 opacity
        "zorder":      <int | null>             # maps to THREE renderOrder
    }

Keys are ``null`` when not set by the user (the loader applies its own default).
``linestyle``, marker *shape*, and ``cmap`` are **not** in the material block —
they are excluded from the threejs backend's ``honored_by`` set (see
:data:`~tsdynamics.viz.style.STYLE_KEYS`) and the loader ignores them.  ``cmap``
in particular cannot be honored: the loader owns a fixed built-in colour ramp for
the ``"c"`` channel, so an arbitrary colormap *name* would be a dead field.

Each ``geometry`` is::

    {
        "type": "line" | "points" | "surface",
        "label": "<layer label or null>",
        "positions": [x0, y0, z0, x1, y1, z1, ...],   # FLAT, plain floats
        "indices":   [...],                            # "surface" only; [] otherwise
        "c":         [c0, c1, ...],                    # optional per-vertex SCALAR
        "n_vertices": <int>,
        "n_vertices_original": <int>                   # before the cap
    }

- A 3-D ``LINE3D`` / a 2-D ``LINE`` (lifted to ``z = 0``) → a ``"line"`` geometry
  with **no** index buffer (a contiguous ``THREE.Line``; vertex order *is* draw
  order).
- A ``SCATTER`` / ``MARKERS`` (2-D lifted to ``z = 0``, or 3-D) → a ``"points"``
  geometry (no ``indices``).
- A ``SURFACE3D`` → a ``"surface"`` geometry whose ``indices`` triangulate the
  grid (two triangles per quad).

``bounds`` are the per-axis ``[min, max]`` over every geometry's vertices; the
``camera`` is derived from those bounds (a corner view looking at the centre)
unless ``spec.meta["camera"]`` overrides it.

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
            "resample": { ... },            # aggregate over every panel
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
        "metadata": { ... }                 # the panel's own labels/bounds/camera
    }

An **animated** composite has no three.js representation: the reference loader
reveals **one** draw range in **one** scene, so it can play a single panel and
nothing more.  Rather than write a payload whose animation directive silently
vanishes, :func:`lower_spec` exports the composite **statically** and emits one
:class:`~tsdynamics.viz.render.caps.VisualizationDegraded` naming what was
dropped and the format that does write it (``.mp4`` / ``.gif``, matplotlib).
A ``layout`` mode of ``"frames"`` is the sharper case — those panels are
consecutive in *time*, so tiling them in space would be a wrong picture, not a
degraded one; the exporter keeps the **last** panel (the still that a
non-animating backend shows for any movie) and says so.

The panel's geometry ``positions`` stay in the panel's **own** local coordinates
(unshifted), so a frontend can render each panel into its own viewport
untouched.  The separate ``offset`` is a convenience translation — each panel's
local-bounds centre laid out on the resolved ``rows`` x ``cols`` grid with unit
cell spacing (column → +x, row → −y, so row 0 is at the top) — for a frontend
that prefers to drop every panel into **one** shared scene rather than tile
viewports.  Either reading is valid: the panel ``grid`` cell and ``offset`` are
redundant placement hints, and the geometry itself is never mutated.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._resample import resample_arclength, uniform_subsample_indices
from ..._visibility import listing_dir
from ...export import SCHEMA_VERSION
from ...spec import PlotKind
from ...style import normalize_style
from ..caps import VisualizationDegraded

if TYPE_CHECKING:
    from ...spec import Animation, Layer, Layout, PlotSpec
    from ...style import Theme

__all__ = ["DEFAULT_DECIMALS", "DEFAULT_MAX_POINTS", "lower_spec"]

__dir__ = listing_dir(__all__)

#: Default ceiling on the vertices of any single geometry.  40 000 vertices of a
#: line is a cheap ``THREE.Line`` and — resampled by **arc length** — holds the
#: worst-case sagitta of even the fastest catalogue attractor (HyperQi) below the
#: 0.008 readability target, while the whole inlined page stays around 1.5 MB.
#: ``None`` opts out (the caller accepts an unbounded payload).
DEFAULT_MAX_POINTS = 40_000

#: Decimal places kept for the bulky float buffers (``positions`` / ``c``).  Four
#: decimals is well below a ``Float32Array``'s own precision at attractor scales
#: and roughly halves the JSON text; ``None`` disables rounding.
DEFAULT_DECIMALS = 4

#: Seed for the point-cloud thinning (:func:`~tsdynamics.viz._resample.uniform_subsample_indices`).
#: Fixed so an export is reproducible byte-for-byte across processes.
_SUBSAMPLE_SEED = 0

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

#: Geometry types the reference loader can play a reveal comet on: a ``"line"``
#: (a swept curve) and a ``"points"`` cloud (a trailing swarm — the loader's
#: ``buildPointsComet``).  A ``"surface"`` is a mesh with no sweep order, so an
#: animated surface-only spec emits no animation block (and warns) rather than a
#: block the loader cannot play.  Mirror this set in the loader's geometry guard.
_REVEALABLE_GEOMETRY: frozenset[str] = frozenset({"line", "points"})


@dataclass
class _CapReport:
    """Accounting for the vertex cap across one :func:`lower_spec` call.

    A composite lowers many panels, and a *per-geometry* warning would spam the
    caller with one message per layer.  This accumulates the totals so the caller
    emits exactly **one** :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`
    naming the original and capped counts.
    """

    max_points: int | None
    original: int = 0
    kept: int = 0
    capped_labels: list[str] = field(default_factory=list)
    #: Whether a **curve** (arc-length resampled) / a **point cloud** (uniformly
    #: subsampled) was among the thinned geometries.  The two are thinned by
    #: different criteria, so the one consolidated warning has to name the one that
    #: actually ran rather than assume every capped layer was a curve.
    capped_curves: bool = False
    capped_clouds: bool = False

    @property
    def capped(self) -> bool:
        """Whether any geometry was actually thinned."""
        return bool(self.capped_labels)

    def record(
        self, label: str | None, original: int, kept: int, *, geom_type: str = "line"
    ) -> None:
        """Note one geometry's vertex counts (and whether it was thinned)."""
        self.original += original
        self.kept += kept
        if kept < original:
            self.capped_labels.append(label or f"layer[{len(self.capped_labels)}]")
            if geom_type == "points":
                self.capped_clouds = True
            else:
                self.capped_curves = True

    def to_dict(self) -> dict[str, Any]:
        """Build the ``metadata["resample"]`` block."""
        return {
            "max_points": None if self.max_points is None else int(self.max_points),
            "original_vertices": int(self.original),
            "vertices": int(self.kept),
            "capped": self.capped,
        }


def lower_spec(
    spec: PlotSpec,
    *,
    max_points: int | None = DEFAULT_MAX_POINTS,
    decimals: int | None = DEFAULT_DECIMALS,
) -> dict[str, Any]:
    """Lower ``spec`` to a three.js BufferGeometry-ready JSON-able payload.

    For a **single-panel** spec, walks the spec's drawable layers, lowering each
    one whose mark is a line / points / surface (other marks — images, bars,
    quivers — have no BufferGeometry analogue and are skipped) into a
    flat-positions geometry, then derives the top-level ``metadata`` (labels /
    bounds / camera / theme / resample) from the axes and the lowered vertices.
    When ``spec.animation`` is set, an ``animation`` block is added to ``metadata``
    (the reveal directive — fps / duration / trail length in samples / head) so
    the reference loader plays a comet reveal.

    For a **composite** spec (``PlotKind.COMPOSITE`` — its drawable content lives
    in :attr:`~tsdynamics.viz.spec.PlotSpec.panels`, not the top-level layers),
    lowers each panel recursively and emits a ``"panels"`` list of per-panel
    payloads (each carrying its identity, grid placement, and a layout-offset)
    plus a top-level ``layout`` block — see the module docstring for the schema.

    Parameters
    ----------
    spec : PlotSpec
        The spec to lower.
    max_points : int or None, optional
        Ceiling on the vertices of any single geometry (default
        :data:`DEFAULT_MAX_POINTS`).  A **line** over the ceiling is resampled by
        arc length (never by a stride — see :mod:`tsdynamics.viz._resample`); a
        **point cloud** is thinned by a deterministic seeded uniform draw.
        ``None`` exports every vertex.  Capping emits exactly one
        :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` naming the
        original and capped counts — it is never silent.
    decimals : int or None, optional
        Decimal places kept in the ``positions`` / ``c`` buffers (default
        :data:`DEFAULT_DECIMALS`); ``None`` disables rounding.

    Returns
    -------
    dict
        A JSON-serializable payload conforming to the module-docstring schema
        (every value is a plain ``str`` / ``int`` / ``float`` / ``list``).

    Warns
    -----
    VisualizationDegraded
        When at least one geometry exceeded ``max_points`` and was thinned.
    """
    report = _CapReport(max_points=max_points)
    if spec.is_composite:
        spec = _degrade_animated_composite(spec)
    if spec.is_composite:
        payload = _lower_composite(spec, report, decimals)
    else:
        payload = _lower_single(spec, report, decimals)
    payload["metadata"]["resample"] = report.to_dict()
    _warn_if_capped(report)
    return payload


def _degrade_animated_composite(spec: PlotSpec) -> PlotSpec:
    """Refuse to animate a composite — out loud — and return what *is* exportable.

    The reference loader reveals **one** draw range in **one** scene, so a
    multi-panel movie has no three.js representation at all.  Two shapes, one rule
    ("animates, **or warns** — never silently drops"):

    - ``layout`` mode ``"frames"`` — the panels are consecutive in *time*, so
      tiling them in space is a wrong picture, not a degraded one.  The **last**
      panel is returned (the still every non-animating backend shows for a movie),
      stripped of its own animation.
    - any other mode — the panels are a genuine spatial layout, so the tiled
      export is kept and only the top-level animation directive is dropped.

    A *static* composite is returned untouched, so nothing about the existing
    export changes.

    Parameters
    ----------
    spec : PlotSpec
        A composite spec (the caller has already checked ``is_composite``).

    Returns
    -------
    PlotSpec
        ``spec`` itself when it is static; otherwise the de-animated composite, or
        — in ``"frames"`` mode — its last panel.

    Warns
    -----
    VisualizationDegraded
        Exactly once, naming the dropped animation and the way to get the movie.
    """
    frames_mode = getattr(getattr(spec, "layout", None), "mode", None) == "frames"
    if spec.animation is None and not frames_mode:
        return spec

    panels = list(spec.panels)
    if frames_mode:
        detail = (
            f"its {len(panels)} panels are frames of a movie, not a spatial layout, "
            "so tiling them would be a different picture; exporting the last panel"
        )
    else:
        detail = (
            f"the reveal comet plays one draw range in one scene, and this is a "
            f"{len(panels)}-panel composite; exporting it statically"
        )
    warnings.warn(
        f"threejs export: the animation on this composite was dropped — {detail}. "
        "For the movie itself, save to .mp4 / .gif (matplotlib writes them).",
        VisualizationDegraded,
        stacklevel=3,
    )
    if frames_mode and panels:
        return dataclasses.replace(panels[-1], animation=None)
    return dataclasses.replace(spec, animation=None)


def _warn_if_capped(report: _CapReport) -> None:
    """Emit the single consolidated cap warning, if anything was thinned."""
    if not report.capped:
        return
    names = ", ".join(report.capped_labels[:4])
    if len(report.capped_labels) > 4:
        names += f", +{len(report.capped_labels) - 4} more"
    # Name the thinning that actually ran.  A curve is resampled by arc length; a
    # point cloud is subsampled uniformly (it has no chord to preserve), and telling
    # a user their scatter was "resampled by arc length" is a false reassurance
    # about the very property this module documents as inapplicable to a set.
    how = {
        (True, False): "The curve was resampled by arc length, so its shape is preserved",
        (False, True): "The point cloud was thinned by a deterministic uniform draw, "
        "so its density structure is preserved",
        (True, True): "Curves were resampled by arc length and point clouds thinned by a "
        "deterministic uniform draw, so their shapes are preserved",
    }[(report.capped_curves, report.capped_clouds)]
    warnings.warn(
        f"threejs export: capped {report.original} vertices to {report.kept} "
        f"(max_points={report.max_points}, layers: {names}). {how}; pass "
        "max_points=None to export every vertex (a 1e6-sample trajectory is a "
        "~26 MB payload).",
        VisualizationDegraded,
        stacklevel=3,
    )


def _lower_single(spec: PlotSpec, report: _CapReport, decimals: int | None) -> dict[str, Any]:
    """Lower a single-panel spec to the ``geometries`` + ``metadata`` payload.

    The non-composite lowering: walk ``spec.layers``, lower each drawable mark,
    and derive the per-spec ``metadata`` (labels / bounds / camera / theme).  Kept
    as a stable helper so the composite path can reuse it per panel.
    """
    theme = spec.resolved_theme

    geometries: list[dict[str, Any]] = []
    for i, layer in enumerate(spec.layers):
        palette_color = theme.palette[i % len(theme.palette)] if theme.palette else None
        geom = _lower_layer(
            layer,
            palette_color=palette_color,
            max_points=report.max_points,
            decimals=decimals,
        )
        if geom is not None:
            report.record(
                geom["label"],
                geom["n_vertices_original"],
                geom["n_vertices"],
                geom_type=geom["type"],
            )
            geometries.append(geom)

    bounds = _bounds(geometries)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "labels": _axis_labels(spec),
        "bounds": bounds,
        "camera": _camera(spec, bounds),
        "theme": _theme_metadata(theme),
    }
    animation = _animation_metadata(spec, geometries)
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


def _lower_composite(spec: PlotSpec, report: _CapReport, decimals: int | None) -> dict[str, Any]:
    """Lower a ``COMPOSITE`` spec to a panelled payload.

    Lowers each child panel via :func:`_lower_single`, places it on the resolved
    ``rows`` x ``cols`` grid (per the :class:`~tsdynamics.viz.spec.Layout`), tags
    it with its identity (index / title / kind) + grid cell + a layout-offset, and
    aggregates the per-panel bounds (each shifted by its offset) into a top-level
    ``bounds`` / ``camera`` that frames the whole laid-out scene.
    """
    panels_in = spec.panels
    rows, cols = _composite_grid(spec.layout, len(panels_in))

    panels_out: list[dict[str, Any]] = []
    placed_geometries: list[dict[str, Any]] = []
    for i, panel in enumerate(panels_in):
        sub = _lower_single(panel, report, decimals)
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
    the explicit ``rows`` x ``cols`` (or a near-square fit when unset), and
    ``"stack"`` (the default) → one column.  ``n == 0`` yields ``(0, 0)``.
    """
    if n <= 0:
        return 0, 0
    grid = getattr(layout, "grid", None)
    if callable(grid):  # Layout.grid(n) — the hoisted single implementation (H3)
        rows, cols = grid(n)
        return int(rows), int(cols)
    mode = getattr(layout, "mode", "stack")
    if mode == "row":
        return 1, n
    if mode == "grid":
        rows_attr = getattr(layout, "rows", None)
        cols_attr = getattr(layout, "cols", None)
        if rows_attr and cols_attr:
            return int(rows_attr), int(cols_attr)
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


def _lower_layer(
    layer: Layer,
    *,
    palette_color: str | None = None,
    max_points: int | None,
    decimals: int | None,
) -> dict[str, Any] | None:
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
    max_points : int or None
        The vertex ceiling (see :func:`lower_spec`).
    decimals : int or None
        Float rounding for the bulky buffers.
    """
    geom_type = _GEOMETRY_TYPE.get(layer.kind)
    if geom_type is None:
        return None

    if geom_type == "surface":
        # A surface is a triangulated grid: thinning it would need a 2-D mesh
        # decimation, not a 1-D resample, so the cap deliberately does not apply.
        return _lower_surface(layer, palette_color=palette_color, decimals=decimals)
    return _lower_line_or_points(
        layer, geom_type, palette_color=palette_color, max_points=max_points, decimals=decimals
    )


def _lower_line_or_points(
    layer: Layer,
    geom_type: str,
    *,
    palette_color: str | None = None,
    max_points: int | None,
    decimals: int | None,
) -> dict[str, Any] | None:
    """Lower a line / points layer to a flat-positions geometry.

    A 2-D layer (no ``"z"`` channel) is lifted to ``z = 0``.  Neither type carries
    an index buffer — a line's vertex order *is* its draw order (a contiguous
    ``THREE.Line``), and points are unindexed by construction.

    **The cap is applied here**, and the two geometry types are capped
    differently on purpose:

    - a ``"line"`` is a *curve*, so it is resampled uniformly in **arc length**
      (:func:`~tsdynamics.viz._resample.resample_arclength`) — every turn keeps
      resolution proportional to the length it occupies;
    - ``"points"`` is a *set* with no chord to bow off, so it is thinned by a
      deterministic seeded uniform draw
      (:func:`~tsdynamics.viz._resample.uniform_subsample_indices`).

    Either way the ``"c"`` channel travels on the **same** parameterisation as the
    positions, so the colour field can never de-register from the vertices.
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

    c_arr = _flat(layer.data.get("c"))
    c = c_arr[:n] if c_arr is not None and c_arr.size >= n else None

    n_original = int(n)
    pts = np.stack((x, y, z), axis=1)
    if max_points is not None and n > max_points:
        if geom_type == "line":
            channels = {"c": c} if c is not None else {}
            pts, out_channels = resample_arclength(pts, int(max_points), channels=channels)
            c = out_channels.get("c")
        else:
            idx = uniform_subsample_indices(n_original, int(max_points), seed=_SUBSAMPLE_SEED)
            pts = pts[idx]
            c = c[idx] if c is not None else None

    geometry: dict[str, Any] = {
        "type": geom_type,
        "label": layer.label,
        "positions": _round_flat(pts.reshape(-1), decimals),
        "indices": [],
        "material": _material_style(layer, palette_color=palette_color),
        "n_vertices": int(len(pts)),
        "n_vertices_original": n_original,
    }
    if c is not None:
        geometry["c"] = _round_flat(c, decimals)
    return geometry


def _lower_surface(
    layer: Layer, *, palette_color: str | None = None, decimals: int | None
) -> dict[str, Any] | None:
    """Lower a ``SURFACE3D`` layer to a triangulated-grid geometry.

    Expects the ``"x"`` / ``"y"`` / ``"z"`` channels as 2-D grids of identical
    shape (rows x cols).  Emits row-major interleaved vertex positions and an
    index list of two triangles per grid quad (``THREE.Mesh`` / ``BufferGeometry``
    order) — the one geometry type whose indices are a real triangulation and not
    a restatement of vertex order, so they are kept.  The scalar height (or the
    ``"c"`` channel) travels as the per-vertex ``"c"`` field.
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

    positions = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1).reshape(-1)
    c = _grid(layer.data.get("c"))
    cflat = c.reshape(-1) if c is not None and c.shape == z.shape else z.reshape(-1)
    return {
        "type": "surface",
        "label": layer.label,
        "positions": _round_flat(positions, decimals),
        "indices": _surface_indices(rows, cols),
        "material": _material_style(layer, palette_color=palette_color),
        "c": _round_flat(cflat, decimals),
        "n_vertices": int(rows * cols),
        "n_vertices_original": int(rows * cols),
    }


# ---------------------------------------------------------------------------
# index buffers
# ---------------------------------------------------------------------------


def _surface_indices(rows: int, cols: int) -> list[int]:
    """Two-triangles-per-quad index list over a ``rows`` x ``cols`` vertex grid.

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
# material style (the honored per-layer style keys for threejs)
# ---------------------------------------------------------------------------


def _parse_rgb(color: Any) -> tuple[float, float, float] | None:
    """Coerce a style color to an ``(r, g, b)`` triple in ``[0, 1]``, or ``None``.

    Accepts a 3- or 4-sequence of floats (an RGB / RGBA tuple); anything else
    (a named color string, ``None``) returns ``None``.
    """
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
        except (TypeError, ValueError):
            return None
        return (r, g, b)
    return None


def _material_style(layer: Layer, *, palette_color: str | None = None) -> dict[str, Any]:
    """Extract the three.js-honored per-layer style keys into a ``material`` dict.

    The three.js backend honors: ``color``, ``linewidth``, ``markersize``,
    ``alpha``, and ``zorder`` (mapped to ``renderOrder`` by the loader).
    ``linestyle``, marker *shape*, and ``cmap`` are excluded from the backend's
    ``honored_by`` set and are **not** serialized here — the loader owns a fixed
    built-in colour ramp for the per-vertex ``"c"`` channel, so an arbitrary
    ``cmap`` name cannot be honored.

    The layer's ``style`` dict is first canonicalized via :func:`normalize_style`
    (aliases → canonical names, values validated); then the honored keys are
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
    zorder = canon.get("zorder")
    return {
        "color": str(color) if color is not None else None,
        "linewidth": float(lw) if lw is not None else None,
        "markersize": float(ms) if ms is not None else None,
        "alpha": float(alpha) if alpha is not None else None,
        "zorder": int(zorder) if zorder is not None else None,
    }


def _theme_metadata(theme: Theme) -> dict[str, Any]:
    """Serialize the resolved theme into a compact ``metadata.theme`` block.

    The loader reads two fields to apply scene-level presentation:

    - ``background``: the scene background color (a CSS string or ``null``).
    - ``palette``: the ordered color cycle for auto-colored layers (a list of
      CSS strings); the loader assigns ``palette[i % len(palette)]`` to geometry
      ``i`` when it carries no per-vertex colors and its ``material.color`` is
      also ``null``.

    Only these two fields travel to the loader — the other Theme fields
    (``foreground``, font, grid, title size, line/marker sizes) are **not**
    honored in a three.js scene (the loader has no axes / text / grid to ink),
    so they are not emitted: a dropped field is honest, a dead field would
    overclaim.  The threejs backend's theme-honored set is therefore
    ``{background, palette}`` — ``caps`` warns for every other theme field.

    Parameters
    ----------
    theme : Theme
        The resolved (never ``None``) theme for the spec.

    Returns
    -------
    dict
        A JSON-serializable mapping with ``"background"`` and ``"palette"`` keys.
    """
    return {
        "background": theme.background,
        "palette": list(theme.palette),
    }


def _axis_labels(spec: PlotSpec) -> dict[str, str]:
    """Return the per-axis label strings (``""`` when an axis carries none)."""
    return {
        "x": spec.x.label,
        "y": spec.y.label,
        "z": spec.z.label if spec.z is not None else "",
    }


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
# animation (the reveal directive — trail-window driven on the frontend)
# ---------------------------------------------------------------------------


def _animation_metadata(spec: PlotSpec, geometries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build the ``metadata["animation"]`` block, or ``None`` for a static spec.

    Mirrors the matplotlib / plotly reveal model so the three.js loader plays the
    same comet (a windowed trail + a head marker sweeping the curve over a faint
    full-curve backdrop).

    **The counts are read off the lowered geometries, not the spec.**  After the
    vertex cap the exported curve genuinely has fewer vertices than the spec's
    layer arrays, and a reveal sized against the *uncapped* count would index past
    the buffer (freezing the comet at the end of the curve) while a trail measured
    in uncapped samples would sweep the wrong fraction of the attractor.  So
    ``n_samples`` is the longest lowered **line** geometry and
    ``trail_length_samples`` is scaled by the same cap ratio.

    Returns ``None`` when ``spec.animation`` is absent (so a static export is
    byte-identical to the pre-animation payload) or when the spec has no
    animatable **line** layer to reveal.  In the latter case — an animation *was*
    requested but the reference loader has no comet to play (a ``surface``-only
    spec) — a :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`
    warning is emitted so the animation is never *silently* dropped.
    """
    anim = spec.animation
    if anim is None:
        return None
    n_samples, n_original = _animated_sample_count(geometries)
    if n_samples < 2:
        _warn_unrevealable(spec)
        return None
    trail = _trail_length_samples(spec, anim)
    if trail is not None and n_original > 0 and n_samples < n_original:
        trail = max(2, int(round(trail * n_samples / n_original)))
    return {
        "fps": float(anim.fps),
        "duration": _playback_seconds(anim, n_samples),
        "n_frames": None if anim.n_frames is None else int(anim.n_frames),
        "loop": bool(anim.loop),
        "pingpong": bool(anim.pingpong),
        "trail_length_samples": trail,
        "head": bool(anim.head),
        "head_size": float(anim.head_size),
        "head_color": _head_color(anim.head_color),
        "n_samples": int(n_samples),
    }


def _playback_seconds(anim: Any, n_samples: int) -> float:
    """Delegate to :meth:`tsdynamics.viz.spec.Animation.playback_seconds` [M41].

    The reference loader has no frame clock: it traverses the whole series in
    ``metadata.animation.duration`` seconds.  The algebra that turns ``fps`` into
    that duration lives on ``Animation`` now, shared with the plotly export, so
    the two browser paths cannot drift.
    """
    return float(anim.playback_seconds(int(n_samples)))


def _warn_unrevealable(spec: PlotSpec) -> None:
    """Warn that an animated spec has no line geometry the threejs loader can reveal.

    Honors the "animates, **or warns** — never silently drops" contract: the
    threejs reveal comet sweeps a **line**, so a ``surface``-only animated spec has
    nothing to reveal.  Rather than emit a block the loader cannot play (which
    would freeze the export *and* the camera), the exporter drops the animation to
    a static payload and warns here.
    """
    kind = spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind)
    warnings.warn(
        f"threejs export: the animation on this {kind!r} spec was dropped — its "
        "reveal comet needs a line (LINE / LINE3D) geometry, but the spec has only "
        "surface layers. Exporting a static payload instead.",
        VisualizationDegraded,
        stacklevel=4,
    )


def _head_color(color: Any) -> list[float] | None:
    """Coerce the head color to a plain ``[r, g, b]`` list, or ``None``.

    Reuses :func:`_parse_rgb` (an RGB / RGBA sequence → a triple; a named-color
    string / ``None`` → ``None``) but returns a JSON-friendly ``list`` so the
    animation block stays plain-list / plain-float like the rest of the payload.
    """
    rgb = _parse_rgb(color)
    return None if rgb is None else [rgb[0], rgb[1], rgb[2]]


def _animated_sample_count(geometries: list[dict[str, Any]]) -> tuple[int, int]:
    """``(capped, original)`` vertices on the longest animatable geometry.

    The reveal length is a property of what was actually **exported**, so it is
    read off the lowered geometries.  ``(0, 0)`` when nothing is revealable.
    """
    n = 0
    n_original = 0
    for geom in geometries:
        # A points cloud has no chord to sweep, but the reference loader plays it
        # as a trailing swarm, so it counts as revealable alongside lines.
        if geom["type"] not in _REVEALABLE_GEOMETRY:
            continue
        if geom["n_vertices"] > n:
            n = int(geom["n_vertices"])
            n_original = int(geom["n_vertices_original"])
    return n, n_original


def _trail_length_samples(spec: PlotSpec, anim: Animation) -> int | None:
    """Resolve the comet tail length to a vertex count (``None`` ⇒ persistent).

    Reuses :meth:`~tsdynamics.viz.spec.Animation.tail_samples` (the same
    ``"time"`` / ``dt`` / ``"steps"`` rule the other backends use), reading the
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


def _round_flat(values: np.ndarray, decimals: int | None) -> list[float]:
    """Flatten an array to plain Python floats, optionally rounded.

    Returns plain floats (never NumPy scalars, never nested arrays), so the result
    is directly JSON-serializable and a ``Float32Array`` source on the frontend.
    Rounding to :data:`DEFAULT_DECIMALS` is well inside ``Float32`` precision at
    attractor scales and roughly halves the JSON text.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if decimals is not None:
        arr = np.round(arr, int(decimals))
    return [float(v) for v in arr]
