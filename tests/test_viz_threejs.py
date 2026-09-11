"""Tests for the ``threejs`` BufferGeometry data-export backend.

Stream VIZ-THREEJS-EXPORT.  Covers
:mod:`tsdynamics.viz.render.threejs` (the backend wiring + ``spec.render("threejs")``
payload + writing to a file) and :mod:`tsdynamics.viz.render.threejs._lower` (the
geometry lowering: flat positions / line-segment indices / surface triangulation /
scalar colors / bounds / camera).  Engine-free — no ``tsdynamics._rust`` import.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tsdynamics.viz.export import SCHEMA_VERSION
from tsdynamics.viz.render.caps import RenderResult, VisualizationDegraded
from tsdynamics.viz.spec import Animation, Axis, Layer, Layout, PlotKind, PlotSpec


def _lorenz_line3d_spec(n: int = 64) -> PlotSpec:
    """A 3-D Lorenz-like attractor spec: one LINE3D layer with a ``c`` channel."""
    t = np.linspace(0.0, 6.0, n)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[
            Layer(
                kind=PlotKind.LINE3D,
                data={"x": np.sin(t), "y": np.cos(t), "z": t, "c": t},
                label="orbit",
            )
        ],
        x=Axis(label="x", tickformat="m"),
        y=Axis(label="y"),
        z=Axis(label="z"),
        ndim=3,
    )


def _phase2d_scatter_spec(n: int = 32) -> PlotSpec:
    """A 2-D phase-portrait spec: one SCATTER layer (no z channel)."""
    t = np.linspace(0.0, 1.0, n)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(kind=PlotKind.SCATTER, data={"x": np.sin(t), "y": np.cos(t)})],
        x=Axis(label="x"),
        y=Axis(label="y"),
        ndim=2,
    )


def _surface_spec(rows: int = 5, cols: int = 4) -> PlotSpec:
    """A SURFACE3D spec over a rows x cols grid."""
    xs = np.linspace(-1.0, 1.0, cols)
    ys = np.linspace(-1.0, 1.0, rows)
    xg, yg = np.meshgrid(xs, ys)
    zg = xg**2 - yg**2
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[Layer(kind=PlotKind.SURFACE3D, data={"x": xg, "y": yg, "z": zg})],
        ndim=3,
    )


def _composite_spec(mode: str = "row") -> PlotSpec:
    """A 2-panel COMPOSITE spec: a 3-D line panel + a 2-D scatter panel."""
    p1 = _lorenz_line3d_spec(48)
    p1.title = "attractor"
    p2 = _phase2d_scatter_spec(24)
    p2.title = "section"
    return PlotSpec(
        kind=PlotKind.COMPOSITE,
        panels=[p1, p2],
        layout=Layout(mode=mode),  # type: ignore[arg-type]
        title="figure",
    )


# ---------------------------------------------------------------------------
# geometry lowering
# ---------------------------------------------------------------------------


def test_line3d_lowers_to_line_geometry() -> None:
    """A LINE3D spec yields one 'line' geometry: positions 3n and NO index buffer.

    Schema v3 drops the ``0,1,1,2,2,3,...`` ``LineSegments`` index for lines: a
    polyline's vertex order *is* its draw order, so the loader draws a contiguous
    ``THREE.Line`` instead.  Same picture, 12.9% fewer payload bytes, half the GPU
    index work.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 64
    payload = lower_spec(_lorenz_line3d_spec(n))
    assert len(payload["geometries"]) == 1
    geom = payload["geometries"][0]
    assert geom["type"] == "line"
    assert geom["label"] == "orbit"
    assert len(geom["positions"]) == 3 * n
    assert geom["indices"] == []
    assert geom["n_vertices"] == n
    assert geom["n_vertices_original"] == n


def test_phase2d_scatter_lowers_to_points_at_z0() -> None:
    """A 2-D SCATTER lowers to a 'points' geometry lifted to z=0 (no indices)."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 32
    payload = lower_spec(_phase2d_scatter_spec(n))
    geom = payload["geometries"][0]
    assert geom["type"] == "points"
    assert geom["indices"] == []
    assert len(geom["positions"]) == 3 * n
    # Every z (index 2, 5, 8, ...) is exactly 0.0.
    z_vals = geom["positions"][2::3]
    assert z_vals == [0.0] * n


def test_surface_triangulates_the_grid() -> None:
    """A SURFACE3D lowers to a 'surface' geometry: 6 indices per quad."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    rows, cols = 5, 4
    payload = lower_spec(_surface_spec(rows, cols))
    geom = payload["geometries"][0]
    assert geom["type"] == "surface"
    assert len(geom["positions"]) == 3 * rows * cols
    # Two triangles (6 indices) per (rows-1)*(cols-1) quad.
    assert len(geom["indices"]) == 6 * (rows - 1) * (cols - 1)
    assert max(geom["indices"]) < rows * cols


def test_scalar_c_channel_travels_as_one_float_per_vertex() -> None:
    """A 'c' channel travels as n SCALARS, not 3n pre-expanded RGB floats.

    Schema v3: expanding one scalar into three floats per vertex was 38.6% of the
    payload, and the colour ramp is a *presentation* choice that belongs to the
    renderer.  The loader owns the ramp now (``colormap`` / ``scalarColors``).
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 64
    geom = lower_spec(_lorenz_line3d_spec(n))["geometries"][0]
    assert "colors" not in geom
    assert len(geom["c"]) == n
    assert geom["c"] == pytest.approx(np.linspace(0.0, 6.0, n), abs=1e-4)


def test_uncolored_layer_takes_color_from_theme_palette() -> None:
    """A layer with no 'c' channel and no rgb style is colored from the theme palette.

    Under the styling overhaul a Theme carries a ``palette`` and renderers color a
    layer that supplies no explicit color from it.  For the threejs exporter that
    means the layer's flat (solid) color lands on ``material.color``.

    A *solid* colour rides on ``material.color`` alone; the per-vertex channel is
    reserved for a genuine scalar ``c`` gradient.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    geom = lower_spec(_phase2d_scatter_spec())["geometries"][0]
    # The uncolored layer now inherits a deterministic per-layer palette color.
    assert geom["material"]["color"] is not None
    assert isinstance(geom["material"]["color"], str)
    # …and the flat color rides on ``material.color`` ALONE — a solid (palette)
    # color emits no redundant per-vertex buffer (that channel is reserved for the
    # scalar ``c``-channel gradient case).
    assert "colors" not in geom
    assert "c" not in geom


# ---------------------------------------------------------------------------
# metadata: schema_version / bounds / camera / labels / units
# ---------------------------------------------------------------------------


def test_metadata_has_schema_bounds_camera() -> None:
    """The payload + metadata carry schema_version, bounds, and a full camera."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_lorenz_line3d_spec())
    assert payload["schema_version"] == SCHEMA_VERSION
    meta = payload["metadata"]
    assert meta["schema_version"] == SCHEMA_VERSION
    for axis in ("x", "y", "z"):
        lo, hi = meta["bounds"][axis]
        assert lo <= hi
    cam = meta["camera"]
    for key in ("position", "target", "up"):
        assert len(cam[key]) == 3
    # Target sits at the bounds centre.
    bx = meta["bounds"]["x"]
    assert cam["target"][0] == pytest.approx((bx[0] + bx[1]) / 2.0)


def test_labels_surface_from_axes_and_units_is_gone() -> None:
    """Axis labels reach metadata; the misnamed ``units`` block does not exist.

    ``metadata.units`` carried each axis's *tickformat* — a number-format string,
    not a physical unit — under a key that invited exactly the wrong reading, and
    the reference loader never read it.  Schema v3 removes it rather than ship a
    field that is both dead and misleading.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    meta = lower_spec(_lorenz_line3d_spec())["metadata"]
    assert meta["labels"] == {"x": "x", "y": "y", "z": "z"}
    assert "units" not in meta


def test_camera_override_from_meta() -> None:
    """A spec.meta['camera'] overrides the derived camera."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    spec = _lorenz_line3d_spec()
    spec.meta["camera"] = {
        "position": [10.0, 20.0, 30.0],
        "target": [1.0, 2.0, 3.0],
        "up": [0.0, 1.0, 0.0],
    }
    cam = lower_spec(spec)["metadata"]["camera"]
    assert cam["position"] == [10.0, 20.0, 30.0]
    assert cam["target"] == [1.0, 2.0, 3.0]
    assert cam["up"] == [0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# JSON-readiness: flat plain floats, round-trips
# ---------------------------------------------------------------------------


def test_positions_are_plain_floats_never_nested() -> None:
    """Flat positions are plain Python floats (Float32-ready), never nested."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    geom = lower_spec(_lorenz_line3d_spec())["geometries"][0]
    assert all(isinstance(v, float) for v in geom["positions"])
    assert all(isinstance(i, int) for i in geom["indices"])


def test_payload_round_trips_through_json() -> None:
    """The whole payload survives json.dumps / json.loads unchanged."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_lorenz_line3d_spec())
    reloaded = json.loads(json.dumps(payload))
    assert reloaded == payload


# ---------------------------------------------------------------------------
# the threejs renderer backend
# ---------------------------------------------------------------------------


def test_threejs_backend_registers_unconditionally() -> None:
    """The threejs backend self-wires via register_builtin_renderers (pure python)."""
    from tsdynamics import registry
    from tsdynamics.viz.render import register_builtin_renderers

    register_builtin_renderers()
    assert "threejs" in registry.renderers
    caps = registry.renderers.get("threejs").capabilities
    assert caps.data_export is True
    assert caps.web_export is True
    assert caps.supports_3d is True
    assert caps.kinds is None  # accepts every kind


def test_render_threejs_returns_payload() -> None:
    """``spec.render('threejs')`` returns a RenderResult carrying the dict payload."""
    spec = _lorenz_line3d_spec()
    result = spec.render("threejs")
    assert isinstance(result, RenderResult)
    assert result.backend == "threejs"
    assert result.mimetype == "application/json"
    assert result.kind is PlotKind.PHASE_PORTRAIT_3D
    assert isinstance(result.payload, dict)
    assert result.payload["geometries"][0]["type"] == "line"


def test_render_threejs_raw_returns_dict() -> None:
    """``raw=True`` returns the bare payload dict instead of a RenderResult."""
    payload = _phase2d_scatter_spec().render("threejs", raw=True)
    assert isinstance(payload, dict)
    assert payload["geometries"][0]["type"] == "points"


def test_render_threejs_2d_phase_portrait_exports() -> None:
    """A 2-D phase portrait exports just like the 3-D attractor (lifted to z=0)."""
    payload = _phase2d_scatter_spec().render("threejs", raw=True)
    assert payload["metadata"]["bounds"]["z"] == [0.0, 0.0]


def test_render_threejs_writes_file(tmp_path) -> None:
    """Passing a ``path`` writes the payload as JSON to disk and returns the path."""
    spec = _lorenz_line3d_spec()
    out = tmp_path / "geometry.json"
    returned = spec.render("threejs", path=out)
    assert returned == out
    assert out.exists()
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == spec.render("threejs", raw=True)


def test_threejs_does_not_shadow_drawing_default() -> None:
    """Default ``spec.render()`` prefers a drawing backend over the threejs exporter.

    The threejs exporter declares ``data_export=True`` / ``kinds=None``, so a naive
    "first capable" default could pick it and return a payload instead of a figure.
    The dispatch skips data-export backends in default selection, so a default
    render returns a matplotlib figure; ``render('threejs')`` by name still reaches
    the exporter.
    """
    matplotlib = pytest.importorskip("matplotlib")
    from tsdynamics import registry
    from tsdynamics.viz.render import register_builtin_renderers

    saved = list(registry.renderers.all())
    registry.renderers.clear()
    try:
        register_builtin_renderers()
        if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present
            pytest.skip("matplotlib backend did not register")
        # A 2-D spec the matplotlib backend can draw (its 3-D drawing is a follow-up).
        spec = _phase2d_scatter_spec()
        assert isinstance(spec.render(), matplotlib.figure.Figure)
        # …while the exporter is still reachable by name.
        assert isinstance(spec.render("threejs"), RenderResult)
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj)


# ---------------------------------------------------------------------------
# animation: the reveal directive in metadata (issue #465)
# ---------------------------------------------------------------------------


def _animated_line3d_spec(n: int = 64, anim: Animation | None = None) -> PlotSpec:
    """A 3-D LINE3D spec carrying an :class:`Animation` (a reveal comet)."""
    spec = _lorenz_line3d_spec(n)
    spec.animation = anim if anim is not None else Animation()
    spec.meta["dt"] = 0.01
    return spec


def test_animated_spec_adds_animation_block() -> None:
    """An animated spec's payload carries a ``metadata.animation`` block."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_animated_line3d_spec(64))
    anim = payload["metadata"]["animation"]
    expected_keys = {
        "fps",
        "duration",
        "n_frames",
        "loop",
        "pingpong",
        "trail_length_samples",
        "head",
        "head_size",
        "head_color",
        "n_samples",
    }
    assert set(anim) == expected_keys
    assert anim["fps"] == 30.0
    assert anim["head"] is True
    assert anim["n_samples"] == 64  # the line's vertex count
    # A persistent (default) trail is null, never a number.
    assert anim["trail_length_samples"] is None


def test_static_spec_has_no_animation_block() -> None:
    """A spec without an Animation carries no ``animation`` key (byte-stable)."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_lorenz_line3d_spec(64))
    assert "animation" not in payload["metadata"]


def test_animation_does_not_touch_geometry() -> None:
    """Adding an Animation leaves the geometry buffers byte-identical (draw-range reveal)."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    static = lower_spec(_lorenz_line3d_spec(64))
    animated = lower_spec(_animated_line3d_spec(64))
    # The only payload difference is the metadata.animation block.
    assert animated["geometries"] == static["geometries"]
    assert animated["metadata"]["bounds"] == static["metadata"]["bounds"]
    assert animated["metadata"]["camera"] == static["metadata"]["camera"]


def test_trail_length_resolves_to_samples() -> None:
    """A ``steps`` trail surfaces as an int sample count; ``time`` divides by dt."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    steps = lower_spec(_animated_line3d_spec(64, Animation(trail_kind="steps", trail_length=20)))
    assert steps["metadata"]["animation"]["trail_length_samples"] == 20

    # "time" trail of 0.2 time-units at dt=0.01 → 20 samples.
    timed = lower_spec(_animated_line3d_spec(64, Animation(trail_kind="time", trail_length=0.2)))
    assert timed["metadata"]["animation"]["trail_length_samples"] == 20


def test_head_color_rgb_surfaces_as_triple() -> None:
    """An RGB ``head_color`` surfaces as an [r, g, b] triple (a named color → null)."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    rgb = lower_spec(_animated_line3d_spec(64, Animation(head_color=(1.0, 0.0, 0.0))))
    assert rgb["metadata"]["animation"]["head_color"] == [1.0, 0.0, 0.0]

    named = lower_spec(_animated_line3d_spec(64, Animation(head_color="red")))
    assert named["metadata"]["animation"]["head_color"] is None


def test_animation_block_round_trips_through_json() -> None:
    """The animated payload (block included) survives json.dumps / json.loads."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_animated_line3d_spec(64))
    assert json.loads(json.dumps(payload)) == payload


def test_animated_2d_phase_portrait_animates() -> None:
    """A 2-D LINE spec (lifted to z=0) also carries the animation block."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 40
    t = np.linspace(0.0, 1.0, n)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(kind=PlotKind.LINE, data={"x": np.sin(t), "y": np.cos(t)})],
        x=Axis(label="x"),
        y=Axis(label="y"),
        ndim=2,
        animation=Animation(),
    )
    anim = lower_spec(spec)["metadata"]["animation"]
    assert anim["n_samples"] == n


def test_surface_only_animation_warns_and_emits_no_block() -> None:
    """An animated surface-only spec → no block + a VisualizationDegraded warning."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    spec = _surface_spec()
    spec.animation = Animation()
    with pytest.warns(VisualizationDegraded):
        payload = lower_spec(spec)
    assert "animation" not in payload["metadata"]


def test_points_only_animation_emits_a_block_the_loader_can_play() -> None:
    """An animated SCATTER/MARKERS-only spec DOES animate — as a trailing swarm.

    Defect found while capping (v6): the exporter refused to emit an animation
    block for a points-only spec, on the premise that the loader's reveal needs a
    line index buffer.  That premise was false — the reference loader has carried
    ``buildPointsComet`` (a ``setDrawRange`` swarm in vertex units) since the
    animation stream landed, and ``test_loader_js_drives_setdrawrange_reveal``
    below asserts it.  So the exporter was silently dropping an animation the
    renderer could play.  Only a ``surface``-only spec has nothing to sweep.
    """
    import warnings

    from tsdynamics.viz.render.threejs._lower import lower_spec

    for mark in (PlotKind.SCATTER, PlotKind.MARKERS):
        n = 32
        t = np.linspace(0.0, 1.0, n)
        spec = PlotSpec(
            kind=PlotKind.PHASE_PORTRAIT_3D,
            layers=[Layer(kind=mark, data={"x": np.sin(t), "y": np.cos(t), "z": t})],
            x=Axis(label="x"),
            y=Axis(label="y"),
            z=Axis(label="z"),
            ndim=3,
            animation=Animation(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisualizationDegraded)
            payload = lower_spec(spec)
        assert payload["metadata"]["animation"]["n_samples"] == n
        assert payload["geometries"][0]["type"] == "points"


def test_static_points_spec_does_not_warn() -> None:
    """A points spec with NO Animation lowers silently (no spurious warning)."""
    import warnings

    from tsdynamics.viz.render.threejs._lower import lower_spec

    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        payload = lower_spec(_phase2d_scatter_spec(32))
    assert "animation" not in payload["metadata"]


def test_to_plot_spec_animate_flag_drives_threejs_payload() -> None:
    """``to_plot_spec(animate=True)`` → animation block; ``animate=False`` → none."""
    pytest.importorskip("tsdynamics._rust")
    import tsdynamics as ts

    tr = ts.systems.Lorenz().run(final_time=10.0, dt=0.01).after(2.0)
    animated = tr.to_plot_spec(animate=True).render("threejs", raw=True)
    static = tr.to_plot_spec(animate=False).render("threejs", raw=True)
    assert "animation" in animated["metadata"]
    assert "animation" not in static["metadata"]
    # The animation toggle never perturbs the geometry buffers.
    assert animated["geometries"] == static["geometries"]


def test_loader_js_drives_setdrawrange_reveal() -> None:
    """The reference loader honors metadata.animation via a reveal comet (string check)."""
    import re
    from pathlib import Path

    loader = Path(__file__).resolve().parents[1] / "docs" / "_static" / "tsdyn-threejs-loader.js"
    src = loader.read_text(encoding="utf-8")
    # The animation path keys off metadata.animation and reveals a comet per geometry.
    assert "meta.animation" in src
    assert "setDrawRange" in src
    # Both LINE and POINTS geometries get a comet (mirrors _ANIMATED_MARKS + the
    # map-iterate point cloud); each exposes the same seek(headVertex, trailVertices)
    # contract that installAnimation drives every frame.
    assert "buildLineComet" in src
    assert "buildPointsComet" in src
    assert 'geom.type === "line"' in src
    assert "function seek(" in src
    # LINE reveal: a fixed-length windowed THREE.Line whose positions AND per-vertex
    # colours are rewritten each seek() — a glowing teal trail fading tail→head — rather
    # than a flat index-unit draw-range slice.  The per-vertex colours are the tell.
    assert "vertexColors: true" in src
    # POINTS reveal: an unindexed Points geometry, so setDrawRange counts VERTICES
    # directly — setDrawRange(lo, …), never a 2*-index window (that mixup would draw a
    # wrong-length swarm).
    assert re.search(r"setDrawRange\(\s*lo\b", src)
    # A defensive animation-block-but-no-comet payload must NOT freeze the camera:
    # autoRotate keys off whether a comet was actually installed, not on `anim`.
    assert "autoRotate = opts.autoRotate ?? !revealing" in src


# composite (multi-panel) export — issue #460
# ---------------------------------------------------------------------------


def test_composite_exports_two_nonempty_panel_groups() -> None:
    """A 2-panel composite yields 2 panel groups, each with non-empty geometry.

    Regression guard for the empty-payload bug: a composite carries its content
    in ``panels``, not ``layers``, so the bare layer-walk used to find nothing.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_composite_spec())
    assert payload["kind"] == PlotKind.COMPOSITE.value
    # The composite emits a ``panels`` block (and no top-level geometry).
    assert payload["geometries"] == []
    panels = payload["panels"]
    assert len(panels) == 2
    for panel in panels:
        assert len(panel["geometries"]) >= 1
        # Each panel group carries real vertices, not an empty buffer.
        assert all(len(geom["positions"]) > 0 for geom in panel["geometries"])


def test_composite_panels_carry_identity_and_grid() -> None:
    """Each panel group carries its index / title / kind and a grid cell."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_composite_spec(mode="row"))
    panels = payload["panels"]
    assert [p["index"] for p in panels] == [0, 1]
    assert [p["title"] for p in panels] == ["attractor", "section"]
    assert panels[0]["kind"] == PlotKind.PHASE_PORTRAIT_3D.value
    assert panels[1]["kind"] == PlotKind.PHASE_PORTRAIT_2D.value
    # A "row" layout places the two panels side by side: same row, cols 0 and 1.
    assert panels[0]["grid"] == {"row": 0, "col": 0}
    assert panels[1]["grid"] == {"row": 0, "col": 1}
    # Every panel carries an [x, y, z] layout offset.
    for panel in panels:
        assert len(panel["offset"]) == 3


def test_composite_metadata_carries_layout_and_bounds() -> None:
    """The composite metadata block carries the resolved layout + aggregate bounds."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    meta = lower_spec(_composite_spec(mode="row"))["metadata"]
    assert meta["schema_version"] == SCHEMA_VERSION
    layout = meta["layout"]
    assert layout["mode"] == "row"
    assert layout["rows"] == 1
    assert layout["cols"] == 2
    # The aggregate bounds frame both panels (offsets applied), so they are wider
    # than a single panel's x-extent.
    bx = meta["bounds"]["x"]
    assert bx[0] <= bx[1]
    cam = meta["camera"]
    for key in ("position", "target", "up"):
        assert len(cam[key]) == 3


def test_composite_stack_grid_is_one_column() -> None:
    """A 'stack' composite resolves to a single-column grid (n rows, 1 col)."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_composite_spec(mode="stack"))
    layout = payload["metadata"]["layout"]
    assert (layout["rows"], layout["cols"]) == (2, 1)
    assert payload["panels"][0]["grid"] == {"row": 0, "col": 0}
    assert payload["panels"][1]["grid"] == {"row": 1, "col": 0}


def test_composite_panel_positions_stay_local() -> None:
    """A panel's geometry positions are its own local coords (unshifted).

    The placement lives in the panel's ``offset`` / ``grid`` hints; the geometry
    itself is never mutated, so a frontend can render a panel in its own viewport.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    standalone = lower_spec(_lorenz_line3d_spec(48))
    payload = lower_spec(_composite_spec(mode="row"))
    panel0 = payload["panels"][0]
    # Panel 0 is the same line3d spec → identical local positions.
    assert panel0["geometries"][0]["positions"] == standalone["geometries"][0]["positions"]


def test_composite_payload_round_trips_through_json() -> None:
    """The whole composite payload survives json.dumps / json.loads unchanged."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_composite_spec())
    reloaded = json.loads(json.dumps(payload))
    assert reloaded == payload


def test_composite_positions_are_plain_floats() -> None:
    """Composite panel positions / indices are plain Python floats / ints."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    panels = lower_spec(_composite_spec())["panels"]
    for panel in panels:
        for geom in panel["geometries"]:
            assert all(isinstance(v, float) for v in geom["positions"])
            assert all(isinstance(i, int) for i in geom["indices"])
        assert all(isinstance(v, float) for v in panel["offset"])


def test_render_threejs_composite_writes_panelled_payload(tmp_path) -> None:
    """``render('threejs', path=...)`` on a composite writes the panelled payload.

    Per VIZ-CHSH, ``.save('.json')`` routes to the JSON spec backend, not threejs;
    selecting the threejs backend by name writes the BufferGeometry payload.
    """
    spec = _composite_spec()
    out = tmp_path / "fig.json"
    returned = spec.render("threejs", path=out)
    assert returned == out
    assert out.exists()
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == spec.render("threejs", raw=True)
    # The written payload is the panelled (non-empty) one, not an empty geometry.
    assert len(reloaded["panels"]) == 2
    assert all(len(p["geometries"]) >= 1 for p in reloaded["panels"])


def test_render_threejs_composite_returns_payload() -> None:
    """``spec.render('threejs')`` on a composite returns a RenderResult payload."""
    result = _composite_spec().render("threejs")
    assert isinstance(result, RenderResult)
    assert result.backend == "threejs"
    assert result.kind is PlotKind.COMPOSITE
    assert len(result.payload["panels"]) == 2


def test_single_panel_export_unchanged_by_composite_path() -> None:
    """Regression guard: a single-panel spec still lowers to the flat payload.

    The composite recursion must not alter the non-composite payload — it carries
    a top-level ``geometries`` block and no ``panels`` key.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_lorenz_line3d_spec(64))
    assert payload["kind"] == PlotKind.PHASE_PORTRAIT_3D.value
    assert "panels" not in payload
    assert len(payload["geometries"]) == 1
    assert payload["geometries"][0]["type"] == "line"


# ---------------------------------------------------------------------------
# the vertex cap — stream VIZ-WEB-EXPORT (H2)
# ---------------------------------------------------------------------------


def _long_curve_spec(n: int, *, animate: bool = False) -> PlotSpec:
    """A synthetic 3-D curve with ``n`` vertices and a per-vertex ``c`` channel."""
    t = np.linspace(0.0, 400.0, n)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        layers=[
            Layer(
                kind=PlotKind.LINE3D,
                data={"x": 10 * np.sin(t), "y": 10 * np.cos(t), "z": t, "c": t},
                label="orbit",
            )
        ],
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z"),
        animation=Animation() if animate else None,
    )


def test_million_sample_payload_stays_under_five_megabytes() -> None:
    """A 1e6-vertex curve exports well under 5 MB (uncapped it was ~137 MB).

    A hard byte assertion — it cannot flake, and it is the whole reason the cap
    exists: an uncapped export is a page no browser can load.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    with pytest.warns(VisualizationDegraded):
        payload = lower_spec(_long_curve_spec(1_000_000))
    size = len(json.dumps(payload))
    assert size < 5_000_000, f"payload is {size / 1e6:.1f} MB"
    assert payload["metadata"]["resample"] == {
        "max_points": 40_000,
        "original_vertices": 1_000_000,
        "vertices": 40_000,
        "capped": True,
    }


def test_cap_warns_exactly_once_naming_both_counts() -> None:
    """Capping emits ONE VisualizationDegraded naming the original and kept counts."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    with pytest.warns(VisualizationDegraded) as record:
        lower_spec(_long_curve_spec(120_000))
    degraded = [w for w in record if issubclass(w.category, VisualizationDegraded)]
    assert len(degraded) == 1
    msg = str(degraded[0].message)
    assert "120000" in msg and "40000" in msg
    assert "max_points" in msg


def test_max_points_none_exports_every_vertex_and_does_not_warn() -> None:
    """``max_points=None`` is a genuine opt-out: no thinning, no warning."""
    import warnings

    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 60_000
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        payload = lower_spec(_long_curve_spec(n), max_points=None)
    geom = payload["geometries"][0]
    assert geom["n_vertices"] == n
    assert len(geom["positions"]) == 3 * n
    assert payload["metadata"]["resample"]["max_points"] is None
    assert payload["metadata"]["resample"]["capped"] is False


def test_under_the_cap_is_untouched() -> None:
    """A curve below the ceiling is exported vertex-for-vertex, silently."""
    import warnings

    from tsdynamics.viz.render.threejs._lower import lower_spec

    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        payload = lower_spec(_long_curve_spec(1000))
    assert payload["geometries"][0]["n_vertices"] == 1000
    assert payload["metadata"]["resample"]["capped"] is False


def test_capping_keeps_positions_and_c_channel_aligned() -> None:
    """The ``c`` channel is resampled on the SAME parameterisation as the positions.

    A channel thinned independently would silently de-register from the vertices it
    colours — the attractor would still draw, in the wrong colours.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    with pytest.warns(VisualizationDegraded):
        geom = lower_spec(_long_curve_spec(100_000), max_points=5_000)["geometries"][0]
    assert geom["n_vertices"] == 5_000
    assert len(geom["positions"]) == 3 * 5_000
    assert len(geom["c"]) == 5_000
    # The channel is monotone in the source, so it must stay monotone after an
    # arc-length resample that shares its parameterisation.
    assert np.all(np.diff(geom["c"]) >= -1e-9)


def test_points_cloud_is_thinned_deterministically_not_by_stride() -> None:
    """A points cloud caps by a seeded uniform draw — reproducible, not a stride.

    A stride would alias a resonant map's orbit onto a handful of points; a seeded
    draw preserves the invariant density that IS the attractor, and being seeded it
    makes the export byte-reproducible.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 50_000
    t = np.linspace(0.0, 1.0, n)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        layers=[Layer(kind=PlotKind.SCATTER, data={"x": np.sin(t), "y": np.cos(t), "z": t})],
    )
    with pytest.warns(VisualizationDegraded):
        first = lower_spec(spec)["geometries"][0]
    with pytest.warns(VisualizationDegraded):
        second = lower_spec(spec)["geometries"][0]
    assert first["n_vertices"] == 40_000
    assert first["positions"] == second["positions"]
    # Not a stride: a stride of a linspace keeps an exactly uniform z spacing.
    zs = np.asarray(first["positions"][2::3])
    assert float(np.std(np.diff(zs))) > 0.0


def test_animation_counts_follow_the_capped_geometry() -> None:
    """After capping, ``n_samples`` / the trail are reported in CAPPED vertices.

    A reveal sized against the uncapped count would index past the buffer and stall
    the comet; a trail measured in uncapped samples would sweep the wrong fraction
    of the attractor.
    """
    from tsdynamics.viz.render.threejs._lower import lower_spec

    spec = _long_curve_spec(200_000, animate=True)
    spec.trail(("steps", 1000))
    with pytest.warns(VisualizationDegraded):
        payload = lower_spec(spec)
    anim = payload["metadata"]["animation"]
    assert anim["n_samples"] == 40_000 == payload["geometries"][0]["n_vertices"]
    # 1000 samples of 200 000 is 0.5% of the curve; of 40 000 that is 200.
    assert anim["trail_length_samples"] == 200


@pytest.mark.slow
def test_arclength_cap_holds_the_sagitta_target_on_the_fastest_attractors() -> None:
    """The capped curve stays visually smooth — and a STRIDE would not.

    The sagitta (bow of the curve off its local chord, over the bounding-box
    diagonal) is the scale-free readability criterion.  This is the test that
    forbids a future "optimisation" back to ``y[::k]``: on HyperQi, a stride at the
    same vertex budget is ~35x worse and reads as a chorded polygon.
    """
    pytest.importorskip("tsdynamics._rust")
    import tsdynamics as ts
    from tsdynamics.viz._resample import max_sagitta_ratio, resample_arclength

    cap = 40_000
    for name in ("HyperQi", "DequanLi", "ZhouChen", "QiChen", "Lorenz", "Rossler"):
        system = getattr(ts.systems, name)()
        traj = system.run(final_time=90.0, dt=0.001, ic=system.resolve_ic(None), solver="rk45")
        y = np.ascontiguousarray(traj.y[len(traj.y) // 5 :, :3], dtype=float)
        arc, _ = resample_arclength(y, cap)
        sag_arc = max_sagitta_ratio(arc)
        assert sag_arc < 0.008, f"{name}: arc-length sagitta {sag_arc:.4f}"
        if name == "HyperQi":
            stride = y[:: max(1, -(-len(y) // cap))][:cap]
            assert max_sagitta_ratio(stride) > 20 * sag_arc


@pytest.mark.slow
def test_sagitta_follows_the_documented_chord_rule_at_a_harsh_thinning_ratio() -> None:
    """The budget rule in the docs is a formula, and this is what pins it.

    The test above measures at a *benign* operating point (a 90-time-unit trace is
    only ~1.8x over the cap), which is not the regime the cap exists for.  Thin the
    same attractors 16x and the honest picture appears: the worst-case sagitta is
    **not** a constant of the attractor but a function of the resampled chord length
    in bounding-box units, ``h = L / (D * n)``.

    Two regimes, both asserted here because the docs promise both:

    - *resolution-limited* (smooth attractors): sagitta ~ ``h**2``, so Lorenz stays
      an order of magnitude inside target even at 16x thinning;
    - *curvature-limited* (cusp-like hyperchaos): sagitta ~ ``h``, so HyperQi
      genuinely **exceeds** the 0.008 target at the default cap — which is why
      ``max_points`` is a parameter and why the docs say to spend vertices rather
      than to change the resampling.

    A future change that made the resample cheaper would break the ``h``
    proportionality here long before it became visible in a rendered page.
    """
    pytest.importorskip("tsdynamics._rust")
    import tsdynamics as ts
    from tsdynamics.viz._resample import max_sagitta_ratio, resample_arclength

    def chord_and_sagitta(name: str, n: int) -> tuple[float, float]:
        system = getattr(ts.systems, name)()
        traj = system.run(final_time=400.0, dt=0.0005, ic=system.resolve_ic(None), solver="rk45")
        y = np.ascontiguousarray(traj.y[len(traj.y) // 5 :, :3], dtype=float)
        length = float(np.linalg.norm(np.diff(y, axis=0), axis=1).sum())
        diagonal = float(np.linalg.norm(y.max(axis=0) - y.min(axis=0)))
        arc, _ = resample_arclength(y, n)
        return length / (diagonal * n), max_sagitta_ratio(arc)

    # Curvature-limited: sagitta tracks h to within a factor of ~2, and the default
    # cap is genuinely NOT enough.  Pinning the overrun stops anyone "fixing" the
    # docs by quietly loosening the target instead of spending vertices.
    h40, sag40 = chord_and_sagitta("HyperQi", 40_000)
    assert 0.02 < sag40 / h40 < 2.0, f"HyperQi sagitta/h = {sag40 / h40:.3f}"
    assert sag40 > 0.008, f"HyperQi at the default cap should overrun, got {sag40:.4f}"

    # …and spending vertices per the documented rule brings it back on target.
    _, sag200 = chord_and_sagitta("HyperQi", 200_000)
    assert sag200 <= 0.0085, f"HyperQi at 200k should be on target, got {sag200:.4f}"

    # Resolution-limited: quadratic, so the same 16x thinning is comfortably inside.
    _, lorenz40 = chord_and_sagitta("Lorenz", 40_000)
    assert lorenz40 < 0.008, f"Lorenz sagitta {lorenz40:.5f}"


# ---------------------------------------------------------------------------
# the loader asset — it must SHIP, and it must not drift from the docs copy
# ---------------------------------------------------------------------------


def test_reference_loader_ships_inside_the_package() -> None:
    """The reference loader is package data, not a docs file.

    Before v6 it lived only under ``docs/`` — so it was in no wheel at all, while
    the documentation told installed users to "copy this loader".
    """
    from tsdynamics.viz.render.threejs import loader_path, loader_source

    assert loader_path().is_file()
    src = loader_source()
    assert "export function renderThreejsPayload" in src
    assert len(src) > 10_000


def test_packaged_loader_is_byte_identical_to_the_docs_copy() -> None:
    """The docs mirror and the packaged loader can never drift.

    ``docs/_static/tsdyn-threejs-loader.js`` is a mirror (the docs site link and the
    style-honoring contract test both read it); the package copy is the source of
    truth.  Two copies of a 700-line file WILL diverge unless something fails when
    they do.
    """
    from pathlib import Path

    from tsdynamics.viz.render.threejs import loader_path

    docs_copy = Path(__file__).resolve().parents[1] / "docs" / "_static" / loader_path().name
    assert docs_copy.read_bytes() == loader_path().read_bytes()


def test_write_loader_asset_emits_the_shared_copy(tmp_path) -> None:
    """``write_loader_asset`` is the companion to ``assets='link'``."""
    from tsdynamics.viz.render.threejs import loader_source, write_loader_asset

    out = write_loader_asset(tmp_path / "static")
    assert out.name == "tsdyn-threejs-loader.js"
    assert out.read_text(encoding="utf-8") == loader_source()


def test_loader_reads_the_scalar_c_channel_and_unindexed_lines() -> None:
    """The loader honors schema v3: a scalar ``c`` ramp and index-free lines."""
    from tsdynamics.viz.render.threejs import loader_source

    src = loader_source()
    # v3: expand the scalar channel in the browser…
    assert "function scalarColors(" in src
    assert "geom.c &&" in src
    # …while still accepting a legacy v2 pre-expanded RGB payload.
    assert "geom.colors &&" in src
    # An index-free line falls through to a contiguous THREE.Line.
    assert "new THREE.Line(geometry, lineMat)" in src


def test_loader_sizes_itself_with_a_resize_observer_and_idles_offscreen() -> None:
    """The loader observes its CONTAINER, not the window, and idles when hidden.

    A `window.resize` listener never fires for a container revealed by a tab / an
    accordion — the single most common embedding context — so such a viewer used to
    render permanently at the 640x420 fallback size.  The *behaviour* is verified in
    a real headless browser by
    ``test_a_container_hidden_at_boot_resizes_when_revealed`` below; this test
    asserts the mechanism structurally so it cannot be refactored away silently in
    an environment where no browser is available to catch it.
    """
    from tsdynamics.viz.render.threejs import loader_source

    src = loader_source()
    assert "new ResizeObserver(" in src
    assert "resizeObserver.observe(container)" in src
    assert "new IntersectionObserver(" in src
    # dispose() must actually tear down: both loops, both observers, the GPU buffers.
    assert "cancelAnimationFrame" in src
    assert "resizeObserver.disconnect()" in src
    assert "intersectionObserver.disconnect()" in src
    assert "disposeObject(scene)" in src
    assert "MAX_PIXEL_RATIO" in src


# ---------------------------------------------------------------------------
# the self-contained HTML page — the actual web deliverable
# ---------------------------------------------------------------------------


def _viewer_page(spec: PlotSpec, **kw) -> str:
    """Render ``spec`` to the viewer page HTML (suppressing the cap warning)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VisualizationDegraded)
        page = spec.render("threejs", html=True, **kw)
    assert isinstance(page, str)
    return page


def test_save_html_writes_a_real_page_not_a_json_blob(tmp_path) -> None:
    """``render('threejs', path='x.html')`` writes HTML.

    It used to write a JSON document into a ``.html`` file — a file no browser
    would render, from the backend whose entire purpose is web embedding.
    """
    out = tmp_path / "attractor.html"
    spec = _lorenz_line3d_spec(256)
    written = spec.render("threejs", path=out, poster=False)
    assert written == out
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html")
    assert not text.startswith('{"schema_version"')


def test_emitted_page_is_self_contained(tmp_path) -> None:
    """The page makes ZERO same-origin requests: loader and payload are inlined.

    That is what lets the artifact open from ``file://``, from a zip, or pasted
    into a CMS.  The one permitted external reference is the pinned three.js build
    in the ES import map — a library we do not vendor.
    """
    import re

    page = _viewer_page(_lorenz_line3d_spec(256), poster=False)
    assert "renderThreejsPayload" in page  # the loader source itself
    assert 'id="tsdyn-payload"' in page  # the inlined geometry
    refs = [
        url for url in re.findall(r'(?:src|href)="([^"]+)"', page) if not url.startswith("data:")
    ]
    assert refs == [], f"page reaches out to {refs}"
    assert "cdn.jsdelivr.net/npm/three@" in page  # the pinned import map


def test_emitted_page_carries_a_poster_and_a_noscript_fallback() -> None:
    """A reader with no WebGL / no CDN / no JavaScript still sees the attractor."""
    pytest.importorskip("matplotlib")
    page = _viewer_page(_lorenz_line3d_spec(256), poster=True)
    assert "data:image/png;base64," in page
    assert "<noscript>" in page


def test_page_escapes_markup_in_a_title_so_the_payload_cannot_break_out() -> None:
    """A ``</script>`` in user data must not terminate the inlined payload block."""
    spec = _lorenz_line3d_spec(32)
    spec.title = "</script><b>pwned</b>"
    page = _viewer_page(spec, poster=False)
    assert "<b>pwned</b>" not in page
    assert "\\u003c/script>" in page or "\\u003cscript" in page


def test_link_assets_mode_imports_the_shared_loader() -> None:
    """``assets='link'`` swaps the inlined loader for one shared import."""
    page = _viewer_page(_lorenz_line3d_spec(64), poster=False, assets="link")
    assert 'await import("./tsdyn-threejs-loader.js")' in page
    assert "function buildLineComet(" not in page  # not inlined


def test_every_page_import_is_dynamic_so_the_poster_can_take_over() -> None:
    """The page must not use a STATIC import anywhere. This is the poster's life.

    A static ``import`` that fails — an unreachable three.js CDN, a missing loader
    — aborts the whole module *before any statement runs*, so the ``try``/``catch``
    that reveals the poster never executes and the reader gets a blank page.  That
    is precisely what the first cut of this page did; it was caught by booting it
    in a real browser with the CDN host broken, not by any dict assertion.
    """
    import re

    for assets in ("inline", "link"):
        page = _viewer_page(_lorenz_line3d_spec(64), poster=False, assets=assets)
        boot = page.split('<script type="module">')[1]
        assert re.search(r"^\s*import\s", boot, flags=re.MULTILINE) is None
        assert 'await import("three")' in boot
        assert "degrade(err)" in boot


def test_inlined_loader_rides_in_an_inert_block_not_a_module() -> None:
    """The inlined loader is data the boot code imports, not an executed script."""
    page = _viewer_page(_lorenz_line3d_spec(64), poster=False)
    assert '<script type="text/plain" id="tsdyn-loader">' in page
    assert "createObjectURL" in page
    # …and it is inlined verbatim: no `export`-stripping surgery on the source.
    from tsdynamics.viz.render.threejs import loader_source

    assert loader_source() in page


@pytest.mark.slow
def test_a_broken_three_js_cdn_falls_back_to_the_poster(tmp_path) -> None:
    """With the CDN unreachable the reader sees the poster, never a blank page."""
    pytest.importorskip("matplotlib")
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    out = tmp_path / "offline.html"
    page_html = _viewer_page(_lorenz_line3d_spec(512), poster=True)
    out.write_text(page_html.replace("cdn.jsdelivr.net", "invalid.example.test"), encoding="utf-8")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": 640, "height": 480})
            page.goto(out.as_uri(), wait_until="load", timeout=60_000)
            page.wait_for_timeout(2500)
            state = page.evaluate(
                "() => [getComputedStyle(document.getElementById('tsdyn-poster')).display,"
                " !!document.querySelector('#tsdyn-viewer canvas')]"
            )
            browser.close()
    except Exception as exc:  # pragma: no cover - no browser available
        pytest.skip(f"headless browser unavailable: {exc}")
    assert state == ["block", False]


def test_unknown_assets_mode_raises_rather_than_guessing() -> None:
    """A page cannot be *partly* self-contained, so an unknown mode is an error."""
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError, match="assets="):
        _viewer_page(_lorenz_line3d_spec(32), poster=False, assets="cdn")


def test_backend_declares_the_extensions_it_can_write() -> None:
    """The backend advertises what it can genuinely write.

    ``PlotSpec.save`` reads this to refuse an unsupported ``(extension, backend)``
    pair loudly instead of returning a path it never wrote.
    """
    from tsdynamics.viz.render.threejs import SAVE_EXTENSIONS

    assert ".html" in SAVE_EXTENSIONS
    assert ".json" in SAVE_EXTENSIONS
    assert ".png" not in SAVE_EXTENSIONS


@pytest.mark.parametrize("ext", [".png", ".svg", ".pdf", ".txt", ""])
def test_backend_refuses_an_extension_it_cannot_honestly_write(tmp_path, ext) -> None:
    """A declared contract that nothing enforces is decoration.

    ``SAVE_EXTENSIONS`` was advertised but never consulted on the write path, so the
    JSON fallback accepted *any* path: ``spec.save("fig.png", backend="threejs")``
    wrote a JSON document into a ``.png`` and returned the path.  That is the same
    "returns a path it did not honestly write" defect the ``.html`` leg had, moved
    to another extension — and it survives an existence check, because a file really
    is there.  Only the extension is a lie.

    The guard must also leave **nothing behind**: a caller who gets an exception must
    not find a half-written file at the path.
    """
    from tsdynamics.errors import InvalidParameterError

    out = tmp_path / f"fig{ext}"
    spec = _lorenz_line3d_spec(256)
    with pytest.raises(InvalidParameterError, match="cannot write"):
        spec.render("threejs", path=out)
    assert not out.exists(), "a refused write must not leave a file behind"


@pytest.mark.parametrize("ext", [".png", ".json", ".txt", ""])
def test_html_true_cannot_smuggle_a_page_into_another_extension(tmp_path, ext) -> None:
    """``html=True`` must not bypass the extension contract.

    The JSON-branch guard was added without a twin on the *page* branch, and
    ``html=True`` short-circuits to the page writer before any check ran — so
    ``render(path="fig.png", html=True)`` wrote a ``<!doctype html>`` document into a
    ``.png`` and returned the path.  Same defect, one branch over: the file exists,
    the size is plausible, and no image viewer can open it.
    """
    from tsdynamics.errors import InvalidParameterError

    out = tmp_path / f"fig{ext}"
    spec = _lorenz_line3d_spec(256)
    with pytest.raises(InvalidParameterError, match="HTML document"):
        spec.render("threejs", path=out, html=True, poster=False)
    assert not out.exists(), "a refused write must not leave a file behind"


def test_capabilities_declare_the_same_extensions_as_the_module_constant() -> None:
    """``can_save`` is the *published* extension point — it must not lie.

    ``PlotSpec.save`` consults ``RendererCapabilities.can_save(ext)`` before falling
    back to its own table.  Declaring the writable formats only on the module
    constant left ``can_save(".html")`` answering ``False`` for the one backend whose
    entire purpose is writing an embeddable ``.html``, so the hard-coded table was
    silently load-bearing for it.
    """
    from tsdynamics import registry
    from tsdynamics.viz.render import register_builtin_renderers
    from tsdynamics.viz.render.threejs import SAVE_EXTENSIONS

    register_builtin_renderers()
    caps = registry.renderers.get("threejs").capabilities
    assert caps.writes == SAVE_EXTENSIONS
    assert caps.can_save(".html") and caps.can_save(".json")
    assert not caps.can_save(".png")


def test_a_composite_page_mounts_every_panel(tmp_path) -> None:
    """A composite ``.html`` draws its panels instead of coming up blank.

    The lowering puts a composite's geometry under ``payload["panels"][i]``, and the
    reference loader used to read only the top-level ``geometries`` — ``[]`` for a
    composite.  So the page booted WebGL successfully, hid the poster fallback
    (JavaScript had plainly worked), and drew nothing: the file existed, weighed
    megabytes, opened in a browser, and showed a black rectangle.  The page writer
    therefore declined a composite outright.

    The loader now normalises single-panel and composite payloads to the same
    ``{geometries, metadata, offset}`` list and mounts each panel in a group at its
    layout offset, so the page is written and draws.  This pins both halves: the
    payload really carries the per-panel geometry, and the loader really reads
    ``payload.panels``.
    """
    import tsdynamics as ts
    from tsdynamics.viz.render.threejs import loader_source

    composite = ts.viz.plot(_lorenz_line3d_spec(256), _lorenz_line3d_spec(256), layout="stack")
    assert composite.is_composite

    out = tmp_path / "composite.html"
    composite.render("threejs", path=out, poster=False)
    assert out.exists() and out.stat().st_size > 0

    payload = composite.render("threejs", raw=True)
    assert payload["geometries"] == []
    assert len(payload["panels"]) == 2
    assert all(p["geometries"] for p in payload["panels"])

    src = loader_source()
    assert "payload.panels" in src, "the loader must read a composite's panels"
    assert "panel.offset" in src, "each panel must be mounted at its layout offset"


def test_the_cap_warning_names_the_thinning_that_actually_ran() -> None:
    """A point cloud is not "resampled by arc length".

    The consolidated cap warning hard-coded the curve wording, so a capped scatter
    (a map's iterate set) was told its shape had been preserved "by arc length" —
    the one property this module documents as *inapplicable* to a set, which is
    thinned by a seeded uniform draw instead.
    """
    from tsdynamics.viz.render.caps import VisualizationDegraded

    spec = _phase2d_scatter_spec(5000)
    with pytest.warns(VisualizationDegraded, match="uniform draw"):
        spec.render("threejs", raw=True, max_points=100)

    with pytest.warns(VisualizationDegraded, match="arc length"):
        _lorenz_line3d_spec(5000).render("threejs", raw=True, max_points=100)


@pytest.mark.slow
def test_emitted_page_boots_in_a_browser(tmp_path) -> None:
    """End-to-end: the emitted file really renders in a browser.

    The only test here that proves the deliverable *works* rather than that its
    bytes look right.  Skipped wherever playwright / Chrome / the three.js CDN are
    unavailable, so it can never block a build.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    out = tmp_path / "viewer.html"
    spec = _lorenz_line3d_spec(4096)
    spec.animation = Animation()
    spec.render("threejs", path=out, poster=False)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"],
            )
            page = browser.new_page(viewport={"width": 640, "height": 480})
            page.goto(out.as_uri(), wait_until="load", timeout=60_000)
            page.wait_for_timeout(3000)
            info = page.evaluate(
                "() => { const c = document.querySelector('#tsdyn-viewer canvas');"
                " return c ? [c.clientWidth, c.clientHeight] : null; }"
            )
            browser.close()
    except Exception as exc:  # pragma: no cover - no browser / no network
        pytest.skip(f"headless browser unavailable: {exc}")
    assert info == [640, 480]


@pytest.mark.slow
def test_a_container_hidden_at_boot_resizes_when_revealed(tmp_path) -> None:
    """The tab / accordion case, in a real browser — the ResizeObserver's whole point.

    A viewer inside a ``display:none`` panel has zero client size while it boots, so
    it falls back to 640x420.  A ``window.resize`` listener never fires when the panel
    is later revealed (the *window* did not change), which left the canvas stuck at
    the fallback forever.  A ``ResizeObserver`` on the container does fire.

    This is the behavioural half of
    ``test_loader_sizes_itself_with_a_resize_observer_and_idles_offscreen``: that one
    greps the source, which cannot tell a working observer from a disconnected one.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    out = tmp_path / "hidden.html"
    spec = _lorenz_line3d_spec(4096)
    spec.render("threejs", path=out, poster=False)
    # Wrap the viewer in a panel that is hidden at boot, then revealed — exactly the
    # markup a docs tab set or an accordion produces.
    out.write_text(
        out.read_text(encoding="utf-8").replace(
            '<div id="tsdyn-viewer"></div>',
            '<div id="panel" style="display:none;position:absolute;inset:0">'
            '<div id="tsdyn-viewer" style="position:absolute;inset:0"></div></div>',
        ),
        encoding="utf-8",
    )

    probe = (
        "() => { const c = document.querySelector('#tsdyn-viewer canvas');"
        " return c ? [c.clientWidth, c.clientHeight] : null; }"
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"],
            )
            page = browser.new_page(viewport={"width": 900, "height": 620})
            page.goto(out.as_uri(), wait_until="load", timeout=60_000)
            page.wait_for_timeout(2500)
            hidden = page.evaluate(probe)
            page.evaluate("() => { document.getElementById('panel').style.display = 'block'; }")
            page.wait_for_timeout(2500)
            revealed = page.evaluate(probe)
            browser.close()
    except Exception as exc:  # pragma: no cover - no browser / no network
        pytest.skip(f"headless browser unavailable: {exc}")

    # Hidden: no layout box at all, so the canvas cannot have been sized to content.
    assert hidden == [0, 0], f"expected a zero-size canvas while hidden, got {hidden}"
    # Revealed: the observer fired and the canvas took the container's real size.
    assert revealed == [900, 620], f"canvas stayed at the fallback size: {revealed}"


# ---------------------------------------------------------------------------
# Legibility: point-cloud sizing and the axes frame
# ---------------------------------------------------------------------------


def test_a_static_point_cloud_is_sized_in_screen_pixels_not_world_units() -> None:
    """The loader must never size a point cloud in *world* units.

    The static ``points`` branch used to build its material with ``size: 0.6`` and
    ``sizeAttenuation`` left at its ``true`` default, i.e. 0.6 **world units**.  On a
    200 000-iterate Hénon export (2.56 wide) every dot was drawn 23% as wide as the
    entire attractor, and the page rendered as a solid opaque slab with the Cantor
    banding — the whole visual content — buried under it.

    The behavioural proof is
    :func:`test_a_dense_map_cloud_shows_its_banding_in_a_browser`; this is the cheap
    structural guard that survives without a browser.
    """
    from tsdynamics.viz.render.threejs import loader_source

    src = loader_source()
    assert "function pointPixelSize(" in src
    assert "function pointOpacity(" in src
    # Whatever the ladder returns, the static branch must switch attenuation off.
    static_branch = src.split('if (geom.type === "points") {', 1)[1][:1200]
    assert "sizeAttenuation: false" in static_branch, (
        "a static point cloud sized in world units renders as a slab"
    )


def test_the_page_asks_the_viewer_for_axes_and_can_be_told_not_to() -> None:
    """``spec.save(...html)`` opts the viewer into its axes frame; ``axes=False`` opts out.

    A saved page is a *plot*, so it carries a scale by default.  The docs-catalogue
    viewer passes no ``axes`` at all and so keeps the loader's own default (off while
    a reveal comet is playing), which is why this must be an explicit page option
    rather than a change of the loader default.
    """
    from tsdynamics.viz.render.threejs import render_page
    from tsdynamics.viz.render.threejs._lower import lower_spec

    payload = lower_spec(_lorenz_line3d_spec(64))
    assert "axes: true" in render_page(payload, poster=False)
    assert "axes: false" in render_page(payload, poster=False, axes=False)


def test_the_axes_keyword_survives_the_dispatcher(tmp_path) -> None:
    """``save(..., axes=False)`` reaches the backend instead of being refused.

    The dispatcher validates render keywords against a hand-written per-backend
    declaration, so a new backend option is unreachable — it raises
    ``InvalidParameterError`` — until that declaration is updated too. A test on
    ``render_page`` alone would not have noticed: it calls the page writer directly
    and never crosses the dispatcher that owns the check.
    """
    out = tmp_path / "no-axes.html"
    _lorenz_line3d_spec(64).save(out, backend="threejs", poster=False, axes=False)
    assert "axes: false" in out.read_text(encoding="utf-8")


def _browser_screenshot(page_path, tmp_path, *, size=(900, 700), wait=4000):
    """Boot an emitted page in headless Chrome and return the screenshot as RGB.

    Skips (never fails) when playwright, Chrome, or the pinned three.js CDN is
    unavailable — a legibility check must not block a build on a network hiccup.
    """
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright
    Image = pytest.importorskip("PIL.Image")

    shot = tmp_path / "shot.png"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"],
            )
            page = browser.new_page(viewport={"width": size[0], "height": size[1]})
            page.goto(page_path.as_uri(), wait_until="load", timeout=60_000)
            page.wait_for_timeout(wait)
            page.screenshot(path=str(shot))
            browser.close()
    except Exception as exc:  # pragma: no cover - no browser / no network
        pytest.skip(f"headless browser unavailable: {exc}")
    return np.asarray(Image.open(shot).convert("RGB")).astype(float)


def _lit_mask(rgb, background=(11, 15, 20)):
    """Pixels that differ from the stage colour — i.e. anything actually drawn."""
    return np.abs(rgb - np.asarray(background, dtype=float)).sum(axis=-1) > 30


@pytest.mark.slow
def test_a_dense_map_cloud_shows_its_banding_in_a_browser(tmp_path) -> None:
    """A 200k-iterate Hénon export must render as *bands*, not as a filled slab.

    Measured on the real screenshots, the two regimes are not close:

    ==========================  ==============  ===========
    render                      lit fraction    max bands
    ==========================  ==============  ===========
    world-unit dots (the bug)   0.097           2
    screen-pixel dots (now)     0.022           10
    ==========================  ==============  ===========

    "Max bands" counts, over every image column, the runs of lit pixels down that
    column: a slab has one (plus its notch), while the Hénon attractor's Cantor
    transversal structure resolves into many.  That count is the whole point of
    plotting this attractor, so it is what the test asserts — a lit-fraction bound
    alone would pass on a blank canvas.
    """
    import tsdynamics as ts

    henon = ts.systems.Henon()
    traj = henon.run(steps=200_000, ic=[0.1, 0.1])
    out = tmp_path / "henon.html"
    with pytest.warns(VisualizationDegraded):
        traj.to_plot_spec().render("threejs", path=out, poster=False)

    rgb = _browser_screenshot(out, tmp_path)
    lit = _lit_mask(rgb)
    assert lit.mean() > 0.001, "nothing was drawn at all"
    assert lit.mean() < 0.06, f"the cloud is inked like a slab: {lit.mean():.3f} of the canvas"

    bands = max(
        int(np.sum(lit[1:, col] & ~lit[:-1, col])) + int(lit[0, col]) for col in range(lit.shape[1])
    )
    assert bands >= 6, f"the Cantor banding collapsed: at most {bands} bands on any column"


@pytest.mark.slow
def test_the_viewer_draws_a_labelled_scale_frame_in_a_browser(tmp_path) -> None:
    """The emitted page carries axes: a frame, ticks and names, in neutral ink.

    A WebGL scene has no axes of its own, and for a long time this viewer drew none —
    an attractor floating in a void with no way to tell whether it spanned 3 units or
    3000.  The frame and its type are inked neutral grey while the data takes the
    palette colour, so counting greyscale pixels distinguishes "the axes rendered"
    from "the attractor rendered" without reaching into the scene graph.
    """
    import tsdynamics as ts

    traj = ts.systems.Lorenz().run(final_time=40.0, dt=0.005, ic=[1.0, 1.0, 1.0])
    out = tmp_path / "lorenz.html"
    traj.to_plot_spec().render("threejs", path=out, poster=False)

    rgb = _browser_screenshot(out, tmp_path)
    lit = _lit_mask(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    grey = lit & (np.abs(r - g) < 18) & (np.abs(g - b) < 18) & (r > 60)
    coloured = lit & ~grey

    assert coloured.sum() > 1000, "the attractor itself did not render"
    assert grey.sum() > 200, "no axes frame / tick labels were drawn"
