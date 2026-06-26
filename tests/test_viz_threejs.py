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
from tsdynamics.viz.render.caps import RenderResult
from tsdynamics.viz.spec import Axis, Layer, Layout, PlotKind, PlotSpec


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
    """A LINE3D spec yields one 'line' geometry: positions 3n, indices 2(n-1)."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 64
    payload = lower_spec(_lorenz_line3d_spec(n))
    assert len(payload["geometries"]) == 1
    geom = payload["geometries"][0]
    assert geom["type"] == "line"
    assert geom["label"] == "orbit"
    assert len(geom["positions"]) == 3 * n
    assert len(geom["indices"]) == 2 * (n - 1)
    # Consecutive segment endpoints: 0,1,1,2,2,3,...
    assert geom["indices"][:6] == [0, 1, 1, 2, 2, 3]


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


def test_scalar_c_channel_maps_to_colors() -> None:
    """A 'c' channel produces a flat per-vertex RGB colors list (3n floats)."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    n = 64
    geom = lower_spec(_lorenz_line3d_spec(n))["geometries"][0]
    assert "colors" in geom
    assert len(geom["colors"]) == 3 * n
    assert all(0.0 <= v <= 1.0 for v in geom["colors"])


def test_uncolored_layer_omits_colors() -> None:
    """A layer with no 'c' channel and no rgb style carries no colors key."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    geom = lower_spec(_phase2d_scatter_spec())["geometries"][0]
    assert "colors" not in geom


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


def test_labels_and_units_surface_from_axes() -> None:
    """Axis labels and tickformat-as-unit are surfaced into metadata."""
    from tsdynamics.viz.render.threejs._lower import lower_spec

    meta = lower_spec(_lorenz_line3d_spec())["metadata"]
    assert meta["labels"] == {"x": "x", "y": "y", "z": "z"}
    assert meta["units"]["x"] == "m"  # from Axis(tickformat="m")


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
