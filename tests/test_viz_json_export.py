"""Tests for the versioned JSON export + the ``json`` data-export backend.

Stream VIZ-JSON-EXPORT.  Covers :mod:`tsdynamics.viz.export`
(:func:`~tsdynamics.viz.export.to_json` / :func:`~tsdynamics.viz.export.from_json`
round-trips, the ``schema_version`` stamp, and back-compat with an unversioned
bare-dict payload) and :mod:`tsdynamics.viz.render.json` (the ``json`` backend's
``spec.render("json")`` payload + writing to a file).  Engine-free — no
``tsdynamics._rust`` import.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tsdynamics.viz.export import (
    SCHEMA_VERSION,
    from_dict_envelope,
    from_json,
    to_dict_envelope,
    to_json,
)
from tsdynamics.viz.render.caps import RenderResult
from tsdynamics.viz.spec import (
    Annotation,
    Axis,
    Colorbar,
    Layer,
    Legend,
    PlotKind,
    PlotSpec,
)


def _time_series_spec() -> PlotSpec:
    """A 2-D time-series spec with a labelled line layer and an annotation."""
    t = np.linspace(0.0, 1.0, 16)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)}, label="x(t)")],
        x=Axis(label="t", scale="linear"),
        y=Axis(label="x", limits=(-1.0, 1.0)),
        legend=Legend(show=True, title="state"),
        title="series",
        annotations=[Annotation(kind="vline", x=0.5, text="onset")],
        meta={"system": "Demo", "dt": 0.01},
    )


def _phase3d_spec() -> PlotSpec:
    """A 3-D phase-portrait spec (exercises the ``z`` axis + LINE3D mark)."""
    s = np.linspace(0.0, 2.0, 12)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[Layer(kind=PlotKind.LINE3D, data={"x": s, "y": s**2, "z": np.cos(s)})],
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z", scale="log"),
        ndim=3,
        aspect="equal",
    )


def _image_spec() -> PlotSpec:
    """An image spec with a color channel + colorbar + clim."""
    img = np.arange(9.0).reshape(3, 3)
    return PlotSpec(
        kind=PlotKind.IMAGE,
        layers=[Layer(kind=PlotKind.IMAGE, data={"z": img})],
        clim=(0.0, 8.0),
        colorbar=Colorbar(label="value", cmap="viridis", discrete=False),
    )


def _assert_specs_equal(a: PlotSpec, b: PlotSpec) -> None:
    """Assert two specs are structurally equal (arrays compared element-wise)."""
    assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [_time_series_spec, _phase3d_spec, _image_spec],
    ids=["time_series", "phase3d", "image"],
)
def test_to_from_json_round_trips(builder) -> None:
    """``from_json(to_json(spec))`` reproduces the spec losslessly."""
    spec = builder()
    restored = from_json(to_json(spec))
    _assert_specs_equal(spec, restored)


def test_round_trip_preserves_array_data() -> None:
    """Array channel data survives the JSON round-trip as a NumPy array."""
    spec = _time_series_spec()
    restored = from_json(to_json(spec))
    layer = restored.layers[0]
    assert isinstance(layer.data["y"], np.ndarray)
    np.testing.assert_allclose(layer.data["y"], spec.layers[0].data["y"])


def test_every_kind_smoke_round_trips() -> None:
    """A minimal spec of *every* PlotKind round-trips through JSON."""
    for kind in PlotKind:
        spec = PlotSpec(kind=kind, layers=[Layer(kind=PlotKind.LINE, data={"x": [0.0, 1.0]})])
        restored = from_json(to_json(spec))
        assert restored.kind is kind
        _assert_specs_equal(spec, restored)


# ---------------------------------------------------------------------------
# schema versioning / back-compat
# ---------------------------------------------------------------------------


def test_schema_version_stamp_present() -> None:
    """The serialized envelope carries the schema_version stamp + spec payload."""
    payload = json.loads(to_json(_time_series_spec()))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["spec"]["kind"] == PlotKind.TIME_SERIES.value


def test_envelope_dict_round_trips() -> None:
    """The dict-envelope helpers round-trip without going through text."""
    spec = _image_spec()
    envelope = to_dict_envelope(spec)
    assert envelope["schema_version"] == SCHEMA_VERSION
    _assert_specs_equal(spec, from_dict_envelope(envelope))


def test_old_dict_without_version_loads() -> None:
    """A legacy bare ``PlotSpec.to_dict()`` document (no envelope) still loads."""
    spec = _time_series_spec()
    legacy_text = json.dumps(spec.to_dict())  # no schema_version, no "spec" wrapper
    restored = from_json(legacy_text)
    _assert_specs_equal(spec, restored)


def test_unrecognized_payload_raises() -> None:
    """A JSON object that is neither an envelope nor a bare spec dict raises."""
    with pytest.raises(ValueError):
        from_json(json.dumps({"not_a_spec": True}))


def test_non_object_json_raises() -> None:
    """A JSON document that is not an object raises a clear error."""
    with pytest.raises(ValueError):
        from_json("[1, 2, 3]")


# ---------------------------------------------------------------------------
# the json renderer backend
# ---------------------------------------------------------------------------


def test_render_json_returns_payload() -> None:
    """``spec.render("json")`` returns a RenderResult carrying the JSON payload."""
    spec = _time_series_spec()
    result = spec.render("json")
    assert isinstance(result, RenderResult)
    assert result.backend == "json"
    assert result.mimetype == "application/json"
    assert result.kind is PlotKind.TIME_SERIES
    # The payload is the to_json text — it loads back into an equal spec.
    _assert_specs_equal(spec, from_json(result.payload))


def test_render_json_raw_returns_str() -> None:
    """``raw=True`` returns the bare JSON string instead of a RenderResult."""
    spec = _time_series_spec()
    text = spec.render("json", raw=True)
    assert isinstance(text, str)
    _assert_specs_equal(spec, from_json(text))


def test_render_json_writes_file(tmp_path) -> None:
    """Passing a ``path`` writes the JSON to disk and returns the path."""
    spec = _phase3d_spec()
    out = tmp_path / "spec.json"
    returned = spec.render("json", path=out)
    assert returned == out
    assert out.exists()
    _assert_specs_equal(spec, from_json(out.read_text(encoding="utf-8")))


def test_render_json_indent_pretty_prints() -> None:
    """An ``indent`` keyword pretty-prints (multi-line) but still round-trips."""
    spec = _image_spec()
    text = spec.render("json", raw=True, indent=2)
    assert "\n" in text
    _assert_specs_equal(spec, from_json(text))


def test_json_backend_registers_unconditionally() -> None:
    """The json backend self-wires via register_builtin_renderers (no optional dep)."""
    from tsdynamics import registry
    from tsdynamics.viz.render import register_builtin_renderers

    register_builtin_renderers()
    assert "json" in registry.renderers
    renderer = registry.renderers.get("json")
    caps = renderer.capabilities
    assert caps.data_export is True
    assert caps.kinds is None  # accepts every kind


def test_json_never_shadows_a_drawing_backend_in_default_selection():
    """Default ``spec.render()`` prefers a drawing backend over the json exporter.

    Regression: the ``json`` exporter declares ``data_export=True`` / ``kinds=None``
    (it serialises any spec), so a naive "first capable" default selection could
    pick it over matplotlib — and then ``result.plot()`` would return a JSON
    payload instead of a figure.  The dispatch skips data-export backends in
    default selection, so *every* default render returns a figure, even when json
    is registered first; ``render("json")`` by name still reaches the exporter.
    """
    matplotlib = pytest.importorskip("matplotlib")
    from tsdynamics import registry
    from tsdynamics.viz.render import register_builtin_renderers

    saved = list(registry.renderers.all())
    registry.renderers.clear()
    try:
        register_builtin_renderers()  # registers json (first) + matplotlib
        if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present
            pytest.skip("matplotlib backend did not register")
        spec = _time_series_spec()
        # Repeated default renders must all be figures (not RenderResult payloads).
        for _ in range(3):
            assert isinstance(spec.render(), matplotlib.figure.Figure)
        # …while the exporter is still reachable by name.
        assert isinstance(spec.render("json"), RenderResult)
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj)
