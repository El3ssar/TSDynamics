"""Tests for the PlotSpec IR (stream WS-VIZSPEC).

Covers the three acceptance pillars:

1. ``to_dict`` / ``from_dict`` round-trips arrays ↔ lists.
2. ``relabel`` / ``rescale`` / ``limits`` / ``ticks`` / ``style`` mutate-and-chain.
3. No plot library is imported anywhere — neither by ``tsdynamics.viz.spec`` nor
   by ``import tsdynamics``.
"""

import subprocess
import sys

import numpy as np
import pytest

from tsdynamics.viz.spec import (
    Annotation,
    Axis,
    Layer,
    PlotKind,
    PlotSpec,
    Plottable,
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _sample_spec() -> PlotSpec:
    """A 3D phase-portrait spec exercising every field that round-trips."""
    t = np.linspace(0.0, 1.0, 16)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        aspect="equal",
        title="Lorenz",
        x=Axis(label="x", scale="linear", limits=(-20.0, 20.0), ticks=[-10, 0, 10]),
        y=Axis(label="y", scale="log"),
        z=Axis(label="z", scale="symlog", tickformat="%.1f"),
        layers=[
            Layer(
                PlotKind.LINE3D,
                {"x": np.sin(t), "y": np.cos(t), "z": t, "c": t**2},
                label="orbit",
                style={"color": "indigo", "lw": 1.5},
            ),
            Layer(PlotKind.SCATTER, {"x": t[::4], "y": t[::4]}, label="markers"),
        ],
        annotations=[
            Annotation(kind="vline", text="r1=3", x=3.0, style={"color": "rose"}),
            Annotation(kind="span", span=(1.0, 2.0), axis="y"),
        ],
        meta={"system": "lorenz", "params": {"sigma": 10.0}, "n": np.int64(16)},
    )


# ---------------------------------------------------------------------------
# 1. Round-trip
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serializable():
    import json

    spec = _sample_spec()
    d = spec.to_dict()
    # Must be plain JSON types — json.dumps would raise on a stray ndarray.
    text = json.dumps(d)
    assert isinstance(text, str)
    # Arrays became nested lists.
    assert isinstance(d["layers"][0]["data"]["x"], list)


def test_round_trip_preserves_structure_and_arrays():
    spec = _sample_spec()
    rebuilt = PlotSpec.from_dict(spec.to_dict())

    assert rebuilt.kind is PlotKind.PHASE_PORTRAIT_3D
    assert rebuilt.ndim == 3
    assert rebuilt.aspect == "equal"
    assert rebuilt.title == "Lorenz"

    # Axes survive (including the optional z axis and its attributes).
    assert rebuilt.x.label == "x"
    assert rebuilt.x.limits == (-20.0, 20.0)
    assert list(rebuilt.x.ticks) == [-10, 0, 10]
    assert rebuilt.y.scale == "log"
    assert rebuilt.z is not None
    assert rebuilt.z.scale == "symlog"
    assert rebuilt.z.tickformat == "%.1f"

    # Layers: marks, labels, styles, and arrays survive bit-for-bit.
    assert [lyr.kind for lyr in rebuilt.layers] == [PlotKind.LINE3D, PlotKind.SCATTER]
    assert rebuilt.layers[0].label == "orbit"
    assert rebuilt.layers[0].style == {"color": "indigo", "lw": 1.5}
    for key in ("x", "y", "z", "c"):
        np.testing.assert_array_equal(rebuilt.layers[0].data[key], spec.layers[0].data[key])
        assert isinstance(rebuilt.layers[0].data[key], np.ndarray)

    # Annotations survive.
    assert rebuilt.annotations[0].kind == "vline"
    assert rebuilt.annotations[0].x == 3.0
    assert rebuilt.annotations[1].span == (1.0, 2.0)
    assert rebuilt.annotations[1].axis == "y"

    # meta numpy scalar was JSONified to a python int.
    assert rebuilt.meta["system"] == "lorenz"
    assert rebuilt.meta["n"] == 16
    assert isinstance(rebuilt.meta["n"], int)


def test_round_trip_with_no_z_axis():
    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        ndim=1,
        x=Axis("t"),
        y=Axis("x(t)"),
        layers=[Layer(PlotKind.LINE, {"x": np.arange(5.0), "y": np.arange(5.0) ** 2})],
    )
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.z is None
    assert rebuilt.x.limits is None
    assert rebuilt.x.ticks is None
    np.testing.assert_array_equal(rebuilt.layers[0].data["y"], np.arange(5.0) ** 2)


def test_layer_coerces_lists_to_arrays():
    lyr = Layer(PlotKind.LINE, {"x": [1, 2, 3], "y": [4, 5, 6]})
    assert isinstance(lyr.data["x"], np.ndarray)
    np.testing.assert_array_equal(lyr.data["y"], [4, 5, 6])


def test_string_kind_is_normalized_to_enum():
    spec = PlotSpec(kind="time_series", layers=[Layer("line", {"x": [0.0]})])
    assert spec.kind is PlotKind.TIME_SERIES
    assert spec.layers[0].kind is PlotKind.LINE


def test_plotkind_is_str_enum():
    # str-enum: members compare equal to their values and serialize as strings.
    assert PlotKind.LINE == "line"
    assert PlotKind.BIFURCATION.value == "bifurcation"


# ---------------------------------------------------------------------------
# 2. Mutate-and-chain tweaks
# ---------------------------------------------------------------------------


def test_tweaks_mutate_and_chain():
    spec = _sample_spec()
    returned = (
        spec.relabel(x="X", y="Y", z="Z", title="Tweaked")
        .rescale(x="log", y="linear", z="symlog")
        .limits(x=(0.0, 1.0), y=(2.0, 3.0))
        .ticks(x=[0.0, 0.5, 1.0])
        .style(color="teal")
    )
    # Chaining returns the SAME object (mutation, not a copy).
    assert returned is spec

    assert spec.x.label == "X"
    assert spec.y.label == "Y"
    assert spec.z.label == "Z"
    assert spec.title == "Tweaked"
    assert spec.x.scale == "log"
    assert spec.x.limits == (0.0, 1.0)
    assert list(spec.x.ticks) == [0.0, 0.5, 1.0]
    # style with no layer index hits every layer.
    assert all(lyr.style.get("color") == "teal" for lyr in spec.layers)


def test_tweaks_only_touch_passed_axes():
    spec = _sample_spec()
    original_y_label = spec.y.label
    spec.relabel(x="new-x")
    assert spec.x.label == "new-x"
    assert spec.y.label == original_y_label  # untouched


def test_z_tweaks_are_ignored_without_z_axis():
    spec = PlotSpec(kind=PlotKind.TIME_SERIES, z=None)
    # Should not raise even though there is no z axis.
    spec.relabel(z="Z").rescale(z="log").limits(z=(0.0, 1.0)).ticks(z=[0.0])
    assert spec.z is None


def test_style_targets_a_single_layer_by_index():
    spec = _sample_spec()
    spec.style(layer=1, marker="o")
    assert "marker" not in spec.layers[0].style
    assert spec.layers[1].style["marker"] == "o"


# ---------------------------------------------------------------------------
# 3. Rendering seam (no backend registered → raises cleanly)
# ---------------------------------------------------------------------------


def test_render_raises_without_a_backend():
    spec = _sample_spec()
    with pytest.raises(ImportError):  # VisualizationNotInstalled subclasses ImportError
        spec.render()


def test_plottable_mixin_plot_raises_without_backend():
    class _Thing(Plottable):
        def to_plot_spec(self):
            return PlotSpec(kind=PlotKind.TIME_SERIES)

    with pytest.raises(ImportError):
        _Thing().plot(xscale="log")


def test_plottable_base_to_plot_spec_raises():
    with pytest.raises(NotImplementedError):
        Plottable().to_plot_spec()


def test_plottable_mimebundle_is_noop_without_backend():
    class _Thing(Plottable):
        def to_plot_spec(self):
            return PlotSpec(kind=PlotKind.TIME_SERIES)

    assert _Thing()._repr_mimebundle_() is None


def test_inline_tweaks_forward_unknown_kwargs_to_backend():
    # No backend, so we can only assert the recognized tweaks are consumed and
    # the spec is mutated before render is attempted.
    from tsdynamics.viz.spec import _apply_inline_tweaks

    spec = PlotSpec(kind=PlotKind.TIME_SERIES)
    leftover = _apply_inline_tweaks(
        spec, {"xscale": "log", "title": "T", "ylim": (0.0, 1.0), "ax": "passthrough"}
    )
    assert spec.x.scale == "log"
    assert spec.title == "T"
    assert spec.y.limits == (0.0, 1.0)
    assert leftover == {"ax": "passthrough"}


# ---------------------------------------------------------------------------
# 4. The no-plot-import invariant
# ---------------------------------------------------------------------------


def test_spec_module_imports_no_plot_library():
    # The spec module itself must not have pulled matplotlib/plotly into sys.modules.
    code = (
        "import sys; import tsdynamics.viz.spec; "
        "bad = [m for m in sys.modules "
        "if m == 'matplotlib' or m.startswith('matplotlib.') "
        "or m == 'plotly' or m.startswith('plotly.')]; "
        "assert not bad, bad; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_import_tsdynamics_pulls_no_plot_library():
    code = (
        "import sys; import tsdynamics; "
        "bad = [m for m in sys.modules "
        "if m == 'matplotlib' or m.startswith('matplotlib.') "
        "or m == 'plotly' or m.startswith('plotly.')]; "
        "assert not bad, bad; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
