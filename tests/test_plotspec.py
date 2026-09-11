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
    # normalize_style canonicalizes the matplotlib marker alias "o" -> "circle".
    assert spec.layers[1].style["marker"] == "circle"


# ---------------------------------------------------------------------------
# 3. Rendering seam (no backend registered → raises cleanly)
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_backend(monkeypatch):
    """Force a genuinely empty renderers registry (stub builtin registration).

    Since stream VIZ-MPL-CORE the matplotlib backend auto-registers on first
    render, so the "no backend installed" behaviour is exercised by clearing the
    registry and stubbing :func:`register_builtin_renderers` to a no-op for the
    test, then restoring it.
    """
    from tsdynamics import registry
    from tsdynamics.viz import render as render_mod

    saved = registry.renderers.all()
    registry.renderers.clear()
    monkeypatch.setattr(render_mod, "register_builtin_renderers", lambda *a, **k: [])
    try:
        yield
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj, replace=True)


def test_render_raises_without_a_backend(_no_backend):
    spec = _sample_spec()
    with pytest.raises(ImportError):  # VisualizationNotInstalled subclasses ImportError
        spec.render()


def test_plottable_mixin_plot_raises_without_backend(_no_backend):
    class _Thing(Plottable):
        def to_plot_spec(self):
            return PlotSpec(kind=PlotKind.TIME_SERIES)

    with pytest.raises(ImportError):
        _Thing().plot(xscale="log").render()


def test_plottable_base_to_plot_spec_raises():
    with pytest.raises(NotImplementedError):
        Plottable().to_plot_spec()


def test_plottable_mimebundle_is_noop_without_backend(_no_backend):
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


# ---------------------------------------------------------------------------
# The versioned JSON envelope is a public, reachable round trip
# ---------------------------------------------------------------------------
#
# ``viz/export.py`` writes the envelope that ``spec.save("fig.json")`` and
# ``spec.render("json")`` produce, but its *load* half had no non-test caller and
# ``ts.viz.from_json`` did not resolve — so the round trip was one-way in
# practice.  Reading a saved spec back is half the "ship a plot to the web"
# story, so the pair is promoted to the public ``tsdynamics.viz`` API rather than
# deleted.


def test_export_round_trip_is_reachable_from_the_public_viz_namespace():
    """``ts.viz.to_json`` / ``from_json`` resolve and round-trip a spec losslessly."""
    import tsdynamics as ts

    spec = _sample_spec()
    restored = ts.viz.from_json(ts.viz.to_json(spec))
    assert restored.to_dict() == spec.to_dict()


def test_export_names_are_in_the_curated_viz_all():
    """The loader half is *listed*, not merely importable (it shows in ``dir()``)."""
    import tsdynamics as ts

    for name in ("to_json", "from_json", "to_dict_envelope", "from_dict_envelope"):
        assert name in ts.viz.__all__, name
        assert name in dir(ts.viz), name
    assert ts.viz.SCHEMA_VERSION >= 3


def test_a_saved_json_file_loads_back_through_the_public_reader(tmp_path):
    """The file ``spec.save(...json)`` writes is exactly what ``from_json`` reads."""
    import tsdynamics as ts

    spec = _sample_spec()
    path = spec.save(tmp_path / "fig.json", backend="json")
    restored = ts.viz.from_json(path.read_text())
    assert restored.kind is spec.kind
    assert restored.to_dict() == spec.to_dict()


# ---------------------------------------------------------------------------
# 5. Frame — what the axes *mean* (P0 composability)
# ---------------------------------------------------------------------------


def test_frame_space_vocabulary_is_frozen():
    """``FrameSpace`` is a closed, reviewed contract — like ``PlotKind``.

    Overlay legality is decided by the frame's ``space``, so adding one is a
    semantic change to what may share an axes, not an implementation detail.
    Changing this set is a deliberate edit of this gate.
    """
    from tsdynamics.viz._frames import FRAME_SPACES, FrameSpace

    assert {s.value for s in FrameSpace} == {
        "time",
        "state2",
        "state3",
        "param1",
        "param2",
        "index",
        "grid2",
        "complex",
        "scaling",
        "category",
    }
    assert frozenset(FrameSpace) == FRAME_SPACES
    assert FrameSpace.STATE2 == "state2"  # StrEnum: a member is its value


def test_frame_requires_its_axes_at_construction():
    """A frame that names no axes would silently overlay onto anything."""
    from tsdynamics.errors import InvalidParameterError
    from tsdynamics.viz.spec import Frame, FrameSpace

    with pytest.raises(InvalidParameterError, match="axis name"):
        Frame(FrameSpace.STATE2, 2, ("x",))
    with pytest.raises(InvalidParameterError, match="axis name"):
        Frame(FrameSpace.STATE2, 2, ())
    assert Frame("state2", 2, ("x", "v")).describe() == "state2(x, v)"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("x", "x"),
        ("$x$", "x"),  # the LaTeX spelling and the bare one are one coordinate
        ("$x_{0}$", "#0"),  # an indexed fallback keeps its index
        ("x_1", "#1"),
        ("y0", ""),  # a bare positional placeholder claims nothing
        ("x2", ""),
        ("", ""),
        (None, ""),
        ("$\\theta$", "\\theta"),
    ],
)
def test_axis_name_normalization(label, expected):
    """Three producers spell one coordinate three ways; the frame compares one."""
    from tsdynamics.viz._frames import axis_name

    assert axis_name(label) == expected


def test_frame_compatibility_is_not_equality():
    """An *unnamed* axis is compatible with a named one; two names must match."""
    from tsdynamics.viz.spec import Frame

    named = Frame("state2", 2, ("x", "v"))
    other = Frame("state2", 2, ("x", "z"))
    unnamed = Frame("state2", 2, ("", ""))
    ordinals = Frame("state2", 2, ("#0", "#1"))
    shifted = Frame("state2", 2, ("#0", "#2"))
    three_d = Frame("state3", 3, ("x", "y", "z"))

    assert named.compatible_with(named)
    assert not named.compatible_with(other)
    assert named.compatible_with(unnamed) and unnamed.compatible_with(named)
    assert not named.compatible_with(three_d)
    # a NAME and an ORDINAL do not contradict each other (nothing says which
    # ordinal "v" is), but two ordinals must agree
    assert named.compatible_with(ordinals)
    assert not ordinals.compatible_with(shifted)
    # merging keeps the most informative axis: name > ordinal > nothing
    assert named.merge(unnamed).axes == ("x", "v")
    assert unnamed.merge(named).axes == ("x", "v")
    assert ordinals.merge(named).axes == ("x", "v")
    assert ordinals.merge(unnamed).axes == ("#0", "#1")


def test_resolved_frame_is_derived_from_kind_and_labels():
    """Every spec has a frame, including one that declares none."""
    from tsdynamics.viz.spec import Frame

    portrait = PlotSpec(kind=PlotKind.PHASE_PORTRAIT_2D, x=Axis(label="x"), y=Axis(label="v"))
    assert portrait.frame is None
    assert portrait.resolved_frame == Frame("state2", 2, ("x", "v"))

    # a time series is a ONE-coordinate frame: the y axis is free, so x(t) and
    # y(t) legitimately overlay
    series = PlotSpec(kind=PlotKind.TIME_SERIES, x=Axis(label="t"), y=Axis(label="x"))
    assert series.resolved_frame == Frame("time", 1, ("t",))

    # a declared frame is taken at its word
    declared = PlotSpec(kind=PlotKind.TIME_SERIES, frame=Frame("index", 2, ("i", "j")))
    assert declared.resolved_frame == Frame("index", 2, ("i", "j"))


def test_a_composite_has_no_frame():
    """A multi-panel figure owns no single set of axes."""
    from tsdynamics.errors import InvalidParameterError

    panel = PlotSpec(kind=PlotKind.TIME_SERIES)
    composite = PlotSpec(kind=PlotKind.COMPOSITE, panels=[panel])
    with pytest.raises(InvalidParameterError, match="no frame"):
        _ = composite.resolved_frame


# ---------------------------------------------------------------------------
# 6. The two additive fields stay additive
# ---------------------------------------------------------------------------


def test_layer_transform_defaults_to_none_and_round_trips():
    layer = Layer(PlotKind.LINE, {"x": [0.0, 1.0], "y": [0.0, 1.0]})
    assert layer.transform is None
    tagged = Layer(PlotKind.LINE, {"x": [0.0]}, transform="basins")
    assert Layer.from_dict(tagged.to_dict()).transform == "basins"


def test_from_dict_accepts_a_payload_written_before_frame_and_transform():
    """Additivity, proven the only way that counts: load an *old* payload.

    Both new fields are read with a default, so a spec serialized by a release
    that did not have them still loads — and a payload carrying an unknown key
    (a *newer* writer) loads too.
    """
    old = _sample_spec().to_dict()
    del old["frame"]
    for layer in old["layers"]:
        del layer["transform"]
    old["some_future_key"] = 42

    rebuilt = PlotSpec.from_dict(old)
    assert rebuilt.frame is None
    assert all(lyr.transform is None for lyr in rebuilt.layers)
    assert rebuilt.kind == PlotKind.PHASE_PORTRAIT_3D
    assert len(rebuilt.layers) == len(_sample_spec().layers)


def test_frame_round_trips_through_to_dict():
    from tsdynamics.viz.spec import Frame

    spec = PlotSpec(kind=PlotKind.PHASE_PORTRAIT_2D, frame=Frame("state2", 2, ("x", "v")))
    assert spec.to_dict()["frame"] == {"space": "state2", "ndim": 2, "axes": ["x", "v"]}
    assert PlotSpec.from_dict(spec.to_dict()).frame == spec.frame


# ---------------------------------------------------------------------------
# The notebook display path (adversarial follow-up to "plot() returns a spec")
#
# ``.plot()`` used to hand back a matplotlib Figure, and a Jupyter cell drew it
# because IPython has a display hook registered for Figure.  Once ``.plot()``
# returned a PlotSpec instead, the display hook still handed a *Figure* back out
# of ``_repr_mimebundle_`` — which is not a mime bundle, so IPython warned
# (``FormatterWarning: ... returned invalid type``) and fell back to ``repr``:
# a multi-thousand-line dump of the spec's data arrays, in place of the picture.
# These pin the contract that broke.
# ---------------------------------------------------------------------------


class _FakeFormatter:
    """Stand-in for ``shell.display_formatter``: reports what it was asked to format."""

    def __init__(self) -> None:
        self.seen: list[object] = []

    def format(self, obj, include=None, exclude=None):  # noqa: D102 - test double
        self.seen.append(obj)
        return {"image/png": b"png-bytes", "text/plain": repr(obj)}, {"image/png": {}}


class _FakeShell:
    def __init__(self) -> None:
        self.display_formatter = _FakeFormatter()


@pytest.fixture()
def fake_ipython(monkeypatch):
    """Install a fake ``IPython.core.getipython`` so the hook believes it is in a notebook."""
    import types

    shell = _FakeShell()
    module = types.ModuleType("IPython.core.getipython")
    module.get_ipython = lambda: shell  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython.core.getipython", module)
    return shell


def test_mimebundle_returns_a_mapping_not_a_figure(fake_ipython):
    """The hook must return ``(data, metadata)``; a figure makes IPython warn and give up."""
    from tsdynamics.viz.spec import _notebook_mimebundle

    drawn = object()
    bundle = _notebook_mimebundle(lambda: drawn, None, None)

    assert isinstance(bundle, tuple)
    data, metadata = bundle
    assert isinstance(data, dict) and isinstance(metadata, dict)
    assert "image/png" in data
    # The figure is handed to IPython's own formatters (so the bundle is whatever
    # the *backend* registered), never returned raw.
    assert fake_ipython.display_formatter.seen == [drawn]


def test_mimebundle_does_not_wait_for_a_previous_render(fake_ipython, monkeypatch):
    """The first notebook cell must draw.

    The hook used to bail out when ``tsdynamics.registry.renderers`` was empty —
    but the in-tree backends only register on the *first* render, so in a fresh
    session the very first ``traj.plot()`` cell no-op'd and printed the repr.
    """
    from tsdynamics.viz import spec as spec_mod

    monkeypatch.setattr(spec_mod, "_resolve_renderers", lambda: None)
    assert spec_mod._notebook_mimebundle(lambda: object(), None, None) is not None


def test_mimebundle_is_a_noop_outside_a_notebook(monkeypatch):
    """A plain console/script must fall back to ``repr`` — and never draw."""
    import types

    from tsdynamics.viz.spec import _notebook_mimebundle

    module = types.ModuleType("IPython.core.getipython")
    module.get_ipython = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython.core.getipython", module)

    def _must_not_draw():
        raise AssertionError("render() must not run outside a notebook")

    assert _notebook_mimebundle(_must_not_draw, None, None) is None


def test_mimebundle_survives_a_render_error(fake_ipython):
    """A backend blow-up must degrade to ``repr``, never propagate out of a repr."""
    from tsdynamics.viz.spec import _notebook_mimebundle

    def _boom():
        raise RuntimeError("no backend")

    assert _notebook_mimebundle(_boom, None, None) is None


# ---------------------------------------------------------------------------
# ts.plot(...) must not swallow a keyword no named transform can use
# ---------------------------------------------------------------------------


def test_front_door_rejects_a_keyword_no_transform_accepts():
    """``ts.plot(traj, "time_series", colour="red")`` drew the wrong picture, silently.

    The per-transform keyword filter (which stops ``grid=`` reaching
    ``trajectory`` in a multi-transform overlay) also swallowed a keyword that
    reached *nothing*.  The sibling door ``traj.plot(colour="red")`` already
    refused, so the two front doors disagreed about the same typo.
    """
    import tsdynamics as ts
    from tsdynamics.errors import InvalidParameterError

    traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    with pytest.raises(InvalidParameterError) as excinfo:
        ts.plot(traj, "time_series", colour="red")
    message = str(excinfo.value)
    assert "colour" in message
    assert "color=" in message  # names the spelling that works
    assert "components" in message  # ... and what this transform does accept


def test_front_door_still_routes_a_keyword_only_one_transform_accepts():
    """The rejection must not break the routing it guards: a used keyword is fine."""
    import tsdynamics as ts

    traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    spec = ts.plot(traj, "time_series", "phase_portrait", components=[0, 1], layout="row")
    assert spec.is_composite


# ---------------------------------------------------------------------------
# The two `plot` doors must agree about a positional string
# ---------------------------------------------------------------------------


def test_a_positional_transform_name_on_the_method_names_the_front_door():
    """``ts.plot(traj, "delay_embedding")`` reads a positional string as a transform.

    A reader who has seen that call tries the same on the method and used to get
    ``TypeError: plot() takes 1 positional argument but 2 were given`` — which
    names neither the concept nor a spelling that works.
    """
    import tsdynamics as ts
    from tsdynamics.errors import InvalidParameterError

    traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    for subject, call in (
        ("traj", lambda: traj.plot("delay_embedding", delay=7)),
        ("system", lambda: ts.systems.Lorenz().plot("phase_portrait")),
    ):
        with pytest.raises(InvalidParameterError) as excinfo:
            call()
        message = str(excinfo.value)
        assert f"ts.plot({subject}, " in message  # the front-door spelling
        assert f"{subject}.plot(kind=" in message  # ... and the kind spelling
