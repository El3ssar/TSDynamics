"""The plot-transform substrate: geometry, registration, front doors, scope (stream P1).

``tests/test_viz_compatibility.py`` renders the *matrix*; this module pins the
*mechanism* underneath it — what a :class:`Geometry` is, what registration
refuses, how ``primitive=`` resolves, and where the scope boundary is drawn.

The single most important claim here is the one in
``test_registering_a_transform_touches_only_its_own_module``: adding a plot must
be **one registration and nothing else**.  If that ever stops being true, every
future contributor pays for it, so it is asserted rather than described.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.errors import InvalidInputError, InvalidParameterError
from tsdynamics.viz._frames import Frame, FrameSpace, OverlayRole
from tsdynamics.viz.spec import Layer, PlotKind, PlotSpec
from tsdynamics.viz.transforms import (
    ADMITTED_SERIES_DIAGNOSTICS,
    EXCLUDED_SERIES_TOOLBOX,
    PRIMITIVES,
    RESERVED_PRIMITIVES,
    Channel,
    ChannelType,
    Geometry,
    Part,
    T,
    build_spec,
    compatibility,
    draw,
    geometry,
    get,
    lower,
    make_frame,
    names,
    plot,
    plot_transform,
    transforms,
)


class _DemoSystem:
    def __init__(self, discrete: bool = False, variables: tuple[str, ...] | None = None) -> None:
        self.is_discrete = discrete
        self.variables = variables


def _traj(dim: int = 3, n: int = 120):
    from tsdynamics.data import Trajectory

    t = np.linspace(0.0, 10.0, n)
    y = np.column_stack([np.sin(t + k) * (k + 1) for k in range(dim)])
    names_ = ("x", "y", "z", "w")[:dim]
    return Trajectory(t, y, _DemoSystem(False, names_), {"system": "demo", "dt": float(t[1])})


def _orbit(n: int = 60):
    from tsdynamics.data import Trajectory

    t = np.arange(n, dtype=float)
    y = np.column_stack([np.cos(0.3 * t), np.sin(0.4 * t)])
    return Trajectory(t, y, _DemoSystem(True, ("a", "b")), {"system": "demo map"})


@pytest.fixture
def scratch_registry():
    """Register throwaway transforms and remove them again."""
    from tsdynamics import registry

    made: list[str] = []

    def register(**kwargs):
        made.append(kwargs["name"])
        return plot_transform(**kwargs)

    yield register
    for name in made:
        if name in registry.plot_transforms:
            registry.plot_transforms.unregister(name)


# ---------------------------------------------------------------------------
# Geometry — the record between "what to plot" and "how to draw it"
# ---------------------------------------------------------------------------


def test_geometry_is_not_an_ir():
    """``Geometry`` deliberately has no serialization: the IR is ``PlotSpec``.

    A second serializable layer above the IR would need its own schema version,
    its own round-trip tests and its own composition rules.  ``Geometry`` exists
    only so the primitive is a separate function reading the same channels.
    """
    g = geometry(_traj(), "phase_portrait", components=[0, 1])
    assert not hasattr(g, "to_dict")
    assert not hasattr(g, "from_dict")
    with pytest.raises(AttributeError):
        g.transform = "something else"  # frozen


def test_geometry_channels_shortcut_refuses_a_multi_part_geometry():
    """A multi-part geometry has no single channel set, and says so.

    Returning the first part's channels would silently hide the rest — exactly
    the quiet wrongness this layer exists to remove.
    """
    single = geometry(_traj(), "phase_portrait", components=[0, 1])
    assert sorted(single.channels) == ["x", "y"]

    multi = geometry(_traj(), "time_series")
    assert len(multi) == 3
    with pytest.raises(InvalidParameterError, match="3 parts"):
        _ = multi.channels
    assert sorted(multi.parts[0].channels) == ["x", "y"]
    assert multi.channel_names() == {"x", "y"}


def test_geometry_needs_exactly_one_of_parts_or_channels():
    frame = make_frame(FrameSpace.STATE2, 2, ("x", "y"))
    with pytest.raises(InvalidParameterError, match="exactly one"):
        Geometry("t", frame)
    with pytest.raises(InvalidParameterError, match="exactly one"):
        Geometry("t", frame, [Part({"x": [0.0]})], channels={"x": [0.0]})


def test_channels_are_typed_and_categorical_ones_say_so():
    """The channel *type* is the fact a renderer should never have to re-derive."""
    labels = Channel("c", np.array([0, 1, 1, 2]), ChannelType.NOMINAL)
    speed = Channel("c", np.array([0.1, 0.2]))
    assert labels.is_categorical and not speed.is_categorical
    assert speed.type is ChannelType.QUANTITATIVE
    part = Part({"c": labels, "x": [0.0, 1.0]})
    assert part.channels["c"] is labels
    assert part.channels["x"].type is ChannelType.QUANTITATIVE


def test_make_frame_normalizes_labels_the_way_the_overlay_check_does():
    """A frame's axis names come from the same normalization the overlay rule uses.

    Otherwise a transform labelling its axis ``"$x$"`` and one labelling it
    ``"x"`` would be refused an overlay on typography.
    """
    assert make_frame(FrameSpace.STATE2, 2, ("$x$", "v")) == Frame(FrameSpace.STATE2, 2, ("x", "v"))
    # A time frame draws two axes but has one coordinate; the extra label is
    # presentation, not a coordinate, so it never reaches the frame.
    assert make_frame(FrameSpace.TIME, 1, ("t", "x")).axes == ("t",)
    # Fewer labels than coordinates is padded with "I did not say", never dropped.
    assert make_frame(FrameSpace.STATE3, 3, ("x",)).axes == ("x", "", "")


# ---------------------------------------------------------------------------
# Registration — what the decorator refuses
# ---------------------------------------------------------------------------


def _ok_geometry(_subject, **_kw):
    return Geometry(
        "scratch",
        make_frame(FrameSpace.STATE2, 2, ("x", "y")),
        channels={"x": np.arange(4.0), "y": np.arange(4.0)},
        axis_labels=("x", "y"),
    )


def test_registering_a_transform_touches_only_its_own_module(scratch_registry):
    """The headline claim: a new plot is one registration, and nothing else.

    No renderer edit, no ``PlotKind`` edit, no ``compose`` edit, no test edit —
    and the result is immediately reachable through every front door.
    """
    from tsdynamics.viz.spec import PlotKind as _PlotKind

    before_kinds = set(_PlotKind)

    @scratch_registry(
        name="scratch",
        source="data",
        kind=PlotKind.PHASE_PORTRAIT_2D,
        frame=FrameSpace.STATE2,
        ndim=2,
        default_primitive="line",
        primitives=("line", "points"),
        example=lambda primitive: (None, {}),
        doc="a throwaway transform",
    )
    def scratch(subject, **kw):
        return _ok_geometry(subject, **kw)

    assert "scratch" in names()
    assert compatibility("scratch") == ("line*", "points")
    spec = build_spec(None, "scratch")
    assert isinstance(spec, PlotSpec)
    assert [str(layer.kind) for layer in spec.layers] == ["line"]
    assert spec.layers[0].transform == "scratch"
    # reachable from the top-level front door with no further wiring
    assert plot(None, "scratch").kind is PlotKind.PHASE_PORTRAIT_2D
    assert plot(None, "scratch.points").layers[0].kind is PlotKind.SCATTER
    # ... and the frozen plot vocabulary is untouched
    assert set(_PlotKind) == before_kinds


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"default_primitive": "points", "primitives": ("line",)}, "not in its row"),
        ({"primitives": ("line", "rainbow")}, "unknown primitive"),
        ({"primitives": ("line",), "exclusive": ("points",)}, "exclusive"),
        (
            {"primitives": ("line", "quiver"), "frame": FrameSpace.SCALING},
            "structurally impossible",
        ),
        ({"name": "spectrogram"}, "generic time-series toolbox"),
    ],
)
def test_registration_refuses_an_impossible_declaration(scratch_registry, kwargs, match):
    """A row that cannot be honored is rejected where it is written, not at call time."""
    base = dict(
        name="scratch",
        source="data",
        kind=PlotKind.PHASE_PORTRAIT_2D,
        frame=FrameSpace.STATE2,
        ndim=2,
        default_primitive="line",
        primitives=("line", "points"),
        example=lambda primitive: (None, {}),
    )
    base.update(kwargs)
    with pytest.raises(InvalidParameterError, match=match):
        scratch_registry(**base)(_ok_geometry)


def test_registration_refuses_a_keyword_that_two_layers_would_claim(scratch_registry):
    """A compute parameter shadowing a primitive option is caught at registration.

    The caller's keywords are split between ``compute`` and the primitive by
    name, so an ambiguous name would otherwise become a silently mis-routed
    argument at call time.
    """
    with pytest.raises(InvalidParameterError, match="unambiguous"):

        @scratch_registry(
            name="scratch",
            source="data",
            kind=PlotKind.PHASE_PORTRAIT_2D,
            frame=FrameSpace.STATE2,
            ndim=2,
            default_primitive="density",
            primitives=("density",),
            example=lambda primitive: (None, {}),
        )
        def scratch(subject, *, bins=10):  # ``bins`` is also the density option
            return _ok_geometry(subject)


def test_a_transform_that_does_not_return_geometry_is_caught(scratch_registry):
    @scratch_registry(
        name="scratch",
        source="data",
        kind=PlotKind.PHASE_PORTRAIT_2D,
        frame=FrameSpace.STATE2,
        ndim=2,
        default_primitive="line",
        primitives=("line",),
        example=lambda primitive: (None, {}),
    )
    def scratch(subject):
        return PlotSpec(kind=PlotKind.PHASE_PORTRAIT_2D)

    with pytest.raises(InvalidInputError, match="not a Geometry"):
        geometry(None, "scratch")


def test_a_transform_that_computes_an_undeclared_frame_is_caught(scratch_registry):
    @scratch_registry(
        name="scratch",
        source="data",
        kind=PlotKind.PHASE_PORTRAIT_2D,
        frame=FrameSpace.STATE2,
        ndim=2,
        default_primitive="line",
        primitives=("line",),
        example=lambda primitive: (None, {}),
    )
    def scratch(subject):
        return Geometry(
            "scratch",
            make_frame(FrameSpace.TIME, 1, ("t",)),
            channels={"x": np.arange(3.0), "y": np.arange(3.0)},
        )

    with pytest.raises(InvalidParameterError, match="declares coordinate space"):
        geometry(None, "scratch")


def test_an_unknown_transform_name_lists_the_registered_ones():
    # Deliberately a name no roadmap entry claims: "nullclines" used to stand
    # here and is now a real row, which is exactly the trap a hard-coded
    # not-yet-implemented name sets for the next contributor.
    with pytest.raises(InvalidParameterError, match="Registered transforms"):
        get("not_a_transform_and_never_will_be")


# ---------------------------------------------------------------------------
# The build path: geometry -> primitive -> spec
# ---------------------------------------------------------------------------


def test_the_default_primitive_can_follow_the_data():
    """A map orbit is a point sequence; a flow is a curve.  Both stay swappable."""
    assert build_spec(_traj(), "time_series").layers[0].kind is PlotKind.LINE
    assert build_spec(_orbit(), "time_series").layers[0].kind is PlotKind.SCATTER
    # ... and an explicit choice still wins over the data-driven default
    assert build_spec(_orbit(), "time_series", primitive="line").layers[0].kind is PlotKind.LINE


def test_a_primitive_valid_for_the_transform_but_not_this_geometry_says_which():
    """The row is a statement about the transform; a shape can narrow it.

    A 2-D spatial field cannot be a line, and the message distinguishes that from
    "this primitive is not valid here at all".
    """
    from tsdynamics.data import Trajectory

    t = np.linspace(0.0, 1.0, 4)
    y = np.stack([np.sin(np.arange(24) * 0.3 + ti) for ti in t])
    field = Trajectory(t, y, _DemoSystem(), {"field_shape": (4, 6)})
    with pytest.raises(InvalidParameterError, match="not for \\*this\\* geometry"):
        build_spec(field, "spatial_field", primitive="line")


def test_options_split_between_the_transform_and_the_primitive():
    """``bins`` reaches the primitive, ``components`` reaches ``compute``."""
    spec = build_spec(_traj(), "phase_portrait", primitive="density", components=[0, 1], bins=17)
    assert spec.layers[0].data["z"].shape == (17, 17)
    # a primitive keyword aimed at a primitive that does not take it is refused,
    # rather than silently dropped
    with pytest.raises(InvalidParameterError, match="does not accept keyword"):
        build_spec(_traj(), "phase_portrait", primitive="line", components=[0, 1], bins=17)


def test_geometry_and_draw_are_the_raw_array_escape_hatch():
    """Rung 4: get the numbers, look at them, hand them back."""
    g = geometry(_traj(), "phase_portrait", components=[0, 1])
    assert g.space is FrameSpace.STATE2
    assert g.axes == ("x", "y")
    assert sorted(g.channels) == ["x", "y"]
    assert g.channels["x"].values.shape == (120,)
    spec = draw(g, "points")
    assert spec.layers[0].kind is PlotKind.SCATTER
    layers = lower(g, "line")
    assert len(layers) == 1 and isinstance(layers[0], Layer)


def test_a_heterogeneous_geometry_pins_the_part_that_needs_it():
    """A field with a host orbit is a quiver part *and* a line part.

    No single primitive draws both, and pretending otherwise would turn the orbit
    into arrows.
    """

    def rhs(u):
        return np.array([-u[1], u[0]])

    g = geometry(rhs, "phase_portrait_field", source=_traj(dim=2), grid=4)
    assert [p.primitive for p in g.parts] == [None, "line"]
    marks = [str(layer.kind) for layer in lower(g, "quiver")]
    assert marks == ["quiver", "line"]


def test_every_lowered_layer_carries_its_transform_provenance():
    """Provenance is what makes per-source restyling inside an overlay possible."""
    for record in transforms():
        subject, options = record.example(record.default_primitive)
        spec = build_spec(subject, record.name, **dict(options))
        assert spec.layers, record.name
        assert {layer.transform for layer in spec.layers} == {record.name}


def test_the_reserved_primitives_draw_when_handed_the_channels_they_need():
    """A primitive with no transform row yet is still exercised, never untested code."""
    from tsdynamics.viz.transforms._base import Geometry as _Geometry

    frame = make_frame(FrameSpace.CATEGORY, 1, ("k",))
    samples = {
        "bars": {"x": np.arange(3.0), "y": np.array([1.0, -2.0, 0.5])},
        "errorbars": {
            "x": np.arange(3.0),
            "y": np.array([1.0, 2.0, 3.0]),
            "err": np.array([0.1, 0.2, 0.1]),
        },
        "band": {
            "x": np.arange(4.0),
            "lo": np.zeros(4),
            "hi": np.ones(4),
            "y": np.full(4, 0.5),
        },
        "markers": {"x": np.arange(3.0), "y": np.arange(3.0)},
        "boundary": {
            "x": np.arange(4.0),
            "y": np.arange(3.0),
            "z": np.array([[0, 0, 1, 1]] * 3, dtype=float),
        },
    }
    assert set(samples) == set(RESERVED_PRIMITIVES)
    for name, channels in samples.items():
        g = _Geometry("reserved", frame, channels=channels)
        layers = PRIMITIVES[name].build(g, g.parts[0], {})
        assert layers, name
        assert all(layer.data for layer in layers), name


# ---------------------------------------------------------------------------
# ts.plot / ts.T
# ---------------------------------------------------------------------------


def test_plot_without_a_transform_is_the_composition_front_door():
    import tsdynamics.viz as viz

    traj = _traj()
    assert plot(traj).to_dict() == viz.plot(traj).to_dict()
    assert plot(traj, components=[0, 1]).kind is PlotKind.PHASE_PORTRAIT_2D


def test_plot_with_a_transform_name_builds_it():
    spec = plot(_traj(), "delay_embedding", tau=7)
    assert spec.kind is PlotKind.PHASE_PORTRAIT_2D
    assert spec.layers[0].transform == "delay_embedding"


def test_plot_overlays_a_new_transform_onto_a_built_in_one(scratch_registry):
    """The flagship shape, and the extension claim at the same time.

    A transform registered from outside this module composes onto a built-in one
    on a single set of axes, with no edit anywhere else — and the draw order
    follows the declared **role** (a curve under its annotation), so the call is
    order-free.
    """

    @scratch_registry(
        name="scratch_marks",
        source="data",
        kind=PlotKind.FIXED_POINTS_OVERLAY,
        frame=FrameSpace.STATE2,
        ndim=2,
        role=OverlayRole.OVERLAY,
        default_primitive="markers",
        primitives=("markers", "points"),
        example=lambda primitive: (None, {}),
    )
    def scratch_marks(subject):
        return Geometry(
            "scratch_marks",
            make_frame(FrameSpace.STATE2, 2, ("x", "y")),
            channels={"x": np.array([0.0, 0.5]), "y": np.array([0.0, -0.5])},
            axis_labels=("x", "y"),
            label="marks",
        )

    forward = plot(_traj(), T("phase_portrait", components=["x", "y"]), "scratch_marks")
    backward = plot(_traj(), "scratch_marks", T("phase_portrait", components=["x", "y"]))
    assert [str(layer.kind) for layer in forward.layers] == ["line", "markers"]
    # z-order is by role, so the two calls are the same picture
    assert [layer.transform for layer in forward.layers] == [
        layer.transform for layer in backward.layers
    ]
    assert forward.resolved_frame == Frame(FrameSpace.STATE2, 2, ("x", "y"))


def test_plot_refuses_an_overlay_of_incompatible_frames():
    with pytest.raises(InvalidParameterError, match="different spaces"):
        plot(_traj(), "time_series", "phase_portrait")


def test_t_carries_per_transform_options_and_style():
    spec = plot(
        _traj(),
        T("phase_portrait", components=[0, 1], color="red", linewidth=3.0),
    )
    assert spec.layers[0].style["color"] == "red"
    assert spec.layers[0].style["linewidth"] == 3.0
    assert repr(T("basins", grid=400)).startswith("T('basins'")
    assert T("basins.boundary").primitive == "boundary"


def test_a_shared_keyword_only_reaches_the_transforms_that_accept_it():
    """``plot(subject, "a", "b", opt=…)`` must not hand ``opt`` to a transform without it.

    ``phase_portrait`` takes no ``tau``; handing it one would raise, so the fact
    that this composes at all is the assertion.
    """
    spec = plot(
        _traj(),
        T("phase_portrait", components=[0, 1]),
        "delay_embedding",
        tau=9,
        layout="row",
    )
    assert spec.kind is PlotKind.COMPOSITE
    assert [p.layers[0].transform for p in spec.panels] == ["phase_portrait", "delay_embedding"]
    assert spec.panels[1].x.label == "x(t)"


def test_plot_needs_exactly_one_subject_when_transforms_are_named():
    with pytest.raises(InvalidParameterError, match="exactly one subject"):
        plot(_traj(), _traj(), "time_series")
    with pytest.raises(InvalidParameterError, match="exactly one subject"):
        plot("time_series")


def test_primitive_without_a_named_transform_is_an_error_not_a_no_op():
    with pytest.raises(InvalidParameterError, match="no transform was named"):
        plot(_traj(), primitive="density")


# ---------------------------------------------------------------------------
# primitive= on the trajectory front door
# ---------------------------------------------------------------------------


def test_trajectory_front_door_takes_a_primitive():
    traj = _traj()
    assert traj.to_plot_spec().layers[0].kind is PlotKind.LINE3D
    assert (
        traj.to_plot_spec(kind="phase_portrait_2d", primitive="points").layers[0].kind
        is PlotKind.SCATTER
    )
    steps = traj.to_plot_spec(kind="time_series", primitive="steps")
    plain = traj.to_plot_spec(kind="time_series")
    assert steps.layers[0].data["x"].size == 2 * plain.layers[0].data["x"].size - 1


def test_trajectory_front_door_default_is_unchanged_by_the_registry():
    """``primitive=None`` is byte-for-byte the pre-registry front door.

    The transforms landed picture-preserving; the front door only consults the
    registry when asked for a *different drawing* of the same numbers.
    """
    spec = _traj().to_plot_spec()
    assert spec.layers[0].transform is None  # not built through a transform
    assert spec.kind is PlotKind.PHASE_PORTRAIT_3D


def test_a_view_with_no_transform_refuses_a_primitive_rather_than_ignoring_it():
    traj = _traj()
    traj.meta["plot_kind"] = "poincare_section"
    with pytest.raises(InvalidParameterError, match="no registered plot transform"):
        traj.to_plot_spec(primitive="points")


def test_system_front_door_forwards_the_primitive():
    """``system.plot(primitive=…)`` must not leak the keyword to the integrator."""
    from tsdynamics.data.trajectory import _PLOT_SPEC_KEYS

    assert "primitive" in _PLOT_SPEC_KEYS


# ---------------------------------------------------------------------------
# The scope boundary — a checked rule, not a convention
# ---------------------------------------------------------------------------


def test_the_admitted_series_diagnostics_are_exactly_this_set():
    """The PSD is admitted; the PSD *toolbox* is not.  Nothing else follows it in.

    The v6 scope surgery deleted the generic time-series layer on the principle
    *phase-space methods stay, generic series statistics go*.  The owner
    re-admitted one member under a rule narrow enough to check:

        the PSD of a phase-space trajectory is a phase-space diagnostic; a PSD
        toolbox with windowing, detrending and filter design is not.

    Changing this assertion is the deliberate act that re-opens that decision.
    """
    assert frozenset({"psd"}) == ADMITTED_SERIES_DIAGNOSTICS


def test_the_registry_docstring_carries_the_admission_rule():
    """The rule lives where someone adding a transform will actually read it."""
    from tsdynamics import registry

    source = __import__("inspect").getsource(registry)
    marker = "phase-space diagnostic"
    assert marker in source, "the PSD admission rule left registry.py"
    assert "plot_transforms" in registry.__all__


@pytest.mark.parametrize("name", sorted(EXCLUDED_SERIES_TOOLBOX))
def test_the_excluded_toolbox_cannot_be_registered_as_a_transform(name, scratch_registry):
    """A plot transform is not a back door for the layer the scope surgery removed."""
    assert name not in names()
    with pytest.raises(InvalidParameterError, match="generic time-series toolbox"):
        scratch_registry(
            name=name,
            source="data",
            kind=PlotKind.PHASE_PORTRAIT_2D,
            frame=FrameSpace.STATE2,
            ndim=2,
            default_primitive="line",
            primitives=("line",),
            example=lambda primitive: (None, {}),
        )(_ok_geometry)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_compatibility_is_readable_and_programmable():
    matrix = compatibility()
    assert set(matrix) == set(names())
    assert "primitives" in repr(matrix) or "transform" in repr(matrix)
    rows = matrix.rows()
    assert {r["transform"] for r in rows} == set(names())
    assert all(set(r) >= {"source", "space", "default", "available"} for r in rows)


def test_an_unavailable_optional_dependency_is_listed_not_hidden(scratch_registry):
    """A missing extra reads as "install it", never as "this plot does not exist"."""
    from tsdynamics.analysis._result_viz import VisualizationNotInstalled

    @scratch_registry(
        name="scratch",
        source="data",
        kind=PlotKind.PHASE_PORTRAIT_2D,
        frame=FrameSpace.STATE2,
        ndim=2,
        default_primitive="line",
        primitives=("line",),
        requires="a_package_that_does_not_exist",
        example=lambda primitive: (None, {}),
    )
    def scratch(subject):
        return _ok_geometry(subject)

    record = get("scratch")
    assert record.available is False
    assert "scratch" in compatibility()  # listed ...
    assert "unavailable" in repr(compatibility())  # ... and flagged
    with pytest.raises(VisualizationNotInstalled, match="a_package_that_does_not_exist"):
        build_spec(None, "scratch")


def test_transform_records_declare_a_role_for_order_free_overlays():
    """Role is what makes ``plot(basins, traj)`` and ``plot(traj, basins)`` one picture."""
    roles = {t.name: t.role for t in transforms()}
    assert roles["vector_field"] is OverlayRole.FIELD
    assert roles["spacetime"] is OverlayRole.FIELD
    assert roles["phase_portrait"] is OverlayRole.BASE
    assert all(isinstance(r, OverlayRole) for r in roles.values())


# ── animate= must survive the per-transform keyword filter ────────────────────


def test_animate_reaches_the_spec_through_the_transform_front_door() -> None:
    """``ts.plot(..., animate=True)`` must animate, in all three spellings.

    ``_build_one`` filters shared keywords against the *transform's* signature, so
    a keyword that is about the FIGURE rather than the computation was dropped on
    the floor.  ``animate`` was: ``ts.plot(traj, "phase_portrait", animate=True)``
    returned a static spec, and — worse than raising — ``.save("x.gif")`` then
    wrote a perfectly valid ONE-FRAME gif, so the failure looked like success.
    Same class as the dropped ``color=``: a keyword accepted and ignored.
    """
    import tsdynamics as ts
    from tsdynamics.viz.spec import Animation

    traj = ts.systems.Lorenz().integrate(final_time=5.0, dt=0.05, ic=[1.0, 1.0, 1.0])

    assert ts.plot(traj, "phase_portrait", animate=True).is_animated
    assert ts.plot(traj, "phase_portrait", animate={"fps": 24}).is_animated
    assert ts.plot(traj, "time_series", animate=Animation()).is_animated
    # and the knob passed through the dict spelling actually lands
    assert ts.plot(traj, "phase_portrait", animate={"fps": 24}).animation.fps == 24
    # not animated unless asked
    assert not ts.plot(traj, "phase_portrait").is_animated


def test_an_animated_transform_spec_writes_a_real_movie(tmp_path) -> None:
    """The gif must have more than one frame — that is what the old bug produced."""
    pytest.importorskip("matplotlib")
    from PIL import Image

    import tsdynamics as ts

    traj = ts.systems.Lorenz().integrate(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    path = tmp_path / "orbit.gif"
    ts.plot(traj, "phase_portrait", animate=True).save(path)

    assert path.stat().st_size > 0
    with Image.open(path) as img:
        assert getattr(img, "n_frames", 1) > 1
