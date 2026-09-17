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
    """A stand-in carrying ``family`` — the v6 spelling a real system has.

    ``is_discrete`` was removed in v6 and is derived here, not stored: a test
    fake that keeps a removed attribute alive is how a rename passes its own
    suite while every real system silently answers the default.
    """

    def __init__(self, discrete: bool = False, variables: tuple[str, ...] | None = None) -> None:
        self.family = "map" if discrete else "ode"
        self.variables = variables

    @property
    def is_discrete(self) -> bool:
        """Deprecated spelling of ``family == "map"`` — for readers not yet repointed."""
        return self.family == "map"


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
        decorate = plot_transform(**kwargs)

        def wrap(fn):
            # ``name`` is derived from ``fn.__name__`` unless declared, so the
            # cleanup list can only be built once the function is in hand.
            made.append(kwargs.get("name", fn.__name__))
            return decorate(fn)

        return wrap

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
    frame = make_frame(FrameSpace.STATE2, ("x", "y"))
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
    assert make_frame(FrameSpace.STATE2, ("$x$", "v")) == Frame(FrameSpace.STATE2, 2, ("x", "v"))
    # A time frame draws two axes but has one coordinate; the extra label is
    # presentation, not a coordinate, so it never reaches the frame.
    assert make_frame(FrameSpace.TIME, ("t", "x")).axes == ("t",)
    # Fewer labels than coordinates is padded with "I did not say", never dropped.
    assert make_frame(FrameSpace.STATE3, ("x",)).axes == ("x", "", "")


# ---------------------------------------------------------------------------
# Registration — what the decorator refuses
# ---------------------------------------------------------------------------


def _ok_geometry(_subject, **_kw):
    return Geometry(
        "scratch",
        make_frame(FrameSpace.STATE2, ("x", "y")),
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
        ({"kind": None}, "declares no kind"),
        ({"source": "oracle"}, "exactly two categories"),
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
            make_frame(FrameSpace.TIME, ("t",)),
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

    frame = make_frame(FrameSpace.CATEGORY, ("k",))
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
    # v6 claimed three of these — `bars` by the lyapunov-spectrum row, `band` by
    # `ensemble_fan`, `boundary` by `basins` — so the compatibility gate renders
    # them now.  What is still reserved must be a subset of what is smoke-tested
    # here, and nothing may be reserved without an example.
    assert set(RESERVED_PRIMITIVES) <= set(samples)
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
    spec = plot(_traj(), "delay_embedding", delay=7)
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
            make_frame(FrameSpace.STATE2, ("x", "y")),
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

    ``phase_portrait`` takes no ``delay``; handing it one would raise, so the fact
    that this composes at all is the assertion.
    """
    spec = plot(
        _traj(),
        T("phase_portrait", components=[0, 1]),
        "delay_embedding",
        delay=9,
        layout="row",
    )
    assert spec.kind is PlotKind.COMPOSITE
    assert [p.layers[0].transform for p in spec.panels] == ["phase_portrait", "delay_embedding"]
    assert spec.panels[1].x.label == "x(t)"


def test_a_named_transform_applies_to_every_subject_that_admits_it():
    """**The headline fix.** Two subjects and one transform is one figure, not a refusal.

    ``ts.plot(a, b, "phase_portrait")`` — the most obvious comparison plot in
    dynamics — used to answer *"naming transform(s) ['phase_portrait'] needs
    exactly one subject to apply them to, got 2"*.
    """
    one = plot(_traj(), "time_series")
    both = plot(_traj(), _traj(), "time_series")
    assert {layer.transform for layer in both.layers} == {"time_series"}
    assert len(both.layers) == 2 * len(one.layers)  # both orbits, one figure
    # ...and every legend entry names exactly one of them
    labels = [layer.label for layer in both.layers if layer.label]
    assert len(set(labels)) == len(labels)


def test_a_transform_no_subject_admits_raises_naming_what_it_needs():
    """A model transform with only data to work on says so, and says what to pass."""
    with pytest.raises(InvalidParameterError, match="none of the 1 subject"):
        plot(_traj(), "nullclines")
    with pytest.raises(InvalidParameterError, match="needs something to apply them to"):
        plot("time_series")


def test_primitive_without_a_named_transform_is_an_error_not_a_no_op():
    with pytest.raises(InvalidParameterError, match="no transform was named"):
        plot(_traj(), primitive="density")


# ---------------------------------------------------------------------------
# primitive= on the trajectory front door
# ---------------------------------------------------------------------------


def test_trajectory_front_door_takes_a_primitive():
    traj = _traj()
    assert traj.__plot_spec__().layers[0].kind is PlotKind.LINE3D
    assert (
        traj.__plot_spec__(kind="phase_portrait_2d", primitive="points").layers[0].kind
        is PlotKind.SCATTER
    )
    steps = traj.__plot_spec__(kind="time_series", primitive="steps")
    plain = traj.__plot_spec__(kind="time_series")
    assert steps.layers[0].data["x"].size == 2 * plain.layers[0].data["x"].size - 1


def test_trajectory_front_door_default_is_unchanged_by_the_registry():
    """``primitive=None`` is byte-for-byte the pre-registry front door.

    The transforms landed picture-preserving; the front door only consults the
    registry when asked for a *different drawing* of the same numbers.
    """
    spec = _traj().__plot_spec__()
    assert spec.layers[0].transform is None  # not built through a transform
    assert spec.kind is PlotKind.PHASE_PORTRAIT_3D


def test_a_view_with_no_transform_refuses_a_primitive_rather_than_ignoring_it():
    traj = _traj()
    traj.meta["plot_kind"] = "poincare_section"
    with pytest.raises(InvalidParameterError, match="no registered plot transform"):
        traj.__plot_spec__(primitive="points")


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

    traj = ts.systems.Lorenz().run(final_time=5.0, dt=0.05, ic=[1.0, 1.0, 1.0])

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

    traj = ts.systems.Lorenz().run(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    path = tmp_path / "orbit.gif"
    ts.plot(traj, "phase_portrait", animate=True).save(path)

    assert path.stat().st_size > 0
    with Image.open(path) as img:
        assert getattr(img, "n_frames", 1) > 1


# ---------------------------------------------------------------------------
# v6: the extension doors — four declarations, and nothing else
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_primitive():
    """Register primitives that are removed again afterwards."""
    from tsdynamics.viz.transforms import PRIMITIVES as _P

    before = dict(_P)
    yield
    _P.clear()
    _P.update(before)


def test_four_declarations_are_enough_to_add_a_plot(scratch_registry):
    """The newcomer's door: source, frame, kind, primitives — everything else derived."""

    @scratch_registry(
        source="data", frame="time", kind="diagnostic_curve", primitives=("line", "points")
    )
    def speed(traj):
        """Instantaneous speed |dx/dt| along the orbit."""
        dt = np.diff(traj.t)
        return {"x": traj.t[1:], "y": np.linalg.norm(np.diff(traj.y, axis=0), axis=1) / dt}

    record = get("speed")
    assert record.name == "speed"  # from __name__
    assert record.doc.startswith("Instantaneous speed")  # from __doc__
    assert record.ndim == (1,)  # from the frame's arity
    assert record.default_primitive == "line"  # from primitives[0]
    assert record.subjects == ("trajectory", "array", "system")  # from source
    # ...and with no other edit anywhere, all of this works:
    assert plot(_traj(), "speed").kind is PlotKind.DIAGNOSTIC_CURVE
    assert plot(_traj(), "speed", primitive="points").layers[0].kind is PlotKind.SCATTER
    assert plot(_traj(), "speed.points").layers[0].kind is PlotKind.SCATTER
    assert "speed" in compatibility()
    assert "speed" in names()
    assert "speed" in ts_find(subject=_traj())


def ts_find(**kw):
    from tsdynamics.viz.transforms import find

    return find(**kw)


def test_a_transform_may_return_a_list_of_mappings(scratch_registry):
    """The plural case without an IR type: one Part per mapping, four reserved keys."""

    @scratch_registry(
        source="data", frame="scaling", kind="scaling_fit", primitives=("line", "points")
    )
    def pair(traj):
        """A measured curve and its fit."""
        x = np.arange(5.0)
        return [
            {"x": x, "y": x**2, "label": "measured"},
            {"x": x, "y": x**2 + 1, "label": "fit", "style": {"linestyle": "dashed"}},
            {"x": x, "y": x, "label": "guide", "primitive": "points"},
        ]

    spec = plot(_traj(), "pair")
    assert [layer.label for layer in spec.layers] == ["measured", "fit", "guide"]
    assert spec.layers[1].style["linestyle"] == "dashed"
    assert [str(layer.kind) for layer in spec.layers] == ["line", "line", "scatter"]


def test_a_new_primitive_is_one_decorator_and_returns_mappings(scratch_primitive, scratch_registry):
    """The second extension door, with the *same* return convention as the first."""
    from tsdynamics.viz.transforms import get_primitive, primitive_names, register_primitive

    @register_primitive("stem", requires=("x", "y"), marks=("line", "points"))
    def stem(part, **options):
        """A vertical drop to the baseline plus a marker at each point."""
        x, y = part["x"], part["y"]
        xs, ys = np.repeat(x, 3), np.empty(3 * len(y))
        ys[0::3], ys[1::3], ys[2::3] = options.get("baseline", 0.0), y, np.nan
        return [{"mark": "line", "x": xs, "y": ys}, {"mark": "points", "x": x, "y": y}]

    assert "stem" in primitive_names()
    assert get_primitive("stem").requires == frozenset({"x", "y"})

    @scratch_registry(
        source="data", frame="time", kind="diagnostic_curve", primitives=("stem", "line")
    )
    def spikes(traj):
        """A spike train."""
        return {"x": traj.t[:6], "y": np.arange(6.0)}

    spec = plot(_traj(), "spikes")
    assert [str(layer.kind) for layer in spec.layers] == ["line", "scatter"]
    # no new PlotKind was needed to add a way of drawing
    assert {str(k) for k in get_primitive("stem").marks} <= {str(k) for k in PlotKind}


def test_draw_takes_plain_arrays_and_gives_back_a_plot():
    """The owner's ask by name: hand arrays to a primitive, get a Plot."""
    r = np.linspace(0.1, 2.0, 20)
    p = draw({"x": r, "y": r**2}, "line", labels=("log r", "log C(r)"))
    assert isinstance(p, PlotSpec)
    assert (p.x.label, p.y.label) == ("log r", "log C(r)")
    assert [str(layer.kind) for layer in p.layers] == ["line"]
    assert np.allclose(p.layers[0].data["y"], r**2)


def test_draw_takes_a_list_of_mappings_with_labels_styles_and_primitives():
    r = np.linspace(0.1, 2.0, 20)
    p = draw(
        [
            {"x": r, "y": r**2, "label": "data"},
            {"x": r, "y": r**2 + 1, "label": "fit", "style": {"linestyle": "dashed"}},
            {"x": r, "lo": r, "hi": r + 1, "primitive": "band", "style": {"alpha": 0.2}},
        ],
        "line",
        title="correlation sum",
    )
    assert [layer.label for layer in p.layers][:2] == ["data", "fit"]
    assert p.layers[1].style["linestyle"] == "dashed"
    assert str(p.layers[2].kind) == "area"
    assert p.title == "correlation sum"


def test_draw_refuses_something_that_is_not_channels():
    with pytest.raises(InvalidInputError, match="channel mapping"):
        draw(_traj(), "line")


def test_a_hand_built_geometry_needs_no_registered_transform():
    """``Geometry.transform`` is provenance, not a lookup key."""
    g = Geometry(
        "mine",
        make_frame(FrameSpace.FREE, ("a", "b")),
        channels={"x": np.arange(5.0), "y": np.arange(5.0)},
    )
    spec = draw(g, "points")
    assert [layer.transform for layer in spec.layers] == ["mine"]


def test_a_model_transform_refuses_measured_data_by_name():
    """Before v6: ``AttributeError: 'numpy.ndarray' object has no attribute 'jacobian'``."""
    with pytest.raises(InvalidInputError, match="needs a dynamical system"):
        build_spec(np.linspace(0.0, 1.0, 10), "nullclines")


def test_an_alias_is_a_second_spelling_not_a_second_row():
    """``direction_field`` resolves to ``vector_field``; the matrix lists one row."""
    assert get("direction_field") is get("vector_field")
    assert "direction_field" not in compatibility()
    assert get("vector_field").aliases == ("direction_field",)


def test_the_guessable_name_is_the_working_one():
    """``ts.plot(system, "vector_field")`` used to raise ``missing 'xlim' and 'ylim'``."""
    from tsdynamics.systems import VanDerPol

    spec = plot(VanDerPol(), "vector_field")
    assert spec.kind is PlotKind.VECTOR_FIELD
    assert "u" in spec.layers[0].data and "v" in spec.layers[0].data


def test_a_primitive_that_would_drop_the_colour_raises(scratch_registry):
    """Measured: ``steps`` discarded the colour channel **and kept the colorbar**."""

    @scratch_registry(
        source="data", frame="time", kind="diagnostic_curve", primitives=("line", "steps")
    )
    def coloured(traj):
        """A curve carrying a colour channel."""
        return {"x": traj.t, "y": traj.y[:, 0], "c": traj.t}

    assert "c" in plot(_traj(), "coloured").layers[0].data
    with pytest.raises(InvalidParameterError, match="cannot draw the colour channel"):
        plot(_traj(), "coloured.steps")


def test_transforms_answers_the_four_shared_registry_verbs():
    """``register`` / ``names`` / ``find`` / ``get`` — the same shape as every registry."""
    import tsdynamics as ts

    for verb in ("register", "names", "find", "get"):
        assert callable(getattr(ts.viz.transforms, verb)), verb
    assert ts.viz.transforms.find(source="model")
    assert set(ts.viz.transforms.find(source="model")) < set(ts.viz.transforms.names())
    assert "nullclines" not in ts.viz.transforms.find(subject=_traj())
    assert "time_series" in ts.viz.transforms.find(subject=_traj())
    assert "psd" in ts.viz.transforms.find("spectral")


def test_the_matrix_reads_as_an_answer_to_what_can_i_draw():
    """``compatibility()`` is the newcomer's first question; it must answer it."""
    text = repr(compatibility())
    assert "FROM DATA" in text and "FROM A MODEL" in text
    assert "* = the default primitive" in text
    assert "ts.plot(subject, 'name')" in text
    # every transform is listed, with its one-line summary under it
    for record in transforms():
        assert record.name in text
        assert record.doc in text


# ---------------------------------------------------------------------------
# v6 — one vocabulary, honest verbs
# ---------------------------------------------------------------------------


def test_draw_speaks_the_same_vocabulary_as_every_other_plotting_door() -> None:
    """``ts.viz.draw`` is a door, not a primitive wrapper.

    Measured before v6: **all ten style keys and sixteen of the seventeen figure
    keywords raised**, and the message blamed the primitive — *"primitive 'line'
    does not accept keyword(s) ['color']; it accepts (none)"* — for a word the
    door had simply never peeled.  Only ``title=`` worked.
    """
    import numpy as np

    import tsdynamics as ts
    from tsdynamics.viz.spec import FIGURE_KEYS
    from tsdynamics.viz.style import STYLE_KEYS

    r = np.logspace(-1.0, 1.0, 24)
    c = r**2.0
    values = {
        "color": "crimson",
        "linewidth": 2.0,
        "alpha": 0.5,
        "cmap": "viridis",
        "marker": "circle",
        "markersize": 4.0,
        "linestyle": "dashed",
        "fill": True,
        "filled": True,
        "fillalpha": 0.2,
        "zorder": 3,
        "title": "t",
        "xlabel": "log r",
        "ylabel": "log C",
        "zlabel": "z",
        "xscale": "log",
        "yscale": "log",
        "zscale": "linear",
        "xlim": (0.1, 10.0),
        "ylim": (0.01, 100.0),
        "zlim": (0.0, 1.0),
        "xticks": [0.1, 1.0],
        "yticks": [1.0],
        "zticks": [0.0],
        "clim": (0.0, 1.0),
        "colorbar": False,
        "legend": True,
        "theme": "dark",
    }
    vocabulary = sorted(set(STYLE_KEYS) | set(FIGURE_KEYS))
    for key in vocabulary:
        assert key in values, f"the gate must exercise every word: {key} is missing"
        ts.viz.draw({"x": r, "y": c}, "line", **{key: values[key]})
    # ...and they land, rather than being accepted and dropped.
    p = ts.viz.draw(
        {"x": r, "y": c}, "line", xlabel="log r", ylabel="log C", xscale="log", color="crimson"
    )
    assert (p.x.label, p.y.label, p.x.scale) == ("log r", "log C", "log")
    assert p.layers[0].style["color"] == "crimson"
    # ...and the aliases canonicalise here exactly as at every other door.
    assert ts.viz.draw({"x": r, "y": c}, "line", lw=3.0).layers[0].style["linewidth"] == 3.0


def test_draw_has_no_kind_keyword_and_says_which_word_names_a_picture() -> None:
    """``kind=`` on ``draw`` was dead by construction: every value raised.

    Passing it skipped the fallback that supplies a kind, so ``spec_of`` raised
    before the line that would have used it — and the message never mentioned
    ``kind`` at all.
    """
    import numpy as np

    import tsdynamics as ts
    from tsdynamics.errors import InvalidParameterError

    r = np.linspace(0.0, 1.0, 8)
    with pytest.raises(InvalidParameterError, match="no kind= keyword"):
        ts.viz.draw({"x": r, "y": r}, "line", kind="time_series")


def test_every_view_kind_uniquely_served_has_a_transform_spelling() -> None:
    """One word, one picture — the transform name, positionally.

    ``kind=`` used to be the only way to ask for three views: a Poincaré section
    (no transform existed), and the two forced dimensionalities of a phase
    portrait.  Each has a positional spelling now, which is what lets the second
    vocabulary retire.
    """
    import numpy as np

    import tsdynamics as ts

    plt = pytest.importorskip("matplotlib.pyplot")
    from tsdynamics.viz.spec import PlotKind

    ros = ts.systems.Rossler()
    tr = ts.systems.Lorenz().run(final_time=6.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    try:
        assert "poincare_section" in ts.viz.transforms.names()
        section = ts.plot(ros, "poincare_section", plane=("y", 0.0, "up"), crossings=40)
        assert section.kind is PlotKind.POINCARE_SECTION
        assert ts.plot(tr, "phase_portrait", ndim=2).kind is PlotKind.PHASE_PORTRAIT_2D
        assert ts.plot(tr, "phase_portrait", ndim=3).kind is PlotKind.PHASE_PORTRAIT_3D
        # The recipe spellings ``kind="delay"`` / ``kind="field"`` are aliases now.
        assert ts.viz.transforms.get("delay").name == "delay_embedding"
        assert ts.viz.transforms.get("field").name == "spatial_field"
        assert ts.viz.transforms.get("recurrence_plot").name == "recurrence"
        # A retired PlotKind spelling is answered with the line that works.
        with pytest.raises(Exception, match=r"phase_portrait', ndim=2"):
            ts.viz.transforms.get("phase_portrait_2d")
        assert np.isfinite(section.layers[0].data["x"]).all()
    finally:
        plt.close("all")


def test_find_subject_advertises_only_what_that_subject_can_be_handed() -> None:
    """``find(subject=…)`` is the user's question and must not over-promise.

    Measured before v6: a **map** was advertised all 38 transforms and 14 raised
    (every 2-D-flow field transform among them); an analysis **result** was
    advertised all 38 and 22 raised.  The old rule accepted everything that was
    not literally an array or a trajectory.
    """
    import numpy as np

    import tsdynamics as ts

    lor, vdp, henon = ts.systems.Lorenz(), ts.systems.VanDerPol(), ts.systems.Henon()
    traj = lor.run(final_time=6.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    spectrum = ts.analysis.lyapunov_spectrum(lor, final_time=20.0)
    field_transforms = {
        "vector_field",
        "flow_speed",
        "nullclines",
        "streamlines",
        "trace_determinant",
        "ftle",
        "escape_time",
        "transient_time",
    }
    for subject, label in ((henon, "a map"), (traj, "a trajectory"), (spectrum, "a result")):
        offered = set(ts.viz.transforms.find(subject=subject))
        assert not (offered & field_transforms), f"{label} was offered a vector-field transform"
    # A cobweb is the staircase of a MAP iteration; a 3-D flow's orbit has none,
    # and used to draw one anyway (kind='cobweb', two layers, no complaint).
    assert "cobweb" in ts.viz.transforms.find(subject=henon)
    assert "cobweb" not in ts.viz.transforms.find(subject=traj)
    # A 2-D flow keeps every field transform, which is the case that must not regress.
    assert field_transforms <= set(ts.viz.transforms.find(subject=vdp))
    # A recorded (times, estimates) pair is an ARRAY; a Trajectory is not one,
    # and was offered the transform that reads such a pair.
    assert "lyapunov_convergence" not in ts.viz.transforms.find(subject=traj)
    assert "lyapunov_convergence" in ts.viz.transforms.find(subject=lor)
    # A bare array is still data.
    assert "psd" in ts.viz.transforms.find(subject=np.sin(np.linspace(0, 40, 400)))


def test_a_user_primitive_can_be_applied_to_a_shipped_transform() -> None:
    """The extension point's missing half — ``ts.viz.transforms.allow``.

    Measured before v6: a newly registered primitive could reach **none** of the
    shipped transforms, because every declared row is a frozen tuple written
    before that primitive existed.  ``PlotTransform`` is frozen and there was no
    public verb to extend a row, so "define your own plots using the primitives"
    worked only with a transform you also wrote yourself.
    """
    import numpy as np

    import tsdynamics as ts
    from tsdynamics.errors import InvalidParameterError

    plt = pytest.importorskip("matplotlib.pyplot")

    @ts.viz.primitives.register("stem_gate", requires=("x", "y"), marks=("line", "points"))
    def stem_gate(part, **options):
        """A vertical drop to the baseline plus a marker at each point."""
        x, y = part["x"], part["y"]
        xs = np.repeat(x, 3)
        ys = np.empty(3 * len(y))
        ys[0::3], ys[1::3], ys[2::3] = 0.0, y, np.nan
        return [{"mark": "line", "x": xs, "y": ys}, {"mark": "points", "x": x, "y": y}]

    traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    try:
        assert "stem_gate" in ts.viz.primitives.names()
        with pytest.raises(InvalidParameterError, match="not valid for transform"):
            ts.plot(traj, "time_series", primitive="stem_gate")
        ts.viz.transforms.allow("time_series", "stem_gate")
        drawn = ts.plot(traj, "time_series", primitive="stem_gate")
        assert len(drawn.layers) == 6  # three components x (line + points)
        drawn.render()
        # A structurally impossible cell is still refused, by allow() itself.
        with pytest.raises(InvalidParameterError, match="structurally impossible"):
            ts.viz.transforms.allow("psd", "quiver")  # a quiver draws in state2 only
        with pytest.raises(InvalidParameterError, match="unknown primitive"):
            ts.viz.transforms.allow("time_series", "no_such_primitive")
    finally:
        plt.close("all")
        record = ts.viz.transforms.get("time_series")
        object.__setattr__(record, "primitives", record.primitives - {"stem_gate"})


def test_geometry_hands_back_numbers_not_objects() -> None:
    """``ts.viz.geometry`` is advertised as the arrays escape hatch.

    ``np.asarray`` of one used to be an ``(n_parts,)`` array of **objects** — a
    silent non-answer that plots as nothing and arithmetics into a ``TypeError``
    far from the call site.
    """
    import numpy as np

    import tsdynamics as ts

    traj = ts.systems.Lorenz().run(final_time=4.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    portrait = ts.viz.geometry(traj, "phase_portrait")
    arr = np.asarray(portrait)
    assert arr.dtype == np.float64 and arr.shape == (3, traj.y.shape[0])
    series = ts.viz.geometry(traj, "time_series")
    stacked = np.asarray(series)
    assert stacked.dtype == np.float64 and stacked.shape[0] == 2  # (x, y) over 3 parts
    assert series["y"].shape == (3, traj.y.shape[0])  # multi-part channel, stacked
