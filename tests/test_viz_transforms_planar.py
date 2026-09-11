r"""The phase-plane and scalar-field transforms as *plots*.

``tests/test_analysis_planar.py`` checks that the numbers are right; this module
checks that the picture drawn from them is right, which is a different question
and the one this whole layer exists to answer.  The failure mode being defended
against is specific: a spec that builds, renders without error, and shows
something other than what it claims.  So the assertions here are about drawn
artists and their coordinates, not about dict shapes.

Nine registered transforms::

    nullclines  direction_field  flow_speed  streamlines  trace_determinant
    ftle  escape_time  transient_time  invariant_density

The compatibility gate (``tests/test_viz_compatibility.py``) already renders
every declared cell of all nine and refuses every undeclared one — those checks
are not repeated here.  What is here is what the gate cannot know: that the
curves land in the right places, that the composition of four of them is one
coherent figure, and that every auto-chosen default is recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from tsdynamics import viz  # noqa: E402
from tsdynamics.analysis import fixed_points  # noqa: E402
from tsdynamics.errors import InvalidInputError, InvalidParameterError  # noqa: E402
from tsdynamics.families import ContinuousSystem  # noqa: E402
from tsdynamics.systems import Brusselator, LotkaVolterra, VanDerPol  # noqa: E402
from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402
from tsdynamics.viz.spec import PlotKind  # noqa: E402
from tsdynamics.viz.transforms import build_spec, compatibility, geometry, get  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _renderers():
    register_builtin_renderers()
    yield


class _Spatial(ContinuousSystem):
    """A 3-D flow, so the slice API has something to slice."""

    params = {"a": 1.0}
    dim = 3
    variables = ("x", "y", "z")
    default_ic = [0.4, 0.3, 0.2]

    @staticmethod
    def _equations(Y, t, *, a):  # noqa: N803, D102
        return a * Y(1), -a * Y(0), -Y(2)


#: The nine rows this work package owns.
_OWNED = (
    "nullclines",
    "direction_field",
    "flow_speed",
    "streamlines",
    "trace_determinant",
    "ftle",
    "escape_time",
    "transient_time",
    "invariant_density",
)

_WINDOW = {"xlim": (0.5, 8.0), "ylim": (0.5, 6.0)}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_every_planar_transform_is_registered_and_introspectable():
    matrix = compatibility()
    for name in _OWNED:
        assert name in matrix, f"{name} is not in ts.viz.compatibility()"
        record = get(name)
        assert record.doc, f"{name} has no one-line doc for the matrix"
        assert record.example is not None, "the compatibility gate needs an example subject"
        assert record.analysis is not None and record.analysis.startswith(
            "tsdynamics.analysis.planar"
        ), "a transform is a thin adapter and must name the estimator it adapts"


def test_the_source_categories_are_the_honest_ones():
    """Model if it evaluates the RHS somewhere new; data if a sample set is enough."""
    for name in (
        "nullclines",
        "direction_field",
        "flow_speed",
        "streamlines",
        "trace_determinant",
        "ftle",
        "escape_time",
        "transient_time",
    ):
        assert get(name).source == "model", name
    assert get("invariant_density").source == "data"


@pytest.mark.parametrize(
    "name",
    [
        "nullclines",
        "direction_field",
        "flow_speed",
        "streamlines",
        "ftle",
        "escape_time",
        "transient_time",
    ],
)
def test_a_model_transform_handed_a_trajectory_names_what_it_needs(name):
    traj = LotkaVolterra().run(final_time=1.0, dt=0.1)
    with pytest.raises(InvalidInputError, match="continuous system"):
        build_spec(traj, name, **_WINDOW)


def test_an_invalid_primitive_names_the_valid_set_and_never_falls_back():
    with pytest.raises(InvalidParameterError) as excinfo:
        build_spec(LotkaVolterra(), "nullclines", primitive="image", **_WINDOW)
    message = str(excinfo.value)
    assert "nullclines" in message and "line" in message and "points" in message


# ---------------------------------------------------------------------------
# nullclines
# ---------------------------------------------------------------------------


def test_nullclines_draw_one_labelled_curve_per_component():
    spec = build_spec(LotkaVolterra(), "nullclines", grid=101, **_WINDOW)
    assert spec.kind is PlotKind.PHASE_PORTRAIT_2D
    assert [layer.label for layer in spec.layers] == ["x' = 0", "y' = 0"]
    assert {layer.transform for layer in spec.layers} == {"nullclines"}
    assert spec.aspect == "equal"


def test_branches_of_one_nullcline_are_one_layer_separated_by_nan():
    """Several branches are one *object*: one colour, one legend entry.

    Van der Pol's ``y' = 0`` set has branches either side of ``x = +-1``.  Drawn
    as N layers the renderer gives each branch its own palette colour and its own
    legend row, which says there are N nullclines when there is one.
    """
    spec = build_spec(VanDerPol(), "nullclines", xlim=(-3.0, 3.0), ylim=(-4.0, 4.0), grid=201)
    assert len(spec.layers) == 2
    y_nullcline = spec.layers[1]
    gaps = np.isnan(y_nullcline.data["x"])
    assert gaps.sum() >= 1, "the separate branches were joined into one stroke"
    assert not np.isnan(y_nullcline.data["x"][0])
    assert not np.isnan(y_nullcline.data["x"][-1])


def test_the_drawn_nullclines_pass_through_the_drawn_equilibria():
    """The composed figure's own consistency check — and it checks both halves.

    The nullclines come from marching squares and the equilibrium markers from
    Newton's method; if the picture is right they must intersect on the marker.
    """
    system = LotkaVolterra()
    nulls = build_spec(system, "nullclines", grid=201, **_WINDOW)
    points = fixed_points(system, seed=0).to_plot_spec()
    marks = np.column_stack([points.layers[0].data["x"], points.layers[0].data["y"]])
    inside = [
        p
        for p in marks
        if _WINDOW["xlim"][0] < p[0] < _WINDOW["xlim"][1]
        and _WINDOW["ylim"][0] < p[1] < _WINDOW["ylim"][1]
    ]
    assert inside
    for point in inside:
        for layer in nulls.layers:
            x, y = layer.data["x"], layer.data["y"]
            good = np.isfinite(x) & np.isfinite(y)
            assert np.hypot(x[good] - point[0], y[good] - point[1]).min() < 0.05


def test_a_window_with_no_nullcline_in_it_raises_rather_than_drawing_a_legend():
    with pytest.raises(InvalidParameterError, match="no nullcline"):
        build_spec(LotkaVolterra(), "nullclines", xlim=(20.0, 30.0), ylim=(20.0, 30.0), grid=21)


# ---------------------------------------------------------------------------
# direction_field / flow_speed
# ---------------------------------------------------------------------------


def test_the_direction_field_draws_unit_arrows_and_the_vector_field_true_ones():
    unit = build_spec(VanDerPol(), "direction_field", xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), grid=9)
    layer = unit.layers[0]
    assert layer.kind is PlotKind.QUIVER
    lengths = np.hypot(layer.data["u"], layer.data["v"])
    # The lattice lands exactly on the equilibrium, where the field has no
    # direction to normalize: that arrow stays zero rather than being invented.
    assert np.allclose(lengths[lengths > 0], 1.0)
    assert (lengths == 0.0).sum() == 1

    true = build_spec(
        VanDerPol(),
        "direction_field",
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        grid=9,
        normalize=False,
    )
    lengths = np.hypot(true.layers[0].data["u"], true.layers[0].data["v"])
    assert lengths.max() / max(lengths.min(), 1e-12) > 5.0, "the magnitudes were flattened"
    assert true.meta["normalize"] is False


def test_colouring_a_direction_field_by_speed_keeps_the_true_magnitude():
    spec = build_spec(
        VanDerPol(),
        "direction_field",
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        grid=9,
        color_by="speed",
    )
    layer = spec.layers[0]
    lengths = np.hypot(layer.data["u"], layer.data["v"])
    assert np.allclose(lengths[lengths > 0], 1.0)
    assert layer.data["c"].max() > 1.0
    assert spec.colorbar is not None and spec.colorbar.label == "|f|"


def test_an_unknown_colour_field_is_refused_rather_than_ignored():
    with pytest.raises(InvalidParameterError, match="color_by"):
        build_spec(
            VanDerPol(),
            "direction_field",
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            grid=5,
            color_by="curvature",
        )


def test_the_slice_api_reaches_a_three_dimensional_flow():
    """The hole the pre-registry ``vector_field`` producer could not fill."""
    spec = build_spec(
        _Spatial(),
        "direction_field",
        plane=("x", "z"),
        at=[0.0, 2.0, 0.0],
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        grid=5,
        normalize=False,
    )
    assert spec.x.label == "x" and spec.y.label == "z"
    assert np.allclose(spec.layers[0].data["u"], 2.0)  # x' = a y, with y frozen at 2
    assert spec.frame is not None and spec.frame.axes == ("x", "z")
    assert spec.meta["at"] == [0.0, 2.0, 0.0]


def test_flow_speed_is_a_field_whose_minimum_is_the_equilibrium():
    system = VanDerPol()  # unique equilibrium at the origin
    spec = build_spec(system, "flow_speed", xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), grid=81)
    layer = spec.layers[0]
    assert layer.kind is PlotKind.IMAGE
    row, col = np.unravel_index(int(np.nanargmin(layer.data["z"])), layer.data["z"].shape)
    assert abs(layer.data["x"][col]) < 0.1 and abs(layer.data["y"][row]) < 0.1
    assert spec.colorbar is not None and spec.colorbar.label == "|f|"
    assert spec.clim is not None, "presentation declares autocolor"


def test_the_log_speed_field_holes_out_the_equilibrium_instead_of_faking_a_minimum():
    spec = build_spec(
        VanDerPol(), "flow_speed", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=41, log=True
    )
    values = spec.layers[0].data["z"]
    assert np.isnan(values[20, 20]), "log of zero speed must be NaN, not a bright pixel"
    assert spec.colorbar is not None and spec.colorbar.label == "log10 |f|"


# ---------------------------------------------------------------------------
# streamlines
# ---------------------------------------------------------------------------


def test_streamlines_are_one_layer_and_stay_inside_the_window():
    spec = build_spec(
        VanDerPol(), "streamlines", xlim=(-3.0, 3.0), ylim=(-4.0, 4.0), seeds=4, steps=40
    )
    assert len(spec.layers) == 1, "the whole field is one object, not one layer per seed"
    layer = spec.layers[0]
    assert layer.kind is PlotKind.LINE
    assert np.isnan(layer.data["x"]).sum() == spec.meta["n_streamlines"] - 1
    good = np.isfinite(layer.data["x"])
    assert layer.data["x"][good].min() >= -3.0 - 1e-9
    assert layer.data["x"][good].max() <= 3.0 + 1e-9


# ---------------------------------------------------------------------------
# trace_determinant
# ---------------------------------------------------------------------------


def test_the_trace_determinant_figure_draws_the_parabola_the_axes_and_the_classes():
    spec = build_spec(LotkaVolterra(), "trace_determinant", points=[[0.0, 0.0], [4.0, 2.75]])
    labels = [layer.label for layer in spec.layers]
    assert labels[:3] == ["det = tr^2 / 4", "det = 0", "tr = 0"]
    assert set(labels[3:]) == {"saddle", "centre"}
    assert spec.legend is not None

    parabola = spec.layers[0]
    assert np.allclose(parabola.data["y"], parabola.data["x"] ** 2 / 4.0)
    assert spec.frame is not None and spec.frame.space == "param2"
    assert spec.x.label == "tr J" and spec.y.label == "det J"

    saddle = next(layer for layer in spec.layers if layer.label == "saddle")
    assert saddle.data["y"][0] < 0.0, "a saddle sits below the det = 0 axis"
    centre = next(layer for layer in spec.layers if layer.label == "centre")
    assert centre.data["x"][0] == pytest.approx(0.0, abs=1e-9)
    assert centre.data["y"][0] > (centre.data["x"][0] ** 2) / 4.0, "a centre is above the parabola"


def test_the_trace_determinant_plane_is_not_a_state_space_and_refuses_to_overlay_on_one():
    """Its axes are matrix invariants, not coordinates — the frame check knows."""
    plane = build_spec(LotkaVolterra(), "trace_determinant", points=[[4.0, 2.75]])
    portrait = build_spec(LotkaVolterra(), "nullclines", grid=41, **_WINDOW)
    with pytest.raises(InvalidParameterError, match="different spaces"):
        viz.plot(portrait, plane)


# ---------------------------------------------------------------------------
# the ensemble fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ftle", "escape_time", "transient_time"])
def test_every_scalar_field_is_an_image_at_the_physical_coordinates(name):
    options = {"grid": 15, **_WINDOW}
    if name == "escape_time":
        options |= {"final_time": 3.0, "chunks": 6}
    if name == "transient_time":
        options |= {"final_time": 3.0, "chunks": 6, "tol": 5.0}
    spec = build_spec(LotkaVolterra(), name, **options)
    layer = spec.layers[0]
    assert layer.kind is PlotKind.IMAGE
    assert layer.data["z"].shape == (15, 15)
    assert layer.data["x"][0] == pytest.approx(_WINDOW["xlim"][0])
    assert layer.data["x"][-1] == pytest.approx(_WINDOW["xlim"][1])
    assert spec.colorbar is not None
    assert spec.aspect == "equal"

    fig = spec.render("matplotlib")
    images = [im for ax in fig.axes for im in ax.images]
    assert images, f"{name} produced no image artist"
    assert np.isfinite(np.asarray(images[0].get_array()).astype(float)).any()


def test_the_ftle_horizon_and_direction_are_recorded_on_the_spec():
    """The field depends on ``T``; a plot that does not say which ``T`` is unreadable."""
    spec = build_spec(LotkaVolterra(), "ftle", grid=11, time=2.5, backward=True, **_WINDOW)
    assert spec.meta["ftle_time"] == 2.5
    assert spec.meta["backward"] is True
    assert spec.colorbar is not None and spec.colorbar.label == "FTLE"


def test_an_auto_chosen_window_is_recorded_not_silent():
    """The one thing a model plot can be confidently wrong about, made auditable."""
    spec = build_spec(LotkaVolterra(), "nullclines", grid=41)
    assert "pilot orbit" in spec.meta["window_source"]
    assert spec.meta["xlim"][0] < 4.0 < spec.meta["xlim"][1]
    explicit = build_spec(LotkaVolterra(), "nullclines", grid=41, **_WINDOW)
    assert explicit.meta["window_source"] == "explicit"


def test_the_time_quantisation_of_a_first_passage_field_is_stated():
    spec = build_spec(LotkaVolterra(), "escape_time", grid=11, final_time=4.0, chunks=8, **_WINDOW)
    assert spec.meta["time_resolution"] == pytest.approx(0.5)
    assert spec.meta["escape"] == "left the drawn window (in-plane)"


# ---------------------------------------------------------------------------
# invariant_density
# ---------------------------------------------------------------------------


def _logistic_orbit(n=20_000):
    from tsdynamics.systems import Logistic

    return Logistic(r=4.0).run(steps=n, ic=[0.4])


def test_the_one_component_density_is_a_histogram_of_the_natural_measure():
    spec = build_spec(_logistic_orbit(), "invariant_density", bins=64)
    layer = spec.layers[0]
    assert layer.kind is PlotKind.HISTOGRAM
    assert layer.data["x"].size == 64
    truth = 1.0 / (np.pi * np.sqrt(np.clip(layer.data["x"] * (1 - layer.data["x"]), 1e-30, None)))
    inside = (layer.data["x"] > 0.1) & (layer.data["x"] < 0.9)
    assert np.median(np.abs(layer.data["y"][inside] - truth[inside]) / truth[inside]) < 0.1
    assert spec.frame is not None and spec.frame.ndim == 1


def test_the_two_component_density_is_the_measure_on_the_attractor():
    traj = LotkaVolterra().run(final_time=60.0, dt=0.01, ic=[4.0, 1.5])
    spec = build_spec(traj, "invariant_density", components=("x", "y"), bins=48)
    layer = spec.layers[0]
    assert layer.kind is PlotKind.IMAGE
    assert layer.data["z"].shape == (48, 48)
    assert spec.frame is not None and spec.frame.ndim == 2
    assert spec.aspect == "equal"


def test_the_row_narrows_to_what_the_geometry_can_honestly_be_drawn_as():
    """A one-component density is not an image, and a two-component one is not a bar chart."""
    orbit = _logistic_orbit(4_000)
    with pytest.raises(InvalidParameterError, match="not for \\*this\\* geometry"):
        build_spec(orbit, "invariant_density", primitive="image", bins=32)

    traj = LotkaVolterra().run(final_time=20.0, dt=0.05, ic=[4.0, 1.5])
    with pytest.raises(InvalidParameterError, match="not for \\*this\\* geometry"):
        build_spec(traj, "invariant_density", components=("x", "y"), primitive="histogram", bins=16)


def test_invariant_density_accepts_a_system_and_records_the_run():
    """A ``data`` transform accepts a system — that is what the source category means.

    This originally asserted the opposite: that handing a system to
    ``invariant_density`` should *refuse*, on the reasoning that a density from an
    arbitrary-length run is not a converged invariant measure and so would
    mislead.  The reasoning is sound; the conclusion is not the library's.

    Two things overrule it.  The declared contract is that a ``data`` transform
    accepts a system, because having the model is having the data — a category
    that thirteen of twenty-two members refused was not a category.  And the
    project has already settled this shape of question the other way: the owner
    rejected exactly this "refuse rather than return something you must interpret"
    move on ``box_counting_dimension``, which now returns its estimate flagged
    rather than raising.  Informing beats refusing.

    So it runs, and the geometry records how many samples it drew, which is the
    thing a caller needs in order to judge convergence for themselves.
    """
    spec = build_spec(LotkaVolterra(), "invariant_density")
    assert spec.layers

    geom = geometry(LotkaVolterra(), "invariant_density")
    assert geom.meta["n_samples"] > 0


def test_too_many_components_is_refused():
    traj = _Spatial().run(final_time=5.0, dt=0.05)
    with pytest.raises(InvalidParameterError, match="one component"):
        build_spec(traj, "invariant_density", components=("x", "y", "z"))


# ---------------------------------------------------------------------------
# THE PAYOFF: they compose on one set of axes
# ---------------------------------------------------------------------------


def test_the_strogatz_composite_is_one_figure_in_role_order():
    """Direction field + nullclines + orbit + equilibria, on one ``state2(x, y)`` plane.

    This is the figure the whole model-only family exists for, and it is also the
    strongest correctness check available: the nullclines are drawn from marching
    squares, the equilibria from Newton's method, and the orbit from the engine —
    three independent computations that only agree if all three are right.
    """
    system = VanDerPol()
    traj = system.run(final_time=30.0, dt=0.01, ic=[0.5, 0.5])
    window = {"xlim": (-3.0, 3.0), "ylim": (-4.0, 4.0)}
    spec = viz.plot(
        build_spec(system, "direction_field", grid=13, **window),
        build_spec(system, "nullclines", grid=201, **window),
        traj.to_plot_spec(components=["x", "y"]),
        fixed_points(system, seed=0).to_plot_spec(annotate=False),
    )
    assert spec.frame is not None
    assert spec.frame.space == "state2" and spec.frame.axes == ("x", "y")

    marks = [str(layer.kind) for layer in spec.layers]
    assert marks[0] == "quiver", "the field must be the backdrop, whatever the argument order"
    assert marks[-1] == "scatter", "the equilibria annotate everything else"

    # order-free: the same call with the arguments shuffled is the same picture
    shuffled = viz.plot(
        fixed_points(system, seed=0).to_plot_spec(annotate=False),
        traj.to_plot_spec(components=["x", "y"]),
        build_spec(system, "nullclines", grid=201, **window),
        build_spec(system, "direction_field", grid=13, **window),
    )
    assert [str(layer.kind) for layer in shuffled.layers] == marks

    fig = spec.render("matplotlib")
    axes = fig.axes[0]
    assert len(axes.lines) >= 3  # two nullclines + the orbit
    assert axes.collections, "the quiver and the equilibrium markers"


def test_a_field_drawn_on_one_plane_refuses_a_curve_drawn_on_another():
    """The wrong-plane bug, structurally impossible for these transforms too."""
    system = _Spatial()
    xy = build_spec(
        system, "direction_field", plane=("x", "y"), xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=5
    )
    xz = build_spec(
        system, "nullclines", plane=("x", "z"), xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=31
    )
    with pytest.raises(InvalidParameterError, match="axes mismatch"):
        viz.plot(xy, xz)


def test_a_scalar_field_and_a_curve_family_compose():
    """FTLE under nullclines: a field and a curve set on the same plane."""
    system = Brusselator(b=1.5)
    window = {"xlim": (0.2, 2.5), "ylim": (0.5, 3.0)}
    spec = viz.plot(
        build_spec(system, "ftle", grid=21, time=2.0, **window),
        build_spec(system, "nullclines", grid=121, **window),
    )
    assert [str(layer.kind) for layer in spec.layers] == ["image", "line", "line"]
    fig = spec.render("matplotlib")
    assert fig.axes[0].images and len(fig.axes[0].lines) == 2


# ---------------------------------------------------------------------------
# The other backends
# ---------------------------------------------------------------------------


_LINE_OPTIONS = {
    "nullclines": {"grid": 41, **_WINDOW},
    "streamlines": {"seeds": 3, "steps": 20, **_WINDOW},
    "trace_determinant": {"points": [[0.0, 0.0], [4.0, 2.75]]},
}


@pytest.mark.parametrize("name", sorted(_LINE_OPTIONS))
@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "json", "threejs"])
def test_the_line_transforms_draw_on_every_backend(name, backend, tmp_path):
    """Contours and streamlines lower to plain lines, which is why all four work.

    ``trace_determinant`` styles its two reference axes dashed, and three.js
    honors no linestyle — so it warns, loudly and by name, which is the honoring
    contract doing its job rather than a failure.
    """
    if backend == "plotly":
        pytest.importorskip("plotly")
    from tsdynamics.viz.render.caps import VisualizationDegraded

    spec = build_spec(LotkaVolterra(), name, **_LINE_OPTIONS[name])
    suffix = {"matplotlib": ".png", "plotly": ".html", "json": ".json", "threejs": ".json"}[backend]
    out = tmp_path / f"{name}{suffix}"
    degrades = backend == "threejs" and name == "trace_determinant"
    if degrades:
        with pytest.warns(VisualizationDegraded, match="linestyle"):
            spec.save(out, backend=backend)
    else:
        spec.save(out, backend=backend)
    assert out.exists() and out.stat().st_size > 0


def test_a_scalar_field_survives_a_json_round_trip_with_its_frame_and_provenance():
    from tsdynamics.viz.spec import PlotSpec

    spec = build_spec(LotkaVolterra(), "ftle", grid=9, time=1.0, **_WINDOW)
    back = PlotSpec.from_dict(spec.to_dict())
    assert back.frame == spec.frame
    assert [layer.transform for layer in back.layers] == ["ftle"]
    assert back.meta["ftle_time"] == 1.0


# ---------------------------------------------------------------------------
# The raw-arrays escape hatch
# ---------------------------------------------------------------------------


def test_geometry_hands_back_typed_channels_in_a_declared_frame():
    geom = geometry(VanDerPol(), "flow_speed", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=7)
    assert geom.frame.describe() == "state2(x, y)"
    assert sorted(geom.channels) == ["c", "x", "y", "z"]
    assert geom.channels["z"].values.shape == (7, 7)
    assert viz.draw(geom, "contour").layers, "and it goes back to the library"
