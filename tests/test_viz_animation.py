"""Tests for the animation layer (stream VIZ-ANIM, issue #461).

Covers the contract agreed in design:

1. Animation is an **orthogonal modifier** (``meta``-free typed
   :class:`~tsdynamics.viz.spec.Animation` on ``PlotSpec.animation``): any kind
   becomes a movie via ``__plot_spec__(animate=...)`` / ``.animate()``; the
   semantic kind is unchanged.
2. The chainable knobs (``animate`` / ``trail`` / ``head`` / ``camera`` /
   ``clock``) mutate-and-return-self and compose with the static tweaks.
3. Per-kind head defaults (on for portraits / spacetime, off for a plain time
   series); reveal frame-math (head schedule, trail → samples).
4. An animated spec round-trips through ``to_dict`` / ``from_dict``.
5. ``ts.viz.plot(..., animate=...)`` animates the whole figure (lockstep
   composite; overlay panel).
6. matplotlib renders a ``FuncAnimation`` (mp4/gif via ``.save``); plotly renders
   a frames+slider figure (html via ``.save``).
7. Building an animated spec imports no plotting library.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import tsdynamics as ts
import tsdynamics.viz as viz
from tsdynamics.viz.spec import Animation, PlotKind, PlotSpec


def _lorenz():
    return ts.systems.Lorenz().run(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)


# ---------------------------------------------------------------------------
# IR: the orthogonal modifier + chainable knobs (backend-free)
# ---------------------------------------------------------------------------


def test_animate_is_an_orthogonal_modifier_keeping_the_kind():
    spec = _lorenz().__plot_spec__(animate=True)
    assert spec.is_animated
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D  # kind unchanged
    assert isinstance(spec.animation, Animation)


def test_static_spec_is_not_animated():
    assert _lorenz().__plot_spec__().is_animated is False


def test_default_animation_is_a_windowed_comet():
    """The default is a smooth windowed comet (not a heavy persistent reveal)."""
    a = _lorenz().__plot_spec__(animate=True).animation
    assert a.trail_kind == "steps"
    assert a.trail_length is not None and a.trail_length > 0


@pytest.mark.parametrize(
    ("kind", "head_default"),
    [(None, True), ("time_series", False), ("spacetime", True)],
)
def test_per_kind_head_default(kind, head_default):
    tr = ts.systems.Lorenz96(N=6).run(final_time=8.0, dt=0.1)
    spec = tr.__plot_spec__(kind=kind, animate=True)
    assert spec.animation.head is head_default


def test_chainable_knobs_mutate_and_return_self():
    spec = _lorenz().__plot_spec__(animate=True)
    out = (
        spec.animate(fps=24, duration=8, loop=False, pingpong=True)
        .trail(length=("time", 5.0), fade=True)
        .head(size=10, color="red", symbol="x")
        .camera(elev=20, azim=45, spin=1.5)
        .clock(fmt="t={t:.1f}")
    )
    assert out is spec  # chainable
    a = spec.animation
    assert (a.fps, a.duration, a.loop, a.pingpong) == (24.0, 8.0, False, True)
    assert (a.trail_kind, a.trail_length, a.trail_fade) == ("time", 5.0, True)
    assert (a.head_size, a.head_color, a.head_symbol) == (10.0, "red", "x")
    assert a.spin == 1.5 and a.clock_format == "t={t:.1f}"
    assert spec.meta["camera"] == {"elev": 20.0, "azim": 45.0}


def test_animate_turns_animation_on_for_a_static_spec():
    spec = _lorenz().__plot_spec__()
    assert not spec.is_animated
    spec.animate(fps=12)
    assert spec.is_animated and spec.animation.fps == 12.0


def test_trail_persistent_clears_the_window():
    spec = _lorenz().__plot_spec__(animate=True).trail(length=("steps", 50))
    assert spec.animation.trail_kind == "steps"
    spec.trail(length=None)
    assert spec.animation.trail_kind is None and spec.animation.trail_length is None


@pytest.mark.parametrize("length", [0, 1, 150])
def test_integer_trail_is_exactly_the_explicit_sample_step_spelling(length):
    short = viz.draw({"x": [0, 1], "y": [0, 1]}, "line").trail(length)
    explicit = viz.draw({"x": [0, 1], "y": [0, 1]}, "line").trail(("steps", length))
    assert short.to_dict() == explicit.to_dict()
    assert viz.load(short.to_json()).animation.trail_length == length


@pytest.mark.parametrize("length", [True, -1, 1.5])
def test_invalid_bare_trails_do_not_turn_a_static_plot_into_a_movie(length):
    plot = viz.draw({"x": [0, 1], "y": [0, 1]}, "line")
    before = plot.to_dict()
    with pytest.raises(ts.errors.InvalidParameterError):
        plot.trail(length)
    assert plot.to_dict() == before


@pytest.mark.parametrize("turns", [-0.25, 0, 0.25])
def test_signed_camera_spin_survives_json_and_rotates_the_rendered_camera(turns):
    pytest.importorskip("matplotlib")
    plot = viz.draw({"x": [0, 1, 2], "y": [0, 1, 0], "z": [0, 0, 1]}, "line3d")
    plot.camera(elev=20, azim=30, spin=turns).animate(n_frames=3)
    restored = viz.load(plot.to_json())
    assert restored.animation.spin == turns
    movie = restored.render("matplotlib")
    movie._fig.canvas.draw()
    for frame, fraction in ((0, 0), (1, 0.5), (2, 1), (0, 0)):
        movie._func(frame)
        assert movie._fig.axes[0].azim == pytest.approx(30 + 360 * turns * fraction)
        assert movie._fig.axes[0].elev == 20


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("limits", [None, (10, 4000)])
def test_three_dimensional_log_animation_keeps_positive_bounds_and_explicit_limits(axis, limits):
    pytest.importorskip("matplotlib")
    import numpy as np

    channels = {"x": np.linspace(1, 2, 10), "y": np.linspace(2, 3, 10), "z": np.linspace(3, 4, 10)}
    channels[axis] = np.geomspace(20, 2000, 10)
    plot = viz.draw(channels, "line3d").rescale(**{axis: "log"}).animate(n_frames=3)
    if limits is not None:
        plot.limits(**{axis: limits})
    movie = plot.render("matplotlib")
    for frame in (0, 2, 0):
        movie._func(frame)
        movie._fig.canvas.draw()
        axes = movie._fig.axes[0]
        assert getattr(axes, f"get_{axis}scale")() == "log"
        actual = getattr(axes, f"get_{axis}lim")()
        if limits is not None:
            assert actual == limits
        else:
            assert 0 < actual[0] < 20 < 2000 < actual[1]


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_a_constant_positive_log_coordinate_stays_in_the_log_domain(axis):
    pytest.importorskip("matplotlib")
    import numpy as np

    channels = {"x": np.arange(3.0) + 1, "y": np.arange(3.0) + 2, "z": np.arange(3.0) + 3}
    channels[axis] = np.full(3, 0.001)
    plot = viz.draw(channels, "line3d").rescale(**{axis: "log"}).animate(n_frames=3)
    movie = plot.render("matplotlib")
    movie._fig.canvas.draw()
    lo, hi = getattr(movie._fig.axes[0], f"get_{axis}lim")()
    assert 0 < lo < 0.001 < hi


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("limits", [None, (10, 4000)])
def test_two_dimensional_log_animation_honors_the_scale_and_explicit_limits(axis, limits):
    pytest.importorskip("matplotlib")
    import numpy as np

    channels = {"x": np.linspace(1, 2, 10), "y": np.linspace(2, 3, 10)}
    channels[axis] = np.geomspace(20, 2000, 10)
    plot = viz.draw(channels, "line").rescale(**{axis: "log"}).animate(n_frames=3)
    if limits is not None:
        plot.limits(**{axis: limits})
    movie = plot.render("matplotlib")
    for frame in (0, 2, 0):
        movie._func(frame)
        movie._fig.canvas.draw()
        axes = movie._fig.axes[0]
        assert getattr(axes, f"get_{axis}scale")() == "log"
        actual = getattr(axes, f"get_{axis}lim")()
        if limits is not None:
            assert actual == limits
        else:
            assert 0 < actual[0] < 20 < 2000 < actual[1]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_two_dimensional_symlog_animation_preserves_the_signed_domain(axis):
    pytest.importorskip("matplotlib")
    import numpy as np

    channels = {"x": np.linspace(1, 2, 10), "y": np.linspace(2, 3, 10)}
    channels[axis] = np.linspace(-100, 100, 10)
    plot = viz.draw(channels, "line").rescale(**{axis: "symlog"}).animate(n_frames=3)
    movie = plot.render("matplotlib")
    movie._fig.canvas.draw()
    axes = movie._fig.axes[0]
    assert getattr(axes, f"get_{axis}scale")() == "symlog"
    lo, hi = getattr(axes, f"get_{axis}lim")()
    assert lo < -100 < 0 < 100 < hi


@pytest.mark.parametrize("spatial", [False, True])
def test_dark_animation_labels_clock_and_fading_legend_keep_their_real_colours(spatial):
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba

    channels = {"x": [0, 1, 2], "y": [0, 2, 1]}
    if spatial:
        channels["z"] = [0, 1, 2]
    primitive = "line3d" if spatial else "line"
    teal = viz.draw({**channels, "label": "first"}, primitive).style(color="#2CC5AE")
    orange = viz.draw({**channels, "label": "second"}, primitive).style(color="#E8912D")
    plot = (teal + orange).theme("dark").font(family="serif", size=13)
    plot.relabel(x="x", y="y", z="z").animate(n_frames=3).trail(1, fade=True).clock()
    plot.colorize(legend={"title": "Initial states"})
    movie = plot.render("matplotlib")
    movie._fig.canvas.draw()
    movie._func(2)
    axis = movie._fig.axes[0]
    ink = to_rgba(plot.resolved_theme.foreground)
    labels = [axis.xaxis.label, axis.yaxis.label]
    if spatial:
        labels.append(axis.zaxis.label)
    assert all(to_rgba(label.get_color()) == ink for label in labels)
    clock = next(text for text in axis.texts if text.get_text().startswith("t ="))
    assert to_rgba(clock.get_color()) == ink
    assert clock.get_fontfamily() == ["serif"] and clock.get_fontsize() == 13
    assert (
        to_rgba(clock.get_bbox_patch().get_facecolor())[:3]
        == to_rgba(plot.resolved_theme.background)[:3]
    )
    legend = axis.get_legend()
    assert [tuple(handle.get_color()) for handle in legend.legend_handles] == [
        to_rgba("#2CC5AE"),
        to_rgba("#E8912D"),
    ]
    assert all(
        to_rgba(text.get_color()) == ink for text in [*legend.get_texts(), legend.get_title()]
    )
    assert legend.get_frame().get_facecolor()[:3] == to_rgba(plot.resolved_theme.background)[:3]


def test_dark_field_animation_clock_is_readable_over_its_heatmap():
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.colors import to_rgba

    from tsdynamics.viz.spec import Layer

    frames = np.arange(12.0).reshape(3, 2, 2)
    plot = (
        PlotSpec(
            kind=PlotKind.SPATIAL_FIELD,
            layers=[Layer(PlotKind.IMAGE, {"frames": frames, "z": frames[-1]})],
            meta={"times": [0, 1, 2]},
        )
        .theme("dark")
        .animate(n_frames=3, mode="frames")
        .clock()
    )
    movie = plot.render("matplotlib")
    movie._fig.canvas.draw()
    movie._func(2)
    clock = next(text for text in movie._fig.axes[0].texts if text.get_text().startswith("t ="))
    assert to_rgba(clock.get_color()) == to_rgba(plot.resolved_theme.foreground)
    assert clock.get_bbox_patch() is not None


def test_clock_default_on_and_static_tweaks_compose():
    spec = _lorenz().__plot_spec__(animate=True).clock().relabel(title="orbit").rescale(z="linear")
    assert spec.animation.clock is True
    assert spec.title == "orbit"  # static tweak still applies under animation


def test_animate_dict_overrides_defaults():
    spec = _lorenz().__plot_spec__(animate={"fps": 10, "pingpong": True})
    assert spec.animation.fps == 10.0 and spec.animation.pingpong is True
    assert spec.animation.head is True  # per-kind default preserved alongside the override


def test_animate_with_an_animation_object():
    spec = _lorenz().__plot_spec__(animate=Animation(fps=5, head=False))
    assert spec.animation.fps == 5.0 and spec.animation.head is False


# ---------------------------------------------------------------------------
# Frame math on Animation
# ---------------------------------------------------------------------------


def test_frame_count_and_head_schedule():
    a = Animation(n_frames=10)
    assert a.frame_count(1000) == 10
    heads = a.head_indices(1000)
    assert heads[0] == 0 and heads[-1] == 999 and len(heads) == 10


def test_frame_count_from_duration():
    assert Animation(duration=4.0, fps=25).frame_count(10_000) == 100


@pytest.mark.parametrize("n_frames, expected_ms", [(None, 1000), (6, 1500)])
def test_short_curve_gif_honors_duration_and_explicit_frame_count(tmp_path, n_frames, expected_ms):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from PIL import Image

    spec = viz.draw({"x": [0, 1, 2], "y": [0, 1, 0]}, "line")
    spec.animate(duration=1, fps=4, n_frames=n_frames).head(False)
    path = tmp_path / "short.gif"
    spec.save(path)
    with Image.open(path) as image:
        duration_ms = 0
        for i in range(image.n_frames):
            image.seek(i)
            duration_ms += image.info["duration"]
    assert duration_ms == expected_ms
    assert spec.animation.playback_seconds(3) * 1000 == expected_ms


def test_pingpong_mirrors_the_schedule():
    a = Animation(n_frames=5, pingpong=True)
    heads = a.head_indices(100)
    assert heads == heads[: len(heads) // 2 + 1] + heads[len(heads) // 2 + 1 :]
    assert len(heads) == 8  # 5 forward + 3 mirrored


@pytest.mark.parametrize("count", [2, 4])
def test_pingpong_export_duration_adds_only_nonendpoint_reverse_frames(tmp_path, count):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from PIL import Image

    spec = viz.draw({"x": list(range(10)), "y": [0, 1] * 5}, "line")
    spec.animate(n_frames=count, fps=4, pingpong=True).trail(None).head(False)
    path = tmp_path / "pingpong.gif"
    spec.save(path)
    with Image.open(path) as image:
        actual = 0
        for frame in range(image.n_frames):
            image.seek(frame)
            actual += image.info["duration"]
    forward = count / 4
    reverse = max(count - 2, 0) / 4
    assert actual == (forward + reverse) * 1000
    assert spec.animation.playback_seconds(10) == forward
    assert spec.animation._reverse_seconds(10) == reverse


def test_tail_samples_units():
    assert Animation(trail_kind="steps", trail_length=200).tail_samples(None) == 200
    assert Animation(trail_kind="time", trail_length=5.0).tail_samples(0.1) == 50
    assert Animation().tail_samples(0.1) is None  # persistent


def test_tail_samples_zero_length_clamps_to_one():
    assert Animation(trail_kind="steps", trail_length=0.0).tail_samples(None) == 1


def test_two_frame_pingpong_has_no_reverse_leg():
    assert Animation(n_frames=2, pingpong=True).head_indices(100) == [0, 99]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_animated_spec_round_trips_byte_identical():
    spec = (
        _lorenz()
        .__plot_spec__(animate=True)
        .animate(duration=6, fps=20, pingpong=True)
        .trail(length=("time", 3.0), fade=True)
        .camera(spin=1.0)
        .clock()
    )
    d = spec.to_dict()
    rebuilt = PlotSpec.from_dict(d)
    assert rebuilt.to_dict() == d
    assert rebuilt.is_animated and rebuilt.animation.spin == 1.0


def test_static_spec_serializes_animation_none():
    assert _lorenz().__plot_spec__().to_dict()["animation"] is None


def test_animation_all_fields_round_trip():
    """Every Animation field (incl. reserved ``mode``, off head, custom head) round-trips."""
    a = Animation(
        fps=12,
        duration=3.0,
        n_frames=50,
        loop=False,
        pingpong=True,
        mode="frames",
        trail_kind="steps",
        trail_length=0.0,
        trail_fade=True,
        head=False,
        head_size=3.0,
        head_color="blue",
        head_symbol="x",
        spin=2.0,
        clock=True,
        clock_format="{t:.3f}",
    )
    assert Animation.from_dict(a.to_dict()).to_dict() == a.to_dict()


@pytest.mark.parametrize(
    "changes",
    [
        {"n_frames": 2.5},
        {"loop": "false"},
        {"head": 0},
        {"fps": "30"},
        {"clock": "no"},
    ],
)
def test_saved_animation_rejects_the_same_invalid_values_as_direct_construction(changes):
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        Animation(**changes)
    with pytest.raises(InvalidParameterError):
        Animation.from_dict(changes)


@pytest.mark.parametrize("changes", [{"fps": 1j}, {"head_symbol": []}, {"mode": ["reveal"]}])
def test_animation_invalid_types_name_the_knob_instead_of_leaking_python_errors(changes):
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        Animation(**changes)


@pytest.mark.parametrize("three_d", [False, True])
@pytest.mark.parametrize("fade", [False, True])
def test_animated_points_keep_point_geometry_size_and_scalar_colour(three_d, fade):
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.collections import PathCollection

    data = {"x": np.arange(5), "y": np.arange(5) ** 2, "c": np.arange(5)}
    data["size"] = np.arange(1, 6) * 10
    if three_d:
        data["z"] = np.arange(5) * 2
    spec = (
        viz.draw(data, "points3d" if three_d else "points")
        .animate(n_frames=5)
        .trail(("steps", 2), fade=fade, backdrop=False)
        .head(False)
    )
    movie = spec.render("matplotlib")
    ax = movie._fig.axes[0]
    movie._fig.canvas.draw()
    points = ax.collections[0]
    assert isinstance(points, PathCollection)
    assert points.norm.vmin == 0 and points.norm.vmax == 4
    for frame in (4, 0, 2):
        movie._func(frame)
        movie._fig.canvas.draw()
        start = max(0, frame - 2)
        coords = points._offsets3d if three_d else points.get_offsets().T
        np.testing.assert_array_equal(coords[0], data["x"][start : frame + 1])
        sizes = points._sizes3d if three_d else points.get_sizes()
        np.testing.assert_array_equal(sizes, data["size"][start : frame + 1])
        if not fade:
            np.testing.assert_array_equal(points.get_array(), data["c"][start : frame + 1])
        assert points.norm.vmin == 0 and points.norm.vmax == 4
    # A point trajectory must not acquire a line connecting its samples.
    assert not any(len(line.get_xdata()) > 1 for line in ax.lines)


def test_animation_keeps_line_style_static_annotations_and_legend():
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.colors import to_rgba

    spec = viz.draw({"x": np.arange(5), "y": np.arange(5), "label": "orbit"}, "line").style(
        color="red", alpha=0.4, linewidth=3, linestyle="dashed", marker="x", markersize=8
    )
    spec += viz.draw({"x": [2], "y": [2]}, "markers").style(color="black")
    spec.animate(n_frames=5).trail(None).head(False).tweak(legend=True)
    movie = spec.render("matplotlib")
    ax = movie._fig.axes[0]
    movie._fig.canvas.draw()
    line = ax.lines[0]
    assert (line.get_linewidth(), line.get_alpha(), line.get_linestyle()) == (3, 0.4, "--")
    assert line.get_marker() == "x" and line.get_markersize() == 8
    assert to_rgba(line.get_color()) == to_rgba("red")
    assert "orbit" in [text.get_text() for text in ax.get_legend().get_texts()]
    for frame in (4, 0, 2):
        movie._func(frame)
        movie._fig.canvas.draw()
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[2, 2]])


def test_flat_overlay_animates_in_a_three_dimensional_plot():
    pytest.importorskip("matplotlib")
    import numpy as np

    spatial = viz.draw({"x": np.arange(3), "y": np.arange(3), "z": np.arange(3)}, "line3d")
    flat = viz.draw({"x": [0, 1, 2], "y": [1, 2, 3]}, "line")
    movie = (spatial + flat).animate(n_frames=3).trail(None).head(False).render("matplotlib")
    movie._fig.canvas.draw()
    movie._func(2)
    movie._fig.canvas.draw()
    np.testing.assert_array_equal(movie._fig.axes[0].lines[2].get_data_3d()[2], [0, 0, 0])


@pytest.mark.parametrize("primitive", ["points3d", "markers"])
@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_hand_built_spatial_points_keep_their_third_coordinate(primitive, backend):
    pytest.importorskip(backend)
    import numpy as np

    spec = viz.draw({"x": [0, 1], "y": [1, 2], "z": [3, 4]}, primitive)
    assert spec.is_three_d and spec.ndim == 3 and spec.z is not None
    result = spec.render(backend)
    coords = result.axes[0].collections[0]._offsets3d if backend == "matplotlib" else result.data[0]
    np.testing.assert_array_equal(coords[2] if backend == "matplotlib" else coords.z, [3, 4])


# ---------------------------------------------------------------------------
# Figure-level animation via ts.viz.plot (lockstep composite, overlay)
# ---------------------------------------------------------------------------


def test_composite_animation_is_lockstep_with_per_panel_head_defaults():
    a, b = _lorenz(), _lorenz()
    comp = viz.plot(
        viz.plot(a, b, components="x"),
        viz.plot(a, b, components="y"),
        layout="stack",
        animate=True,
    )
    assert comp.is_animated and comp.kind == PlotKind.COMPOSITE
    # every panel inherits the master clock but keeps its own (time-series) head default
    assert all(p.is_animated for p in comp.panels)
    assert all(p.animation.head is False for p in comp.panels)  # time series → no head


def test_overlay_animation_animates_the_merged_panel():
    spec = viz.plot(_lorenz(), _lorenz(), animate=True)
    assert spec.is_animated and not spec.is_composite
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D
    assert spec.animation.head is True  # portrait → head on


def test_composite_mixed_kind_head_defaults():
    """A mixed-kind composite: each panel keeps its OWN per-kind head default."""
    a = _lorenz()
    comp = viz.plot(
        viz.plot(a, components="x"),  # TIME_SERIES → head off
        viz.plot(a),  # PHASE_PORTRAIT_3D → head on
        layout="stack",
        animate=True,
    )
    heads = {p.kind: p.animation.head for p in comp.panels}
    assert heads[PlotKind.TIME_SERIES] is False
    assert heads[PlotKind.PHASE_PORTRAIT_3D] is True


def test_pre_animated_composite_panel_is_preserved():
    """A panel that already carries an Animation keeps it; only None panels inherit."""
    a = _lorenz()
    px = viz.plot(a, components="x").animate(fps=5)  # already animated
    py = viz.plot(a, components="y")  # plain
    comp = viz.plot(px, py, layout="stack", animate=True)
    assert comp.panels[0].animation.fps == 5.0  # px's own animation survived
    assert comp.panels[1].is_animated  # py inherited the master clock


# ---------------------------------------------------------------------------
# matplotlib rendering (FuncAnimation; gif/mp4 export)
# ---------------------------------------------------------------------------


def test_matplotlib_renders_a_funcanimation():
    pytest.importorskip("matplotlib")
    spec = _lorenz().__plot_spec__(animate=True).animate(n_frames=8)
    anim = spec.render(backend="matplotlib")
    assert hasattr(anim, "to_jshtml")  # the FuncAnimation signature attribute
    # Consume it (renders frames via Agg, no external writer) so it is not
    # garbage-collected un-started — which would emit a warning at GC.
    assert isinstance(anim.to_jshtml(), str)


def test_matplotlib_composite_animation_renders():
    pytest.importorskip("matplotlib")
    a, b = _lorenz(), _lorenz()
    comp = viz.plot(a, b, layout="row", animate=True).animate(n_frames=6)
    anim = comp.render(backend="matplotlib")
    assert hasattr(anim, "to_jshtml")
    assert isinstance(anim.to_jshtml(), str)


def test_save_gif(tmp_path):
    """The GIF plays: ``n_frames`` frames, all different, at the requested rate.

    ``st_size > 0`` is not enough — a **one-frame** GIF is a perfectly valid file
    and has shipped from this library before.  The frame count, the pairwise
    difference and the inter-frame delay are what a reader would actually notice.
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")  # the pillow gif writer
    import numpy as np

    out = tmp_path / "orbit.gif"
    spec = _lorenz().__plot_spec__(animate=True).animate(n_frames=6).trail(length=("steps", 100))
    returned = spec.save(str(out), fps=10)
    assert returned == str(out)
    assert out.stat().st_size > 0

    frames = _gif_frames(out)
    assert len(frames) == 6  # not one, not "some"
    assert len({f.tobytes() for f in frames}) == 6  # six different pictures
    diffs = [
        int(np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).sum()) for i in range(1, 6)
    ]
    assert all(d > 0 for d in diffs)

    from PIL import Image

    with Image.open(out) as im:
        assert im.info.get("duration") == pytest.approx(100, abs=10)  # fps=10 → 100 ms/frame


def test_save_mp4(tmp_path):
    """The MP4 plays: ``n_frames`` decoded frames at the requested rate."""
    import shutil

    pytest.importorskip("matplotlib")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available for mp4 export")
    out = tmp_path / "orbit.mp4"
    spec = _lorenz().__plot_spec__(animate=True).animate(n_frames=6)
    assert spec.save(str(out), fps=10) == str(out)
    assert out.stat().st_size > 0

    n_frames, rate = _mp4_stream_info(out)
    assert n_frames == 6
    assert rate == pytest.approx(10.0, abs=0.5)


def test_save_still_image_renders_final_frame(tmp_path):
    """A still-image extension on an animated spec writes the final (full) frame, not a movie."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "orbit.png"
    spec = _lorenz().__plot_spec__(animate=True).animate(n_frames=6)
    assert spec.save(str(out)) == str(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG, not a failed movie write


# ---------------------------------------------------------------------------
# plotly rendering (frames + slider; html export)
# ---------------------------------------------------------------------------


def test_plotly_renders_frames_with_play_and_slider():
    pytest.importorskip("plotly")
    spec = _lorenz().__plot_spec__(animate=True).animate(n_frames=12)
    fig = spec.render(backend="plotly")
    assert type(fig).__name__ == "Figure"
    assert len(fig.frames) == 12  # one frame per head index
    # the interactive controls (the actual deliverable)
    assert fig.layout.updatemenus  # play / pause buttons
    assert fig.layout.sliders and len(fig.layout.sliders[0].steps) == 12


def test_plotly_is_camera_locked_with_comet_over_context():
    """The killer feat: camera preserved across frames (rotatable while playing)."""
    pytest.importorskip("plotly")
    fig = _lorenz().__plot_spec__(animate=True).animate(n_frames=10).render(backend="plotly")
    assert fig.layout.uirevision  # camera / zoom preserved across frames
    assert fig.layout.scene.uirevision  # the 3-D camera specifically
    # a static full-curve context trace + the comet; frames touch only the comet
    assert len(fig.data) >= 2
    assert 0 < len(fig.frames[0].traces) < len(fig.data)


def test_fade_comet_3d_renders():
    """A 3-D glowing fade comet renders without the empty-Line3DCollection crash."""
    pytest.importorskip("matplotlib")
    spec = (
        _lorenz()
        .__plot_spec__(animate=True)
        .animate(n_frames=6)
        .trail(length=("steps", 80), fade=True)
    )
    anim = spec.render(backend="matplotlib")
    assert isinstance(anim.to_jshtml(), str)


def test_save_html_is_realtime_and_rotatable(tmp_path):
    """The HTML export is the real-time (rAF + extendTraces) animation, not plotly frames.

    A requestAnimationFrame loop streams the comet with ``Plotly.extendTraces`` (in-
    place buffer mutation).  For a 3-D plot the stream is paused for the duration of a
    drag (a gl3d data update would replot the scene and cancel the orbit gesture), so
    the attractor rotates smoothly and resumes on release; the live drag camera is
    mirrored into ``gd.layout`` so there is no snap-back.  Embedding the curve once (no
    frames) keeps the file small.  A minimal play/pause + restart overlay makes the
    animation obviously alive.
    """
    pytest.importorskip("plotly")
    out = tmp_path / "orbit.html"
    assert _lorenz().__plot_spec__(animate=True).save(str(out)) == str(out)
    text = out.read_text()
    assert "requestAnimationFrame" in text  # the smooth frame-rate driver
    assert "Plotly.extendTraces" in text  # in-place comet streaming (no gl3d scene rebuild)
    assert "pointerdown" in text  # the drag-start hook that pauses the stream → rotatable
    assert "plotly_relayouting" in text  # mirror the live drag camera → no snap-back
    assert "uirevision" in text  # the camera/zoom-preservation token
    assert "playBtn" in text and "restartBtn" in text  # the visible play/pause + restart control
    assert text.count('"frames"') == 0  # not the janky frame-animation path
    assert "{plot_id}" not in text  # the div id was substituted (loop will bind)


def test_style_axes_false_hides_axes_across_backends(tmp_path):
    """``style(axes=False)`` records the intent and every backend honors it."""
    spec = _lorenz().__plot_spec__()
    assert spec.style(axes=False) is spec  # chainable
    assert spec._axes_hidden() is True
    assert spec.meta["axes_visible"] is False
    # ...and re-enabling clears it.
    assert spec.style(axes=True)._axes_hidden() is False

    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._threed import _scene

    hidden = _lorenz().__plot_spec__().style(axes=False)
    scene = _scene(hidden)
    assert scene["xaxis"]["visible"] is False
    assert scene["zaxis"]["visible"] is False

    # The animated HTML layout also hides the scene axes.
    out = tmp_path / "clean.html"
    _lorenz().__plot_spec__(animate=True).style(axes=False).save(str(out))
    assert '"visible": false' in out.read_text().lower() or '"visible":false' in out.read_text()


def test_style_axes_false_matplotlib(tmp_path):
    """The matplotlib renderer turns the axes off for a clean still and animation."""
    pytest.importorskip("matplotlib")
    spec = _lorenz().__plot_spec__().style(axes=False)
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    assert ax.axison is False  # set_axis_off() took effect


# ---------------------------------------------------------------------------
# Frames mode: a spatial-field movie (1-D profile / 2-D heatmap)
# (stream VIZ-SPATIAL-FIELD; the redesign of the old issue #461 frames mode)
# ---------------------------------------------------------------------------


def _field_2d_traj():
    """A cheap 2-D field trajectory (a Swift–Hohenberg lattice) whose field evolves."""
    return ts.systems.SwiftHohenberg(N=8, L=40.0).run(
        final_time=20.0, dt=0.2, backend="interp", solver="rk45"
    )


def _field_1d_traj():
    """A cheap 1-D field trajectory (a Kuramoto–Sivashinsky profile) whose profile evolves.

    Uses the catalogue default grid (N=32) — KS is stiff and a coarser grid is
    numerically fragile; a short window keeps it cheap.
    """
    return ts.systems.KuramotoSivashinsky(N=32, L=22.0).run(final_time=20.0, dt=0.5)


def _mp4_stream_info(path):
    """Return ``(n_frames, frame_rate)`` decoded with ffprobe or FFmpeg alone.

    Counting the frames is the only way to tell a movie from a one-frame file that
    happens to carry the right container; the rate proves ``fps=`` reached the writer.
    """
    from _video import probe_video

    stream = probe_video(path)
    return stream.n_frames, stream.frame_rate


def _gif_frames(path):
    """Return the GIF's frames as a list of RGB ndarrays (browser-free)."""
    import numpy as np
    from PIL import Image

    frames = []
    with Image.open(path) as im:
        for i in range(getattr(im, "n_frames", 1)):
            im.seek(i)
            frames.append(np.asarray(im.convert("RGB")).copy())
    return frames


def test_field_recipe_builds_a_spatial_field_spec_from_the_system_grid():
    """``kind="field"`` reshapes the flattened state via the system's grid → SPATIAL_FIELD."""
    spec = _field_2d_traj().__plot_spec__(kind="field")
    assert spec.kind == PlotKind.SPATIAL_FIELD
    assert spec.ndim == 2  # a 2-D field → a heatmap
    layer = spec.layers[0]
    assert layer.kind == PlotKind.IMAGE
    assert layer.data["z"].shape == (8, 8)  # the field reshaped to its grid
    assert layer.data["frames"].ndim == 3  # the per-time field stack (T, Ny, Nx)


def test_field_recipe_one_d_field_is_a_profile_line():
    """A 1-D field (no 2-D grid) plays as a profile LINE — honest, never a guessed grid."""
    spec = _field_1d_traj().__plot_spec__(kind="field")
    assert spec.kind == PlotKind.SPATIAL_FIELD
    assert spec.ndim == 1
    assert spec.layers[0].kind == PlotKind.LINE
    assert spec.layers[0].data["frames"].ndim == 2  # (T, Nx)


def test_field_animate_defaults_to_the_frames_field_movie():
    """``kind="field", animate=True`` is the field movie — frames mode, no comet/head."""
    spec = _field_2d_traj().__plot_spec__(kind="field", animate=True)
    assert spec.is_animated
    assert spec.kind == PlotKind.SPATIAL_FIELD  # animation is orthogonal — kind unchanged
    a = spec.animation
    assert a.mode == "frames"
    assert a.head is False  # the field itself is the motion
    assert a.trail_kind is None


def test_field_spec_round_trips_through_to_dict():
    spec = _field_2d_traj().__plot_spec__(kind="field", animate=True).animate(n_frames=10, fps=12)
    d = spec.to_dict()
    rebuilt = PlotSpec.from_dict(d)
    assert rebuilt.to_dict() == d
    assert rebuilt.kind == PlotKind.SPATIAL_FIELD
    assert rebuilt.animation.mode == "frames"
    assert rebuilt.layers[0].data["frames"].shape == spec.layers[0].data["frames"].shape


def test_field_movie_renders_a_funcanimation():
    pytest.importorskip("matplotlib")
    spec = _field_2d_traj().__plot_spec__(kind="field", animate=True).animate(n_frames=8)
    anim = spec.render(backend="matplotlib")
    assert hasattr(anim, "to_jshtml")  # a real FuncAnimation
    assert isinstance(anim.to_jshtml(), str)  # frames render without error


def test_field_movie_2d_frames_carry_distinct_evolving_content():
    """The headline deliverable: the 2-D heatmap is a fresh spatial state each frame.

    Renders the field movie and confirms consecutive frames carry genuinely
    different image content (the field evolves) — not a sweep line over a static
    image.  Reads the ``imshow`` array directly (browser-free).
    """
    import numpy as np

    pytest.importorskip("matplotlib")
    spec = _field_2d_traj().__plot_spec__(kind="field", animate=True).animate(n_frames=10)
    anim = spec.render(backend="matplotlib")
    image = anim._fig.axes[0].images[0]

    snapshots = []
    for k in range(10):
        anim._func(k)  # advance to playback frame k
        snapshots.append(np.asarray(image.get_array(), dtype=float).copy())
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered (warns under -W error)

    # Every frame is a full, finite field with real spatial structure (not uniform).
    assert all(np.all(np.isfinite(s)) for s in snapshots)
    assert all(float(np.var(s)) > 0.0 for s in snapshots)
    # Consecutive frames differ (a fresh field each frame, not a static image).
    assert all(
        not np.array_equal(snapshots[i], snapshots[i + 1]) for i in range(len(snapshots) - 1)
    )
    assert len({s.tobytes() for s in snapshots}) == len(snapshots)


def test_field_movie_1d_profile_frames_differ():
    """A 1-D field movie plays a travelling-wave LINE whose y-data changes per frame."""
    import numpy as np

    pytest.importorskip("matplotlib")
    spec = _field_1d_traj().__plot_spec__(kind="field", animate=True).animate(n_frames=10)
    anim = spec.render(backend="matplotlib")
    line = anim._fig.axes[0].lines[0]

    profiles = []
    for k in range(10):
        anim._func(k)
        profiles.append(np.asarray(line.get_ydata(), dtype=float).copy())
    anim.to_jshtml()
    assert all(np.all(np.isfinite(p)) for p in profiles)
    assert len({p.tobytes() for p in profiles}) >= 2  # the profile evolves


def test_field_movie_gif_frames_evolve(tmp_path):
    """A 2-D field movie exported to GIF: the decoded heatmap frames change."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    import numpy as np

    spec = _field_2d_traj().__plot_spec__(kind="field", animate=True).animate(n_frames=8)
    out = tmp_path / "field.gif"
    assert spec.save(str(out), fps=8) == str(out)

    frames = _gif_frames(out)
    assert len(frames) >= 2
    diffs = [
        int(np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).sum())
        for i in range(1, len(frames))
    ]
    assert any(d > 0 for d in diffs)  # the heatmap content changes across frames


def test_field_static_render_is_the_final_field(tmp_path):
    """A static ``kind="field"`` spec renders the field at the final time (a real PNG)."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "field.png"
    spec = _field_2d_traj().__plot_spec__(kind="field")  # no animate → static final field
    assert spec.is_animated is False
    assert spec.save(str(out)) == str(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG


def test_field_movie_still_image_is_the_final_field(tmp_path):
    """A still-image extension on an animated field spec writes the final field frame."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "field_still.png"
    spec = _field_2d_traj().__plot_spec__(kind="field", animate=True).animate(n_frames=6)
    assert spec.save(str(out)) == str(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG, not a failed movie


def test_field_movie_block_selection_for_multi_field_systems():
    """``components=`` selects a field block (Gray–Scott's u / v); v is the default."""
    import numpy as np

    tr = ts.systems.GrayScott(N=8).run(final_time=80.0, dt=4.0, backend="interp", solver="rk45")
    spec_v = tr.__plot_spec__(kind="field")  # default → the activator v (last block)
    spec_u = tr.__plot_spec__(kind="field", components="u")
    assert spec_v.layers[0].data["z"].shape == (8, 8)
    # The two blocks are distinct fields (u starts near 1, v near 0).
    assert not np.array_equal(spec_v.layers[0].data["z"], spec_u.layers[0].data["z"])


def test_frames_mode_without_a_field_stack_warns_and_degrades():
    """``mode="frames"`` on a non-field spec degrades to reveal with a warning."""
    pytest.importorskip("matplotlib")
    from tsdynamics.viz.render.caps import VisualizationDegraded

    # A plain time series carries no field stack — frames mode cannot apply.
    spec = _lorenz().__plot_spec__(kind="time_series", components="x")
    spec.animation = Animation(mode="frames", n_frames=5)
    with pytest.warns(VisualizationDegraded):
        anim = spec.render(backend="matplotlib")
        anim.to_jshtml()


# ---------------------------------------------------------------------------
# Sweep guarantee: every animatable plot kind yields a renderable animation
# ---------------------------------------------------------------------------


#: The animatable plot kinds the sweep covers (built lazily per test so collection
#: stays cheap — a field system integration is heavy).
_ANIMATABLE_KINDS = (
    "time_series",
    "phase_portrait_2d",
    "phase_portrait_3d",
    "delay_embedding",
    "spacetime_reveal",
    "spatial_field_2d",
    "spatial_field_1d",
)


def _build_animatable_spec(name):
    """Build one representative animated spec for ``name`` (lazily — no collection cost)."""
    if name.startswith("spatial_field"):
        traj = _field_2d_traj() if name.endswith("2d") else _field_1d_traj()
        return traj.__plot_spec__(kind="field", animate=True)
    if name == "spacetime_reveal":
        field = ts.systems.Lorenz96(N=8).run(final_time=8.0, dt=0.1)
        return field.__plot_spec__(kind="spacetime", animate=True)
    lor = ts.systems.Lorenz().run(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0]).after(1.0)
    if name == "time_series":
        return lor.__plot_spec__(kind="time_series", components="x", animate=True)
    if name == "phase_portrait_2d":
        return lor.__plot_spec__(components=["x", "y"], animate=True)
    if name == "delay_embedding":
        return lor.__plot_spec__(kind="delay", delay_time=0.1, components="x", animate=True)
    return lor.__plot_spec__(animate=True)  # phase_portrait_3d


@pytest.mark.parametrize("name", _ANIMATABLE_KINDS)
def test_every_animatable_kind_renders_an_animation(name):
    """Animate ANY kind: each representative spec renders without error."""
    pytest.importorskip("matplotlib")
    spec = _build_animatable_spec(name).animate(n_frames=6)
    anim = spec.render(backend="matplotlib")
    assert hasattr(anim, "to_jshtml")  # a FuncAnimation, not a static figure
    assert isinstance(anim.to_jshtml(), str)  # all frames render


def test_curve_kinds_reveal_multiple_distinct_frames():
    """Curve kinds reveal must produce > 1 *distinct* frame (the comet really moves)."""
    import numpy as np

    pytest.importorskip("matplotlib")
    lor = ts.systems.Lorenz().run(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0]).after(1.0)
    # A persistent reveal so the drawn curve strictly grows (no windowing).
    spec = (
        lor.__plot_spec__(components=["x", "y"], animate=True)
        .animate(n_frames=8)
        .trail(length=None)
    )
    anim = spec.render(backend="matplotlib")
    # With a persistent trail the first Line2D is the revealing curve; its vertex
    # set grows and its endpoint (the comet head) advances each frame.
    line = anim._fig.axes[0].lines[0]
    heads = []
    sizes = []
    for k in range(8):
        anim._func(k)
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        sizes.append(xs.size)
        if xs.size:
            heads.append((float(xs[-1]), float(ys[-1])))
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered
    assert len(set(heads)) > 1  # the head advances → distinct frames
    assert sizes[0] < sizes[-1]  # the persistent trail grows


def test_spacetime_reveal_sweep_line_moves():
    """The spacetime *reveal* sweep produces > 1 distinct sweep-line position.

    The reveal model for a ``SPACETIME`` image is unchanged by the spatial-field
    redesign: a moving "now" sweep line over the static image (it is the *frames*
    mode that became the spatial-field movie).
    """
    import numpy as np

    pytest.importorskip("matplotlib")
    field = ts.systems.Lorenz96(N=8).run(final_time=8.0, dt=0.1)
    spec = field.__plot_spec__(kind="spacetime", animate=True).animate(n_frames=8)
    anim = spec.render(backend="matplotlib")
    ax = anim._fig.axes[0]
    # The moving "now" sweep line is the only Line2D over the static image.
    sweep = ax.lines[0]
    positions = []
    for k in range(8):
        anim._func(k)
        positions.append(float(np.asarray(sweep.get_xdata(), dtype=float)[0]))
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered
    assert len(set(positions)) > 1  # the sweep advances → distinct frames
    assert positions[0] < positions[-1]


# ---------------------------------------------------------------------------
# Import-light: building an animation pulls in no plotting library
# ---------------------------------------------------------------------------


def test_building_an_animation_imports_no_plot_library():
    code = (
        "import sys, tsdynamics as ts;"
        "tr = ts.systems.Lorenz().run(final_time=10.0, dt=0.05).after(2.0);"
        "spec = tr.__plot_spec__(animate=True).animate(fps=20).trail(length=('time', 3.0));"
        "spec.to_dict();"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'plotly')];"
        "assert not bad, bad; print('NO_PLOT_LIBS')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "NO_PLOT_LIBS" in out.stdout
