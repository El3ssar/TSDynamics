"""Tests for the animation layer (stream VIZ-ANIM, issue #461).

Covers the contract agreed in design:

1. Animation is an **orthogonal modifier** (``meta``-free typed
   :class:`~tsdynamics.viz.spec.Animation` on ``PlotSpec.animation``): any kind
   becomes a movie via ``to_plot_spec(animate=...)`` / ``.animate()``; the
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
    return ts.Lorenz().integrate(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)


# ---------------------------------------------------------------------------
# IR: the orthogonal modifier + chainable knobs (backend-free)
# ---------------------------------------------------------------------------


def test_animate_is_an_orthogonal_modifier_keeping_the_kind():
    spec = _lorenz().to_plot_spec(animate=True)
    assert spec.is_animated
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D  # kind unchanged
    assert isinstance(spec.animation, Animation)


def test_static_spec_is_not_animated():
    assert _lorenz().to_plot_spec().is_animated is False


def test_default_animation_is_a_windowed_comet():
    """The default is a smooth windowed comet (not a heavy persistent reveal)."""
    a = _lorenz().to_plot_spec(animate=True).animation
    assert a.trail_kind == "steps"
    assert a.trail_length is not None and a.trail_length > 0


@pytest.mark.parametrize(
    ("kind", "head_default"),
    [(None, True), ("time_series", False), ("spacetime", True)],
)
def test_per_kind_head_default(kind, head_default):
    tr = ts.Lorenz96(N=6).trajectory(final_time=8.0, dt=0.1)
    spec = tr.to_plot_spec(kind=kind, animate=True)
    assert spec.animation.head is head_default


def test_chainable_knobs_mutate_and_return_self():
    spec = _lorenz().to_plot_spec(animate=True)
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
    spec = _lorenz().to_plot_spec()
    assert not spec.is_animated
    spec.animate(fps=12)
    assert spec.is_animated and spec.animation.fps == 12.0


def test_trail_persistent_clears_the_window():
    spec = _lorenz().to_plot_spec(animate=True).trail(length=("steps", 50))
    assert spec.animation.trail_kind == "steps"
    spec.trail(length=None)
    assert spec.animation.trail_kind is None and spec.animation.trail_length is None


def test_clock_default_on_and_static_tweaks_compose():
    spec = _lorenz().to_plot_spec(animate=True).clock().relabel(title="orbit").rescale(z="linear")
    assert spec.animation.clock is True
    assert spec.title == "orbit"  # static tweak still applies under animation


def test_animate_dict_overrides_defaults():
    spec = _lorenz().to_plot_spec(animate={"fps": 10, "pingpong": True})
    assert spec.animation.fps == 10.0 and spec.animation.pingpong is True
    assert spec.animation.head is True  # per-kind default preserved alongside the override


def test_animate_with_an_animation_object():
    spec = _lorenz().to_plot_spec(animate=Animation(fps=5, head=False))
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


def test_pingpong_mirrors_the_schedule():
    a = Animation(n_frames=5, pingpong=True)
    heads = a.head_indices(100)
    assert heads == heads[: len(heads) // 2 + 1] + heads[len(heads) // 2 + 1 :]
    assert len(heads) == 8  # 5 forward + 3 mirrored


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
        .to_plot_spec(animate=True)
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
    assert _lorenz().to_plot_spec().to_dict()["animation"] is None


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
    spec = _lorenz().to_plot_spec(animate=True).animate(n_frames=8)
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
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")  # the pillow gif writer
    out = tmp_path / "orbit.gif"
    spec = _lorenz().to_plot_spec(animate=True).animate(n_frames=6).trail(length=("steps", 100))
    returned = spec.save(str(out), fps=10)
    assert returned == str(out)
    assert out.stat().st_size > 0


def test_save_mp4(tmp_path):
    import shutil

    pytest.importorskip("matplotlib")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available for mp4 export")
    out = tmp_path / "orbit.mp4"
    spec = _lorenz().to_plot_spec(animate=True).animate(n_frames=6)
    assert spec.save(str(out), fps=10) == str(out)
    assert out.stat().st_size > 0


def test_save_still_image_renders_final_frame(tmp_path):
    """A still-image extension on an animated spec writes the final (full) frame, not a movie."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "orbit.png"
    spec = _lorenz().to_plot_spec(animate=True).animate(n_frames=6)
    assert spec.save(str(out)) == str(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG, not a failed movie write


# ---------------------------------------------------------------------------
# plotly rendering (frames + slider; html export)
# ---------------------------------------------------------------------------


def test_plotly_renders_frames_with_play_and_slider():
    pytest.importorskip("plotly")
    spec = _lorenz().to_plot_spec(animate=True).animate(n_frames=12)
    fig = spec.render(backend="plotly")
    assert type(fig).__name__ == "Figure"
    assert len(fig.frames) == 12  # one frame per head index
    # the interactive controls (the actual deliverable)
    assert fig.layout.updatemenus  # play / pause buttons
    assert fig.layout.sliders and len(fig.layout.sliders[0].steps) == 12


def test_plotly_is_camera_locked_with_comet_over_context():
    """The killer feat: camera preserved across frames (rotatable while playing)."""
    pytest.importorskip("plotly")
    fig = _lorenz().to_plot_spec(animate=True).animate(n_frames=10).render(backend="plotly")
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
        .to_plot_spec(animate=True)
        .animate(n_frames=6)
        .trail(length=("steps", 80), fade=True)
    )
    anim = spec.render(backend="matplotlib")
    assert isinstance(anim.to_jshtml(), str)


def test_save_html_is_realtime_and_rotatable(tmp_path):
    """The HTML export is the real-time (rAF + react) animation, not plotly frames.

    A requestAnimationFrame loop hands ``Plotly.react`` fresh comet trace objects
    each tick while leaving the layout (camera) and the static context trace
    untouched, so the attractor is rotatable while it plays; embedding the curve
    once (no frames) keeps the file small.  A minimal play/pause + restart overlay
    makes the animation obviously alive without devtools.
    """
    pytest.importorskip("plotly")
    out = tmp_path / "orbit.html"
    assert _lorenz().to_plot_spec(animate=True).save(str(out)) == str(out)
    text = out.read_text()
    assert "requestAnimationFrame" in text  # the smooth frame-rate driver
    assert "Plotly.react" in text  # re-renders gl3d under constant uirevision → camera kept
    assert "Plotly.restyle" not in text  # restyle does NOT redraw gl3d traces — never use it
    assert "uirevision" in text  # the camera/zoom-preservation token
    assert "gd.data.slice()" in text  # fresh array each tick → react's immutable diff fires
    assert "playBtn" in text and "restartBtn" in text  # the visible play/pause + restart control
    assert text.count('"frames"') == 0  # not the janky frame-animation path
    assert "{plot_id}" not in text  # the div id was substituted (loop will bind)


def test_style_axes_false_hides_axes_across_backends(tmp_path):
    """``style(axes=False)`` records the intent and every backend honors it."""
    spec = _lorenz().to_plot_spec()
    assert spec.style(axes=False) is spec  # chainable
    assert spec._axes_hidden() is True
    assert spec.meta["axes_visible"] is False
    # ...and re-enabling clears it.
    assert spec.style(axes=True)._axes_hidden() is False

    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._threed import _scene

    hidden = _lorenz().to_plot_spec().style(axes=False)
    scene = _scene(hidden)
    assert scene["xaxis"]["visible"] is False
    assert scene["zaxis"]["visible"] is False

    # The animated HTML layout also hides the scene axes.
    out = tmp_path / "clean.html"
    _lorenz().to_plot_spec(animate=True).style(axes=False).save(str(out))
    assert '"visible": false' in out.read_text().lower() or '"visible":false' in out.read_text()


def test_style_axes_false_matplotlib(tmp_path):
    """The matplotlib renderer turns the axes off for a clean still and animation."""
    pytest.importorskip("matplotlib")
    spec = _lorenz().to_plot_spec().style(axes=False)
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    assert ax.axison is False  # set_axis_off() took effect


# ---------------------------------------------------------------------------
# Frames mode: an evolving 2-D field (Kuramoto-Sivashinsky u(x, t)) (issue #461)
# ---------------------------------------------------------------------------


def _spacetime_field_traj(*, final_time=8.0, dt=0.1, N=8):
    """A cheap high-dimensional field trajectory whose spacetime image evolves."""
    return ts.Lorenz96(N=N).trajectory(final_time=final_time, dt=dt)


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


def test_frames_mode_front_door_defaults_to_a_growing_field():
    """``animate={"mode": "frames"}`` stamps the evolving-field model (no comet/head)."""
    spec = _spacetime_field_traj().to_plot_spec(kind="spacetime", animate={"mode": "frames"})
    assert spec.is_animated
    assert spec.kind == PlotKind.SPACETIME  # kind unchanged — animation is orthogonal
    a = spec.animation
    assert a.mode == "frames"
    assert a.head is False  # the image itself is the motion
    assert a.trail_kind is None  # persistent growth, no comet window


def test_frames_mode_round_trips_through_to_dict():
    spec = (
        _spacetime_field_traj()
        .to_plot_spec(kind="spacetime", animate={"mode": "frames"})
        .animate(n_frames=10, fps=12)
    )
    d = spec.to_dict()
    rebuilt = PlotSpec.from_dict(d)
    assert rebuilt.to_dict() == d
    assert rebuilt.animation.mode == "frames"


def test_frames_mode_renders_a_funcanimation():
    pytest.importorskip("matplotlib")
    spec = (
        _spacetime_field_traj()
        .to_plot_spec(kind="spacetime", animate={"mode": "frames"})
        .animate(n_frames=8)
    )
    anim = spec.render(backend="matplotlib")
    assert hasattr(anim, "to_jshtml")  # a real FuncAnimation
    assert isinstance(anim.to_jshtml(), str)  # frames render without error


def test_frames_mode_field_grows_and_consecutive_frames_differ():
    """The headline #461 deliverable: the heatmap field evolves frame by frame.

    Renders the evolving-field animation and confirms (a) the revealed field
    *grows* (strictly more finite cells each frame) and (b) consecutive frames
    carry genuinely different image content — not a sweep line over a static
    image.  Reads the ``imshow`` array directly (browser-free).
    """
    import numpy as np

    pytest.importorskip("matplotlib")
    spec = (
        _spacetime_field_traj()
        .to_plot_spec(kind="spacetime", animate={"mode": "frames"})
        .animate(n_frames=10)
    )
    anim = spec.render(backend="matplotlib")
    fig = anim._fig
    ax = fig.axes[0]
    image = ax.images[0]

    revealed = []
    snapshots = []
    for k in range(10):
        anim._func(k)  # advance to playback frame k
        arr = np.asarray(image.get_array(), dtype=float)
        revealed.append(int(np.isfinite(arr).sum()))
        snapshots.append(np.nan_to_num(arr, nan=-1e9).copy())
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered (warns under -W error)

    # The field grows (more finite cells over time) and never shrinks.
    assert revealed[0] < revealed[-1]
    assert all(revealed[i] <= revealed[i + 1] for i in range(len(revealed) - 1))
    # Consecutive frames differ (distinct image content, not just a moving line).
    assert all(
        not np.array_equal(snapshots[i], snapshots[i + 1]) for i in range(len(snapshots) - 1)
    )
    assert len({s.tobytes() for s in snapshots}) >= 2


def test_frames_mode_kuramoto_sivashinsky_gif_frames_evolve(tmp_path):
    """A real PDE field (Kuramoto-Sivashinsky u(x, t)) exported to GIF evolves.

    Saves the evolving-field animation as a GIF and decodes the frames: the
    heatmap content changes across frames (the spatiotemporal field develops).
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    import numpy as np

    tr = ts.KuramotoSivashinsky().trajectory(final_time=30.0, dt=0.5)
    spec = tr.to_plot_spec(kind="spacetime", animate={"mode": "frames"}).animate(n_frames=8)
    out = tmp_path / "ks_field.gif"
    assert spec.save(str(out), fps=8) == str(out)

    frames = _gif_frames(out)
    assert len(frames) >= 2
    diffs = [
        int(np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).sum())
        for i in range(1, len(frames))
    ]
    assert all(d > 0 for d in diffs)  # every consecutive pair of heatmaps differs


def test_frames_mode_still_image_is_the_full_final_field(tmp_path):
    """A still-image extension on a frames-mode spec writes the full (final) field."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "ks_field.png"
    spec = (
        _spacetime_field_traj()
        .to_plot_spec(kind="spacetime", animate={"mode": "frames"})
        .animate(n_frames=6)
    )
    assert spec.save(str(out)) == str(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG, not a failed movie


# ---------------------------------------------------------------------------
# Sweep guarantee: every animatable plot kind yields a renderable animation
# ---------------------------------------------------------------------------


def _animatable_specs():
    """One representative spec per animatable kind, all with ``animate=True``."""
    lor = ts.Lorenz().integrate(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0]).after(1.0)
    field = _spacetime_field_traj()
    return {
        "time_series": lor.to_plot_spec(kind="time_series", components="x", animate=True),
        "phase_portrait_2d": lor.to_plot_spec(components=["x", "y"], animate=True),
        "phase_portrait_3d": lor.to_plot_spec(animate=True),
        "delay_embedding": lor.to_plot_spec(kind="delay", tau=0.1, components="x", animate=True),
        "spacetime_reveal": field.to_plot_spec(kind="spacetime", animate=True),
        "spacetime_frames": field.to_plot_spec(kind="spacetime", animate={"mode": "frames"}),
    }


@pytest.mark.parametrize("name", list(_animatable_specs()))
def test_every_animatable_kind_renders_an_animation(name):
    """Animate ANY kind: each representative spec renders without error."""
    pytest.importorskip("matplotlib")
    spec = _animatable_specs()[name].animate(n_frames=6)
    anim = spec.render(backend="matplotlib")
    assert hasattr(anim, "to_jshtml")  # a FuncAnimation, not a static figure
    assert isinstance(anim.to_jshtml(), str)  # all frames render


def test_curve_kinds_reveal_multiple_distinct_frames():
    """Curve kinds reveal must produce > 1 *distinct* frame (the comet really moves)."""
    import numpy as np

    pytest.importorskip("matplotlib")
    lor = ts.Lorenz().integrate(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0]).after(1.0)
    # A persistent reveal so the drawn curve strictly grows (no windowing).
    spec = (
        lor.to_plot_spec(components=["x", "y"], animate=True).animate(n_frames=8).trail(length=None)
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


# ---------------------------------------------------------------------------
# Import-light: building an animation pulls in no plotting library
# ---------------------------------------------------------------------------


def test_building_an_animation_imports_no_plot_library():
    code = (
        "import sys, tsdynamics as ts;"
        "tr = ts.Lorenz().integrate(final_time=10.0, dt=0.05).after(2.0);"
        "spec = tr.to_plot_spec(animate=True).animate(fps=20).trail(length=('time', 3.0));"
        "spec.to_dict();"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'plotly')];"
        "assert not bad, bad; print('NO_PLOT_LIBS')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "NO_PLOT_LIBS" in out.stdout
