"""``layout="frames"`` — the parameter-sweep movie (contract §6.6 / M9).

The fifth composition mode.  Panels are already the composition unit and already
nest, so a movie is a composite whose panels are consecutive **in time** rather
than in space::

    ts.plot(*[ts.plot(logistic.with_params(r=r), "cobweb")
              for r in np.linspace(2.8, 4.0, 60)],
            layout="frames", fps=15).save("cascade.mp4")

Scope in v6 is matplotlib (mp4 / gif); plotly and three.js **decline** an animated
composite and say what they dropped.

What these tests pin, all of it user-facing:

1. The panels are played, not tiled — ``N`` panels give ``N`` genuinely different
   frames (a movie that renders one frame N times is the defect this file exists
   to catch: a valid one-frame GIF has shipped from this library before).
2. The view does not jump: axis ranges are the union over the panels, so what
   moves in the movie is the dynamics and not the axes.
3. ``share_color=True`` unifies the colour scale for the same reason.
4. ``n_frames`` / ``pingpong`` index the **panel list**, exactly as they index a
   sample axis everywhere else.
5. The other backends refuse out loud, never silently.

.. note::
   The contract splits this feature across slots: ``Layout.mode``'s literal
   (``viz/spec.py``) and the ``layout="frames"`` spelling at the ``ts.plot`` door
   (``viz/compose.py``) are other slots' files, so these tests build the composite
   and then set ``Layout(mode="frames")`` directly — which is exactly what the
   front door will do once it accepts the string.  ``test_the_front_door_spelling``
   records the remaining half and is expected to fail until it lands.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError
from tsdynamics.viz.render.caps import VisualizationDegraded
from tsdynamics.viz.spec import Layout


def _sweep(n=5, transform="cobweb", param="r", lo=2.8, hi=4.0):
    """``n`` single-panel plots of one system across a parameter — the movie's frames."""
    logistic = ts.systems.Logistic()
    return [
        ts.plot(logistic.with_params(**{param: float(r)}), transform, title=f"{param} = {r:.2f}")
        for r in np.linspace(lo, hi, n)
    ]


def _movie(panels, *, fps=6, share_color=False, **anim_kw):
    """Build the ``layout="frames"`` composite the front door will build."""
    tiled = ts.plot(*panels, layout="row")
    spec = dataclasses.replace(tiled, layout=Layout(mode="frames", share_color=share_color))
    return spec.animate(fps=fps, **anim_kw)


def _gif_frames(path):
    """Return a GIF's frames as RGB ndarrays (browser-free)."""
    from PIL import Image

    frames = []
    with Image.open(path) as im:
        for i in range(getattr(im, "n_frames", 1)):
            im.seek(i)
            frames.append(np.asarray(im.convert("RGB")).copy())
    return frames


# ---------------------------------------------------------------------------
# 1. the panels are PLAYED, and every frame is different
# ---------------------------------------------------------------------------


def test_one_frame_per_panel_not_one_tiled_figure():
    """A 5-panel ``frames`` composite renders a 5-frame animation, not a 1x5 grid."""
    pytest.importorskip("matplotlib")
    movie = _movie(_sweep(5))
    anim = movie.render(backend="matplotlib")

    assert type(anim).__name__ == "FuncAnimation"
    assert anim._save_count == 5  # one frame per panel
    assert len(anim._fig.axes) == 1  # one axes, re-drawn — NOT five tiled axes
    assert isinstance(anim.to_jshtml(), str)  # every frame renders (and consumes the anim)


def test_saved_gif_carries_one_distinct_frame_per_panel(tmp_path):
    """The decoded GIF has N frames and **no two are identical** — it genuinely plays.

    Asserting only ``st_size > 0`` would pass for a one-frame GIF, which is the
    exact failure this library has shipped before.
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    out = tmp_path / "cascade.gif"
    _movie(_sweep(5)).save(str(out), fps=5)

    frames = _gif_frames(out)
    assert out.stat().st_size > 0
    assert len(frames) == 5
    assert len({f.tobytes() for f in frames}) == 5  # five different pictures
    diffs = [
        int(np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).sum()) for i in range(1, 5)
    ]
    assert all(d > 0 for d in diffs)


def test_a_still_of_a_movie_is_its_final_frame(tmp_path):
    """``movie.save("x.png")`` writes the last panel — never a blank page.

    A still save writes the animation's *underlying figure*; if that figure is only
    populated once the frame loop runs, the PNG is empty and nothing says so.
    """
    pytest.importorskip("matplotlib")
    panels = _sweep(4)
    anim = _movie(panels).render(backend="matplotlib")
    assert anim._fig.axes[0].get_title() == panels[-1].title
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered (warns under -W error)

    out = tmp_path / "cascade.png"
    _movie(panels).save(str(out))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 5_000  # a real picture, not an empty canvas


def test_a_frame_is_the_panel_it_came_from():
    """Frame ``k`` draws panel ``k`` — title and all — so the movie shows the sweep."""
    pytest.importorskip("matplotlib")
    panels = _sweep(4)
    anim = _movie(panels).render(backend="matplotlib")
    fig = anim._fig

    seen = []
    for k in range(4):
        anim._func(k)
        seen.append(fig.axes[0].get_title())
    assert seen == [p.title for p in panels]
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered


# ---------------------------------------------------------------------------
# 2 + 3. the view does not jump; share_color unifies the colour meaning
# ---------------------------------------------------------------------------


def test_the_axes_do_not_jump_between_frames():
    """Every frame is drawn on the union range, so the motion is the data, not the axes."""
    pytest.importorskip("matplotlib")
    panels = _sweep(4)
    anim = _movie(panels).render(backend="matplotlib")
    ax = anim._fig.axes[0]

    limits = []
    for k in range(4):
        anim._func(k)
        limits.append((ax.get_xlim(), ax.get_ylim()))
    assert len(set(limits)) == 1, f"the view moved between frames: {limits}"

    # ...and the one range covers every panel's data.
    xs = np.concatenate(
        [layer.data["x"] for p in panels for layer in p.layers if "x" in layer.data]
    )
    (x0, x1), _ = limits[0]
    assert x0 <= float(xs.min()) and x1 >= float(xs.max())
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered


def _frame_clims(movie, n):
    """The colour limits actually painted on each of the first ``n`` frames."""
    anim = movie.render(backend="matplotlib")
    out = []
    for k in range(n):
        anim._func(k)
        ax = anim._fig.axes[0]
        out.append([c.get_clim() for c in ax.collections if hasattr(c, "get_clim")])
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered
    return out


def test_share_color_unifies_the_colour_scale_across_frames():
    """``share_color=True`` gives every frame one colour meaning — otherwise nothing compares.

    Two orbits coloured by time over different windows: unshared, frame 0 paints
    ``t in [0, 4]`` and frame 1 ``t in [2, 4]``, so the same hue means two different
    times and the "comparison" is a wrong answer.  Shared, both frames paint the
    union.
    """
    pytest.importorskip("matplotlib")
    traj = ts.systems.Lorenz().run(final_time=4.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    panels = [
        ts.plot(traj, "phase_portrait", components=["x", "z"], color_by="time"),
        ts.plot(traj.after(2.0), "phase_portrait", components=["x", "z"], color_by="time"),
    ]
    assert panels[0].clim != panels[1].clim  # the panels genuinely disagree

    unshared = _frame_clims(_movie(panels, share_color=False), 2)
    assert unshared[0] != unshared[1]

    shared = _frame_clims(_movie(panels, share_color=True), 2)
    assert shared[0] == shared[1] != [], shared
    lo = min(c[0] for c in (panels[0].clim, panels[1].clim))
    hi = max(c[1] for c in (panels[0].clim, panels[1].clim))
    assert shared[0][0] == (lo, hi)  # ...and the one scale is the union


# ---------------------------------------------------------------------------
# 4. n_frames / pingpong index the PANEL list
# ---------------------------------------------------------------------------


def test_n_frames_resamples_the_panel_list():
    """``n_frames`` means the same thing here as anywhere: how many frames to play."""
    pytest.importorskip("matplotlib")
    anim = _movie(_sweep(6), n_frames=3).render(backend="matplotlib")
    assert anim._save_count == 3
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered


def test_pingpong_plays_the_sweep_back():
    """``pingpong`` mirrors the panel schedule — a cascade that runs forward then back."""
    pytest.importorskip("matplotlib")
    movie = _movie(_sweep(4)).animate(pingpong=True)
    anim = movie.render(backend="matplotlib")
    assert anim._save_count == 6  # 4 forward + 2 back (endpoints not repeated)

    titles = []
    for k in range(6):
        anim._func(k)
        titles.append(anim._fig.axes[0].get_title())
    assert titles == titles[:4] + titles[2::-1][:2]
    anim.to_jshtml()  # consume so the animation is not GC'd un-rendered


# ---------------------------------------------------------------------------
# 5. the other backends refuse OUT LOUD
# ---------------------------------------------------------------------------


def test_plotly_declines_and_says_so():
    """plotly cannot animate a composite; it falls back to matplotlib with one warning."""
    pytest.importorskip("plotly")
    pytest.importorskip("matplotlib")
    movie = _movie(_sweep(3))
    with pytest.warns(VisualizationDegraded, match="composite"):
        result = movie.render(backend="plotly")
    assert type(result).__name__ == "FuncAnimation"
    result.to_jshtml()  # consume so the animation is not GC'd un-rendered


def test_plotly_html_refuses_a_movie_with_the_way_out():
    """``.save('.html')`` on a movie names the formats that DO write it."""
    pytest.importorskip("plotly")
    movie = _movie(_sweep(3))
    with pytest.raises(InvalidParameterError, match=r"\.mp4"):
        movie.save("cascade.html")


def test_threejs_exports_the_last_panel_and_names_what_it_dropped(tmp_path):
    """three.js reveals one draw range in one scene, so a movie becomes its last still."""
    pytest.importorskip("matplotlib")
    panels = _sweep(3, transform="cobweb")
    movie = _movie(panels)

    with pytest.warns(VisualizationDegraded, match="frames of a movie"):
        payload = movie.render(backend="threejs").payload

    assert payload["kind"] != "composite"  # not tiled — one still
    assert "animation" not in payload["metadata"]  # and honestly static
    assert payload["geometries"]  # ...but it is a real export, not an empty one


# ---------------------------------------------------------------------------
# the remaining half of the feature (another slot's files)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="contract §9.5 C4 + S5: Layout.mode's literal and compose.py's layout= "
    "vocabulary are other slots' files; the renderer half is landed and tested above",
    strict=False,
)
def test_the_front_door_spelling():
    """``ts.plot(*panels, layout="frames", fps=15)`` — the spelling §6.6 advertises."""
    movie = ts.plot(*_sweep(3), layout="frames", fps=15)
    assert movie.is_animated
    assert movie.layout is not None and movie.layout.mode == "frames"
