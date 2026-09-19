"""Writing a movie blits — and the blit is byte-identical (stream VIZ-ANIM-FAST).

The frame drivers always mutated artists rather than rebuilding them, but
*writing* a frame still cost a whole figure: matplotlib's ``Animation.save``
draws once through ``draw_idle`` and discards it, ``print_figure`` draws again to
settle the layout engine, and ``savefig`` draws a third time — each re-solving
constrained layout and re-measuring every tick label.  Measured on a default
360-frame Lorenz comet: **34.3 s** to write an ``.mp4``.

:class:`tsdynamics.viz.render.mpl._anim._FrameCompositor` caches the static
prefix of the figure as one raster and re-draws only the artists the drivers
touch, which takes the same movie to **1.06 s**.  That is only allowed to be a
speed change, so this file pins the two halves:

1. **answer preservation** — the frames a movie writer receives are *byte-for-byte*
   what the unoptimised path produced, on a 2-D comet, a 3-D comet, a lockstep
   composite and a spatial-field movie, through both writer paths (the
   transparency-supporting one and the facecolor-overriding one ``.mp4`` uses);
2. **the saving is structural, not incidental** — the number of whole-figure draws
   during a save does not grow with the frame count, the drivers declare every
   artist they mutate (a driver that forgets would freeze that artist at its first
   frame), and each driver is a **pure function of the frame index** (the
   compositor restores the frame it calibrated on, and ``pingpong`` replays frame
   0 every cycle);
3. **the optimisation applies**, not merely "falls back safely" — every eligible
   kind is asserted to reach ``blitting is True``, because a correct movie drawn
   the slow way is indistinguishable from a fast one by looking at it.

Plus the exemptions: a spinning 3-D camera and a ``layout="frames"`` composite
genuinely re-draw the axes, so the compositor is never armed for them — though the
spinning one still gets the layout freeze, and must, since its axes never settles
on its own.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import tsdynamics as ts

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.animation import AbstractMovieWriter, FuncAnimation  # noqa: E402

from tsdynamics.viz.render.mpl import _anim  # noqa: E402

#: Small enough that a frame buffer is ~240 kB, big enough to carry real ticks.
FIGSIZE = (3.0, 2.0)


def _dispose(fig) -> None:
    """Drop a figure's artists.

    ``pyplot.close`` is not the tool here: this renderer never touches pyplot, so
    its figures are not in pyplot's registry and closing does nothing.
    """
    fig.clear()


def _dispose_anim(anim) -> None:
    """Drop an animation that was inspected rather than written.

    ``Animation.__del__`` warns when nothing was ever drawn, and this suite runs
    under ``filterwarnings = error``, so a test that only looks at the built
    figure would fail a *later*, unrelated test when the garbage collector got
    round to it.  The flag is matplotlib's own.
    """
    anim._draw_was_started = True
    _dispose(anim._fig)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _BufferWriter(AbstractMovieWriter):
    """Writer protocol that keeps every grabbed frame as raw RGBA bytes.

    Goes through the real ``fig.savefig`` the movie writers use, so it exercises
    exactly the path ``PillowWriter`` / ``FFMpegWriter`` take — but keeps the
    result lossless and in memory, which a ``.gif`` (palettised) would not.
    """

    def __init__(self, fps: float = 5.0, transparency: bool = True) -> None:
        super().__init__(fps=fps)
        self._transparency = transparency
        self.frames: list[bytes] = []

    def _supports_transparency(self) -> bool:
        return self._transparency

    def setup(self, fig, outfile, dpi=None):  # noqa: D102, ANN001, ANN201
        super().setup(fig, outfile, dpi=dpi)
        self.frames = []

    def grab_frame(self, **savefig_kwargs):  # noqa: D102, ANN003, ANN201
        import io

        buf = io.BytesIO()
        self.fig.savefig(buf, format="rgba", dpi=self.dpi, **savefig_kwargs)
        self.frames.append(buf.getvalue())

    def finish(self) -> None:  # noqa: D102
        pass


def _lorenz(final_time: float = 3.0):
    """A short, pinned Lorenz orbit (pinned: a random IC is a different movie)."""
    return ts.systems.Lorenz().run(final_time=final_time, dt=0.02, ic=[1.0, 1.0, 1.0])


def _spec(kind: str, n_frames: int = 5):
    """One animated spec per animation kind under test."""
    if kind == "2d":
        return ts.plot(_lorenz(), components=("x", "z"), animate={"n_frames": n_frames})
    if kind == "3d":
        return ts.plot(_lorenz(), animate={"n_frames": n_frames})
    if kind == "series":
        return ts.plot(_lorenz(), components="x", animate={"n_frames": n_frames})
    if kind == "composite":
        traj = _lorenz()
        return ts.viz.plot(
            ts.plot(traj, components=("x", "z")),
            ts.plot(traj, components="x"),
            layout="row",
            animate={"n_frames": n_frames},
        )
    if kind == "field2d":
        traj = ts.systems.SwiftHohenberg(N=16).run(final_time=1.0, dt=0.1)
        return ts.plot(traj, "spatial_field", animate={"n_frames": n_frames})
    if kind == "field1d":
        traj = ts.systems.KuramotoSivashinsky(N=24).run(final_time=2.0, dt=0.2)
        return ts.plot(traj, "spatial_field", animate={"n_frames": n_frames})
    if kind == "fade3d":
        return ts.plot(_lorenz(), animate={"n_frames": n_frames}).trail(
            length=("steps", 40), fade=True
        )
    if kind == "fade2d":
        return ts.plot(_lorenz(), components=("x", "z"), animate={"n_frames": n_frames}).trail(
            length=("steps", 40), fade=True
        )
    if kind == "annotated":
        return (
            ts.plot(_lorenz(), components="x", animate={"n_frames": n_frames})
            .vline(1.0, label="onset")
            .hline(0.0)
            .text(1.5, 5.0, "HERE")
        )
    raise AssertionError(kind)  # pragma: no cover


def _record(spec, *, blitting: bool, transparency: bool = True) -> list[bytes]:
    """Write ``spec`` through the writer protocol and return the raw frames.

    ``blitting=False`` reproduces the pre-change path exactly: the compositor
    disabled *and* matplotlib's own ``_post_draw`` (the discarded full draw per
    frame) restored.

    Both paths are handed a figure whose layout engine has already converged.
    Constrained layout is iterative — on a 3-D axes it takes five draws to settle
    — so an un-converged figure has each path drawing its opening frames at a
    *different* framing, which is a pre-existing wobble in the unoptimised
    renderer and not a property of the composite.  Settling first is what makes
    the comparison about the change under test.
    """
    anim = _anim.render_animation(spec, figsize=FIGSIZE)
    if not blitting:
        anim._tsd_compositor = None
        anim._post_draw = FuncAnimation._post_draw.__get__(anim)
    for _ in range(8):
        anim._fig.canvas.draw()
    writer = _BufferWriter(transparency=transparency)
    anim.save("unused", writer=writer)
    _dispose(anim._fig)
    return writer.frames


def _worst_pixel_difference(a: list[bytes], b: list[bytes]) -> int:
    """Max per-channel difference over matching frames (``255`` on a shape mismatch)."""
    assert len(a) == len(b) and a, (len(a), len(b))
    worst = 0
    for fa, fb in zip(a, b, strict=True):
        if len(fa) != len(fb):
            return 255
        ua = np.frombuffer(fa, dtype=np.uint8).astype(np.int16)
        ub = np.frombuffer(fb, dtype=np.uint8).astype(np.int16)
        worst = max(worst, int(np.abs(ua - ub).max()))
    return worst


# ---------------------------------------------------------------------------
# 1. answer preservation — the movie is byte-for-byte what it always was
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["2d", "3d", "series", "composite", "field2d", "field1d", "fade2d", "fade3d", "annotated"],
)
def test_blitted_frames_are_byte_identical_to_a_full_draw(kind):
    """Every frame the writer receives is identical with and without the compositor."""
    slow = _record(_spec(kind), blitting=False)
    fast = _record(_spec(kind), blitting=True)
    assert _worst_pixel_difference(slow, fast) == 0


@pytest.mark.parametrize("kind", ["2d", "3d", "series", "fade2d", "fade3d", "annotated"])
def test_every_eligible_kind_actually_blits(kind):
    """Calibration passing is the point; falling back safely is only the safety net.

    ``.trail(fade=True)`` used to arm the compositor and then **fail** calibration,
    so a fading comet quietly paid the full per-frame draw while the table said the
    optimisation applied.  The cause was a 3-D axes rewriting its collections'
    ``zorder`` as it depth-sorts them (see ``_draw_foreground``).  A test that only
    asserted the movie came out right could not see the difference — this one can.
    """
    anim = _anim.render_animation(_spec(kind), figsize=FIGSIZE)
    compositor = anim._tsd_compositor
    assert compositor is not None, f"{kind}: no compositor was wired at all"
    anim.save("unused", writer=_BufferWriter())
    assert compositor.blitting is True, f"{kind}: armed but calibration failed"
    _dispose(anim._fig)


def test_annotations_survive_an_animated_render():
    """A ``.vline()`` on a movie must draw, exactly as it does on a still.

    The reveal renderer builds its own axes and used to call neither annotation
    applier, so every annotation on an animated plot vanished with no warning —
    while the static render of the same spec drew them all.
    """
    spec = _spec("annotated")
    assert len(spec.annotations) == 3
    anim = _anim.render_animation(spec, figsize=FIGSIZE)
    ax = anim._fig.axes[0]
    assert "HERE" in [t.get_text() for t in ax.texts]
    assert "onset" in [t.get_text() for t in ax.texts]  # the vline's inline label
    # the comet's line + head, plus the vline and the hline
    assert len(ax.lines) == 4
    _dispose_anim(anim)


def test_annotations_survive_an_animated_3d_render():
    """The 3-D applier is reached too (it places a reference line on the mid-z plane)."""
    spec = ts.plot(_lorenz(), animate={"n_frames": 4}).text(0.0, 0.0, "MARK")
    anim = _anim.render_animation(spec, figsize=FIGSIZE)
    assert "MARK" in [t.get_text() for t in anim._fig.axes[0].texts]
    _dispose_anim(anim)


def test_a_composite_is_played_on_the_composites_own_clock():
    """``.animate(duration=…)`` on a composite must decide how many frames are written.

    Composing with ``animate=`` stamps a default ``Animation`` on every panel that
    carries none; the renderer then read the panel's stamp in preference to the
    composite's own, so a two-second request wrote the 360-frame default instead.
    """
    traj = _lorenz()
    comp = ts.viz.plot(
        ts.plot(traj, components=("x", "z")),
        ts.plot(traj, components="x"),
        layout="row",
        animate=True,
    ).animate(fps=30.0, duration=0.2)
    anim = _anim.render_animation(comp, figsize=FIGSIZE)
    assert len(list(anim.new_frame_seq())) == comp.animation.frame_count(len(traj.t)) == 6
    _dispose_anim(anim)


@pytest.mark.parametrize("kind", ["2d", "field2d"])
def test_byte_identical_through_the_facecolor_overriding_writer_path(kind):
    """``.mp4`` writers cannot do transparency, so ``save`` forces a ``facecolor``.

    That reaches ``print_figure``, which sets it on the figure for the duration of
    the write — i.e. it changes the very background the compositor caches.  The
    compositor keys its cache on the facecolor for exactly this reason, so the
    movie must come out identical on this path too.
    """
    slow = _record(_spec(kind), blitting=False, transparency=False)
    fast = _record(_spec(kind), blitting=True, transparency=False)
    assert _worst_pixel_difference(slow, fast) == 0


def test_the_no_blit_env_var_is_a_real_bypass(monkeypatch):
    """``TSDYNAMICS_NO_BLIT`` disables the compositor and still writes the movie."""
    monkeypatch.setenv("TSDYNAMICS_NO_BLIT", "1")
    anim = _anim.render_animation(_spec("2d"), figsize=FIGSIZE)
    assert anim._tsd_compositor is not None
    assert anim._tsd_compositor.enabled is False
    writer = _BufferWriter()
    anim.save("unused", writer=writer)
    assert anim._tsd_compositor.blitting is False
    assert len(writer.frames) == 5
    _dispose(anim._fig)


# ---------------------------------------------------------------------------
# 2. the saving is structural — full draws do not scale with the frame count
# ---------------------------------------------------------------------------


def _full_draws_during_save(kind: str, n_frames: int) -> int:
    """How many whole-figure draws a save of ``n_frames`` frames performs."""
    anim = _anim.render_animation(_spec(kind, n_frames), figsize=FIGSIZE)
    compositor = anim._tsd_compositor
    assert compositor is not None
    real = compositor._real_draw
    calls = [0]

    def counted(renderer):
        calls[0] += 1
        real(renderer)

    compositor._real_draw = counted
    anim.save("unused", writer=_BufferWriter())
    assert compositor.blitting is True
    _dispose(anim._fig)
    return calls[0]


@pytest.mark.parametrize("kind", ["2d", "3d", "field2d"])
def test_full_figure_draws_do_not_grow_with_the_frame_count(kind):
    """The per-frame cost no longer contains a whole-figure draw.

    A counting assertion rather than a timing one: wall clock on a shared machine
    is not a gate, but "the tenth frame costs another full figure" is.  Before the
    compositor this number was ~3 per frame (``draw_idle`` + ``print_figure``'s
    layout pre-pass + ``savefig``); it is now a fixed calibration cost.
    """
    few = _full_draws_during_save(kind, 4)
    many = _full_draws_during_save(kind, 16)
    assert few == many, f"{few} draws for 4 frames vs {many} for 16 — cost is per-frame"


@pytest.mark.parametrize("kind", ["2d", "3d", "series", "fade2d", "fade3d", "field2d", "field1d"])
def test_a_frame_is_a_pure_function_of_its_index(kind):
    """Drawing frame 0 after frame N must give exactly frame 0.

    The compositor calibrates on a handful of scattered frames and then *restores*
    the one the writer is about to grab, so a driver that only ever moves forward
    would be calibrated against its own stale state — and the mismatch would be
    invisible, because the reference draw sees the same stale artists.  It bites a
    user directly too: a ``loop`` or ``pingpong`` playback replays frame 0 on every
    cycle.  Measured before the fix, the fading-comet driver left frame 0 showing
    the *last* frame's comet (135 differing pixels).
    """
    from tsdynamics.viz.render.mpl._core import new_figure

    spec = _spec(kind, n_frames=6)
    fig = new_figure(FIGSIZE, None, None)
    ax = fig.add_subplot(1, 1, 1, projection="3d" if spec.is_three_d else None)
    updater = _anim._build_panel_animation(fig, ax, spec, three_d=spec.is_three_d)

    def snapshot() -> bytes:
        fig.canvas.draw()
        return bytes(fig.canvas.get_renderer().buffer_rgba())

    updater.update(0)
    # Constrained layout is iterative; converge it out of the measurement so this
    # test is about the driver.  The production freeze is the tool for that.
    _anim._LayoutFreeze(fig).prepare(fig.draw)
    fresh = snapshot()
    updater.update(updater.n_steps - 1)
    updater.update(0)
    assert snapshot() == fresh, f"{kind}: frame 0 depends on what was drawn before it"
    _dispose(fig)


def test_drivers_declare_every_artist_they_mutate():
    """An artist whose data changes between frames must be in the compositor's set.

    This is the invariant the whole optimisation rests on: the compositor caches
    everything *not* in that set, so a driver that mutates an artist without
    declaring it would freeze that artist at its first frame — a silently wrong
    movie, not a crash.
    """
    from matplotlib.figure import Figure

    from tsdynamics.viz.render.mpl._core import new_figure

    spec = _spec("2d", n_frames=8)
    fig: Figure = new_figure(FIGSIZE, None, None)
    ax = fig.add_subplot(1, 1, 1)
    updater = _anim._build_panel_animation(fig, ax, spec, three_d=False)
    declared = {id(a) for a in updater.artists}

    def snapshot() -> dict[int, tuple]:
        out = {}
        for child in ax.get_children():
            getter = getattr(child, "get_data", None)
            if callable(getter):
                x, y = getter()
                out[id(child)] = (np.asarray(x).tobytes(), np.asarray(y).tobytes())
        return out

    updater.update(0)
    first = snapshot()
    updater.update(updater.n_steps - 1)
    last = snapshot()
    changed = {key for key, value in last.items() if first.get(key) != value}
    assert changed, "the sanity of the probe itself: something must move"
    assert changed <= declared
    _dispose(fig)


# ---------------------------------------------------------------------------
# 3. the exemptions — what genuinely re-draws is left alone
# ---------------------------------------------------------------------------


def _spinning(n_frames: int = 12):
    """A 3-D comet whose camera spins — the one kind that cannot be blitted."""
    return ts.plot(_lorenz(), animate={"n_frames": n_frames}).camera(spin=90.0)


def _axes_rects_per_written_frame(anim) -> list[tuple[float, ...]]:
    """The axes rectangle as each frame is handed to the writer."""
    rects: list[tuple[float, ...]] = []
    real = anim._fig.canvas.print_figure

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        rects.append(tuple(round(v, 9) for v in anim._fig.axes[0].get_position().bounds))
        return out

    anim._fig.canvas.print_figure = spy
    anim.save("unused", writer=_BufferWriter())
    return rects


def test_a_spinning_camera_is_not_blitted():
    """``.camera(spin=...)`` moves the axes themselves, so the cache would be stale."""
    anim = _anim.render_animation(_spinning(4), figsize=FIGSIZE)
    assert anim._tsd_compositor is None
    anim.save("unused", writer=_BufferWriter())  # still writes, just not blitted
    _dispose(anim._fig)


def test_a_spinning_camera_still_gets_a_fixed_framing():
    """Un-blittable is not un-freezable: the axes must not walk while the camera turns.

    A spinning 3-D axes re-measures its tick labels every frame, so constrained
    layout never converges and keeps resizing the axes underneath the attractor.
    Measured before the freeze was wired for this path: **11 distinct axes
    rectangles** over 20 frames, moving on 14 of the 19 transitions — the library's
    own stated failure mode, *a movie whose axes rescale every frame shows you the
    axes moving, not the dynamics*.  It is also what made this path's output depend
    on how many times each frame happened to be drawn, which is why nothing about
    it was reproducible.
    """
    anim = _anim.render_animation(_spinning(), figsize=FIGSIZE)
    assert anim._tsd_compositor is None
    assert anim._tsd_freeze is not None, "an unblitted animation still freezes its layout"
    rects = _axes_rects_per_written_frame(anim)
    assert len(rects) == 12
    assert len(set(rects)) == 1, f"the axes moved during the spin: {sorted(set(rects))}"
    _dispose(anim._fig)


def test_the_freeze_is_undone_when_the_save_ends():
    """An unblitted save must hand the figure back with its layout engine, like a blitted one."""
    anim = _anim.render_animation(_spinning(4), figsize=FIGSIZE)
    before = anim._fig.get_layout_engine()
    assert before is not None
    anim.save("unused", writer=_BufferWriter())
    assert anim._fig.get_layout_engine() is before
    _dispose(anim._fig)


def test_a_frames_layout_composite_is_not_blitted():
    """``layout="frames"`` plays whole panels, re-drawing the figure each frame.

    It is the one path that gets **neither** optimisation: with the figure cleared
    and an axes re-added per frame there is no static prefix to cache and no single
    framing to settle on, so its layout engine has to keep running.
    """
    logistic = ts.systems.Logistic()
    panels = [ts.plot(logistic.with_params(r=r), "cobweb") for r in (3.2, 3.6, 3.9)]
    composite = ts.viz.plot(*panels, layout="row", animate={"n_frames": 3})
    movie = dataclasses.replace(
        composite, layout=dataclasses.replace(composite.layout, mode="frames")
    )
    anim = _anim.render_animation(movie, figsize=FIGSIZE)
    assert anim._tsd_compositor is None
    assert anim._tsd_freeze is None
    anim.save("unused", writer=_BufferWriter())  # still writes, just not blitted
    assert anim._fig.get_layout_engine() is not None
    _dispose(anim._fig)


# ---------------------------------------------------------------------------
# 4. the figure survives the save
# ---------------------------------------------------------------------------


def test_the_figure_is_handed_back_unhooked_and_with_its_layout_engine():
    """A save must not leave the figure holding the compositor or a frozen layout.

    ``Plot.fig`` hands this exact figure to the caller, and a still ``.png`` save
    of a movie writes it — so a hook or a dropped layout engine surviving the save
    would leak into every later render of the same plot.
    """
    anim = _anim.render_animation(_spec("2d"), figsize=FIGSIZE)
    fig = anim._fig
    before = fig.get_layout_engine()
    assert before is not None
    anim.save("unused", writer=_BufferWriter())
    assert "draw" not in vars(fig), "the compositor is still installed as Figure.draw"
    assert fig.get_layout_engine() is before
    fig.canvas.draw()  # the plain draw path still works
    _dispose(fig)


def test_a_still_png_of_a_movie_is_still_drawn_in_full(tmp_path):
    """``.save('x.png')`` of an animated plot writes a real picture, not a blank page."""
    out = tmp_path / "still.png"
    _spec("2d").save(str(out))
    assert out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# 5. the probe schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [1, 2, 3, 5, 12, 360])
def test_probe_schedule_is_in_range_and_ends_on_the_last_frame(n_steps):
    """A reveal comet is longest on the last frame, so that frame is always probed."""
    probes = _anim._probe_schedule(n_steps)
    assert all(0 <= p <= n_steps - 1 for p in probes)
    assert probes == sorted(set(probes))
    if n_steps > 1:
        assert probes[-1] == n_steps - 1
    else:
        assert probes == []
