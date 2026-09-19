"""The pixel gate: tests that look at the picture.

The beta report's own conclusion was that *"nothing tests the pixels — almost
every wrong picture this round passed every existing test, because the tests
inspect the DESCRIPTION of a figure and never the figure."*  Four wrong pictures
shipped with a green suite:

* a recurrence plot at 5 % recurrence rate rendered as 14 % ink — a near-solid
  black square, with a perfectly correct ``Plot``;
* a 3-D attractor that dropped ``cmap="plasma"`` while the 2-D one honoured it;
* a vector field left on its old evaluation box after ``xlim=`` moved the axes;
* a ``legend=`` keyword that built fine and crashed inside matplotlib.

Every one of them is a statement about pixels, and a spec-dict assertion cannot
make a statement about pixels.  This module renders to an Agg buffer and measures
the array (see :mod:`tests._pixels` for the three ideas the measurements rest
on).  Seven questions, in the order the plan lists them:

1. **not blank** — every transform × its default primitive puts ink in its panel,
   and (slow tier) so does every declared cell of the compatibility matrix;
2. **ink density matches the data** — a recurrence plot at a known rate draws
   that much ink;
3. **colour is honoured** — two colours, and two colormaps, give two different
   pictures, in 3-D as well as 2-D;
4. **limits are honoured** — ``xlim=`` moves what is drawn, and a field
   transform's evaluation box follows the window;
5. **figure keywords render** — all seventeen build, render, and change the
   picture;
6. **animation** — consecutive frames differ and the frame count is the one asked
   for, in memory and in a written ``.gif``;
7. **composition** — an overlay carries more ink than either orbit alone, and a
   grid of four draws in all four panels.

Everything asserted here is a *robust* property — is there ink, did it change, is
the ink roughly this fraction.  Never a hash, never a golden image: fonts and
layout solvers move pixels, and a flaky pixel test is worse than none.

Two wrong pictures this gate found, and now pins
------------------------------------------------
Both were shipping, both passed every existing test, and both are fixed here.
The tests that caught them are ordinary green tests now — deliberately kept, and
kept named after the picture rather than the patch, because these are exactly the
regressions a refactor reintroduces without anything else noticing.

**A.** ``viz/render/mpl/_threed.py::_apply_3d_axes`` applied the three axis
*labels* and *limits* and nothing else, so ``xticks`` / ``yticks`` / ``zticks``
/ ``xscale`` / ``yscale`` / ``zscale`` were honoured on a 2-D axes and silently
dropped on a 3-D one — the *common* axes, for 106 of the catalogue's ODEs.
Measured: ``ts.plot(traj, components=("x","y","z"), zticks=[10, 40])`` rendered
``ax.get_zticks() == [-10, 0, 10, 20, 30, 40, 50]`` and the picture was
bit-identical to the one without the keyword.

**B.** ``viz/transforms/_primitives.py::_build_density`` binned unconditionally
onto 400 x 400.  Right for the million-point orbit diagram its docstring cites,
**blank** for anything smaller: 200 points over 160 000 bins is 0.125 %
occupancy and nothing survives the panel downsample.  Ink at each transform's
own shipped example was ``phase_portrait`` 0.0000, ``delay_embedding`` 0.0008,
``poincare_section`` 0.0005; it is now 0.193 / 0.218 / 0.084.  The bin count
follows the sample count and still caps at 400, so the picture the old number
was chosen for is byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

import _pixels as px  # noqa: E402

import tsdynamics as ts  # noqa: E402
from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402
from tsdynamics.viz.render.caps import VisualizationDegraded  # noqa: E402
from tsdynamics.viz.spec import FIGURE_KEYS  # noqa: E402
from tsdynamics.viz.transforms import build_spec, get, transforms  # noqa: E402

# ---------------------------------------------------------------------------
# floors, each with the margin it was set from
# ---------------------------------------------------------------------------

#: Ink floor for a transform drawn with its **default** primitive.  Measured
#: minimum over the 39 registered transforms: ``psd``/``line`` at 0.0319, so this
#: is a 3.2× margin.  A plot that drew nothing scores exactly 0.0 — spines and
#: ticks fall outside the measured region by construction.
DEFAULT_FLOOR = 0.01

#: Ink floor for any declared cell of the compatibility matrix.  Measured
#: minimum over the 101 cells, excluding the three the density defect blanks:
#: ``orbit_diagram``/``density`` at 0.0116, a 2.9× margin.
CELL_FLOOR = 0.004

#: How far a rendered recurrence plot's ink may sit from the recurrence rate the
#: matrix actually realised.  Measured agreement is within 0.002 at both 5 % and
#: 20 %; this is a 10× margin, and still a tenth of the 5 %→14 % error that
#: shipped.
INK_TOLERANCE = 0.02

#: How much of a panel two renders must disagree over before "the picture
#: changed".  Every effect this is applied to measures ≥ 0.014; two renders of
#: the same spec differ by exactly 0.0, because Agg is deterministic.
CHANGED = 0.002

#: The same question asked as a *count*, for the figure-keyword sweep.  Effect
#: sizes there span four orders of magnitude — ``theme="dark"`` repaints the
#: whole canvas, ``zlabel="…"`` edits one short string and moves 92 pixels of
#: 102 400 — so a single fractional floor either lets a dropped keyword through
#: or fails an honoured one.  What separates the two cleanly is that a **dropped
#: keyword moves exactly zero pixels**; this floor sits 5.75× above the smallest
#: real effect and infinitely above none.
MIN_CHANGED_PIXELS = 16


@pytest.fixture(autouse=True)
def _isolated_matplotlib():
    """Pin Agg, quarantine rcParams, and leave no figure behind.

    The renderer builds bare ``Figure`` objects pyplot has never seen, so
    ``close("all")`` alone frees nothing — :func:`tests._pixels.render` clears
    each figure itself.  This is the belt for anything that slipped past.
    """
    with matplotlib.rc_context():
        yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="module", autouse=True)
def _renderers():
    from tsdynamics import registry

    register_builtin_renderers()
    if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present here
        pytest.skip("matplotlib backend did not register")


# ---------------------------------------------------------------------------
# pinned subjects
#
# Every system is run with an explicit ``ic=``.  A catalogue system with no
# ``_default_ic`` draws a fresh random start on each call, so an un-pinned
# subject makes "did the picture change" a question about the initial condition.
# ---------------------------------------------------------------------------


def _lorenz(final_time: float = 8.0, dt: float = 0.02):
    return ts.systems.Lorenz().run(final_time=final_time, dt=dt, ic=[1.0, 1.0, 1.0])


def _quadrants(pic: px.Picture) -> list[float]:
    """Ink in each quarter of the main panel, reading rows then columns."""
    data = pic.data
    h, w = data.shape[:2]
    return [
        px.ink_fraction(data[r * h // 2 : (r + 1) * h // 2, c * w // 2 : (c + 1) * w // 2])
        for r in (0, 1)
        for c in (0, 1)
    ]


# ===========================================================================
# 1. NOT BLANK
# ===========================================================================


def _defaults() -> list[tuple[str, str]]:
    return sorted((t.name, t.default_primitive) for t in transforms())


def _cells() -> list[object]:
    """Every declared (transform, primitive) cell of the compatibility matrix.

    No carve-outs: the three ``density`` cells that used to be excused here are
    the three defect B made blank, and they draw now.
    """
    return [
        pytest.param(record.name, primitive, id=f"{record.name}-{primitive}")
        for record in transforms()
        for primitive in sorted(record.primitives)
    ]


@pytest.mark.parametrize(("name", "primitive"), _defaults(), ids=lambda v: str(v))
def test_every_transform_actually_draws_something(name, primitive):
    """Rendering a transform leaves ink in its panel.

    The one test that would have caught most of this round's wrong pictures: a
    figure whose description is perfect and whose data area is empty scores
    exactly 0.0 here, because spines, ticks and the colorbar strip are trimmed
    out of the measured region before anything is counted.
    """
    subject, options = get(name).example(primitive)
    pic = px.render(build_spec(subject, name, primitive=primitive, **dict(options)))
    assert pic.ink() >= DEFAULT_FLOOR, (
        f"{name}.{primitive} rendered a panel that is {pic.ink():.4%} ink — below the "
        f"{DEFAULT_FLOOR:.1%} floor. The spec may be perfect; the picture is blank."
    )


@pytest.mark.slow
@pytest.mark.parametrize(("name", "primitive"), _cells())
def test_every_declared_cell_draws_something(name, primitive):
    """Every pair the compatibility matrix promises puts ink on the canvas.

    ``tests/test_viz_compatibility.py`` already renders the whole matrix and
    checks the artists carry finite data.  An artist can carry finite data and
    still be invisible — which is exactly how three ``density`` cells ship a
    blank image today.
    """
    subject, options = get(name).example(primitive)
    pic = px.render(build_spec(subject, name, primitive=primitive, **dict(options)))
    assert pic.ink() >= CELL_FLOOR, (
        f"declared cell {name}.{primitive} drew {pic.ink():.4%} ink — the matrix promises a "
        f"plot this pair can produce, and this one is empty."
    )


# ===========================================================================
# 2. INK DENSITY MATCHES THE DATA
# ===========================================================================


@pytest.mark.parametrize("rate", [0.05, 0.20])
def test_a_recurrence_plot_draws_as_much_ink_as_its_recurrence_rate(rate):
    """5 % of recurrent points is 5 % of ink, not 14 %.

    The bug this pins shipped green: the matrix was right, the ``Plot`` was
    right, and the picture was a near-solid black square.  Nothing between the
    estimator and the reader was checking that the *amount of black* is the
    number the estimator computed.
    """
    series = _lorenz(final_time=40.0, dt=0.05)["x"]
    matrix = ts.analysis.recurrence_matrix(series, recurrence_rate=rate)
    realised = float(np.asarray(matrix).mean())

    pic = px.render(ts.plot(matrix), size=4.0)
    drawn = pic.dark()

    assert abs(drawn - realised) <= INK_TOLERANCE, (
        f"a recurrence matrix that is {realised:.1%} recurrent rendered as {drawn:.1%} ink "
        f"(tolerance {INK_TOLERANCE:.0%}). The picture is not the data."
    )


def test_more_recurrence_means_more_ink():
    """Two rates, and the picture has to tell them apart by the right amount.

    A single-point check can be passed by a plot whose ink happens to sit near
    one rate.  This is the complement, not the replacement: an error that is
    *affine* in the density (a dilation, say) shifts both rates together and is
    invisible here — which is why the absolute check above is the load-bearing
    one, and this is the one that catches a rescaling.
    """
    series = _lorenz(final_time=40.0, dt=0.05)["x"]
    sparse = ts.plot(ts.analysis.recurrence_matrix(series, recurrence_rate=0.05))
    dense = ts.plot(ts.analysis.recurrence_matrix(series, recurrence_rate=0.20))

    gap = px.render(dense, size=4.0).dark() - px.render(sparse, size=4.0).dark()
    assert abs(gap - 0.15) <= INK_TOLERANCE, (
        f"raising the recurrence rate from 5% to 20% changed the ink by {gap:.1%}, "
        "not the 15 percentage points the data moved by."
    )


# ===========================================================================
# 3. COLOUR IS HONOURED
# ===========================================================================


@pytest.mark.parametrize("components", [("x", "z"), ("x", "y", "z")], ids=["2d", "3d"])
def test_two_colours_make_two_different_pictures(components):
    """``color="red"`` and ``color="blue"`` cannot render the same pixels."""
    traj = _lorenz()
    red = px.render(ts.plot(traj, components=components, color="red"))
    blue = px.render(ts.plot(traj, components=components, color="blue"))
    changed = px.difference(red, blue)
    assert changed > CHANGED, (
        f"a {len(components)}-D portrait drawn in red and in blue differs over only "
        f"{changed:.4%} of the figure — the colour was accepted and never reached the canvas."
    )


@pytest.mark.parametrize("components", [("x", "z"), ("x", "y", "z")], ids=["2d", "3d"])
def test_two_colormaps_make_two_different_pictures(components):
    """``cmap=`` reaches a 3-D attractor, not only a 2-D one.

    The shipped bug exactly: 2-D honoured ``cmap="plasma"``, 3-D silently drew
    the default, and every spec-level assertion agreed the colormap was set.
    """
    traj = _lorenz()
    plasma = px.render(ts.plot(traj, components=components, color_by="time", cmap="plasma"))
    viridis = px.render(ts.plot(traj, components=components, color_by="time", cmap="viridis"))
    changed = px.difference(plasma, viridis)
    assert changed > CHANGED, (
        f"a {len(components)}-D portrait coloured by time renders identically under 'plasma' and "
        f"'viridis' ({changed:.4%} of pixels differ) — the colormap is being dropped."
    )


def test_a_colormap_reaches_an_image_as_well_as_a_line():
    """The image primitives take the colormap too — a different code path."""
    field = px.render(ts.plot(ts.systems.VanDerPol(), "flow_speed", cmap="magma"))
    other = px.render(ts.plot(ts.systems.VanDerPol(), "flow_speed", cmap="viridis"))
    assert px.difference(field, other) > CHANGED


# ===========================================================================
# 4. LIMITS ARE HONOURED
# ===========================================================================


def _flat_then_busy():
    """A series that is flat for its first half and oscillating for its second.

    Built so that *what the window selects* is legible as ink: the two halves
    differ by ~85× in how much of a panel they fill.
    """
    t = np.linspace(0.0, 10.0, 2001)
    y = np.where(t < 5.0, 0.0, np.sin(40.0 * t))
    return ts.Trajectory(t=t, y=y.reshape(-1, 1))


def test_moving_the_x_window_moves_what_is_drawn():
    """``xlim=`` selects part of the data, and the picture shows that part."""
    traj = _flat_then_busy()
    frame = {"ylim": (-1.2, 1.2)}
    flat = px.render(ts.plot(traj, xlim=(0.0, 4.5), **frame))
    busy = px.render(ts.plot(traj, xlim=(5.5, 10.0), **frame))

    assert flat.ink() < 0.05, f"the flat half of the series drew {flat.ink():.1%} ink"
    assert busy.ink() > 0.30, f"the oscillating half of the series drew {busy.ink():.1%} ink"
    assert busy.ink() > 5.0 * flat.ink(), (
        "the two windows of a series that is flat then busy rendered near-identical amounts of "
        "ink — xlim= is framing the axes without selecting the data."
    )


@pytest.mark.parametrize(
    "domain",
    [[(-3.0, 3.0), (-3.0, 3.0)], [(2.0, 6.0), (2.0, 6.0)], [(-8.0, -4.0), (1.0, 5.0)]],
    ids=["origin", "shifted", "disjoint"],
)
def test_a_field_is_evaluated_over_the_window_it_is_drawn_on(domain):
    """Move the window and the arrows follow it — the whole panel, not a corner.

    The shipped bug left the field on its *previous* evaluation box while the
    axes moved, so the arrows survived only in the overlap and the rest of the
    panel was empty.  Requiring ink in all four quadrants is the measurement
    that catches it, and it does not care which box the old one was.
    """
    pic = px.render(ts.plot(ts.systems.VanDerPol(), "vector_field", domain=domain))
    quadrants = _quadrants(pic)
    assert min(quadrants) > 0.02, (
        f"a vector field on {domain} left quadrant ink {[round(q, 4) for q in quadrants]} — the "
        "arrows are not spread over the window the axes show."
    )


def test_two_disjoint_windows_are_two_different_field_pictures():
    """Two windows that share no points cannot draw the same arrows."""
    vdp = ts.systems.VanDerPol()
    near = px.render(ts.plot(vdp, "vector_field", domain=[(-3.0, 3.0), (-3.0, 3.0)]))
    far = px.render(ts.plot(vdp, "vector_field", domain=[(2.0, 6.0), (2.0, 6.0)]))
    assert px.difference(near, far) > 0.02


def test_the_axes_window_also_moves_the_field_it_frames():
    """``xlim=``/``ylim=`` on a model transform reaches the evaluation box too.

    It warns while doing it — the keyword is doing two jobs and says so — but the
    picture is the one the caller wanted, and that is what is checked here.
    """
    with pytest.warns(VisualizationDegraded, match="two jobs"):
        pic = px.render(
            ts.plot(ts.systems.VanDerPol(), "vector_field", xlim=(2.0, 6.0), ylim=(2.0, 6.0))
        )
    quadrants = _quadrants(pic)
    assert min(quadrants) > 0.02, (
        f"xlim=/ylim= moved the axes to [2, 6]² but the arrows only reached "
        f"{[round(q, 4) for q in quadrants]} of each quadrant."
    )


# ===========================================================================
# 5. FIGURE KEYWORDS RENDER
# ===========================================================================

#: One natural value per figure keyword, plus the subject it applies to.  Written
#: as *what a user would type*: a title is a sentence, a window is a pair of
#: floats, ``legend="upper left"`` is where you move the key on a two-orbit
#: overlay.  That last one is chosen deliberately over ``legend=True``: an
#: overlay already legends itself, so ``True`` is a no-op, and a no-op keyword
#: reaches ``ax.legend`` through **no** call — which is how a crash *inside*
#: matplotlib's legend construction stayed invisible.
#:
#: The 3-D rows render a little larger (4 in at dpi 64): at 3 × 3 the ``mplot3d``
#: z-label is laid out off-canvas, so a test at that size would report "the
#: keyword does nothing" about the figure size rather than about the keyword.
_FIGURE_KEYWORDS: dict[str, tuple[str, object, float]] = {
    "title": ("series", "A Lorenz orbit", 3.0),
    "xlabel": ("series", "time (s)", 3.0),
    "ylabel": ("series", "x(t)", 3.0),
    "xlim": ("series", (0.0, 5.0), 3.0),
    "ylim": ("series", (-5.0, 5.0), 3.0),
    "xticks": ("series", [0.0, 6.0, 12.0], 3.0),
    "yticks": ("series", [-10.0, 10.0], 3.0),
    "xscale": ("positive", "log", 3.0),
    "yscale": ("positive", "log", 3.0),
    "legend": ("overlay", "upper left", 3.0),
    "theme": ("series", "dark", 3.0),
    "clim": ("coloured", (4.0, 8.0), 3.0),
    "colorbar": ("coloured", False, 3.0),
    "zlabel": ("portrait3d", "DEPTH AXIS", 4.0),
    "zlim": ("portrait3d", (0.0, 80.0), 4.0),
    "zticks": ("portrait3d", [10.0, 40.0], 4.0),
    "zscale": ("portrait3d", "log", 4.0),
}


def _subject(kind: str):
    traj = _lorenz(final_time=12.0)
    if kind == "series":
        return (traj,), {"components": "x"}
    if kind == "positive":
        return (traj,), {"components": "z"}
    if kind == "portrait3d":
        return (traj,), {"components": ("x", "y", "z")}
    if kind == "coloured":
        return (traj,), {"components": ("x", "z"), "color_by": "time"}
    if kind == "overlay":
        other = ts.systems.Lorenz().run(final_time=12.0, dt=0.02, ic=[1.0, 1.0, 1.2])
        return (traj["x"], other["x"]), {}
    raise AssertionError(kind)  # pragma: no cover


def test_the_figure_keyword_table_is_the_whole_vocabulary():
    """Every figure keyword the library declares is exercised below, and no other.

    Without this, a seventeenth keyword becomes an eighteenth and the new one is
    tested by nobody.
    """
    assert set(_FIGURE_KEYWORDS) == set(FIGURE_KEYS)


@pytest.mark.parametrize("key", sorted(_FIGURE_KEYWORDS), ids=lambda v: str(v))
def test_every_figure_keyword_renders(key):
    """It builds *and it draws*.

    A keyword that raises inside matplotlib hundreds of lines after the call that
    accepted it is the defect this closes: ``legend=`` shipped doing exactly
    that, because the only thing that ever stopped it was that nothing drew it.
    """
    kind, value, size = _FIGURE_KEYWORDS[key]
    subjects, base = _subject(kind)
    pic = px.render(ts.plot(*subjects, **base, **{key: value}), size=size, dpi=64)
    assert pic.ink() > 0.0, f"{key}={value!r} rendered an empty panel"


@pytest.mark.parametrize(
    "key",
    sorted(_FIGURE_KEYWORDS),
)
def test_every_figure_keyword_changes_the_picture(key):
    """Rendering is not enough: the keyword has to *do* something.

    Each value is chosen to be visibly different from the default, so "no pixels
    moved" means the keyword was accepted and dropped — the failure mode that
    cannot be seen from the spec, because the spec records it faithfully either
    way.
    """
    kind, value, size = _FIGURE_KEYWORDS[key]
    subjects, base = _subject(kind)
    before = px.render(ts.plot(*subjects, **base), size=size, dpi=64)
    after = px.render(ts.plot(*subjects, **base, **{key: value}), size=size, dpi=64)
    moved = px.changed_pixels(before, after)
    assert moved >= MIN_CHANGED_PIXELS, (
        f"{key}={value!r} moved {moved} pixels (floor {MIN_CHANGED_PIXELS}). "
        "The keyword is accepted, recorded on the Plot, and never drawn."
    )


def test_tick_and_scale_keywords_are_honoured_on_a_three_dimensional_axes():
    """What a 2-D axes does with ticks and scales, a 3-D axes must do too.

    Checked against the rendered axes rather than the pixels, because that is the
    precise diagnosis: the 2-D half of each pair passes today, so this test
    cannot mask a regression in the half that works.
    """
    import matplotlib.pyplot as plt

    traj = _lorenz()
    asked = {"xticks": [-10.0, 10.0], "yticks": [-10.0, 10.0], "xscale": "log"}

    flat = ts.plot(traj, components=("x", "z"), **asked)
    flat.size(4.0, 4.0, dpi=64)
    fig = flat.render("matplotlib")
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert list(ax.get_xticks()) == asked["xticks"]
        assert list(ax.get_yticks()) == asked["yticks"]
        assert ax.get_xscale() == "log"
    finally:
        fig.clear()
        plt.close("all")

    cube = ts.plot(
        traj,
        components=("x", "y", "z"),
        xticks=[-10.0, 10.0],
        yticks=[-10.0, 10.0],
        zticks=[10.0, 40.0],
        zscale="log",
    )
    cube.size(4.0, 4.0, dpi=64)
    fig = cube.render("matplotlib")
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert list(ax.get_xticks()) == [-10.0, 10.0]
        assert list(ax.get_yticks()) == [-10.0, 10.0]
        assert list(ax.get_zticks()) == [10.0, 40.0]
        assert ax.get_zscale() == "log"
    finally:
        fig.clear()
        plt.close("all")


# ===========================================================================
# 6. ANIMATION
# ===========================================================================

_ANIMATED = {
    "2d": dict(components=("x", "z")),
    "3d": dict(components=("x", "y", "z")),
    "series": dict(components="x"),
}


@pytest.mark.parametrize("kind", sorted(_ANIMATED), ids=lambda v: str(v))
def test_consecutive_animation_frames_are_not_the_same_picture(kind):
    """A movie has to move.

    A still repeated N times satisfies every frame-count assertion in the suite,
    writes a valid file, and is not an animation.
    """
    plot = ts.plot(_lorenz(), **_ANIMATED[kind], animate={"n_frames": 6})
    written = px.frames(plot)

    assert len(written) == 6
    gaps = [px.difference(written[i - 1], written[i]) for i in range(1, len(written))]
    assert min(gaps) > 0.0, (
        f"a {kind} movie repeated a frame: consecutive differences {[round(g, 5) for g in gaps]}"
    )
    assert max(gaps) > CHANGED, (
        f"a {kind} movie's frames all but coincide (largest change {max(gaps):.4%}) — "
        "the picture is a still, whatever the file says."
    )


@pytest.mark.parametrize("n_frames", [3, 9])
def test_the_number_of_frames_written_is_the_number_asked_for(n_frames):
    """``n_frames=`` is a promise about the file, not about the directive."""
    plot = ts.plot(_lorenz(), components=("x", "z"), animate={"n_frames": n_frames})
    assert len(px.frames(plot)) == n_frames


def test_a_written_gif_holds_frames_that_move(tmp_path):
    """End to end: what is on disk, read back.

    A one-frame ``.gif`` has shipped from here before, and every in-memory check
    in the suite passed while it did.
    """
    pytest.importorskip("PIL")
    out = tmp_path / "orbit.gif"
    plot = ts.plot(_lorenz(), components=("x", "z"), animate={"n_frames": 6})
    plot.size(3.0, 3.0, dpi=50)
    plot.save(str(out))

    read_back = px.gif_frames(out)
    assert len(read_back) == 6, f"asked for 6 frames, the file holds {len(read_back)}"
    gaps = [px.difference(read_back[i - 1], read_back[i]) for i in range(1, len(read_back))]
    assert min(gaps) > 0.0, f"the written gif repeats a frame: {[round(g, 5) for g in gaps]}"


# ===========================================================================
# 7. COMPOSITION
# ===========================================================================


def test_an_overlay_has_more_ink_than_either_orbit_alone():
    """Two orbits on one axes draw both of them.

    The window is pinned on all three renders, because otherwise the overlay
    rescales to hold both orbits and "more ink" would be a question about the
    auto-limits rather than about the overlay.
    """
    lor = ts.systems.Lorenz()
    first = lor.run(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    second = lor.run(final_time=8.0, dt=0.02, ic=[-8.0, -6.0, 26.0])
    window = {"components": ("x", "z"), "xlim": (-25.0, 25.0), "ylim": (0.0, 50.0)}

    alone_first = px.render(ts.plot(first, **window)).ink()
    alone_second = px.render(ts.plot(second, **window)).ink()
    together = px.render(ts.plot(first, second, **window)).ink()

    assert together > max(alone_first, alone_second) + CHANGED, (
        f"an overlay of two orbits ({together:.2%} ink) carries no more than the heavier one "
        f"alone ({alone_first:.2%} / {alone_second:.2%}) — one of them is not being drawn."
    )


def test_a_grid_of_four_draws_in_all_four_panels():
    """Four panels, four pictures — not one picture and three empty boxes."""
    lor = ts.systems.Lorenz()
    first = lor.run(final_time=8.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    second = lor.run(final_time=8.0, dt=0.02, ic=[-8.0, -6.0, 26.0])
    grid = ts.viz.grid(
        ts.plot(first, components="x"),
        ts.plot(first, components=("x", "z")),
        ts.plot(second, components="y"),
        ts.plot(second, components=("y", "z")),
        rows=2,
        cols=2,
    )
    pic = px.render(grid, size=4.0, dpi=60)

    assert len(pic.panels) == 4, f"a 2x2 grid rendered {len(pic.panels)} measurable panels"
    inks = [pic.ink(i) for i in range(4)]
    assert min(inks) > DEFAULT_FLOOR, (
        f"a grid of four drew {[round(i, 4) for i in inks]} — a panel of the figure is empty."
    )
