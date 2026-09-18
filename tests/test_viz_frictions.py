"""The plotting frictions six blind beta testers hit, each pinned by its cause.

Every test here is the inverse of something a user *did* — with no docs and no
source — and got a wrong picture, a dead end, or a message that pointed
elsewhere.  The rule that decides what belongs in this file: **a fix earns a test
when the defect was invisible**, i.e. nothing raised and nothing warned.

Three clusters, worst first:

1. **Plain Python at every door.**  ``legend="upper left"`` and
   ``colorbar="right"`` were accepted and died inside the renderer; ``labels=``
   had no spelling at all; a colour value was validated by matplotlib, hundreds
   of frames later.
2. **A declared thing, silently dropped.**  ``cmap`` on a 3-D line; ``xlim=``
   swallowed by the figure while the transform that also declares it computed on
   its auto window; an animated frame themed differently from the still of the
   same spec.
3. **The library knowing something and not saying it.**  A field windowed so
   tightly that two of the five orbits drawn on it fell outside the axes; a
   recurrence plot with more markers than the figure has pixels; one linear
   colour scale flattening three of four panels.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

import tsdynamics as ts
import tsdynamics.viz as viz
from tsdynamics.errors import InvalidInputError, InvalidParameterError
from tsdynamics.viz.render.caps import VisualizationDegraded
from tsdynamics.viz.spec import Colorbar, Legend, PlotKind


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure this module opens (the machine has been OOM-killed)."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _lorenz(final_time: float = 10.0, dt: float = 0.02):
    return ts.systems.Lorenz().run(final_time=final_time, dt=dt, ic=[1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 1. Plain Python at every door
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("upper left", Legend(location="upper left")),
        ({"location": "lower right", "ncol": 2}, Legend(location="lower right", ncol=2)),
        (True, Legend()),
        (Legend(title="runs"), Legend(title="runs")),
    ],
)
def test_legend_takes_what_a_caller_would_type_and_renders(value, expected, tmp_path):
    """``legend='upper left'`` built fine and died in ``_apply_legend`` at save time.

    ``AttributeError: 'str' object has no attribute 'show'`` — 15 of the 17
    figure keywords took plain Python and the two carrying a typed IR noun did
    not.  Building is not the test: this **renders**, because the only thing that
    ever stopped these two working is that nothing drew them.
    """
    plot = ts.plot(_lorenz(), legend=value)
    assert plot.legend == expected
    plot.save(tmp_path / "legend.png")


@pytest.mark.parametrize("value", ["left", {"location": "bottom"}, True])
def test_colorbar_takes_what_a_caller_would_type_and_renders(value, tmp_path):
    """The twin defect: ``'str' object has no attribute 'cmap'``, from the renderer."""
    plot = ts.plot(_lorenz(), color_by="time", colorbar=value)
    assert isinstance(plot.colorbar, Colorbar)
    plot.save(tmp_path / "colorbar.png")


def test_a_colorbar_placement_keeps_the_label_it_already_had():
    """``colorbar='left'`` means *move the one I have*, not *throw its label away*."""
    plot = ts.plot(_lorenz(), color_by="time")
    assert plot.colorbar is not None and plot.colorbar.label == "time"
    plot.colorize(colorbar="left")
    assert plot.colorbar.location == "left" and plot.colorbar.label == "time"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("legend", "uper left", "upper left"),
        ("legend", 3.5, "placement string"),
        ("colorbar", "middle", "Accepted"),
    ],
)
def test_a_bad_placement_is_refused_at_the_door_naming_the_vocabulary(key, value, match):
    """Named here, not as an ``AttributeError`` from inside a renderer later."""
    with pytest.raises(InvalidParameterError, match=match):
        ts.plot(_lorenz(), **{key: value})


def test_reading_a_property_that_raises_does_not_report_it_as_missing():
    """``p.ax`` said *"'Plot' object has no attribute 'ax'"* on a plot that has one.

    A ``@property`` whose body raises ``AttributeError`` fails lookup exactly
    like one that does not exist, so ``__getattr__`` answered for it — the
    escape hatch accusing itself of not existing, which is the most expensive
    kind of wrong message.  It cost a tester several minutes and nearly three of
    their five figures.
    """
    plot = ts.plot(_lorenz())
    plot.legend = "not a Legend"  # bypass the door's coercion on purpose
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is the assertion
        _ = plot.ax
    message = str(excinfo.value)
    assert "no attribute 'ax'" not in message
    assert "Plot.ax exists" in message
    assert excinfo.value.__cause__ is not None


def test_a_genuinely_missing_attribute_is_still_a_plain_attribute_error():
    """The guard must not break ``hasattr`` for the rest of the universe."""
    plot = ts.plot(_lorenz())
    assert not hasattr(plot, "definitely_not_a_plot_attribute")


def test_labels_names_the_curves_at_the_front_door():
    """Comparing two parameter values is the commonest figure here — and had no spelling.

    ``label=`` / ``labels=`` / ``legend_labels=`` were all refused, two of them
    suggesting ``zlabel=`` (an *axis* name), leaving ``p.layers[i].label`` — the
    IR — as the only route.
    """
    a, b = _lorenz(), _lorenz(final_time=9.0)
    plot = ts.plot(a, b, components="x", labels=["reference", "perturbed"])
    assert [layer.label for layer in plot.layers] == ["reference", "perturbed"]
    assert plot.legend is not None


def test_labels_works_through_the_composition_door_too():
    """One spelling, both doors: ``ts.plot`` and ``ts.viz.plot``."""
    a, b = _lorenz(), _lorenz(final_time=9.0)
    plot = viz.plot(a, b, components="x", labels=["one", "two"])
    assert [layer.label for layer in plot.layers] == ["one", "two"]


def test_labels_is_matched_to_subjects_not_to_the_specs_they_produced():
    """A subject can draw several specs; the count a caller can predict is subjects."""
    vdp = ts.systems.VanDerPol()
    orbit = vdp.run(final_time=8.0, dt=0.02, ic=[2.0, 0.0])
    plot = ts.plot(vdp, orbit, "vector_field", "nullclines", labels=[None, "limit cycle"])
    assert "limit cycle" in [layer.label for layer in plot.layers]


def test_a_named_subject_is_exempt_from_the_automatic_disambiguation():
    """You already said what it is; ``(1)`` / ``(2)`` would be noise on top."""
    a, b = _lorenz(), _lorenz()
    plot = ts.plot(a, b, components="x", labels=["A", "B"])
    assert [layer.label for layer in plot.layers] == ["A", "B"]


def test_a_multi_curve_subject_gets_the_label_as_a_prefix():
    """One name over several curves would legend two different things identically."""
    plot = ts.plot(_lorenz(), "time_series", labels=["run A"], components=["x", "y"])
    assert [layer.label for layer in plot.layers] == ["run A: x", "run A: y"]


def test_a_wrong_label_count_names_both_counts():
    """A silently short sequence would label the wrong curves."""
    a, b = _lorenz(), _lorenz()
    with pytest.raises(InvalidParameterError, match="2 were plotted and 1 label"):
        ts.plot(a, b, labels=["only one"])


def test_the_singular_label_is_answered_with_the_plural_not_with_an_axis_name():
    """``difflib`` tie-broke on the ``z`` of ``zlabel``, which encodes nothing."""
    with pytest.raises(InvalidParameterError, match=r"label= — did you mean labels=\?"):
        ts.plot(_lorenz(), "phase_portrait", label="x")


@pytest.mark.parametrize("value", ["crimson", "#d81b60", "0.4", (0.1, 0.2, 0.3)])
def test_every_real_colour_spelling_is_accepted(value):
    """The validation must not cost the vocabulary matplotlib already understands."""
    assert ts.plot(_lorenz(), color=value).layers[0].style["color"] == value


def test_a_colour_that_is_not_a_colour_is_refused_at_the_door():
    """``c='time'`` was accepted and died in ``.save()`` with a raw matplotlib error.

    The one untranslated backend error a tester met all session — and a plausible
    guess, because ``color_by='time'`` is real.
    """
    with pytest.raises(ValueError, match="color_by='time'"):
        ts.plot(_lorenz(), c="time")


def test_plot_title_is_a_string_that_teaches_when_you_call_it():
    """Every neighbour of ``title`` in ``dir(p)`` is a verb; ``p.title('x')`` was not.

    It answered ``TypeError: 'str' object is not callable`` — the one message in
    the library that names nothing.  There is deliberately no second way to set
    it: calling it says where the title is set.
    """
    plot = ts.plot(_lorenz(), title="Lorenz")
    assert plot.title == "Lorenz" and isinstance(plot.title, str)
    with pytest.raises(InvalidInputError, match=r"relabel\(title='Rossler'\)"):
        plot.title("Rossler")


def test_the_title_serializes_as_a_plain_string():
    """The JSON envelope must not learn about the subclass."""
    plot = ts.plot(_lorenz(), title="Lorenz")
    assert type(plot.to_dict()["title"]) is str
    assert viz.Plot.from_dict(plot.to_dict()).title == "Lorenz"


# ---------------------------------------------------------------------------
# 2. A declared thing, silently dropped
# ---------------------------------------------------------------------------


def test_cmap_reaches_a_three_d_line_exactly_as_it_reaches_a_two_d_one():
    """A declared style key, dropped with no warning, on the 3-D majority of the catalogue.

    The 3-D path read ``spec.colorbar.cmap`` only; the front door writes every
    style key onto the **layer**.  The identical 2-D call honoured it, which is
    what made the drop invisible — a tester shipped a viridis hero figure
    believing it was plasma.
    """
    traj = _lorenz()
    names = {}
    for tag, extra in (("3d", {}), ("2d", {"components": ["x", "z"]})):
        plot = ts.plot(traj, color_by="time", cmap="plasma", **extra)
        axes = plot.fig.axes[0]
        names[tag] = {c.cmap.name for c in axes.collections if getattr(c, "cmap", None)}
    assert "plasma" in names["3d"] and "plasma" in names["2d"]


def test_a_shared_window_reaches_the_transform_that_also_declares_it():
    """``ts.plot(sys, 'flow_speed', ylim=(-8, 8))`` moved the axes and not the field.

    Measured: the geometry door (``ts.viz.geometry``) windowed it correctly and
    the plot door did not — two doors, one word, two pictures, no warning.  It
    reaches both now, and says so (see the ``domain=`` tests below).
    """
    vdp = ts.systems.VanDerPol()
    with pytest.warns(VisualizationDegraded):
        plot = ts.plot(vdp, "flow_speed", ylim=(-8.0, 8.0))
    field_y = np.asarray(plot.layers[0].data["y"], dtype=float)
    assert (float(field_y.min()), float(field_y.max())) == (-8.0, 8.0)
    assert plot.y.limits == (-8.0, 8.0)


def test_a_transforms_own_window_still_wins_over_the_shared_one():
    """The escape hatch stays exact: computing over one box, drawing over another."""
    vdp = ts.systems.VanDerPol()
    with pytest.warns(VisualizationDegraded):
        plot = ts.plot(vdp, ("flow_speed", {"ylim": (-3.0, 3.0)}), ylim=(-10.0, 10.0))
    field_y = np.asarray(plot.layers[0].data["y"], dtype=float)
    assert (float(field_y.min()), float(field_y.max())) == (-3.0, 3.0)
    assert plot.y.limits == (-10.0, 10.0)


def test_an_animated_frame_is_themed_exactly_like_the_still_of_the_same_spec(tmp_path):
    """One spec rendered ``#e6e6e6`` as a ``.png`` and near-black as a ``.gif``.

    A talk slide and a paper figure that do not match.  The cause: the theme
    coloured a title *if it already existed*, and the animator set it afterwards.
    """
    from PIL import Image

    traj = _lorenz()
    for extra in ({"components": ["x", "z"]}, {}):
        base = {"theme": "dark", "title": "Lorenz", **extra}
        still = tmp_path / "still.png"
        movie = tmp_path / "movie.gif"
        ts.plot(traj, **base).save(still)
        ts.plot(traj, animate=True, **base).animate(n_frames=4).save(movie, fps=4)
        with Image.open(still) as opened:
            a = np.asarray(opened.convert("RGB"), dtype=float)
        with Image.open(movie) as frames:
            frames.seek(frames.n_frames - 1)
            b = np.asarray(frames.convert("RGB"), dtype=float)
        # The title band: light ink on a dark page in both, or neither.
        assert np.allclose(a[:38].mean(axis=(0, 1)), b[:38].mean(axis=(0, 1)), atol=1.0)


def test_the_animation_backdrop_has_a_switch_and_it_changes_frame_zero(tmp_path):
    """The faint full curve is drawn from frame 0 — it shows the ending first.

    Undocumented, absent from ``.trail``'s help, and unreachable by any of the
    14 keys in ``to_dict()["animation"]``: a tester gave up and used a
    persistent trail instead, which is a different animation.
    """
    from PIL import Image

    traj = ts.systems.Lorenz().run(final_time=30.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    ink = {}
    for tag, backdrop in (("on", True), ("off", False)):
        path = tmp_path / f"{tag}.gif"
        plot = ts.plot(traj, components=["x", "z"], animate=True).animate(n_frames=6)
        plot.trail(("time", 2.0), backdrop=backdrop).save(path, fps=6)
        with Image.open(path) as frames:
            frames.seek(0)
            pixels = np.asarray(frames.convert("RGB"), dtype=float).reshape(-1, 3)
        ink[tag] = int((pixels.max(axis=1) < 250).sum())
    assert ink["off"] < ink["on"] / 2


def test_the_backdrop_switch_round_trips_through_the_envelope():
    """A knob that does not survive ``to_dict``/``from_dict`` is not a knob."""
    plot = ts.plot(_lorenz(), animate=True).trail(("time", 2.0), backdrop=False)
    assert viz.Plot.from_dict(plot.to_dict()).animation.backdrop is False


def test_shared_axes_drop_the_redundant_inner_tick_labels():
    """Removing the repetition is the main reason to ask for shared axes in a paper.

    ``sharex=`` links the *limits*; only ``plt.subplots`` hides the labels, and
    this renderer builds axes one at a time so a 3-D panel can sit beside a 2-D
    one.
    """
    traj = _lorenz()
    grid = viz.grid(
        ts.plot(traj, components="x"),
        ts.plot(traj, components="z"),
        rows=2,
        cols=1,
        share_x=True,
    )
    axes = grid.fig.axes
    assert axes[0].get_xlabel() == "" and axes[1].get_xlabel() == "t"
    assert axes[0].xaxis.get_tick_params()["labelbottom"] is False
    assert axes[1].xaxis.get_tick_params()["labelbottom"] is True


def test_a_shared_colorbar_spans_the_panels_instead_of_sitting_inside_one():
    """ "One figure-level colorbar" landed between panels 1 and 2 of a 2x2 top row."""
    vdp = ts.systems.VanDerPol()
    panels = [
        ts.plot(vdp.with_params(mu=mu), ("flow_speed", {"log": True}), grid=16)
        for mu in (0.5, 1.0, 2.0, 4.0)
    ]
    grid = viz.grid(*panels, rows=2, cols=2, share_color=True)
    figure = grid.fig
    bars = [ax for ax in figure.axes if ax.get_label() == "<colorbar>"]
    assert len(bars) == 1
    panel_axes = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    right_edge = max(ax.get_position().x1 for ax in panel_axes)
    assert bars[0].get_position().x0 >= right_edge


def test_a_colorbar_matches_the_height_of_an_aspect_locked_image():
    """On a wide, short basin region the bar was ~3x the height of the picture.

    ``Figure.colorbar(..., ax=ax)`` sizes from the axes' *rectangle*; an
    equal-aspect image draws inside a smaller box than its rectangle.
    """
    henon = ts.systems.Henon()
    result = ts.analysis.basins(henon, [(-3.0, 3.0, 24), (-0.6, 0.6, 12)])
    figure = ts.plot(result).fig
    figure.canvas.draw()
    image_ax = figure.axes[0]
    # The bar is an INSET of the image (axes coordinates are the drawn box), so
    # it is a child axes rather than a figure axes — which is what makes it track
    # the picture under constrained layout.
    assert len(image_ax.child_axes) == 1
    bar_ax = image_ax.child_axes[0]
    drawn = image_ax.get_window_extent().height
    assert bar_ax.get_window_extent().height == pytest.approx(drawn, rel=0.02)
    assert bar_ax.get_window_extent().width < 0.1 * image_ax.get_window_extent().width


# ---------------------------------------------------------------------------
# 3. The library knowing something and not saying it
# ---------------------------------------------------------------------------


def test_a_field_is_computed_over_every_orbit_drawn_on_it():
    """Two of five orbits fell entirely outside the axes — and stayed in the legend.

    The field chose its window from the *system* alone and, as the overlay's
    base spec, imposed it on the figure.  A figure asserting that a curve is
    somewhere it is not.
    """
    vdp = ts.systems.VanDerPol()
    orbits = [vdp.run(final_time=8.0, dt=0.01, ic=[a, 0.0]) for a in (0.5, 2.0, 4.0)]
    plot = ts.plot(vdp, *orbits, "vector_field")
    lo, hi = plot.x.limits
    for layer in plot.layers:
        x = np.asarray(layer.data["x"], dtype=float)
        assert lo <= np.nanmin(x) and np.nanmax(x) <= hi, layer.label


def test_a_system_drawn_alone_keeps_its_own_auto_window():
    """The union rule must not move a picture that has no data to union with."""
    vdp = ts.systems.VanDerPol()
    alone = ts.plot(vdp, "vector_field")
    x = np.asarray(alone.layers[0].data["x"], dtype=float)
    assert float(x.min()) < -2.0 and float(x.max()) > 2.0


def test_share_color_says_so_when_one_linear_scale_flattens_the_panels():
    """Honest and hazardous: unifying |f| over a 100x spread made 3 of 4 panels black.

    The unification destroyed the very comparison it was asked for.  A log norm
    is the fix and the caller has to choose it, so this says so.
    """
    vdp = ts.systems.VanDerPol()
    panels = [ts.plot(vdp.with_params(mu=mu), "flow_speed", grid=16) for mu in (0.5, 2.0, 4.0)]
    with pytest.warns(VisualizationDegraded, match="log scale"):
        viz.plot(*panels, layout="row", share_color=True)


def test_share_color_is_silent_when_the_panels_are_already_comparable():
    """A warning on every shared colorbar would be noise."""
    vdp = ts.systems.VanDerPol()
    panels = [
        ts.plot(vdp.with_params(mu=mu), ("flow_speed", {"log": True}), grid=16) for mu in (0.5, 2.0)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        viz.plot(*panels, layout="row", share_color=True)


def test_a_recurrence_plot_says_so_when_it_has_more_markers_than_pixels():
    """A marker-per-recurrence RECURRENCE_PLOT warns once it cannot fit.

    The shipped route no longer builds one — ``RecurrenceMatrix`` draws a
    density image, which is what made the near-solid black square go away — so
    the scatter is built here directly.  The guard stays because it is keyed on
    the *kind*, not on that one route: a caller who draws a recurrence plot with
    ``primitive="points"``, or a third-party transform that registers one, lands
    in exactly the same place and must be told rather than handed the black box.
    """
    from tsdynamics.analysis import _plotbuilder as pb

    traj = ts.systems.Lorenz().run(final_time=30.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    result = ts.analysis.recurrence_matrix(traj, recurrence_rate=0.05)
    coo = result.matrix.tocoo()
    spec = pb.spec(
        None,
        "recurrence_plot",
        layers=[pb.scatter(np.asarray(coo.row, float), np.asarray(coo.col, float))],
        aspect="equal",
    )
    with pytest.warns(VisualizationDegraded, match="overlap"):
        _ = spec.fig


def test_a_large_recurrence_plot_draws_its_true_density_not_a_black_square():
    """The picture's ink must be the matrix's recurrence rate, at any length.

    Measured on a 1501x1501 matrix at a verified 5% density: the old
    marker-per-recurrence scatter inked 14.4% of the canvas — 3.3x the truth,
    and rising with the record length until it is solid — while the binned
    density image inks 4.4%, which is the answer.
    """
    traj = ts.systems.Lorenz().run(final_time=30.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    result = ts.analysis.recurrence_matrix(traj, recurrence_rate=0.05)
    field, side = result._display_field()
    assert result.size > side, "this record is meant to exceed the display cap"
    # Every recurrence is accounted for EXACTLY: a pixel is the local density,
    # so multiplying each back by the cells it covers recovers the stored count.
    per_axis = np.bincount((np.arange(result.size) * side) // result.size, minlength=side).astype(
        float
    )
    recovered = float((field * np.outer(per_axis, per_axis)).sum())
    assert recovered == pytest.approx(float(result.matrix.tocoo().nnz), rel=1e-12)
    # And the ink is the answer: the mean pixel is the recurrence rate.  It is a
    # mean of RATIOS, and the bins are not all the same width when the display
    # side does not divide the record length, so it lands near rather than on it.
    assert field.mean() == pytest.approx(float(result.recurrence_rate), rel=1e-2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        _ = ts.plot(result).fig


def test_a_recurrence_plot_that_fits_is_the_exact_matrix():
    """Below the display cap the field is the 0/1 matrix, bit-for-bit."""
    traj = ts.systems.Lorenz().run(final_time=6.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    result = ts.analysis.recurrence_matrix(traj, recurrence_rate=0.05)
    field, side = result._display_field()
    assert side == result.size
    assert np.array_equal(field, np.asarray(result.matrix.todense(), dtype=float))


def test_an_orbit_diagram_labels_the_observable_not_the_slice_it_was_sampled_on():
    """A flow's cascade labelled its y axis ``Poincaré section (1, 0.0)``.

    The plane *spec* — where the points were sampled — not what is plotted, and
    identical for two different component choices.
    """
    rossler = ts.systems.Rossler()
    section = rossler.poincare("y", 0.0, direction="up")
    plot = ts.plot(section, "orbit_diagram", param="c", values=(4.0, 6.0, 6), points=8)
    assert plot.y.label == "x at y = 0"


def test_a_maps_orbit_diagram_names_its_own_variable():
    """A map has no slice, so the label is just the observable."""
    logistic = ts.systems.Logistic()
    plot = ts.plot(logistic, "orbit_diagram", param="r", values=(3.4, 4.0, 10), points=10)
    assert plot.y.label == "x"


def test_a_cobweb_can_be_drawn_over_the_maps_own_domain():
    """The hump is the whole point, and a converging orbit never visits it.

    The curve was clipped to the orbit's visited span, so the picture whose
    point is *the diagonal crosses the hump* routinely omitted the hump.  A 1-D
    map has no domain the library can read off it, so it is a keyword.
    """
    orbit = ts.systems.Logistic(r=2.8).run(steps=30, ic=[0.1])
    plot = ts.plot(orbit, "cobweb", domain=(0.0, 1.0))
    graph = next(layer for layer in plot.layers if layer.label == "f(x)")
    x = np.asarray(graph.data["x"], dtype=float)
    y = np.asarray(graph.data["y"], dtype=float)
    assert float(x.min()) == 0.0 and float(x.max()) == 1.0
    assert float(y.max()) == pytest.approx(0.7, abs=1e-3)  # the hump, r/4
    assert plot.x.limits == (0.0, 1.0)


def test_a_cobweb_domain_that_is_not_an_interval_is_refused():
    """A reversed pair would silently draw nothing."""
    orbit = ts.systems.Logistic(r=2.8).run(steps=10, ic=[0.1])
    with pytest.raises(InvalidParameterError, match=r"domain=\(0, 1\)"):
        ts.plot(orbit, "cobweb", domain=(1.0, 0.0))


def test_a_delay_embedding_names_the_channel_it_embedded():
    """It said ``x(t)`` for every input, including a channel called something else."""
    traj = _lorenz(final_time=20.0, dt=0.01)
    assert ts.plot(traj, "delay_embedding", delay=16, components="z").y.label == "z(t - 0.16)"
    named = ts.Trajectory(traj.t, traj["x"][:, None], meta={"variables": ("voltage",)})
    assert ts.plot(named, "delay_embedding", delay=16).x.label == "voltage(t)"


def test_a_delay_embeddings_lag_is_stated_in_the_unit_it_is_in():
    """``t - 16`` on an axis whose ``t`` is a time read as a time; the lag is samples."""
    bare = np.sin(np.linspace(0.0, 40.0, 2000))
    assert ts.plot(bare, "delay_embedding", delay=16).y.label == "x(t - 16 samples)"


def test_a_dense_marker_cloud_is_sized_so_it_does_not_overplot():
    """The theme's constant marker draws a hundred points well and a hundred thousand as a blob."""
    traj = ts.systems.Lorenz().run(final_time=200.0, dt=0.005, ic=[1.0, 1.0, 1.0])
    dense = viz.draw({"x": traj["x"], "y": traj["z"]}, "points").fig
    sparse = viz.draw({"x": traj["x"][:500], "y": traj["z"][:500]}, "points").fig
    big = {float(np.ravel(c.get_sizes())[0]) for ax in dense.axes for c in ax.collections}
    small = {float(np.ravel(c.get_sizes())[0]) for ax in sparse.axes for c in ax.collections}
    assert max(big) < min(small)  # ...and below the pivot nothing moved
    assert small == {36.0}


# ---------------------------------------------------------------------------
# Discovery: one verb, one question, one quality of answer
# ---------------------------------------------------------------------------


def test_transform_find_prints_a_table_and_is_still_a_list():
    """``ts.analysis.find`` gives a grouped table; this gave 21 bare strings.

    Same verb, same question.  It must stay a ``list[str]`` so nothing that
    consumed the old return value changes.
    """
    traj = _lorenz(final_time=2.0, dt=0.05)
    found = viz.transforms.find(subject=traj)
    assert isinstance(found, list)
    assert found == sorted(found)
    assert "phase_portrait" in found
    text = repr(found)
    assert "FROM DATA" in text and "ts.plot(subject" in text
    assert viz.transforms.get("psd").doc in text


def test_transform_find_with_no_match_says_what_to_do_next():
    """A bare ``[]`` is the least useful possible answer to "what can I draw?"."""
    assert "compatibility" in repr(viz.transforms.find("definitely-not-a-transform"))


def test_the_analysis_spelling_of_a_field_transform_is_an_alias_not_a_second_row():
    """One concept must not need two words depending on which door you are at."""
    for plot_name, analysis_name in (
        ("ftle", "ftle_field"),
        ("escape_time", "escape_time_field"),
        ("transient_time", "transient_time_field"),
    ):
        assert viz.transforms.get(analysis_name).name == plot_name
        assert analysis_name not in viz.transforms.names()  # one record, one matrix row


def test_the_frames_layout_is_documented_where_the_layout_keyword_is():
    """The most impressive thing in the library was findable only by typing a wrong word."""
    import inspect

    doc = inspect.getdoc(ts.plot) or ""
    assert '"frames"' in doc and "layout=" in doc


def test_a_layer_repr_describes_the_layer_instead_of_dumping_its_arrays():
    """``p.layers`` printed ~9 000 characters of floats on a 200-point trajectory.

    Introspecting a plot is the first thing anyone does when a picture looks
    wrong, and the generated dataclass repr made that step unusable.
    """
    plot = ts.plot(_lorenz(final_time=2.0))
    text = repr(plot.layers)
    assert len(text) < 200
    assert "Layer(line3d" in text and "n=" in text
    assert plot.layers[0].data["x"].size > 50  # the arrays are still there


def test_every_figure_keyword_survives_a_render_not_merely_a_build(tmp_path):
    """The gate the two crashing keywords needed: BUILD and DRAW each of the 17."""
    from tsdynamics.viz.spec import FIGURE_KEYS

    values: dict[str, object] = {
        "title": "t",
        "xlabel": "x",
        "ylabel": "y",
        "zlabel": "z",
        "xlim": (-1.0, 1.0),
        "ylim": (-1.0, 1.0),
        "zlim": (-1.0, 1.0),
        "xscale": "linear",
        "yscale": "linear",
        "zscale": "linear",
        "xticks": [-1.0, 0.0, 1.0],
        "yticks": [-1.0, 0.0, 1.0],
        "zticks": [-1.0, 0.0, 1.0],
        "clim": (0.0, 1.0),
        "colorbar": "right",
        "legend": "upper left",
        "theme": "dark",
    }
    assert set(values) == set(FIGURE_KEYS)
    traj = _lorenz(final_time=4.0)
    for key, value in values.items():
        plot = ts.plot(traj, color_by="time", **{key: value})
        assert plot.kind is PlotKind.PHASE_PORTRAIT_3D
        plot.save(tmp_path / f"{key}.png")


# ---------------------------------------------------------------------------
# 4. One vocabulary, one window, one whole map
#
# The second pass over the same six sessions.  Everything here is a place where
# the library accepted TWO spellings of one thing and quietly meant different
# pictures by them, or drew a frame that left out what the picture is for.
# ---------------------------------------------------------------------------


def test_a_picture_is_named_one_way_and_kind_says_which():
    """``kind=`` and the positional name were both accepted and BUILT DIFFERENT SPECS.

    Measured at v6 round 7: ``ts.plot(tr, kind="time_series")`` and
    ``ts.plot(tr, "time_series")`` both returned a ``Plot``, neither warned, and
    ``a.to_dict() != b.to_dict()`` — the ``kind=`` route produced a spec with no
    ``frame`` (so it could not be frame-checked for an overlay) and a y axis
    labelled with the *first* component of a three-component plot.
    """
    traj = _lorenz(final_time=4.0)
    with pytest.raises(InvalidParameterError) as excinfo:
        ts.plot(traj, kind="time_series")
    message = str(excinfo.value)
    assert "kind=" in message and "ts.plot(subject, 'time_series')" in message
    assert ts.plot(traj, "time_series").frame is not None  # the spelling that stays


@pytest.mark.parametrize(
    ("value", "transform"),
    [
        ("delay", "delay_embedding"),
        ("field", "spatial_field"),
        ("phase_portrait_3d", "phase_portrait"),
        ("spacetime", "spacetime"),
        ("recurrence_plot", "recurrence"),
    ],
)
def test_everything_kind_uniquely_served_survives_under_one_spelling(value, transform):
    """Including the two *recipes* that were never ``PlotKind`` members."""
    traj = _lorenz(final_time=4.0)
    with pytest.raises(InvalidParameterError, match=re.escape(f"{transform!r})")):
        ts.plot(traj, kind=value)
    assert transform in viz.transforms.names()


def test_the_delay_recipes_options_are_the_transforms_options():
    """``kind="delay", delay_time=17`` had to keep working as ``"delay_embedding"``."""
    traj = _lorenz(final_time=20.0, dt=0.01)
    assert ts.plot(traj, "delay_embedding", delay_time=0.16, components="z").y.label == (
        "z(t - 0.16)"
    )


def test_a_field_has_its_own_word_for_the_box_it_is_evaluated_over():
    """``xlim`` meant two things — axis limits, and where to evaluate the equations.

    *"Cost me more time than everything else combined, and it produced a
    plausible wrong picture."*  ``domain=`` is the transform's word, read the
    way every ``region=`` in the library is read: one ``(lo, hi)`` pair per axis.
    """
    vdp = ts.systems.VanDerPol()
    plot = ts.plot(vdp, "flow_speed", domain=((-3.0, 3.0), (-8.0, 8.0)))
    field_x = np.asarray(plot.layers[0].data["x"], dtype=float)
    field_y = np.asarray(plot.layers[0].data["y"], dtype=float)
    assert (float(field_x.min()), float(field_x.max())) == (-3.0, 3.0)
    assert (float(field_y.min()), float(field_y.max())) == (-8.0, 8.0)
    assert plot.x.limits == (-3.0, 3.0) and plot.y.limits == (-8.0, 8.0)


def test_the_new_window_word_reaches_the_arrays_door_too():
    """One word at every door, or it is not one word."""
    geom = viz.geometry(ts.systems.VanDerPol(), "flow_speed", domain=((-1.0, 1.0), (-2.0, 2.0)))
    assert geom.axis_limits == ((-1.0, 1.0), (-2.0, 2.0))


def test_a_square_domain_may_be_written_once():
    """``domain=(0, 1)`` is the unit square — the shape a caller reaches for."""
    plot = ts.plot(ts.systems.VanDerPol(), "flow_speed", domain=(-2.0, 2.0))
    assert plot.x.limits == (-2.0, 2.0) and plot.y.limits == (-2.0, 2.0)


@pytest.mark.parametrize(
    "bad", [(1.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), "nope", ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))]
)
def test_a_malformed_domain_is_refused_with_the_shape_it_wanted(bad):
    with pytest.raises(InvalidParameterError, match="one .lo, hi. pair per axis"):
        ts.plot(ts.systems.VanDerPol(), "flow_speed", domain=bad)


def test_the_field_box_and_the_axes_may_be_named_separately_in_one_call():
    """The two words do two jobs, so both at once is an ordinary figure.

    It used to raise *"two spellings of one box in one call"* — while the
    ambiguity warning next door was, in the same breath, teaching ``domain=`` as
    the evaluation window and ``xlim=`` as "only the axes".  A beta tester read
    that as an invitation, took it, and was refused: field over a big box with
    the axes zoomed into part of it is a completely ordinary panel.
    """
    plot = ts.plot(
        ts.systems.VanDerPol(),
        "flow_speed",
        domain=((-6.0, 6.0), (-6.0, 6.0)),
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
    )
    assert plot.x.limits == (-2.0, 2.0)
    # ...and the field really was evaluated over the big box, not the small one.
    xs = np.concatenate([np.asarray(layer.data["x"]).ravel() for layer in plot.layers])
    assert xs.max() > 2.5


def test_naming_the_box_twice_inside_one_transform_call_is_refused():
    """Written into ONE transform's own options they really are two spellings."""
    with pytest.raises(InvalidParameterError, match="two spellings of one box"):
        ts.plot(
            ts.systems.VanDerPol(),
            ("flow_speed", {"domain": ((-1.0, 1.0), (-1.0, 1.0)), "xlim": (-2.0, 2.0)}),
        )


def test_a_window_doing_two_jobs_says_so_and_names_the_word_for_one_of_them():
    """The collision is gone as a *defect* and kept as a convenience — out loud."""
    with pytest.warns(VisualizationDegraded, match="domain="):
        ts.plot(ts.systems.VanDerPol(), "flow_speed", xlim=(-3.0, 3.0))


def test_an_axis_limit_on_a_plain_trajectory_is_silent():
    """Nothing collides when no transform wanted the window: that must stay quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts.plot(_lorenz(final_time=4.0), "time_series", xlim=(0.0, 1.0))


def test_a_cobweb_draws_the_whole_map_not_the_orbits_bounding_box():
    """The picture whose entire point is *the diagonal crosses the hump* omitted the hump.

    Measured: ``Logistic(r=2.8).run(steps=40, ic=[0.6])`` converges onto the
    fixed point, so the orbit's bounding box is ``[0.6, 0.643]`` — ``f`` was
    drawn over *that*, and neither the critical point nor the second fixed point
    was on the canvas.  A 1-D map does not declare a domain but it **is** one,
    and it can be read off the kernel: the unit square, for every ``r``.
    """
    orbit = ts.systems.Logistic(r=2.8).run(steps=40, ic=[0.6])
    plot = ts.plot(orbit, "cobweb")
    visited = np.asarray(orbit.y[:, 0], dtype=float)
    assert float(visited.max()) < 0.7  # the orbit never goes near the hump...
    lo, hi = plot.x.limits
    assert lo == pytest.approx(0.0, abs=1e-3) and hi == pytest.approx(1.0, abs=1e-3)
    assert plot.y.limits == plot.x.limits  # ...and the box is square
    graph = next(layer for layer in plot.layers if layer.label == "f(x)")
    x = np.asarray(graph.data["x"], dtype=float)
    y = np.asarray(graph.data["y"], dtype=float)
    assert (float(x.min()), float(x.max())) == (lo, hi)  # f spans the AXIS range
    assert float(y.max()) == pytest.approx(0.7, abs=1e-3)  # the hump, r/4


@pytest.mark.parametrize("r", [2.8, 3.2, 3.6, 3.9])
def test_the_logistic_cobweb_is_the_unit_square_whatever_the_orbit_did(r):
    plot = ts.plot(ts.systems.Logistic(r=r).run(steps=40, ic=[0.6]), "cobweb")
    lo, hi = plot.x.limits
    assert lo == pytest.approx(0.0, abs=1e-3) and hi == pytest.approx(1.0, abs=1e-3)


def test_a_bare_series_cobweb_still_keeps_the_orbits_span():
    """No kernel, no domain to read — and inventing one would be a made-up answer."""
    geom = viz.geometry(np.array([0.2, 0.5, 0.75, 0.6, 0.7]), "cobweb")
    assert geom.axis_limits == ((pytest.approx(0.2), pytest.approx(0.75)),) * 2
    assert not any(part.label == "f(x)" for part in geom.parts)


def test_a_field_is_computed_over_orbits_a_named_transform_also_claimed():
    """The union was taken over the subjects no transform claimed — and no others.

    ``ts.plot(vdp, t1, t2, "vector_field", "phase_portrait")`` names a transform
    for the orbits too, so they contributed nothing to the field's window: the
    field came out on ``[-3, 3]`` and the orbit reaching ``x = 4.2`` ran off the
    canvas while staying in the legend.
    """
    vdp = ts.systems.VanDerPol()
    far = vdp.run(final_time=8.0, dt=0.01, ic=[4.0, 4.0])
    plot = ts.plot(vdp, far, "vector_field", "phase_portrait")
    lo, hi = plot.x.limits
    for layer in plot.layers:
        x = np.asarray(layer.data["x"], dtype=float)
        assert lo <= np.nanmin(x) and np.nanmax(x) <= hi, layer.label


def test_a_legended_curve_that_is_entirely_off_screen_says_so():
    """A legend entry is a claim that the curve is in this picture."""
    vdp = ts.systems.VanDerPol()
    a = vdp.run(final_time=8.0, dt=0.01, ic=[0.5, 0.0])
    b = vdp.run(final_time=8.0, dt=0.01, ic=[4.0, 4.0])
    with pytest.warns(VisualizationDegraded, match="entirely outside the axes"):
        ts.plot(a, b, "phase_portrait", xlim=(10.0, 20.0))


def test_a_reference_line_outside_the_data_scale_is_not_an_off_screen_curve():
    """``lambda = 0`` is a datum a transform draws to be measured against."""
    t = np.linspace(1.0, 50.0, 200)
    est = 1.0 + 8.0 * np.exp(-t / 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts.plot((t, est), "lyapunov_convergence")


def test_two_runs_of_one_system_are_legended_by_the_parameter_that_differs():
    """``VanDerPol (1)`` / ``VanDerPol (2)`` names argument slots, not dynamics.

    Overlaying two parameter values is the commonest figure in this field; the
    values were in ``traj.meta["params"]`` the whole time.  The system's name —
    which both curves share — stays in the title, where it belongs.
    """
    vdp = ts.systems.VanDerPol()
    a = vdp.with_params(mu=1.0).run(final_time=20.0, dt=0.02, ic=[0.1, 0.1])
    b = vdp.with_params(mu=3.0).run(final_time=20.0, dt=0.02, ic=[0.1, 0.1])
    plot = ts.plot(a, b)
    assert [layer.label for layer in plot.layers] == ["mu = 1", "mu = 3"]
    assert plot.title == "VanDerPol"


def test_the_parameter_legend_only_fires_when_it_can_name_every_curve_apart():
    """Same parameters, different starts: ``(1)`` / ``(2)`` is the honest answer."""
    vdp = ts.systems.VanDerPol()
    a = vdp.run(final_time=6.0, dt=0.05, ic=[0.1, 0.1])
    b = vdp.run(final_time=6.0, dt=0.05, ic=[0.5, 0.5])
    assert [layer.label for layer in ts.plot(a, b).layers] == ["VanDerPol (1)", "VanDerPol (2)"]


def test_an_explicit_label_still_wins_over_the_parameter_it_would_have_used():
    vdp = ts.systems.VanDerPol()
    a = vdp.with_params(mu=1.0).run(final_time=6.0, dt=0.05, ic=[0.1, 0.1])
    b = vdp.with_params(mu=3.0).run(final_time=6.0, dt=0.05, ic=[0.1, 0.1])
    assert [layer.label for layer in ts.plot(a, b, labels=["A", "B"]).layers] == ["A", "B"]


def test_a_recorded_cascade_names_its_observable_at_every_door():
    """``ts.plot(od, "orbit_diagram")`` labelled the y axis ``x0`` — a made-up index name.

    Identical for ``components=0`` and ``components="z"``, which is exactly the
    defect the same label was fixed for at the system door: two different
    pictures, one wrong caption.
    """
    rossler = ts.systems.Rossler()
    section = rossler.poincare("y", 0.0, direction="up")
    values = np.linspace(4.0, 6.0, 5)
    for observable, expected in ((None, "x"), ("z", "z")):
        kw = {} if observable is None else {"components": observable}
        recorded = ts.analysis.orbit_diagram(
            section, "c", values, points_per_value=6, transient=80, **kw
        )
        assert ts.plot(recorded, "orbit_diagram").y.label == expected
        assert ts.plot(recorded).y.label == expected


# ---------------------------------------------------------------------------
# A refused animation knob must leave the plot exactly as it found it
# ---------------------------------------------------------------------------


#: Every animation knob a tester abused, with the value that must be refused.
#: Eleven of the twelve were already refused by name — the defect was WHERE the
#: refusal happened relative to the write.
_BAD_ANIMATION_KNOBS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("animate", "fps", {"fps": -5}),
    ("animate", "duration", {"duration": "2s"}),
    ("animate", "n_frames", {"n_frames": 0}),
    ("animate", "loop", {"loop": "yes"}),
    ("animate", "pingpong", {"pingpong": "sure"}),
    ("animate", "mode", {"mode": "typo"}),
    ("trail", "length-unit", {"length": ("whatever", 3.0)}),
    ("trail", "length-bare", {"length": 3.0}),
    ("trail", "fade", {"fade": "lots"}),
    ("trail", "backdrop", {"backdrop": "on"}),
    ("trail", "backdrop_alpha", {"backdrop_alpha": 3}),
    ("head", "symbol", {"symbol": "banana"}),
    ("head", "show", {"show": "yes"}),
    ("head", "size", {"size": -1}),
    ("camera", "spin", {"spin": "fast"}),
)


def _animated_plot():
    traj = ts.systems.Lorenz().run(final_time=4.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    return ts.plot(traj, animate=True)


@pytest.mark.parametrize(
    ("method", "knob", "kwargs"),
    _BAD_ANIMATION_KNOBS,
    ids=[f"{m}-{k}" for m, k, _ in _BAD_ANIMATION_KNOBS],
)
def test_a_refused_animation_value_is_never_written(method, knob, kwargs):
    """Measured before the fix: ``p.animate(fps=-5)`` raised AND stored ``-5``.

    Every later tweak on that plot then re-raised the *old* error, naming a knob
    the caller had not touched — so a sweep of twelve knobs reported three
    results about the wrong argument, and in an interactive session one bad
    value made the plot permanently unusable with misleading messages.

    The messages themselves were already good; they were attached to the wrong
    call.  A refusal must be a no-op.
    """
    plot = _animated_plot()
    before = plot.animation.to_dict()
    with pytest.raises(ts.errors.InvalidParameterError):
        getattr(plot, method)(**kwargs)
    assert plot.animation.to_dict() == before


@pytest.mark.parametrize(
    ("method", "knob", "kwargs"),
    _BAD_ANIMATION_KNOBS,
    ids=[f"{m}-{k}" for m, k, _ in _BAD_ANIMATION_KNOBS],
)
def test_the_plot_still_takes_every_other_knob_after_one_was_refused(method, knob, kwargs):
    """The tail of the same bug: the plot must stay usable, and stay honest."""
    plot = _animated_plot()
    with pytest.raises(ts.errors.InvalidParameterError):
        getattr(plot, method)(**kwargs)
    plot.animate(fps=24, loop=False).trail(("steps", 40)).head(symbol="s").camera(spin=1.0)
    assert plot.animation.fps == 24.0
    assert plot.animation.loop is False
    assert (plot.animation.trail_kind, plot.animation.trail_length) == ("steps", 40.0)
    assert plot.animation.head_symbol == "s"
    assert plot.animation.spin == 1.0


def test_a_switch_that_is_not_a_switch_is_refused_and_names_the_two_values():
    """``loop='yes'`` and ``fade='lots'`` were ACCEPTED as truthy strings.

    They sat beside eight knobs that validate strictly, so a silent accept read
    as "that was a valid value" — the inconsistency is what misled, not the
    permissiveness on its own.
    """
    plot = _animated_plot()
    with pytest.raises(ts.errors.InvalidParameterError, match="True or False"):
        plot.animate(loop="yes")
    with pytest.raises(ts.errors.InvalidParameterError, match="True or False"):
        plot.trail(fade="lots")
    # The real values still work, False included (it is not "unset").
    assert plot.animate(loop=False).animation.loop is False
    assert plot.trail(fade=True).animation.trail_fade is True


def test_a_duration_that_is_not_a_number_names_the_knob_and_its_unit():
    """It was the one knob of eleven with no typed message.

    ``ValueError: could not convert string to float: '2s'`` named neither
    ``duration``, nor the unit, nor a remedy.
    """
    plot = _animated_plot()
    with pytest.raises(ts.errors.InvalidParameterError) as excinfo:
        plot.animate(duration="2s")
    message = str(excinfo.value)
    assert "duration" in message
    assert "seconds" in message
    assert "could not convert string to float" not in message
