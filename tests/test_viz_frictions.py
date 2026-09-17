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
    plot = ts.plot(_lorenz(), labels=["run A"], kind="time_series", components=["x", "y"])
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
    the plot door did not — two doors, one word, two pictures, no warning.
    """
    vdp = ts.systems.VanDerPol()
    plot = ts.plot(vdp, "flow_speed", ylim=(-8.0, 8.0))
    field_y = np.asarray(plot.layers[0].data["y"], dtype=float)
    assert (float(field_y.min()), float(field_y.max())) == (-8.0, 8.0)
    assert plot.y.limits == (-8.0, 8.0)


def test_a_transforms_own_window_still_wins_over_the_shared_one():
    """The escape hatch stays exact: computing over one box, drawing over another."""
    vdp = ts.systems.VanDerPol()
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
