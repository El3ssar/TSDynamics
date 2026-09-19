"""The space-filling-curve transforms (``viz/transforms/hilbert.py``).

Three things are worth testing here and one of them is unusual.

The unusual one: a Hilbert plot's **pixel-to-sample map** can be wrong with *no
symptom at all* — the image still renders, and only a hover read-out or an
annotation is a lie.  So the map is checked against the image it describes, for
every curve family, fit and granularity this module can produce, and the guard
that is supposed to catch a disagreement is itself provoked into firing.

The ordinary two: the optional dependency behaves (absent means a clear error
naming the extra, **never** a silent substitution), and the pictures are right —
which for a locality plot means the *measurement* is pinned, not an adjective.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import tsdynamics as ts  # noqa: E402
from tsdynamics.analysis._result_viz import VisualizationNotInstalled  # noqa: E402
from tsdynamics.errors import BackendError, InvalidParameterError  # noqa: E402
from tsdynamics.viz._frames import FrameSpace  # noqa: E402
from tsdynamics.viz.transforms import build_spec, compatibility, geometry, get  # noqa: E402
from tsdynamics.viz.transforms import hilbert as hp  # noqa: E402

HILBERT_TRANSFORMS = ("hilbert", "hilbert_fourier", "hilbert_difference", "hilbert_labels")

#: Skips the tests that need the forty real curves.  The in-tree orderings are
#: always exercised, so this file is meaningful with or without the extra.
needs_extra = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("hilbertplot") is None,
    reason="the optional hilbertplot extra is not installed",
)


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    """A deterministic quasi-periodic record — 1024 samples (a full 32x32 grid)."""
    t = np.linspace(0.0, 60.0, 1024)
    return np.sin(t) + 0.4 * np.sin(np.sqrt(3.0) * t)


@pytest.fixture(scope="module")
def lorenz():
    """A pinned Lorenz orbit (5,001 samples) — a real, non-synthetic subject."""
    return ts.systems.Lorenz().run(final_time=50.0, dt=0.01, ic=[1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", HILBERT_TRANSFORMS)
def test_each_view_is_a_registered_data_transform_on_a_lattice(name):
    record = get(name)
    assert record.source == "data"
    assert record.frame == (FrameSpace.GRID2,)
    assert record.default_primitive == "image"
    assert name in compatibility()
    assert record.doc


def test_the_row_is_listed_whether_or_not_the_extra_is_installed():
    """A silently-absent row reads as "this plot does not exist"; these always show.

    The transforms declare **no** ``requires``, and that is deliberate: they draw
    with the in-tree orderings on any machine, so the extra genuinely is an
    upgrade rather than a gate.  What needs it is the *Hilbert curve*, and that
    is reported per call, naming the extra.
    """
    for name in HILBERT_TRANSFORMS:
        record = get(name)
        assert record.requires is None
        assert record.available is True


def test_curve_names_always_offers_the_in_tree_orderings():
    names = hp.curve_names()
    assert set(hp.ORDERINGS) <= set(names)
    assert names[: len(hp.ORDERINGS)] == tuple(sorted(hp.ORDERINGS))


@needs_extra
def test_curve_names_grows_to_the_forty_curves_when_the_extra_is_installed():
    import hilbertplot

    names = hp.curve_names()
    assert len(names) == len(hp.ORDERINGS) + 40
    assert "Hilbert" in names and "Moore" in names
    assert len(hilbertplot.catalog()) == 40


# ---------------------------------------------------------------------------
# The optional dependency: an error, never a substitution
# ---------------------------------------------------------------------------


@pytest.fixture
def without_hilbertplot(monkeypatch):
    """Make ``import hilbertplot`` fail, whether or not it is installed."""
    monkeypatch.setitem(sys.modules, "hilbertplot", None)
    return None


def test_a_hilbert_curve_without_the_extra_raises_naming_it(without_hilbertplot, series):
    with pytest.raises(VisualizationNotInstalled) as excinfo:
        geometry(series, "hilbert", curve="Hilbert")
    message = str(excinfo.value)
    assert "hilbertplot" in message
    assert "tsdynamics[hilbert]" in message
    assert "rowmajor" in message, "the error must name the alternatives it did not take"
    assert "not substituted silently" in message


def test_the_in_tree_orderings_still_draw_without_the_extra(without_hilbertplot, series):
    """The plot exists on any machine; only the *Hilbert curve* needs the extra."""
    for curve in sorted(hp.ORDERINGS):
        geom = geometry(series, "hilbert", curve=curve)
        assert geom.meta["curve_source"] == "in-tree"
        assert geom.channels["z"].values.shape == (32, 32)


def test_curve_names_reports_only_the_in_tree_orderings_without_the_extra(without_hilbertplot):
    assert hp.curve_names() == tuple(sorted(hp.ORDERINGS))


def test_an_unknown_curve_name_raises_rather_than_falling_back(series):
    with pytest.raises((InvalidParameterError, VisualizationNotInstalled)):
        geometry(series, "hilbert", curve="definitely-not-a-curve")


# ---------------------------------------------------------------------------
# The grid-sizing rule — stated here, not inferred from another package
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "fit", "any_side", "expected"),
    [
        (1024, "square", True, 32),
        (1000, "square", True, 32),
        (1001, "square", True, 32),
        (1025, "square", True, 33),
        (1000, "pad", False, 32),
        (1025, "pad", False, 64),
        (1000, "truncate", False, 16),
        (1024, "truncate", False, 32),
        (1, "square", True, 1),
    ],
)
def test_the_grid_side_rule(n, fit, any_side, expected):
    assert hp.grid_side(n, fit, any_side=any_side) == expected


def test_a_power_of_two_ordering_refuses_the_tight_square():
    with pytest.raises(InvalidParameterError, match="fit='square' needs an ordering"):
        hp.grid_side(1000, "square", any_side=False)
    with pytest.raises(InvalidParameterError, match="unknown fit"):
        hp.grid_side(1000, "auto", any_side=True)


def test_morton_needs_a_power_of_two_side(series):
    """It is refused, not silently padded — the grid would be a different plot."""
    with pytest.raises(InvalidParameterError, match="only 2\\*\\*k grids"):
        geometry(series[:1000], "hilbert", curve="morton", fit="square")
    with pytest.raises(InvalidParameterError, match="power-of-two side"):
        hp._ordering_points("morton", 6)


# ---------------------------------------------------------------------------
# The in-tree orderings are real space-filling paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ordering", sorted(hp.ORDERINGS))
def test_every_in_tree_ordering_visits_each_cell_exactly_once(ordering):
    """An ordering that is not a bijection onto the grid would silently lose data."""
    side = 16
    points = hp._ordering_points(ordering, side)
    assert points.shape == (side * side, 2)
    flat = points[:, 1] * side + points[:, 0]
    assert np.array_equal(np.sort(flat), np.arange(side * side))


def test_snake_reverses_alternate_rows_and_rowmajor_does_not():
    rm = hp._ordering_points("rowmajor", 4)
    sn = hp._ordering_points("snake", 4)
    assert np.array_equal(rm[:4, 0], [0, 1, 2, 3])
    assert np.array_equal(sn[:4, 0], [0, 1, 2, 3])
    assert np.array_equal(rm[4:8, 0], [0, 1, 2, 3])
    assert np.array_equal(sn[4:8, 0], [3, 2, 1, 0])


# ---------------------------------------------------------------------------
# The pixel-to-sample map: the failure with no symptom
# ---------------------------------------------------------------------------


def _check_map(values, *, curve, fit=None, granularity=1):
    """Every finite pixel must hold exactly the sample its index map names."""
    image = (
        geometry(values, "hilbert", curve=curve, fit=fit, granularity=granularity)
        .channels["z"]
        .values
    )
    index = hp.sample_index_map(values, curve=curve, fit=fit, granularity=granularity)
    assert index.shape == image.shape
    ys, xs = np.nonzero(np.isfinite(image))
    assert ys.size, "the image is entirely padding"
    expected = hp._granulate(np.asarray(values, dtype=float), granularity)[index[ys, xs]]
    assert np.allclose(image[ys, xs], expected)
    return image, index


@pytest.mark.parametrize("ordering", sorted(hp.ORDERINGS))
@pytest.mark.parametrize("granularity", [1, 8])
def test_the_in_tree_pixel_to_sample_map_is_exact(series, ordering, granularity):
    fit = "square" if ordering != "morton" else "pad"
    _check_map(series, curve=ordering, fit=fit, granularity=granularity)


@needs_extra
@pytest.mark.parametrize("curve", ["Hilbert", "Moore", "Liu1"])
@pytest.mark.parametrize("fit", ["pad", "truncate"])
def test_the_hilbertplot_pixel_to_sample_map_is_exact(series, curve, fit):
    """Checked against the *public* ``label_map``, so no undocumented behaviour is assumed."""
    _check_map(series[:1000], curve=curve, fit=fit)


@needs_extra
def test_the_tight_square_fit_is_exact_for_a_generalizing_curve(series):
    image, _ = _check_map(series[:1000], curve="Hilbert", fit="square")
    assert image.shape == (32, 32)


@needs_extra
@pytest.mark.parametrize("granularity", [1, 4, 16])
def test_the_granularity_transform_agrees_with_hilbertplots(series, granularity):
    """The two coarsenings must be the same function, or the map check is meaningless."""
    import hilbertplot

    ours = hp._granulate(np.asarray(series, dtype=float), granularity)
    theirs = hilbertplot.granulate(series, granularity)
    assert np.allclose(ours, theirs)
    _check_map(series[:1000], curve="Hilbert", granularity=granularity)


@needs_extra
def test_a_layout_that_disagrees_with_this_module_raises_instead_of_drawing(series, monkeypatch):
    """Provoke the guard: if the grid is not the one we computed, refuse to draw.

    This is the only failure mode in the module that would otherwise be silent,
    so the guard is tested by making it fire rather than by trusting it.
    """
    monkeypatch.setattr(hp, "grid_side", lambda n, fit, *, any_side: 999)
    with pytest.raises(BackendError, match="pixel-to-sample map would be wrong"):
        geometry(series[:1000], "hilbert", curve="Hilbert")


def test_the_sample_index_map_stays_off_meta_unless_asked(series):
    """It is as large as the image, and ``meta`` is serialized by ``to_json``."""
    lean = geometry(series, "hilbert", curve="snake")
    assert "sample_index" not in lean.meta
    fat = geometry(series, "hilbert", curve="snake", with_sample_index=True)
    assert fat.meta["sample_index"].shape == fat.channels["z"].values.shape


def test_padding_cells_are_reported_and_drawn_as_nan(series):
    geom = geometry(series[:1000], "hilbert", curve="snake", fit="square")
    z = geom.channels["z"].values
    assert geom.meta["n_padding_cells"] == 32 * 32 - 1000
    assert int(np.isnan(z).sum()) == geom.meta["n_padding_cells"]


# ---------------------------------------------------------------------------
# The pictures are right
# ---------------------------------------------------------------------------


def _locality(curve: str, side: int = 128) -> tuple[float, float, float]:
    """Median / mean / P(<=8) record-distance between 8-adjacent cells."""
    labels = hp.sample_index_map(np.zeros(side * side), curve=curve, fit="pad").astype(float)
    gaps = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            a = labels[max(0, dy) : side + min(0, dy), max(0, dx) : side + min(0, dx)]
            b = labels[max(0, -dy) : side + min(0, -dy), max(0, -dx) : side + min(0, -dx)]
            gaps.append(np.abs(a - b).ravel())
    g = np.concatenate(gaps)
    return float(np.median(g)), float(g.mean()), float((g <= 8).mean())


@pytest.mark.parametrize(
    ("curve", "median", "mean", "p8"),
    [
        ("morton", 5.0, 94.5, 0.656),
        ("snake", 85.5, 96.1, 0.275),
        ("rowmajor", 127.5, 96.1, 0.251),
        pytest.param("Hilbert", 3.0, 114.3, 0.683, marks=needs_extra),
    ],
)
def test_the_locality_claim_is_a_measurement(curve, median, mean, p8):
    """Pin the numbers the module docstring quotes — including the ones against it.

    Hilbert's median gap is 43x better than row-major's and its **mean** is 19%
    *worse*: it buys the median by lengthening its rare worst jumps.  Both live
    in this table so neither can be quietly dropped from the docs.
    """
    got_median, got_mean, got_p8 = _locality(curve)
    assert got_median == pytest.approx(median, abs=0.51)
    assert got_mean == pytest.approx(mean, rel=0.01)
    assert got_p8 == pytest.approx(p8, abs=0.002)


def test_the_difference_map_of_a_raster_layout_is_a_featureless_plateau(series):
    """Row-major has no locality anywhere, and its locality map says exactly that."""
    flat = geometry(series, "hilbert_difference", curve="rowmajor").channels["z"].values
    interior = flat[1:-1, 1:-1]
    assert np.ptp(interior) == pytest.approx(0.0)


@needs_extra
def test_the_difference_map_of_a_hilbert_curve_shows_its_locality_barriers(series):
    """The recursive ridge structure is the whole content of the plot."""
    field = geometry(series, "hilbert_difference", curve="Hilbert").channels["z"].values
    interior = field[1:-1, 1:-1]
    assert interior.std() > 0.05 * interior.mean(), "a Hilbert difference map is not flat"
    assert interior.max() > 20.0 * np.median(interior), "the barriers must stand out"


def test_the_fourier_map_of_a_periodic_record_is_more_structured_than_of_noise():
    """ "Periodicity as symmetry": a periodic record concentrates power, noise does not."""
    n = 4096
    periodic = np.sin(np.linspace(0.0, 2.0 * np.pi * 64.0, n))
    noise = np.random.default_rng(7).standard_normal(n)

    def concentration(values):
        spec = geometry(values, "hilbert_fourier", curve="snake").channels["z"].values
        flat = np.sort(spec.ravel())[::-1]
        return float(flat[: flat.size // 100].sum() / flat.sum())

    assert concentration(periodic) > 1.5 * concentration(noise)


def test_the_visit_order_view_is_the_pixel_to_sample_map(series):
    """``hilbert_labels`` is a view, not a debugging aid — it *is* the map."""
    labels = geometry(series, "hilbert_labels", curve="snake").channels["z"].values
    assert np.array_equal(labels, hp.sample_index_map(series, curve="snake").astype(float))


def test_the_difference_map_does_not_depend_on_the_values(series):
    """It describes the curve and the grid, so two different records must agree."""
    a = geometry(series, "hilbert_difference", curve="snake").channels["z"].values
    b = geometry(series * 3.0 + 1.0, "hilbert_difference", curve="snake").channels["z"].values
    assert np.array_equal(a, b)


def test_granularity_coarsens_rather_than_decimating(series):
    """The *l*-granularity transform keeps the length and lowers the total variation.

    Both halves matter: a *decimation* would shrink the grid (and quietly change
    which sample a pixel is), while a coarsening leaves the layout identical and
    only flattens the record.
    """
    fine = geometry(series, "hilbert", curve="snake").channels["z"].values
    coarse = geometry(series, "hilbert", curve="snake", granularity=16).channels["z"].values
    assert fine.shape == coarse.shape
    assert np.nanmin(coarse) >= np.nanmin(fine) and np.nanmax(coarse) <= np.nanmax(fine)
    # The definition: blocks of l samples replaced by l copies of the block mean.
    granulated = hp._granulate(series, 16)
    assert granulated.size == series.size
    blocks = granulated.reshape(-1, 16)
    assert np.allclose(blocks, blocks[:, :1])
    assert np.allclose(blocks[:, 0], series.reshape(-1, 16).mean(axis=1))
    assert np.unique(granulated).size == series.size // 16


# ---------------------------------------------------------------------------
# Which TSDynamics series can feed it
# ---------------------------------------------------------------------------


def test_a_trajectory_component_lays_out_on_a_curve(lorenz):
    geom = geometry(lorenz, "hilbert", components="x", curve="snake")
    assert geom.meta["n_samples"] == lorenz.y.shape[0]
    assert geom.meta["side"] == int(np.ceil(np.sqrt(lorenz.y.shape[0])))
    assert np.isfinite(geom.channels["z"].values).sum() == lorenz.y.shape[0]


def test_an_rqa_measure_over_time_lays_out_on_a_curve(lorenz):
    """A windowed RQA measure is a 1-D record like any other — no special case."""
    windowed = ts.analysis.windowed_rqa(
        lorenz.y[::20][:400], window=80, step=4, recurrence_rate=0.1
    )
    det = np.asarray(windowed.determinism)
    geom = geometry(det, "hilbert", curve="snake")
    assert geom.meta["n_samples"] == det.size
    assert np.isfinite(geom.channels["z"].values).sum() == det.size


def test_an_inter_event_series_lays_out_on_a_curve(lorenz):
    """The return-time series the ``return_time`` transform hands back in ``meta``."""
    times = np.asarray(geometry(lorenz, "return_time", components="z").meta["return_times"])
    geom = geometry(times, "hilbert", curve="snake")
    assert geom.meta["n_samples"] == times.size


def test_a_symbolic_sequence_lays_out_on_a_curve():
    """Symbols encoded as integers: a period-3 word must tile the image."""
    symbols = np.tile([0.0, 1.0, 2.0], 400)
    geom = geometry(symbols, "hilbert", curve="rowmajor")
    z = geom.channels["z"].values
    assert set(np.unique(z[np.isfinite(z)])) == {0.0, 1.0, 2.0}


def test_a_system_subject_is_integrated_for_its_data():
    geom = geometry(
        ts.systems.Lorenz(), "hilbert", components="x", curve="snake", final_time=10.0, dt=0.01
    )
    assert geom.meta["integrated_for_plot"]["n_samples"] == 1001


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", HILBERT_TRANSFORMS)
@pytest.mark.parametrize("backend", ["matplotlib", "json"])
def test_every_view_renders_on_more_than_one_backend(series, name, backend):
    """The whole point of wrapping the array in an IMAGE layer: every backend draws it."""
    from tsdynamics.viz.render import register_builtin_renderers

    register_builtin_renderers()
    spec = build_spec(series, name, curve="snake")
    out = spec.render(backend)
    assert out is not None
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        assert any(ax.images for ax in out.axes), "a lattice must draw as an image"
        plt.close(out)


def test_a_hilbert_spec_is_themed_and_styled_like_any_other_plot(series):
    """It goes through PlotSpec, so the Theme and STYLE_KEYS reach it."""
    spec = build_spec(series, "hilbert", curve="snake")
    assert spec.aspect == "equal"
    assert spec.colorbar is not None
    assert {layer.transform for layer in spec.layers} == {"hilbert"}
    assert spec.frame.space is FrameSpace.GRID2
    round_tripped = ts.viz.from_json(ts.viz.to_json(spec))
    assert round_tripped.layers[0].transform == "hilbert"


def test_the_fourier_view_lives_in_its_own_frame_and_refuses_a_wrong_overlay(series):
    """``(k_x, k_y)`` is not ``(cell x, cell y)``, so the two must not share axes."""
    image = build_spec(series, "hilbert", curve="snake")
    spectrum = build_spec(series, "hilbert_fourier", curve="snake")
    assert image.frame != spectrum.frame
    with pytest.raises(InvalidParameterError):
        ts.viz.plot(image, spectrum, layout="overlay")
