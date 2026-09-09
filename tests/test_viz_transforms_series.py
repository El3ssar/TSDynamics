"""The data-capable diagnostic transforms (``viz/transforms/series.py``).

Seven registered transforms, and the thing this file exists to check is **not**
that they return a well-shaped spec — a plot that draws without error and shows
the wrong thing is the failure mode of a plotting module, and a spec-dict
assertion cannot see it.  So the tests here assert *physics*:

* the PSD of a pure sinusoid is a single line at the right frequency;
* the PSD of a chaotic Lorenz orbit is broadband;
* the autocorrelation of white noise drops to zero at lag 1;
* the return times of a sinusoid are its period;
* the recurrence line-length distributions reproduce, exactly, the ``L_max`` and
  ``DET`` that ``rqa`` computes from them.

Plus the registry contract each transform signs (source category, declared
frame, the row, the invalid-pair error) and the scope boundary the PSD is
admitted under.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import tsdynamics as ts  # noqa: E402
from tsdynamics.errors import InvalidInputError, InvalidParameterError  # noqa: E402
from tsdynamics.viz._frames import FrameSpace  # noqa: E402
from tsdynamics.viz.spec import PlotKind  # noqa: E402
from tsdynamics.viz.transforms import (  # noqa: E402
    ADMITTED_SERIES_DIAGNOSTICS,
    build_spec,
    compatibility,
    geometry,
    get,
)
from tsdynamics.viz.transforms.series import series_of  # noqa: E402

#: Everything this module registers.
SERIES_TRANSFORMS = (
    "psd",
    "autocorrelation",
    "mutual_information",
    "fnn",
    "cao",
    "line_lengths",
    "return_time",
)

FS = 100.0  # samples per time unit for the synthetic signals
DT = 1.0 / FS


@pytest.fixture(scope="module")
def sinusoid() -> np.ndarray:
    """A 3 Hz sinusoid sampled at 100 Hz — 4096 samples."""
    return np.sin(2.0 * np.pi * 3.0 * np.arange(4096) * DT)


@pytest.fixture(scope="module")
def white() -> np.ndarray:
    """Reproducible white noise — 4096 samples."""
    return np.random.default_rng(20260909).standard_normal(4096)


@pytest.fixture(scope="module")
def lorenz():
    """A pinned Lorenz orbit: 10,001 samples, deterministic (an explicit ``ic``)."""
    return ts.systems.Lorenz().integrate(final_time=100.0, dt=0.01, ic=[1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SERIES_TRANSFORMS)
def test_each_transform_is_registered_as_a_data_transform(name):
    """Every one is ``source="data"``: it computes from samples it was given.

    None of them evaluates a right-hand side anywhere new — a system subject is
    integrated *once* to obtain the samples, which is exactly what the ``data``
    category permits.
    """
    record = get(name)
    assert record.source == "data"
    assert name in compatibility()
    assert record.default_primitive in record.primitives
    assert record.example is not None, "the compatibility gate needs an example subject"
    assert record.doc, "a row with no doc is invisible in ts.viz.compatibility()"


@pytest.mark.parametrize("name", SERIES_TRANSFORMS)
def test_each_transform_declares_the_diagnostic_frame_it_computes(name):
    """The declared frame is ``scaling`` for all seven, and the geometry honours it."""
    record = get(name)
    assert record.frame == (FrameSpace.SCALING,)
    assert record.kind is PlotKind.DIAGNOSTIC_CURVE
    subject, options = record.example(record.default_primitive)
    geom = geometry(subject, name, **dict(options))
    assert geom.frame.space is FrameSpace.SCALING
    assert geom.frame.ndim == 1


def test_an_invalid_primitive_raises_naming_the_valid_set(sinusoid):
    with pytest.raises(InvalidParameterError, match="not valid for transform 'psd'"):
        build_spec(sinusoid, "psd", primitive="quiver")


# ---------------------------------------------------------------------------
# The scope boundary the PSD is admitted under
# ---------------------------------------------------------------------------


def test_the_psd_is_the_admitted_series_diagnostic_and_is_registered_under_that_name():
    """The re-admitted transform must *be* the one the admitted set names.

    A transform called ``power_spectrum`` while the checked constant says ``psd``
    would leave the gate guarding nothing.
    """
    assert "psd" in ADMITTED_SERIES_DIAGNOSTICS
    assert "psd" in compatibility()


def test_the_psd_is_not_a_spectral_toolbox(sinusoid):
    """The knob set is closed: two estimators, and an unknown one raises.

    This is the mechanical half of the admission rule — *a PSD toolbox with
    windowing options, detrending and filter design is not a phase-space
    diagnostic*.  Growing this signature is the deliberate act that re-opens the
    scope decision.
    """
    import inspect

    params = set(inspect.signature(get("psd").compute).parameters)
    assert params == {
        "subject",
        "component",
        "method",
        "nperseg",
        "final_time",
        "dt",
        "steps",
    }
    with pytest.raises(InvalidParameterError, match="not a spectral toolbox"):
        build_spec(sinusoid, "psd", method="multitaper")


# ---------------------------------------------------------------------------
# Physics: the PSD separates periodic from chaotic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["periodogram", "welch"])
def test_the_psd_of_a_sinusoid_is_one_line_at_the_right_frequency(sinusoid, method):
    """A pure tone must give a single peak at 3 Hz carrying nearly all the power."""
    geom = geometry(sinusoid, "psd", method=method, dt=DT)
    freq = geom.channels["x"].values
    power = geom.channels["y"].values
    resolution = float(freq[1] - freq[0])  # Welch's segments are coarser than a periodogram
    peak = int(np.argmax(power))
    assert freq[peak] == pytest.approx(3.0, abs=resolution)
    # "A single line" measured honestly: Welch's Hann window spreads one tone over
    # its three adjacent bins by construction (56% in the peak bin, 99% in three),
    # so the concentration is read over the line, not over one sample of it.
    line = power[max(0, peak - 1) : peak + 2].sum() / power.sum()
    assert line > 0.95, "a pure tone must not be spread over the band"
    assert geom.meta["sampling_rate"] == pytest.approx(FS)


def test_the_psd_of_a_chaotic_orbit_is_broadband(lorenz):
    """Lorenz has no discrete lines: no bin dominates, and many bins are populated."""
    power = geometry(lorenz, "psd", component="x").channels["y"].values
    assert power.max() / power.sum() < 0.25, "a chaotic spectrum must not be a line"
    assert (power > 0.01 * power.max()).sum() > 20, "a chaotic spectrum must be continuous"


def test_the_psd_of_a_sinusoid_and_of_noise_are_told_apart_by_the_same_number(sinusoid, white):
    """The discriminant the figure is read with, as a number: peak concentration."""
    tone = geometry(sinusoid, "psd", dt=DT).channels["y"].values
    noise = geometry(white, "psd", dt=DT).channels["y"].values
    assert tone.max() / tone.sum() > 20.0 * (noise.max() / noise.sum())


# ---------------------------------------------------------------------------
# Physics: the delay diagnostics
# ---------------------------------------------------------------------------


def test_the_autocorrelation_of_white_noise_drops_to_zero_at_lag_one(white):
    geom = geometry(white, "autocorrelation", max_delay=20)
    acf = geom.parts[0].array("y")
    assert acf[0] == pytest.approx(1.0)
    assert abs(acf[1]) < 0.05
    assert np.abs(acf[1:]).max() < 0.1
    assert geom.meta["tau_1_over_e"] == 1, "white noise decorrelates in one sample"


def test_the_autocorrelation_marks_both_crossings_of_a_smooth_orbit(lorenz):
    """The 1/e and first-zero lags are drawn, not merely stashed in ``meta``."""
    geom = geometry(lorenz, "autocorrelation", component="x", max_delay=1000)
    assert geom.meta["tau_1_over_e"] is not None
    assert geom.meta["tau_first_zero"] is not None
    assert 0 < geom.meta["tau_1_over_e"] < geom.meta["tau_first_zero"]
    labels = [p.label for p in geom.parts]
    assert len(geom.parts) == 3
    assert any("1/e" in str(lbl) for lbl in labels)
    assert any("first zero" in str(lbl) for lbl in labels)
    # The markers are annotations: they stay lines whatever the curve is drawn as.
    assert [p.primitive for p in geom.parts] == [None, "line", "line"]


def test_a_curve_that_never_crosses_reports_none_rather_than_extrapolating(sinusoid):
    """An honest ``None`` beats a lag invented past the end of the curve."""
    geom = geometry(sinusoid, "autocorrelation", max_delay=3)
    assert geom.meta["tau_1_over_e"] is None
    assert len(geom.parts) == 1


def test_mutual_information_marks_the_delay_its_estimator_would_choose(lorenz):
    """The transform's marker must be the estimator's answer, not a second opinion."""
    from tsdynamics.analysis.embedding import mutual_information as mi

    geom = geometry(lorenz, "mutual_information", component="x", max_delay=100)
    assert geom.meta["optimal_lag"] == int(mi(lorenz.y[:, 0], max_delay=100).optimal_lag)
    marker = geom.parts[-1]
    assert marker.array("x")[0] == pytest.approx(float(geom.meta["optimal_lag"]))


def test_the_two_delay_diagnostics_share_a_frame_and_therefore_overlay(lorenz):
    """``tau`` is one coordinate: the whole point is reading them against each other."""
    acf = build_spec(lorenz, "autocorrelation", component="x", max_delay=100)
    mi = build_spec(lorenz, "mutual_information", component="x", max_delay=100)
    assert acf.frame == mi.frame
    merged = ts.viz.plot(acf, mi, layout="overlay")
    assert len(merged.layers) == len(acf.layers) + len(mi.layers)


def test_a_psd_refuses_to_overlay_a_delay_curve(lorenz):
    """Different coordinates, different frames — the overlay check does its job."""
    psd = build_spec(lorenz, "psd", component="x")
    acf = build_spec(lorenz, "autocorrelation", component="x")
    assert psd.frame != acf.frame
    with pytest.raises(InvalidParameterError):
        ts.viz.plot(psd, acf, layout="overlay")


# ---------------------------------------------------------------------------
# Physics: the dimension diagnostics
# ---------------------------------------------------------------------------


def test_fnn_decays_to_zero_at_the_dimension_it_reports(lorenz):
    """The curve *is* the evidence: it must start high and reach the threshold at m."""
    geom = geometry(lorenz, "fnn", component="x", delay=17, max_dim=8)
    dims = geom.parts[0].array("x")
    fraction = geom.parts[0].array("y")
    m = int(geom.meta["dimension"])
    assert 2 <= m <= 4, f"Lorenz x should embed in ~3 dimensions, got {m}"
    assert fraction[0] > 0.5, "d = 1 must show a large false-neighbour fraction"
    assert fraction[dims == m][0] <= 0.01
    assert geom.parts[-1].array("x")[0] == pytest.approx(float(m))


def test_cao_draws_both_curves_because_e2_is_the_determinism_test(lorenz):
    """E1 saturates at 1; E2 departing from 1 is what says the data is deterministic."""
    geom = geometry(lorenz, "cao", component="x", delay=17, max_dim=8)
    labels = [p.label for p in geom.parts]
    assert "$E_1(d)$" in labels and "$E_2(d)$" in labels
    e1 = geom.parts[0].array("y")
    e2 = geom.parts[1].array("y")
    assert e1[-1] == pytest.approx(1.0, abs=0.1), "E1 must saturate at 1"
    assert np.abs(e2 - 1.0).max() > 0.1, "E2 must leave 1 for deterministic data"


# ---------------------------------------------------------------------------
# Physics: the recurrence line-length distributions
# ---------------------------------------------------------------------------


def test_the_line_length_distributions_reproduce_the_rqa_numbers_they_are_read_from(lorenz):
    """``L_max``, ``DET``, ``LAM`` and ``TT`` must be recoverable from the drawn curves.

    This is the claim the transform makes — that these histograms are exactly
    what ``rqa`` reduces to four numbers — so it is checked against ``rqa``
    itself, to the last bit, rather than asserted in a docstring.
    """
    points = lorenz.y[::20][:600]
    matrix = ts.recurrence_matrix(points, recurrence_rate=0.05, theiler=2)
    quant = ts.rqa(matrix, min_diagonal=2, min_vertical=2)
    geom = geometry(matrix, "line_lengths", min_diagonal=2, min_vertical=2, normalize=False)

    lengths = geom.parts[0].array("x")
    counts = geom.parts[0].array("y")
    assert int(lengths[counts > 0].max()) == int(quant.max_diagonal_length)
    # DET = points on diagonal lines >= min / all recurrence points (upper triangle).
    assert float((lengths * counts).sum()) / (matrix.matrix.nnz / 2.0) == pytest.approx(
        quant.determinism
    )

    v_lengths = geom.parts[1].array("x")
    v_counts = geom.parts[1].array("y")
    # LAM is the same reduction of P(v), and TT is its mean.
    assert float((v_lengths * v_counts).sum()) / matrix.matrix.nnz == pytest.approx(
        quant.laminarity
    )
    assert float((v_lengths * v_counts).sum() / v_counts.sum()) == pytest.approx(
        quant.trapping_time
    )


def test_the_line_length_distributions_are_probabilities_when_normalized(lorenz):
    geom = geometry(lorenz.y[::20][:400], "line_lengths", recurrence_rate=0.05)
    for part in geom.parts:
        assert part.array("y").sum() == pytest.approx(1.0)
    assert [p.label for p in geom.parts] == ["$P(l)$ diagonal", "$P(v)$ vertical"]


def test_line_lengths_uses_the_full_state_vector_by_default(lorenz):
    """Phase-space recurrence is of the *state*, not of one coordinate."""
    full = geometry(lorenz.y[::40][:300], "line_lengths", recurrence_rate=0.05)
    one = geometry(lorenz.y[::40][:300], "line_lengths", component=0, recurrence_rate=0.05)
    assert full.meta["epsilon"] != one.meta["epsilon"]


def test_a_threshold_too_small_to_make_lines_says_so(white):
    with pytest.raises(InvalidInputError, match="no distribution to draw"):
        geometry(white[:200], "line_lengths", threshold=1e-12)


# ---------------------------------------------------------------------------
# Physics: return times
# ---------------------------------------------------------------------------


def test_the_return_times_of_a_sinusoid_are_its_period(sinusoid):
    """3 Hz upward zero-crossings recur every 1/3 of a time unit, and nowhere else."""
    geom = geometry(sinusoid, "return_time", threshold=0.0, direction="up", dt=DT, n_bins=40)
    times = np.asarray(geom.meta["return_times"])
    assert times.mean() == pytest.approx(1.0 / 3.0, rel=0.02)
    assert times.std() < 0.02, "a periodic orbit has one return time, not a distribution"
    assert geom.meta["n_returns"] == times.size


def test_the_return_times_of_a_chaotic_orbit_are_a_distribution(lorenz):
    geom = geometry(lorenz, "return_time", component="z", n_bins=25)
    times = np.asarray(geom.meta["return_times"])
    assert times.size > 20
    assert times.std() / times.mean() > 0.1, "a chaotic orbit must not return periodically"
    assert geom.channels["y"].values.sum() > 0


def test_return_time_defaults_to_the_mean_level_so_crossings_always_exist(lorenz):
    geom = geometry(lorenz, "return_time", component="x")
    assert geom.meta["threshold"] == pytest.approx(float(lorenz.y[:, 0].mean()))


def test_return_time_rejects_an_unknown_direction_and_an_uncrossable_level(sinusoid):
    with pytest.raises(InvalidParameterError, match="use 'up', 'down' or 'both'"):
        geometry(sinusoid, "return_time", direction="sideways")
    with pytest.raises(InvalidInputError, match="no return time is defined"):
        geometry(sinusoid, "return_time", threshold=99.0)


# ---------------------------------------------------------------------------
# Subject coercion: an array, a Trajectory, or a system
# ---------------------------------------------------------------------------


def test_a_data_transform_accepts_a_bare_array_a_trajectory_and_a_system(lorenz):
    """A model gives you data for free — the ``data`` category's own rule."""
    from_array = geometry(lorenz.y[:, 0], "psd", dt=0.01).channels["y"].values
    from_traj = geometry(lorenz, "psd", component="x").channels["y"].values
    assert np.allclose(from_array, from_traj)

    geom = geometry(ts.systems.Lorenz(), "psd", component="x", final_time=20.0, dt=0.01)
    assert geom.channels["y"].values.size > 8
    assert geom.meta["integrated_for_plot"]["n_samples"] > 100


def test_a_discrete_system_subject_is_iterated_not_integrated():
    geom = geometry(ts.systems.Logistic(r=3.9), "autocorrelation", steps=400, max_delay=10)
    assert geom.meta["integrated_for_plot"]["n_samples"] == 400
    assert geom.meta["sample_spacing"] == 1.0


def test_series_of_reads_the_sample_spacing_off_the_time_axis(lorenz):
    values, spacing, meta, title = series_of(lorenz, component="z")
    assert spacing == pytest.approx(0.01)
    assert values.shape == (lorenz.y.shape[0],)
    assert meta["component"] == 2
    assert title


def test_a_component_name_is_resolved_against_the_declared_variables(lorenz):
    by_name = series_of(lorenz, component="z")[0]
    by_index = series_of(lorenz, component=2)[0]
    assert np.array_equal(by_name, by_index)


def test_a_series_too_short_to_diagnose_says_so():
    with pytest.raises(InvalidInputError, match="at least 8 samples"):
        geometry(np.arange(4.0), "psd")


# ---------------------------------------------------------------------------
# Rendering — the only evidence that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SERIES_TRANSFORMS)
@pytest.mark.parametrize("backend", ["matplotlib", "json"])
def test_every_series_transform_renders_on_more_than_one_backend(name, backend):
    """A transform lowering to plain marks draws everywhere, by construction."""
    from tsdynamics.viz.render import register_builtin_renderers

    register_builtin_renderers()
    record = get(name)
    subject, options = record.example(record.default_primitive)
    spec = build_spec(subject, name, **dict(options))
    out = spec.render(backend)
    assert out is not None
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        assert out.axes and any(ax.lines or ax.collections for ax in out.axes)
        plt.close(out)


def test_a_diagnostic_spec_can_be_put_on_log_axes(sinusoid):
    """The substrate carries no axis *scale*, so the documented route is ``rescale``."""
    spec = build_spec(sinusoid, "psd", dt=DT).rescale(x="log", y="log")
    assert (spec.x.scale, spec.y.scale) == ("log", "log")


def test_every_layer_carries_its_transform_provenance(lorenz):
    spec = ts.plot(lorenz, "autocorrelation", component="x")
    assert {layer.transform for layer in spec.layers} == {"autocorrelation"}
