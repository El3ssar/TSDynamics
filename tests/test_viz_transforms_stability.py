"""The stability / spectrum transforms — validated against analytic truth.

Two modules are covered: ``viz.transforms.stability`` (``eigenvalue_plane``,
``floquet_multipliers``) and ``viz.transforms.spectra`` (``lyapunov_spectrum``,
``lyapunov_convergence``, ``gali_curves``, ``zero_one_pq_plane``,
``scaling_fit``).

The bar here is deliberately not "it did not raise".  Every one of these plots
is a *quantitative* claim, so each is checked against a number that exists
independently of this library:

* the Lorenz origin's Jacobian spectrum, derived by hand from the analytic
  characteristic polynomial;
* a converged limit cycle's Floquet multipliers (one trivial ``~1``, the rest
  strictly inside the unit circle);
* the Lorenz leading exponent, 0.906;
* the Kaplan-Yorke dimension of the literature Lorenz spectrum, 2.062;
* the GALI power law's exact slope ``-(k - s)``;
* the 0-1 test's bounded-vs-diffusive geometry, as a measured extent ratio.

Plus the cross-cutting invariants the transform substrate promises: the
declared row draws, an undeclared pair raises, the frame refuses a
wrong-quantity overlay, and the reference geometry survives a JSON round trip
(which an ``Annotation``-based reference would not).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import tsdynamics as ts  # noqa: E402
from tsdynamics.errors import InvalidInputError, InvalidParameterError  # noqa: E402
from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402
from tsdynamics.viz.transforms import (  # noqa: E402
    PRIMITIVES,
    build_spec,
    compatibility,
    draw,
    geometry,
    get,
)

MINE = (
    "eigenvalue_plane",
    "floquet_multipliers",
    "lyapunov_spectrum",
    "lyapunov_convergence",
    "gali_curves",
    "zero_one_pq_plane",
    "scaling_fit",
)

#: Which matplotlib artist container each primitive fills (the same table
#: ``tests/test_viz_compatibility.py`` uses; duplicated deliberately so this file
#: covers its own cells regardless of collection order).
_CONTAINER = {
    "line": "lines",
    "points": "collections",
    "density": "images",
    # v6: ``lyapunov_spectrum`` declares ``bars`` (§6.5 — a primitive is *how*
    # geometry is drawn, and a bar chart is one), which mpl draws as patches.
    "bars": "patches",
}


@pytest.fixture(scope="module", autouse=True)
def _renderers():
    from tsdynamics import registry

    register_builtin_renderers()
    if "matplotlib" not in registry.renderers:  # pragma: no cover
        pytest.skip("matplotlib backend did not register")
    yield


def _artists(fig, container):
    out = []
    for ax in fig.axes:
        out.extend(getattr(ax, container, []))
    return out


def _labels(spec):
    return [layer.label for layer in spec.layers if layer.label]


# ---------------------------------------------------------------------------
# Registration and the declared matrix
# ---------------------------------------------------------------------------


def test_all_seven_transforms_are_registered_and_appear_in_the_matrix():
    matrix = compatibility()
    for name in MINE:
        assert name in matrix, f"{name} is not in ts.viz.compatibility()"
        assert get(name).doc, f"{name} has no one-line doc"


@pytest.mark.parametrize("name", MINE)
def test_source_category_is_one_of_the_two(name):
    record = get(name)
    assert record.source in ("data", "model")
    # The greppable rule: a transform that must build a Jacobian / integrate the
    # variational equation at a point not in its input is a model transform.
    expected = "model" if name in ("eigenvalue_plane", "floquet_multipliers") else "data"
    assert record.source == expected


@pytest.mark.parametrize("name", MINE)
def test_every_declared_cell_draws_something_finite(name):
    record = get(name)
    assert record.example is not None
    for primitive in sorted(record.primitives):
        subject, options = record.example(primitive)
        spec = build_spec(subject, name, primitive=primitive, **dict(options))
        fig = spec.render("matplotlib")
        drawn = _artists(fig, _CONTAINER[primitive])
        assert drawn, f"{name}.{primitive} produced no {_CONTAINER[primitive]}"


@pytest.mark.parametrize("name", MINE)
def test_an_undeclared_primitive_raises_and_names_the_valid_set(name):
    record = get(name)
    undeclared = sorted(set(PRIMITIVES) - record.primitives)
    subject, options = record.example(record.default_primitive)
    with pytest.raises(InvalidParameterError) as excinfo:
        build_spec(subject, name, primitive=undeclared[0], **dict(options))
    message = str(excinfo.value)
    assert undeclared[0] in message and name in message
    for valid in record.primitives:
        assert valid in message


# ---------------------------------------------------------------------------
# TRUTH 1 — the Lorenz origin is a saddle with a known 2-D stable manifold
# ---------------------------------------------------------------------------


def _lorenz_origin_truth():
    r"""Analytic eigenvalues of the Lorenz Jacobian at the origin.

    ``J(0) = [[-sigma, sigma, 0], [rho, -1, 0], [0, 0, -beta]]`` is block
    diagonal, so the spectrum is ``-beta`` together with the roots of
    ``lam^2 + (sigma + 1) lam + sigma (1 - rho) = 0``.  With the standard
    ``sigma = 10, rho = 28, beta = 8/3`` that is
    ``lam = (-11 +/- sqrt(1201)) / 2`` and ``-8/3``.
    """
    root = np.sqrt(1201.0)
    return np.sort(np.array([(-11.0 + root) / 2.0, (-11.0 - root) / 2.0, -8.0 / 3.0]))[::-1]


def test_lorenz_origin_eigenvalues_match_the_analytic_spectrum():
    geom = geometry(ts.systems.Lorenz(), "eigenvalue_plane", at=[0.0, 0.0, 0.0])
    drawn = np.concatenate(
        [
            part.array("x") + 1j * part.array("y")
            for part in geom.parts
            if part.primitive is None  # skip the boundary geometry
        ]
    )
    got = np.sort(drawn.real)[::-1]
    truth = _lorenz_origin_truth()
    assert np.allclose(got, truth, atol=1e-9), f"{got} != {truth}"
    # The physics the picture must show: one expanding direction, two contracting.
    assert int(np.sum(got > 0)) == 1
    assert int(np.sum(got < 0)) == 2
    assert np.max(np.abs(drawn.imag)) == 0.0  # the origin is a real node/saddle


def test_the_lorenz_origin_plane_separates_contracting_from_expanding():
    spec = ts.plot(ts.systems.Lorenz(), "eigenvalue_plane", at=[0.0, 0.0, 0.0])
    labels = _labels(spec)
    assert any("contracting" in text for text in labels)
    assert any("expanding" in text for text in labels)
    # ... and draws the boundary the classification is read against.
    assert any("= 0$" in text for text in labels), labels
    # Two eigenvalues left of the axis, one to the right — in the drawn data.
    left = right = 0
    for layer in spec.layers:
        if layer.label and "contracting" in layer.label:
            left += int(np.size(layer.data["x"]))
        if layer.label and "expanding" in layer.label:
            right += int(np.size(layer.data["x"]))
    assert (left, right) == (2, 1)


def test_a_map_fixed_point_is_judged_on_the_unit_circle_not_the_imaginary_axis():
    from tsdynamics.analysis.fixedpoints import FixedPoint

    fp = FixedPoint(
        x=np.zeros(2),
        eigenvalues=np.array([1.9, -0.16], dtype=complex),
        stable=False,
        continuous=False,
    )
    geom = geometry(fp, "eigenvalue_plane")
    boundary = next(part for part in geom.parts if part.primitive == "line")
    radius = np.hypot(boundary.array("x"), boundary.array("y"))
    assert np.allclose(radius, 1.0)  # the unit circle, not a straight axis
    assert "|\\lambda| < 1" in " ".join(part.label or "" for part in geom.parts)


# ---------------------------------------------------------------------------
# TRUTH 2 — a stable limit cycle: everything inside |mu| = 1 but the trivial one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "options"),
    [
        (lambda: ts.systems.VanDerPol(params={"mu": 1.0}), {"ic": [2.0, 0.0], "period_guess": 6.6}),
        (ts.systems.StuartLandau, {}),
    ],
    ids=["VanDerPol", "StuartLandau"],
)
def test_a_stable_cycle_has_one_trivial_multiplier_and_the_rest_inside(factory, options):
    options = dict(options)
    guess = options.pop("period_guess", None)
    orbits = ts.analysis.periodic_orbits(factory(), guess, **options)
    # v6: indexing a result collection gives you NUMBERS (``orbits[0]`` is the
    # orbit's points); the record carrying the multipliers is ``.details[0]``.
    orbit = orbits.details[0]
    assert orbit.stable
    mu = np.asarray(orbit.multipliers, dtype=complex)
    trivial = np.argmin(np.abs(mu - 1.0))
    assert abs(mu[trivial] - 1.0) < 1e-6, "the trivial Floquet multiplier is not ~1"
    rest = np.delete(mu, trivial)
    assert np.all(np.abs(rest) < 1.0), f"a non-trivial multiplier is outside: {rest}"

    geom = geometry(orbit, "floquet_multipliers")
    assert geom.meta["trivial_marked"] is True
    labelled = {part.label: part for part in geom.parts}
    trivial_part = next(part for label, part in labelled.items() if label and "trivial" in label)
    assert np.isclose(float(trivial_part.array("x")[0]), 1.0, atol=1e-6)
    # Every remaining drawn multiplier is inside the circle, and it is the
    # *contracting* series that carries them.
    inside = next(part for label, part in labelled.items() if label and "contracting" in label)
    assert np.all(np.hypot(inside.array("x"), inside.array("y")) < 1.0)
    assert not any(label and "expanding" in label for label in labelled)


def test_the_unit_circle_stays_in_view_for_a_strongly_stable_cycle():
    """A cycle with |mu| ~ 1e-11 must not autoscale the circle off the axes."""
    from tsdynamics.analysis.fixedpoints import PeriodicOrbit

    orbit = PeriodicOrbit(
        points=np.zeros((4, 2)),
        period=6.0,
        multipliers=np.array([1.0, 1e-11], dtype=complex),
        stable=True,
        continuous=True,
    )
    spec = draw(geometry(orbit, "floquet_multipliers"), "points")
    assert spec.x.limits is not None and spec.y.limits is not None
    assert spec.x.limits[0] <= -1.0 and spec.x.limits[1] >= 1.0
    assert spec.y.limits[0] <= -1.0 and spec.y.limits[1] >= 1.0


def test_a_map_cycle_gets_no_trivial_multiplier_marked():
    from tsdynamics.analysis.fixedpoints import PeriodicOrbit

    orbit = PeriodicOrbit(
        points=np.zeros((2, 2)),
        period=2,
        multipliers=np.array([0.4, -1.7], dtype=complex),
        stable=False,
        continuous=False,
    )
    geom = geometry(orbit, "floquet_multipliers")
    assert geom.meta["trivial_marked"] is False
    assert not any(part.label and "trivial" in part.label for part in geom.parts)


# ---------------------------------------------------------------------------
# TRUTH 3 — Lorenz convergence settles on 0.906
# ---------------------------------------------------------------------------


def test_lorenz_lyapunov_convergence_settles_on_the_literature_value():
    spec = ts.plot(
        ts.systems.Lorenz(), "lyapunov_convergence", steps=10_000, dt=0.5, ic=[1.0, 1.0, 1.0]
    )
    settled = np.asarray(spec.meta["settled"], dtype=float)
    assert settled[0] == pytest.approx(0.906, abs=0.02), settled
    assert abs(settled[1]) < 0.01  # the flow's zero exponent
    assert settled[2] == pytest.approx(-14.57, abs=0.1)

    # "Settled" is a claim about the *curve*, so check the curve: over its last
    # third the leading estimate must move by far less than it did overall.
    leading = next(layer for layer in spec.layers if layer.label and "lambda_{1}" in layer.label)
    y = np.asarray(leading.data["y"], dtype=float)
    tail = y[int(0.66 * y.size) :]
    assert np.ptp(tail) < 0.02, f"the leading estimate is still moving: ptp={np.ptp(tail)}"
    assert np.ptp(tail) < 0.05 * np.ptp(y[10:])


def test_the_convergence_label_carries_the_value_the_curve_settled_on():
    spec = ts.plot(
        ts.systems.Lorenz(), "lyapunov_convergence", steps=400, dt=0.5, ic=[1.0, 1.0, 1.0]
    )
    for i, settled in enumerate(spec.meta["settled"]):
        label = next(
            layer.label
            for layer in spec.layers
            if layer.label and f"lambda_{{{i + 1}}}" in layer.label
        )
        assert f"{float(settled):.4g}" in label


def test_the_opening_transient_is_drawn_but_does_not_set_the_vertical_scale():
    t = np.linspace(1.0, 100.0, 200)
    est = np.column_stack([0.9 + 50.0 / t])  # a huge opening, settling at 0.9
    spec = ts.plot((t, est), "lyapunov_convergence")
    curve = next(layer for layer in spec.layers if layer.label and "lambda_" in layer.label)
    peak = float(np.max(curve.data["y"]))
    assert peak > 40.0, "the transient must still be drawn"
    assert spec.y.limits[1] < 0.25 * peak, "the opening transient captured the vertical range"

    bare = ts.plot((t, est), "lyapunov_convergence", autoscale_after=0.0)
    assert bare.y.limits is None or bare.y.limits[1] > 40.0  # opting out keeps the full range


def test_convergence_rejects_a_mismatched_recorded_pair():
    with pytest.raises(InvalidInputError, match="shapes do not match"):
        geometry((np.arange(5.0), np.zeros((3, 2))), "lyapunov_convergence")


# ---------------------------------------------------------------------------
# TRUTH 4 — the spectrum plot: ordering, the zero line, and D_KY
# ---------------------------------------------------------------------------


def test_lyapunov_spectrum_orders_the_exponents_and_annotates_kaplan_yorke():
    spec = ts.plot([-14.572, 0.906, 0.0], "lyapunov_spectrum")
    assert spec.meta["exponents"] == sorted(spec.meta["exponents"], reverse=True)
    assert spec.meta["n_positive"] == 1
    # Kaplan-Yorke of the literature Lorenz spectrum, computed independently:
    # 2 + (0.906 + 0.0) / 14.572 = 2.0622.
    assert spec.meta["kaplan_yorke"] == pytest.approx(2.0 + 0.906 / 14.572, rel=1e-6)
    assert f"{spec.meta['kaplan_yorke']:.4g}" in spec.title
    assert any("lambda = 0" in text for text in _labels(spec))


def test_every_exponent_gets_a_stem_from_zero_to_its_value():
    exps = [0.906, 0.0, -14.572]
    geom = geometry(exps, "lyapunov_spectrum")
    stems = [
        part
        for part in geom.parts
        if part.primitive == "line"
        and np.size(part.array("x")) == 2
        and float(part.array("x")[0]) == float(part.array("x")[1])
    ]
    assert len(stems) == len(exps)
    for i, part in enumerate(stems):
        assert float(part.array("y")[0]) == 0.0
        assert float(part.array("y")[1]) == pytest.approx(sorted(exps, reverse=True)[i])


def test_a_precomputed_spectrum_refuses_estimator_options_instead_of_ignoring_them():
    with pytest.raises(InvalidInputError, match="already computed"):
        geometry([0.9, 0.0, -14.0], "lyapunov_spectrum", final_time=100.0)


def test_lyapunov_spectrum_runs_the_estimator_when_given_a_system():
    spec = ts.plot(ts.systems.Lorenz(), "lyapunov_spectrum", final_time=200.0, dt=0.5)
    assert spec.meta["n_positive"] == 1
    assert spec.meta["exponents"][0] == pytest.approx(0.906, abs=0.15)


# ---------------------------------------------------------------------------
# TRUTH 5 — GALI: the reference slope is exactly -(k - s)
# ---------------------------------------------------------------------------


def test_the_gali_reference_has_the_analytic_power_law_slope():
    from tsdynamics.analysis.chaos.gali import GALIResult

    t = np.geomspace(1.0, 1000.0, 200)
    results = [GALIResult(k=kk, times=t, values=t ** -(kk - 2.0)) for kk in (3, 4)]
    geom = geometry(results, "gali_curves", frequencies=2)
    for order in (3, 4):
        guide = next(
            part for part in geom.parts if part.label and f"t^{{{-(order - 2)}}}" in part.label
        )
        x, y = guide.array("x"), guide.array("y")
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        assert slope == pytest.approx(-(order - 2), rel=1e-9)


def test_gali_separates_a_regular_from_a_chaotic_henon_heiles_orbit():
    """The order/chaos discriminator, on the canonical 2-dof Hamiltonian."""
    hh = ts.systems.HenonHeiles()
    common = {"k": 2, "frequencies": 2, "final_time": 1000.0, "dt": 1.0, "transient": 0.0}
    regular = ts.plot(hh, "gali_curves", ic=[0.0, 0.1, 0.15, 0.0], **common)
    chaotic = ts.plot(hh, "gali_curves", ic=[0.0, -0.1, 0.49, 0.0], **common)
    # GALI_2 holds near 1 on the torus and collapses on the chaotic orbit.
    assert regular.meta["final"][0] > 0.1
    assert chaotic.meta["final"][0] < 1e-6
    assert chaotic.meta["final"][0] < regular.meta["final"][0] / 1e5


def test_gali_clips_the_underflowed_tail_instead_of_dropping_it():
    from tsdynamics.analysis.chaos.gali import GALIResult

    t = np.linspace(1.0, 10.0, 20)
    values = np.concatenate([np.ones(10), np.zeros(10)])  # exact underflow to 0
    geom = geometry(GALIResult(k=2, times=t, values=values), "gali_curves", frequencies=0)
    y = geom.parts[0].array("y")
    assert y.size == t.size, "the underflowed samples were dropped"
    assert np.all(np.isfinite(y))
    assert float(np.min(y)) == pytest.approx(np.log10(1e-16))


def test_gali_reference_is_omitted_when_no_frequency_count_is_claimed():
    from tsdynamics.analysis.chaos.gali import GALIResult

    t = np.geomspace(1.0, 100.0, 50)
    geom = geometry(GALIResult(k=3, times=t, values=1.0 / t), "gali_curves", frequencies=0)
    assert len(geom.parts) == 1


# ---------------------------------------------------------------------------
# TRUTH 6 — the 0-1 plane: bounded for a cycle, diffusive for chaotic Lorenz
# ---------------------------------------------------------------------------


def test_the_zero_one_plane_is_bounded_for_a_cycle_and_diffusive_for_lorenz():
    regular = ts.plot(
        ts.systems.VanDerPol(params={"mu": 1.0}),
        "zero_one_pq_plane",
        ic=[2.0, 0.0],
        components=0,
        final_time=2000.0,
        dt=0.3,
        transient=200.0,
        seed=0,
    )
    chaotic = ts.plot(
        ts.systems.Lorenz(),
        "zero_one_pq_plane",
        ic=[1.0, 1.0, 1.0],
        components=0,
        final_time=2000.0,
        dt=0.5,
        transient=100.0,
        seed=0,
    )
    assert regular.meta["verdict"] == "regular" and regular.meta["K"] < 0.1
    assert chaotic.meta["verdict"] == "chaotic" and chaotic.meta["K"] > 0.9

    def extent(spec):
        layer = spec.layers[0]
        return max(float(np.ptp(layer.data["x"])), float(np.ptp(layer.data["y"])))

    # The visible claim: the chaotic walk sprawls orders of magnitude further
    # than the bounded blob.  (Measured here: ~4 units vs ~600.)
    assert extent(chaotic) > 50.0 * extent(regular)


def test_the_zero_one_plane_marks_where_the_walk_started():
    result = ts.analysis.zero_one_test(np.cos(np.arange(2000) * 0.7), seed=0)
    geom = geometry(result, "zero_one_pq_plane")
    start = next(part for part in geom.parts if part.label == "start")
    assert np.size(start.array("x")) == 1
    assert float(start.array("x")[0]) == pytest.approx(float(geom.parts[0].array("x")[0]))


def test_a_precomputed_zero_one_result_refuses_test_options():
    result = ts.analysis.zero_one_test(np.cos(np.arange(500) * 0.7), seed=0)
    with pytest.raises(InvalidInputError, match="configure the test"):
        geometry(result, "zero_one_pq_plane", dt=0.5)


# ---------------------------------------------------------------------------
# TRUTH 7 — the scaling fit: the window is visible and the residuals inspectable
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lorenz_dimension():
    traj = ts.systems.Lorenz().run(final_time=200.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    return ts.analysis.correlation_dimension(traj)


def test_the_fitted_window_is_drawn_as_geometry_not_merely_marker_colour(lorenz_dimension):
    geom = geometry(lorenz_dimension, "scaling_fit")
    lo, hi = lorenz_dimension.fit_region
    x = np.asarray(lorenz_dimension.abscissa, dtype=float)
    delimiters = [
        part
        for part in geom.parts
        if part.primitive == "line"
        and np.size(part.array("x")) == 2
        and float(part.array("x")[0]) == float(part.array("x")[1])
    ]
    assert len(delimiters) == 2, "the fit window must be delimited on both sides"
    edges = sorted(float(part.array("x")[0]) for part in delimiters)
    assert edges == pytest.approx([x[lo], x[hi]])


def test_the_highlighted_points_are_exactly_the_fitted_ones(lorenz_dimension):
    geom = geometry(lorenz_dimension, "scaling_fit")
    lo, hi = lorenz_dimension.fit_region
    region = next(part for part in geom.parts if part.label and part.label.startswith("fit region"))
    assert np.size(region.array("x")) == hi - lo + 1
    assert region.array("x") == pytest.approx(np.asarray(lorenz_dimension.abscissa)[lo : hi + 1])
    assert f"{hi - lo + 1} points" in region.label


def test_the_residual_view_shows_a_good_fit_as_a_flat_band(lorenz_dimension):
    """A residual view that cannot distinguish inside from outside is useless."""
    geom = geometry(lorenz_dimension, "scaling_fit", view="residuals")
    lo, hi = lorenz_dimension.fit_region
    residual = next(part for part in geom.parts if part.label == "residual")
    r = np.asarray(residual.array("y"), dtype=float)
    inside = np.max(np.abs(r[lo : hi + 1]))
    outside = np.max(np.abs(np.concatenate([r[:lo], r[hi + 1 :]])))
    assert inside < 0.02, f"the fitted region is not straight: max|residual| = {inside}"
    assert outside > 3.0 * inside, "the residual view does not separate in from out"


def test_the_local_slope_view_plateaus_at_the_reported_estimate(lorenz_dimension):
    geom = geometry(lorenz_dimension, "scaling_fit", view="local_slopes")
    lo, hi = lorenz_dimension.fit_region
    local = next(part for part in geom.parts if part.label == "local slope")
    plateau = np.asarray(local.array("y"), dtype=float)[lo : hi + 1]
    assert np.median(plateau) == pytest.approx(float(lorenz_dimension), abs=0.05)
    assert np.ptp(plateau) < 0.2, "the 'scaling region' has no plateau"
    # Lorenz's correlation dimension, independently: D2 ~ 2.05.
    assert float(lorenz_dimension) == pytest.approx(2.05, abs=0.1)


def test_the_slope_and_its_uncertainty_reach_the_figure(lorenz_dimension):
    spec = ts.plot(lorenz_dimension, "scaling_fit")
    assert spec.legend is not None, "a labelled figure with no legend says nothing"
    assert any(f"{float(lorenz_dimension):.4g}" in text for text in _labels(spec)), _labels(spec)


def test_an_unknown_scaling_view_raises_and_lists_the_three():
    subject, _ = get("scaling_fit").example("points")
    with pytest.raises(InvalidParameterError, match="local_slopes"):
        geometry(subject, "scaling_fit", view="resid")


def test_scaling_fit_accepts_every_in_tree_scaling_result_shape():
    """The schema claim: one transform covers the whole ScalingResult family."""
    traj = ts.systems.Lorenz().run(final_time=60.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    for result in (
        ts.analysis.correlation_dimension(traj),
        ts.analysis.lyapunov_from_data(traj["x"], dt=0.05),
    ):
        spec = ts.plot(result, "scaling_fit")
        assert spec.kind is ts.viz.PlotKind.SCALING_FIT
        assert spec.layers


# ---------------------------------------------------------------------------
# Cross-cutting: frames, provenance, backends, error paths
# ---------------------------------------------------------------------------


def test_an_eigenvalue_plane_refuses_to_overlay_a_multiplier_plane():
    eig, _ = get("eigenvalue_plane").example("points")
    orbit, _ = get("floquet_multipliers").example("points")
    a = ts.plot(eig, "eigenvalue_plane")
    b = ts.plot(orbit, "floquet_multipliers")
    assert a.frame.space is b.frame.space  # both live in the complex plane ...
    assert a.frame.axes != b.frame.axes  # ... but they are different quantities
    with pytest.raises(InvalidParameterError, match="axes mismatch"):
        ts.viz.plot(a, b)


def test_two_eigenvalue_planes_do_overlay():
    from tsdynamics.analysis.fixedpoints import FixedPoint

    specs = [
        ts.plot(
            FixedPoint(x=np.zeros(2), eigenvalues=np.array([c], dtype=complex), continuous=True),
            "eigenvalue_plane",
        )
        for c in (-1.0 + 2.0j, 0.5)
    ]
    from tsdynamics.viz._frames import FrameSpace

    merged = ts.viz.plot(*specs)
    assert merged.frame.space is FrameSpace.COMPLEX
    assert len(merged.layers) == len(specs[0].layers) + len(specs[1].layers)


@pytest.mark.parametrize("name", MINE)
def test_every_layer_carries_its_transform_as_provenance(name):
    record = get(name)
    subject, options = record.example(record.default_primitive)
    spec = build_spec(subject, name, **dict(options))
    assert {layer.transform for layer in spec.layers} == {name}


@pytest.mark.parametrize("name", MINE)
def test_the_reference_geometry_survives_a_json_round_trip(name):
    """Reference lines are *parts*, so unlike annotations they export."""
    from tsdynamics.viz.export import from_json, to_json

    record = get(name)
    subject, options = record.example(record.default_primitive)
    spec = build_spec(subject, name, **dict(options))
    back = from_json(to_json(spec))
    assert len(back.layers) == len(spec.layers)
    assert back.frame == spec.frame
    assert [layer.label for layer in back.layers] == [layer.label for layer in spec.layers]


@pytest.mark.parametrize("name", MINE)
def test_every_transform_renders_on_plotly_and_json(name):
    pytest.importorskip("plotly")
    record = get(name)
    subject, options = record.example(record.default_primitive)
    spec = build_spec(subject, name, **dict(options))
    assert spec.render("plotly") is not None
    assert spec.render("json") is not None


def test_eigenvalue_plane_refuses_a_trajectory_rather_than_drawing_its_samples():
    traj = ts.systems.Lorenz().run(final_time=5.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    with pytest.raises(InvalidInputError, match="needs a dynamical system"):
        geometry(traj, "eigenvalue_plane")


def test_at_on_a_non_system_raises_rather_than_being_ignored():
    fp, _ = get("eigenvalue_plane").example("points")
    with pytest.raises(InvalidInputError, match="at="):
        geometry(fp, "eigenvalue_plane", at=[0.0, 0.0, 0.0])


def test_an_empty_spectrum_raises_instead_of_drawing_an_empty_plane():
    with pytest.raises(InvalidInputError):
        geometry(np.empty(0), "eigenvalue_plane")


def test_shared_keywords_reach_the_estimator_through_the_composition_front_door():
    """Regression: a ``**kwargs`` compute would have these silently dropped.

    ``ts.plot`` routes a shared keyword only to a transform whose *signature*
    names it, so every forwarded estimator option is declared explicitly.
    """
    import inspect

    for name in MINE:
        params = inspect.signature(get(name).compute).parameters
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
            f"{name} takes **kwargs; ts.plot would drop those options silently"
        )

    spec = ts.plot(
        ts.systems.Lorenz(), "lyapunov_convergence", steps=50, dt=0.2, ic=[1.0, 1.0, 1.0]
    )
    assert spec.meta["steps"] == 50
