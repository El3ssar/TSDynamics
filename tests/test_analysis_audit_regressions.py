"""Numerical and input-contract regressions from the library-wide audit."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis.dimensions._scaling import fit_scaling_region
from tsdynamics.analysis.lyapunov.from_data import ScalingRegionWarning


class _TimeStretch(ts.ContinuousSystem):
    dim = 2
    params = {}

    @staticmethod
    def _equations(y, t):
        return t * y(0), 0.0


class _NoEquilibrium(ts.ContinuousSystem):
    dim = 2
    params = {}

    @staticmethod
    def _equations(y, t):
        return 1.0, 1.0


class _AdditiveForcing(ts.ContinuousSystem):
    dim = 1
    params = {"amplitude": 1.0}

    @staticmethod
    def _equations(y, t, amplitude):
        from symengine import sin

        return [-y(0) + amplitude * sin(t)]


class _ForcedDelay(ts.DelaySystem):
    dim = 1
    params = {"tau": 0.1, "amplitude": 0.2}

    @staticmethod
    def _equations(y, t, tau, amplitude):
        from symengine import sin

        return [-2.0 * y(0) + 0.1 * y(0, t - tau) + amplitude * sin(t)]


class _ForcedReturn(ts.ContinuousSystem):
    dim = 1
    params = {}

    @staticmethod
    def _equations(y, t):
        from symengine import cos

        return [cos(t)]


class _OutOfPlaneShear(ts.ContinuousSystem):
    dim = 3
    params = {"rate": 2.0}

    @staticmethod
    def _equations(y, t, rate):
        return [0.0, 0.0, rate * y(0)]


class _PlanarShear(ts.ContinuousSystem):
    dim = 2
    params = {"rate": 2.0}

    @staticmethod
    def _equations(y, t, rate):
        return [rate * y(1), 0.0]


class _StiffField(ts.ContinuousSystem):
    dim = 2
    params = {}
    _default_method = "bdf"

    @staticmethod
    def _equations(y, t):
        return [-100.0 * y(0), -y(1)]


class _HarmonicSection(ts.ContinuousSystem):
    dim = 2
    params = {}
    _default_ic = [0.0, 1.0]

    @staticmethod
    def _equations(y, t):
        return [y(1), -y(0)]


class _ClockDrift(ts.ContinuousSystem):
    dim = 2
    params = {"offset": 0.0, "slope": 1.0}

    @staticmethod
    def _equations(y, t, offset, slope):
        return [offset + slope * t, 0.0]


def test_fields_resolve_instance_sized_defaults_and_preserve_explicit_state_precedence() -> None:
    system = ts.systems.MultiChua(n_circuits=5)
    settings = dict(plane=(1, 2), xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), grid=3)
    field = ts.analysis.flow_field(system, **settings)
    declared = np.asarray(system._default_ic, dtype=float)
    assert declared.size == 15
    np.testing.assert_array_equal(field.meta["at"], declared)
    # The central arrow samples y=z=0 with the first circuit's x pinned at .1;
    # its y'=x-y+z distinguishes the declared state from the old all-zero slice.
    assert field.u[1, 1] == pytest.approx(0.1)
    configured = declared.copy()
    configured[0] = 0.7
    explicit = ts.systems.MultiChua(n_circuits=5, ic=configured)
    selected = ts.analysis.flow_field(explicit, **settings)
    assert selected.u[1, 1] == pytest.approx(0.7)
    overridden = ts.analysis.flow_field(explicit, at=declared, **settings)
    assert overridden.u[1, 1] == pytest.approx(0.1)
    np.testing.assert_array_equal(explicit.ic, configured)


@pytest.mark.parametrize("analysis", ["escape_time_field", "transient_time_field"])
def test_first_passage_fields_advance_the_initial_clock_but_return_durations(analysis) -> None:
    settings = dict(
        xlim=(0.0, 0.1),
        ylim=(-1.0, 1.0),
        grid=3,
        t0=2.0,
        final_time=1.0,
        chunks=4,
        dt=0.05,
        solver="rk4",
    )
    settings["escape" if analysis == "escape_time_field" else "settled"] = lambda states: (
        states[:, 0] >= 1.25
    )
    result = getattr(ts.analysis, analysis)(_ClockDrift(), **settings)
    # x(2+s)-x(2)=2s+s²/2. The .5 checkpoint has x<=1.225;
    # every row first satisfies the predicate at the .75 checkpoint.
    np.testing.assert_allclose(result.values, 0.75)
    assert result.meta["t0"] == 2.0


def test_default_transient_speed_is_evaluated_at_each_checkpoint_time() -> None:
    result = ts.analysis.transient_time_field(
        _ClockDrift(offset=3.0, slope=-1.0),
        xlim=(0.0, 0.1),
        ylim=(-1.0, 1.0),
        grid=3,
        t0=2.0,
        final_time=1.0,
        chunks=4,
        tol=0.4,
        dt=0.05,
        solver="rk4",
    )
    # |f(t)|=|3-t| first falls below .4 at the t=2.75 checkpoint.
    np.testing.assert_allclose(result.values, 0.75)


@pytest.mark.parametrize("analysis", ["poincare_section", "return_map"])
def test_section_and_section_return_map_honor_explicit_initial_state(analysis: str) -> None:
    system = _HarmonicSection(ic=[0.0, 1.0])
    common = dict(plane=(0, 0.0), ic=[0.0, 2.0], dt=0.05)
    if analysis == "poincare_section":
        result = ts.analysis.poincare_section(system, crossings=3, **common)
        values = result.y[:, 1]
    else:
        result = ts.analysis.return_map(system, 1, kind="poincare", n=3, **common)
        values = result.values
    # x'=y, y'=-x preserves x²+y²; upward x=0 crossings have y=+radius.
    np.testing.assert_allclose(values, 2.0, atol=1e-6)
    np.testing.assert_array_equal(system.ic, [0.0, 1.0])


def test_automatic_section_is_chosen_from_the_requested_orbit() -> None:
    class ShiftedOscillator(ts.ContinuousSystem):
        dim = 3
        params = {}
        _default_ic = [0.0, 1.0, 0.0]

        @staticmethod
        def _equations(y, t):
            return [2.0 * y(1), -0.5 * (y(0) - y(2)), 0.0]

    result = ts.analysis.poincare_section(
        ShiftedOscillator(), ic=[10.0, 2.0, 10.0], crossings=2, max_time=20.0, dt=0.05
    )
    axis, offset = result.meta["plane"]
    assert axis == 0  # x has twice the oscillation amplitude of y.
    assert 8.0 < offset < 12.0  # the requested center, not the default orbit at 0
    np.testing.assert_allclose(result.y[:, 2], 10.0)
    np.testing.assert_allclose(
        (result.y[:, 0] - 10.0) ** 2 + 4.0 * result.y[:, 1] ** 2, 16.0, atol=2e-6
    )


@pytest.mark.parametrize("analysis", ["ftle_field", "escape_time_field", "transient_time_field"])
def test_integrating_fields_execute_the_declared_and_requested_native_solvers(
    analysis: str, monkeypatch
) -> None:
    native = pytest.importorskip("tsdynamics._rust")
    original = native.integrate_ensemble_final
    executed = []

    def record(*args):
        executed.append(args[-5])  # Method at the actual native kernel boundary.
        return original(*args)

    monkeypatch.setattr(native, "integrate_ensemble_final", record)
    settings = dict(xlim=(0.5, 1.0), ylim=(0.5, 1.0), grid=3, final_time=0.01, dt=0.001)
    if analysis == "escape_time_field":
        settings.update(escape=lambda states: states[:, 0] <= 0.4, chunks=4)
    elif analysis == "transient_time_field":
        settings.update(settled=lambda states: states[:, 0] <= 0.4, chunks=4)
    function = getattr(ts.analysis, analysis)
    default = function(_StiffField(), **settings)
    assert executed and set(executed) == {"bdf"}
    assert default.meta["solver"] == "bdf"
    if analysis == "ftle_field":
        np.testing.assert_allclose(default.values, -1.0, atol=1e-5)
    executed.clear()
    requested = function(_StiffField(), solver="rk4", **settings)
    assert executed and set(executed) == {"rk4"}
    assert requested.meta["solver"] == "rk4"
    with pytest.raises(ValueError, match="pass only solver"):
        function(_StiffField(), solver="rk4", method="bdf", **settings)


@pytest.mark.parametrize("kind, sign", [("max", 1.0), ("min", -1.0)])
def test_return_map_refines_an_irregularly_sampled_quadratic_in_actual_time(
    kind: str, sign: float
) -> None:
    times = np.array([0.0, 0.6, 1.4])
    signal = sign * (2.0 - (times - 0.8) ** 2)
    result = ts.analysis.return_map(ts.Trajectory(t=times, y=signal), kind=kind)
    np.testing.assert_allclose(result.values, [sign * 2.0], atol=1e-14)
    np.testing.assert_allclose(result.times, [0.8], atol=1e-14)


def test_ftle_counts_growth_out_of_the_displayed_initial_slice() -> None:
    horizon = 0.5
    field = ts.analysis.ftle_field(
        _OutOfPlaneShear(), xlim=(-1.0, 1.0), ylim=(-2.0, 2.0), grid=5, final_time=horizon
    )
    # phi_T(x,y,0)=(x,y,2Tx), hence F^T F=diag(1+4T^2,1).
    expected = np.log1p(4 * horizon**2) / (2 * horizon)
    np.testing.assert_allclose(field.values, expected, rtol=1e-10, atol=1e-12)
    assert field.meta["plane"] == (0, 1)
    assert field.meta["deformation_output"] == "full_state"


def test_ftle_preserves_the_analytic_two_dimensional_shear_result() -> None:
    horizon = 0.5
    field = ts.analysis.ftle_field(
        _PlanarShear(), xlim=(-1.0, 1.0), ylim=(-2.0, 2.0), grid=5, final_time=horizon
    )
    b = 2 * horizon
    singular_value = (np.sqrt(4 + b**2) + abs(b)) / 2
    np.testing.assert_allclose(field.values, np.log(singular_value) / horizon, rtol=1e-10)


def test_an_endpoint_return_in_a_forced_flow_is_not_an_autonomous_cycle() -> None:
    # x(t)=sin(t) returns to x(0) at pi but has period 2*pi. The autonomous
    # shooting equations previously accepted pi and reported a stable cycle.
    assert np.sin(np.pi) == pytest.approx(np.sin(0.0), abs=1e-15)
    assert np.sin(np.pi / 2 + np.pi) != pytest.approx(np.sin(np.pi / 2))
    with pytest.raises(TypeError, match="autonomous-flow shooting"):
        ts.analysis.periodic_orbits(_ForcedReturn(), np.pi, ic=[0.0])


@pytest.mark.parametrize("method", ["fft", "autocorrelation"])
def test_period_estimation_refuses_irregular_sampling_and_accepts_explicit_spacing(
    method: str,
) -> None:
    uniform = np.arange(1000) * 0.01
    signal = np.sin(2 * np.pi * uniform / 0.8)
    irregular = uniform + 0.001 * np.sin(np.arange(uniform.size))
    recording = ts.Trajectory(t=irregular, y=signal)
    with pytest.raises(ValueError, match="not uniformly spaced"):
        ts.analysis.estimate_period(recording, method=method)
    explicit = ts.analysis.estimate_period(recording, dt=0.01, method=method)
    assert float(explicit) == pytest.approx(0.8, rel=0.01)


@pytest.mark.parametrize("method", ["fft", "autocorrelation"])
def test_period_estimation_is_independent_of_small_measurement_units(method: str) -> None:
    times = np.arange(1000) * 0.01
    signal = 1e-12 * np.sin(2 * np.pi * times / 0.8)
    assert float(ts.analysis.estimate_period(signal, dt=0.01, method=method)) == pytest.approx(
        0.8, rel=0.01
    )


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan])
def test_period_estimation_requires_a_finite_positive_sampling_step(dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        ts.analysis.estimate_period(np.sin(np.arange(100)), dt=dt)


def test_period_estimation_does_not_assign_a_period_to_undetrended_constant_data() -> None:
    with pytest.raises(ValueError, match="signal is constant"):
        ts.analysis.estimate_period(np.full(100, 3.0), detrend=False, method="fft")


class _IdentityMap(ts.DiscreteMap):
    dim = 1
    params = {}

    @staticmethod
    def _step(x):
        return x


class _TranslationMap(ts.DiscreteMap):
    dim = 2
    params = {"shift": 0.2}

    @staticmethod
    def _step(x, shift):
        return np.array([x[0] + shift, x[1]])

    @staticmethod
    def _jacobian(x, shift):
        return np.eye(2)


def test_exact_zero_expansion_is_applicable_when_the_region_has_survivors() -> None:
    result = ts.analysis.expansion_entropy(_IdentityMap(), [(0.0, 1.0)], n_samples=20, n=5)
    assert float(result) == 0.0
    assert result.stderr == 0.0
    assert result.n_survivors == 20
    assert result.applicable
    assert "not applicable" not in repr(result)


def test_expansion_entropy_preserves_a_ball_restraining_region() -> None:
    ball = ts.data.Ball([0.0, 0.0], r=1.0)
    draw = ts.data.sampler(ball, seed=7)
    initial = np.array([draw() for _ in range(200)])
    # Translation on a convex ball stays inside at every iterate iff its final
    # point is inside; no derivative direction expands, so G(DF^n) is exactly1.
    expected = int(np.count_nonzero(np.linalg.norm(initial + [0.8, 0.0], axis=1) <= 1.0))
    result = ts.analysis.expansion_entropy(_TranslationMap(), ball, n_samples=200, n=4, seed=7)
    assert result.n_survivors == expected
    assert result.ordinate[-1] == pytest.approx(np.log(expected / 200))


def test_windowed_rqa_plot_can_select_the_measure_the_result_already_exposes() -> None:
    result = ts.analysis.windowed_rqa(
        np.random.default_rng(0).normal(size=120), window=30, threshold=0.3
    )
    spec = result.__plot_spec__(measure="laminarity")
    np.testing.assert_array_equal(spec.layers[0].data["y"], result.laminarity)
    assert spec.y.label == "laminarity"
    assert "laminarity" in spec.title
    with pytest.raises(ValueError, match="unknown RQA measure"):
        result.__plot_spec__(measure="missing")


def test_an_interior_fnn_minimum_is_not_reported_as_successful_saturation() -> None:
    data = np.random.default_rng(0).normal(size=500)
    result = ts.analysis.false_nearest_neighbors(data, delay=1, max_dim=10)
    assert result.dimension < 10  # Best candidate lies inside the scanned range.
    assert np.min(result.fnn_fraction) > 0.01  # None met the actual criterion.
    assert not result.saturated
    assert result.meta["threshold_reached"] is False
    assert "criterion not met" in repr(result)
    assert result.to_dict(full=True)["saturated"] is False


@pytest.mark.parametrize("analysis", ["cao_dimension", "false_nearest_neighbors"])
def test_negative_theiler_windows_cannot_admit_self_neighbours(analysis: str) -> None:
    data = np.random.default_rng(0).normal(size=200)
    with pytest.raises(ValueError, match="theiler must be non-negative"):
        getattr(ts.analysis, analysis)(data, delay=1, max_dim=3, theiler=-1)


@pytest.mark.parametrize("keyword", ["dimension", "delay"])
def test_multivariate_embedding_rejects_fractional_per_channel_counts(keyword: str) -> None:
    data = np.arange(40).reshape(20, 2)
    settings = {"dimension": [2, 2], "delay": [1, 1], keyword: [2.5, 2]}
    with pytest.raises(ValueError, match="whole number"):
        ts.analysis.embed(data, **settings)


def test_multivariate_embedding_broadcasts_whole_numeric_counts_consistently() -> None:
    data = np.arange(40).reshape(20, 2)
    integral = ts.analysis.embed(data, dimension=2, delay=1)
    equivalent = ts.analysis.embed(data, dimension=2.0, delay=1.0)
    np.testing.assert_array_equal(np.asarray(integral), np.asarray(equivalent))


def test_renyi_result_exposes_the_actual_fit_when_informative_scales_have_holes() -> None:
    points = np.random.default_rng(0).uniform(size=(100, 2))
    scales = np.geomspace(0.08, 0.25, 100)[:9]
    result = ts.analysis.generalized_dimension(
        points, 2, scales=scales, min_occupancy=100 / 70, min_resolution=1.0
    )
    assert result.abscissa.size < scales.size
    lo, hi = result.fit_region
    x, y = result.abscissa[lo : hi + 1], result.ordinate[lo : hi + 1]
    slope, intercept = np.polyfit(x, y, 1)
    assert float(result) == pytest.approx(slope, abs=1e-10)
    assert result.intercept == pytest.approx(intercept, abs=1e-10)
    assert result.n_fit == x.size


def test_delay_access_is_not_confused_with_external_forcing() -> None:
    from tsdynamics.analysis._tangent import flow_has_explicit_time

    assert flow_has_explicit_time(_ForcedDelay())
    assert not flow_has_explicit_time(_ForcedDelay(amplitude=0.0))
    assert not flow_has_explicit_time(ts.systems.MackeyGlass())
    result = ts.analysis.lyapunov_spectrum(
        _ForcedDelay(), ic=[0.0], k=1, final_time=5.0, transient=1.0, dt=0.01
    )
    assert result.meta["autonomous"] is False
    assert np.isnan(result.kaplan_yorke)


@pytest.mark.parametrize("method", ["newton", "interval"])
def test_fixed_points_does_not_freeze_external_forcing_at_time_zero(method: str) -> None:
    # The old solver reported x=0, stable, by solving only f(x, 0)=0.
    # That state immediately moves when the external forcing advances.
    with pytest.raises(TypeError, match="depends explicitly on time"):
        ts.analysis.fixed_points(_AdditiveForcing(), region=[(-2.0, 2.0)], method=method)


def test_disabling_the_forcing_restores_the_autonomous_equilibrium_problem() -> None:
    result = ts.analysis.fixed_points(
        _AdditiveForcing(amplitude=0.0), region=[(-2.0, 2.0)], n_seeds=2
    )
    np.testing.assert_allclose(np.asarray(result), [[0.0]])
    assert result.is_stable.tolist() == [True]


@pytest.mark.parametrize("keyword", ["final_time", "dt", "transient"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0, True, "0.1"])
def test_flow_lyapunov_rejects_nonphysical_time_settings(keyword: str, value: object) -> None:
    with pytest.raises(ValueError, match=keyword):
        ts.analysis.lyapunov_spectrum(_TimeStretch(), **{keyword: value})


def test_a_single_positive_flow_exponent_is_not_mistaken_for_its_neutral_direction() -> None:
    result = ts.analysis.lyapunov_spectrum(
        ts.systems.Lorenz(), k=1, final_time=100.0, transient=50.0, ic=[1.0, 1.0, 1.0]
    )
    assert float(result[0]) > 0.5
    assert result.n_positive == 1
    assert result.chaotic is None
    assert result.regime == "indeterminate"
    assert result.verdict.startswith("indeterminate")
    assert np.isnan(result.kaplan_yorke)
    assert "D_KY" not in repr(result)
    assert result.to_dict(full=True)["kaplan_yorke"] is None
    with pytest.raises(ValueError, match="truncated spectrum"):
        ts.analysis.kaplan_yorke_dimension(result)


def test_a_truncated_spectrum_can_define_a_dimension_once_its_cumulative_sum_closes() -> None:
    from tsdynamics.analysis.results import LyapunovSpectrum

    result = LyapunovSpectrum(values=np.array([0.3, 0.0, -2.0]), meta={"spectrum_complete": False})
    assert result.kaplan_yorke == pytest.approx(2.15)
    assert float(ts.analysis.kaplan_yorke_dimension(result)) == pytest.approx(2.15)


@pytest.mark.parametrize(
    ("peak", "start", "expected"),
    [
        (1e10, 1e10, False),
        (1e10, 1.0, True),
        (1e6, 1.0, False),
        (np.inf, 1.0, True),
        (np.nan, 1.0, True),
        (1e10, None, True),  # Legacy records do not contain a starting scale.
    ],
)
def test_lyapunov_escape_evidence_uses_growth_as_well_as_physical_scale(
    peak: float, start: float | None, expected: bool
) -> None:
    from tsdynamics.analysis.results import LyapunovSpectrum

    metadata = {"orbit_peak": peak}
    if start is not None:
        metadata["orbit_start"] = start
    result = LyapunovSpectrum(values=np.array([-1.0]), meta=metadata)
    assert result.unbounded is expected


@pytest.mark.parametrize("q", [2000.0, 1e300])
def test_large_finite_renyi_orders_do_not_underflow_the_partition_function(q: float) -> None:
    from tsdynamics.analysis.dimensions.generalized import _partition_ordinate

    # For two equal masses the Renyi entropy is log(2) at every order,
    # although directly raising 0.5 to these powers underflows to zero.
    assert _partition_ordinate(np.array([1, 1]), 2, q) == pytest.approx(-np.log(2.0))


def test_rqa_plot_does_not_clip_entropy_above_one_nat() -> None:
    from dataclasses import replace

    result = replace(
        ts.analysis.rqa(np.sin(np.linspace(0.0, 20.0, 100)), recurrence_rate=0.1),
        diagonal_entropy=2.0,
    )
    spec = result.__plot_spec__()
    assert spec.y.limits is not None
    assert spec.y.limits[1] > result.diagonal_entropy


def test_gali_does_not_call_an_attracting_limit_cycle_chaotic() -> None:
    result = ts.analysis.gali(
        ts.systems.VanDerPol(), k=2, ic=[2.0, 0.0], final_time=30.0, transient=20.0
    )
    assert result.is_chaotic()  # The legacy predicate detects alignment collapse.
    assert result.chaotic is None
    assert "alignment collapsed" in repr(result)
    assert "chaos indeterminate" in repr(result)


def test_a_center_is_not_described_as_an_unstable_equilibrium() -> None:
    from tsdynamics.analysis.results import FixedPoint, FixedPointSet

    point = FixedPoint(
        x=np.zeros(2), eigenvalues=np.array([1j, -1j]), continuous=True, stable=False
    )
    assert "nonhyperbolic" in point.verdict
    assert "unstable" not in point.verdict
    collection = FixedPointSet(items=(point,))
    assert "0 unstable" in repr(collection)
    assert "1 nonhyperbolic" in repr(collection)


def test_positive_finite_time_shear_does_not_label_a_periodic_orbit_chaotic() -> None:
    result = ts.analysis.lyapunov_spectrum(
        ts.systems.LotkaVolterra(ic=[1.0, 1.0]), k=1, final_time=40.0, transient=20.0
    )
    assert result[0] > 0.0
    assert result.chaotic is None
    assert result.regime == "indeterminate"


def test_fixed_point_repr_and_subsets_use_the_same_neutral_boundary() -> None:
    from tsdynamics.analysis.results import FixedPoint, FixedPointSet

    point = FixedPoint(x=np.zeros(1), eigenvalues=np.array([-1e-15]), continuous=True, stable=True)
    collection = FixedPointSet(items=(point,))
    assert "1 stable, 0 unstable" in repr(collection)
    assert collection.n_stable == len(collection.stable) == 1
    assert collection.to_dict(full=True)["n_stable"] == 1


def test_public_spectrum_sorts_finite_time_qr_rates_without_reordering_live_frames(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        _TimeStretch, "_lyapunov_spectrum", lambda self, **kw: np.array([-0.5, 0.7])
    )
    result = ts.analysis.lyapunov_spectrum(_TimeStretch(), final_time=1.0)
    np.testing.assert_array_equal(result.exponents, [0.7, -0.5])


def test_fixed_point_overlay_keeps_actual_saddles_distinct_from_neutral_points() -> None:
    from tsdynamics.analysis.results import FixedPoint, FixedPointSet

    saddle = FixedPoint(x=np.zeros(2), eigenvalues=np.array([-1.0, 1.0]), continuous=True)
    center = FixedPoint(x=np.ones(2), eigenvalues=np.array([-1j, 1j]), continuous=True)
    spec = FixedPointSet(items=(saddle, center)).__plot_spec__()
    assert {layer.label for layer in spec.layers} == {"unstable", "nonhyperbolic"}


def test_mutual_information_delay_is_independent_of_logarithm_units() -> None:
    series = np.sin(np.linspace(0.0, 40.0 * np.pi, 2000))
    nats = ts.analysis.mutual_information(series, max_delay=60, bins=8)
    scaled = ts.analysis.mutual_information(series, max_delay=60, bins=8, base=np.exp(100.0))
    np.testing.assert_allclose(np.asarray(scaled), np.asarray(nats) / 100.0)
    assert nats.optimal_lag > 1
    assert scaled.optimal_lag == nats.optimal_lag


@pytest.mark.parametrize("base", [0.0, 0.5, 1.0, np.nan, np.inf])
def test_mutual_information_rejects_invalid_logarithm_base(base: float) -> None:
    with pytest.raises(ValueError, match="base"):
        ts.analysis.mutual_information(np.arange(30.0), base=base)


def test_recurrence_rejects_a_constant_multicomponent_state() -> None:
    points = np.tile([2.0, -7.0], (64, 1))
    with pytest.raises(ValueError, match="constant"):
        ts.analysis.recurrence_matrix(points, recurrence_rate=0.05)


def test_zero_one_rejects_a_three_dimensional_observable() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        ts.analysis.zero_one_test(np.arange(300.0).reshape(10, 10, 3), oversampling="ignore")


@pytest.mark.parametrize("kwargs", [{"dt": 0.1}, {"final_time": 100.0}])
def test_zero_one_refuses_flow_horizons_on_maps(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="map uses n"):
        ts.analysis.zero_one_test(ts.systems.Logistic(), **kwargs)


@pytest.mark.parametrize("analysis", ["gali", "expansion_entropy"])
@pytest.mark.parametrize("n_internal", [0, -1, 0.5, True])
def test_flow_chaos_indicators_cannot_disable_the_integrator(
    analysis: str, n_internal: float
) -> None:
    system = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    kwargs = {"k": 2, "transient": 0.0} if analysis == "gali" else {"n_samples": 2}
    if analysis == "expansion_entropy":
        kwargs["region"] = [(-100.0, 100.0)] * 3
    with pytest.raises(ValueError, match="n_internal.*positive integer"):
        getattr(ts.analysis, analysis)(
            system, final_time=0.2, dt=0.1, n_internal=n_internal, **kwargs
        )


def test_rosenstein_keeps_references_when_the_unobservable_tail_clusters_near_them() -> None:
    # Only the first 80 states have all 160 forward images.  The tail's dense
    # cluster must never crowd an eligible reference's true neighbour out of
    # the query: all 80 distinct reference states have a distance-one neighbour.
    series = np.concatenate([np.arange(80.0), np.linspace(40.01, 40.02, 160)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScalingRegionWarning)
        result = ts.analysis.lyapunov_from_data(
            series, dimension=1, delay=1, theiler=0, k_max=160, method="rosenstein", fit=(0, 3)
        )
    assert result.n_reference == 80
    assert result.ordinate[0] == pytest.approx(0.0, abs=1e-14)


def test_scaling_fit_is_invariant_to_large_coordinate_offsets() -> None:
    x = 1e9 + np.linspace(0.0, 1.0, 30)
    y = -2.0 * x + 3.0
    fit = fit_scaling_region(x, y)
    assert fit.slope == pytest.approx(-2.0, abs=1e-10)
    assert fit.intercept == pytest.approx(3.0, abs=1e-6)


def test_interval_fixed_points_honors_the_requested_deduplication_distance() -> None:
    points = ts.analysis.fixed_points(
        ts.systems.Henon(), region=[(-3.0, 3.0)] * 2, method="interval", dedup_tol=10.0
    )
    assert len(points) == 1


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("t0", [0.0, 1.0])
def test_ftle_reverses_the_forcing_clock_for_a_nonautonomous_flow(
    backward: bool, t0: float
) -> None:
    horizon = 0.5
    field = ts.analysis.ftle_field(
        _TimeStretch(),
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        grid=3,
        final_time=horizon,
        t0=t0,
        backward=backward,
    )
    # x(t) = x(t0) exp((t*t - t0*t0)/2), while y is unchanged.
    # The largest stretching is max(exp(integral), 1), in either direction.
    integral = (-t0 * horizon if backward else t0 * horizon) + horizon**2 / 2.0
    np.testing.assert_allclose(field.values, max(integral, 0.0) / horizon, atol=1e-7)


def test_trace_determinant_handles_a_flow_without_equilibria() -> None:
    result = ts.analysis.trace_determinant(
        _NoEquilibrium(), region=[(-1.0, 1.0)] * 2, n_seeds=2, max_iter=2
    )
    assert result.points.shape == (0, 2)
    assert result.trace.size == 0
    assert result.trace_range == (-2.0, 2.0)


def test_wada_verdict_uses_the_largest_requested_radius_independent_of_order() -> None:
    labels = np.tile(np.repeat([1, 2, 3], 3), (9, 1))
    ordered = ts.analysis.wada_property(labels, radii=(1, 4), threshold=0.5)
    reversed_radii = ts.analysis.wada_property(labels, radii=(4, 1), threshold=0.5)
    assert ordered.W > 0.0
    assert reversed_radii.W == ordered.W
    assert reversed_radii.wada == ordered.wada


@pytest.mark.parametrize("base", [0.5, 1.0, np.nan, np.inf])
def test_basin_entropy_rejects_nonentropy_logarithm_bases(base: float) -> None:
    labels = np.tile([1, 2], (10, 5))
    with pytest.raises(ValueError, match="base"):
        ts.analysis.basin_entropy(labels, base=base)


def test_basin_entropy_qualifies_a_coarse_partition_of_smooth_stripes() -> None:
    # Two straight parallel boundaries. One coarse box sees three colours even
    # though neither boundary is fractal; refinement resolves each separately.
    stripes = np.tile(np.repeat([1, 2, 3], 4), (12, 1))
    coarse = ts.analysis.basin_entropy(stripes, box_size=12)
    fine = ts.analysis.basin_entropy(stripes, box_size=3)
    assert coarse.sbb == pytest.approx(np.log(3.0))
    assert coarse.fractal_boundary is True
    assert fine.fractal_boundary is False
    assert "criterion met" in repr(coarse)
    assert "resolution convergence" in repr(coarse)


@pytest.mark.parametrize("analysis", ["attractors", "basin_fractions"])
@pytest.mark.parametrize("n_seeds", [0, -1, 0.5, True])
def test_basin_sampling_rejects_empty_or_fractional_seed_counts(analysis, n_seeds) -> None:
    with pytest.raises(ValueError, match="n_seeds.*positive integer"):
        getattr(ts.analysis, analysis)(ts.systems.Logistic(), [(0.1, 0.9)], n_seeds=n_seeds)


@pytest.mark.parametrize("radii", [(), (0, 1), (-1, 2), (0.5, 2)])
def test_wada_rejects_invalid_grid_radii(radii: tuple[int, ...]) -> None:
    labels = np.tile(np.repeat([1, 2, 3], 3), (9, 1))
    with pytest.raises(ValueError, match="radii"):
        ts.analysis.wada_property(labels, radii=radii)


def test_uncertainty_exponent_requires_distinct_physical_scales() -> None:
    labels = np.tile(np.repeat([1, 2], 10), (20, 1))
    with pytest.raises(ValueError, match="distinct radii"):
        ts.analysis.uncertainty_exponent(labels, radii=(1, 1))


@pytest.mark.parametrize("flatness", [0.5, np.nan, np.inf])
def test_scaling_fit_reports_invalid_flatness_before_window_selection(flatness: float) -> None:
    with pytest.raises(ValueError, match="flatness"):
        fit_scaling_region(np.arange(10.0), np.arange(10.0), tol=flatness)
