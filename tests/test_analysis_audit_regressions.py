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
