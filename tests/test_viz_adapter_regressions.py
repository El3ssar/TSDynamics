"""Scientific quantities and sampling must survive the plotting adapters."""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidInputError, InvalidParameterError


@pytest.mark.parametrize("component", [1, "y"])
def test_orbit_diagram_plots_the_selected_observable(component):
    model = ts.systems.Henon()
    values = [1.1, 1.2]
    measured = ts.analysis.orbit_diagram(
        model, "a", values, components=component, points_per_value=12, transient=50
    )
    plot = ts.plot(
        model,
        "orbit_diagram",
        param="a",
        values=values,
        components=component,
        points=12,
        transient=50,
    )
    expected = np.concatenate([points[:, 0] for points in measured.points])
    np.testing.assert_array_equal(plot.layers[0].data["y"], expected)
    assert plot.y.label == "y"


def test_ensemble_percentiles_refuse_different_physical_sample_times():
    first = ts.Trajectory(np.array([0.0, 1.0, 2.0]), np.array([[1.0], [2.0], [3.0]]))
    shifted = ts.Trajectory(np.array([0.0, 2.0, 4.0]), np.array([[1.0], [2.0], [3.0]]))
    with pytest.raises(InvalidInputError, match="same time grid"):
        ts.plot([first, shifted], "ensemble_fan")


def test_ensemble_percentiles_keep_the_shared_time_axis_and_real_values():
    time = np.array([2.0, 3.0, 4.0])
    first = ts.Trajectory(time, np.array([[1.0], [2.0], [3.0]]))
    second = ts.Trajectory(time, np.array([[3.0], [4.0], [5.0]]))
    plot = ts.plot([first, second], "ensemble_fan", spread=(25, 75))
    band, median = plot.layers
    np.testing.assert_array_equal(band.data["x"], time)
    np.testing.assert_array_equal(band.data["lo"], [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(band.data["hi"], [2.5, 3.5, 4.5])
    np.testing.assert_array_equal(median.data["y"], [2.0, 3.0, 4.0])


def test_the_actual_ensemble_batch_is_one_aggregate_not_a_single_trajectory():
    from tsdynamics.derived.ensemble import TrajectoryBatch
    from tsdynamics.viz.transforms import get

    time = np.array([2.0, 3.0, 4.0])
    first = ts.Trajectory(time, np.array([[1.0], [2.0], [3.0]]))
    second = ts.Trajectory(time, np.array([[3.0], [4.0], [5.0]]))
    batch = TrajectoryBatch([first, second])
    assert get("ensemble_fan").accepts_subject(batch)
    assert not get("time_series").accepts_subject(batch)
    plot = ts.plot(batch, "ensemble_fan", spread=(25, 75))
    band, median = plot.layers
    np.testing.assert_array_equal(band.data["x"], time)
    np.testing.assert_array_equal(band.data["lo"], [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(band.data["hi"], [2.5, 3.5, 4.5])
    np.testing.assert_array_equal(median.data["y"], [2.0, 3.0, 4.0])


@pytest.mark.parametrize("component, label", [(0, "x"), (2, "z"), ("z", "z")])
def test_a_section_cascade_keeps_both_observable_values_and_section_context(component, label):
    section = ts.systems.Rossler().poincare("y", 0.0, direction="up")
    values = [4.0, 5.0]
    measured = ts.analysis.orbit_diagram(
        section, "c", values, components=component, points_per_value=6, transient=30
    )
    plot = ts.plot(
        section,
        "orbit_diagram",
        param="c",
        values=values,
        components=component,
        points=6,
        transient=30,
    )
    assert plot.y.label == f"{label} at y = 0"
    np.testing.assert_array_equal(
        plot.layers[0].data["y"], np.concatenate([points[:, 0] for points in measured.points])
    )


def test_an_ordinary_trajectory_list_still_means_a_comparison():
    time = np.arange(3.0)
    first = ts.Trajectory(time, np.array([[1.0], [2.0], [3.0]]))
    second = ts.Trajectory(time, np.array([[3.0], [4.0], [5.0]]))
    plot = ts.plot([first, second], "time_series")
    assert len(plot.layers) == 2
    np.testing.assert_array_equal(plot.layers[0].data["y"], first.y[:, 0])
    np.testing.assert_array_equal(plot.layers[1].data["y"], second.y[:, 0])


@pytest.mark.parametrize("spread", [(90, 10), (-1, 99), (1, 101), (50, 50)])
def test_ensemble_spread_names_a_real_ordered_percentile_interval(spread):
    with pytest.raises(InvalidParameterError, match="0 <= low < high <= 100"):
        ts.plot(np.ones((2, 3)), "ensemble_fan", spread=spread)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
@pytest.mark.parametrize("component", [None, "u", "v", 0, 1, -1, -2])
def test_spatial_field_labels_identify_the_block_drawn_in_stills_and_movies(shape, component):
    cells = int(np.prod(shape))
    values = np.concatenate((np.ones((3, cells)), 2 * np.ones((3, cells))), axis=1)
    trajectory = ts.Trajectory(
        np.arange(3.0), values, meta={"field_shape": shape, "field_labels": ("u", "v")}
    )
    plot = ts.plot(trajectory, "spatial_field", components=component)
    selected = "u" if component in ("u", 0, -2) else "v"
    layer = plot.layers[0]
    expected = 1.0 if selected == "u" else 2.0
    assert np.all(layer.data["frames"] == expected)
    assert np.all(layer.data["z" if len(shape) == 2 else "y"] == expected)
    label = plot.colorbar.label if len(shape) == 2 else plot.y.label
    assert label == selected


def test_spatial_field_does_not_wrap_an_out_of_range_positive_block_to_another_quantity():
    trajectory = ts.Trajectory(
        np.arange(3.0), np.ones((3, 6)), meta={"field_shape": (3,), "field_labels": ("u", "v")}
    )
    with pytest.raises(InvalidParameterError, match="out of range"):
        ts.plot(trajectory, "spatial_field", components=2)


def test_neutral_eigenvalues_are_not_labelled_as_expanding_or_unstable():
    fixed = ts.analysis.results.FixedPoint(
        x=np.zeros(2), eigenvalues=np.array([0.0, -1.0]), continuous=True, stable=False
    )
    plot = ts.plot(fixed, "eigenvalue_plane")
    assert "nonhyperbolic" in plot.title
    assert "unstable equilibrium" not in plot.title
    assert all("expanding" not in (layer.label or "") for layer in plot.layers)


def test_a_jacobian_at_an_arbitrary_state_does_not_claim_it_is_an_equilibrium():
    plot = ts.plot(ts.systems.Lorenz(), "eigenvalue_plane", at=[1.0, 1.0, 1.0])
    assert "Jacobian at" in plot.title
    assert "equilibrium" not in plot.title


def test_a_data_lyapunov_scaling_view_keeps_time_units_instead_of_claiming_log_time():
    result = ts.analysis.results.LyapunovFromData(
        estimate=0.5,
        abscissa=np.arange(5.0),
        ordinate=2 + 0.5 * np.arange(5.0),
        intercept=2.0,
        fit_region=(0, 4),
        trusted=False,
    )
    for view in ("fit", "residuals", "local_slopes"):
        plot = ts.plot(result, "scaling_fit", view=view)
        assert plot.x.label == "time"
        curve = next(
            layer for layer in plot.layers if layer.label in ("curve", "residual", "local slope")
        )
        np.testing.assert_array_equal(curve.data["x"], result.times)


def test_a_sample_lag_on_an_irregular_axis_does_not_invent_constant_physical_time():
    trajectory = ts.Trajectory(
        np.array([0.0, 0.1, 0.3, 0.6]), np.arange(4.0).reshape(-1, 1), meta={"dt": 0.1}
    )
    plot = ts.plot(trajectory, "delay_embedding", delay=1)
    assert "1 samples" in plot.y.label
    assert "0.1" not in plot.y.label
    np.testing.assert_array_equal(plot.layers[0].data["x"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(plot.layers[0].data["y"], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("transform", ["flow_speed", "vector_field"])
def test_slice_speed_is_labelled_as_projected_when_the_full_state_still_moves(
    transform, monkeypatch
):
    from tsdynamics import registry

    monkeypatch.setitem(registry._BY_NAME, "OutOfPlaneMotion", [])

    class OutOfPlaneMotion(ts.ContinuousSystem):
        variables = ("x", "y", "z")

        @staticmethod
        def _equations(y, t):
            return (0 * y(0), 0 * y(1), 1 + 0 * y(2))

    options = {"color_by": "speed"} if transform == "vector_field" else {}
    geometry = ts.viz.geometry(
        OutOfPlaneMotion(), transform, domain=((-1, 1), (-1, 1)), grid=3, **options
    )
    assert geometry.color_label == "in-plane speed"
    assert np.all(geometry["c"] == 0.0)


def test_a_stable_jacobian_block_does_not_claim_full_equilibrium_stability(monkeypatch):
    from tsdynamics import registry

    monkeypatch.setitem(registry._BY_NAME, "CoupledLinearSlice", [])

    class CoupledLinearSlice(ts.ContinuousSystem):
        variables = ("x", "y", "z")

        @staticmethod
        def _equations(y, t):
            return (-y(0) + 3 * y(2), -y(1), 3 * y(0) - y(2))

    with pytest.warns(ts.viz.VisualizationDegraded, match="2x2"):
        plot = ts.plot(CoupledLinearSlice(), "trace_determinant", points=[[0.0, 0.0, 0.0]])
    assert "frozen 2-D block" in plot.title
    assert plot.meta["stability_scope"] == "frozen_block"
    assert any("2-D block" in (layer.label or "") for layer in plot.layers)


@pytest.mark.parametrize("time_scale", [1e-6, 1e6])
def test_curvature_of_a_geometric_orbit_is_independent_of_time_units(time_scale):
    angle = np.linspace(0.0, 2 * np.pi, 201)
    points = np.column_stack((np.cos(angle), np.sin(angle)))
    reference = ts.plot(ts.Trajectory(angle, points), color_by="curvature")
    rescaled = ts.plot(ts.Trajectory(angle * time_scale, points), color_by="curvature")
    expected = reference.layers[0].data["c"][2:-2]
    actual = rescaled.layers[0].data["c"][2:-2]
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
    np.testing.assert_allclose(actual, 1.0, rtol=1e-3)
