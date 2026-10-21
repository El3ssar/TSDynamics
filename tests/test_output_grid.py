"""Sampling grids preserve exact bounds across time units and time offsets."""

import numpy as np
import pytest

from tsdynamics._utils.grids import make_output_grid
from tsdynamics.errors import InvalidParameterError


@pytest.mark.parametrize(
    ("t0", "tf", "dt"),
    [(0.0, 1e-13, 1e-13), (0.0, 1e-12, 3e-13), (-2e-13, 2e-13, 1e-13)],
)
def test_short_windows_retain_both_bounds(t0, tf, dt):
    grid = make_output_grid(t0, tf, dt)
    assert grid.size >= 2
    assert grid[0] == t0
    assert grid[-1] == tf
    assert np.all(np.diff(grid) > 0)


def test_roundoff_sized_tail_is_snapped_to_exact_endpoint():
    tf = np.nextafter(0.3, np.inf)
    grid = make_output_grid(0.0, tf, 0.1)
    assert grid.size == 4
    assert grid[-1] == tf
    assert np.all(np.diff(grid) > 0)


def test_large_time_offset_does_not_accumulate_increment_roundoff():
    t0, tf, dt = 1e9, 1e9 + 1.0, 0.01
    grid = make_output_grid(t0, tf, dt)
    assert grid.size == 101
    np.testing.assert_array_equal(grid, t0 + np.arange(101) * dt)


@pytest.mark.parametrize("dt", [0.1, 1.0, 1.5])
def test_unrepresentable_sampling_interval_is_rejected_before_allocation(dt):
    with pytest.raises(InvalidParameterError, match="dt.*too small"):
        make_output_grid(1e16, 1e16 + 10.0, dt)


def test_large_window_roundoff_cannot_append_a_descending_endpoint():
    grid = make_output_grid(-1e16, 3.0, 1e15)
    assert grid[0] == -1e16
    assert grid[-1] == 3.0
    assert np.all(np.diff(grid) > 0)


def test_single_rounded_increment_needs_no_interior_sampling_lattice():
    t0, dt = 5896.019602918814, 6.950792873080231e-13
    tf = t0 + dt
    assert tf - t0 > dt  # subtraction recovers a full ULP after rounding
    np.testing.assert_array_equal(make_output_grid(t0, tf, dt), [t0, tf])


def test_finite_endpoints_cannot_request_infinite_window_length():
    with pytest.raises(InvalidParameterError, match="finite window"):
        make_output_grid(-1e308, 1e308, 1e308)
