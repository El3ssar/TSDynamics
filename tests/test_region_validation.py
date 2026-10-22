"""Region construction refuses invalid domains and never truncates grid sizes."""

import numpy as np
import pytest

from tsdynamics.data import Ball, Box, Grid, as_region, grid_points, region, sampler


@pytest.mark.parametrize("count", [1.5, True, np.nan, np.inf, -np.inf, 0, -1])
@pytest.mark.parametrize("route", ["grid", "region", "reader", "resolution"])
def test_grid_counts_are_validated_before_conversion(count, route):
    with pytest.raises(ValueError, match="counts.*whole numbers"):
        if route == "grid":
            Grid([-1], [1], (count,))
        elif route == "region":
            region([(-1, 1, count)])
        elif route == "reader":
            as_region([(-1, 1, count)], want_grid=True)
        else:
            grid_points([(-1, 1)], resolution=count)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_region_bounds_centers_and_radii_must_be_finite(bad):
    for build in (
        lambda: Box([bad], [1.0]),
        lambda: Box([-1.0], [bad]),
        lambda: Grid([bad], [1.0], (3,)),
        lambda: Ball([bad], 1.0),
        lambda: Ball([0.0], bad),
    ):
        with pytest.raises(ValueError, match="finite"):
            build()


@pytest.mark.parametrize("coordinates", [[], [[-1.0, 1.0]]])
def test_region_coordinates_are_nonempty_vectors(coordinates):
    with pytest.raises(ValueError, match="coordinate vector"):
        Box(coordinates, coordinates)
    with pytest.raises(ValueError, match="coordinate vector"):
        Ball(coordinates, 1.0)
    with pytest.raises(ValueError, match="coordinate vector"):
        Grid(coordinates, coordinates, (3,))


def test_direct_grid_rejects_backwards_bounds():
    with pytest.raises(ValueError, match="hi >= lo"):
        Grid([1.0], [-1.0], (3,))


def test_scalar_one_dimensional_regions_sample_coordinate_vectors():
    for domain in (Box(-1.0, 1.0), Ball(0.0, 1.0), Grid(-1.0, 1.0, (3,))):
        draw = sampler(domain, seed=42)()
        assert draw.shape == (1,)
        assert domain.contains(draw)
    np.testing.assert_array_equal(grid_points(Grid(-1.0, 1.0, (3.0,))), [[-1.0], [0.0], [1.0]])
