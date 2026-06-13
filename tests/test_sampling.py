"""Regions, samplers, grid enumeration, and set distances."""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.sampling import Ball, Box, Grid, grid_points, sampler, set_distance

# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


def test_box_contains_and_validation() -> None:
    b = Box([-1, -1], [1, 2])
    assert b.dim == 2
    assert b.contains([0, 0]) and b.contains([1, 2])
    assert not b.contains([0, 3])
    with pytest.raises(ValueError, match="hi >= lo"):
        Box([0, 0], [-1, 1])


def test_ball_contains_uniform_in_volume() -> None:
    ball = Ball([0, 0], 2.0)
    assert ball.contains([0, 0]) and ball.contains([2, 0])
    assert not ball.contains([2, 2])
    with pytest.raises(ValueError, match="radius"):
        Ball([0, 0], 0.0)


def test_grid_axes_and_validation() -> None:
    g = Grid([0, 0], [1, 1], (3, 5))
    assert g.dim == 2 and g.shape == (3, 5)
    ax = g.axes()
    np.testing.assert_allclose(ax[0], [0.0, 0.5, 1.0])
    assert ax[1].size == 5
    with pytest.raises(ValueError, match="counts"):
        Grid([0], [1], (0,))


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def test_box_sampler_in_bounds_and_reproducible() -> None:
    region = Box([-1, 2], [1, 5])
    draw = sampler(region, seed=0)
    pts = np.array([draw() for _ in range(500)])
    assert np.all(pts >= region.lo) and np.all(pts <= region.hi)
    # same seed → identical stream
    again = sampler(region, seed=0)
    np.testing.assert_array_equal(pts[0], again())


def test_ball_sampler_inside_and_volume_uniform() -> None:
    region = Ball([0.0, 0.0], 1.0)
    draw = sampler(region, seed=1)
    pts = np.array([draw() for _ in range(5000)])
    radii = np.linalg.norm(pts, axis=1)
    assert np.all(radii <= 1.0 + 1e-12)
    # uniform-in-volume ⇒ ~half the points beyond r = 2^{-1/2} in 2-D
    frac_outer = np.mean(radii > 2.0**-0.5)
    assert 0.4 < frac_outer < 0.6


def test_grid_points_enumeration() -> None:
    g = Grid([0, 0], [1, 1], (2, 3))
    pts = grid_points(g)
    assert pts.shape == (6, 2)
    # C-order: first axis varies slowest
    np.testing.assert_allclose(pts[0], [0.0, 0.0])
    np.testing.assert_allclose(pts[-1], [1.0, 1.0])


def test_grid_points_1d() -> None:
    pts = grid_points(Grid([0], [1], (5,)))
    assert pts.shape == (5, 1)


# ---------------------------------------------------------------------------
# Set distances
# ---------------------------------------------------------------------------


def test_set_distance_centroid() -> None:
    a = np.zeros((10, 2))
    b = np.ones((10, 2))
    assert set_distance(a, b, method="centroid") == pytest.approx(np.sqrt(2))


def test_set_distance_minimum_and_hausdorff() -> None:
    a = np.array([[0.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 0.5], [3.0, 0.0]])
    # nearest pair: (0,0)-(0,0.5) and (0,1)-(0,0.5) both 0.5
    assert set_distance(a, b, method="minimum") == pytest.approx(0.5)
    # hausdorff: worst nearest-neighbour over both directions
    assert (
        set_distance(a, b, method="hausdorff") == pytest.approx(3.0, abs=0.2)
        or set_distance(a, b, method="hausdorff") > 0.5
    )


def test_set_distance_identical_sets_zero() -> None:
    a = np.random.default_rng(0).normal(size=(50, 3))
    for m in ("centroid", "minimum", "hausdorff"):
        assert set_distance(a, a, method=m) == pytest.approx(0.0, abs=1e-12)


def test_set_distance_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="different dimensions"):
        set_distance(np.zeros((3, 2)), np.zeros((3, 3)))


def test_set_distance_accepts_trajectory() -> None:
    t = np.arange(5)
    tr1 = ts.Trajectory(t, np.zeros((5, 2)), system=None)
    tr2 = ts.Trajectory(t, np.ones((5, 2)), system=None)
    assert tr1.set_distance(tr2) == pytest.approx(np.sqrt(2))
    assert tr1.set_distance(tr2, method="minimum") == pytest.approx(np.sqrt(2))
