"""
The data layer as the canonical home of :class:`Trajectory`.

Stream C-DATA re-homes ``Trajectory`` from ``families.base`` into
``tsdynamics.data``.  These tests guard the migration's two invariants:

1. there is exactly **one** ``Trajectory`` class, reachable from every public
   path (no diverging copy / no stale shim), and
2. its full v2 surface works when imported from its new home.

The families-facing behaviour (named components, point-set ops, provenance) is
exercised in detail in ``test_base.py``; here we only assert it is reachable
and identical through the data layer.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.data import Trajectory


class _NamedStub:
    """Minimal system stub declaring component names (no engine, no compile)."""

    variables = ("x", "y", "z")


# ---------------------------------------------------------------------------
# Canonical identity — one class, every path
# ---------------------------------------------------------------------------


def test_trajectory_canonical_home_is_data() -> None:
    """``Trajectory`` is defined in tsdynamics.data.trajectory."""
    assert Trajectory.__module__ == "tsdynamics.data.trajectory"
    assert "Trajectory" in ts.data.__all__


def test_trajectory_is_single_object_across_paths() -> None:
    """Every public import path resolves to the very same class object."""
    from tsdynamics.data.trajectory import Trajectory as TrajFromModule
    from tsdynamics.families import Trajectory as TrajFromFamilies
    from tsdynamics.families.base import Trajectory as TrajFromBase

    assert (
        ts.Trajectory
        is ts.data.Trajectory
        is TrajFromFamilies
        is TrajFromBase
        is TrajFromModule
        is Trajectory
    )


# ---------------------------------------------------------------------------
# Full surface reachable from the data layer (parity at the new home)
# ---------------------------------------------------------------------------


def test_trajectory_surface_from_data_layer() -> None:
    t = np.arange(10, dtype=float)
    y = np.arange(30, dtype=float).reshape(10, 3)
    traj = Trajectory(t, y, system=_NamedStub(), meta={"k": 1})

    # shape / unpacking
    assert (traj.dim, traj.n_steps) == (3, 10)
    t_, y_ = traj
    np.testing.assert_array_equal(t_, t)

    # named + integer component access
    np.testing.assert_array_equal(traj["y"], y[:, 1])
    np.testing.assert_array_equal(traj[["x", "z"]], y[:, [0, 2]])
    np.testing.assert_array_equal(traj.component("z"), traj.component(2))

    # transient trimming preserves meta
    tail = traj.after(5.0)
    assert tail.t[0] == 5.0 and tail.meta == {"k": 1}

    # point-set ops
    lo, hi = traj.minmax()
    np.testing.assert_array_equal(lo, y.min(axis=0))
    np.testing.assert_array_equal(hi, y.max(axis=0))

    std = traj.standardize()
    np.testing.assert_allclose(std.y.mean(axis=0), 0.0, atol=1e-12)
    assert "standardized" in std.meta


def test_trajectory_neighbors_kdtree_from_data_layer() -> None:
    y = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    traj = Trajectory(np.arange(3), y, system=None)
    _, idx = traj.neighbors([0.9, 0.1], k=1)
    assert idx == 1
    # lazy KD-tree is built and cached on first query
    assert traj._kdtree is not None


def test_trajectory_set_distance_roundtrips_through_data() -> None:
    t = np.arange(5)
    a = Trajectory(t, np.zeros((5, 2)), system=None)
    b = Trajectory(t, np.ones((5, 2)), system=None)
    assert a.set_distance(b) == pytest.approx(np.sqrt(2))
    assert a.set_distance(b, method="hausdorff") == pytest.approx(np.sqrt(2))
