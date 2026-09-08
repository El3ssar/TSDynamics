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
from tsdynamics.data import Ball, Box, Grid, Trajectory


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
    t_, y_ = traj.unpack()
    np.testing.assert_array_equal(t_, t)
    np.testing.assert_array_equal(y_, y)

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


# ---------------------------------------------------------------------------
# Region.contains: single point returns a scalar bool; a batch returns a mask
# ---------------------------------------------------------------------------


def test_box_contains_single_point_is_scalar_bool() -> None:
    box = Box([-1.0, -1.0], [1.0, 1.0])
    assert box.contains([0.0, 0.0]) is True
    assert box.contains([2.0, 0.0]) is False
    # exact return type is a Python bool (not a numpy scalar)
    assert isinstance(box.contains([0.0, 0.0]), bool)


def test_box_contains_batch_returns_per_row_mask() -> None:
    """A (n, dim) batch must yield an (n,) mask, never one conflated bool.

    Before the fix ``np.all`` collapsed the whole array, so a batch with one
    inside and one outside point silently returned a single ``False``.
    """
    box = Box([-1.0, -1.0], [1.0, 1.0])
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [-0.5, 0.5]])
    mask = box.contains(pts)
    assert isinstance(mask, np.ndarray)
    np.testing.assert_array_equal(mask, [True, False, True])


def test_ball_contains_batch_returns_per_row_mask() -> None:
    ball = Ball([0.0, 0.0], r=1.0)
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 0.5]])
    mask = ball.contains(pts)
    np.testing.assert_array_equal(mask, [True, False, True])
    assert ball.contains([0.5, 0.5]) is True


def test_grid_contains_batch_returns_per_row_mask() -> None:
    grid = Grid([-1.0, -1.0], [1.0, 1.0], (5, 5))
    pts = np.array([[0.0, 0.0], [3.0, 0.0]])
    np.testing.assert_array_equal(grid.contains(pts), [True, False])


def test_contains_rejects_wrong_shape() -> None:
    """A flat array whose length is not ``dim`` must raise, not silently reduce."""
    box = Box([-1.0, -1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="containment query"):
        box.contains([0.0, 0.0, 0.0])  # 3-vector against a 2-D box
    with pytest.raises(ValueError, match="containment query"):
        Ball([0.0, 0.0], r=1.0).contains(np.zeros((2, 2, 2)))  # 3-D array


def test_box_contains_1d_scalar_point() -> None:
    """A bare scalar is still a valid single point of a 1-D region (back-compat)."""
    box = Box([-1.0], [1.0])
    assert box.contains(0.5) is True
    assert box.contains(2.0) is False


# ---------------------------------------------------------------------------
# Trajectory as a first-class data object (stream v6 api-core)
#
# ``Trajectory`` is the library's primary data type, yet it exposed none of the
# protocols a data type is expected to: ``len(traj)`` raised ``TypeError``,
# ``np.asarray(traj)`` silently produced a **0-d object array** (so any
# downstream ``arr.shape`` / ``arr[:, 0]`` failed far from the cause), and there
# was no tabular export at all.
# ---------------------------------------------------------------------------


def _demo_traj() -> Trajectory:
    t = np.linspace(0.0, 1.0, 11)
    y = np.column_stack([t, t**2, -t])
    return Trajectory(t, y, system=_NamedStub(), meta={"system": "Stub", "dt": 0.1})


def test_len_is_the_sample_count() -> None:
    traj = _demo_traj()
    assert len(traj) == 11 == traj.n_steps


def test_array_protocol_returns_the_state_block() -> None:
    """``np.asarray(traj)`` is the ``(T, dim)`` state array, not a 0-d object."""
    traj = _demo_traj()
    arr = np.asarray(traj)
    assert arr.shape == (11, 3)
    assert arr.dtype == traj.y.dtype
    # No copy for the identity conversion — asarray hands back ``y`` itself.
    assert arr is traj.y
    np.testing.assert_array_equal(arr[:, 0], traj["x"])


def test_array_protocol_honours_dtype_and_copy() -> None:
    traj = _demo_traj()
    assert np.asarray(traj, dtype=np.float32).dtype == np.float32
    assert np.array(traj, copy=True) is not traj.y
    np.testing.assert_array_equal(np.array(traj, copy=True), traj.y)


def test_array_protocol_composes_with_numpy() -> None:
    """The point of the protocol: NumPy functions just work on a trajectory."""
    traj = _demo_traj()
    np.testing.assert_allclose(np.mean(traj, axis=0), traj.y.mean(axis=0))
    assert np.vstack([traj, traj]).shape == (22, 3)


def test_to_frame_round_trips_the_data() -> None:
    pd = pytest.importorskip("pandas")
    traj = _demo_traj()
    frame = traj.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (11, 3)
    assert list(frame.columns) == ["x", "y", "z"]
    assert frame.index.name == "t"
    np.testing.assert_array_equal(frame.to_numpy(), traj.y)
    np.testing.assert_array_equal(frame.index.to_numpy(), traj.t)
    assert frame.attrs["meta"]["system"] == "Stub"


def test_to_frame_without_variables_uses_generic_names() -> None:
    pytest.importorskip("pandas")
    traj = Trajectory(np.arange(4.0), np.zeros((4, 2)), system=None)
    assert list(traj.to_frame().columns) == ["y0", "y1"]


def test_to_frame_missing_pandas_points_at_pip_install_pandas(monkeypatch) -> None:
    """The hint must name a real install, not a non-existent extra."""
    import builtins

    real_import = builtins.__import__

    def _no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pandas)
    with pytest.raises(ImportError, match=r"pip install pandas"):
        _demo_traj().to_frame()


def test_trajectory_pickles_with_its_meta() -> None:
    """A Trajectory (and the system it references) survives a pickle round-trip."""
    import pickle

    traj = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=1.0, dt=0.1)
    clone = pickle.loads(pickle.dumps(traj))
    np.testing.assert_array_equal(clone.y, traj.y)
    np.testing.assert_array_equal(clone.t, traj.t)
    assert clone.meta["system"] == "Lorenz"
    assert clone.meta["params"] == traj.meta["params"]
    np.testing.assert_array_equal(clone.meta["ic"], traj.meta["ic"])
    assert type(clone.system) is type(traj.system)
    assert clone["x"].shape == traj["x"].shape


def test_map_and_dde_systems_pickle() -> None:
    """One system per family survives a round trip and still runs afterwards."""
    import pickle

    for system in (
        ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]),
        ts.systems.Henon(ic=[0.1, 0.1]),
        ts.systems.MackeyGlass(),
        ts.systems.OrnsteinUhlenbeck(ic=[0.5]),
    ):
        clone = pickle.loads(pickle.dumps(system))
        assert type(clone) is type(system)
        assert clone.params.as_dict() == system.params.as_dict()
        np.testing.assert_array_equal(clone.ic, system.ic)


def test_a_live_stepped_system_still_pickles() -> None:
    """A system holding a live Rust stepper pickles (cold) instead of failing."""
    import pickle

    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    lor.reinit([1.0, 1.0, 1.0])
    lor.step(0.01)
    clone = pickle.loads(pickle.dumps(lor))
    clone.reinit([1.0, 1.0, 1.0])
    assert np.isfinite(clone.step(0.01)).all()


# ---------------------------------------------------------------------------
# Container contract: ``len`` and ``iter`` must agree
# ---------------------------------------------------------------------------


def test_len_and_iter_report_the_same_number_of_items() -> None:
    """``__iter__`` used to yield the two columns while ``__len__`` counted samples.

    Any code that sizes an iterable and then walks it (a progress bar, a
    ``zip`` against another series, a chunked writer) saw two different lengths.
    """
    t = np.arange(7, dtype=float)
    y = np.arange(21, dtype=float).reshape(7, 3)
    traj = Trajectory(t, y, system=None)

    assert len(traj) == traj.n_steps == 7
    assert len(list(traj)) == len(traj)


def test_iterating_yields_time_state_samples() -> None:
    t = np.arange(4, dtype=float)
    y = np.arange(8, dtype=float).reshape(4, 2)
    traj = Trajectory(t, y, system=None)
    for k, (t_i, y_i) in enumerate(traj):
        assert t_i == t[k]
        np.testing.assert_array_equal(y_i, y[k])


def test_unpack_is_the_explicit_column_spelling() -> None:
    t = np.arange(4, dtype=float)
    y = np.arange(8, dtype=float).reshape(4, 2)
    traj = Trajectory(t, y, system=None)
    t_, y_ = traj.unpack()
    assert t_ is traj.t and y_ is traj.y


def test_poincare_section_subclass_inherits_the_contract() -> None:
    """The ``Trajectory`` subclass used by the section API must not diverge."""
    from tsdynamics.derived.poincare import PoincareSection

    sec = PoincareSection(np.arange(5.0), np.arange(15.0).reshape(5, 3), system=None)
    assert len(sec) == len(list(sec)) == 5
