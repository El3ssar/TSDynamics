"""Tests for ``tsdynamics.families``: ParamSet, Trajectory, MetaStore, SystemBase."""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.families import MetaStore, ParamSet, SystemBase, Trajectory

# ---------------------------------------------------------------------------
# A minimal SystemBase subclass for direct testing
# ---------------------------------------------------------------------------


class _Stub(SystemBase):
    params = {"a": 1.0, "b": 2.0}
    dim = 3


class _StubNoParams(SystemBase):
    dim = 2


class _StubWithDefaultIC(SystemBase):
    params = {"a": 1.0}
    dim = 2
    default_ic = np.array([0.5, -0.25])


# ---------------------------------------------------------------------------
# ParamSet
# ---------------------------------------------------------------------------


class TestParamSet:
    def test_attribute_access(self) -> None:
        p = ParamSet({"sigma": 10.0, "rho": 28.0})
        assert p.sigma == 10.0
        assert p.rho == 28.0

    def test_attribute_write_updates_dict(self) -> None:
        p = ParamSet({"sigma": 10.0})
        p.sigma = 15.0
        assert p["sigma"] == 15.0

    def test_unknown_key_attribute_raises(self) -> None:
        p = ParamSet({"sigma": 10.0})
        with pytest.raises(AttributeError, match="Unknown parameter"):
            p.unknown_key = 5.0

    def test_unknown_key_item_raises(self) -> None:
        p = ParamSet({"sigma": 10.0})
        with pytest.raises(KeyError, match="Unknown parameter"):
            p["nope"] = 5.0

    def test_delete_forbidden(self) -> None:
        p = ParamSet({"sigma": 10.0})
        with pytest.raises(TypeError):
            del p["sigma"]

    def test_iter_preserves_insertion_order(self) -> None:
        p = ParamSet({"a": 1, "b": 2, "c": 3})
        assert list(p) == ["a", "b", "c"]

    def test_as_tuple_and_as_dict(self) -> None:
        p = ParamSet({"a": 1.0, "b": 2.0})
        assert p.as_tuple() == (1.0, 2.0)
        assert p.as_dict() == {"a": 1.0, "b": 2.0}

    def test_param_hash_stable_across_instances(self) -> None:
        p1 = ParamSet({"a": 1.0, "b": 2.0})
        p2 = ParamSet({"a": 1.0, "b": 2.0})
        assert p1.param_hash() == p2.param_hash()

    def test_param_hash_changes_with_values(self) -> None:
        p = ParamSet({"a": 1.0, "b": 2.0})
        h0 = p.param_hash()
        p.a = 99.0
        assert p.param_hash() != h0


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_tuple_unpacking(self) -> None:
        t = np.linspace(0, 1, 5)
        y = np.zeros((5, 3))
        traj = Trajectory(t, y, system=None)
        t_, y_ = traj
        np.testing.assert_array_equal(t_, t)
        np.testing.assert_array_equal(y_, y)

    def test_dim_and_n_steps(self) -> None:
        traj = Trajectory(np.zeros(10), np.zeros((10, 4)), system=None)
        assert traj.dim == 4
        assert traj.n_steps == 10

    def test_slicing_returns_trajectory(self) -> None:
        traj = Trajectory(np.arange(10), np.arange(30).reshape(10, 3), system="sys")
        sl = traj[2:7]
        assert isinstance(sl, Trajectory)
        assert sl.n_steps == 5
        assert sl.system == "sys"

    def test_component_extraction(self) -> None:
        y = np.arange(30).reshape(10, 3)
        traj = Trajectory(np.arange(10), y, system=None)
        np.testing.assert_array_equal(traj.component(1), y[:, 1])

    def test_after_drops_transient(self) -> None:
        t = np.linspace(0, 10, 11)
        y = np.zeros((11, 2))
        traj = Trajectory(t, y, system=None)
        sl = traj.after(5.0)
        assert sl.t[0] == 5.0
        assert sl.n_steps == 6


# ---------------------------------------------------------------------------
# SystemBase — params / dim / ic resolution
# ---------------------------------------------------------------------------


class TestSystemBase:
    def test_class_params_accessible_as_attributes(self) -> None:
        s = _Stub()
        assert s.a == 1.0
        assert s.b == 2.0

    def test_attribute_write_syncs_params(self) -> None:
        s = _Stub()
        s.a = 99.0
        assert s.params["a"] == 99.0

    def test_constructor_override_merges_with_defaults(self) -> None:
        s = _Stub(params={"a": 5.0})
        assert s.params["a"] == 5.0
        assert s.params["b"] == 2.0  # default preserved

    def test_unknown_constructor_param_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            _Stub(params={"zzz": 1.0})

    def test_no_params_class_has_empty_paramset(self) -> None:
        s = _StubNoParams()
        assert len(s.params) == 0

    def test_dim_from_class(self) -> None:
        assert _Stub().dim == 3

    def test_dim_override_via_constructor(self) -> None:
        assert _Stub(dim=7).dim == 7

    # IC resolution priority: kwarg > self.ic > default_ic > random
    def test_ic_none_falls_back_to_random(self, rng: np.random.Generator) -> None:
        # We re-seed numpy directly because resolve_ic uses np.random
        np.random.seed(123)
        s = _Stub()
        ic = s.resolve_ic()
        assert ic.shape == (3,)
        assert np.all((ic >= 0.0) & (ic < 1.0))

    def test_ic_kwarg_takes_priority(self) -> None:
        s = _Stub()
        s.resolve_ic([7.0, 8.0, 9.0])
        np.testing.assert_array_equal(s.ic, [7.0, 8.0, 9.0])

    def test_default_ic_used_when_no_kwarg(self) -> None:
        s = _StubWithDefaultIC()
        ic = s.resolve_ic()
        np.testing.assert_array_equal(ic, [0.5, -0.25])

    def test_default_ic_overridden_by_kwarg(self) -> None:
        s = _StubWithDefaultIC()
        ic = s.resolve_ic([1.0, 2.0])
        np.testing.assert_array_equal(ic, [1.0, 2.0])

    # Copy / with_params
    def test_copy_is_independent(self) -> None:
        s = _Stub(params={"a": 9.0})
        c = s.copy()
        c.a = 17.0
        assert s.a == 9.0
        assert c.a == 17.0

    def test_with_params_returns_new_instance(self) -> None:
        s = _Stub()
        new = s.with_params(a=42.0)
        assert new is not s
        assert new.a == 42.0
        assert s.a == 1.0  # original untouched

    def test_meta_dict_is_per_instance(self) -> None:
        s1 = _Stub()
        s2 = _Stub()
        s1.meta["foo"] = 1
        assert s2.meta == {}


# ---------------------------------------------------------------------------
# MetaStore
# ---------------------------------------------------------------------------


class TestMetaStore:
    def test_dict_style_read_write(self) -> None:
        m = MetaStore()
        m["x"] = 1
        assert m["x"] == 1
        assert "x" in m
        assert len(m) == 1

    def test_writes_append_history(self) -> None:
        m = MetaStore()
        m["x"] = 1
        m["x"] = 2
        assert m["x"] == 2  # latest wins on read
        hist = m.history("x")
        assert [h["value"] for h in hist] == [1, 2]

    def test_record_stores_context(self) -> None:
        m = MetaStore()
        m.record("spec", [0.9, 0.0], dt=0.1, final_time=200.0)
        rec = m.history("spec")[-1]
        assert rec["context"] == {"dt": 0.1, "final_time": 200.0}
        assert "timestamp" in rec

    def test_equality_against_plain_dict(self) -> None:
        m = MetaStore()
        assert m == {}
        m["a"] = 5
        assert m == {"a": 5}
        assert m != {"a": 6}

    def test_missing_key_raises(self) -> None:
        m = MetaStore()
        with pytest.raises(KeyError):
            m["nope"]

    def test_latest_snapshot(self) -> None:
        m = MetaStore()
        m["a"] = 1
        m["b"] = 2
        m["a"] = 3
        assert m.latest() == {"a": 3, "b": 2}


# ---------------------------------------------------------------------------
# Trajectory — named components, point-set ops, provenance
# ---------------------------------------------------------------------------


class _NamedStub(SystemBase):
    params = {"a": 1.0}
    dim = 3
    variables = ("x", "y", "z")


class TestTrajectoryNamedAccess:
    def _traj(self) -> Trajectory:
        y = np.arange(30, dtype=float).reshape(10, 3)
        return Trajectory(np.arange(10), y, system=_NamedStub())

    def test_named_component(self) -> None:
        traj = self._traj()
        np.testing.assert_array_equal(traj["y"], traj.y[:, 1])

    def test_named_multi_component(self) -> None:
        traj = self._traj()
        np.testing.assert_array_equal(traj[["x", "z"]], traj.y[:, [0, 2]])

    def test_component_accepts_names_and_ints(self) -> None:
        traj = self._traj()
        np.testing.assert_array_equal(traj.component("z"), traj.component(2))

    def test_unknown_name_raises_with_options(self) -> None:
        traj = self._traj()
        with pytest.raises(KeyError, match="Declared variables"):
            traj["w"]

    def test_unnamed_system_raises_helpfully(self) -> None:
        traj = Trajectory(np.arange(3), np.zeros((3, 2)), system=_StubNoParams())
        with pytest.raises(KeyError, match="declares no"):
            traj["x"]

    def test_row_slicing_still_works(self) -> None:
        traj = self._traj()
        sl = traj[2:5]
        assert isinstance(sl, Trajectory)
        assert sl.n_steps == 3


class TestTrajectoryPointSetOps:
    def test_minmax(self) -> None:
        y = np.array([[0.0, 5.0], [2.0, -1.0], [1.0, 3.0]])
        traj = Trajectory(np.arange(3), y, system=None)
        lo, hi = traj.minmax()
        np.testing.assert_array_equal(lo, [0.0, -1.0])
        np.testing.assert_array_equal(hi, [2.0, 5.0])

    def test_standardize(self) -> None:
        rng = np.random.default_rng(1)
        traj = Trajectory(np.arange(100), rng.normal(5.0, 3.0, size=(100, 2)), system=None)
        std = traj.standardize()
        np.testing.assert_allclose(std.y.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(std.y.std(axis=0), 1.0, atol=1e-12)
        assert "standardized" in std.meta

    def test_standardize_constant_component_is_safe(self) -> None:
        y = np.column_stack([np.ones(10), np.arange(10.0)])
        traj = Trajectory(np.arange(10), y, system=None)
        std = traj.standardize()
        assert np.all(np.isfinite(std.y))

    def test_neighbors(self) -> None:
        y = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        traj = Trajectory(np.arange(3), y, system=None)
        dist, idx = traj.neighbors([0.9, 0.1], k=1)
        assert idx == 1
        dist2, idx2 = traj.neighbors([0.0, 0.0], k=2)
        assert list(idx2) == [0, 1]

    def test_meta_preserved_through_slicing_and_after(self) -> None:
        traj = Trajectory(np.arange(10.0), np.zeros((10, 2)), system=None, meta={"k": 1})
        assert traj[2:].meta == {"k": 1}
        assert traj.after(5.0).meta == {"k": 1}
