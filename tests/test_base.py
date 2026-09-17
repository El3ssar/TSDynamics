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
    def test_unpack_returns_the_two_columns(self) -> None:
        t = np.linspace(0, 1, 5)
        y = np.zeros((5, 3))
        traj = Trajectory(t, y, system=None)
        t_, y_ = traj.unpack()
        np.testing.assert_array_equal(t_, t)
        np.testing.assert_array_equal(y_, y)

    def test_len_and_iter_agree(self) -> None:
        # The container contract: sizing an iterable and then walking it must
        # see the same number of items.  ``__iter__`` used to be a
        # tuple-unpacking convenience yielding the two *columns*, so
        # ``len(traj)`` said 5 while ``len(list(traj))`` said 2.
        t = np.linspace(0, 1, 5)
        y = np.arange(15, dtype=float).reshape(5, 3)
        traj = Trajectory(t, y, system=None)
        samples = list(traj)
        assert len(traj) == len(samples) == 5
        for k, (t_i, y_i) in enumerate(samples):
            assert t_i == t[k]
            np.testing.assert_array_equal(y_i, y[k])

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
    def test_ic_none_falls_back_to_random(self) -> None:
        # We re-seed numpy directly because resolve_ic uses np.random
        np.random.seed(123)
        s = _Stub()
        ic = s.resolve_ic()
        assert ic.shape == (3,)
        assert np.all((ic >= 0.0) & (ic < 1.0))

    def test_ic_kwarg_takes_priority(self) -> None:
        """An EXPLICIT ic is used and does not latch (§3.1: run(ic=) is not a setter)."""
        s = _Stub()
        np.testing.assert_array_equal(s.resolve_ic([7.0, 8.0, 9.0]), [7.0, 8.0, 9.0])
        assert s.ic is None

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

    def test_system_meta_is_gone_and_says_where_provenance_lives(self) -> None:
        """§8.3 — a system no longer accumulates metadata; a RUN records its own."""
        with pytest.raises(AttributeError, match="a run records its own"):
            _ = _Stub().meta


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
        picked = traj["x", "z"]
        np.testing.assert_array_equal(picked.y, traj.y[:, [0, 2]])
        # the sub-trajectory names its OWN columns, so a second selection is right
        assert picked.variables == ("x", "z")
        np.testing.assert_array_equal(picked["z"], traj.y[:, 2])

    def test_component_accepts_names_and_ints(self) -> None:
        traj = self._traj()
        np.testing.assert_array_equal(traj.component("z"), traj.component(2))

    def test_unknown_name_raises_with_options(self) -> None:
        traj = self._traj()
        with pytest.raises(KeyError, match="Declared variables"):
            traj["w"]

    def test_unnamed_system_raises_helpfully(self) -> None:
        traj = Trajectory(np.arange(3), np.zeros((3, 2)), system=_StubNoParams())
        with pytest.raises(KeyError, match="Declared variables"):
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


# ---------------------------------------------------------------------------
# Copying, pickling and IC reproducibility  (stream v6 api-core)
#
# Four defects the v6 audit reproduced independently, all on the same surface:
#
#   1. ``copy.copy(system)`` returned an ALIASED system — the "copy" shared the
#      original's ParamSet / MetaStore / ic array, so ``copy.copy(lor).sigma = 99``
#      rewrote the original's sigma.
#   2. Systems and Trajectories could not be pickled or deep-copied at all
#      (``ParamSet`` has ``__slots__`` plus a validating ``__setattr__``, and the
#      generic slot restorer tripped over it), which blocks multiprocessing /
#      joblib / caching — i.e. every way a user parallelises a parameter study.
#   3. The random-IC fallback drew from the GLOBAL numpy RNG, so a run silently
#      perturbed the caller's ``np.random.seed(0)`` stream and was itself
#      irreproducible (no seed was recorded anywhere).
#   4. ``resolve_ic`` commits the IC to the instance BEFORE the run, so a run that
#      then failed left the bad IC latched on the object and every later,
#      unrelated analysis silently started from it.
# ---------------------------------------------------------------------------


class TestCopySemantics:
    def test_copy_copy_is_independent(self) -> None:
        """``copy.copy`` must not alias params / meta / ic (defect 1)."""
        import copy

        base = _Stub(params={"a": 1.0}, ic=[1.0, 2.0, 3.0])
        clone = copy.copy(base)

        assert clone.params is not base.params
        assert clone.ic is not base.ic

        clone.a = 99.0
        clone.ic[0] = 42.0

        assert base.a == 1.0
        np.testing.assert_array_equal(base.ic, [1.0, 2.0, 3.0])

    def test_copy_copy_preserves_content(self) -> None:
        """The clone still *carries* the original's params, ic and meta."""
        import copy

        base = _Stub(params={"a": 7.0}, ic=[1.0, 2.0, 3.0])
        clone = copy.copy(base)
        assert clone.a == 7.0
        np.testing.assert_array_equal(clone.ic, [1.0, 2.0, 3.0])

    def test_deepcopy_is_independent(self) -> None:
        """``copy.deepcopy`` works at all, and is independent (defects 1 + 2)."""
        import copy

        base = _Stub(params={"a": 1.0}, ic=[1.0, 2.0, 3.0])
        clone = copy.deepcopy(base)

        clone.a = 99.0
        clone.ic[1] = -5.0

        assert base.a == 1.0
        np.testing.assert_array_equal(base.ic, [1.0, 2.0, 3.0])

    def test_deepcopy_keeps_the_subclass_and_dim(self) -> None:
        """A variable-dimension instance keeps its resolved ``dim``."""
        import copy

        base = _Stub(dim=7)
        assert copy.copy(base).dim == 7
        assert copy.deepcopy(base).dim == 7


class TestPickling:
    def test_paramset_round_trips(self) -> None:
        """The ``__slots__`` + validating-setattr container pickles (defect 2)."""
        import pickle

        p = ParamSet({"sigma": 10.0, "rho": 28.0})
        q = pickle.loads(pickle.dumps(p))
        assert q.as_dict() == p.as_dict()
        q.sigma = 1.0
        assert p.sigma == 10.0

    def test_metastore_round_trips_with_history(self) -> None:
        import pickle

        m = MetaStore()
        m.record("k", 1, note="first")
        m.record("k", 2, note="second")
        n = pickle.loads(pickle.dumps(m))
        assert n["k"] == 2
        assert [r["context"]["note"] for r in n.history("k")] == ["first", "second"]

    def test_stub_system_round_trips(self) -> None:
        import pickle

        base = _Stub(params={"a": 3.0}, ic=[1.0, 2.0, 3.0])
        clone = pickle.loads(pickle.dumps(base))
        assert type(clone) is _Stub
        assert clone.a == 3.0
        np.testing.assert_array_equal(clone.ic, [1.0, 2.0, 3.0])
        clone.a = 99.0
        assert base.a == 3.0

    def test_pickle_drops_only_the_runtime_caches(self) -> None:
        """Transient caches are dropped; identity state survives."""
        import pickle

        base = _Stub(ic=[1.0, 2.0, 3.0], seed=11)
        base.resolve_ic()  # populate the runtime RNG cache
        state = base.__getstate__()
        assert "_ic_rng" not in state
        clone = pickle.loads(pickle.dumps(base))
        assert clone.__dict__["_ic_seed"] == 11


class TestSeededInitialConditions:
    def test_random_ic_does_not_touch_the_global_rng(self) -> None:
        """A random-IC draw must not perturb the caller's stream (defect 3)."""
        np.random.seed(0)
        expected = np.random.rand(4)

        np.random.seed(0)
        _Stub().resolve_ic()
        _Stub().resolve_ic()
        got = np.random.rand(4)

        np.testing.assert_array_equal(expected, got)

    def test_same_seed_gives_the_same_random_ic(self) -> None:
        a = _Stub(seed=1234).resolve_ic()
        b = _Stub(seed=1234).resolve_ic()
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = _Stub(seed=1).resolve_ic()
        b = _Stub(seed=2).resolve_ic()
        assert not np.array_equal(a, b)

    def test_unseeded_draw_records_a_reproducible_seed(self) -> None:
        """No ``seed=`` still records the seed actually used, so a run replays."""
        s = _Stub()
        drawn = s.resolve_ic()
        seed = s._provenance()["ic_seed"]
        np.testing.assert_array_equal(_Stub(seed=seed).resolve_ic(), drawn)

    def test_resolve_ic_seed_keyword(self) -> None:
        np.testing.assert_array_equal(_Stub().resolve_ic(seed=7), _Stub().resolve_ic(seed=7))

    def test_explicit_ic_ignores_the_seed(self) -> None:
        np.testing.assert_array_equal(_Stub(seed=1).resolve_ic([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

    def test_default_ic_wins_over_the_draw(self) -> None:
        np.testing.assert_array_equal(_StubWithDefaultIC(seed=1).resolve_ic(), [0.5, -0.25])

    def test_provenance_omits_ic_seed_when_no_draw_happened(self) -> None:
        assert "ic_seed" not in _Stub(ic=[1.0, 2.0, 3.0])._provenance()


class TestExplicitICTracking:
    def test_constructor_ic_is_explicit(self) -> None:
        s = _Stub(ic=[1.0, 2.0, 3.0])
        s.resolve_ic()
        assert s._ic_explicit is True

    def test_random_draw_is_not_explicit(self) -> None:
        s = _Stub()
        s.resolve_ic()
        assert s._ic_explicit is False

    def test_default_ic_is_not_explicit(self) -> None:
        """A class-declared default keeps the random-IC retry available."""
        s = _StubWithDefaultIC()
        s.resolve_ic()
        assert s._ic_explicit is False


class TestICRollback:
    def test_failed_block_restores_the_previous_ic(self) -> None:
        """A failed run must leave the object as it was (defect 4)."""
        s = _Stub(ic=[1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError), s._ic_rollback():
            s.resolve_ic([9.0, 9.0, 9.0])
            raise RuntimeError("boom")
        np.testing.assert_array_equal(s.ic, [1.0, 2.0, 3.0])
        assert s._ic_explicit is True

    def test_successful_block_commits(self) -> None:
        """An AUTO-resolved ic latches; an explicit one does not (§3.1)."""
        s = _Stub()
        with s._ic_rollback():
            drawn = s.resolve_ic()
        np.testing.assert_array_equal(s.ic, drawn)


class TestICArrayIsNotShared:
    """The constructor must own its ``ic`` array (found while fixing the copies).

    ``np.asarray`` on a float64 array is a no-op, so ``Lorenz(ic=arr)`` used to
    *share* the caller's array: mutating ``arr`` afterwards silently moved the
    system's initial condition, and ``with_params`` (which forwards
    ``ic=self.ic``) handed back a system aliasing its source's ic.
    """

    def test_constructor_copies_the_caller_array(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        s = _Stub(ic=arr)
        assert s.ic is not arr
        arr[0] = 99.0
        np.testing.assert_array_equal(s.ic, [1.0, 2.0, 3.0])

    def test_with_params_does_not_share_the_ic(self) -> None:
        base = _Stub(ic=[1.0, 2.0, 3.0])
        derived = base.with_params(a=5.0)
        assert derived.ic is not base.ic
        derived.ic[0] = 42.0
        np.testing.assert_array_equal(base.ic, [1.0, 2.0, 3.0])

    def test_copy_method_does_not_share_the_ic(self) -> None:
        base = _Stub(ic=[1.0, 2.0, 3.0])
        clone = base.copy()
        clone.ic[0] = 42.0
        np.testing.assert_array_equal(base.ic, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Constructor keywords: ``Lorenz(sigma=12.0)`` (the natural spelling)
# ---------------------------------------------------------------------------


class TestConstructorParameterKeywords:
    """Every declared parameter is reachable as a plain constructor keyword."""

    def test_keyword_sets_the_parameter(self) -> None:
        s = _Stub(a=5.0)
        assert s.a == 5.0
        assert s.b == 2.0  # untouched default

    def test_keyword_equals_the_params_dict_spelling(self) -> None:
        assert (
            _Stub(a=5.0, b=7.0).params.as_dict()
            == _Stub(params={"a": 5.0, "b": 7.0}).params.as_dict()
        )

    def test_unknown_keyword_raises_and_names_the_valid_options(self) -> None:
        from tsdynamics.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError) as excinfo:
            _Stub(aa=5.0)
        msg = str(excinfo.value)
        assert "aa" in msg
        # The error must name the declared parameters, so the fix is obvious.
        assert "'a'" in msg and "'b'" in msg

    def test_unknown_keyword_is_a_value_error_subclass(self) -> None:
        # A bare TypeError was the pre-fix behaviour ("unexpected keyword
        # argument"); the library's own typed error is a ValueError subclass.
        with pytest.raises(ValueError):
            _Stub(aa=5.0)

    def test_giving_the_same_parameter_twice_is_an_error(self) -> None:
        from tsdynamics.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError, match="given twice"):
            _Stub(a=5.0, params={"a": 9.0})

    def test_the_other_channel_still_works_when_names_do_not_collide(self) -> None:
        s = _Stub(a=5.0, params={"b": 9.0})
        assert (s.a, s.b) == (5.0, 9.0)

    def test_reserved_constructor_keywords_stay_reserved(self) -> None:
        # params= / ic= / dim= / field_shape= / seed= must keep their meaning.
        s = _Stub(ic=[1.0, 2.0, 3.0], dim=3, seed=11, a=4.0)
        np.testing.assert_array_equal(s.ic, [1.0, 2.0, 3.0])
        assert s.dim == 3
        assert s._ic_seed == 11
        assert s.a == 4.0

    def test_a_parameter_shadowing_a_reserved_keyword_is_refused_at_import(self) -> None:
        # It could never be passed as a keyword — ``Sys(dim=3)`` would set the
        # dimension.  Refuse the class rather than ship a silent shadow.
        with pytest.raises(TypeError, match="collide"):

            class _Shadow(SystemBase):
                params = {"dim": 1.0}
                dim = 1

    def test_every_builtin_param_carrying_system_accepts_keyword_parameters(self) -> None:
        """The headline claim, swept over the whole registry (all four families)."""
        from tsdynamics import registry

        checked: dict[str, int] = {}
        for entry in registry.all_systems():
            declared = entry.cls.params or {}
            if not declared:
                continue
            sys = entry.cls(**dict(declared))
            for key, value in declared.items():
                assert sys.params[key] == value, f"{entry.name}.{key}"
            checked[entry.family] = checked.get(entry.family, 0) + 1
        assert sum(checked.values()) == 164
        assert set(checked) == {"ode", "dde", "map", "sde"}


class TestCustomInitSystemsKeepResolvingDim:
    """The five variable-dimension systems own their ``__init__`` — check them."""

    @pytest.mark.parametrize(
        ("name", "kwargs", "expected_dim"),
        [
            ("Lorenz96", {"N": 6}, 6),
            ("GrayScott", {"N": 8}, 128),
            ("KuramotoSivashinsky", {"N": 16}, 16),
            ("MultiChua", {"n_circuits": 2}, 6),
            ("SwiftHohenberg", {"N": 8}, 64),
        ],
    )
    def test_dim_resolution(self, name: str, kwargs: dict, expected_dim: int) -> None:
        import tsdynamics as ts

        assert getattr(ts.systems, name)(**kwargs).dim == expected_dim

    def test_non_structural_parameters_are_still_plain_keywords(self) -> None:
        import tsdynamics as ts

        assert ts.systems.GrayScott(N=8, F=0.03).F == 0.03
        assert ts.systems.MultiChua(2, alpha=9.0).alpha == 9.0
