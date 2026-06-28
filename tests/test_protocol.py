"""
System-protocol tests: step/state/set_state/time/reinit/trajectory across the
three families, including the two-live-steppers isolation guarantee for ODEs.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidInputError, InvalidParameterError
from tsdynamics.families import MetaStore, ParamSet, System

# ---------------------------------------------------------------------------
# Base-class contracts: ParamSet / MetaStore / resolve_ic (fast)
# ---------------------------------------------------------------------------


class TestParamSet:
    def test_fixed_key_rejects_unknown(self) -> None:
        p = ParamSet({"sigma": 10.0})
        with pytest.raises(AttributeError, match="Unknown parameter"):
            p.rho = 1.0
        with pytest.raises(KeyError, match="Unknown parameter"):
            p["rho"] = 1.0
        with pytest.raises(InvalidInputError, match="fixed-key"):
            del p["sigma"]

    def test_attr_and_item_access_agree(self) -> None:
        p = ParamSet({"sigma": 10.0, "rho": 28.0})
        p.sigma = 15.0
        assert p["sigma"] == 15.0
        assert p.as_tuple() == (15.0, 28.0)
        assert p.as_dict() == {"sigma": 15.0, "rho": 28.0}

    def test_param_hash_process_stable(self) -> None:
        # MD5-backed hash must be identical for equal contents (no hash seed).
        a = ParamSet({"sigma": 10.0, "rho": 28.0})
        b = ParamSet({"sigma": 10.0, "rho": 28.0})
        assert a.param_hash() == b.param_hash()
        b.rho = 29.0
        assert a.param_hash() != b.param_hash()


class TestMetaStore:
    def test_record_history_and_latest(self) -> None:
        m = MetaStore()
        m.record("k", 1, dt=0.1)
        m.record("k", 2, dt=0.2)
        assert m["k"] == 2  # latest wins
        hist = m.history("k")
        assert [r["value"] for r in hist] == [1, 2]  # oldest first
        assert hist[0]["context"] == {"dt": 0.1}
        assert m.latest() == {"k": 2}

    def test_equality_against_plain_dict(self) -> None:
        m = MetaStore()
        assert m == {}
        m["a"] = 1
        m["a"] = 1  # overwrite — equality is on the latest value
        assert m == {"a": 1}
        other = MetaStore()
        other["a"] = 1
        assert m == other

    def test_repr_annotates_overwritten_keys(self) -> None:
        m = MetaStore()
        m["dt"] = 0.01
        m["system"] = "lorenz"
        m["system"] = "lorenz"
        r = repr(m)
        assert "dt=0.01" in r
        assert "system='lorenz' (x2)" in r
        assert repr(MetaStore()) == "MetaStore()"


class TestResolveIc:
    def test_priority_arg_over_self_over_default(self) -> None:
        # arg > self.ic > default_ic
        lor = ts.Lorenz(ic=[1.0, 2.0, 3.0])
        np.testing.assert_array_equal(lor.resolve_ic([4.0, 5.0, 6.0]), [4.0, 5.0, 6.0])
        # the explicit arg is now stored, so a subsequent bare call reproduces it
        np.testing.assert_array_equal(lor.resolve_ic(), [4.0, 5.0, 6.0])

    def test_falls_back_to_self_ic(self) -> None:
        lor = ts.Lorenz(ic=[1.0, 2.0, 3.0])
        np.testing.assert_array_equal(lor.resolve_ic(), [1.0, 2.0, 3.0])

    def test_random_fallback_is_deterministic_under_seed(self) -> None:
        # No ic, no default_ic -> random U[0,1)^dim, reproducible under a seed.
        np.random.seed(0)
        a = ts.Lorenz().resolve_ic()
        np.random.seed(0)
        b = ts.Lorenz().resolve_ic()
        assert a.shape == (3,)
        np.testing.assert_array_equal(a, b)


def test_constructor_rejects_unknown_param() -> None:
    with pytest.raises(InvalidParameterError, match="unknown parameter"):
        ts.Lorenz(params={"sigmaa": 99.0})


# ---------------------------------------------------------------------------
# Structural conformance (fast)
# ---------------------------------------------------------------------------


def test_families_satisfy_protocol() -> None:
    assert isinstance(ts.Lorenz(), System)
    assert isinstance(ts.MackeyGlass(), System)
    assert isinstance(ts.Henon(), System)


def test_is_discrete_flags() -> None:
    assert ts.Henon().is_discrete is True
    assert ts.Lorenz().is_discrete is False
    assert ts.MackeyGlass().is_discrete is False


def test_dde_set_state_raises_helpfully() -> None:
    mg = ts.MackeyGlass()
    with pytest.raises(NotImplementedError, match="history function"):
        mg.set_state([1.0])


# ---------------------------------------------------------------------------
# Map stepping (fast — pure-Python _step loop)
# ---------------------------------------------------------------------------


class TestMapStepping:
    def test_step_matches_iterate(self) -> None:
        ic = np.array([0.1, 0.2])
        h1 = ts.Henon()
        h1.reinit(ic)
        stepped = [h1.step().copy() for _ in range(5)]

        h2 = ts.Henon()
        traj = h2.iterate(steps=5, ic=ic)
        np.testing.assert_allclose(np.array(stepped), traj.y, rtol=1e-12)

    def test_batch_step_equals_single_steps(self) -> None:
        ic = np.array([0.1, 0.2])
        a, b = ts.Henon(), ts.Henon()
        a.reinit(ic)
        b.reinit(ic)
        for _ in range(40):
            a.step()
        b.step(40)
        np.testing.assert_allclose(a.state(), b.state(), rtol=1e-10)
        assert a.time() == b.time() == 40.0

    def test_set_state_and_time(self) -> None:
        h = ts.Henon()
        h.reinit([0.0, 0.0], t=7)
        assert h.time() == 7.0
        h.set_state([0.5, -0.5])
        np.testing.assert_array_equal(h.state(), [0.5, -0.5])

    def test_implicit_reinit_on_cold_step(self) -> None:
        h = ts.Henon(ic=[0.1, 0.1])
        out = h.step()
        assert out.shape == (2,)
        assert h.time() == 1.0

    def test_one_dim_map_steps(self) -> None:
        m = ts.Logistic(ic=[0.3])
        out = m.step()
        assert out.shape == (1,)
        np.testing.assert_allclose(out[0], 3.9 * 0.3 * 0.7, rtol=1e-12)

    def test_trajectory_with_transient(self) -> None:
        h = ts.Henon()
        traj = h.trajectory(steps=100, transient=50, ic=[0.1, 0.1])
        assert traj.n_steps == 100


# ---------------------------------------------------------------------------
# ODE stepping (slow — engine integration runtime)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestODEStepping:
    def test_step_advances_time_and_state(self) -> None:
        lor = ts.Lorenz()
        lor.reinit([1.0, 1.0, 1.0])
        s0 = lor.state()
        s1 = lor.step(0.05)
        assert lor.time() == pytest.approx(0.05)
        assert not np.allclose(s0, s1)

    def test_step_matches_integrate(self) -> None:
        ic = [1.0, 1.0, 1.0]
        lor = ts.Lorenz()
        lor.reinit(ic, rtol=1e-9, atol=1e-12)
        for _ in range(10):
            lor.step(0.1)

        traj = ts.Lorenz().integrate(final_time=1.0, dt=0.1, ic=ic, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(lor.state(), traj.y[-1], rtol=1e-5, atol=1e-6)

    def test_two_live_steppers_are_isolated(self) -> None:
        a, b = ts.Lorenz(), ts.Lorenz()
        a.reinit([1.0, 1.0, 1.0])
        b.reinit([-5.0, 2.0, 30.0])
        # interleave
        for _ in range(5):
            a.step(0.02)
            b.step(0.02)
        a_alone = ts.Lorenz()
        a_alone.reinit([1.0, 1.0, 1.0])
        for _ in range(5):
            a_alone.step(0.02)
        np.testing.assert_allclose(a.state(), a_alone.state(), rtol=1e-9)

    def test_set_state_redirects_orbit(self) -> None:
        lor = ts.Lorenz()
        lor.reinit([1.0, 1.0, 1.0])
        lor.step(0.1)
        lor.set_state([0.0, 1.0, 20.0])
        np.testing.assert_array_equal(lor.state(), [0.0, 1.0, 20.0])
        out = lor.step(0.05)
        assert np.all(np.isfinite(out))

    def test_reinit_applies_param_overrides(self) -> None:
        lor = ts.Lorenz()
        lor.reinit([1.0, 1.0, 1.0], params={"rho": 35.0})
        assert lor.params["rho"] == 35.0
        assert np.all(np.isfinite(lor.step(0.05)))


# ---------------------------------------------------------------------------
# DDE stepping (slow — engine method-of-steps runtime)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDDEStepping:
    def test_step_forward(self) -> None:
        mg = ts.MackeyGlass()
        mg.reinit([1.2])
        t0 = mg.time()
        s = mg.step(0.5)
        assert s.shape == (1,)
        assert np.isfinite(s).all()
        assert mg.time() == pytest.approx(t0 + 0.5)

    def test_trajectory_protocol(self) -> None:
        mg = ts.MackeyGlass()
        traj = mg.trajectory(final_time=5.0, dt=0.5, transient=2.0, ic=[1.2])
        assert traj.t[0] >= 2.0
        assert np.all(np.isfinite(traj.y))
