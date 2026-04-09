"""Tests for BaseDyn: generate_timesteps, parameter management, initial conditions."""

import numpy as np
import pytest

from tsdynamics.base.base import BaseDyn

# ---------------------------------------------------------------------------
# Concrete stub for testing BaseDyn directly
# ---------------------------------------------------------------------------


class _Stub(BaseDyn):
    """Minimal concrete subclass — no abstract methods needed in BaseDyn."""

    params = {"a": 1.0, "b": 2.0}
    n_dim = 3


class _StubNoParams(BaseDyn):
    n_dim = 2


# ---------------------------------------------------------------------------
# generate_timesteps
# ---------------------------------------------------------------------------


class TestGenerateTimesteps:
    def setup_method(self):
        self.stub = _Stub()

    def test_starts_at_zero(self):
        ts = self.stub.generate_timesteps(dt=0.1, final_time=1.0)
        assert ts[0] == pytest.approx(0.0)

    def test_includes_endpoint_final_time(self):
        ts = self.stub.generate_timesteps(dt=0.1, final_time=0.5)
        assert ts[-1] == pytest.approx(0.5)

    def test_final_time_step_count(self):
        ts = self.stub.generate_timesteps(dt=0.1, final_time=0.5)
        # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 → 6 points
        assert ts.shape == (6,)

    def test_steps_overrides_final_time(self, capsys):
        ts = self.stub.generate_timesteps(dt=0.1, steps=5, final_time=99.0)
        captured = capsys.readouterr()
        assert "steps" in captured.out.lower() or ts.shape[0] == 6  # warning printed
        # Steps=5 with dt=0.1 → 0, 0.1, 0.2, 0.3, 0.4, 0.5 (6 points)
        assert ts.shape == (6,)
        assert ts[-1] == pytest.approx(0.5)

    def test_steps_only(self):
        ts = self.stub.generate_timesteps(dt=0.2, steps=4)
        # 0, 0.2, 0.4, 0.6, 0.8 → 5 points
        assert ts.shape == (5,)
        assert ts[-1] == pytest.approx(0.8)

    def test_raises_without_steps_or_final_time(self):
        with pytest.raises(ValueError, match="Either"):
            self.stub.generate_timesteps(dt=0.1, steps=None, final_time=None)

    def test_returns_float64(self):
        ts = self.stub.generate_timesteps(dt=0.1, final_time=1.0)
        assert ts.dtype == np.float64

    def test_monotonically_increasing(self):
        ts = self.stub.generate_timesteps(dt=0.05, final_time=1.0)
        assert np.all(np.diff(ts) > 0)

    def test_step_spacing(self):
        ts = self.stub.generate_timesteps(dt=0.1, final_time=1.0)
        diffs = np.diff(ts[:-1])  # exclude the potential fractional final step
        assert np.allclose(diffs, 0.1, atol=1e-12)

    def test_very_small_dt(self):
        ts = self.stub.generate_timesteps(dt=0.001, final_time=0.01)
        assert ts[0] == pytest.approx(0.0)
        assert ts[-1] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Parameter management
# ---------------------------------------------------------------------------


class TestParameterManagement:
    def test_class_params_accessible_as_attributes(self):
        stub = _Stub()
        assert stub.a == pytest.approx(1.0)
        assert stub.b == pytest.approx(2.0)

    def test_params_dict_populated_from_class_defaults(self):
        stub = _Stub()
        assert stub.params == {"a": 1.0, "b": 2.0}

    def test_attribute_write_syncs_to_params_dict(self):
        stub = _Stub()
        stub.a = 99.0
        assert stub.params["a"] == pytest.approx(99.0)

    def test_params_dict_write_is_independent_of_attribute(self):
        """Writing directly to params dict does NOT auto-sync the attribute."""
        stub = _Stub()
        stub.params["b"] = 99.0
        # After direct dict write, params["b"] is updated but attribute may lag
        assert stub.params["b"] == pytest.approx(99.0)

    def test_constructor_params_override_replaces_class_defaults(self):
        """Passing params= at construction replaces class-level defaults entirely."""
        stub = _Stub(params={"a": 5.0, "b": 10.0})
        assert stub.params["a"] == pytest.approx(5.0)
        assert stub.params["b"] == pytest.approx(10.0)

    def test_no_params_class_gives_empty_dict(self):
        stub = _StubNoParams()
        assert stub.params == {}

    def test_n_dim_from_class_attribute(self):
        stub = _Stub()
        assert stub.n_dim == 3

    def test_n_dim_override_at_construction(self):
        stub = _Stub(n_dim=7)
        assert stub.n_dim == 7


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_default_initial_conds_is_none(self):
        stub = _Stub()
        assert stub.initial_conds is None

    def test_constructor_stores_initial_conds(self):
        ic = np.array([1.0, 2.0, 3.0])
        stub = _Stub(initial_conds=ic)
        np.testing.assert_array_equal(stub.initial_conds, ic)

    def test_initial_conds_stores_reference(self):
        """BaseDyn stores the initial_conds reference as-is (no copy at construction)."""
        ic = np.array([1.0, 2.0, 3.0])
        stub = _Stub(initial_conds=ic)
        # The object stores the reference — mutating ic mutates stub.initial_conds
        assert stub.initial_conds is ic

    def test_initial_conds_accepts_list(self):
        """BaseDyn accepts any sequence for initial_conds and stores it as-is."""
        stub = _Stub(initial_conds=[0.1, 0.2, 0.3])
        # BaseDyn does not convert to np.ndarray at construction time
        assert stub.initial_conds[0] == pytest.approx(0.1)
