"""WS-ERRORS — the TSDynamicsError hierarchy + value-naming + silent-footgun gate.

These tests pin three guarantees the v4 error standard makes:

1. The hierarchy exists and the leaf classes multiply-inherit from the stdlib
   type a caller would already be catching, so ``except ValueError`` /
   ``except TypeError`` keep working (the break is purely additive).
2. Rejected values get a *value-naming* message: it names the offending value,
   states the rule or options, and (where one exists) suggests a fix.
3. The headline silent footguns now **raise** instead of returning garbage:
   ``final_time <= 0``, ``dt <= 0``, a typo'd system attribute, an unknown
   constructor parameter, and a System handed where a measured series is wanted.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import (
    BackendError,
    ConvergenceError,
    InvalidInputError,
    InvalidParameterError,
    TSDynamicsError,
    invalid_value,
)
from tsdynamics.utils.grids import make_output_grid

# ---------------------------------------------------------------------------
# 1. The hierarchy (and the multiple-inheritance contract)
# ---------------------------------------------------------------------------


def test_errors_module_is_reachable():
    """``tsdynamics.errors`` is importable as a submodule."""
    import tsdynamics.errors as errs

    assert errs.TSDynamicsError is TSDynamicsError


@pytest.mark.parametrize(
    ("cls", "stdlib"),
    [
        (InvalidParameterError, ValueError),
        (InvalidInputError, TypeError),
        (ConvergenceError, RuntimeError),
        (BackendError, RuntimeError),
    ],
)
def test_leaf_classes_multiply_inherit(cls, stdlib):
    """Every leaf is a TSDynamicsError *and* the expected stdlib exception."""
    assert issubclass(cls, TSDynamicsError)
    assert issubclass(cls, stdlib)
    err = cls("boom")
    assert isinstance(err, TSDynamicsError)
    assert isinstance(err, stdlib)


def test_invalid_parameter_still_caught_by_value_error():
    """A plain ``except ValueError`` must still catch InvalidParameterError."""
    with pytest.raises(ValueError):
        raise InvalidParameterError("bad")


def test_invalid_input_still_caught_by_type_error():
    """A plain ``except TypeError`` must still catch InvalidInputError."""
    with pytest.raises(TypeError):
        raise InvalidInputError("bad")


# ---------------------------------------------------------------------------
# 2. The value-naming message builder
# ---------------------------------------------------------------------------


def test_invalid_value_rule_form_names_value():
    err = invalid_value("final_time", -5, rule="must be > 0")
    assert isinstance(err, InvalidParameterError)
    assert str(err) == "final_time must be > 0, got -5"


def test_invalid_value_options_form_lists_choices():
    err = invalid_value("backend", "gpu", options=["interp", "jit"])
    assert "gpu" in str(err)
    assert "interp" in str(err) and "jit" in str(err)


def test_invalid_value_appends_hint():
    err = invalid_value("dt", 0.0, rule="must be > 0", hint="use a small positive step")
    assert str(err).endswith("use a small positive step")


# ---------------------------------------------------------------------------
# 3a. Silent footguns — the output grid (dt<=0 / backwards window)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_grid_rejects_nonpositive_dt(dt):
    with pytest.raises(InvalidParameterError) as ei:
        make_output_grid(0.0, 10.0, dt)
    assert "dt" in str(ei.value) and repr(dt) in str(ei.value)
    assert isinstance(ei.value, ValueError)  # back-compat


@pytest.mark.parametrize(("t0", "tf"), [(0.0, -5.0), (0.0, 0.0), (5.0, 1.0)])
def test_grid_rejects_backwards_window(t0, tf):
    with pytest.raises(InvalidParameterError) as ei:
        make_output_grid(t0, tf, 0.1)
    assert "final_time" in str(ei.value)
    assert isinstance(ei.value, ValueError)


def test_grid_happy_path_unchanged():
    grid = make_output_grid(0.0, 1.0, 0.5)
    assert np.allclose(grid, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# 3b. Silent footguns — end-to-end through a real integrate()
# ---------------------------------------------------------------------------


def test_integrate_negative_final_time_raises():
    """Was a silent one-step garbage Trajectory; now raises with the value named."""
    lor = ts.systems.Lorenz()
    with pytest.raises(InvalidParameterError) as ei:
        lor.integrate(final_time=-5, dt=0.01)
    assert "final_time" in str(ei.value) and "-5" in str(ei.value)


def test_integrate_zero_dt_raises():
    """Was a bare ZeroDivisionError from the grid helper; now a domain error."""
    lor = ts.systems.Lorenz()
    with pytest.raises(InvalidParameterError) as ei:
        lor.integrate(final_time=10, dt=0.0)
    assert "dt" in str(ei.value)
    # the old failure mode was a ZeroDivisionError — assert we no longer leak it
    assert not isinstance(ei.value, ZeroDivisionError)


def test_integrate_happy_path_still_works():
    lor = ts.systems.Lorenz()
    traj = lor.integrate(final_time=2.0, dt=0.1)
    assert traj.y.shape[1] == 3
    assert traj.y.shape[0] > 1


# ---------------------------------------------------------------------------
# 3c. Silent footguns — typo'd attribute / unknown constructor parameter
# ---------------------------------------------------------------------------


def test_typo_attribute_raises_naming_value():
    """``lor.sigmaa = 99`` was silently stored; now rejected like with_params."""
    lor = ts.systems.Lorenz()
    with pytest.raises(InvalidParameterError) as ei:
        lor.sigmaa = 99
    msg = str(ei.value)
    assert "sigmaa" in msg and "sigma" in msg
    assert isinstance(ei.value, ValueError)
    # and the real parameter is untouched
    assert lor.sigma != 99


def test_setting_a_real_parameter_still_works():
    lor = ts.systems.Lorenz()
    lor.sigma = 12.0
    assert lor.sigma == 12.0


def test_private_and_known_attributes_pass_through():
    """Internal step state (``_``-prefixed) and class attrs are never rejected."""
    lor = ts.systems.Lorenz()
    lor._scratch = object()  # private — allowed
    lor.ic = [1.0, 2.0, 3.0]  # real instance attribute — allowed
    assert np.allclose(lor.ic, [1.0, 2.0, 3.0])


def test_unknown_constructor_parameter_raises():
    with pytest.raises(InvalidParameterError) as ei:
        ts.systems.Lorenz(params={"sigmaa": 9.0})
    assert "sigmaa" in str(ei.value) and "Declared" in str(ei.value)
    assert isinstance(ei.value, ValueError)


# ---------------------------------------------------------------------------
# 3d. Silent footguns — a System handed where a measured series is wanted
# ---------------------------------------------------------------------------


def test_entropy_on_a_system_raises_domain_error():
    """Was a leaked ``float() argument`` TypeError; now names the System + the fix."""
    from tsdynamics.analysis import permutation_entropy

    lor = ts.systems.Lorenz()
    with pytest.raises(InvalidInputError) as ei:
        permutation_entropy(lor)
    msg = str(ei.value)
    assert "System" in msg and "Lorenz" in msg
    assert isinstance(ei.value, TypeError)  # back-compat: was a TypeError


def test_entropy_on_a_series_still_works():
    from tsdynamics.analysis import permutation_entropy

    rng = np.random.default_rng(0)
    series = rng.standard_normal(500)
    val = float(permutation_entropy(series))
    assert 0.0 <= val <= 1.0


def test_catch_all_via_base_class():
    """A user can catch everything deliberate through the one base class."""
    lor = ts.systems.Lorenz()
    with pytest.raises(TSDynamicsError):
        lor.integrate(final_time=-1.0, dt=0.1)
