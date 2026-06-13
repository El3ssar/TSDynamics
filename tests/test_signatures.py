"""
Subclass-contract enforcement: parameter signatures must match params dicts.

The import-time checker in ``DiscreteMap.__init_subclass__`` is the first
line of defence; these tests pin its behaviour and sweep the ODE/DDE families
(whose ``_equations`` take keyword params, so a mismatch raises at call time).
"""

from __future__ import annotations

import numpy as np
import pytest
from jitcode import t as t_sym
from jitcode import y as y_sym

import tsdynamics as ts

# ---------------------------------------------------------------------------
# DiscreteMap import-time checker
# ---------------------------------------------------------------------------


def test_map_swapped_step_signature_raises() -> None:
    with pytest.raises(TypeError, match="ORDER must match"):

        class _SwappedStep(ts.DiscreteMap):
            params = {"a": 1.0, "b": 2.0}
            dim = 1

            @staticmethod
            def _step(X, b, a):  # wrong order
                return (a * X[0] + b,)


def test_map_swapped_jacobian_signature_raises() -> None:
    with pytest.raises(TypeError, match="_jacobian"):

        class _SwappedJac(ts.DiscreteMap):
            params = {"a": 1.0, "b": 2.0}
            dim = 1

            @staticmethod
            def _step(X, a, b):
                return (a * X[0] + b,)

            @staticmethod
            def _jacobian(X, b, a):  # wrong order
                return ((a,),)


def test_map_missing_param_raises() -> None:
    with pytest.raises(TypeError, match="ORDER must match"):

        class _MissingParam(ts.DiscreteMap):
            params = {"a": 1.0, "b": 2.0}
            dim = 1

            @staticmethod
            def _step(X, a):  # b missing
                return (a * X[0],)


def test_map_correct_signature_accepted() -> None:
    class _Fine(ts.DiscreteMap):
        params = {"a": 1.0, "b": 2.0}
        dim = 1

        @staticmethod
        def _step(X, a, b):
            return (a * X[0] + b,)

    m = _Fine()
    assert m.dim == 1


def test_map_inherited_step_with_reordered_params_raises() -> None:
    """Subclassing with a reordered params dict must also be caught."""
    with pytest.raises(TypeError, match="ORDER must match"):

        class _Reordered(ts.Henon):
            params = {"b": 0.3, "a": 1.4}  # Henon._step is (X, a, b)


# ---------------------------------------------------------------------------
# ODE/DDE keyword-param sweep — _equations must accept exactly the params
# ---------------------------------------------------------------------------


def test_ode_equations_accept_declared_params(ode_entry) -> None:
    sys = ode_entry.cls()
    exprs = list(type(sys)._equations(y_sym, t_sym, **sys.params.as_dict()))
    assert len(exprs) == sys.dim


def test_dde_equations_accept_declared_params(dde_entry) -> None:
    from jitcdde import t as t_dde
    from jitcdde import y as y_dde

    sys = dde_entry.cls()
    exprs = list(type(sys)._equations(y_dde, t_dde, **sys.params.as_dict()))
    assert len(exprs) == sys.dim


def test_map_step_executes_with_declared_params(map_entry) -> None:
    """One numeric ``_step`` call with the declared params must succeed."""
    sys = map_entry.cls()
    x0 = sys.resolve_ic(None)
    nxt = np.asarray(type(sys)._step(x0, *sys.params.as_tuple()), dtype=float).ravel()
    assert nxt.shape == (sys.dim,)
