"""
Subclass-contract enforcement: parameter signatures must match params dicts.

The import-time checker in ``DiscreteMap.__init_subclass__`` is the first
line of defence; these tests pin its behaviour and sweep the ODE/DDE families
(whose ``_equations`` take keyword params, so a mismatch raises at call time).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine.symbols import state_time_symbols

# The engine-native symbolic state/time accessors (`y(i)` / `t`), byte-identical
# to the callables a system's `_equations` is written against.
y_sym, t_sym = state_time_symbols()

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
    """A map declaring both kernels in ``params`` order instantiates.

    ``_step`` **and** ``_jacobian`` are both required: ``DiscreteMap`` is an
    ``abc.ABC`` like every other family base, so a subclass that omits either
    cannot be instantiated (it used to be the one non-ABC family base, which
    let a missing kernel surface far downstream as a ``TapeCompileError``).
    """

    class _Fine(ts.DiscreteMap):
        params = {"a": 1.0, "b": 2.0}
        dim = 1

        @staticmethod
        def _step(X, a, b):
            return (a * X[0] + b,)

        @staticmethod
        def _jacobian(X, a, b):
            return ((a,),)

    m = _Fine()
    assert m.dim == 1
    assert np.allclose(m._step(np.array([1.0]), 1.0, 2.0), [3.0])


def test_map_inherited_step_with_reordered_params_raises() -> None:
    """Subclassing with a reordered params dict must also be caught."""
    with pytest.raises(TypeError, match="ORDER must match"):

        class _Reordered(ts.systems.Henon):
            params = {"b": 0.3, "a": 1.4}  # Henon._step is (X, a, b)


# ---------------------------------------------------------------------------
# ODE/DDE keyword-param sweep — _equations must accept exactly the params
# ---------------------------------------------------------------------------


def test_ode_equations_accept_declared_params(ode_entry) -> None:
    sys = ode_entry.cls()
    exprs = list(type(sys)._equations(y_sym, t_sym, **sys.params.as_dict()))
    assert len(exprs) == sys.dim


def test_dde_equations_accept_declared_params(dde_entry) -> None:
    # The same engine-native y/t accessors carry the 2-argument delayed form
    # y(i, t - tau) that DDE _equations use.
    y_dde, t_dde = state_time_symbols()

    sys = dde_entry.cls()
    exprs = list(type(sys)._equations(y_dde, t_dde, **sys.params.as_dict()))
    assert len(exprs) == sys.dim


def test_map_step_executes_with_declared_params(map_entry) -> None:
    """One numeric ``_step`` call with the declared params must succeed."""
    sys = map_entry.cls()
    x0 = sys.resolve_ic(None)
    nxt = np.asarray(type(sys)._step(x0, *sys.params.as_tuple()), dtype=float).ravel()
    assert nxt.shape == (sys.dim,)


# ---------------------------------------------------------------------------
# SystemBase import-time checker: a parameter may not shadow a constructor keyword
# ---------------------------------------------------------------------------


def test_parameter_named_like_a_constructor_keyword_is_refused() -> None:
    """Since every parameter is a constructor keyword, the two namespaces meet.

    ``Sys(dim=3)`` must keep meaning "state-space dimension".  A class declaring
    a parameter called ``dim`` could therefore never have it set by keyword — so
    the class is refused at definition time rather than silently shadowing it.
    """
    for reserved in ("params", "ic", "dim", "field_shape", "seed"):
        with pytest.raises(TypeError, match="collide"):

            class _Shadow(ts.families.ContinuousSystem):
                params = {reserved: 1.0}
                dim = 1

                @staticmethod
                def _equations(y, t, **p):
                    return [-y(0)]


def test_the_reserved_set_is_derived_from_the_real_signature() -> None:
    """The guard's name list cannot drift from the constructor it describes."""
    import inspect

    from tsdynamics.families.base import _RESERVED_INIT_KEYWORDS, SystemBase

    sig = inspect.signature(SystemBase.__init__)
    expected = {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }
    assert expected == _RESERVED_INIT_KEYWORDS
    # ... and the constructor really does take free parameter keywords.
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
