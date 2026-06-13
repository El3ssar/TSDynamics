"""
Fast symbolic checks for every ODE system's ``_equations`` and ``_jacobian``.

These tests evaluate the RHS / Jacobian symbolically — no C compilation, no
integration — so they run in <1 s even for the full registry sweep.  The
intent is to catch shape errors, wrong parameter signatures, and bad imports.
"""

from __future__ import annotations

from jitcode import t as t_sym
from jitcode import y as y_sym


def _eval_equations(sys: object):
    """Call ``sys._equations(y, t, **params)`` and return the result list."""
    return list(type(sys)._equations(y_sym, t_sym, **sys.params.as_dict()))


def test_ode_equations_returns_dim_expressions(ode_entry) -> None:
    """``_equations`` must yield exactly ``dim`` symbolic expressions."""
    sys = ode_entry.cls()
    expr_list = _eval_equations(sys)
    assert len(expr_list) == sys.dim, (
        f"{ode_entry.name}._equations returned {len(expr_list)} expressions, expected {sys.dim}"
    )


def test_ode_jacobian_shape_if_defined(ode_entry) -> None:
    """Where ``_jacobian`` is defined, it must return a ``dim × dim`` matrix."""
    import pytest

    cls = ode_entry.cls
    if "_jacobian" not in cls.__dict__:
        pytest.skip(f"{ode_entry.name} does not define _jacobian")
    sys = cls()
    jac = list(cls._jacobian(y_sym, t_sym, **sys.params.as_dict()))
    assert len(jac) == sys.dim, (
        f"{ode_entry.name}._jacobian returned {len(jac)} rows, expected {sys.dim}"
    )
    for i, row in enumerate(jac):
        cols = list(row)
        assert len(cols) == sys.dim, (
            f"{ode_entry.name}._jacobian row {i} has {len(cols)} cols, expected {sys.dim}"
        )


# ---------------------------------------------------------------------------
# Variable-dim systems with non-default sizes
# ---------------------------------------------------------------------------


def test_lorenz96_equations_returns_n_expressions() -> None:
    import tsdynamics as ts

    lor = ts.Lorenz96(N=8)
    expr_list = _eval_equations(lor)
    assert len(expr_list) == 8


def test_kuramoto_sivashinsky_equations_returns_n_expressions() -> None:
    import tsdynamics as ts

    ks = ts.KuramotoSivashinsky(N=8, L=8.0)
    expr_list = _eval_equations(ks)
    assert len(expr_list) == 8


def test_multichua_equations_returns_3n_expressions() -> None:
    import tsdynamics as ts

    mc = ts.MultiChua(n_circuits=4)
    expr_list = _eval_equations(mc)
    assert len(expr_list) == 12
