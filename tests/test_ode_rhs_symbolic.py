"""
Fast symbolic checks for every ODE system's ``_equations`` and ``_jacobian``.

These tests evaluate the RHS / Jacobian symbolically — no C compilation, no
integration — so they run in <1 s even for the full suite.  The intent is to
catch shape errors, wrong parameter signatures, and bad imports.
"""

from __future__ import annotations

import importlib

import pytest
from jitcode import t as t_sym
from jitcode import y as y_sym

# Import the canonical system list from the integration tests.
from test_ode_systems import _IDS, ALL_ODE_SYSTEMS  # noqa: E402


def _eval_equations(sys: object):
    """Call ``sys._equations(y, t, **params)`` and return the result list."""
    return list(type(sys)._equations(y_sym, t_sym, **sys.params.as_dict()))


def _eval_jacobian(sys: object) -> list | None:
    """Call ``sys._jacobian(y, t, **params)`` if defined, else return None."""
    cls = type(sys)
    if "_jacobian" not in cls.__dict__:
        # Check inherited too — only base ABCs have it as abstract.
        return None
    fn = cls._jacobian
    return list(fn(y_sym, t_sym, **sys.params.as_dict()))


@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_equations_returns_dim_expressions(module_path: str, class_name: str) -> None:
    """``_equations`` must yield exactly ``dim`` symbolic expressions."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()
    expr_list = _eval_equations(sys)
    assert len(expr_list) == sys.dim, (
        f"{class_name}._equations returned {len(expr_list)} expressions, expected {sys.dim}"
    )


@pytest.mark.parametrize("module_path,class_name", ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_jacobian_shape_if_defined(module_path: str, class_name: str) -> None:
    """Where ``_jacobian`` is defined, it must return a ``dim × dim`` matrix."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    sys = cls()
    jac = _eval_jacobian(sys)
    if jac is None:
        pytest.skip(f"{class_name} does not define _jacobian")
    assert len(jac) == sys.dim, (
        f"{class_name}._jacobian returned {len(jac)} rows, expected {sys.dim}"
    )
    for i, row in enumerate(jac):
        cols = list(row)
        assert len(cols) == sys.dim, (
            f"{class_name}._jacobian row {i} has {len(cols)} cols, expected {sys.dim}"
        )


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
