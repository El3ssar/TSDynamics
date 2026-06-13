"""
Experimental Rust-backed ODE solving via *diffsol* (``pip install tsdynamics[diffsol]``).

The system's symbolic ``_equations`` are translated to DiffSL — a small DSL
that diffsol JIT-compiles at runtime (no C compiler, sub-second) — and
integrated by Rust solver kernels (Tsit45, BDF, TR-BDF2, ESDIRK34).

Initial conditions and control parameters are both DiffSL *inputs*, so one
compiled module per (class, structural-params) serves every IC and every
parameter value — the same caching economics as the JiTCODE path, without
the C-compiler dependency.

Usage::

    traj = ts.Lorenz().integrate(final_time=100.0, dt=0.01, backend="diffsol")

Limitations (experimental): ODEs only (no DDEs); the RHS must use functions
DiffSL provides (``sin``/``cos``/``tan``/``exp``/``log``/``sqrt``/``abs``/
``tanh``/...); unsupported constructs raise :class:`DiffSLTranslationError`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DiffSLTranslationError", "available", "integrate", "to_diffsl"]

#: (class name, structural hash, solver) → compiled pydiffsol.Ode
_ODE_CACHE: dict[tuple, Any] = {}

_STATE_PREFIX = "tsdstate"  # collision-proof DiffSL state names


class DiffSLTranslationError(NotImplementedError):
    """The system's RHS uses constructs DiffSL cannot express."""


def available() -> bool:
    """Check whether the pydiffsol extra is installed."""
    try:
        import pydiffsol  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# SymEngine → DiffSL translation
# ---------------------------------------------------------------------------


def _diffsl_printer():
    from sympy.printing.str import StrPrinter

    direct = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "exp": "exp",
        "log": "log",
        "sqrt": "sqrt",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "asinh": "arcsinh",
        "acosh": "arccosh",
    }

    class DiffSLPrinter(StrPrinter):
        def _print_Pow(self, expr):  # noqa: N802 — sympy printer dispatch name
            import sympy

            base = self._print(expr.base)
            if expr.exp == -1:
                return f"(1.0 / ({base}))"
            if expr.exp == sympy.Rational(1, 2):
                return f"sqrt({base})"
            if expr.exp == sympy.Integer(2):
                return f"(({base}) * ({base}))"
            return f"pow({base}, {self._print(expr.exp)})"

        def _print_Abs(self, expr):  # noqa: N802 — sympy printer dispatch name
            return f"abs({self._print(expr.args[0])})"

        def _print_sign(self, expr):
            arg = self._print(expr.args[0])
            return f"(2.0 * heaviside({arg}) - 1.0)"

        def _print_Function(self, expr):  # noqa: N802 — sympy printer dispatch name
            name = expr.func.__name__
            if name in direct:
                args = ", ".join(self._print(a) for a in expr.args)
                return f"{direct[name]}({args})"
            raise DiffSLTranslationError(f"DiffSL has no equivalent for function {name!r}.")

    return DiffSLPrinter()


def to_diffsl(system: Any) -> tuple[str, list[str]]:
    """
    Translate a :class:`~tsdynamics.base.ContinuousSystem`'s RHS to DiffSL.

    Returns
    -------
    (code, control_names)
        The DiffSL source and the parameter input order.  The full input
        vector at solve time is ``[*control_values, *initial_state]``.
    """
    import symengine
    from jitcode import t as t_sym
    from jitcode import y

    sys_obj = system
    dim = sys_obj.dim
    struct_vals = sys_obj._structural_vals()
    control_names = list(sys_obj._control_params())
    control_syms = {k: symengine.Symbol(k) for k in control_names}

    exprs = list(type(sys_obj)._equations(y, t_sym, **{**struct_vals, **control_syms}))
    if len(exprs) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(exprs)}")

    state_names = [f"{_STATE_PREFIX}{i}" for i in range(dim)]
    subs = {y(i): symengine.Symbol(state_names[i]) for i in range(dim)}

    printer = _diffsl_printer()
    rhs_lines = []
    for e in exprs:
        sympy_expr = symengine.sympify(e).subs(subs)._sympy_()
        rhs_lines.append(f"  {printer.doprint(sympy_expr)},")

    inputs = [f"  {k} = {float(sys_obj.params[k])!r}," for k in control_names]
    inputs += [f"  ic{i} = 0.0," for i in range(dim)]
    states = [f"  {state_names[i]} = ic{i}," for i in range(dim)]

    code = "\n".join(
        [
            "in_i {",
            *inputs,
            "}",
            "u_i {",
            *states,
            "}",
            "F_i {",
            *rhs_lines,
            "}",
            "",
        ]
    )
    return code, control_names


# ---------------------------------------------------------------------------
# Integration entry point (called by ContinuousSystem.integrate)
# ---------------------------------------------------------------------------

_SOLVER_MAP = {
    "tsit45": "tsit45",
    "RK45": "tsit45",
    "dopri5": "tsit45",
    "bdf": "bdf",
    "LSODA": "bdf",
    "lsoda": "bdf",
    "tr_bdf2": "tr_bdf2",
    "esdirk34": "esdirk34",
}


def _compiled_ode(system: Any, solver: str) -> tuple[Any, list[str]]:
    import pydiffsol as pds

    key = (str(system._module_path()), solver)
    hit = _ODE_CACHE.get(key)
    if hit is not None:
        return hit

    code, control_names = to_diffsl(system)
    try:
        ode = pds.Ode(code, pds.llvm, ode_solver=getattr(pds, solver))
    except RuntimeError as err:
        raise DiffSLTranslationError(
            f"DiffSL rejected the translated RHS for {type(system).__name__}:\n{err}\n"
            f"--- generated code ---\n{code}"
        ) from err
    _ODE_CACHE[key] = (ode, control_names)
    return ode, control_names


def integrate(
    system: Any,
    final_time: float,
    dt: float,
    *,
    t0: float = 0.0,
    ic: np.ndarray,
    method: str | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
):
    """
    Integrate ``system`` with the diffsol backend; mirrors the JiTCODE path.

    Returns ``(t_eval, y)`` — the caller wraps them into a Trajectory.
    """
    if not available():
        raise ImportError(
            "backend='diffsol' needs the optional dependency: pip install tsdynamics[diffsol]"
        )
    solver = _SOLVER_MAP.get(method or "tsit45")
    if solver is None:
        raise ValueError(
            f"diffsol backend does not provide method {method!r}; "
            f"use one of {sorted(set(_SOLVER_MAP))}."
        )

    ode, control_names = _compiled_ode(system, solver)
    ode.rtol = rtol
    ode.atol = atol

    from tsdynamics.base.ode_base import _make_t_eval

    t_eval = _make_t_eval(t0, final_time, dt)
    control_vals = [float(system.params[k]) for k in control_names]
    params_vec = np.concatenate([control_vals, np.asarray(ic, dtype=float)])

    sol = ode.solve_dense(params_vec, t_eval - t0)
    y = np.asarray(sol.ys, dtype=float).T
    if not np.all(np.isfinite(y)):
        raise RuntimeError(
            f"{type(system).__name__}: diffsol integration produced non-finite values."
        )
    return t_eval, y
