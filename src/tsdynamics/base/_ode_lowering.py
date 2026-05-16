"""Lower a :class:`ContinuousSystem` subclass to the symbolic IR.

The pipeline:

1. Build SymEngine placeholders — one ``Symbol`` per state component
   (``_v0, _v1, ...``), one per non-structural parameter (``_p_<name>``),
   one for time (``_t``).  Structural parameters are substituted with
   their literal value at trace time (same machinery as JiTCODE — they
   bake into the symbolic structure of ``_equations``).
2. Call ``cls._equations(y, t_sym, **kwargs)`` to get a length-``dim``
   list of SymEngine expression trees.
3. Walk each tree → IR :class:`~tsdynamics.base._ir.Node`.  Numeric
   constants (``Integer``, ``Rational``, ``RealDouble``, ``Constant`` —
   the last covers ``pi``) lower to :class:`~tsdynamics.base._ir.Const`.
   ``Add`` / ``Mul`` fold left.  ``Pow`` with integer exponent →
   :class:`~tsdynamics.base._ir.Pow`; with fractional exponent →
   :class:`~tsdynamics.base._ir.PowF`.  The special case ``x ** (1/2)``
   is lifted to ``UnaryOp("sqrt", x)`` for a cheaper Rust ``sqrt``
   intrinsic than a generic ``powf``.
4. Jacobian: if ``cls`` provides its own ``_jacobian``, trace it the
   same way.  Otherwise call ``symengine.diff`` cell by cell.  SymEngine
   leaves ``Abs`` / ``sign`` derivatives unevaluated as ``Derivative``
   nodes — six systems (MultiChua, AnishchenkoAstakhov,
   StickSlipOscillator, CellularNeuralNetwork, Colpitts,
   FluidTrampoline) hit this.  For them we set ``has_jacobian = False``
   on the resulting :class:`~tsdynamics.base._ir.CompiledOde`; the RHS
   still lowers cleanly.  Stiff Rosenbrock methods (N2.c) and
   variational Lyapunov (N3) are the only consumers of J, and both
   arrive later.

Raises :class:`~tsdynamics.base._ir.NotLowerableError` for any RHS
construct the IR can't represent — caller falls back to JiTCODE for
that one system.  In N2.a's lifetime every built-in ODE lowers
without error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import symengine as se
from symengine.lib.symengine_wrapper import Constant as _SymEngineConstant

from ._ir import (
    BinOp,
    CompiledOde,
    Const,
    Node,
    NotLowerableError,
    Param,
    Pow,
    PowF,
    Time,
    UnaryOp,
    Var,
    serialize_ode,
)

# ---------------------------------------------------------------------------
# SymEngine class → IR tag mapping.  Looked up by ``type(expr).__name__``
# because the SymEngine function classes are not all exposed at the top
# level under the same name (e.g. ``Abs`` vs ``abs`` lower / upper case).
# ---------------------------------------------------------------------------

_UNARY_FUNCTION_TAGS: dict[str, str] = {
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "Abs": "abs",
    "sign": "sign",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
    "acos": "arccos",
}


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


class _SymEngineWalker:
    """One-shot walker.

    Encapsulates the var/param symbol → index maps so each ``_walk`` call
    stays O(tree size).  An instance is created once per
    :func:`lower_ode_to_ir` call.
    """

    __slots__ = ("_var_idx", "_par_idx", "_time_name")

    def __init__(
        self,
        var_idx: dict[str, int],
        par_idx: dict[str, int],
        time_name: str,
    ) -> None:
        self._var_idx = var_idx
        self._par_idx = par_idx
        self._time_name = time_name

    def walk(self, expr) -> Node:
        # 0. Plain Python numeric leaves — surface when a user writes a
        #    ``_jacobian`` row with literal ints / floats (e.g.
        #    ``row1 = [0, 1, 0]`` in Rossler).  Coerce before any
        #    SymEngine ``isinstance`` checks so the walker is a single
        #    dispatch.
        if isinstance(expr, bool):
            return Const(1.0 if expr else 0.0)
        if isinstance(expr, (int, float, np.integer, np.floating)):
            return Const(float(expr))

        # 1. Symbols dispatch on name.
        if isinstance(expr, se.Symbol):
            name = expr.name
            if name in self._var_idx:
                return Var(self._var_idx[name])
            if name in self._par_idx:
                return Param(self._par_idx[name])
            if name == self._time_name:
                return Time()
            raise NotLowerableError(f"unbound symbol {name!r} in lowered RHS")

        # 2. Numeric constants — anything ``float()``-able that isn't a
        #    Symbol / Function / Add / Mul / Pow.  Covers ``Integer``,
        #    ``Rational``, ``RealDouble``, ``Float``, ``One``, plus the
        #    named ``Constant`` subclass (``pi``, ``E``).  ``se.Constant``
        #    is not exposed at the package top level, so we import it
        #    directly from ``symengine_wrapper`` above.
        if isinstance(expr, (se.Number, _SymEngineConstant)):
            return Const(float(expr))

        # 3. Add / Mul: variadic in SymEngine, fold left into chained BinOps.
        if isinstance(expr, se.Add):
            args = list(expr.args)
            if not args:
                raise NotLowerableError("empty Add")
            node = self.walk(args[0])
            for a in args[1:]:
                node = BinOp("add", node, self.walk(a))
            return node

        if isinstance(expr, se.Mul):
            args = list(expr.args)
            if not args:
                raise NotLowerableError("empty Mul")
            node = self.walk(args[0])
            for a in args[1:]:
                node = BinOp("mul", node, self.walk(a))
            return node

        # 4. Pow.  Three sub-cases:
        #    a. Integer exponent (incl. negative)        → Pow(base, i32)
        #    b. Rational(1, 2) exponent (i.e. sqrt)     → UnaryOp("sqrt", base)
        #    c. Numeric non-integer exponent            → PowF(base, f64)
        #    d. Symbolic exponent (param / state expr)  → BinOp("pow", base, exp)
        if isinstance(expr, se.Pow):
            base, expnt = expr.args
            base_node = self.walk(base)
            if isinstance(expnt, se.Integer):
                return Pow(base_node, int(expnt))
            if isinstance(expnt, se.Rational) and expnt.p == 1 and expnt.q == 2:
                # Specialise ``base ** 1/2`` to ``sqrt(base)`` — the Rust
                # ``sqrt`` intrinsic is faster than a generic ``powf``.
                return UnaryOp("sqrt", base_node)
            if isinstance(expnt, (se.Number, _SymEngineConstant)):
                # Rational / Float / RealDouble — encode as PowF(f64).
                k = float(expnt)
                if k.is_integer():
                    # Defensive — SymEngine sometimes builds
                    # ``Pow(x, Rational(4, 2))`` which simplifies to int 2
                    # but reaches us as a Rational.
                    return Pow(base_node, int(k))
                return PowF(base_node, k)
            # Symbolic exponent: lower both operands and emit Pow2.
            # Hits CircadianRhythm (`x ** n` with n a parameter),
            # YuWang (`x ** (y * z)`), JerkCircuit (`x ** (y / p)`).
            return BinOp("pow", base_node, self.walk(expnt))

        # 5. Function: dispatch on class name.
        if isinstance(expr, se.Function):
            cls_name = type(expr).__name__
            if cls_name == "Derivative":
                # Unevaluated derivative — the user's RHS or our auto-diff
                # produced ``Derivative(Abs(x), x)`` or similar.  We can't
                # encode this in the IR.
                raise NotLowerableError(
                    f"unevaluated symbolic Derivative reached lowering: "
                    f"{expr!r} (likely diff(Abs(...)) or diff(sign(...))); "
                    f"provide an explicit _jacobian or rewrite without Abs/sign"
                )
            op = _UNARY_FUNCTION_TAGS.get(cls_name)
            if op is None:
                raise NotLowerableError(
                    f"unsupported SymEngine function {cls_name!r} (args={list(expr.args)!r})"
                )
            if len(expr.args) != 1:
                raise NotLowerableError(
                    f"function {cls_name!r} reached lowering with arity "
                    f"{len(expr.args)} (only unary supported)"
                )
            return UnaryOp(op, self.walk(expr.args[0]))

        raise NotLowerableError(f"unsupported SymEngine node type {type(expr).__name__} ({expr!r})")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lower_ode_to_ir(
    cls: type,
    *,
    dim: int,
    params: dict[str, Any],
    structural_params: frozenset[str] = frozenset(),
) -> CompiledOde:
    """Lower ``cls._equations`` (and optionally ``cls._jacobian``) to IR.

    Parameters
    ----------
    cls
        The :class:`~tsdynamics.base.ContinuousSystem` subclass to lower.
    dim
        State-space dimension.
    params
        ``{name: value}`` for every declared parameter.  Insertion order
        matters: the non-structural params keep their order in the
        resulting ``Param(idx)`` mapping, so callers must pass parameter
        values in that same order when invoking the Rust kernel.
    structural_params
        Names of parameters whose values are baked into the symbolic
        structure (e.g. loop length ``N`` for Lorenz-96).  These are
        substituted with their literal values during the trace; changing
        them therefore changes the IR (caller is expected to key the
        cache on ``hash(structural_params)``).

    Returns
    -------
    CompiledOde
        Serialised RHS (+ optional Jacobian) bytecode + metadata.

    Raises
    ------
    NotLowerableError
        If the RHS contains an op the IR can't represent.  Caller may
        catch and dispatch to the JiTCODE fallback path.
    """
    if not isinstance(structural_params, frozenset):
        structural_params = frozenset(structural_params)

    # 1. Build the placeholder symbols.
    var_syms = [se.Symbol(f"_v{i}") for i in range(dim)]
    t_sym = se.Symbol("_t")

    # Non-structural params keep their original insertion order.
    nonstructural = [k for k in params if k not in structural_params]
    par_syms: dict[str, se.Symbol] = {name: se.Symbol(f"_p_{name}") for name in nonstructural}

    var_idx = {sym.name: i for i, sym in enumerate(var_syms)}
    par_idx = {sym.name: i for i, sym in enumerate(par_syms.values())}

    def y_accessor(i: int) -> se.Symbol:
        return var_syms[i]

    kwargs: dict[str, Any] = {}
    for name, value in params.items():
        if name in structural_params:
            kwargs[name] = value
        else:
            kwargs[name] = par_syms[name]

    # 2. Trace the RHS.
    walker = _SymEngineWalker(var_idx, par_idx, t_sym.name)
    rhs_exprs = _eval_callable(cls._equations, y_accessor, t_sym, kwargs)
    if len(rhs_exprs) != dim:
        raise NotLowerableError(
            f"{cls.__name__}._equations returned {len(rhs_exprs)} expressions, expected dim={dim}"
        )
    rhs_nodes = [walker.walk(e) for e in rhs_exprs]

    # 3. Jacobian — explicit user override wins; otherwise symbolic diff.
    jac_nodes = _try_lower_jacobian(
        cls,
        walker,
        var_syms=var_syms,
        rhs_exprs=rhs_exprs,
        t_sym=t_sym,
        kwargs=kwargs,
        dim=dim,
    )

    return serialize_ode(
        dim=dim,
        param_names=tuple(nonstructural),
        rhs=rhs_nodes,
        jacobian=jac_nodes,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_callable(
    fn: Callable,
    y_accessor: Callable[[int], se.Symbol],
    t_sym: se.Symbol,
    kwargs: dict[str, Any],
) -> list:
    """Call ``fn(y, t, **kwargs)`` and normalise to a list of expressions.

    Both ``_equations`` and ``_jacobian`` historically accept positional
    or keyword params and may return tuples / lists / generators.  We
    normalise here so the walker sees a flat list.
    """
    try:
        result = fn(y_accessor, t_sym, **kwargs)
    except TypeError:
        # Some subclasses (e.g. Halvorsen._jacobian, Lorenz84._jacobian)
        # declare positional-only param signatures like
        # ``def _jacobian(Y, t, a, b)`` — the ``**kwargs`` call is the
        # contract everywhere, so a positional-only signature is a
        # latent bug.  We retry with positional args to keep N2 from
        # tripping on it; this exactly mirrors what
        # ``test_ode_rhs_symbolic.py`` does today.
        result = fn(y_accessor, t_sym, *kwargs.values())
    return list(result)


def _try_lower_jacobian(
    cls: type,
    walker: _SymEngineWalker,
    *,
    var_syms: list[se.Symbol],
    rhs_exprs: list,
    t_sym: se.Symbol,
    kwargs: dict[str, Any],
    dim: int,
) -> list[list[Node]] | None:
    """Best-effort Jacobian lowering — returns None on first failure.

    Three cases:

    1. ``cls.__dict__`` declares ``_jacobian`` directly — trace it the
       same way as ``_equations``.  Any walker failure raises (the user
       supplied an explicit Jacobian; if it doesn't lower they want to
       know).
    2. No explicit ``_jacobian`` — auto-differentiate the RHS using
       ``symengine.diff``.  If any cell contains a SymEngine
       ``Derivative`` (or otherwise fails to lower), bail out and
       return ``None``.
    3. Anything raises during the explicit-jacobian trace — caller
       should not silently drop, so we propagate.
    """
    if "_jacobian" in cls.__dict__:
        jac_rows = _eval_callable(cls._jacobian, lambda i: var_syms[i], t_sym, kwargs)
        if len(jac_rows) != dim:
            raise NotLowerableError(
                f"{cls.__name__}._jacobian returned {len(jac_rows)} rows, expected dim={dim}"
            )
        jac_grid: list[list[Node]] = []
        for row in jac_rows:
            cells = list(row)
            if len(cells) != dim:
                raise NotLowerableError(
                    f"{cls.__name__}._jacobian row has {len(cells)} cols, expected dim={dim}"
                )
            jac_grid.append([walker.walk(c) for c in cells])
        return jac_grid

    # Auto-diff path.
    try:
        return [
            [walker.walk(se.diff(rhs_exprs[i], var_syms[j])) for j in range(dim)]
            for i in range(dim)
        ]
    except NotLowerableError:
        return None


__all__ = ["lower_ode_to_ir"]
