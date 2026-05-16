"""Tracer for lowering ``DiscreteMap._step`` / ``_jacobian`` into the IR.

We run the user-supplied method once with :class:`Tracer` instances in
place of state and parameters. NumPy ufunc dispatch (`np.cos`,
`np.where`, …) routes through ``__array_ufunc__`` / ``__array_function__``
on the Tracer, producing IR nodes instead of float values. Python
operators (``+``, ``*``, ``**``, ``<``, ``&`` …) are wired the same way.

Operations the IR can't represent — `__bool__`, float `__pow__`, an
unhandled ufunc — raise :class:`NotLowerableError`; the caller catches it
and falls back to the Numba dispatch path.
"""

from __future__ import annotations

import numpy as np

from ._ir import BinOp, Const, NotLowerableError, Param, Pow, UnaryOp, Var, Where

# Numpy ufunc name → BinOp / UnaryOp tag.
_UFUNC_UNARY = {
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "absolute": "abs",
    "fabs": "abs",
    "sqrt": "sqrt",
    "arccos": "arccos",
    "sign": "sign",
    "negative": "neg",
}
_UFUNC_BINARY = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "true_divide": "div",
    "floor_divide": None,  # not supported
    "mod": "mod",
    "remainder": "mod",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "bitwise_and": "and",
    "logical_and": "and",
}


def _to_node(value):
    """Promote a Python/NumPy scalar (or pass through a Tracer) to a Node."""
    if isinstance(value, Tracer):
        return value._node
    if isinstance(value, bool):
        return Const(1.0 if value else 0.0)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return Const(float(value))
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return Const(float(value))
    raise NotLowerableError(f"cannot promote value of type {type(value).__name__} to IR")


class Tracer:
    """An IR node masquerading as a numeric scalar.

    Equality / hashing intentionally fall back to identity — comparison
    operators are used for control-flow elsewhere (e.g. ``y < alpha``)
    and must build IR nodes, not Python booleans.
    """

    __slots__ = ("_node",)

    # Make NumPy prefer our protocols over its own coercion.
    __array_priority__ = 1000.0

    def __init__(self, node) -> None:
        object.__setattr__(self, "_node", node)

    # -- internal --------------------------------------------------------

    def _bin(self, op: str, other, *, reflected: bool = False) -> Tracer:
        if reflected:
            left = _to_node(other)
            right = self._node
        else:
            left = self._node
            right = _to_node(other)
        return Tracer(BinOp(op, left, right))

    def _unary(self, op: str) -> Tracer:
        return Tracer(UnaryOp(op, self._node))

    # -- arithmetic ------------------------------------------------------

    def __add__(self, other):
        try:
            return self._bin("add", other)
        except NotLowerableError:
            return NotImplemented

    def __radd__(self, other):
        try:
            return self._bin("add", other, reflected=True)
        except NotLowerableError:
            return NotImplemented

    def __sub__(self, other):
        try:
            return self._bin("sub", other)
        except NotLowerableError:
            return NotImplemented

    def __rsub__(self, other):
        try:
            return self._bin("sub", other, reflected=True)
        except NotLowerableError:
            return NotImplemented

    def __mul__(self, other):
        try:
            return self._bin("mul", other)
        except NotLowerableError:
            return NotImplemented

    def __rmul__(self, other):
        try:
            return self._bin("mul", other, reflected=True)
        except NotLowerableError:
            return NotImplemented

    def __truediv__(self, other):
        try:
            return self._bin("div", other)
        except NotLowerableError:
            return NotImplemented

    def __rtruediv__(self, other):
        try:
            return self._bin("div", other, reflected=True)
        except NotLowerableError:
            return NotImplemented

    def __mod__(self, other):
        try:
            return self._bin("mod", other)
        except NotLowerableError:
            return NotImplemented

    def __rmod__(self, other):
        try:
            return self._bin("mod", other, reflected=True)
        except NotLowerableError:
            return NotImplemented

    def __neg__(self):
        return self._unary("neg")

    def __pos__(self):
        return self

    def __pow__(self, exponent):
        if isinstance(exponent, Tracer):
            raise NotLowerableError("power with non-constant exponent is not supported")
        if isinstance(exponent, bool):
            exponent = int(exponent)
        if isinstance(exponent, (np.integer, int)):
            return Tracer(Pow(self._node, int(exponent)))
        if isinstance(exponent, (np.floating, float)):
            if float(exponent).is_integer():
                return Tracer(Pow(self._node, int(exponent)))
            raise NotLowerableError(f"non-integer power not supported in IR (got {exponent!r})")
        raise NotLowerableError(f"unsupported exponent type {type(exponent).__name__}")

    def __rpow__(self, base):
        raise NotLowerableError("base of power must be a Tracer / constant exponent")

    # -- comparison ------------------------------------------------------

    def __lt__(self, other):
        try:
            return self._bin("lt", other)
        except NotLowerableError:
            return NotImplemented

    def __le__(self, other):
        try:
            return self._bin("le", other)
        except NotLowerableError:
            return NotImplemented

    def __gt__(self, other):
        try:
            return self._bin("gt", other)
        except NotLowerableError:
            return NotImplemented

    def __ge__(self, other):
        try:
            return self._bin("ge", other)
        except NotLowerableError:
            return NotImplemented

    def __and__(self, other):
        try:
            return self._bin("and", other)
        except NotLowerableError:
            return NotImplemented

    def __rand__(self, other):
        try:
            return self._bin("and", other, reflected=True)
        except NotLowerableError:
            return NotImplemented

    # Equality / hashing fall back to identity: equality on a Tracer
    # almost certainly indicates a coding error in a map's ``_step``.
    def __eq__(self, other):  # type: ignore[override]
        return self is other

    def __hash__(self):  # type: ignore[override]
        return id(self)

    def __bool__(self):
        raise NotLowerableError(
            "Tracer.__bool__() called — does the map's _step / _jacobian "
            "contain Python `if`/`and`/`or` on a state or parameter? "
            "Rewrite with np.where to make it lowerable."
        )

    # -- numpy dispatch --------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented

        name = ufunc.__name__
        if name in _UFUNC_UNARY and len(inputs) == 1:
            (a,) = inputs
            if not isinstance(a, Tracer):
                return NotImplemented
            return a._unary(_UFUNC_UNARY[name])

        if name in _UFUNC_BINARY and len(inputs) == 2:
            op = _UFUNC_BINARY[name]
            if op is None:
                raise NotLowerableError(f"ufunc {name!r} is not supported")
            a, b = inputs
            return Tracer(BinOp(op, _to_node(a), _to_node(b)))

        raise NotLowerableError(f"ufunc {name!r} is not supported")

    def __array_function__(self, func, types, args, kwargs):
        # We only need `np.where` for branching maps (Tent, Baker).
        if func is np.where:
            if len(args) == 3:
                cond, t, f = args
            elif len(args) == 1:
                raise NotLowerableError("np.where(cond) with no branches is not supported")
            else:
                return NotImplemented
            return Tracer(Where(_to_node(cond), _to_node(t), _to_node(f)))
        return NotImplemented

    # -- introspection ---------------------------------------------------

    def __repr__(self) -> str:
        return f"Tracer({self._node!r})"


def state_tracer(dim: int):
    """Build the ``X`` argument for a map's ``_step`` / ``_jacobian``.

    For ``dim == 1`` the existing convention is ``x = X`` (no unpacking),
    so we hand back a single :class:`Tracer`. For ``dim >= 2`` we hand
    back a tuple — supports both ``x, y = X`` unpacking and ``X[i]``
    indexing.
    """
    if dim == 1:
        return Tracer(Var(0))
    return tuple(Tracer(Var(i)) for i in range(dim))


def param_tracers(n_params: int) -> tuple[Tracer, ...]:
    return tuple(Tracer(Param(i)) for i in range(n_params))
