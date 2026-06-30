r"""
Interval-Newton / Krawczyk root finding over a box (stream **A-FP**, ``method="interval"``).

A *rigorous*, deterministic alternative to multi-start Newton: instead of seeding
many local solves and hoping to hit every basin, the Krawczyk operator brackets
**all** roots of the residual :math:`g(x)=0` inside a search box by interval
branch-and-prune, with a uniqueness/existence certificate per sub-box.

The pieces:

- :class:`Interval` — a scalar interval ``[lo, hi]`` with the arithmetic and
  elementary functions the catalogue kernels use (``+ - * / **`` for integer
  powers, ``sin``/``cos``/``cosh``/``tanh``/``exp``/``log``/``sqrt``/``abs``).  It
  also implements ``__array_ufunc__`` so a kernel written with ``numpy`` ufuncs
  (``np.sin`` …) evaluates on intervals unchanged.
- :class:`IntervalJet` — forward-mode automatic differentiation over
  :class:`Interval`: a value-interval plus a gradient-interval vector.  Pushing
  ``dim`` seeded jets through the residual yields the residual interval **and**
  the interval Jacobian in one pass, with **no symbolic differentiation** (so a
  flow whose symbolic Jacobian would blow up into reciprocal/variable powers is
  handled exactly through the operations the kernel itself performs).
- :func:`krawczyk_roots` — the branch-and-prune loop.  For a box ``[x]`` with
  midpoint ``m`` and a preconditioner ``Y ≈ J(m)^{-1}`` it forms the Krawczyk
  image :math:`K([x]) = m - Y g(m) + (I - Y G([x]))([x] - m)`.  If
  :math:`K \subset \operatorname{int}[x]` the box holds a **unique** root
  (existence + uniqueness certificate); if :math:`K \cap [x] = \varnothing` the
  box holds **no** root and is pruned; otherwise the box is contracted to
  :math:`K \cap [x]` (or bisected when contraction stalls).

The interval arithmetic here is *not* outward-rounded (it uses plain ``float``
ops), so the bracketing is rigorous only up to floating-point round-off — fixed
points are recovered to machine precision (each box is polished by a final
Newton step), which is the contract :func:`tsdynamics.fixed_points` documents.
A certified-rounded kernel would live engine-side (a larger future project; see
``docs/theory/fixed-points-interval.md``).

References
----------
Krawczyk (1969), *Computing* 4, 187.
Neumaier (1990), *Interval Methods for Systems of Equations*, CUP.
Moore, Kearfott & Cloud (2009), *Introduction to Interval Analysis*, SIAM.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tsdynamics.errors import InvalidInputError

if TYPE_CHECKING:
    from tsdynamics.families import ContinuousSystem, DiscreteMap
    from tsdynamics.families.base import ParamSet

__all__ = ["Interval", "IntervalJet", "krawczyk_roots", "map_interval_fn", "flow_interval_fn"]


# ── scalar interval ───────────────────────────────────────────────────────────


class Interval:
    r"""A scalar real interval ``[lo, hi]`` with overloaded arithmetic.

    Supports the operations the system kernels use; an unsupported operation
    (a non-integer/variable power, a comparison, a modulo) raises
    :class:`NotImplementedError`, which the public entry point turns into a clear
    :class:`~tsdynamics.errors.InvalidInputError`.  Not outward-rounded — rigorous
    to floating-point round-off (see the module docstring).
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: float | None = None) -> None:
        if hi is None:
            hi = lo
        self.lo = float(lo)
        self.hi = float(hi)

    # numpy interop: let np.sin(iv) etc. dispatch to the Interval methods, so a
    # kernel written with numpy ufuncs evaluates on intervals unchanged.
    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__" or kwargs:
            return NotImplemented
        name = ufunc.__name__
        fn = _UFUNC_MAP.get(name)
        if fn is None:
            raise NotImplementedError(
                f"interval method does not support numpy ufunc {name!r} in the kernel."
            )
        return fn(*(_coerce(a) for a in inputs))

    def __add__(self, o: Any) -> Interval:
        o = _coerce(o)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, o: Any) -> Interval:
        o = _coerce(o)
        return Interval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o: Any) -> Interval:
        return _coerce(o).__sub__(self)

    def __mul__(self, o: Any) -> Interval:
        o = _coerce(o)
        p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(min(p), max(p))

    __rmul__ = __mul__

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __pos__(self) -> Interval:
        return Interval(self.lo, self.hi)

    def __abs__(self) -> Interval:
        if self.lo >= 0.0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0.0:
            return Interval(-self.hi, -self.lo)
        return Interval(0.0, max(-self.lo, self.hi))

    def __pow__(self, n: Any) -> Interval:
        if isinstance(n, Interval):
            if n.lo != n.hi:
                raise NotImplementedError("interval method: variable exponent in a power.")
            n = n.lo
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if not isinstance(n, int):
            raise NotImplementedError(f"interval method: non-integer power {n!r}.")
        if n == 0:
            return Interval(1.0, 1.0)
        if n < 0:
            return Interval(1.0, 1.0) / (self ** (-n))
        a, b = self.lo**n, self.hi**n
        if n % 2 == 0 and self.lo <= 0.0 <= self.hi:
            return Interval(0.0, max(a, b))
        return Interval(min(a, b), max(a, b))

    def __truediv__(self, o: Any) -> Interval:
        o = _coerce(o)
        if o.lo <= 0.0 <= o.hi:
            # division by an interval straddling zero -> unbounded; the caller's
            # Krawczyk loop treats an infinite image as "inconclusive -> bisect".
            return Interval(-math.inf, math.inf)
        recip = Interval(1.0 / o.hi, 1.0 / o.lo)
        return self * recip

    def __rtruediv__(self, o: Any) -> Interval:
        return _coerce(o).__truediv__(self)

    def width(self) -> float:
        return self.hi - self.lo

    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Interval([{self.lo:g}, {self.hi:g}])"


def _coerce(x: Any) -> Interval:
    return x if isinstance(x, Interval) else Interval(float(x))


# ── elementary interval functions ─────────────────────────────────────────────


def _i_sin(x: Interval) -> Interval:
    lo, hi = x.lo, x.hi
    vals = [math.sin(lo), math.sin(hi)]
    # extrema of sin at pi/2 + k*pi
    k0 = math.ceil((lo - math.pi / 2.0) / math.pi)
    k1 = math.floor((hi - math.pi / 2.0) / math.pi)
    for k in range(k0, k1 + 1):
        vals.append(1.0 if k % 2 == 0 else -1.0)
    return Interval(min(vals), max(vals))


def _i_cos(x: Interval) -> Interval:
    lo, hi = x.lo, x.hi
    vals = [math.cos(lo), math.cos(hi)]
    # extrema of cos at k*pi
    k0 = math.ceil(lo / math.pi)
    k1 = math.floor(hi / math.pi)
    for k in range(k0, k1 + 1):
        vals.append(1.0 if k % 2 == 0 else -1.0)
    return Interval(min(vals), max(vals))


def _i_tan(x: Interval) -> Interval:
    # tan is monotone on each branch; if the box spans a pole the image is
    # unbounded (Krawczyk -> bisect).  Detect a pole at pi/2 + k*pi in [lo, hi].
    lo, hi = x.lo, x.hi
    k0 = math.ceil((lo - math.pi / 2.0) / math.pi)
    k1 = math.floor((hi - math.pi / 2.0) / math.pi)
    if k1 >= k0:
        return Interval(-math.inf, math.inf)
    return Interval(math.tan(lo), math.tan(hi))


def _i_exp(x: Interval) -> Interval:
    return Interval(math.exp(x.lo), math.exp(x.hi))


def _i_log(x: Interval) -> Interval:
    if x.lo <= 0.0:
        return Interval(-math.inf, math.log(x.hi) if x.hi > 0.0 else -math.inf)
    return Interval(math.log(x.lo), math.log(x.hi))


def _i_sqrt(x: Interval) -> Interval:
    lo = math.sqrt(x.lo) if x.lo > 0.0 else 0.0
    hi = math.sqrt(x.hi) if x.hi > 0.0 else 0.0
    return Interval(lo, hi)


def _i_rpow(x: Interval, p: float) -> Interval:
    r"""Interval image of ``x ** p`` for a real (non-integer) exponent ``p``.

    Defined for a non-negative base (``x ** p = exp(p log x)`` is real only there);
    the function is monotone on ``[0, ∞)`` (increasing for ``p > 0``, decreasing for
    ``p < 0``), so the image is the ``p``-power of the endpoints.  A base straddling
    or below zero yields a ``[0, ∞)`` / unbounded enclosure (the Krawczyk loop then
    bisects), matching the conservative handling of :func:`_i_sqrt` / :func:`_i_log`.
    """
    lo = x.lo if x.lo > 0.0 else 0.0
    hi = x.hi if x.hi > 0.0 else 0.0
    if p >= 0.0:
        return Interval(lo**p, hi**p)
    # p < 0: x**p is decreasing; a lower endpoint at 0 blows up to +inf.
    a = math.inf if lo == 0.0 else lo**p
    b = math.inf if hi == 0.0 else hi**p
    return Interval(min(a, b), max(a, b))


def _i_sinh(x: Interval) -> Interval:
    return Interval(math.sinh(x.lo), math.sinh(x.hi))


def _i_cosh(x: Interval) -> Interval:
    lo, hi = x.lo, x.hi
    if lo >= 0.0:
        return Interval(math.cosh(lo), math.cosh(hi))
    if hi <= 0.0:
        return Interval(math.cosh(hi), math.cosh(lo))
    return Interval(1.0, max(math.cosh(lo), math.cosh(hi)))


def _i_tanh(x: Interval) -> Interval:
    return Interval(math.tanh(x.lo), math.tanh(x.hi))


def _i_atan(x: Interval) -> Interval:
    return Interval(math.atan(x.lo), math.atan(x.hi))


def _i_abs(x: Interval) -> Interval:
    return abs(x)


def _i_sign(x: Interval) -> Interval:
    lo = -1.0 if x.lo < 0.0 else (1.0 if x.lo > 0.0 else 0.0)
    hi = -1.0 if x.hi < 0.0 else (1.0 if x.hi > 0.0 else 0.0)
    return Interval(min(lo, hi), max(lo, hi))


_UFUNC_MAP: dict[str, Callable[[Interval], Interval]] = {
    "sin": _i_sin,
    "cos": _i_cos,
    "tan": _i_tan,
    "exp": _i_exp,
    "log": _i_log,
    "sqrt": _i_sqrt,
    "sinh": _i_sinh,
    "cosh": _i_cosh,
    "tanh": _i_tanh,
    "arctan": _i_atan,
    "absolute": _i_abs,
    "fabs": _i_abs,
    "sign": _i_sign,
}


# ── forward-mode AD over intervals ────────────────────────────────────────────


class IntervalJet:
    r"""Interval value plus interval gradient (forward-mode AD over :class:`Interval`).

    Pushing ``dim`` jets seeded at the box (the ``k``-th with unit gradient in
    direction ``k``) through the residual produces, for each output, its value
    interval and one Jacobian row — the interval residual and interval Jacobian in
    a single pass, with no symbolic differentiation.
    """

    __slots__ = ("v", "g")

    def __init__(self, v: Interval, g: list[Interval]) -> None:
        self.v = v
        self.g = g

    @staticmethod
    def const(c: float, n: int) -> IntervalJet:
        return IntervalJet(Interval(c), [Interval(0.0) for _ in range(n)])

    @staticmethod
    def var(value: Interval, k: int, n: int) -> IntervalJet:
        g = [Interval(0.0) for _ in range(n)]
        g[k] = Interval(1.0)
        return IntervalJet(value, g)

    def _coerce(self, o: Any) -> IntervalJet:
        if isinstance(o, IntervalJet):
            return o
        n = len(self.g)
        return IntervalJet(_coerce(o), [Interval(0.0) for _ in range(n)])

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__" or kwargs:
            return NotImplemented
        name = ufunc.__name__
        fn = _JET_UFUNC_MAP.get(name)
        if fn is None:
            raise NotImplementedError(
                f"interval method does not support numpy ufunc {name!r} in the kernel."
            )
        (a,) = inputs
        return fn(a if isinstance(a, IntervalJet) else self._coerce(a))

    def __add__(self, o: Any) -> IntervalJet:
        o = self._coerce(o)
        return IntervalJet(self.v + o.v, [a + b for a, b in zip(self.g, o.g, strict=True)])

    __radd__ = __add__

    def __sub__(self, o: Any) -> IntervalJet:
        o = self._coerce(o)
        return IntervalJet(self.v - o.v, [a - b for a, b in zip(self.g, o.g, strict=True)])

    def __rsub__(self, o: Any) -> IntervalJet:
        return self._coerce(o).__sub__(self)

    def __mul__(self, o: Any) -> IntervalJet:
        o = self._coerce(o)
        return IntervalJet(
            self.v * o.v,
            [self.v * b + a * o.v for a, b in zip(self.g, o.g, strict=True)],
        )

    __rmul__ = __mul__

    def __neg__(self) -> IntervalJet:
        return IntervalJet(-self.v, [-a for a in self.g])

    def __pos__(self) -> IntervalJet:
        return self

    def __abs__(self) -> IntervalJet:
        # |x| is non-smooth at 0; if 0 is strictly inside the value interval the
        # derivative is set-valued ([-1, 1]).  Box that case so the Jacobian
        # enclosure stays valid (Krawczyk then bisects through it).
        if self.v.lo < 0.0 < self.v.hi:
            d = Interval(-1.0, 1.0)
            return IntervalJet(abs(self.v), [d * a for a in self.g])
        s = -1.0 if self.v.hi <= 0.0 else 1.0
        return IntervalJet(abs(self.v), [Interval(s) * a for a in self.g])

    def __pow__(self, n: Any) -> IntervalJet:
        if isinstance(n, IntervalJet):
            if n.v.lo != n.v.hi or any(gi.lo != 0.0 or gi.hi != 0.0 for gi in n.g):
                raise NotImplementedError("interval method: variable exponent in a power.")
            n = n.v.lo
        if isinstance(n, Interval):
            if n.lo != n.hi:
                raise NotImplementedError("interval method: variable exponent in a power.")
            n = n.lo
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if not isinstance(n, int):
            raise NotImplementedError(f"interval method: non-integer power {n!r}.")
        if n == 0:
            return IntervalJet.const(1.0, len(self.g))
        coef = Interval(float(n)) * (self.v ** (n - 1))
        return IntervalJet(self.v**n, [coef * a for a in self.g])

    def real_pow(self, p: float) -> IntervalJet:
        r"""Forward-AD of ``self ** p`` for a real (non-integer) exponent ``p``.

        The value is the interval real power (:func:`_i_rpow`) and the gradient
        carries the chain rule ``d/dx (x ** p) = p * x ** (p - 1)`` over intervals.
        Used by the flow path for ``sqrt`` / fractional powers of a state expression.
        """
        coef = Interval(p) * _i_rpow(self.v, p - 1.0)
        return IntervalJet(_i_rpow(self.v, p), [coef * a for a in self.g])

    def __truediv__(self, o: Any) -> IntervalJet:
        o = self._coerce(o)
        val = self.v / o.v
        v2 = o.v * o.v
        return IntervalJet(
            val,
            [(a * o.v - self.v * b) / v2 for a, b in zip(self.g, o.g, strict=True)],
        )

    def __rtruediv__(self, o: Any) -> IntervalJet:
        return self._coerce(o).__truediv__(self)


def _jet_unary(
    f: Callable[[Interval], Interval], df: Callable[[Interval], Interval]
) -> Callable[[IntervalJet], IntervalJet]:
    def wrap(x: IntervalJet) -> IntervalJet:
        d = df(x.v)
        return IntervalJet(f(x.v), [d * a for a in x.g])

    return wrap


_JET_UFUNC_MAP: dict[str, Callable[[IntervalJet], IntervalJet]] = {
    "sin": _jet_unary(_i_sin, _i_cos),
    "cos": _jet_unary(_i_cos, lambda x: -_i_sin(x)),
    "exp": _jet_unary(_i_exp, _i_exp),
    "log": _jet_unary(_i_log, lambda x: Interval(1.0) / x),
    "sqrt": _jet_unary(_i_sqrt, lambda x: Interval(0.5) / _i_sqrt(x)),
    "sinh": _jet_unary(_i_sinh, _i_cosh),
    "cosh": _jet_unary(_i_cosh, _i_sinh),
    "tanh": _jet_unary(_i_tanh, lambda x: Interval(1.0) - _i_tanh(x) * _i_tanh(x)),
    "absolute": lambda x: abs(x),
    "fabs": lambda x: abs(x),
}


# ── symengine tree evaluation with IntervalJet (flows) ────────────────────────

# symengine elementary-function node name -> the IntervalJet wrapper.  ``exp`` and
# ``sqrt`` are deliberately absent: symengine canonicalizes ``exp(z)`` to
# ``Pow(E, z)`` and ``sqrt(z)`` to ``Pow(z, 1/2)``, so they reach the tree-walk as
# ``Pow`` nodes handled in :func:`_eval_jet` (never as named function nodes).
_SE_JET_FUNCS: dict[str, Callable[[IntervalJet], IntervalJet]] = {
    "sin": _JET_UFUNC_MAP["sin"],
    "cos": _JET_UFUNC_MAP["cos"],
    "log": _JET_UFUNC_MAP["log"],
    "sinh": _JET_UFUNC_MAP["sinh"],
    "cosh": _JET_UFUNC_MAP["cosh"],
    "tanh": _JET_UFUNC_MAP["tanh"],
    "Abs": lambda x: abs(x),
}


def _eval_jet(expr: Any, env: dict[str, Any]) -> IntervalJet:
    """Evaluate a symengine ``expr`` over :class:`IntervalJet` values in ``env``.

    ``env`` maps the state-symbol strings (``"y(0)"`` …) to seeded jets and
    carries ``"__n__"`` (the dimension).  Raises :class:`NotImplementedError` on a
    node the interval engine does not model.
    """
    import symengine as se

    if isinstance(expr, se.Symbol):
        key = str(expr)
        if key in env:
            return cast("IntervalJet", env[key])
        raise NotImplementedError(f"interval method: free symbol {key!r} in the RHS.")
    if expr.is_Number:
        return IntervalJet.const(float(expr), env["__n__"])

    name = type(expr).__name__
    if name in ("Exp1", "E"):  # the constant e: symengine does not flag it is_Number
        return IntervalJet.const(float(expr), env["__n__"])
    args = expr.args
    if name == "Add":
        acc = IntervalJet.const(0.0, env["__n__"])
        for a in args:
            acc = acc + _eval_jet(a, env)
        return acc
    if name == "Mul":
        acc = IntervalJet.const(1.0, env["__n__"])
        for a in args:
            acc = acc * _eval_jet(a, env)
        return acc
    if name == "Pow":
        # symengine canonicalizes exp(z) -> Pow(E, z) and sqrt(z) -> Pow(z, 1/2),
        # so the elementary functions reach the tree-walk as Pow nodes, not as
        # named function nodes.  Handle: (a) base E -> exp jet of the exponent;
        # (b) constant Pow(number, number) -> folded constant (e.g. sqrt(2)); the
        # foldable case must precede the integer/real-power dispatch so a numeric
        # 1/2 exponent never raises.  (c) integer exponent -> repeated product;
        # (d) real (non-integer) numeric exponent -> the real-power jet.
        base_expr, exp_expr = args[0], args[1]
        if type(base_expr).__name__ in ("Exp1", "E"):  # exp(exp_expr)
            return _JET_UFUNC_MAP["exp"](_eval_jet(exp_expr, env))
        if base_expr.is_Number and exp_expr.is_Number:
            return IntervalJet.const(float(base_expr) ** float(exp_expr), env["__n__"])
        base = _eval_jet(base_expr, env)
        if exp_expr.is_Number:
            pf = float(exp_expr)
            if pf.is_integer():
                return base ** int(pf)
            return base.real_pow(pf)
        raise NotImplementedError(f"interval method: non-integer power {exp_expr!r} in the RHS.")
    fn = _SE_JET_FUNCS.get(name)
    if fn is not None:
        return fn(_eval_jet(args[0], env))
    # A FunctionSymbol y(i) appears as a Symbol above; anything else is unmodelled.
    key = str(expr)
    if key in env:
        return cast("IntervalJet", env[key])
    raise NotImplementedError(f"interval method: unsupported RHS node {name!r} ({expr}).")


# ── residual builders (map / flow) ────────────────────────────────────────────

ResidJacIv = Callable[[np.ndarray, np.ndarray], tuple[list[Interval], list[list[Interval]]]]


def map_interval_fn(system: DiscreteMap) -> ResidJacIv:
    r"""Build the interval residual+Jacobian of ``f(x) - x`` for a discrete map.

    Pushes seeded :class:`IntervalJet` state through the class's pure-Python
    ``_step`` (so the operations are exactly those the kernel performs).  Raises
    :class:`~tsdynamics.errors.InvalidInputError` at build time if a probe through
    ``_step`` hits an operation the interval engine cannot model (a comparison, a
    modulo, a non-integer power) — the additive ``method="interval"`` then asks the
    caller to use ``method="newton"`` for that system.
    """
    cls = type(system)
    dim = int(cast("int", system.dim))
    params = tuple(cast("ParamSet", system.params).as_tuple())

    def to_native(jets: list[IntervalJet]) -> Any:
        return jets[0] if dim == 1 else jets

    def resid_jac(lo: np.ndarray, hi: np.ndarray) -> tuple[list[Interval], list[list[Interval]]]:
        jets = [IntervalJet.var(Interval(lo[i], hi[i]), i, dim) for i in range(dim)]
        out = cls._step(to_native(jets), *params)
        out_list = [out] if dim == 1 else list(out)
        g: list[Interval] = []
        jac: list[list[Interval]] = []
        for i in range(dim):
            oi = out_list[i]
            oi = oi if isinstance(oi, IntervalJet) else IntervalJet.const(float(oi), dim)
            # g_i = f_i(x) - x_i  -> value and gradient (subtract the i-th unit)
            gi_val = oi.v - Interval(lo[i], hi[i])
            row = [oi.g[c] - (Interval(1.0) if c == i else Interval(0.0)) for c in range(dim)]
            g.append(gi_val)
            jac.append(row)
        return g, jac

    _probe(resid_jac, system, dim)
    return resid_jac


def flow_interval_fn(system: ContinuousSystem) -> ResidJacIv:
    r"""Build the interval residual+Jacobian of ``f(x) = 0`` for a continuous flow.

    Walks the symengine RHS expression tree over :class:`IntervalJet`, so the
    interval Jacobian is the forward-AD gradient of the *residual itself* — never
    a symbolically differentiated (and possibly reciprocal-/variable-power) form.
    Raises :class:`~tsdynamics.errors.InvalidInputError` if the RHS contains a node
    the interval engine cannot model.
    """
    from tsdynamics.engine.symbols import state_time_symbols

    cls = type(system)
    dim = int(cast("int", system.dim))
    y, t = state_time_symbols()
    exprs = list(cls._equations(y, t, **dict(system.params)))
    keys = [f"y({i})" for i in range(dim)]

    def resid_jac(lo: np.ndarray, hi: np.ndarray) -> tuple[list[Interval], list[list[Interval]]]:
        env: dict[str, Any] = {"__n__": dim}
        for i in range(dim):
            env[keys[i]] = IntervalJet.var(Interval(lo[i], hi[i]), i, dim)
        outs = [_eval_jet(exprs[r], env) for r in range(dim)]
        g = [o.v for o in outs]
        jac = [[outs[r].g[c] for c in range(dim)] for r in range(dim)]
        return g, jac

    _probe(resid_jac, system, dim)
    return resid_jac


def _probe(resid_jac: ResidJacIv, system: Any, dim: int) -> None:
    """Evaluate the interval residual once on a tiny box to surface an unmodelled op.

    Runs at build time so a system the interval engine cannot handle fails *before*
    the branch-and-prune loop, with a clear, actionable message.
    """
    lo = np.full(dim, -0.1)
    hi = np.full(dim, 0.1)
    try:
        resid_jac(lo, hi)
    except (NotImplementedError, TypeError, ValueError) as exc:
        # An unmodelled op surfaces either as our explicit NotImplementedError
        # (a non-integer power, a numpy ufunc we do not enclose) or as a bare
        # TypeError/ValueError from the kernel (``%``, a ``<`` comparison, an
        # ``np.where``) hitting an operator the Interval/IntervalJet lacks.
        raise InvalidInputError(
            f"method='interval' cannot enclose {type(system).__name__}: {exc} "
            f"Use method='newton' (or 'sd'/'dl' for a map) for this system."
        ) from exc


# ── Krawczyk branch-and-prune ─────────────────────────────────────────────────


def _krawczyk_image(
    lo: np.ndarray,
    hi: np.ndarray,
    resid_jac: ResidJacIv,
    jac_float: Callable[[np.ndarray], np.ndarray],
) -> tuple[str, np.ndarray, np.ndarray]:
    """One Krawczyk step on ``[lo, hi]``; returns ``(status, new_lo, new_hi)``.

    ``status`` is ``"unique"`` (K ⊂ int box: a unique root), ``"empty"`` (no
    root), ``"shrunk"`` (contracted to a nonempty intersection), or ``"bisect"``
    (singular preconditioner — caller bisects).
    """
    dim = lo.size
    m = 0.5 * (lo + hi)
    try:
        ydir = np.linalg.inv(jac_float(m))
    except np.linalg.LinAlgError:
        return "bisect", lo, hi
    if not np.all(np.isfinite(ydir)):
        return "bisect", lo, hi

    gm, _ = resid_jac(m, m)  # residual at the midpoint (a point interval)
    _, g_mat = resid_jac(lo, hi)  # interval Jacobian over the box

    # K_i = m_i - (Y g(m))_i + sum_j (I - Y G)_ij (box_j - m_j)
    eye = np.eye(dim)
    klo = np.empty(dim)
    khi = np.empty(dim)
    dbox = [Interval(lo[j] - m[j], hi[j] - m[j]) for j in range(dim)]
    for i in range(dim):
        acc = Interval(m[i])
        ygm = Interval(0.0)
        for j in range(dim):
            ygm = ygm + float(ydir[i, j]) * gm[j]
        acc = acc - ygm
        for j in range(dim):
            mij = Interval(0.0)
            for k in range(dim):
                mij = mij + float(ydir[i, k]) * g_mat[k][j]
            mij = Interval(eye[i, j]) - mij
            acc = acc + mij * dbox[j]
        klo[i], khi[i] = acc.lo, acc.hi

    if not (np.all(np.isfinite(klo)) and np.all(np.isfinite(khi))):
        return "bisect", lo, hi

    nlo = np.maximum(klo, lo)
    nhi = np.minimum(khi, hi)
    if np.any(nlo > nhi):
        return "empty", lo, hi
    if np.all(klo > lo) and np.all(khi < hi):
        return "unique", nlo, nhi
    return "shrunk", nlo, nhi


def krawczyk_roots(
    lo0: np.ndarray,
    hi0: np.ndarray,
    resid_jac: ResidJacIv,
    residual_float: Callable[[np.ndarray], np.ndarray],
    jac_float: Callable[[np.ndarray], np.ndarray],
    *,
    tol: float = 1e-12,
    max_boxes: int = 2_000_000,
    polish_iter: int = 25,
) -> list[np.ndarray]:
    r"""Branch-and-prune for all roots of the residual in the box ``[lo0, hi0]``.

    Returns the deduplicated root points (midpoints of the certified sub-boxes,
    polished by a final float Newton).  Each returned root carries a Krawczyk
    existence-and-uniqueness certificate for its sub-box.  Raises
    :class:`~tsdynamics.errors.ConvergenceError` if the box budget is exhausted.
    """
    from tsdynamics.errors import ConvergenceError

    stack: list[tuple[np.ndarray, np.ndarray]] = [
        (np.asarray(lo0, dtype=float), np.asarray(hi0, dtype=float))
    ]
    roots: list[np.ndarray] = []
    n_boxes = 0
    while stack:
        lo, hi = stack.pop()
        n_boxes += 1
        if n_boxes > max_boxes:
            raise ConvergenceError(
                f"method='interval' exceeded the box budget ({max_boxes}); the search "
                f"box may be too large or the residual too flat — shrink 'region'."
            )
        width = float(np.max(hi - lo))
        if width < 1e-15:
            m = 0.5 * (lo + hi)
            if float(np.linalg.norm(residual_float(m))) < 1e-7:
                roots.append(m)
            continue

        status, nlo, nhi = _krawczyk_image(lo, hi, resid_jac, jac_float)
        if status == "empty":
            continue
        if status == "unique":
            blo, bhi = nlo, nhi
            ok = True
            for _ in range(80):
                if float(np.max(bhi - blo)) < tol:
                    break
                st, l2, h2 = _krawczyk_image(blo, bhi, resid_jac, jac_float)
                if st == "empty":
                    ok = False
                    break
                blo, bhi = l2, h2
            if ok:
                roots.append(
                    _polish(0.5 * (blo + bhi), residual_float, jac_float, tol, polish_iter)
                )
            continue
        # inconclusive: contract if Krawczyk genuinely shrank the box, else bisect.
        if status == "shrunk" and float(np.max(nhi - nlo)) < 0.5 * width:
            stack.append((nlo, nhi))
            continue
        k = int(np.argmax(nhi - nlo))
        mid = 0.5 * (nlo[k] + nhi[k])
        h1 = nhi.copy()
        h1[k] = mid
        l2b = nlo.copy()
        l2b[k] = mid
        stack.append((nlo.copy(), h1))
        stack.append((l2b, nhi.copy()))
    return roots


def _polish(
    m: np.ndarray,
    residual_float: Callable[[np.ndarray], np.ndarray],
    jac_float: Callable[[np.ndarray], np.ndarray],
    tol: float,
    polish_iter: int,
) -> np.ndarray:
    """Run a few float Newton steps to land the certified-box midpoint on the root."""
    x = np.asarray(m, dtype=float).copy()
    for _ in range(polish_iter):
        g = residual_float(x)
        if float(np.linalg.norm(g)) < tol:
            break
        try:
            dx = np.linalg.solve(jac_float(x), -g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dx)):
            break
        x = x + dx
    return x
