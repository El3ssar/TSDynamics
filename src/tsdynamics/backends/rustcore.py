"""Experimental Rust-backed RHS evaluation and ensemble integration (``tsdynamics-core``).

The same symbolic ``_equations`` that feed the DiffSL backend are lowered here
to a flat *instruction tape* — single-static-assignment primitives over a
register file — which the Rust crate evaluates with no Python callbacks and no
runtime compiler.  That makes ensemble integration GIL-free and rayon-parallel
(the basin/Monte-Carlo primitive), and is the foundation the future Rust SDE
and DDE solvers build on.

``tsdynamics`` itself stays pure-Python; this module is inert unless the
optional accelerator is installed::

    pip install tsdynamics-core    # or the [rustcore] extra

Limitations (experimental): ODEs only; fixed-step explicit RK4 (no stiff
support — use the ``jitcode``/``diffsol`` backends for stiff systems); the RHS
must use functions the tape VM provides (the same set the DiffSL backend
supports).  Unsupported constructs raise :class:`TapeCompileError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "TapeCompileError",
    "available",
    "compile_tape",
    "ensemble_final",
    "eval_rhs",
    "integrate_dense",
]

# ---------------------------------------------------------------------------
# Opcodes — MUST stay in sync with crates/tsdynamics-core/src/vm.rs
# ---------------------------------------------------------------------------
OP_CONST = 0
OP_STATE = 1
OP_PARAM = 2
OP_TIME = 3
OP_ADD = 10
OP_SUB = 11
OP_MUL = 12
OP_DIV = 13
OP_POW = 14  # regs[a] ** regs[b]
OP_POWI = 15  # regs[a] ** b   (b is the integer exponent)
OP_NEG = 20
OP_RECIP = 21
_FUNC_OPS = {
    "sin": 30,
    "cos": 31,
    "tan": 32,
    "exp": 33,
    "log": 34,
    "sqrt": 35,
    "Abs": 36,
    "sign": 37,
    "sinh": 38,
    "cosh": 39,
    "tanh": 40,
    "asin": 41,
    "acos": 42,
    "atan": 43,
    "asinh": 44,
    "acosh": 45,
    "atanh": 46,
}
_OP_SQRT = 35


class TapeCompileError(NotImplementedError):
    """The system's RHS uses a construct the tape VM cannot express."""


def available() -> bool:
    """Whether the optional ``tsdynamics-core`` accelerator is installed."""
    try:
        import tsdynamics_core  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Symbolic RHS  →  instruction tape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledTape:
    """A system RHS lowered to flat arrays for the Rust tape VM."""

    ops: np.ndarray  # int32 (n_instr,)
    a: np.ndarray  # int32
    b: np.ndarray  # int32
    imm: np.ndarray  # float64
    outputs: np.ndarray  # int32 (dim,) — register holding each derivative
    n_state: int
    n_param: int
    control_names: list[str]


class _Emitter:
    """Lower a sympy expression DAG to SSA instructions, sharing subexpressions."""

    def __init__(self) -> None:
        self.ops: list[int] = []
        self.a: list[int] = []
        self.b: list[int] = []
        self.imm: list[float] = []
        self._cache: dict[Any, int] = {}

    def _push(self, op: int, a: int = 0, b: int = 0, imm: float = 0.0) -> int:
        idx = len(self.ops)
        self.ops.append(op)
        self.a.append(a)
        self.b.append(b)
        self.imm.append(imm)
        return idx

    def emit(self, expr: Any) -> int:
        cached = self._cache.get(expr)
        if cached is not None:
            return cached
        idx = self._emit(expr)
        self._cache[expr] = idx
        return idx

    def _emit(self, expr: Any) -> int:
        import sympy

        # Any symbol-free subexpression (numbers, pi, e, constant folds).
        if not expr.free_symbols:
            return self._push(OP_CONST, imm=float(expr))

        if isinstance(expr, sympy.Symbol):
            name = expr.name
            if name == "t":
                return self._push(OP_TIME)
            if name.startswith("u"):
                return self._push(OP_STATE, a=int(name[1:]))
            if name.startswith("p"):
                return self._push(OP_PARAM, a=int(name[1:]))
            raise TapeCompileError(f"unexpected symbol {name!r} in RHS")

        if isinstance(expr, sympy.Add):
            args = expr.args
            acc = self.emit(args[0])
            for term in args[1:]:
                acc = self._push(OP_ADD, a=acc, b=self.emit(term))
            return acc

        if isinstance(expr, sympy.Mul):
            args = expr.args
            acc = self.emit(args[0])
            for fac in args[1:]:
                acc = self._push(OP_MUL, a=acc, b=self.emit(fac))
            return acc

        if isinstance(expr, sympy.Pow):
            return self._emit_pow(expr)

        name = type(expr).__name__
        op = _FUNC_OPS.get(name)
        if op is not None:
            if len(expr.args) != 1:
                raise TapeCompileError(f"function {name!r} expects 1 argument")
            return self._push(op, a=self.emit(expr.args[0]))

        raise TapeCompileError(f"tape VM has no equivalent for {name!r}.")

    def _emit_pow(self, expr: Any) -> int:
        import sympy

        base, exp = expr.base, expr.exp
        base_reg = self.emit(base)
        if isinstance(exp, sympy.Integer):
            e = int(exp)
            if e == -1:
                return self._push(OP_RECIP, a=base_reg)
            return self._push(OP_POWI, a=base_reg, b=e)
        if exp == sympy.Rational(1, 2):
            return self._push(_OP_SQRT, a=base_reg)
        if exp == sympy.Rational(-1, 2):
            sqrt_reg = self._push(_OP_SQRT, a=base_reg)
            return self._push(OP_RECIP, a=sqrt_reg)
        # general exponent: a constant non-integer power, or a symbolic exponent
        return self._push(OP_POW, a=base_reg, b=self.emit(exp))


def compile_tape(system: Any) -> CompiledTape:
    """
    Lower a :class:`~tsdynamics.base.ContinuousSystem`'s RHS to a tape.

    Structural parameters are folded to constants; control parameters become
    inputs in ``control_names`` order (the layout the solve-time params vector
    must follow).
    """
    import symengine
    from jitcode import t as t_sym
    from jitcode import y

    dim = system.dim
    struct_vals = system._structural_vals()
    control_names = list(system._control_params())
    control_syms = {k: symengine.Symbol(f"p{i}") for i, k in enumerate(control_names)}

    exprs = list(type(system)._equations(y, t_sym, **{**struct_vals, **control_syms}))
    if len(exprs) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(exprs)}")

    subs = {y(i): symengine.Symbol(f"u{i}") for i in range(dim)}
    subs[t_sym] = symengine.Symbol("t")

    em = _Emitter()
    outputs = [em.emit(symengine.sympify(e).subs(subs)._sympy_()) for e in exprs]

    return CompiledTape(
        ops=np.asarray(em.ops, dtype=np.int32),
        a=np.asarray(em.a, dtype=np.int32),
        b=np.asarray(em.b, dtype=np.int32),
        imm=np.asarray(em.imm, dtype=np.float64),
        outputs=np.asarray(outputs, dtype=np.int32),
        n_state=dim,
        n_param=len(control_names),
        control_names=control_names,
    )


# ---------------------------------------------------------------------------
# Thin wrappers over the Rust kernels
# ---------------------------------------------------------------------------


def _require():
    if not available():
        raise ImportError(
            "the Rust core is not installed: pip install tsdynamics-core (or the 'rustcore' extra)"
        )
    import tsdynamics_core

    return tsdynamics_core


def _params_vec(system: Any, tape: CompiledTape) -> np.ndarray:
    return np.asarray([float(system.params[k]) for k in tape.control_names], dtype=np.float64)


def _args(tape: CompiledTape):
    return (tape.ops, tape.a, tape.b, tape.imm, tape.outputs, tape.n_state, tape.n_param)


def eval_rhs(
    system: Any, u: Any, t: float = 0.0, *, tape: CompiledTape | None = None
) -> np.ndarray:
    """Evaluate ``du/dt`` once in Rust — used to cross-check the tape."""
    core = _require()
    tape = tape or compile_tape(system)
    u = np.asarray(u, dtype=np.float64)
    return np.asarray(core.eval_rhs(*_args(tape), u, _params_vec(system, tape), float(t)))


def integrate_dense(
    system: Any,
    ic: Any,
    t_eval: Any,
    *,
    h: float | None = None,
    tape: CompiledTape | None = None,
) -> np.ndarray:
    """Fixed-step RK4 trajectory at ``t_eval`` (internal step ``h``)."""
    core = _require()
    tape = tape or compile_tape(system)
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if h is None:
        h = float(np.min(np.diff(t_eval))) if t_eval.size > 1 else 1e-2
    ic = np.asarray(ic, dtype=np.float64)
    return np.asarray(
        core.integrate_dense_py(*_args(tape), ic, _params_vec(system, tape), t_eval, float(h))
    )


def ensemble_final(
    system: Any,
    u0_batch: Any,
    t0: float,
    t1: float,
    *,
    h: float = 1e-2,
    tape: CompiledTape | None = None,
) -> np.ndarray:
    """Integrate a batch of initial conditions in parallel; return final states.

    ``u0_batch`` is ``(n, dim)``; each row is integrated from ``t0`` to ``t1``
    and its final state returned as row of the ``(n, dim)`` result.
    """
    core = _require()
    tape = tape or compile_tape(system)
    u0_batch = np.ascontiguousarray(u0_batch, dtype=np.float64)
    return np.asarray(
        core.integrate_ensemble_final_py(
            *_args(tape), u0_batch, _params_vec(system, tape), float(t0), float(t1), float(h)
        )
    )
