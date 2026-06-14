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

Solvers (select with ``method=``): fixed-step ``RK4``, adaptive Dormand-Prince
``RK45`` (default), and an L-stable linearly-implicit ``stiff`` kernel (aliases
``Rosenbrock``/``LSODA``/``BDF``) that uses the system's analytic Jacobian —
also lowered into the tape. Limitations (experimental): ODEs only; the RHS must
use functions the tape VM provides (the same set the DiffSL backend supports).
Unsupported constructs raise :class:`TapeCompileError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # Register index of each Jacobian entry, row-major dim×dim; empty unless the
    # tape was compiled with_jacobian=True (only the stiff solver needs it).
    jac_outputs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


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


def compile_tape(system: Any, *, with_jacobian: bool = False) -> CompiledTape:
    """
    Lower a :class:`~tsdynamics.families.ContinuousSystem`'s RHS to a tape.

    Structural parameters are folded to constants; control parameters become
    inputs in ``control_names`` order (the layout the solve-time params vector
    must follow).

    With ``with_jacobian=True`` the analytic Jacobian ``∂f_k/∂u_j`` is
    differentiated symbolically and emitted into the *same* tape (sharing
    common subexpressions with the RHS) — the stiff solver consumes it.
    """
    import symengine
    from jitcode import t as t_sym
    from jitcode import y

    from tsdynamics.families.continuous import _resolve_derivative_nodes

    dim = system.dim
    struct_vals = system._structural_vals()
    control_names = list(system._control_params())
    control_syms = {k: symengine.Symbol(f"p{i}") for i, k in enumerate(control_names)}

    exprs = list(type(system)._equations(y, t_sym, **{**struct_vals, **control_syms}))
    if len(exprs) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(exprs)}")

    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    subs = {y(i): u_syms[i] for i in range(dim)}
    subs[t_sym] = symengine.Symbol("t")
    rhs = [symengine.sympify(e).subs(subs) for e in exprs]

    em = _Emitter()
    outputs = [em.emit(e._sympy_()) for e in rhs]
    jac_outputs: list[int] = []
    if with_jacobian:
        # Row-major dim×dim: ∂f_k/∂u_j. _resolve_derivative_nodes rewrites the
        # unevaluated d|u|/du → sign(u) and d·sign/du → 0 a.e. (the same a.e.
        # convention the jitcode/diffsol Jacobian autogen uses), so systems with
        # abs/sign still lower cleanly.
        for e in rhs:
            for s in u_syms:
                jac_outputs.append(em.emit(_resolve_derivative_nodes(e.diff(s))._sympy_()))

    return CompiledTape(
        ops=np.asarray(em.ops, dtype=np.int32),
        a=np.asarray(em.a, dtype=np.int32),
        b=np.asarray(em.b, dtype=np.int32),
        imm=np.asarray(em.imm, dtype=np.float64),
        outputs=np.asarray(outputs, dtype=np.int32),
        n_state=dim,
        n_param=len(control_names),
        control_names=control_names,
        jac_outputs=np.asarray(jac_outputs, dtype=np.int32),
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


def _args_stiff(tape: CompiledTape):
    # Stiff kernels take jac_outputs after outputs (mirrors the Rust signature).
    return (
        tape.ops,
        tape.a,
        tape.b,
        tape.imm,
        tape.outputs,
        tape.jac_outputs,
        tape.n_state,
        tape.n_param,
    )


# Method names → kernel. RK4 = fixed step; RK45/dopri5 = adaptive Dormand-Prince
# 5(4); stiff = L-stable linearly-implicit solver with the analytic Jacobian.
# The SciPy-style stiff names (LSODA/BDF/Radau) route to the stiff kernel so
# `method="LSODA"` does the expected thing rather than silently downgrading.
_FIXED_RK4 = {"RK4", "rk4"}
_ADAPTIVE = {"RK45", "rk45", "dopri5", "DP45"}
_STIFF = {
    "stiff",
    "Rosenbrock",
    "rosenbrock",
    "ROS1",
    "LSODA",
    "lsoda",
    "BDF",
    "bdf",
    "Radau",
    "radau",
}
_METHODS = _FIXED_RK4 | _ADAPTIVE | _STIFF


def _check_method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(
            f"rustcore: unknown method {method!r}; supported: {sorted(_METHODS)} "
            "(RK4 = fixed step, RK45/dopri5 = adaptive explicit, "
            "stiff/Rosenbrock/LSODA/BDF = L-stable implicit)."
        )


def _stiff_tape(system: Any, tape: CompiledTape | None) -> CompiledTape:
    """Return a tape carrying the analytic Jacobian (recompiled if the given one lacks it)."""
    if tape is not None and tape.jac_outputs.size:
        return tape
    return compile_tape(system, with_jacobian=True)


def _as_state(x: Any, dim: int, name: str) -> np.ndarray:
    """Coerce to a contiguous float64 1-D vector of length ``dim`` (or raise)."""
    a = np.ascontiguousarray(x, dtype=np.float64).ravel()
    if a.size != dim:
        raise ValueError(f"rustcore: {name} has length {a.size}, expected dim={dim}")
    return a


def eval_rhs(
    system: Any, u: Any, t: float = 0.0, *, tape: CompiledTape | None = None
) -> np.ndarray:
    """Evaluate ``du/dt`` once in Rust — used to cross-check the tape."""
    core = _require()
    tape = tape or compile_tape(system)
    u = _as_state(u, tape.n_state, "u")
    return np.asarray(core.eval_rhs(*_args(tape), u, _params_vec(system, tape), float(t)))


def integrate_dense(
    system: Any,
    ic: Any,
    t_eval: Any,
    *,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h: float | None = None,
    tape: CompiledTape | None = None,
) -> np.ndarray:
    """Integrate a trajectory and sample it at ``t_eval``.

    ``method="RK45"`` (default) uses error-controlled adaptive Dormand-Prince
    5(4) with Hermite dense output (``rtol``/``atol`` set the tolerance);
    ``method="RK4"`` uses fixed-step RK4 with internal step ``h``;
    ``method="stiff"`` (aliases ``Rosenbrock``/``LSODA``/``BDF``) uses an
    L-stable linearly-implicit solver with the system's analytic Jacobian.

    Raises ``RuntimeError`` if the trajectory diverges or the step collapses
    before reaching the final time (matching the jitcode/diffsol backends).
    """
    _check_method(method)
    core = _require()
    tape = _stiff_tape(system, tape) if method in _STIFF else (tape or compile_tape(system))
    t_eval = np.ascontiguousarray(t_eval, dtype=np.float64)
    ic = _as_state(ic, tape.n_state, "ic")
    p = _params_vec(system, tape)
    if method in _FIXED_RK4:
        if h is None:
            h = float(np.min(np.diff(t_eval))) if t_eval.size > 1 else 1e-2
        y = np.asarray(core.integrate_dense_py(*_args(tape), ic, p, t_eval, float(h)))
    elif method in _STIFF:
        y = np.asarray(
            core.integrate_dense_stiff_py(
                *_args_stiff(tape), ic, p, t_eval, float(rtol), float(atol)
            )
        )
    else:
        y = np.asarray(
            core.integrate_dense_rk45_py(*_args(tape), ic, p, t_eval, float(rtol), float(atol))
        )
    if not np.all(np.isfinite(y)):
        raise RuntimeError(
            f"{type(system).__name__}: rustcore integration diverged or the step "
            "collapsed before reaching the final time."
        )
    return y


def ensemble_final(
    system: Any,
    u0_batch: Any,
    t0: float,
    t1: float,
    *,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h: float = 1e-2,
    tape: CompiledTape | None = None,
) -> np.ndarray:
    """Integrate a batch of initial conditions in parallel; return final states.

    ``u0_batch`` is ``(n, dim)``; each row is integrated from ``t0`` to ``t1``
    and its final state returned as a row of the ``(n, dim)`` result.
    ``method`` selects the adaptive (``"RK45"``, default), fixed-step
    (``"RK4"``), or stiff (``"stiff"``/``"LSODA"``) kernel, as in
    :func:`integrate_dense`.

    A trajectory that diverges (escapes to infinity) yields a row of ``NaN``
    rather than aborting the batch — the basin/ensemble caller classifies those
    initial conditions as escaped.
    """
    _check_method(method)
    core = _require()
    tape = _stiff_tape(system, tape) if method in _STIFF else (tape or compile_tape(system))
    u0_batch = np.ascontiguousarray(u0_batch, dtype=np.float64)
    if u0_batch.ndim != 2 or u0_batch.shape[1] != tape.n_state:
        raise ValueError(
            f"rustcore: u0_batch must be (n, {tape.n_state}); got shape {u0_batch.shape}"
        )
    p = _params_vec(system, tape)
    if method in _FIXED_RK4:
        return np.asarray(
            core.integrate_ensemble_final_py(
                *_args(tape), u0_batch, p, float(t0), float(t1), float(h)
            )
        )
    if method in _STIFF:
        return np.asarray(
            core.integrate_ensemble_final_stiff_py(
                *_args_stiff(tape), u0_batch, p, float(t0), float(t1), float(rtol), float(atol)
            )
        )
    return np.asarray(
        core.integrate_ensemble_final_rk45_py(
            *_args(tape), u0_batch, p, float(t0), float(t1), float(rtol), float(atol)
        )
    )
