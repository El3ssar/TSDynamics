"""Symbolic ``_equations`` → instruction-tape IR compiler (the engine front end).

Every system family in TSDynamics defines its dynamics symbolically — an ODE's
``_equations``, a map's ``_step``, a DDE's delayed ``_equations``, an SDE's
``_drift`` / ``_diffusion``.  This module lowers any of those to a flat
**single-static-assignment instruction tape** (:class:`Tape`): a list of
primitive operations over a register file that the Rust engine evaluates with no
Python callbacks and no runtime compiler.  It is the Python half of the frozen
IR contract — the opcodes, operand layout, and well-formedness rules mirror the
``tsdyn-ir`` crate exactly (``crates/tsdyn-ir/src/{op,tape}.rs``), so a tape
produced here is consumed by the interpreter (``tsdyn-vm``) and the JIT
(``tsdyn-jit``) unchanged.

What this module is *not*: it does not integrate.  Time-stepping lives in the
Rust solver kernels (reached through :mod:`tsdynamics.engine.run`).  The only
evaluation here is a small, dependency-light **reference evaluator**
(:func:`eval_tape` / :func:`eval_tape_jac`) that mirrors
``tsdyn-ir``'s ``reference.rs`` operational semantics — the oracle the lowering
is validated against (a lowered RHS must reproduce the symbolic RHS to machine
precision) and a pure-Python fallback for callers without the compiled engine.

Family coverage
---------------
- **ODE** (:func:`lower_ode`) — RHS, optional analytic Jacobian.  Structural
  parameters fold to constants; control parameters become tape inputs in
  ``control_names`` order.
- **Map** (:func:`lower_map`) — the numeric ``_step`` is *traced* symbolically
  (evaluated on symbolic state) and lowered; the Jacobian is the symbolic
  derivative of the traced step.  Maps whose ``_step`` branches on the state
  (e.g. piecewise/discontinuous orbits) cannot be traced and raise
  :class:`TapeCompileError`.
- **DDE** (:func:`lower_dde`) — delayed accesses ``y(i, t - τ)`` become extra
  *delay-slot inputs*; the lowered tape is an ordinary RHS over
  ``dim + n_slots`` inputs, and the returned :class:`DelaySlot` list tells the
  engine which (component, delay) feeds each extra input.  The frozen IR is
  untouched — delays are data, not a new opcode.
- **SDE** (:func:`lower_sde`) — diagonal-Itô ``_drift`` + ``_diffusion`` lower to
  two ordinary tapes; Milstein additionally needs ``∂g/∂u``, emitted as the
  diffusion tape's Jacobian.

The ``abs``/``sign`` Jacobian convention is resolved a.e. (``d|u|/du = sign u``,
``d sign/du = 0``) via :func:`tsdynamics.families.continuous._resolve_derivative_nodes`,
the same convention the symbolic Jacobian autogen uses.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DelaySlot",
    "Tape",
    "TapeCompileError",
    "eval_tape",
    "eval_tape_jac",
    "eval_tape_param_jac",
    "lower_dde",
    "lower_expressions",
    "lower_map",
    "lower_ode",
    "lower_sde",
    "run_tape",
]

# ---------------------------------------------------------------------------
# Opcodes — the wire values of the frozen IR (crates/tsdyn-ir/src/op.rs).
# These integers ARE the FFI contract: the Rust ``Op::from_i32`` decodes them
# and a round-trip test on the Rust side pins every one.  Never renumber.
# ---------------------------------------------------------------------------
OP_CONST = 0
OP_STATE = 1
OP_PARAM = 2
OP_TIME = 3
OP_ADD = 10
OP_SUB = 11
OP_MUL = 12
OP_DIV = 13
OP_POW = 14  # regs[a] ** regs[b]   (runtime / non-integer exponent)
OP_POWI = 15  # regs[a] ** b          (b is the literal integer exponent)
OP_NEG = 20
OP_RECIP = 21

# ---------------------------------------------------------------------------
# Non-smooth / piecewise opcodes — wire range 50-69 (stream E-OPS).
# Additive to the frozen IR (range reserved by ROADMAP §13d); they let modular
# and piecewise maps (Circle's ``% 1``, Baker's branch) lower onto the engine.
# Comparisons yield 1.0 (true) / 0.0 (false); ``Min``/``Max`` follow ``f64::min``/
# ``max`` (NaN returns the other operand); ``Floor``/``Ceil`` are IEEE round to
# integral; ``Mod`` is the floored modulo (Python ``%`` / ``np.mod``) and ``Rem``
# the truncated remainder (Rust ``%`` / C ``fmod``).
# ---------------------------------------------------------------------------
OP_LT = 50  # regs[a] <  regs[b]  -> 1.0 / 0.0
OP_LE = 51  # regs[a] <= regs[b]
OP_GT = 52  # regs[a] >  regs[b]
OP_GE = 53  # regs[a] >= regs[b]
OP_EQ = 54  # regs[a] == regs[b]
OP_NE = 55  # regs[a] != regs[b]
OP_MIN = 56  # min(regs[a], regs[b])
OP_MAX = 57  # max(regs[a], regs[b])
OP_FLOOR = 58  # floor(regs[a])
OP_CEIL = 59  # ceil(regs[a])
OP_MOD = 60  # floored modulo: regs[a] - regs[b] * floor(regs[a] / regs[b])
OP_REM = 61  # truncated remainder: regs[a] % regs[b]  (C fmod)

#: SymEngine/SymPy function spelling → unary opcode.  Matches ``tsdyn-ir``'s
#: ``Op::name`` spellings for the elementary functions.
_FUNC_OPS: dict[str, int] = {
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
    "floor": OP_FLOOR,
    "ceiling": OP_CEIL,
}
_OP_SQRT = 35

#: SymPy relational ``rel_op`` string → comparison opcode (yields 1.0 / 0.0).
_REL_OPS: dict[str, int] = {
    "<": OP_LT,
    "<=": OP_LE,
    ">": OP_GT,
    ">=": OP_GE,
    "==": OP_EQ,
    "!=": OP_NE,
}

#: Opcodes whose register file slot is filled by reading a single source
#: register ``a`` (the unary functions plus ``Neg``/``Recip`` and the
#: round-to-integral ops).
_UNARY_OPS: frozenset[int] = frozenset({OP_NEG, OP_RECIP, *_FUNC_OPS.values()})
#: Binary opcodes (read registers ``a`` and ``b``).
_BINARY_OPS: frozenset[int] = frozenset(
    {
        OP_ADD,
        OP_SUB,
        OP_MUL,
        OP_DIV,
        OP_POW,
        *_REL_OPS.values(),
        OP_MIN,
        OP_MAX,
        OP_MOD,
        OP_REM,
    }
)
#: Leaf opcodes (read an input or an immediate; no register operands).
_LEAF_OPS: frozenset[int] = frozenset({OP_CONST, OP_STATE, OP_PARAM, OP_TIME})


class TapeCompileError(NotImplementedError):
    """A symbolic definition uses a construct the instruction tape cannot express.

    Raised when lowering hits a function with no opcode, a symbol that is not a
    declared state/parameter/time input, or a map ``_step`` that branches on its
    state (and so cannot be traced to a single straight-line expression).
    """


# ---------------------------------------------------------------------------
# The Tape
# ---------------------------------------------------------------------------


class DelaySlot(NamedTuple):
    """One delayed-state input of a lowered DDE tape.

    A DDE's ``y(component, t - delay)`` access is lowered to an *extra* state
    input appended after the ``dim`` real components.  Slot ``k`` occupies input
    index ``dim + k``; the engine fills it each step with component ``component``
    of the history evaluated ``delay`` time units in the past.

    Attributes
    ----------
    input_index : int
        The tape input index (``>= dim``) this slot occupies.
    component : int
        Which state component (``0 <= component < dim``) is delayed.
    delay : float
        The (positive) delay magnitude τ.
    """

    input_index: int
    component: int
    delay: float


@dataclass(frozen=True)
class Tape:
    """A symbolic right-hand side lowered to a flat instruction tape.

    The tape is a list of ``n_reg`` instructions held as parallel arrays
    (``ops``/``a``/``b``/``imm``) of equal length.  Instruction ``i`` writes
    register ``i`` (single static assignment) and may read only strictly earlier
    registers; common subexpressions are shared at build time.  Evaluating it is
    one linear pass over the arrays.  This is the Python mirror of the
    ``tsdyn-ir`` ``Tape``; :meth:`to_arrays` yields exactly the wire arrays its
    FFI constructor (``Tape::from_arrays``) ingests.

    Operand layout, by opcode kind (identical to the Rust contract):

    ===========  =========================================  ====================
    Field        Leaf                                       Unary / Binary / Powi
    ===========  =========================================  ====================
    ``ops[i]``   the opcode                                 the opcode
    ``a[i]``     ``State``/``Param``: input index; else —   source register ``a``
    ``b[i]``     —                                          ``Binary``: register ``b``;
                                                            ``Powi``: integer exponent
    ``imm[i]``   ``Const``: the constant; else —            —
    ===========  =========================================  ====================

    Attributes
    ----------
    ops, a, b : ndarray of int32, shape (n_reg,)
        The opcode and operand arrays.
    imm : ndarray of float64, shape (n_reg,)
        Per-instruction immediates (read only by ``Const``).
    outputs : ndarray of int32, shape (dim,)
        Register holding each derivative / next-state component ``k``.
    jac_outputs : ndarray of int32, shape (dim*dim,) or (0,)
        Registers of the row-major ``dim × dim`` Jacobian ``∂f_k/∂u_j``
        (``jac_outputs[k*dim + j]``), or empty when no Jacobian was emitted.
    n_state, n_param : int
        Declared input widths (bound the ``State``/``Param`` leaf indices).
    control_names : list[str]
        Parameter names, in the order the runtime parameter vector must follow
        (``params[control_names[i]]`` feeds ``Param`` leaf ``i``).  Empty when
        all parameters were folded to constants.
    param_jac_outputs : ndarray of int32, shape (dim*n_param,) or (0,)
        Registers of the row-major ``dim × n_param`` parameter Jacobian
        ``∂f_k/∂p_i`` (``param_jac_outputs[k*n_param + i]``), or empty when no
        parameter Jacobian was emitted (stream E-SENS).  This is a *Python-side*
        convenience (like :attr:`control_names`): it shares the register file and
        common subexpressions with the RHS, but is **not** part of the frozen FFI
        wire payload — :meth:`to_arrays` never emits it, so the Rust evaluators
        (which only know ``outputs`` / ``jac_outputs``) are untouched.  Evaluate
        it through the engine by re-exposing it as an RHS-shaped tape's outputs
        (:meth:`with_param_jac_as_outputs`), or in pure Python via
        :func:`eval_tape_param_jac`.
    """

    ops: np.ndarray
    a: np.ndarray
    b: np.ndarray
    imm: np.ndarray
    outputs: np.ndarray
    n_state: int
    n_param: int
    jac_outputs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    control_names: list[str] = field(default_factory=list)
    param_jac_outputs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    # -- derived sizes ------------------------------------------------------

    @property
    def n_reg(self) -> int:
        """Number of instructions (= number of registers)."""
        return int(self.ops.size)

    @property
    def dim(self) -> int:
        """System dimension (number of derivative / next-state outputs)."""
        return int(self.outputs.size)

    @property
    def has_jacobian(self) -> bool:
        """Whether the tape carries a Jacobian (``jac_outputs`` populated)."""
        return bool(self.jac_outputs.size)

    @property
    def has_param_jacobian(self) -> bool:
        """Whether the tape carries a parameter Jacobian (``param_jac_outputs`` populated)."""
        return bool(self.param_jac_outputs.size)

    @property
    def n_param_jac(self) -> int:
        """Number of parameter columns the parameter Jacobian spans (``0`` if none)."""
        return self.param_jac_outputs.size // self.dim if self.dim else 0

    def with_param_jac_as_outputs(self) -> Tape:
        """Return an RHS-shaped tape whose ``outputs`` are the flattened ``∂f/∂p``.

        The parameter Jacobian lives in this tape's register file but, like
        :attr:`jac_outputs`, is not part of the frozen FFI wire payload — so the
        compiled engine cannot read :attr:`param_jac_outputs` directly.  This
        helper re-exposes those same registers as an ordinary tape's
        ``outputs`` (row-major ``dim × n_param``, length ``dim·n_param``), sharing
        the original ops/operands/immediates with no recompute.  The result is a
        plain RHS tape the engine evaluates through the existing ``eval_rhs`` FFI
        — so exact ``∂f/∂p`` *round-trips through* :mod:`tsdynamics._rust` (the
        interpreter or the JIT) with no Rust-side change.

        Returns
        -------
        Tape
            A tape with ``outputs = self.param_jac_outputs`` and no Jacobians.

        Raises
        ------
        TapeCompileError
            If this tape carries no parameter Jacobian.
        """
        if not self.has_param_jacobian:
            raise TapeCompileError(
                "with_param_jac_as_outputs requires a tape compiled with a parameter "
                "Jacobian (lower_expressions(..., param_jacobian=True))"
            )
        out = Tape(
            ops=self.ops,
            a=self.a,
            b=self.b,
            imm=self.imm,
            outputs=np.ascontiguousarray(self.param_jac_outputs, dtype=np.int32),
            n_state=self.n_state,
            n_param=self.n_param,
            control_names=list(self.control_names),
        )
        # Self-defend at the FFI boundary: every tape leaving this module is
        # validated (matching lower_expressions), so a parent whose
        # param_jac_outputs skipped validation cannot hand a malformed RHS tape to
        # the engine.
        out.validate()
        return out

    # -- FFI / serialization ------------------------------------------------

    def to_arrays(self) -> tuple:
        """Return the wire arrays the Rust ``Tape::from_arrays`` ingests.

        The tuple is ``(ops, a, b, imm, outputs, jac_outputs, n_state,
        n_param)`` with the integer arrays as ``int32`` and ``imm`` as
        ``float64`` — contiguous and ready for zero-copy hand-off.
        """
        return (
            np.ascontiguousarray(self.ops, dtype=np.int32),
            np.ascontiguousarray(self.a, dtype=np.int32),
            np.ascontiguousarray(self.b, dtype=np.int32),
            np.ascontiguousarray(self.imm, dtype=np.float64),
            np.ascontiguousarray(self.outputs, dtype=np.int32),
            np.ascontiguousarray(self.jac_outputs, dtype=np.int32),
            int(self.n_state),
            int(self.n_param),
        )

    # -- validation (mirrors tsdyn-ir Tape::validate) -----------------------

    def validate(self) -> None:
        """Check every structural invariant of the frozen IR contract.

        Mirrors ``tsdyn-ir``'s ``Tape::validate`` so a malformed tape is caught
        on the Python side rather than at (or worse, after) the FFI boundary.

        Raises
        ------
        TapeCompileError
            On a length mismatch, an unknown opcode, a forward/self register
            reference, a state/param index out of range, an output index out of
            range, or a Jacobian whose length is not ``dim * dim``.
        """
        n = self.n_reg
        if not (self.a.size == n and self.b.size == n and self.imm.size == n):
            raise TapeCompileError(
                f"tape arrays have mismatched lengths: ops={n}, a={self.a.size}, "
                f"b={self.b.size}, imm={self.imm.size}"
            )
        ops = self.ops
        a = self.a
        b = self.b
        for i in range(n):
            op = int(ops[i])
            if op in _LEAF_OPS:
                if op == OP_STATE and not (0 <= a[i] < self.n_state):
                    raise TapeCompileError(
                        f"instruction {i}: state index {a[i]} out of range "
                        f"for n_state={self.n_state}"
                    )
                if op == OP_PARAM and not (0 <= a[i] < self.n_param):
                    raise TapeCompileError(
                        f"instruction {i}: param index {a[i]} out of range "
                        f"for n_param={self.n_param}"
                    )
            elif op in _UNARY_OPS:
                _check_reg(i, int(a[i]))
            elif op in _BINARY_OPS:
                _check_reg(i, int(a[i]))
                _check_reg(i, int(b[i]))
            elif op == OP_POWI:
                _check_reg(i, int(a[i]))  # b is the literal exponent, not a register
            else:
                raise TapeCompileError(f"unknown opcode {op} at instruction {i}")

        for k, reg in enumerate(self.outputs):
            if not (0 <= reg < n):
                raise TapeCompileError(f"outputs[{k}] = {reg} is out of range for n_reg={n}")

        dim = self.dim
        if self.jac_outputs.size and self.jac_outputs.size != dim * dim:
            raise TapeCompileError(
                f"jac_outputs length {self.jac_outputs.size} is not dim*dim = {dim}*{dim}"
            )
        for k, reg in enumerate(self.jac_outputs):
            if not (0 <= reg < n):
                raise TapeCompileError(f"jac_outputs[{k}] = {reg} is out of range for n_reg={n}")

        if self.param_jac_outputs.size and self.param_jac_outputs.size != dim * self.n_param:
            raise TapeCompileError(
                f"param_jac_outputs length {self.param_jac_outputs.size} is not "
                f"dim*n_param = {dim}*{self.n_param}"
            )
        for k, reg in enumerate(self.param_jac_outputs):
            if not (0 <= reg < n):
                raise TapeCompileError(
                    f"param_jac_outputs[{k}] = {reg} is out of range for n_reg={n}"
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tape):
            return NotImplemented
        return (
            self.n_state == other.n_state
            and self.n_param == other.n_param
            and self.control_names == other.control_names
            and np.array_equal(self.ops, other.ops)
            and np.array_equal(self.a, other.a)
            and np.array_equal(self.b, other.b)
            and np.array_equal(self.imm, other.imm)
            and np.array_equal(self.outputs, other.outputs)
            and np.array_equal(self.jac_outputs, other.jac_outputs)
            and np.array_equal(self.param_jac_outputs, other.param_jac_outputs)
        )

    __hash__ = None  # type: ignore[assignment]  # mutable arrays → unhashable


def _check_reg(at: int, reg: int) -> None:
    """Require ``reg`` to be a strictly earlier instruction than ``at`` (SSA)."""
    if not (0 <= reg < at):
        raise TapeCompileError(
            f"instruction {at} reads register {reg}, which is not a strictly earlier register"
        )


# ---------------------------------------------------------------------------
# Emitter: symbolic DAG → SSA instructions (with common-subexpression sharing)
# ---------------------------------------------------------------------------


class _Emitter:
    """Lower SymPy expression DAGs to SSA instructions, sharing subexpressions.

    Leaves (state/param/time symbols) are resolved through ``leaf_for_name``: a
    map from a symbol's name to a ``(op, index)`` pair.  All other nodes —
    ``Add``/``Mul``/``Pow`` and the elementary functions — are emitted
    structurally.  The ``_cache`` keyed on the expression object makes shared
    subexpressions (the whole point of an SSA tape) emit once.
    """

    def __init__(self, leaf_for_name: dict[str, tuple[int, int]]) -> None:
        self._leaf = leaf_for_name
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
        """Emit ``expr``, returning the register holding its value (CSE-cached)."""
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
            leaf = self._leaf.get(expr.name)
            if leaf is None:
                raise TapeCompileError(
                    f"unexpected symbol {expr.name!r} in symbolic definition — "
                    f"not a declared state/parameter/time input"
                )
            op, idx = leaf
            if op == OP_TIME:
                return self._push(OP_TIME)
            return self._push(op, a=idx)

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

        # Piecewise / selection (maps with ``np.where`` or a branch).  Lowered
        # to a comparison-masked arithmetic blend; see ``_emit_piecewise``.
        if isinstance(expr, sympy.Piecewise):
            return self._emit_piecewise(expr)

        # A relational (``a < b`` …) used as a value yields 1.0 / 0.0 via a
        # comparison opcode.  These reach the emitter only as Piecewise
        # conditions (SymPy forbids ``Relational`` inside ``Add``/``Mul``).
        if isinstance(expr, sympy.core.relational.Relational):
            return self._emit_relational(expr)

        # Boolean connectives over 0/1-valued conditions: And → product,
        # Or → a + b - a*b, Not → 1 - a (each input is already 0/1).
        if isinstance(expr, sympy.logic.boolalg.BooleanFunction):
            return self._emit_boolean(expr)

        name = type(expr).__name__

        # n-ary Min / Max fold left into binary OP_MIN / OP_MAX.
        if name == "Min" or name == "Max":
            op = OP_MIN if name == "Min" else OP_MAX
            args = expr.args
            acc = self.emit(args[0])
            for term in args[1:]:
                acc = self._push(op, a=acc, b=self.emit(term))
            return acc

        # Floored modulo ``Mod(a, b)`` (SymPy's ``%``; the maps' bare ``%`` is
        # canonicalised to ``a - floor(a/b)*b`` instead, but a literal Mod node
        # lowers directly).
        if name == "Mod":
            if len(expr.args) != 2:
                raise TapeCompileError(f"Mod expects 2 arguments, got {len(expr.args)}")
            return self._push(OP_MOD, a=self.emit(expr.args[0]), b=self.emit(expr.args[1]))

        op = _FUNC_OPS.get(name)
        if op is not None:
            if len(expr.args) != 1:
                raise TapeCompileError(
                    f"function {name!r} expects 1 argument, got {len(expr.args)}"
                )
            return self._push(op, a=self.emit(expr.args[0]))

        raise TapeCompileError(f"the instruction tape has no equivalent for {name!r}.")

    def _emit_relational(self, expr: Any) -> int:
        """Emit a comparison opcode (1.0 if the relation holds, else 0.0)."""
        op = _REL_OPS.get(expr.rel_op)
        if op is None:
            raise TapeCompileError(f"the instruction tape has no equivalent for {expr.rel_op!r}.")
        lhs, rhs = expr.args
        return self._push(op, a=self.emit(lhs), b=self.emit(rhs))

    def _emit_boolean(self, expr: Any) -> int:
        """Lower And/Or/Not over 0/1-valued conditions to arithmetic."""
        import sympy

        if isinstance(expr, sympy.And):
            acc = self.emit(expr.args[0])
            for term in expr.args[1:]:
                acc = self._push(OP_MUL, a=acc, b=self.emit(term))  # c1 * c2 * …
            return acc
        if isinstance(expr, sympy.Or):
            # a + b - a*b, folded so the running accumulator stays in {0, 1}.
            acc = self.emit(expr.args[0])
            for term in expr.args[1:]:
                t = self.emit(term)
                s = self._push(OP_ADD, a=acc, b=t)
                p = self._push(OP_MUL, a=acc, b=t)
                acc = self._push(OP_SUB, a=s, b=p)
            return acc
        if isinstance(expr, sympy.Not):
            one = self._push(OP_CONST, imm=1.0)
            return self._push(OP_SUB, a=one, b=self.emit(expr.args[0]))
        raise TapeCompileError(
            f"the instruction tape has no equivalent for boolean {type(expr).__name__!r}."
        )

    def _emit_piecewise(self, expr: Any) -> int:
        """Lower ``Piecewise((e0, c0), …, (en, True))`` to a masked blend.

        Each condition ``ck`` emits to a 1.0/0.0 mask; the value is built from
        the last (default) branch backwards as ``mk*ek + (1 - mk)*acc``.  This
        evaluates **both** branches, so a branch that is singular on the
        unselected region (``0*inf = nan``) would poison the result — fine for
        the finite-branch piecewise maps this targets, a caveat for general use.
        """
        import sympy

        pairs = [(p.args[0], p.args[1]) for p in expr.args]
        if pairs[-1][1] != sympy.true:
            raise TapeCompileError(
                "Piecewise must end with a default (True) branch to lower to a tape "
                f"(got condition {pairs[-1][1]!r}); the engine cannot represent a "
                "partial/undefined region."
            )
        acc = self.emit(pairs[-1][0])
        for value, cond in reversed(pairs[:-1]):
            mask = self.emit(cond)
            val = self.emit(value)
            one = self._push(OP_CONST, imm=1.0)
            inv = self._push(OP_SUB, a=one, b=mask)  # 1 - mask
            sel = self._push(OP_MUL, a=mask, b=val)  # mask * value
            other = self._push(OP_MUL, a=inv, b=acc)  # (1 - mask) * acc
            acc = self._push(OP_ADD, a=sel, b=other)
        return acc

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
        # General exponent: a non-integer constant power or a symbolic exponent.
        return self._push(OP_POW, a=base_reg, b=self.emit(exp))


# ---------------------------------------------------------------------------
# Core lowering: a list of symbolic expressions → a Tape
# ---------------------------------------------------------------------------


def lower_expressions(
    exprs: Sequence[Any],
    state_syms: Sequence[Any],
    *,
    param_syms: Sequence[Any] = (),
    time_sym: Any = None,
    jacobian: bool = False,
    param_jacobian: bool = False,
    control_names: Sequence[str] | None = None,
) -> Tape:
    """Lower a list of symbolic expressions to a validated :class:`Tape`.

    This is the family-agnostic core every ``lower_*`` helper funnels through.
    ``exprs`` are SymEngine expressions over ``state_syms`` (the state inputs,
    in order), ``param_syms`` (the parameter inputs, in order), and optionally
    ``time_sym``.  Each input symbol must be a distinct SymEngine ``Symbol``.

    Parameters
    ----------
    exprs : sequence of SymEngine expressions
        The ``dim`` right-hand-side (or next-state) components.
    state_syms : sequence of SymEngine Symbol
        The state inputs ``u_0 … u_{n_state-1}`` in order; ``State`` leaf ``i``
        reads ``state_syms[i]``.
    param_syms : sequence of SymEngine Symbol, optional
        The parameter inputs ``p_0 … p_{n_param-1}`` in order.
    time_sym : SymEngine Symbol, optional
        The independent-variable symbol, lowered to the ``Time`` leaf.
    jacobian : bool, default False
        If true, also emit the row-major ``dim × dim`` Jacobian
        ``∂exprs_k/∂state_syms_j`` into the same tape (sharing subexpressions
        with the RHS).  ``abs``/``sign`` derivatives are resolved a.e.
    param_jacobian : bool, default False
        If true, also emit the row-major ``dim × n_param`` parameter Jacobian
        ``∂exprs_k/∂param_syms_i`` into the tape's :attr:`Tape.param_jac_outputs`
        (sharing subexpressions with the RHS), the exact-symbolic ``∂f/∂p`` the
        continuation and gradient-fitting tracks build on (stream E-SENS).
        ``abs``/``sign`` derivatives are resolved a.e., exactly as for the state
        Jacobian.  Empty ``param_syms`` yields an empty parameter Jacobian.
    control_names : sequence of str, optional
        Parameter names in ``param_syms`` order; attached to the tape so the
        runtime parameter vector can be built by name.

    Returns
    -------
    Tape
        Already validated against the frozen IR invariants.

    Raises
    ------
    TapeCompileError
        If an expression uses an unsupported function or a free symbol that is
        not one of the declared inputs.
    """
    import symengine

    from tsdynamics.families.continuous import _resolve_derivative_nodes

    state_syms = list(state_syms)
    param_syms = list(param_syms)

    leaf_for_name: dict[str, tuple[int, int]] = {}
    for i, s in enumerate(state_syms):
        leaf_for_name[_sym_name(s)] = (OP_STATE, i)
    for i, s in enumerate(param_syms):
        leaf_for_name[_sym_name(s)] = (OP_PARAM, i)
    if time_sym is not None:
        leaf_for_name[_sym_name(time_sym)] = (OP_TIME, 0)

    rhs = [symengine.sympify(e) for e in exprs]

    em = _Emitter(leaf_for_name)
    outputs = [em.emit(e._sympy_()) for e in rhs]

    jac_outputs: list[int] = []
    if jacobian:
        # Row-major dim×dim: ∂f_k/∂u_j with abs/sign derivatives resolved a.e.
        for e in rhs:
            for s in state_syms:
                jac_outputs.append(em.emit(_resolve_derivative_nodes(e.diff(s))._sympy_()))

    param_jac_outputs: list[int] = []
    if param_jacobian:
        # Row-major dim×n_param: ∂f_k/∂p_i with abs/sign derivatives resolved a.e.
        # (the exact-symbolic engine moat the FD-based competition cannot match;
        # the one-loop analogue of the state-Jacobian emit above, over param_syms).
        for e in rhs:
            for s in param_syms:
                param_jac_outputs.append(em.emit(_resolve_derivative_nodes(e.diff(s))._sympy_()))

    tape = Tape(
        ops=np.asarray(em.ops, dtype=np.int32),
        a=np.asarray(em.a, dtype=np.int32),
        b=np.asarray(em.b, dtype=np.int32),
        imm=np.asarray(em.imm, dtype=np.float64),
        outputs=np.asarray(outputs, dtype=np.int32),
        n_state=len(state_syms),
        n_param=len(param_syms),
        jac_outputs=np.asarray(jac_outputs, dtype=np.int32),
        control_names=list(control_names) if control_names is not None else [],
        param_jac_outputs=np.asarray(param_jac_outputs, dtype=np.int32),
    )
    tape.validate()
    return tape


def _sym_name(sym: Any) -> str:
    """Return a SymEngine symbol's name (``.name`` attr, else ``str``)."""
    return getattr(sym, "name", None) or str(sym)


# ---------------------------------------------------------------------------
# ODE lowering
# ---------------------------------------------------------------------------


def lower_ode(
    system: Any, *, with_jacobian: bool = False, with_param_jacobian: bool = False
) -> Tape:
    """Lower a :class:`~tsdynamics.families.ContinuousSystem` RHS to a tape.

    Structural parameters are folded to constants; control parameters become
    ``Param`` inputs in ``system._control_params()`` order (recorded on the
    tape as ``control_names``).  With ``with_jacobian=True`` the analytic
    Jacobian ``∂f_k/∂u_j`` is emitted into the same tape — the stiff/implicit
    solver family consumes it.  With ``with_param_jacobian=True`` the analytic
    parameter Jacobian ``∂f_k/∂p_i`` is emitted into
    :attr:`Tape.param_jac_outputs` (stream E-SENS).

    Parameters
    ----------
    system : ContinuousSystem
        The system instance (its current structural-parameter values are baked
        in; control-parameter *values* are not — only their layout).
    with_jacobian : bool, default False
        Emit the analytic state Jacobian ``∂f_k/∂u_j`` alongside the RHS.
    with_param_jacobian : bool, default False
        Emit the analytic parameter Jacobian ``∂f_k/∂p_i`` (over the control
        parameters) into the tape's :attr:`Tape.param_jac_outputs`.

    Returns
    -------
    Tape
    """
    import symengine

    from tsdynamics.engine.symbols import state_time_symbols

    y, t_sym = state_time_symbols()

    dim = system.dim
    struct_vals = system._structural_vals()
    control_names = list(system._control_params())
    control_syms = {k: symengine.Symbol(f"p{i}") for i, k in enumerate(control_names)}

    exprs = list(type(system)._equations(y, t_sym, **{**struct_vals, **control_syms}))
    if len(exprs) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(exprs)}")

    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    t_canon = symengine.Symbol("t")
    subs = {y(i): u_syms[i] for i in range(dim)}
    subs[t_sym] = t_canon
    rhs = [symengine.sympify(e).subs(subs) for e in exprs]

    return lower_expressions(
        rhs,
        u_syms,
        param_syms=[control_syms[k] for k in control_names],
        time_sym=t_canon,
        jacobian=with_jacobian,
        param_jacobian=with_param_jacobian,
        control_names=control_names,
    )


# ---------------------------------------------------------------------------
# Map lowering (trace the numeric _step symbolically, then lower)
# ---------------------------------------------------------------------------


class _SymbolicNumpy:
    """A drop-in ``numpy`` whose array math returns SymEngine expressions.

    A map's ``_step`` is written for numeric (NumPy) evaluation with ``np.sin``,
    ``np.where``, ``%`` and friends.  NumPy's object-dtype ufunc loop calls a
    method *named after the ufunc* on each element (``elem.sin()``), which a raw
    SymEngine symbol does not have — so ``np.sin(symbolic_state)`` raises.  When
    *tracing* a step we rebind its ``np`` global to this shim: the elementary
    functions, ``floor``/``ceil``, ``abs``/``sign``, ``min``/``max``, ``mod`` and
    ``where`` (→ ``Piecewise``) lower to SymEngine, while every other attribute
    (``array``, ``zeros``, constants other than ``pi``/``e``, …) falls through to
    the real :mod:`numpy`.  Arithmetic and comparisons need no shim — SymEngine
    already overloads ``+``/``*``/``%``/``<`` on its expressions.
    """

    def __init__(self) -> None:
        import symengine as se

        def unary(fn: Any) -> Any:
            def f(x: Any) -> Any:
                arr = np.asarray(x, dtype=object)
                if arr.ndim == 0:
                    return fn(arr.item())
                out = np.empty(arr.shape, dtype=object)
                flat = out.ravel()
                for i, v in enumerate(arr.ravel()):
                    flat[i] = fn(v)
                return out

            return f

        def binary(fn: Any) -> Any:
            def f(x: Any, y: Any) -> Any:
                ax = np.asarray(x, dtype=object)
                ay = np.asarray(y, dtype=object)
                if ax.ndim == 0 and ay.ndim == 0:
                    return fn(ax.item(), ay.item())
                bx, by = np.broadcast_arrays(ax, ay)
                out = np.empty(bx.shape, dtype=object)
                flat = out.ravel()
                for i, (a, b) in enumerate(zip(bx.ravel(), by.ravel(), strict=True)):
                    flat[i] = fn(a, b)
                return out

            return f

        def where(cond: Any, a: Any, b: Any) -> Any:
            def pw(c: Any, x: Any, y: Any) -> Any:
                return se.Piecewise((x, c), (y, True))

            ac = np.asarray(cond, dtype=object)
            aa = np.asarray(a, dtype=object)
            ab = np.asarray(b, dtype=object)
            if ac.ndim == 0 and aa.ndim == 0 and ab.ndim == 0:
                return pw(ac.item(), aa.item(), ab.item())
            bc, ba, bb = np.broadcast_arrays(ac, aa, ab)
            out = np.empty(bc.shape, dtype=object)
            flat = out.ravel()
            for i, (c, x, y) in enumerate(zip(bc.ravel(), ba.ravel(), bb.ravel(), strict=True)):
                flat[i] = pw(c, x, y)
            return out

        self.pi = math.pi
        self.e = math.e
        for name, fn in (
            ("sin", se.sin),
            ("cos", se.cos),
            ("tan", se.tan),
            ("exp", se.exp),
            ("log", se.log),
            ("sqrt", se.sqrt),
            ("arcsin", se.asin),
            ("arccos", se.acos),
            ("arctan", se.atan),
            ("sinh", se.sinh),
            ("cosh", se.cosh),
            ("tanh", se.tanh),
            ("arcsinh", se.asinh),
            ("arccosh", se.acosh),
            ("arctanh", se.atanh),
            ("floor", se.floor),
            ("ceil", se.ceiling),
            ("abs", se.Abs),
            ("absolute", se.Abs),
            ("sign", se.sign),
        ):
            setattr(self, name, unary(fn))
        self.minimum = binary(se.Min)
        self.maximum = binary(se.Max)
        self.fmin = self.minimum
        self.fmax = self.maximum
        self.mod = binary(lambda a, b: a % b)
        self.remainder = self.mod
        self.power = binary(lambda a, b: a**b)
        self.where = where

    def __getattr__(self, name: str) -> Any:
        # Anything not overridden above (array, zeros, dtype, …) is the real
        # numpy.  ``__getattr__`` runs only on a miss, so overrides win.
        return getattr(np, name)


def _trace_step(step_fn: Any) -> Any:
    """Return a copy of ``step_fn`` whose ``np`` global is the symbolic shim.

    Rebinding the global on a fresh :class:`types.FunctionType` (sharing the
    original code object) keeps the real module untouched and the operation
    thread-safe — no monkeypatching of shared state.
    """
    g = dict(step_fn.__globals__)
    g["np"] = _SymbolicNumpy()
    return types.FunctionType(
        step_fn.__code__,
        g,
        step_fn.__name__,
        step_fn.__defaults__,
        step_fn.__closure__,
    )


def lower_map(system: Any, *, with_jacobian: bool = False) -> Tape:
    """Lower a :class:`~tsdynamics.families.DiscreteMap` step to a tape.

    A map's ``_step`` is a numeric (``staticjit``) function, so it is *traced*
    symbolically — evaluated on a symbolic state vector — to recover the
    straight-line next-state expression, which is then lowered.  With
    ``with_jacobian=True`` the map Jacobian ``∂step_k/∂u_j`` is the symbolic
    derivative of the traced step (the single source of truth; it agrees with a
    hand-written ``_jacobian`` where one exists).

    Parameters arrive positionally to ``_step`` in declaration order and are
    folded to constants — maps have no runtime control-parameter inputs here, so
    the returned tape has ``n_param = 0`` and empty ``control_names``.

    Parameters
    ----------
    system : DiscreteMap
        The map instance.
    with_jacobian : bool, default False
        Emit the symbolic step Jacobian alongside the next-state expression.

    Returns
    -------
    Tape

    Raises
    ------
    TapeCompileError
        If ``_step`` cannot be traced symbolically — typically because it
        branches on the state (piecewise/discontinuous maps) or calls a NumPy
        ufunc that does not dispatch onto symbolic operands.
    """
    import symengine

    from tsdynamics.families.discrete import _unwrap_static

    dim = system.dim
    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    state = np.array(u_syms, dtype=object)
    step = _trace_step(_unwrap_static(type(system)._step))
    params = system.params.as_tuple()

    try:
        out = step(state, *params)
        exprs = [symengine.sympify(e) for e in list(out)]
    except TapeCompileError:
        raise
    except Exception as err:  # noqa: BLE001 - any tracing failure means "not lowerable"
        # A numeric ``_step`` can still fail to trace on symbolic state: a Python
        # ``if`` on the state (``TypeError`` on a Relational's truth value — use
        # ``np.where`` for a branchless step instead), a NumPy routine the
        # symbolic shim does not cover, shape/index errors, etc.  Whatever the
        # cause, it means this map cannot lower to a straight-line tape — surface
        # one clear error type.
        raise TapeCompileError(
            f"{type(system).__name__}: _step cannot be traced symbolically "
            f"({type(err).__name__}: {err}). Maps that branch on the state with a "
            f"Python ``if`` (rewrite the branch with ``np.where``) or call a NumPy "
            f"routine the tracer does not model cannot lower to a straight-line tape."
        ) from err

    if len(exprs) != dim:
        raise TapeCompileError(f"_step traced to {len(exprs)} components, expected dim={dim}")

    return lower_expressions(exprs, u_syms, jacobian=with_jacobian)


# ---------------------------------------------------------------------------
# DDE lowering (delayed accesses → extra delay-slot inputs)
# ---------------------------------------------------------------------------


def lower_dde(system: Any) -> tuple[Tape, list[DelaySlot]]:
    """Lower a :class:`~tsdynamics.families.DelaySystem` RHS to a tape + delay slots.

    Delayed accesses ``y(component, t - τ)`` cannot be a leaf of the frozen IR
    (there is no delay opcode), so each distinct ``(component, τ)`` pair is
    lowered to an **extra state input** appended after the ``dim`` real
    components.  The returned tape is therefore an ordinary RHS over
    ``dim + n_slots`` inputs; the :class:`DelaySlot` list records, for each extra
    input, which component is delayed and by how much, so the DDE engine
    (history buffer + dense interpolation) can fill those inputs each step.

    Parameters are folded to constants (matching the v2 DDE compile path), so
    the tape has ``n_param = 0``.  Only constant delays are supported — a
    state-dependent delay (``τ`` depending on ``y``) raises.

    Parameters
    ----------
    system : DelaySystem
        The DDE instance.

    Returns
    -------
    (Tape, list[DelaySlot])
        The lowered RHS over the extended input space, and the ordered delay
        slots (slot ``k`` is input index ``dim + k``).

    Raises
    ------
    TapeCompileError
        If a delayed access has a state-dependent delay, or the RHS uses an
        unsupported construct.
    """
    import symengine

    from tsdynamics.engine.symbols import state_time_symbols

    y, t_sym = state_time_symbols()

    dim = system.dim
    exprs = list(type(system)._equations(y, t_sym, **system.params.as_dict()))
    if len(exprs) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(exprs)}")

    t_canon = symengine.Symbol("t")
    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]

    # First pass: collect distinct (component, delay) delayed accesses and build
    # a substitution mapping each delayed term to a fresh extra-input symbol.
    slots: list[DelaySlot] = []
    slot_key_to_sym: dict[tuple[int, float], Any] = {}
    delayed_subs: dict[Any, Any] = {}

    def scan(node: Any) -> None:
        node = symengine.sympify(node)
        if _is_past_y(node):
            comp, delay = _past_y_component_and_delay(node, t_sym, system)
            if not (0 <= comp < dim):
                raise TapeCompileError(
                    f"{type(system).__name__}: delayed access {node} references component "
                    f"{comp}, outside the state range 0..{dim - 1}."
                )
            if delay == 0.0:
                # ``y(i, t)`` is the current state — substitute the real input.
                delayed_subs[node] = u_syms[comp]
                return
            key = (comp, delay)
            if key not in slot_key_to_sym:
                k = len(slots)
                sym = symengine.Symbol(f"u{dim + k}")
                slot_key_to_sym[key] = sym
                slots.append(DelaySlot(input_index=dim + k, component=comp, delay=delay))
            delayed_subs[node] = slot_key_to_sym[key]
            return
        for arg in node.args:
            scan(arg)

    for e in exprs:
        scan(e)

    # Second pass: substitute delayed terms → extra inputs and current state →
    # real inputs, then lower over the extended input space (dim + n_slots).
    subs = {y(i): u_syms[i] for i in range(dim)}
    subs[t_sym] = t_canon
    subs.update(delayed_subs)
    extra_syms = [slot_key_to_sym[(s.component, s.delay)] for s in slots]
    rhs = [symengine.sympify(e).subs(subs) for e in exprs]

    tape = lower_expressions(rhs, [*u_syms, *extra_syms], time_sym=t_canon)
    return tape, slots


def _is_past_y(node: Any) -> bool:
    """Whether a SymEngine node is a delayed-state access ``y(component, t - τ)``.

    The engine-native state symbol is ``symengine.Function("y")``: a *current*
    access ``y(i)`` is a one-argument ``FunctionSymbol`` named ``y`` and a
    *delayed* access ``y(i, t - τ)`` is the two-argument form — so delayed
    accesses are distinguished from current ones by arity.
    """
    return (
        type(node).__name__ == "FunctionSymbol"
        and str(node).startswith("y(")
        and len(node.args) == 2
    )


def _past_y_component_and_delay(node: Any, t_sym: Any, system: Any) -> tuple[int, float]:
    """Extract ``(component, delay)`` from a ``y(component, t - τ)`` delayed access.

    The delay magnitude is ``t - delay_time``; it must be a positive constant
    (state-independent) for the slot scheme to apply.
    """
    import symengine

    args = node.args
    component = int(args[0])
    delay_time = symengine.sympify(args[1])  # symbolic time of the access, e.g. ``t - tau``

    # The delay magnitude is τ = t - delay_time.  SymEngine does not fold
    # ``t - (t - τ)`` to ``τ``, so evaluate delay_time at t = 0 (→ -τ) instead.
    if delay_time.free_symbols - {symengine.sympify(t_sym)}:
        # A leftover non-``t`` symbol → state-dependent delay or unresolved param.
        raise TapeCompileError(
            f"{type(system).__name__}: delayed access {node} has a non-constant delay "
            f"(delay time {delay_time}); only constant delays lower to fixed delay slots."
        )
    delay = -float(delay_time.subs({symengine.sympify(t_sym): symengine.Integer(0)}))
    if delay < 0.0:
        # A negative delay is a *future* access (``y(i, t + τ)``) — not causal.
        raise TapeCompileError(
            f"{type(system).__name__}: delayed access {node} resolves to a "
            f"negative (future) delay {delay}."
        )
    # delay == 0.0 is an explicit current-time access ``y(i, t)`` (== ``y(i)``);
    # the caller maps it to the current state rather than a delay slot.
    return component, delay


# ---------------------------------------------------------------------------
# SDE lowering (diagonal-Itô: drift + per-component diffusion)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweredSDE:
    """A diagonal-Itô SDE lowered to a drift tape and a diffusion tape.

    Both tapes share the same input layout (state + control parameters).  The
    drift tape's ``outputs`` are ``f_k(u, t)``; the diffusion tape's ``outputs``
    are the per-component noise coefficients ``g_k(u, t)``.  For the Milstein
    scheme the diffusion tape additionally carries ``∂g_k/∂u_j`` in its
    ``jac_outputs`` (requested with ``with_diffusion_jacobian=True``).

    Attributes
    ----------
    drift : Tape
        The deterministic part ``f`` (one output per component).
    diffusion : Tape
        The diagonal noise coefficients ``g`` (one output per component);
        carries ``∂g/∂u`` when lowered for Milstein.
    """

    drift: Tape
    diffusion: Tape


def lower_sde(system: Any, *, with_diffusion_jacobian: bool = False) -> LoweredSDE:
    """Lower a diagonal-Itô SDE (``_drift`` + ``_diffusion``) to two tapes.

    Follows the resolved noise contract (ROADMAP §11): ``_drift(y, t, **params)``
    is the deterministic part (exactly like an ODE's ``_equations``) and
    ``_diffusion(y, t, **params)`` returns one noise coefficient per state
    component, each multiplying an independent Wiener increment (Itô).  Both
    lower to ordinary tapes over the same (state, control-parameter) input
    layout.  Milstein (order 1.0) needs ``∂g/∂u``; pass
    ``with_diffusion_jacobian=True`` to emit it into the diffusion tape.

    Parameters
    ----------
    system : object
        Anything exposing ``_drift`` and ``_diffusion`` staticmethods with the
        ``(y, t, **params)`` signature, plus ``dim`` / ``params`` (the SDE
        family base class, stream E-SDE).
    with_diffusion_jacobian : bool, default False
        Emit ``∂g_k/∂u_j`` as the diffusion tape's Jacobian (for Milstein).

    Returns
    -------
    LoweredSDE

    Raises
    ------
    TapeCompileError
        If ``_drift`` / ``_diffusion`` is missing or returns the wrong length.
    """
    import symengine

    from tsdynamics.engine.symbols import state_time_symbols

    y, t_sym = state_time_symbols()

    drift_fn = getattr(type(system), "_drift", None)
    diff_fn = getattr(type(system), "_diffusion", None)
    if drift_fn is None or diff_fn is None:
        raise TapeCompileError(
            f"{type(system).__name__}: SDE lowering needs both _drift and _diffusion "
            f"staticmethods (diagonal-Itô contract)."
        )

    dim = system.dim
    struct_vals = system._structural_vals() if hasattr(system, "_structural_vals") else {}
    control_names = (
        list(system._control_params())
        if hasattr(system, "_control_params")
        else list(system.params)
    )
    control_syms = {k: symengine.Symbol(f"p{i}") for i, k in enumerate(control_names)}
    call_kwargs = {**struct_vals, **control_syms}

    drift_exprs = list(_unwrap_static_callable(drift_fn)(y, t_sym, **call_kwargs))
    diff_exprs = list(_unwrap_static_callable(diff_fn)(y, t_sym, **call_kwargs))
    if len(drift_exprs) != dim:
        raise TapeCompileError(f"_drift must return {dim} expressions, got {len(drift_exprs)}")
    if len(diff_exprs) != dim:
        raise TapeCompileError(f"_diffusion must return {dim} expressions, got {len(diff_exprs)}")

    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    t_canon = symengine.Symbol("t")
    subs = {y(i): u_syms[i] for i in range(dim)}
    subs[t_sym] = t_canon
    param_syms = [control_syms[k] for k in control_names]

    drift = lower_expressions(
        [symengine.sympify(e).subs(subs) for e in drift_exprs],
        u_syms,
        param_syms=param_syms,
        time_sym=t_canon,
        control_names=control_names,
    )
    diffusion = lower_expressions(
        [symengine.sympify(e).subs(subs) for e in diff_exprs],
        u_syms,
        param_syms=param_syms,
        time_sym=t_canon,
        jacobian=with_diffusion_jacobian,
        control_names=control_names,
    )
    return LoweredSDE(drift=drift, diffusion=diffusion)


def _unwrap_static_callable(fn: Any) -> Any:
    """Peel the ``staticmethod`` wrapper off a method, returning the raw callable."""
    return getattr(fn, "__func__", fn)


# ---------------------------------------------------------------------------
# Reference evaluator — mirrors crates/tsdyn-ir/src/reference.rs.
#
# Every opcode replicates the Rust reference's IEEE-754 semantics. The integer
# power ``OP_POWI`` uses the same square-and-multiply reduction as Rust's
# ``f64::powi`` (see ``_powi`` below) rather than NumPy's ``pow`` (an
# exp·log reduction that differs by a few ULP); cross-language *bit-exact*
# agreement on every op is asserted by the I-XVAL migration gate once the
# compiled wheel is built, not here.
# ---------------------------------------------------------------------------


def _powi(base: np.float64, exp: int) -> np.float64:
    """Integer power by square-and-multiply, matching Rust's ``f64::powi``.

    NumPy's ``base ** int`` promotes the exponent to a float and takes an
    ``exp·log`` path, which drifts from the Rust evaluators' repeated-multiply
    reduction by up to a few ULP.  Replicating square-and-multiply here keeps the
    pure-Python reference a faithful oracle on the ``OP_POWI`` path.
    """
    n = int(exp)
    b = base
    if n < 0:
        b = np.float64(1.0) / b
        n = -n
    result = np.float64(1.0)
    while n > 0:
        if n & 1:
            result = result * b
        n >>= 1
        if n > 0:
            b = b * b
    return result


def run_tape(tape: Tape, u: Any, p: Any = (), t: float = 0.0) -> np.ndarray:
    """Run the instruction tape, returning the full register file.

    A direct, unoptimised port of the IR's reference semantics
    (``tsdyn-ir``'s ``reference.rs``) — the executable specification of what each
    opcode means.  One linear pass over the arrays; no allocation beyond the
    register vector.  Used to validate lowering against the symbolic RHS and as
    a pure-Python fallback when the compiled engine is unavailable.

    Parameters
    ----------
    tape : Tape
    u : array-like
        State inputs, length ``tape.n_state``.
    p : array-like, optional
        Parameter inputs, length ``tape.n_param``.
    t : float, optional
        The independent variable.

    Returns
    -------
    ndarray, shape (n_reg,)
        The value written to each register.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    ops = tape.ops
    a = tape.a
    b = tape.b
    imm = tape.imm
    regs = np.empty(tape.n_reg, dtype=np.float64)
    # Match the Rust evaluator's IEEE-754 semantics: a singular state (1/0,
    # sqrt of a negative, a negative base to a fractional power, …) yields
    # inf/NaN silently — never an exception — so this stays a faithful oracle
    # even when probed outside the physical domain (escaped/diverged states,
    # solver overshoot).  Without this, NumPy's scalar warnings would escalate
    # to errors under a strict ``filterwarnings`` policy.
    with np.errstate(all="ignore"):
        for i in range(tape.n_reg):
            op = int(ops[i])
            ai = int(a[i])
            if op == OP_CONST:
                r = imm[i]
            elif op == OP_STATE:
                r = u[ai]
            elif op == OP_PARAM:
                r = p[ai]
            elif op == OP_TIME:
                r = t
            elif op == OP_ADD:
                r = regs[ai] + regs[int(b[i])]
            elif op == OP_SUB:
                r = regs[ai] - regs[int(b[i])]
            elif op == OP_MUL:
                r = regs[ai] * regs[int(b[i])]
            elif op == OP_DIV:
                r = regs[ai] / regs[int(b[i])]
            elif op == OP_POW:
                # ``regs`` is float64, so ``**`` dispatches to NumPy power: a
                # negative base to a fractional power yields NaN (matching Rust
                # ``powf``), never a Python ``complex``.  This relies on both
                # operands being NumPy scalars — never let a Python ``float`` in.
                r = regs[ai] ** regs[int(b[i])]
            elif op == OP_POWI:
                # Square-and-multiply (matches Rust f64::powi), not NumPy pow.
                r = _powi(regs[ai], int(b[i]))
            elif op == OP_NEG:
                r = -regs[ai]
            elif op == OP_RECIP:
                r = 1.0 / regs[ai]
            elif op in _BINARY_FUNC:
                r = _BINARY_FUNC[op](regs[ai], regs[int(b[i])])
            else:
                r = _UNARY_FUNC[op](regs[ai])
            regs[i] = r
    return regs


def _fmin(x: np.float64, y: np.float64) -> np.float64:
    """``f64::min`` semantics: a NaN operand returns the other; else the smaller."""
    if x != x:
        return y
    if y != y:
        return x
    return x if x < y else y


def _fmax(x: np.float64, y: np.float64) -> np.float64:
    """``f64::max`` semantics: a NaN operand returns the other; else the larger."""
    if x != x:
        return y
    if y != y:
        return x
    return x if x > y else y


# Unary opcode → NumPy implementation (sign matches the a.e. convention:
# sign(0) = 0, exactly as tsdyn-ir's Op::Sign).
_UNARY_FUNC: dict[int, Any] = {
    30: np.sin,
    31: np.cos,
    32: np.tan,
    33: np.exp,
    34: np.log,
    35: np.sqrt,
    36: np.abs,
    37: lambda x: float(np.sign(x)),
    38: np.sinh,
    39: np.cosh,
    40: np.tanh,
    41: np.arcsin,
    42: np.arccos,
    43: np.arctan,
    44: np.arcsinh,
    45: np.arccosh,
    46: np.arctanh,
    OP_FLOOR: np.floor,
    OP_CEIL: np.ceil,
}

# Binary opcode → implementation.  Comparisons yield 1.0 / 0.0; Min/Max follow
# ``f64::min``/``max`` (NaN returns the other operand); Mod is the floored
# modulo and Rem the truncated remainder (C ``fmod``) — matching the Rust
# evaluators' IEEE-754 *values* op-for-op.
#
# Caveat (sub-ULP): on the measure-zero edges these can disagree with the Rust
# arms in the SIGN of a zero or NaN result — ``min``/``max`` of a ``±0.0`` tie
# and ``mod``/``rem`` by a zero divisor (Python yields a ``-NaN``, Rust a
# ``+NaN``).  The values are equal; only the sign bit differs, and no built-in
# map reaches these inputs (``%`` lowers via ``floor``, not ``OP_MOD``).  True
# bit-for-bit agreement across the FFI boundary is asserted by the I-XVAL gate
# against the compiled engine, not promised by this pure-Python oracle.
_BINARY_FUNC: dict[int, Any] = {
    OP_LT: lambda x, y: 1.0 if x < y else 0.0,
    OP_LE: lambda x, y: 1.0 if x <= y else 0.0,
    OP_GT: lambda x, y: 1.0 if x > y else 0.0,
    OP_GE: lambda x, y: 1.0 if x >= y else 0.0,
    OP_EQ: lambda x, y: 1.0 if x == y else 0.0,
    OP_NE: lambda x, y: 1.0 if x != y else 0.0,
    OP_MIN: _fmin,
    OP_MAX: _fmax,
    OP_MOD: lambda x, y: x - y * np.floor(x / y),
    OP_REM: np.fmod,  # C fmod (truncated); NaN on a zero divisor under errstate
}


def eval_tape(tape: Tape, u: Any, p: Any = (), t: float = 0.0) -> np.ndarray:
    """Evaluate ``du/dt`` (or next state) at ``(u, p, t)`` via the reference evaluator.

    Returns
    -------
    ndarray, shape (dim,)
    """
    regs = run_tape(tape, u, p, t)
    return regs[tape.outputs]


def eval_tape_jac(tape: Tape, u: Any, p: Any = (), t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``(du/dt, Jacobian)`` in one tape pass via the reference evaluator.

    Requires a tape carrying ``jac_outputs`` (see :attr:`Tape.has_jacobian`).

    Returns
    -------
    (ndarray, ndarray)
        The derivative ``(dim,)`` and the row-major ``(dim, dim)`` Jacobian.

    Raises
    ------
    ValueError
        If the tape carries no Jacobian.
    """
    if not tape.has_jacobian:
        raise ValueError("eval_tape_jac requires a tape compiled with a Jacobian")
    regs = run_tape(tape, u, p, t)
    dim = tape.dim
    deriv = regs[tape.outputs]
    jac = regs[tape.jac_outputs].reshape(dim, dim)
    return deriv, jac


def eval_tape_param_jac(
    tape: Tape, u: Any, p: Any = (), t: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``(du/dt, ∂f/∂p)`` in one tape pass via the reference evaluator.

    Requires a tape carrying ``param_jac_outputs`` (see
    :attr:`Tape.has_param_jacobian`), produced by
    ``lower_expressions(..., param_jacobian=True)`` (stream E-SENS).  The
    parameter Jacobian is the exact symbolic ``∂f_k/∂p_i`` — never a finite
    difference.

    Returns
    -------
    (ndarray, ndarray)
        The derivative ``(dim,)`` and the row-major ``(dim, n_param)`` parameter
        Jacobian ``∂f_k/∂p_i``.

    Raises
    ------
    ValueError
        If the tape carries no parameter Jacobian.
    """
    if not tape.has_param_jacobian:
        raise ValueError("eval_tape_param_jac requires a tape compiled with a parameter Jacobian")
    regs = run_tape(tape, u, p, t)
    deriv = regs[tape.outputs]
    param_jac = regs[tape.param_jac_outputs].reshape(tape.dim, tape.n_param)
    return deriv, param_jac
