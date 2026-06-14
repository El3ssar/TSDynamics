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
}
_OP_SQRT = 35

#: Opcodes whose register file slot is filled by reading a single source
#: register ``a`` (the unary functions plus ``Neg``/``Recip``).
_UNARY_OPS: frozenset[int] = frozenset({OP_NEG, OP_RECIP, *_FUNC_OPS.values()})
#: Binary opcodes (read registers ``a`` and ``b``).
_BINARY_OPS: frozenset[int] = frozenset({OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW})
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

        name = type(expr).__name__
        op = _FUNC_OPS.get(name)
        if op is not None:
            if len(expr.args) != 1:
                raise TapeCompileError(
                    f"function {name!r} expects 1 argument, got {len(expr.args)}"
                )
            return self._push(op, a=self.emit(expr.args[0]))

        raise TapeCompileError(f"the instruction tape has no equivalent for {name!r}.")

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
    )
    tape.validate()
    return tape


def _sym_name(sym: Any) -> str:
    """Return a SymEngine symbol's name (``.name`` attr, else ``str``)."""
    return getattr(sym, "name", None) or str(sym)


# ---------------------------------------------------------------------------
# ODE lowering
# ---------------------------------------------------------------------------


def lower_ode(system: Any, *, with_jacobian: bool = False) -> Tape:
    """Lower a :class:`~tsdynamics.families.ContinuousSystem` RHS to a tape.

    Structural parameters are folded to constants; control parameters become
    ``Param`` inputs in ``system._control_params()`` order (recorded on the
    tape as ``control_names``).  With ``with_jacobian=True`` the analytic
    Jacobian ``∂f_k/∂u_j`` is emitted into the same tape — the stiff/implicit
    solver family consumes it.

    Parameters
    ----------
    system : ContinuousSystem
        The system instance (its current structural-parameter values are baked
        in; control-parameter *values* are not — only their layout).
    with_jacobian : bool, default False
        Emit the analytic Jacobian alongside the RHS.

    Returns
    -------
    Tape
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
        control_names=control_names,
    )


# ---------------------------------------------------------------------------
# Map lowering (trace the numeric _step symbolically, then lower)
# ---------------------------------------------------------------------------


def lower_map(system: Any, *, with_jacobian: bool = False) -> Tape:
    """Lower a :class:`~tsdynamics.families.DiscreteMap` step to a tape.

    A map's ``_step`` is a numeric (Numba/staticjit) function, so it is *traced*
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
    step = _unwrap_static(type(system)._step)
    params = system.params.as_tuple()

    try:
        out = step(state, *params)
        exprs = [symengine.sympify(e) for e in list(out)]
    except TapeCompileError:
        raise
    except Exception as err:  # noqa: BLE001 - any tracing failure means "not lowerable"
        # A numeric ``_step`` can fail to trace on symbolic state in many ways:
        # branching (``TypeError`` on a Relational), NumPy ufuncs that don't
        # dispatch (``AttributeError``), shape/index errors (``ValueError`` /
        # ``IndexError``), etc.  Whatever the cause, it means this map cannot
        # lower to a straight-line tape — surface one clear error type.
        raise TapeCompileError(
            f"{type(system).__name__}: _step cannot be traced symbolically "
            f"({type(err).__name__}: {err}). Maps that branch on the state "
            f"(piecewise/discontinuous) or use NumPy ufuncs on the state vector "
            f"cannot lower to a straight-line tape."
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
    from jitcdde import t as t_sym
    from jitcdde import y

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
    """Whether a SymEngine node is a JiTCDDE delayed-state access (``past_y(...)``)."""
    return type(node).__name__ == "FunctionSymbol" and str(node).startswith("past_y")


def _past_y_component_and_delay(node: Any, t_sym: Any, system: Any) -> tuple[int, float]:
    """Extract ``(component, delay)`` from a ``past_y(t - τ, component, anchors)`` node.

    The delay magnitude is ``t - delay_time``; it must be a positive constant
    (state-independent) for the slot scheme to apply.
    """
    import symengine

    args = node.args
    delay_time = symengine.sympify(args[0])  # symbolic time of the access, e.g. ``t - tau``
    component = int(args[1])

    # The delay magnitude is τ = t - delay_time.  SymEngine does not fold
    # ``t - (t - τ)`` to ``τ``, so evaluate delay_time at t = 0 (→ -τ) instead.
    if delay_time.free_symbols - {symengine.sympify(t_sym)}:
        # A leftover non-``t`` symbol → state-dependent delay or unresolved param.
        raise TapeCompileError(
            f"{type(system).__name__}: delayed access {node} has a non-constant delay "
            f"(delay time {delay_time}); only constant delays lower to fixed delay slots."
        )
    delay = -float(delay_time.subs({symengine.sympify(t_sym): symengine.Integer(0)}))
    if not (delay > 0.0):
        raise TapeCompileError(
            f"{type(system).__name__}: delayed access {node} resolves to a "
            f"non-positive delay {delay}."
        )
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
    from jitcode import t as t_sym
    from jitcode import y

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
    """Peel ``staticmethod``/Numba wrappers off a method, returning the callable."""
    fn = getattr(fn, "__func__", fn)
    return getattr(fn, "py_func", fn)


# ---------------------------------------------------------------------------
# Reference evaluator — mirrors crates/tsdyn-ir/src/reference.rs exactly.
# ---------------------------------------------------------------------------


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
                r = regs[ai] ** int(b[i])
            elif op == OP_NEG:
                r = -regs[ai]
            elif op == OP_RECIP:
                r = 1.0 / regs[ai]
            else:
                r = _UNARY_FUNC[op](regs[ai])
            regs[i] = r
    return regs


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
