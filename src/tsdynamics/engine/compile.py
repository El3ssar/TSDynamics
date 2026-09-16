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

Two emitters, one tape
----------------------
The symbolic frontend is SymEngine, so :class:`_SymEngineEmitter` reads the
SymEngine tree **directly** — no ``._sympy_()`` round-trip, hence no ~260 ms
SymPy import on the first lowering in a process (Lorenz: 300 ms → 0.4 ms; a cold
``integrate()``: 208 ms → 1.6 ms; Gray–Scott: 2.5 s → 0.63 s).  Where SymEngine's
canonical form differs from SymPy's it is reproduced exactly, because the tape is
a **byte** contract (``tests/_equation_reference_golden.txt``); where reproducing
it would mean re-implementing ``signsimp`` / ``default_sort_key``, the node kind
is *gated* and the whole tape falls back to :class:`_SymPyEmitter`, which is
unchanged.  ``TSDYNAMICS_NO_NATIVE_LOWERING=1`` forces the fallback everywhere —
the bypass that proves the two emitters agree.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import types
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DelaySlot",
    "LoweredSDE",
    "Tape",
    "TapeCompileError",
    "clear_tape_cache",
    "eval_tape",
    "eval_tape_jac",
    "lower_dde",
    "lower_dde_cached",
    "lower_expressions",
    "lower_map",
    "lower_map_cached",
    "map_jacobian_fn",
    "lower_map_sweep",
    "lower_map_sweep_cached",
    "lower_ode",
    "lower_ode_cached",
    "lower_sde",
    "lower_sde_cached",
    "run_tape",
    "tape_cache_stats",
    "tape_jacobian_is_smooth",
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
# Additive to the frozen IR (range reserved in the frozen IR opcode table); they let modular
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

#: Opcodes whose **derivative** is non-smooth (a kink or a step): ``Abs``/``sign``
#: (resolved a.e. as ``sign u`` / ``0``, so the lowered IR derivative is ``0`` *at*
#: a kink), the round-to-integral / modular ops (``floor``/``ceil``/``mod``/``rem``,
#: zero derivative a.e. with a jump at the integers), the comparisons (a piecewise
#: 0/1 selection with zero derivative), and ``min``/``max`` (a kink at the
#: crossover).  A map whose lowered Jacobian tape contains any of these is
#: *piecewise*: the lowered IR derivative is a.e.-correct but collapses to the
#: a.e. value (``0``) exactly on the kink/step, where a hand-written one-sided
#: ``_jacobian`` instead returns the meaningful slope.  See
#: :func:`tape_jacobian_is_smooth`.
_NONSMOOTH_OPS: frozenset[int] = frozenset(
    {
        _FUNC_OPS["Abs"],
        _FUNC_OPS["sign"],
        OP_FLOOR,
        OP_CEIL,
        OP_MOD,
        OP_REM,
        OP_MIN,
        OP_MAX,
        *_REL_OPS.values(),
    }
)


def tape_jacobian_is_smooth(tape: Any) -> bool:
    """Whether a lowered tape's Jacobian is smooth (no kink/step opcodes).

    Returns ``False`` when the tape uses any non-smooth opcode
    (:data:`_NONSMOOTH_OPS`: ``abs``/``sign``/``floor``/``ceil``/``mod``/``rem``/
    comparisons/``min``/``max``) — i.e. the map is *piecewise* and its lowered IR
    Jacobian is a.e.-correct but collapses to the a.e. value at a kink/step (where
    a hand-written one-sided ``_jacobian`` returns the meaningful slope).  The map
    Lyapunov **kernel** propagates the lowered Jacobian, so a piecewise map whose
    orbit lands on a kink (e.g. the Tent map at full height, whose dyadic orbit
    hits ``x = 0.5`` exactly) collapses the QR growth to ``log 0`` and poisons the
    spectrum; such maps must use the pure-Python QR loop (which reads the
    one-sided hand-written ``_jacobian``) instead.  A smooth map (Hénon, Ikeda,
    logistic, …) returns ``True`` and keeps the fast kernel.

    Parameters
    ----------
    tape : Tape
        A lowered map tape (typically ``with_jacobian=True``).

    Returns
    -------
    bool
        ``True`` if the tape contains no non-smooth opcode, else ``False``.
    """
    ops = np.asarray(tape.ops, dtype=np.int64)
    return not bool(np.isin(ops, list(_NONSMOOTH_OPS)).any())


class TapeCompileError(NotImplementedError):
    """A symbolic definition uses a construct the instruction tape cannot express.

    Raised when lowering hits a function with no opcode, a symbol that is not a
    declared state/parameter/time input, or a map ``_step`` that branches on its
    state (and so cannot be traced to a single straight-line expression).
    """


# ---------------------------------------------------------------------------
# Kernel-tracing diagnostics
#
# The #1 documented pitfall is calling ``math.*`` / ``numpy.*`` / ``scipy.*``
# inside a symbolic kernel: those routines try to convert a SymEngine symbol to
# a float, and the raw failure ("RuntimeError: Symbol cannot be evaluated.", or
# a ufunc ``TypeError``) says nothing about what to do instead.  Every family's
# kernel call is wrapped in :func:`_trace_kernel`, which locates the offending
# source line and raises one actionable :class:`TapeCompileError`.
# ---------------------------------------------------------------------------

#: Modules whose functions are *numeric* and therefore cannot be traced.  The
#: aliases are the spellings that actually appear in user code.
_NUMERIC_MODULES = ("np", "numpy", "math", "scipy", "sp", "cmath")

#: The same list for a map ``_step``, whose ``np`` global *is* traced through the
#: :class:`_SymbolicNumpy` shim — so a ``np.`` call is not the culprit there.
_MAP_NUMERIC_MODULES = ("math", "scipy", "sp", "cmath")


def _numeric_call_re(modules: Sequence[str]) -> re.Pattern[str]:
    """Regex matching ``<module>.<attr>(`` for any alias in ``modules``."""
    return re.compile(r"\b(" + "|".join(modules) + r")\.([A-Za-z_][A-Za-z_0-9.]*)\s*\(")


#: The SymEngine functions the tape can express — quoted verbatim in the hint so
#: a user has the replacement in front of them.
_SYMENGINE_HINT = (
    "symengine.sin/cos/tan/asin/acos/atan/exp/log/sqrt/sinh/cosh/tanh/"
    "Abs/sign/Min/Max (and the plain operators + - * / ** for arithmetic and powers)"
)

#: A control parameter reaches the kernel as a *symbol*, so using it where Python
#: needs a concrete ``int`` (``range(N)``, an index, ``%``) fails with one of
#: these.  That is the second documented pitfall — a missing
#: ``_structural_params`` — and its fix has nothing to do with numeric routines,
#: so it must never be answered with the SymEngine-replacement advice.
_INT_COERCION_RE = re.compile(
    r"cannot be interpreted as an integer|"
    r"list indices must be integers|"
    r"tuple indices must be integers|"
    r"only integer scalar arrays"
)

#: The kernels whose state argument is the **callable accessor** ``y`` (``y(0)``),
#: not an array.  A map's ``_step`` receives a real state vector, so ``x[0]``
#: there is correct and must never draw the ``y(0)`` advice.
_ACCESSOR_KERNELS: frozenset[str] = frozenset({"_equations", "_drift", "_diffusion"})

#: The raw Python error for ``y[0]`` when ``y`` is the accessor — the single most
#: likely first-timer mistake in the whole library, and the one the generic
#: "don't use numpy" paragraph answered with advice about numeric routines and
#: structural parameters, neither of which is the problem.
_SUBSCRIPT_RE = re.compile(r"object is not subscriptable")

#: The accessor's own Python type, as it appears in that error.  ``y`` is a
#: ``symengine.Function``, which Python reports as ``'function' object`` — so this
#: is the one message that names the state itself, and it is what lets the hint be
#: given when the source line could not be recovered.
_ACCESSOR_TYPE_RE = re.compile(r"'function' object is not subscriptable")

#: The *other* way a first-timer reaches for the state as a container: ``x, y, z =
#: u``.  It is the spelling every ``DiscreteMap._step`` in this library uses (a
#: map's state genuinely IS a vector), so a reader who wrote a map first will
#: write it in an ODE — and Python answers "cannot unpack non-iterable function
#: object", which names neither the kernel, nor the state, nor the fix.
_UNPACK_RE = re.compile(r"cannot unpack non-iterable|is not iterable")

#: The unpack error that names the accessor's own type, so the hint can be given
#: even when the source line is unavailable (a class defined in a REPL / stdin,
#: where ``inspect`` has no file to read).
_UNPACK_ACCESSOR_TYPE_RE = re.compile(r"non-iterable function object|'function' object is not")

#: A tuple-unpack of the state on the offending source line: ``x, y, z = u``.
_UNPACK_LINE_RE = re.compile(r"^\s*([A-Za-z_][\w\s,]*?)\s*=\s*([A-Za-z_]\w*)\s*$")


def _kernel_state_name(system: Any, kernel: str) -> str:
    """Name of the state parameter of ``kernel`` (the accessor), defaulting to ``y``."""
    import inspect

    try:
        raw = inspect.getattr_static(type(system), kernel)
        params = list(inspect.signature(getattr(raw, "__func__", raw)).parameters)
    except Exception:  # noqa: BLE001 - the hint is best-effort, never the failure
        return "y"
    params = [p for p in params if p != "self"]
    return params[0] if params else "y"


def _unpacked_accessor_rewrite(line: str, state: str) -> str | None:
    """Rewrite ``x, y, z = u`` as ``x, y, z = u(0), u(1), u(2)``, or ``None``.

    Only fires when the right-hand side is exactly the state name: anything else
    that failed to unpack is a different mistake, and offering this rewrite for
    it would send the reader to the one name on the line that is already right.
    """
    match = _UNPACK_LINE_RE.match(line)
    if match is None or match.group(2) != state:
        return None
    targets = [t.strip() for t in match.group(1).split(",") if t.strip()]
    if len(targets) < 2:
        return None
    indent = line[: len(line) - len(line.lstrip())]
    calls = ", ".join(f"{state}({i})" for i in range(len(targets)))
    return f"{indent}{', '.join(targets)} = {calls}"


def _subscripted_accessor_hint(
    system: Any, kernel: str, line: str | None, message: str
) -> list[str] | None:
    """Explain ``y[0]`` → ``y(0)``, echoing the offending line rewritten.

    The state reaches a symbolic kernel as a **function**: ``y(i)`` is component
    ``i`` and ``y(i, t - tau)`` is a delayed access, which an array simply cannot
    express.  So ``y[0]`` is a ``TypeError`` from Python's subscript machinery,
    whose text (``'function' object is not subscriptable``) never mentions ``y``,
    the kernel, or the one-character fix.

    Returns ``None`` when the subscript was **not** of the state, so the caller
    falls through to the general advice.  "Something in this kernel was
    subscripted" is not the same claim as "the state was subscripted": a kernel
    writing ``a[0]`` for a scalar *parameter* raises the same shape of
    ``TypeError`` (``'Symbol' object is not subscriptable``) and was being told,
    wrongly and with no line to show for it, that ``y`` is an accessor — sending
    the reader to inspect the one name on the line that is already correct.  The
    claim is therefore made only when the source line really does subscript the
    state, or when the raised error names the accessor's own type (``function``).
    """
    state = _kernel_state_name(system, kernel)
    rewritten: str | None = None
    unpacked = False
    if line is not None:
        fixed = re.sub(rf"\b{re.escape(state)}\s*\[([^][]*)\]", rf"{state}(\1)", line)
        rewritten = fixed if fixed != line else None
    if rewritten is None and _UNPACK_RE.search(message) is not None:
        if line is not None:
            rewritten = _unpacked_accessor_rewrite(line, state)
        unpacked = rewritten is not None or _UNPACK_ACCESSOR_TYPE_RE.search(message) is not None
        if not unpacked:
            return None
    if rewritten is None and not unpacked and _ACCESSOR_TYPE_RE.search(message) is None:
        return None
    verb = f"`{state}[0]` is a subscript of a function"
    if unpacked:
        verb = f"unpacking it (`x, y, z = {state}`) is an iteration over a function"
    parts = [
        f"`{state}` is the state *accessor*, not an array: inside `{kernel}` the state is "
        f"read by CALLING it — `{state}(0)`, `{state}(1)`, … — so {verb} "
        "and cannot work."
    ]
    if rewritten is not None:
        parts.append(f"  you wrote: {line}")
        parts.append(f"  write:     {rewritten}")
    parts.append(
        f"(A call is what makes a delayed access expressible too: `{state}(0, t - tau)` in a "
        "DelaySystem.)"
    )
    return parts


def _kernel_owner(system: Any) -> type:
    """Return the class that owns a kernel, given an instance *or* a class.

    Every trace-time diagnostic names the system class.  Lowering normally has an
    instance in hand, but :func:`map_jacobian_fn` derives a Jacobian from the
    class alone, so both spellings must reach the same name.
    """
    return system if isinstance(system, type) else type(system)


def _missing_staticmethod_hint(system: Any, kernel: str) -> list[str] | None:
    """Return the "you forgot ``@staticmethod``" advice, or ``None`` if that is not it.

    A symbolic kernel is called **off the class** (``type(system)._equations(y, t,
    …)``), because there is no instance state in the math.  Declared as an
    ordinary method, its ``self`` therefore swallows the state accessor, ``y``
    swallows ``t``, and Python reports a missing argument named ``t`` — a message
    that points at the wrong parameter entirely.  The mistake is decidable from
    the class, not from the message, so it is checked structurally.

    ``system`` may be an instance *or* the class itself: the autogenerated map
    Jacobian (:func:`map_jacobian_fn`) traces a kernel with no instance in hand.
    """
    import inspect

    try:
        raw = inspect.getattr_static(_kernel_owner(system), kernel)
    except AttributeError:  # pragma: no cover - the kernel exists by construction
        return None
    if not inspect.isfunction(raw):  # a staticmethod object → correctly declared
        return None
    try:
        params = list(inspect.signature(raw).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not params or params[0] != "self":
        return None
    name = _kernel_owner(system).__name__
    rest = ", ".join(params[1:]) or "y, t"
    return [
        f"`{kernel}` is declared as an ordinary method (its first parameter is `self`), but "
        f"the engine calls it off the class — there is no instance state in the math — so "
        f"`self` receives the state and every later argument is shifted by one (which is why "
        f"the error names a parameter you did pass).",
        "Add the decorator:",
        "    @staticmethod",
        f"    def {kernel}({rest}):        # on {name}, no `self`",
    ]


def _kernel_source_frame(err: BaseException) -> tuple[str, int, str] | None:
    """Locate the deepest *user* frame of ``err`` as ``(file, lineno, source)``.

    Walks the traceback from the raise site back towards the caller and returns
    the deepest frame that is not inside this module or a third-party numeric
    library (SymEngine / NumPy internals), i.e. the line of the kernel body (or
    of a helper it called) that actually made the untraceable call.  Returns
    ``None`` when no such frame can be identified or its source is unavailable
    (e.g. a kernel defined in an interactive session).
    """
    import linecache

    frames: list[tuple[str, int]] = []
    tb = err.__traceback__
    while tb is not None:
        name = tb.tb_frame.f_code.co_filename
        frames.append((name, tb.tb_lineno))
        tb = tb.tb_next

    skip = (
        os.path.join("symengine", ""),
        os.path.join("numpy", ""),
        os.path.join("scipy", ""),
    )
    this_file = os.path.abspath(__file__)
    for filename, lineno in reversed(frames):
        if os.path.abspath(filename) == this_file:
            continue
        if any(part in filename for part in skip) or filename.startswith("<"):
            continue
        line = linecache.getline(filename, lineno).strip()
        if line:
            return filename, lineno, line
    return None


def _trace_kernel(
    system: Any,
    kernel: str,
    call: Callable[[], Any],
    *,
    numeric_modules: Sequence[str] = _NUMERIC_MODULES,
    hint: str = "",
) -> Any:
    """Run a symbolic kernel call, converting any failure into a guided error.

    Parameters
    ----------
    system : SystemBase
        The system being lowered (named in the message).
    kernel : str
        The kernel attribute name, e.g. ``"_equations"`` / ``"_step"``.
    call : callable
        A zero-argument thunk that invokes the kernel on symbolic arguments.
    numeric_modules : sequence of str, optional
        Module aliases whose functions are numeric (and so untraceable) *in this
        kernel*.  A map's ``_step`` traces ``np`` through a symbolic shim, so
        ``np``/``numpy`` are excluded there.
    hint : str, optional
        An extra family-specific paragraph appended to the message.

    Returns
    -------
    Any
        Whatever ``call()`` returned.

    Raises
    ------
    TapeCompileError
        If ``call()`` raised anything at all.  The message names the system, the
        kernel, the offending source line, the numeric call detected on it (when
        one is), and the SymEngine replacements.
    """
    try:
        return call()
    except TapeCompileError:
        raise
    except Exception as err:  # noqa: BLE001 - any kernel failure means "not lowerable"
        name = _kernel_owner(system).__name__
        parts = [f"{name}: `{kernel}` could not be lowered to an engine tape."]
        frame = _kernel_source_frame(err)
        offender = None
        if frame is not None:
            filename, lineno, line = frame
            parts.append(f"  {filename}:{lineno}: {line}")
            match = _numeric_call_re(numeric_modules).search(line)
            if match is not None:
                offender = f"{match.group(1)}.{match.group(2)}"
        parts.append(f"  raised {type(err).__name__}: {err}")
        listed = ", ".join(f"{m}.*" for m in numeric_modules)
        forgot_static = _missing_staticmethod_hint(system, kernel)
        # Two spellings of the same misconception — "the state is a container":
        # ``y[0]`` (a subscript) and ``x, y, z = u`` (an unpack).  Both are
        # answered by the same sentence, so both route to the same hint.
        reached_for_a_container = (
            _SUBSCRIPT_RE.search(str(err)) is not None or _UNPACK_RE.search(str(err)) is not None
        )
        subscripted = (
            _subscripted_accessor_hint(
                system, kernel, frame[2] if frame is not None else None, str(err)
            )
            if kernel in _ACCESSOR_KERNELS and reached_for_a_container
            else None
        )
        if forgot_static is not None:
            # Structural, so decided from the class rather than from the message:
            # whatever the kernel then failed on, the declaration is the cause.
            parts.extend(forgot_static)
        elif subscripted is not None:
            parts.extend(subscripted)
        elif offender is not None:
            parts.append(
                f"`{offender}(...)` is a *numeric* function: it converts its argument to a "
                f"float, but `{kernel}` is called with SymEngine symbols, not numbers."
            )
            parts.append(f"Use the SymEngine equivalents instead: {_SYMENGINE_HINT}.")
        elif _INT_COERCION_RE.search(str(err)) is not None:
            # A *different* documented pitfall: a control parameter arrives as a
            # symbol, so `range(N)` / `x[N]` / `i % N` cannot work.  Diagnosing
            # this as "don't use numeric routines" would send the user the wrong
            # way entirely, so name the real fix and the candidate parameters.
            parts.append(
                f"A parameter was used where Python needs a concrete integer. Control "
                f"parameters reach `{kernel}` as SymEngine *symbols*, so `range(N)`, an "
                f"index or `%` on one cannot work."
            )
            parts.append(_structural_params_hint(system))
        else:
            parts.append(
                f"`{kernel}` is traced *symbolically* — it is called with SymEngine symbols, "
                f"not numbers — so it cannot use numeric routines ({listed}), branch on the "
                f"state with a Python `if`, or use a parameter as a concrete int."
            )
            parts.append(
                f"If it calls a numeric routine, use the SymEngine equivalents: {_SYMENGINE_HINT}."
            )
            parts.append(_structural_params_hint(system))
        if hint:
            parts.append(hint)
        raise TapeCompileError("\n".join(parts)) from err


def _structural_params_hint(system: Any) -> str:
    """Return the ``_structural_params`` advice, naming the integer-valued candidates.

    A parameter that fixes the *shape* of the equations (a lattice size, a mode
    count) must be declared structural so lowering sees its value rather than a
    symbol.  The candidates are the control parameters currently holding an
    ``int``, which is what a shape parameter looks like.
    """
    declared = frozenset(getattr(type(system), "_structural_params", frozenset()))
    try:
        control = system._control_params()
    except Exception:  # noqa: BLE001 - the hint is best-effort, never the failure
        control = {
            k: v for k, v in dict(getattr(system, "params", {})).items() if k not in declared
        }
    candidates = [k for k, v in control.items() if isinstance(v, int) and not isinstance(v, bool)]
    name = type(system).__name__
    if candidates:
        listed = ", ".join(repr(c) for c in sorted(candidates))
        return (
            f"If one of them fixes the *shape* of the equations, declare it structural so "
            f"lowering bakes in its value: on {name}, "
            f"`_structural_params = frozenset({{{listed}}})` (integer-valued candidates)."
        )
    return (
        f"If one of them fixes the *shape* of the equations, declare it structural so "
        f"lowering bakes in its value: `_structural_params = frozenset({{'N'}})` on {name}."
    )


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

    def to_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        r"""Return the wire arrays the Rust ``Tape::from_arrays`` ingests.

        The tuple is ``(ops, a, b, imm, outputs, jac_outputs, n_state,
        n_param)`` with the integer arrays as ``int32`` and ``imm`` as
        ``float64`` — each contiguous and of the right dtype, ready for the FFI.

        Notes
        -----
        ``np.ascontiguousarray`` is a no-op (returns the *same* object) when the
        field is already C-contiguous and of the requested dtype, so these are
        **views onto the tape's own arrays, not guaranteed-fresh copies**.  That
        is safe because the consumer never writes through them: the FFI
        constructor ``Tape::from_arrays`` copies the data into owned Rust
        ``Vec``\\ s, and the reference evaluator only reads.  Callers must not
        mutate the returned arrays in place (it would corrupt the cached tape).
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
    """Lower expression DAGs to SSA instructions, sharing subexpressions.

    Leaves (state/param/time symbols) are resolved through ``leaf_for_name``: a
    map from a symbol's name to a ``(op, index)`` pair.  All other nodes —
    ``Add``/``Mul``/``Pow`` and the elementary functions — are emitted
    structurally.  The ``_cache`` keyed on the expression object makes shared
    subexpressions (the whole point of an SSA tape) emit once.

    This base class owns the SSA machinery (register file, CSE cache) and the
    node-kind-agnostic blends; the per-node dispatch lives in the two concrete
    subclasses, :class:`_SymEngineEmitter` (the native path) and
    :class:`_SymPyEmitter` (the fallback / oracle).
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

    #: The emitter's own spelling of the literal ``1`` (set by each subclass).
    #: Routed through :meth:`emit` so the CSE cache shares one ``Const 1``.
    _const_one: Any = None

    def _one(self) -> int:
        """Return the register holding the constant ``1.0`` (CSE-shared).

        The piecewise / boolean blends need a literal one (``1 - mask``).  Routing
        it through :meth:`emit` of the backend's ``Integer(1)`` — rather than a raw
        ``_push(OP_CONST, imm=1.0)`` — lets the CSE cache dedup it across every
        branch, so a tape with several piecewise selections carries a single
        ``Const 1`` register instead of one per branch.
        """
        return self.emit(self._const_one)

    def _emit(self, expr: Any) -> int:
        raise NotImplementedError  # pragma: no cover - abstract

    # -- node-kind-agnostic folds (shared by both emitters) ------------------

    def _fold_binary(self, op: int, args: Sequence[Any]) -> int:
        """Left-fold an n-ary commutative node (``Add``/``Mul``/``Min``/``Max``)."""
        acc = self.emit(args[0])
        for term in args[1:]:
            acc = self._push(op, a=acc, b=self.emit(term))
        return acc

    def _fold_and(self, args: Sequence[Any]) -> int:
        """``c1 ∧ c2 ∧ …`` over 0/1-valued conditions → a product."""
        acc = self.emit(args[0])
        for term in args[1:]:
            acc = self._push(OP_MUL, a=acc, b=self.emit(term))
        return acc

    def _fold_or(self, args: Sequence[Any]) -> int:
        """``c1 ∨ c2 ∨ …`` → ``a + b - a*b``, folded so the accumulator stays 0/1."""
        acc = self.emit(args[0])
        for term in args[1:]:
            t = self.emit(term)
            s = self._push(OP_ADD, a=acc, b=t)
            p = self._push(OP_MUL, a=acc, b=t)
            acc = self._push(OP_SUB, a=s, b=p)
        return acc

    def _emit_not(self, arg: Any) -> int:
        """``¬c`` → ``1 - c`` (the input is already 0/1-valued)."""
        return self._push(OP_SUB, a=self._one(), b=self.emit(arg))

    def _blend_piecewise(self, pairs: Sequence[tuple[Any, Any]]) -> int:
        """Lower ``[(e0, c0), …, (en, True)]`` to a comparison-masked blend.

        Each condition ``ck`` emits to a 1.0/0.0 mask; the value is built from
        the last (default) branch backwards as ``mk*ek + (1 - mk)*acc``.

        .. important::
           Because the blend is *arithmetic*, **every branch is evaluated on
           every input**, then masked.  The IR has no control flow that could
           skip the unselected arm.  So each branch expression must be **finite
           on the whole domain**, not merely on the region its condition
           selects: a branch that is singular off its own region (``±inf`` or
           ``NaN`` there) poisons the result through ``0 * inf = NaN`` /
           ``0 * inf + finite = NaN``.  This holds for the finite-branch
           piecewise maps this targets (Baker's modular branches), but a
           ``Piecewise((1/u, u != 0), (0, True))``-style guard against a
           singularity would *not* lower correctly — rewrite it so both arms are
           finite (e.g. blend on a regularised expression).
        """
        acc = self.emit(pairs[-1][0])
        for value, cond in reversed(pairs[:-1]):
            mask = self.emit(cond)
            val = self.emit(value)
            inv = self._push(OP_SUB, a=self._one(), b=mask)  # 1 - mask
            sel = self._push(OP_MUL, a=mask, b=val)  # mask * value
            other = self._push(OP_MUL, a=inv, b=acc)  # (1 - mask) * acc
            acc = self._push(OP_ADD, a=sel, b=other)
        return acc

    def _emit_leaf_symbol(self, name: str) -> int:
        """Emit a declared state / parameter / time input by symbol name."""
        leaf = self._leaf.get(name)
        if leaf is None:
            raise TapeCompileError(
                f"unexpected symbol {name!r} in symbolic definition — "
                f"not a declared state/parameter/time input"
            )
        op, idx = leaf
        if op == OP_TIME:
            return self._push(OP_TIME)
        return self._push(op, a=idx)

    def _emit_pow_parts(self, base_reg: int, exp_is_int: bool, exp_int: int, exp: Any) -> int:
        """Emit ``base ** exp`` given the already-emitted base register.

        ``exp_is_int`` selects the integer-exponent fast paths (``Recip`` /
        ``PowI``); otherwise ``exp`` is inspected for ``±1/2`` (``Sqrt``) and
        falls back to the general two-register ``Pow``.
        """
        if exp_is_int:
            if exp_int == -1:
                return self._push(OP_RECIP, a=base_reg)
            return self._push(OP_POWI, a=base_reg, b=exp_int)
        half = self._exponent_half(exp)
        if half == 1:
            return self._push(_OP_SQRT, a=base_reg)
        if half == -1:
            sqrt_reg = self._push(_OP_SQRT, a=base_reg)
            return self._push(OP_RECIP, a=sqrt_reg)
        # General exponent: a non-integer constant power or a symbolic exponent.
        return self._push(OP_POW, a=base_reg, b=self.emit(exp))

    def _exponent_half(self, exp: Any) -> int:
        """Return ``1`` for a ``1/2`` exponent, ``-1`` for ``-1/2``, else ``0``."""
        raise NotImplementedError  # pragma: no cover - abstract


# ---------------------------------------------------------------------------
# The SymPy emitter — the fallback path and the lowering oracle
# ---------------------------------------------------------------------------


class _SymPyEmitter(_Emitter):
    """Lower **SymPy** expression DAGs (the fallback / reference emitter).

    Reached by converting each SymEngine expression with ``._sympy_()``, which
    imports SymPy (~260 ms) and rebuilds the whole tree through SymPy's
    constructors.  :class:`_SymEngineEmitter` is the native path that avoids
    both; this one stays as the oracle it is validated against, and as the
    escape hatch for a node kind the native path declines
    (:class:`_NativeLoweringUnsupportedError`).
    """

    def __init__(self, leaf_for_name: dict[str, tuple[int, int]]) -> None:
        super().__init__(leaf_for_name)
        import sympy

        self._sympy = sympy
        self._const_one = sympy.Integer(1)

    def _emit(self, expr: Any) -> int:
        sympy = self._sympy

        # Any symbol-free subexpression (numbers, pi, e, constant folds).
        if not expr.free_symbols:
            return self._push(OP_CONST, imm=float(expr))

        if isinstance(expr, sympy.Symbol):
            return self._emit_leaf_symbol(expr.name)

        if isinstance(expr, sympy.Add):
            return self._fold_binary(OP_ADD, expr.args)

        if isinstance(expr, sympy.Mul):
            return self._fold_binary(OP_MUL, expr.args)

        if isinstance(expr, sympy.Pow):
            return self._emit_pow(expr)

        # Piecewise / selection (maps with ``np.where`` or a branch).  Lowered
        # to a comparison-masked arithmetic blend; see ``_blend_piecewise``.
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
            return self._fold_binary(OP_MIN if name == "Min" else OP_MAX, expr.args)

        # Floored modulo ``Mod(a, b)`` (SymPy's ``%``; the maps' bare ``%`` is
        # canonicalised to ``a - floor(a/b)*b`` instead, but a literal Mod node
        # lowers directly).
        if name == "Mod":
            if len(expr.args) != 2:
                raise TapeCompileError(f"Mod expects 2 arguments, got {len(expr.args)}")
            return self._push(OP_MOD, a=self.emit(expr.args[0]), b=self.emit(expr.args[1]))

        func_op = _FUNC_OPS.get(name)
        if func_op is not None:
            if len(expr.args) != 1:
                raise TapeCompileError(
                    f"function {name!r} expects 1 argument, got {len(expr.args)}"
                )
            return self._push(func_op, a=self.emit(expr.args[0]))

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
        sympy = self._sympy

        if isinstance(expr, sympy.And):
            return self._fold_and(expr.args)
        if isinstance(expr, sympy.Or):
            return self._fold_or(expr.args)
        if isinstance(expr, sympy.Not):
            return self._emit_not(expr.args[0])
        raise TapeCompileError(
            f"the instruction tape has no equivalent for boolean {type(expr).__name__!r}."
        )

    def _emit_piecewise(self, expr: Any) -> int:
        """Lower ``Piecewise((e0, c0), …, (en, True))`` to a masked blend."""
        pairs = [(p.args[0], p.args[1]) for p in expr.args]
        if pairs[-1][1] != self._sympy.true:
            raise TapeCompileError(
                "Piecewise must end with a default (True) branch to lower to a tape "
                f"(got condition {pairs[-1][1]!r}); the engine cannot represent a "
                "partial/undefined region."
            )
        return self._blend_piecewise(pairs)

    def _emit_pow(self, expr: Any) -> int:
        base, exp = expr.base, expr.exp
        base_reg = self.emit(base)
        is_int = isinstance(exp, self._sympy.Integer)
        return self._emit_pow_parts(base_reg, is_int, int(exp) if is_int else 0, exp)

    def _exponent_half(self, exp: Any) -> int:
        sympy = self._sympy
        if exp == sympy.Rational(1, 2):
            return 1
        if exp == sympy.Rational(-1, 2):
            return -1
        return 0


# ---------------------------------------------------------------------------
# The SymEngine emitter — the native (SymPy-free) lowering path
# ---------------------------------------------------------------------------
#
# Everything upstream of the tape is already SymEngine: ``_equations`` /
# ``_drift`` / the traced ``_step`` build SymEngine trees and ``.diff`` takes the
# Jacobian on them.  Converting each node to SymPy with ``._sympy_()`` just to
# read its structure back out cost a ~260 ms SymPy import on the first lowering
# in every process, plus the whole tree rebuilt through SymPy's constructors
# (measured: 5.1 s of Gray–Scott's 6.3 s profiled lowering).  This emitter reads
# the SymEngine tree directly.
#
# The hard constraint is that the emitted tape must stay **byte-identical**: the
# register layout is the emission order, so the tape encodes SymPy's canonical
# form, not merely the mathematics.  SymEngine's canonical form differs from
# SymPy's in exactly three ways, and this emitter reproduces all three:
#
# 1. **Commutative argument order.**  Both sort ``Add``/``Mul`` arguments, but by
#    different keys (SymEngine by its internal hash, SymPy by ``Basic.compare``).
#    :func:`_sympy_compare_key` reproduces SymPy's key; see its docstring.
# 2. **``Number * Add`` distributes in SymPy** (``Mul.flatten``'s "2*(1+a) ->
#    2 + 2*a" rule) but not in SymEngine.  :meth:`_SymEngineEmitter._canon`
#    rewrites it — in SymEngine, so SymEngine's own ``Add`` constructor does the
#    re-flattening.
# 3. **SymEngine has no ``exp`` node**: ``exp(x)`` is ``Pow(E, x)``.  The emitter
#    recognises that base and emits ``OP_EXP``.
#
# Anything outside the validated subset raises :class:`_NativeLoweringUnsupportedError`
# and the whole tape falls back to :class:`_SymPyEmitter` — the fallback is
# per-*tape*, never per-node, because a half-native emitter would key its CSE
# cache on two different node types and share nothing between them.


class _NativeLoweringUnsupportedError(Exception):
    """The SymEngine-native emitter declines this tree; fall back to SymPy.

    Internal control flow only — never surfaces to a caller.  Raised for node
    kinds whose SymPy canonical ordering this module does not reproduce
    (``And`` / ``Or`` / ``Not``, whose SymPy argument order comes from
    ``LatticeOp``'s ``default_sort_key`` rather than ``Basic.compare``), and for
    a node kind the canonicaliser cannot rebuild.
    """


#: ``sympy.core.basic.ordering_of_classes`` — the class-name precedence
#: ``Basic.compare`` consults before falling back to a plain name comparison.
#: Vendored (not imported) because importing it would import SymPy, which is the
#: whole point of this path; ``test_engine_compile.py`` pins it against the
#: installed SymPy so a SymPy release that reorders it fails loudly here rather
#: than silently moving every lowered tape.
_ORDERING_OF_CLASSES: tuple[str, ...] = (
    "Zero",
    "One",
    "Half",
    "Infinity",
    "NaN",
    "NegativeOne",
    "NegativeInfinity",
    "Integer",
    "Rational",
    "Float",
    "Exp1",
    "Pi",
    "ImaginaryUnit",
    "Symbol",
    "Wild",
    "Pow",
    "Mul",
    "Add",
    "Derivative",
    "Integral",
    "Abs",
    "Sign",
    "Sqrt",
    "Floor",
    "Ceiling",
    "Re",
    "Im",
    "Arg",
    "Conjugate",
    "Exp",
    "Log",
    "Sin",
    "Cos",
    "Tan",
    "Cot",
    "ASin",
    "ACos",
    "ATan",
    "ACot",
    "Sinh",
    "Cosh",
    "Tanh",
    "Coth",
    "ASinh",
    "ACosh",
    "ATanh",
    "ACoth",
    "RisingFactorial",
    "FallingFactorial",
    "factorial",
    "binomial",
    "Gamma",
    "LowerGamma",
    "UpperGamma",
    "PolyGamma",
    "Erf",
    "Chebyshev",
    "Chebyshev2",
    "Function",
    "WildFunction",
    "Lambda",
    "Order",
    "Equality",
    "Unequality",
    "StrictGreaterThan",
    "StrictLessThan",
    "GreaterThan",
    "LessThan",
)

_CLASS_RANK: dict[str, int] = {name: i for i, name in enumerate(_ORDERING_OF_CLASSES)}
#: ``_cmp_name``'s rank for a class name absent from the table above.
_UNKNOWN_CLASS_RANK: int = len(_ORDERING_OF_CLASSES) + 1

#: SymEngine relational class name → the SymPy ``rel_op`` string.  SymEngine
#: already canonicalises ``a > b`` to ``StrictLessThan(b, a)`` exactly as SymPy
#: does, so ``GreaterThan``/``StrictGreaterThan`` never reach the emitter.
_SE_REL_NAMES: dict[str, str] = {
    "StrictLessThan": "<",
    "LessThan": "<=",
    "Equality": "==",
    "Unequality": "!=",
}


def _mpf_tuple(x: float) -> tuple[int, int, int, int]:
    """Return mpmath's normalised ``(sign, man, exp, bc)`` for a finite float.

    ``sympy.Float._hashable_content()`` is ``(self._mpf_, self._prec)``, and
    ``Basic.compare`` orders two ``Float``s by that tuple (lexicographically —
    *not* numerically).  Reproducing it needs only bit twiddling: ``mpmath``'s
    ``from_float`` takes the IEEE mantissa/exponent and strips trailing zero
    bits so the mantissa is odd.
    """
    if x == 0.0:
        return (0, 0, 0, 0)
    sign = 1 if x < 0.0 else 0
    frac, exp2 = math.frexp(abs(x))
    man = int(frac * (1 << 53))
    exp = exp2 - 53
    trailing = (man & -man).bit_length() - 1
    man >>= trailing
    exp += trailing
    return (sign, man, exp, man.bit_length())


def _number_class_name(value: Any) -> str:
    """SymPy's class name for a SymEngine number.

    SymPy gives the distinguished small rationals their own classes, and
    ``ordering_of_classes`` ranks those *before* the generic ``Integer`` /
    ``Rational`` — so ``One`` sorts before ``Integer(7)``.  SymEngine spells them
    identically (``Zero`` / ``One`` / ``NegativeOne`` / ``Half`` / ``Integer`` /
    ``Rational`` / ``Pi`` / ``Exp1`` / ``NaN`` / ``Infinity``), so the only
    rename needed is the float type.
    """
    name = type(value).__name__
    if name in ("RealDouble", "RealMPFR"):
        return "Float"
    return name


def _sympy_compare_key(expr: Any, memo: dict[Any, Any]) -> Any:
    """Return a sort key reproducing SymPy's ``Basic.compare`` on a SymEngine tree.

    ``Add``/``Mul`` arguments are stored in SymPy's canonical order, which is
    ``[Number coefficient] + sorted(rest, key=cmp_to_key(Basic.compare))`` — so
    the emitted register layout (hence the tape bytes) depends on that order and
    this module has to reproduce it exactly.

    ``Basic.compare`` compares, in order: the class name (via
    ``ordering_of_classes``, else a plain string comparison), then the length of
    ``_hashable_content()``, then its elements — recursing into the ones that are
    themselves expressions.  That is a lexicographic comparison, so it maps onto
    a plain sort *key*: ``(rank, name, len(content), content)``.  Two keys only
    ever compare their ``content`` when the class names matched, so the content
    tuples are type-homogeneous and Python's tuple ordering is well defined.

    Raises
    ------
    _NativeLoweringUnsupportedError
        For a node whose SymPy hashable content this function does not model.
    """
    hit = memo.get(expr)
    if hit is not None:
        return hit

    name = type(expr).__name__
    content: tuple[Any, ...]

    if name == "Symbol":
        # SymPy: ``(self.name,) + tuple(sorted(self.assumptions0.items()))``.  A
        # SymEngine-born symbol carries only ``commutative=True``, so the second
        # element is a constant and only the name discriminates; it is kept so
        # the content length matches SymPy's.
        sname = "Symbol"
        content = (str(expr.name), 0)
    elif expr.is_Number:
        sname = _number_class_name(expr)
        if sname != "Float" and sname not in _SE_EXACT_RATIONAL_CLASSES:
            # SymEngine reports ``is_Number`` for several classes that carry no
            # ``.p``/``.q`` and no real ``float()``: ``ImaginaryUnit`` /
            # ``Complex`` / ``ComplexDouble`` (from e.g. ``sqrt(-1)``),
            # ``Infinity``, ``ComplexInfinity`` and ``NaN``.  Decline so the whole
            # tape falls back to the SymPy emitter, which reports the same
            # ``TypeError: Cannot convert complex to float`` it always has,
            # instead of leaking an ``AttributeError`` from this module.
            raise _NativeLoweringUnsupportedError(f"unmodelled number node {sname!r}")
        content = (_mpf_tuple(float(expr)), 53) if sname == "Float" else (int(expr.p), int(expr.q))
    elif name in ("Pi", "Exp1", "ImaginaryUnit"):
        # SymPy's ``NumberSymbol`` singletons: ``is_Number`` but not ``Number``,
        # so they sort as their own class with empty hashable content.
        sname = name
        content = ()
    elif name == "BooleanTrue" or name == "BooleanFalse":
        sname = name
        content = ()
    elif name == "Add" or name == "Mul":
        sname = name
        content = tuple(_sympy_compare_key(a, memo) for a in _canonical_args(expr, memo))
    elif name == "Pow":
        base, exponent = expr.args
        if _is_exp_base(base):
            sname = "exp"
            content = (_sympy_compare_key(exponent, memo),)
        else:
            sname = "Pow"
            content = (
                _sympy_compare_key(base, memo),
                _sympy_compare_key(exponent, memo),
            )
    elif name in _NATIVE_GATED_NAMES:
        raise _NativeLoweringUnsupportedError(f"no SymPy ordering model for {name!r}")
    elif name in _FUNC_OPS:
        sname = name
        content = tuple(_sympy_compare_key(a, memo) for a in expr.args)
    else:
        raise _NativeLoweringUnsupportedError(f"no SymPy ordering model for {name!r}")

    key = (_CLASS_RANK.get(sname, _UNKNOWN_CLASS_RANK), sname, len(content), content)
    memo[expr] = key
    return key


def _is_exp_base(node: Any) -> bool:
    """Whether ``node`` is SymEngine's ``E`` (so ``Pow(E, x)`` is SymPy's ``exp``)."""
    return type(node).__name__ == "Exp1"


#: SymEngine leaf numbers whose ``float()`` is the correctly-rounded double of an
#: exactly-representable value, hence bit-identical to SymPy's.  A *compound*
#: symbol-free subtree is **not** on this list: SymPy folds it through ``evalf``
#: (mpmath, guard digits, one final rounding) while SymEngine folds it through
#: ``eval_double`` (a C double per operation), and the two disagree by an ULP for
#: e.g. ``pi/12`` (0.2617993877991494 vs 0.26179938779914946 — measured on
#: ``CircadianRhythm``).  Since an ULP in the immediate pool is a changed tape,
#: the native path folds only these leaves and hands anything compound back to
#: the SymPy emitter.
_FOLDABLE_NUMBER_CLASSES: frozenset[str] = frozenset(
    {
        "Zero",
        "One",
        "NegativeOne",
        "Half",
        "Integer",
        "Rational",
        "RealDouble",
        "RealMPFR",
        "Pi",
        "Exp1",
    }
)

#: Node kinds the native path declines outright, because SymPy's ``eval`` for
#: them rewrites the node in a way SymEngine does not mirror:
#:
#: - ``Abs`` / ``sign`` run SymPy's ``signsimp`` on their argument and may flip
#:   its sign (``Abs(1 - x)`` → ``Abs(x - 1)``), choosing the representative with
#:   ``could_extract_minus_sign``, whose tie-break is ``default_sort_key``.
#: - ``Piecewise`` collapses branches SymEngine keeps (``Piecewise((0, c), (0,
#:   True))`` → ``0``).
#: - ``Min`` / ``Max`` and the boolean connectives order their arguments with
#:   ``default_sort_key`` (via ``MinMaxBase`` / ``LatticeOp``), not
#:   ``Basic.compare``.
#: - The relationals disagree on the canonical direction (SymEngine's
#:   ``0 <= x`` is SymPy's ``x >= 0``).
#:
#: All five appear only in a handful of catalogue kernels, so the fallback is
#: cheap; modelling ``default_sort_key`` and ``signsimp`` would not be.
_NATIVE_GATED_NAMES: frozenset[str] = frozenset(
    {
        "Abs",
        "sign",
        "Piecewise",
        "Min",
        "Max",
        "And",
        "Or",
        "Not",
        "StrictLessThan",
        "LessThan",
        "StrictGreaterThan",
        "GreaterThan",
        "Equality",
        "Unequality",
    }
)

#: Unary functions SymPy folds a leading minus through (odd → ``f(-x) = -f(x)``,
#: even → ``f(-x) = f(x)``).  Both libraries do this, but they can pick opposite
#: representatives of ``±arg`` when the argument is a *sum* — SymPy decides with
#: ``could_extract_minus_sign``, SymEngine with its own internal ordering.  So
#: the native path declines these when the argument is sign-ambiguous
#: (:func:`_sign_ambiguous`); a plain symbol / product argument has one
#: representative in both and is emitted natively.
_SIGN_FOLDING_FUNCS: frozenset[str] = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
    }
)


#: SymEngine class names for an exact integer.  SymEngine (like SymPy) gives
#: ``0`` / ``1`` / ``-1`` their own singleton classes, so a bare ``== "Integer"``
#: test would miss the ``x**-1`` → ``Recip`` fast path.
_SE_INTEGER_CLASSES: frozenset[str] = frozenset({"Integer", "Zero", "One", "NegativeOne"})
#: SymEngine class names for an exact non-integer rational (``1/2`` is ``Half``).
_SE_RATIONAL_CLASSES: frozenset[str] = frozenset({"Rational", "Half"})
#: Every SymEngine number class that exposes the ``.p`` / ``.q`` pair
#: :func:`_sympy_compare_key` reads.  ``is_Number`` is *broader* than this —
#: ``ImaginaryUnit`` / ``Complex`` / ``Infinity`` / ``NaN`` all report it and
#: carry neither — so the ordering key tests membership here, not ``is_Number``.
_SE_EXACT_RATIONAL_CLASSES: frozenset[str] = _SE_INTEGER_CLASSES | _SE_RATIONAL_CLASSES


def _sign_ambiguous(arg: Any) -> bool:
    """Whether ``±arg`` has two representatives SymPy and SymEngine may disagree on.

    A sum is ambiguous (``1 - x`` vs ``x - 1``), and so is a product with a sum
    factor (``a*(x - y)``).  Everything else — a symbol, a power, a nested
    function, a product of those, a product with a negative numeric coefficient —
    has a single canonical spelling in both libraries.
    """
    name = type(arg).__name__
    if name == "Add":
        return True
    if name == "Mul":
        return any(type(f).__name__ == "Add" for f in arg.args)
    return False


def _canonical_args(expr: Any, memo: dict[Any, Any]) -> list[Any]:
    """Return an ``Add``/``Mul``'s arguments in SymPy's canonical order.

    SymPy's ``Add.flatten`` / ``Mul.flatten`` sort the non-numeric arguments with
    ``Basic.compare`` and then ``insert(0, coeff)`` — the numeric coefficient is
    first and is *not* part of the sort.  SymEngine's ``get_args`` already puts
    its coefficient first (when it is not the identity), so the coefficient is
    detected positionally and only the tail is re-sorted.
    """
    args = list(expr.args)
    if not args:
        return args
    head: list[Any] = []
    if args[0].is_Number:
        head = [args[0]]
        args = args[1:]
    if len(args) > 1:
        args.sort(key=lambda a: _sympy_compare_key(a, memo))
    return head + args


class _SymEngineEmitter(_Emitter):
    """Lower **SymEngine** expression DAGs straight to SSA (no SymPy round-trip).

    See the module-level commentary above the class for the three canonical-form
    differences between SymEngine and SymPy that this emitter reproduces so the
    tape stays byte-identical to the SymPy path's.
    """

    def __init__(self, leaf_for_name: dict[str, tuple[int, int]]) -> None:
        super().__init__(leaf_for_name)
        import symengine

        self._se = symengine
        self._const_one = symengine.Integer(1)
        self._key_memo: dict[Any, Any] = {}
        self._canon_memo: dict[Any, Any] = {}

    # -- canonicalisation ----------------------------------------------------

    def _canon(self, expr: Any) -> Any:
        """Rewrite ``expr`` into SymPy's canonical *structure* (still SymEngine).

        The only structural rewrite needed is SymPy ``Mul.flatten``'s
        ``Number * Add`` distribution ("2*(1+a) -> 2 + 2*a"), which SymEngine does
        not perform.  Rebuilding the distributed sum through SymEngine's own
        ``Add`` constructor also re-flattens it into any enclosing sum, matching
        what SymPy's ``Add.flatten`` would have seen.
        """
        hit = self._canon_memo.get(expr)
        if hit is not None:
            return hit
        out = self._canon_impl(expr)
        self._canon_memo[expr] = out
        return out

    def _canon_impl(self, expr: Any) -> Any:
        se = self._se
        name = type(expr).__name__
        # Decline early so a gated node deep in a large tree aborts the native
        # attempt before the whole tree has been walked twice.
        if name in _NATIVE_GATED_NAMES:
            raise _NativeLoweringUnsupportedError(f"gated node kind {name!r}")
        if name == "Pow":
            self._check_pow_base(expr.args[0])
        if name == "Mul":
            args = expr.args
            if len(args) == 2 and args[0].is_Number and type(args[1]).__name__ == "Add":
                return self._canon(se.Add(*[args[0] * term for term in args[1].args]))
        args = expr.args
        if not args:
            return expr
        new = [self._canon(a) for a in args]
        if all(n is o or n == o for n, o in zip(new, args, strict=True)):
            return expr
        return self._rebuild(expr, name, new)

    @staticmethod
    def _check_pow_base(base: Any) -> None:
        """Decline the ``Pow`` bases whose exponent SymPy redistributes.

        ``Pow.eval`` rewrites a power in two ways SymEngine does not mirror, and
        both change the emitted *structure*, not just the register order:

        - a **nested power** collapses its exponents when the inner one is a
          rational of magnitude ``<= 1`` (``sqrt(x)**(-1/2)`` → ``x**(-1/4)``,
          ``sqrt(sqrt(x))`` → ``x**(1/4)``) — but not always (``(x**2)**(-1/2)``
          stays put, since ``x`` may be negative);
        - an **exact constant inside a product base** is extracted, because SymPy
          knows it is positive (``sqrt(pi*x)`` → ``sqrt(pi)*sqrt(x)``, which then
          folds ``sqrt(pi)`` to a ``Float``, so the *value* differs in the last
          bit as well as the layout).

        Modelling either would mean re-implementing ``Pow.eval``'s positivity
        reasoning, which is exactly what the gate exists to avoid.  No catalogue
        system uses either construct, so the fallback is never taken for the
        catalogue and this costs nothing.
        """
        bname = type(base).__name__
        if bname == "Pow":
            raise _NativeLoweringUnsupportedError("nested power base")
        if bname == "Mul" and any(type(f).__name__ in ("Pi", "Exp1") for f in base.args):
            raise _NativeLoweringUnsupportedError("exact constant inside a power base")

    def _rebuild(self, expr: Any, name: str, args: list[Any]) -> Any:
        """Rebuild ``expr`` from canonicalised children (only on an actual change)."""
        se = self._se
        if name == "Add":
            return se.Add(*args)
        if name == "Mul":
            return se.Mul(*args)
        if name == "Pow":
            return se.Pow(args[0], args[1])
        if name == "Piecewise":
            return se.Piecewise(*[(args[i], args[i + 1]) for i in range(0, len(args), 2)])
        if name in _SE_REL_NAMES:
            builder = {"<": se.Lt, "<=": se.Le, "==": se.Eq, "!=": se.Ne}[_SE_REL_NAMES[name]]
            return builder(args[0], args[1])
        func = getattr(expr, "func", None)
        if func is None:
            raise _NativeLoweringUnsupportedError(f"cannot rebuild a {name!r} node")
        return func(*args)

    # -- emission ------------------------------------------------------------

    def emit(self, expr: Any) -> int:
        """Emit ``expr`` (canonicalised first), returning its register."""
        return super().emit(self._canon(expr))

    def _emit(self, expr: Any) -> int:
        name = type(expr).__name__

        # A symbol-free subexpression folds to one ``Const`` register, exactly as
        # the SymPy emitter does — but only for the leaves whose ``float()`` is
        # provably SymPy's (see ``_FOLDABLE_NUMBER_CLASSES``).
        if not expr.free_symbols:
            if name not in _FOLDABLE_NUMBER_CLASSES:
                raise _NativeLoweringUnsupportedError(f"compound constant subtree {name!r}")
            return self._push(OP_CONST, imm=float(expr))

        if name == "Symbol":
            return self._emit_leaf_symbol(str(expr.name))

        if name == "Add":
            return self._fold_binary(OP_ADD, _canonical_args(expr, self._key_memo))

        if name == "Mul":
            return self._fold_binary(OP_MUL, _canonical_args(expr, self._key_memo))

        if name == "Pow":
            return self._emit_pow(expr)

        if name in _NATIVE_GATED_NAMES:
            raise _NativeLoweringUnsupportedError(f"gated node kind {name!r}")

        func_op = _FUNC_OPS.get(name)
        if func_op is not None:
            if len(expr.args) != 1:
                raise TapeCompileError(
                    f"function {name!r} expects 1 argument, got {len(expr.args)}"
                )
            arg = expr.args[0]
            if name in _SIGN_FOLDING_FUNCS and _sign_ambiguous(arg):
                raise _NativeLoweringUnsupportedError(f"sign-ambiguous argument to {name!r}")
            return self._push(func_op, a=self.emit(arg))

        raise TapeCompileError(f"the instruction tape has no equivalent for {name!r}.")

    def _emit_pow(self, expr: Any) -> int:
        base, exponent = expr.args
        # SymEngine has no ``exp`` node: ``exp(x)`` is ``Pow(E, x)``.
        if _is_exp_base(base):
            # SymPy's ``exp.eval`` splits a sum whose terms evaluate to a number
            # out of the exponent — ``exp(4.108 - x)`` becomes
            # ``60.82…*exp(-x)`` — which SymEngine does not do.  Decline the
            # whole tape rather than model which terms SymPy would peel off.
            if type(exponent).__name__ == "Add" and any(not t.free_symbols for t in exponent.args):
                raise _NativeLoweringUnsupportedError("exp of a sum with a constant term")
            return self._push(_FUNC_OPS["exp"], a=self.emit(exponent))
        base_reg = self.emit(base)
        is_int = type(exponent).__name__ in _SE_INTEGER_CLASSES
        return self._emit_pow_parts(base_reg, is_int, int(exponent) if is_int else 0, exponent)

    def _exponent_half(self, exp: Any) -> int:
        if type(exp).__name__ not in _SE_RATIONAL_CLASSES:
            return 0
        p, q = int(exp.p), int(exp.q)
        if q == 2 and p == 1:
            return 1
        if q == 2 and p == -1:
            return -1
        return 0


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

    def run(em: _Emitter, to_sympy: bool) -> tuple[list[int], list[int]]:
        """Emit the RHS (and Jacobian) with ``em``; ``to_sympy`` picks the path.

        The Jacobian expressions are differentiated *inside* the loop rather than
        materialised as a ``dim × dim`` list: Gray–Scott's 4608² entries would not
        fit in memory.  A fallback therefore re-differentiates, which is why the
        fallback is (measurably) never taken for the catalogue.
        """
        conv: Callable[[Any], Any] = (lambda e: e._sympy_()) if to_sympy else (lambda e: e)
        outs = [em.emit(conv(e)) for e in rhs]
        jac: list[int] = []
        if jacobian:
            # Row-major dim×dim: ∂f_k/∂u_j with abs/sign derivatives resolved a.e.
            for e in rhs:
                for s in state_syms:
                    jac.append(em.emit(conv(_resolve_derivative_nodes(e.diff(s)))))
        return outs, jac

    em: _Emitter
    if _native_lowering_enabled():
        em = _SymEngineEmitter(leaf_for_name)
        try:
            outputs, jac_outputs = run(em, to_sympy=False)
        except _NativeLoweringUnsupportedError:
            # Whole-tape fallback: a half-native emitter would key its CSE cache
            # on two node types and share nothing across them, so the partially
            # built emitter is discarded rather than continued.
            em = _SymPyEmitter(leaf_for_name)
            outputs, jac_outputs = run(em, to_sympy=True)
    else:
        em = _SymPyEmitter(leaf_for_name)
        outputs, jac_outputs = run(em, to_sympy=True)

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


#: env var: set truthy to force every lowering through the SymPy emitter.  The
#: bypass that proves NATIVE == SYMPY (``tests/test_engine_compile.py``) and the
#: escape hatch if a user's exotic kernel ever lowers differently on the two
#: paths.  Mirrors ``TSDYNAMICS_NO_TAPE_CACHE`` / ``TSDYNAMICS_NO_JIT_CACHE``.
_NATIVE_LOWERING_ENV = "TSDYNAMICS_NO_NATIVE_LOWERING"


def _native_lowering_enabled() -> bool:
    """Whether the SymEngine-native emitter is used (off if the env var is truthy)."""
    val = os.environ.get(_NATIVE_LOWERING_ENV, "")
    return val.strip().lower() not in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Lowered-tape cache (stream PERF-LOWER-CACHE)
# ---------------------------------------------------------------------------
#
# Lowering a system's symbolic dynamics to an IR :class:`Tape` is a pure function
# of the *math*: the kernel body, the dimension, the baked-in structural
# parameters, the DDE delays, and the ``with_jacobian`` flag.  Control parameters
# are **not** baked into the tape — they are read live at runtime through
# ``problem.params_vec()`` — so a control-parameter sweep (continuation, an
# orbit-diagram over a ``PoincareMap``, a Lyapunov sweep) re-lowers a *byte
# identical* tape on every value.  For small ODEs that is cheap (~0.1 ms), but for
# high-dimensional method-of-lines fields it dominates wholesale: a 4608-state
# Gray–Scott lowers in ~2.2 s while a short integration runs in ~0.1 s, so a
# parameter sweep that re-lowers per value spends >95 % of its time re-deriving an
# identical tape.
#
# This module memoises the lowered tape, keyed on **everything that affects the
# tape and nothing that does not**.  The key includes the kernel callable object
# itself (not merely ``id()``), so the entry is held alive exactly as long as the
# kernel is, and a runtime monkeypatch / redefinition of the kernel — swapping in
# a new function object — is a cache *miss* (no stale tape).  Control-parameter
# values are deliberately absent from the key.
#
# Correctness rests on the lowered :class:`Tape` (and :class:`LoweredSDE`) being
# treated as **immutable**: ``Tape`` is a frozen dataclass over ndarray fields
# that no downstream consumer writes to.  ``to_arrays`` returns contiguous,
# correct-dtype arrays for the FFI, but they are *views* onto the tape's own
# arrays whenever those are already contiguous (``np.ascontiguousarray`` is a
# no-op there) — the data is copied into owned Rust ``Vec``\s by
# ``Tape::from_arrays``, and no consumer mutates them, so a shared cached tape
# can be handed to every problem safely.

#: Maximum number of distinct lowered tapes retained (LRU eviction).  Bounds the
#: memory of a long session sweeping many distinct systems; a single sweep keys on
#: one entry, so this is generous.
_TAPE_CACHE_MAXSIZE = 256

#: env var: set truthy to disable the cache process-wide (always re-lower).  Lets
#: tests prove WITH-cache == WITHOUT-cache and gives users an escape hatch.
_TAPE_CACHE_ENV = "TSDYNAMICS_NO_TAPE_CACHE"

#: LRU store (insertion-ordered) + a coarse lock; lowering can run on worker
#: threads (e.g. an ensemble fan-out building a problem), so guard the dict.
_tape_cache: OrderedDict[Any, Any] = OrderedDict()
_tape_cache_lock = threading.Lock()
_tape_cache_hits = 0
_tape_cache_misses = 0


def _cache_enabled() -> bool:
    """Whether the lowered-tape cache is active (off if the env var is truthy)."""
    val = os.environ.get(_TAPE_CACHE_ENV, "")
    return val.strip().lower() not in ("1", "true", "yes", "on")


def clear_tape_cache() -> None:
    """Empty the lowered-tape cache and reset its hit/miss counters.

    The bypass hook for tests (prove a cached sweep equals a re-lowered one) and
    for freeing memory.  Combined with the ``TSDYNAMICS_NO_TAPE_CACHE`` env var
    (disable entirely), it gives a full clear/bypass surface.
    """
    global _tape_cache_hits, _tape_cache_misses
    with _tape_cache_lock:
        _tape_cache.clear()
        _tape_cache_hits = 0
        _tape_cache_misses = 0


def tape_cache_stats() -> dict[str, int]:
    """Return the cache ``{"hits", "misses", "size", "maxsize"}`` counters.

    Lets a test assert a repeat lowering was actually served from the cache.
    """
    with _tape_cache_lock:
        return {
            "hits": _tape_cache_hits,
            "misses": _tape_cache_misses,
            "size": len(_tape_cache),
            "maxsize": _TAPE_CACHE_MAXSIZE,
        }


def _hashable_value(v: Any) -> Any:
    """Coerce a parameter value into a collision-resistant hashable key part.

    Most parameters are plain scalars (already hashable, returned as-is).  An
    unhashable value — most plausibly a NumPy array used as a structural
    parameter — must be coerced without losing information: ``repr`` is unsafe
    because NumPy truncates large arrays (``array([0., 1., ..., 1997., ...])``),
    so two genuinely different arrays could collide and serve a *stale* tape.
    Arrays are keyed on ``(shape, dtype, sha256(bytes))`` instead; any other
    unhashable type falls back to a typed ``repr`` (and never to a bare ``repr``
    that could alias across types).
    """
    try:
        hash(v)
        return v
    except TypeError:
        pass
    arr = getattr(v, "tobytes", None)
    if arr is not None and hasattr(v, "shape") and hasattr(v, "dtype"):
        digest = hashlib.sha256(v.tobytes()).hexdigest()
        return ("ndarray", tuple(v.shape), str(v.dtype), digest)
    return (type(v).__name__, repr(v))


def _structural_key(system: Any) -> tuple[tuple[str, Any], ...]:
    """Return the ``(name, value)`` of every structural parameter, sorted by name.

    Structural parameters are baked into the tape as constants, so they belong in
    the key; control parameters are read live and must NOT be.  Values are coerced
    to a collision-resistant hashable form (see :func:`_hashable_value`).
    """
    struct_fn = getattr(system, "_structural_vals", None)
    struct = struct_fn() if struct_fn is not None else {}
    return tuple((k, _hashable_value(struct[k])) for k in sorted(struct))


def _all_params_key(system: Any) -> tuple[tuple[str, Any], ...]:
    """Return the ``(name, value)`` of every parameter, sorted — for map/DDE tapes.

    Maps and DDEs fold *all* parameters into the tape (``n_param == 0``), so every
    parameter value affects the tape and belongs in the key (a delay value is one
    such parameter, so DDE delays are covered here).
    """
    params = getattr(system, "params", {})
    return tuple((k, _hashable_value(params[k])) for k in sorted(params))


def _kernel_identity(cls: type, *names: str) -> tuple[Any, ...]:
    """Hold the raw kernel callables in the key so identity drives invalidation.

    Storing the function object (the staticmethod unwrapped) — not its ``id()`` —
    makes the key compare by object identity *and* keeps the kernel alive for the
    entry's lifetime, so a monkeypatched / redefined kernel (a new function object)
    misses the cache and re-lowers, while an unchanged kernel hits.
    """
    return tuple(
        getattr(getattr(cls, n), "__func__", getattr(cls, n))
        for n in names
        if getattr(cls, n, None) is not None
    )


def _cache_get_or_build[T](key: Any, build: Callable[[], T]) -> T:
    """Return the cached value for ``key`` (LRU), building + storing it on a miss.

    ``build`` is only called outside the lock (lowering can be slow and may itself
    recurse), so two threads racing the same cold key may both build — harmless,
    since the result is a value-identical immutable tape; the last writer wins and
    both callers get a correct tape.  When the cache is disabled (env var) this is
    a straight passthrough that never touches the store.
    """
    global _tape_cache_hits, _tape_cache_misses
    if not _cache_enabled():
        return build()
    with _tape_cache_lock:
        hit = _tape_cache.get(key, _MISS)
        if hit is not _MISS:
            _tape_cache.move_to_end(key)
            _tape_cache_hits += 1
            return cast("T", hit)
        _tape_cache_misses += 1
    value = build()
    with _tape_cache_lock:
        _tape_cache[key] = value
        _tape_cache.move_to_end(key)
        while len(_tape_cache) > _TAPE_CACHE_MAXSIZE:
            _tape_cache.popitem(last=False)
    return value


_MISS = object()  # sentinel distinguishing "absent" from a stored ``None``


def lower_ode_cached(system: Any, *, with_jacobian: bool = False) -> Tape:
    """Return a cached :func:`lower_ode` tape, memoised across a parameter sweep.

    Keyed on the system class, ``with_jacobian``, dimension, the structural
    parameters (baked into the tape), and the ``_equations`` kernel object.
    Control-parameter values are absent from the key (they feed the tape live via
    ``params_vec``), so a control-parameter sweep reuses one cached tape.

    The tape's ``control_names`` (the runtime parameter-vector *layout*) are not a
    key part of their own: they are a pure function of the system **class** — they
    derive from ``_control_params()`` / the ``params`` ⨯ ``_structural_params``
    split, neither of which a control-parameter *value* changes — so the class in
    the key captures them transitively.  A construct that could change the control
    layout without changing the class (e.g. per-instance ``_structural_params``)
    would break this invariant; the catalogue does not do that.
    """
    key = (
        "ode",
        type(system),
        bool(with_jacobian),
        int(system.dim),
        _structural_key(system),
        _kernel_identity(type(system), "_equations"),
    )
    return _cache_get_or_build(key, lambda: lower_ode(system, with_jacobian=with_jacobian))


def lower_map_cached(system: Any, *, with_jacobian: bool = False) -> Tape:
    """Return a cached :func:`lower_map` next-state tape.

    Maps fold *all* parameters into the tape, so the key carries every parameter
    value (alongside the class, ``with_jacobian``, dim, and the ``_step`` kernel
    object); a parameter change is therefore a deliberate miss.
    """
    key = (
        "map",
        type(system),
        bool(with_jacobian),
        int(system.dim),
        _all_params_key(system),
        _kernel_identity(type(system), "_step"),
    )
    return _cache_get_or_build(key, lambda: lower_map(system, with_jacobian=with_jacobian))


def lower_dde_cached(system: Any) -> tuple[Tape, list[DelaySlot]]:
    """Return a cached :func:`lower_dde` extended tape + delay slots.

    DDEs bake every parameter (delays included) into the tape, so the key carries
    all parameter values plus the class, dim and ``_equations`` kernel object.  The
    returned ``(tape, slots)`` is immutable (the ``DelaySlot`` list holds plain
    namedtuples); a fresh slot list is returned per call so a consumer cannot
    mutate the cached one.
    """
    key = (
        "dde",
        type(system),
        int(system.dim),
        _all_params_key(system),
        _kernel_identity(type(system), "_equations"),
    )
    tape, slots = _cache_get_or_build(key, lambda: lower_dde(system))
    return tape, list(slots)


def lower_sde_cached(system: Any, *, with_diffusion_jacobian: bool = False) -> LoweredSDE:
    """Return a cached :func:`lower_sde` drift + diffusion tape pair.

    Keyed like the ODE path (class, ``with_diffusion_jacobian``, dim, structural
    parameters) plus *both* kernel objects (``_drift`` and ``_diffusion``), so a
    monkeypatch of either invalidates the entry.  Control parameters feed both
    tapes live, so they stay out of the key.

    As in :func:`lower_ode_cached`, the shared ``control_names`` layout of both
    tapes is captured *transitively* by the class: the control-name layout is a
    pure function of the class (``_control_params()`` / the ``params`` ⨯
    ``_structural_params`` split), independent of any control-parameter value, so
    it needs no key part of its own.
    """
    key = (
        "sde",
        type(system),
        bool(with_diffusion_jacobian),
        int(system.dim),
        _structural_key(system),
        _kernel_identity(type(system), "_drift", "_diffusion"),
    )
    return _cache_get_or_build(
        key, lambda: lower_sde(system, with_diffusion_jacobian=with_diffusion_jacobian)
    )


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
        The lowered RHS (and Jacobian, if requested), already validated.

    Raises
    ------
    ValueError
        If ``_equations`` does not return exactly ``system.dim`` expressions.
    TapeCompileError
        If the RHS uses a construct the instruction tape cannot express (an
        unsupported function, or a free symbol that is not a declared input).

    Examples
    --------
    >>> import tsdynamics as ts
    >>> from tsdynamics.engine.compile import lower_ode, eval_tape
    >>> tape = lower_ode(ts.systems.Lorenz())
    >>> tape.dim
    3
    """
    import symengine

    dim = system.dim
    struct_vals = system._structural_vals()
    control_names = list(system._control_params())
    control_syms = {k: symengine.Symbol(f"p{i}") for i, k in enumerate(control_names)}

    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    t_canon = symengine.Symbol("t")

    # Build the RHS directly over the canonical state symbols u_i (accessor
    # ``y(i) -> u_syms[i]``) and time ``t``.  The previous path built the RHS over
    # a Function ``y(i)`` and then substituted ``{y(i): u_i}`` for every i — an
    # O(dim²) operation (a dim-entry subs applied to each of dim expressions) that
    # dominated lowering for high-dimensional method-of-lines fields (~18 s for a
    # 4608-state Gray-Scott; sub-second now).  The lowered tape is identical: the
    # expressions are the same, merely constructed over u_i from the start.
    def y(i: int) -> Any:
        return u_syms[i]

    exprs = list(
        _trace_kernel(
            system,
            "_equations",
            lambda: type(system)._equations(y, t_canon, **{**struct_vals, **control_syms}),
        )
    )
    if len(exprs) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(exprs)}")
    rhs = [symengine.sympify(e) for e in exprs]

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

    A map's ``_step`` is a numeric (``staticmethod``) function, so it is *traced*
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

    # A numeric ``_step`` can still fail to trace on symbolic state: a Python
    # ``if`` on the state (``TypeError`` on a Relational's truth value — use
    # ``np.where`` for a branchless step instead), a NumPy routine the symbolic
    # shim does not cover, shape/index errors, etc.  Whatever the cause, it means
    # this map cannot lower to a straight-line tape.  ``np`` *is* traced here (the
    # ``_SymbolicNumpy`` shim), so it is not listed as a numeric module.
    out = _trace_kernel(
        system,
        "_step",
        lambda: step(state, *params),
        numeric_modules=_MAP_NUMERIC_MODULES,
        hint=(
            "A map `_step` may use `np.*` (it is traced through a symbolic shim), but it "
            "must not branch on the state with a Python `if` — rewrite the branch with "
            "`np.where` — and the shim only models the routines the tape has opcodes for."
        ),
    )
    exprs = _trace_kernel(system, "_step", lambda: [symengine.sympify(e) for e in list(out)])

    if len(exprs) != dim:
        raise TapeCompileError(f"_step traced to {len(exprs)} components, expected dim={dim}")

    return lower_expressions(exprs, u_syms, jacobian=with_jacobian)


#: Memo for :func:`map_jacobian_fn`, keyed like every other lowering cache: the
#: ``_step`` **function object** (so a monkeypatched or redefined kernel is a
#: deliberate miss) plus the concrete parameter values it was traced against.
_MAP_JACOBIAN_MEMO: dict[tuple[Any, ...], Callable[[Any], np.ndarray]] = {}


def map_jacobian_fn(cls: Any, params: Sequence[Any], *, dim: int) -> Callable[[Any], np.ndarray]:
    """Return a numeric ``J(x)`` for a map, derived symbolically from ``_step``.

    This is the map twin of :meth:`ContinuousSystem.jacobian`: a map's Jacobian
    is the symbolic derivative ``∂step_k/∂u_j`` of its own ``_step``, so writing
    one by hand is transcription work the library can do exactly.  ``_step`` is
    traced on a symbolic state (the same trace :func:`lower_map` performs),
    differentiated with SymEngine, and compiled to a numeric callable with
    ``Lambdify``.

    Parameters are folded in as **constants**, matching :func:`lower_map`, so the
    result is memoised per ``(kernel object, parameter values)``.

    Parameters
    ----------
    cls : type[DiscreteMap]
        The map class (its ``_step`` is the kernel to differentiate).
    params : sequence
        Parameter values in declaration order — exactly what ``_step`` receives.
    dim : int
        State dimension.

    Returns
    -------
    callable
        ``J(x) -> ndarray`` of shape ``(dim, dim)``.

    Raises
    ------
    TapeCompileError
        If ``_step`` cannot be traced symbolically (a Python ``if`` on the
        state, or a routine the symbolic shim does not model).  Such a map must
        supply its own ``_jacobian``.
    """
    import symengine

    from tsdynamics.families.discrete import _unwrap_static

    key = (_unwrap_static(cls._step), dim, tuple(map(_hashable_param, params)))
    cached = _MAP_JACOBIAN_MEMO.get(key)
    if cached is not None:
        return cached

    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    state = np.array(u_syms, dtype=object)
    step = _trace_step(_unwrap_static(cls._step))

    out = _trace_kernel(
        cls,
        "_step",
        lambda: step(state, *params),
        numeric_modules=_MAP_NUMERIC_MODULES,
        hint=(
            "A map `_step` may use `np.*` (it is traced through a symbolic shim), but it "
            "must not branch on the state with a Python `if` — rewrite the branch with "
            "`np.where`. A map whose step genuinely cannot be traced must define its own "
            "`_jacobian` (a @staticmethod returning the dim x dim matrix)."
        ),
    )
    exprs = _trace_kernel(cls, "_step", lambda: [symengine.sympify(e) for e in list(out)])
    if len(exprs) != dim:
        raise TapeCompileError(f"_step traced to {len(exprs)} components, expected dim={dim}")

    rows = [[e.diff(u) for u in u_syms] for e in exprs]
    flat = [_resolve_map_derivative(entry) for row in rows for entry in row]
    lam = symengine.Lambdify(u_syms, flat, real=True)

    def jac(x: Any) -> np.ndarray:
        arr = np.asarray(lam(np.asarray(x, dtype=float).ravel()), dtype=float)
        return arr.reshape(dim, dim)

    if len(_MAP_JACOBIAN_MEMO) >= _TAPE_CACHE_MAXSIZE:
        _MAP_JACOBIAN_MEMO.clear()
    _MAP_JACOBIAN_MEMO[key] = jac
    return jac


def _hashable_param(value: Any) -> Any:
    """Make a parameter value usable in the memo key (arrays are not hashable)."""
    if isinstance(value, np.ndarray):
        return (value.shape, value.tobytes())
    return value


def _resolve_map_derivative(entry: Any) -> Any:
    """Resolve the a.e. derivative nodes SymEngine leaves unevaluated.

    ``d|u|/du``, ``d sign(u)/du`` and ``d floor(u)/du`` are left as unevaluated
    ``Derivative`` nodes that ``Lambdify`` cannot compile; a map built from
    ``np.abs`` / ``np.sign`` / the ``%`` opcode hits all three.  This reuses the
    resolution :class:`~tsdynamics.families.ContinuousSystem` already applies to
    its autogenerated Jacobian, so both families answer the same way.
    """
    from tsdynamics.families.continuous import _resolve_derivative_nodes

    return _resolve_derivative_nodes(entry)


def lower_map_sweep(system: Any, sweep_param: str) -> Tape:
    """Lower a map's ``_step`` keeping ``sweep_param`` as a runtime ``Param``.

    The variant :func:`lower_map` uses for the orbit-diagram parameter sweep
    (stream ``perf/param-sweep-kernel``): unlike :func:`lower_map`, which folds
    *every* parameter into the tape as a constant, this keeps the **swept**
    parameter as the tape's single runtime ``Param`` input (``n_param == 1``,
    ``control_names == [sweep_param]``) while the other parameters stay folded.
    The sweep kernel then varies that one input per value, so the whole sweep is
    one engine call with no per-value re-lowering.

    The lowered next-state expression is **byte-identical** to
    :func:`lower_map`'s at runtime for the swept parameter: a ``Param`` leaf
    reads the same ``f64`` value a folded ``Const`` would, then feeds the same
    op — so a value swept here lands bit-for-bit where :func:`lower_map` with
    that value baked in would (verified for the logistic map).

    Parameters
    ----------
    system : DiscreteMap
        The map instance (its non-swept parameter values are baked in).
    sweep_param : str
        The parameter to keep as the runtime input.  Must be a declared
        parameter of ``system``.

    Returns
    -------
    Tape
        ``n_param == 1`` over the single swept parameter (no Jacobian — the sweep
        records states, not stability).

    Raises
    ------
    InvalidParameterError
        If ``sweep_param`` is not a declared parameter of ``system``.
    TapeCompileError
        If ``_step`` cannot be traced symbolically (see :func:`lower_map`).
    """
    import symengine

    from tsdynamics.families.discrete import _unwrap_static

    param_names = list(system.params)
    if sweep_param not in param_names:
        from tsdynamics.errors import invalid_value

        raise invalid_value(
            "sweep_param",
            sweep_param,
            options=param_names,
            hint="orbit_diagram sweeps a declared parameter of the map.",
        )

    dim = system.dim
    u_syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    state = np.array(u_syms, dtype=object)
    step = _trace_step(_unwrap_static(type(system)._step))

    # Pass the swept parameter as a symbol (a runtime Param) and every other
    # parameter as its current numeric value (folded to a constant), positionally
    # in declaration order — exactly the order ``_step`` expects.
    sweep_sym = symengine.Symbol("psweep")
    args: list[Any] = [
        sweep_sym if name == sweep_param else system.params[name] for name in param_names
    ]

    out = _trace_kernel(
        system,
        "_step",
        lambda: step(state, *args),
        numeric_modules=_MAP_NUMERIC_MODULES,
        hint=(
            "A map `_step` may use `np.*` (it is traced through a symbolic shim), but it "
            "must not branch on the state with a Python `if` — rewrite the branch with "
            "`np.where` — and the shim only models the routines the tape has opcodes for."
        ),
    )
    exprs = _trace_kernel(system, "_step", lambda: [symengine.sympify(e) for e in list(out)])

    if len(exprs) != dim:
        raise TapeCompileError(f"_step traced to {len(exprs)} components, expected dim={dim}")

    return lower_expressions(
        exprs,
        u_syms,
        param_syms=[sweep_sym],
        control_names=[sweep_param],
    )


def lower_map_sweep_cached(system: Any, sweep_param: str) -> Tape:
    """Return a cached :func:`lower_map_sweep` tape, memoised across the sweep.

    Keyed like :func:`lower_map_cached` (class, dim, the ``_step`` kernel object)
    plus the swept parameter name and the values of every **other** parameter (the
    folded-in constants).  The swept parameter's value is deliberately absent —
    it is the runtime input — so the whole sweep reuses one cached tape.
    """
    others = tuple(
        (k, _hashable_value(system.params[k])) for k in sorted(system.params) if k != sweep_param
    )
    key = (
        "map_sweep",
        type(system),
        str(sweep_param),
        int(system.dim),
        others,
        _kernel_identity(type(system), "_step"),
    )
    return _cache_get_or_build(key, lambda: lower_map_sweep(system, sweep_param))


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

    Parameters are folded to constants, so the tape has ``n_param = 0``: a delay
    value bakes into the tape, so a DDE re-lowers on any parameter change and
    carries no runtime parameter vector (see the "No compilation cache" section
    of ``CLAUDE.md``).  Only constant delays are supported — a state-dependent
    delay (``τ`` depending on ``y``) raises.

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
    exprs = list(
        _trace_kernel(
            system,
            "_equations",
            lambda: type(system)._equations(y, t_sym, **system.params.as_dict()),
            hint=(
                "A delayed access is spelled `y(i, t - tau)`; it is a symbolic node too, so "
                "it cannot be fed to a numeric routine either."
            ),
        )
    )
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

    Follows the resolved noise contract (see CLAUDE.md, StochasticSystem): ``_drift(y, t, **params)``
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
        ``(y, t, **params)`` signature, plus ``dim`` / ``params`` — the
        :class:`~tsdynamics.families.stochastic.StochasticSystem` contract
        (duck-typed here so the engine layer stays below ``families`` in the
        import graph).
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
    from tsdynamics.families.discrete import _unwrap_static

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

    drift_exprs = list(
        _trace_kernel(system, "_drift", lambda: _unwrap_static(drift_fn)(y, t_sym, **call_kwargs))
    )
    diff_exprs = list(
        _trace_kernel(
            system, "_diffusion", lambda: _unwrap_static(diff_fn)(y, t_sym, **call_kwargs)
        )
    )
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
    return np.asarray(regs[tape.outputs])


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


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
