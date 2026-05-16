"""Symbolic IR for compiled discrete-map kernels.

The IR is intentionally small — sized for the 26 built-in maps that ship
today. Later N-milestones (variational ODE, cranelift JIT) extend it but
never break the wire format: opcode byte values are stable.

The Python side builds an IR tree from a tracer pass over ``_step`` and
``_jacobian`` (see :mod:`tsdynamics.base._lowering`), then serialises to a
postfix bytecode that Rust decodes once per system in
``crates/tsdyn-core/src/ir.rs``. Booleans live in the same f64 stack:
comparison opcodes push ``1.0`` / ``0.0``; ``Where`` selects on
``cond != 0``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


class NotLowerableError(Exception):
    """Raised when an expression contains an operation the IR can't encode.

    Caught by callers (``DiscreteMap._compile_ir``) which then mark the
    system as non-lowerable and dispatch through the Numba fallback path
    instead of the Rust kernel.
    """


# ---------------------------------------------------------------------------
# Opcodes — must match crates/tsdyn-core/src/ir.rs.
# ---------------------------------------------------------------------------

OP_CONST = 0x00
OP_VAR = 0x01
OP_PARAM = 0x02

OP_ADD = 0x10
OP_SUB = 0x11
OP_MUL = 0x12
OP_DIV = 0x13
OP_NEG = 0x14
OP_POW = 0x15
OP_MOD = 0x16

OP_SIN = 0x20
OP_COS = 0x21
OP_EXP = 0x22
OP_LOG = 0x23
OP_ABS = 0x24
OP_SQRT = 0x25
OP_ARCCOS = 0x26
OP_SIGN = 0x27

OP_WHERE = 0x30
OP_LT = 0x31
OP_LE = 0x32
OP_GT = 0x33
OP_GE = 0x34
OP_AND = 0x35

_BINOP_CODES = {
    "add": OP_ADD,
    "sub": OP_SUB,
    "mul": OP_MUL,
    "div": OP_DIV,
    "mod": OP_MOD,
    "and": OP_AND,
    "lt": OP_LT,
    "le": OP_LE,
    "gt": OP_GT,
    "ge": OP_GE,
}

_UNARY_CODES = {
    "neg": OP_NEG,
    "sin": OP_SIN,
    "cos": OP_COS,
    "exp": OP_EXP,
    "log": OP_LOG,
    "abs": OP_ABS,
    "sqrt": OP_SQRT,
    "arccos": OP_ARCCOS,
    "sign": OP_SIGN,
}


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Const:
    value: float


@dataclass(frozen=True, slots=True)
class Var:
    idx: int


@dataclass(frozen=True, slots=True)
class Param:
    idx: int


@dataclass(frozen=True, slots=True)
class BinOp:
    op: str
    left: Node
    right: Node


@dataclass(frozen=True, slots=True)
class UnaryOp:
    op: str
    arg: Node


@dataclass(frozen=True, slots=True)
class Pow:
    base: Node
    exp: int


@dataclass(frozen=True, slots=True)
class Where:
    cond: Node
    t: Node
    f: Node


Node = Const | Var | Param | BinOp | UnaryOp | Pow | Where


# ---------------------------------------------------------------------------
# Postfix emit
# ---------------------------------------------------------------------------


def _emit_program(node: Node, buf: bytearray) -> int:
    """Append the postfix encoding of ``node`` to ``buf``; return op count."""
    n_ops = 0
    if isinstance(node, Const):
        buf.append(OP_CONST)
        buf.extend(struct.pack("<d", float(node.value)))
        n_ops += 1
    elif isinstance(node, Var):
        buf.append(OP_VAR)
        buf.extend(struct.pack("<I", int(node.idx)))
        n_ops += 1
    elif isinstance(node, Param):
        buf.append(OP_PARAM)
        buf.extend(struct.pack("<I", int(node.idx)))
        n_ops += 1
    elif isinstance(node, BinOp):
        n_ops += _emit_program(node.left, buf)
        n_ops += _emit_program(node.right, buf)
        buf.append(_BINOP_CODES[node.op])
        n_ops += 1
    elif isinstance(node, UnaryOp):
        n_ops += _emit_program(node.arg, buf)
        buf.append(_UNARY_CODES[node.op])
        n_ops += 1
    elif isinstance(node, Pow):
        n_ops += _emit_program(node.base, buf)
        buf.append(OP_POW)
        buf.extend(struct.pack("<i", int(node.exp)))
        n_ops += 1
    elif isinstance(node, Where):
        n_ops += _emit_program(node.cond, buf)
        n_ops += _emit_program(node.t, buf)
        n_ops += _emit_program(node.f, buf)
        buf.append(OP_WHERE)
        n_ops += 1
    else:
        raise NotLowerableError(f"unknown IR node type: {type(node).__name__}")
    return n_ops


def _emit_with_header(node: Node, buf: bytearray) -> None:
    """Emit ``u32 n_ops`` then the program bytes."""
    inner = bytearray()
    n_ops = _emit_program(node, inner)
    buf.extend(struct.pack("<I", n_ops))
    buf.extend(inner)


# ---------------------------------------------------------------------------
# CompiledMap payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledMap:
    """Serialised IR payload + metadata. Passed across the PyO3 boundary."""

    dim: int
    n_params: int
    bytecode: bytes


def serialize(
    *, dim: int, n_params: int, step: list[Node], jacobian: list[list[Node]]
) -> CompiledMap:
    """Serialise the step + Jacobian programs into the wire format.

    Layout (little-endian):

    - ``u32`` dim
    - ``u32`` n_params
    - ``u32`` n_step (== dim)
    - per step expr: ``u32`` n_ops, ops…
    - ``u32`` n_jac_rows (== dim)
    - per row: ``u32`` n_cols (== dim); per cell: ``u32`` n_ops, ops…
    """
    if len(step) != dim:
        raise NotLowerableError(f"step length {len(step)} != dim {dim}")
    if len(jacobian) != dim or any(len(row) != dim for row in jacobian):
        raise NotLowerableError(
            f"jacobian shape != ({dim}, {dim}); got {[len(row) for row in jacobian]}"
        )

    buf = bytearray()
    buf.extend(struct.pack("<I", int(dim)))
    buf.extend(struct.pack("<I", int(n_params)))

    buf.extend(struct.pack("<I", int(dim)))
    for expr in step:
        _emit_with_header(expr, buf)

    buf.extend(struct.pack("<I", int(dim)))
    for row in jacobian:
        buf.extend(struct.pack("<I", int(dim)))
        for cell in row:
            _emit_with_header(cell, buf)

    return CompiledMap(dim=dim, n_params=n_params, bytecode=bytes(buf))
