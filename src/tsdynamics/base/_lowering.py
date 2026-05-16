"""Tracer-driven lowering from ``DiscreteMap._step`` / ``_jacobian`` to IR.

We run the user method once with :class:`Tracer` instances in place of
state and parameters, walk the returned structure into a fixed
``(dim, dim×dim)`` shape, and serialise to bytecode. The output
:class:`CompiledMap` (in :mod:`tsdynamics.base._ir`) is what crosses the
PyO3 boundary.

If anything along the way raises :class:`NotLowerableError`, the caller
catches it and falls back to the Numba dispatch path for that one map.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ._ir import CompiledMap, Const, NotLowerableError, serialize
from ._tracer import Tracer, param_tracers, state_tracer

if TYPE_CHECKING:
    pass


def _coerce_node(value):
    """Take an arbitrary ``_step``-returned value and return an IR Node.

    Plain numbers (``2.0``) appearing in a Jacobian row are wrapped as
    ``Const``; Tracers are unwrapped to their inner node. Anything else
    is non-lowerable.
    """
    if isinstance(value, Tracer):
        return value._node
    if isinstance(value, bool):
        return Const(1.0 if value else 0.0)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return Const(float(value))
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return Const(float(value))
    raise NotLowerableError(f"unexpected leaf of type {type(value).__name__} in lowered output")


def _flatten_step(result, dim: int):
    """Normalise the ``_step`` return value to a length-``dim`` list of Nodes."""
    if dim == 1:
        # Maps may return either a bare scalar/Tracer or a length-1 sequence.
        if isinstance(result, (tuple, list)):
            if len(result) != 1:
                raise NotLowerableError(f"dim=1 step returned sequence of length {len(result)}")
            return [_coerce_node(result[0])]
        return [_coerce_node(result)]

    if not isinstance(result, (tuple, list)):
        raise NotLowerableError(
            f"dim={dim} step must return tuple/list, got {type(result).__name__}"
        )
    if len(result) != dim:
        raise NotLowerableError(f"step returned {len(result)} items, expected dim={dim}")
    return [_coerce_node(v) for v in result]


def _flatten_jacobian(result, dim: int):
    """Normalise the ``_jacobian`` return value to a ``dim × dim`` Node grid."""
    if not isinstance(result, (tuple, list)):
        raise NotLowerableError(f"_jacobian must return tuple/list, got {type(result).__name__}")

    if dim == 1:
        # Per existing convention: ``_jacobian`` returns ``[scalar]``.
        if len(result) != 1:
            raise NotLowerableError(f"dim=1 jacobian returned sequence of length {len(result)}")
        return [[_coerce_node(result[0])]]

    if len(result) != dim:
        raise NotLowerableError(f"_jacobian returned {len(result)} rows, expected {dim}")

    grid = []
    for row in result:
        if not isinstance(row, (tuple, list, Sequence)) or isinstance(row, str):
            raise NotLowerableError(f"_jacobian row must be tuple/list, got {type(row).__name__}")
        if not isinstance(row, (tuple, list)):
            row = list(row)  # accept nalgebra-style sequences
        if len(row) != dim:
            raise NotLowerableError(f"_jacobian row has length {len(row)}, expected {dim}")
        grid.append([_coerce_node(cell) for cell in row])
    return grid


def lower_to_ir(cls: type, params_tuple: tuple, dim: int) -> CompiledMap:
    """Trace ``cls._step`` / ``cls._jacobian`` and produce a CompiledMap.

    ``params_tuple`` is unused at trace time — params come in as
    :class:`Tracer` placeholders so they can be bound at evaluation time
    in Rust. We still pass the tuple's length so the IR carries the
    right ``n_params`` count.

    Parameters
    ----------
    cls
        The :class:`DiscreteMap` subclass.
    params_tuple
        Parameter values (only its length is used here; values are bound
        per-call in the Rust kernel).
    dim
        State-space dimension.

    Raises
    ------
    NotLowerableError
        If ``_step`` or ``_jacobian`` contains operations the IR can't
        represent. Caller should fall back to the Numba path.
    """
    n_params = len(params_tuple)
    X = state_tracer(dim)
    P = param_tracers(n_params)

    # @staticjit wraps with numba.njit; the dispatcher exposes ``.py_func``
    # so we can call the underlying Python implementation with Tracer inputs.
    # When numba is unavailable, ``_step`` is already the raw function.
    step_fn = getattr(cls._step, "py_func", cls._step)
    jac_fn = getattr(cls._jacobian, "py_func", cls._jacobian)

    try:
        step_result = step_fn(X, *P)
    except NotLowerableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NotLowerableError(f"_step trace of {cls.__name__} failed: {exc!r}") from exc

    try:
        jac_result = jac_fn(X, *P)
    except NotLowerableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NotLowerableError(f"_jacobian trace of {cls.__name__} failed: {exc!r}") from exc

    step_nodes = _flatten_step(step_result, dim)
    jac_nodes = _flatten_jacobian(jac_result, dim)
    return serialize(dim=dim, n_params=n_params, step=step_nodes, jacobian=jac_nodes)


__all__ = ["lower_to_ir"]


# Suppress unused import warning when we re-export Iterable only for typing
_ = Iterable
