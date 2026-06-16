"""Utility decorators shared across TSDynamics modules."""

from __future__ import annotations

from collections.abc import Callable


def staticjit(func: Callable) -> Callable:
    """Mark a map's ``_step`` / ``_jacobian`` as a plain ``staticmethod``.

    Historically this also applied Numba's ``njit``; since the M3 migration the
    Rust engine lowers ``_step`` to the IR and iterates it natively, so the
    Python form is just a ``staticmethod`` (also used by the pure-Python
    reference evaluator and the analytic-Jacobian analyses).  The name is kept
    for backward compatibility with every built-in map definition.
    """
    return staticmethod(func)
