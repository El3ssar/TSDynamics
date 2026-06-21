"""Utility decorators shared across TSDynamics modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # To a type checker ``staticjit`` *is* ``staticmethod``: this makes a decorated
    # ``_step(x, y, a, b)`` read with ``x`` as the first *parameter* (not ``self``).
    # Aliasing the builtin (rather than a custom ``-> staticmethod[P, R]`` decorator)
    # is what mypy actually honours — a custom decorator still binds the first
    # argument to the enclosing class and mis-types every state component as the
    # class itself.
    staticjit = staticmethod
else:

    def staticjit(func):
        """Mark a map's ``_step`` / ``_jacobian`` as a plain ``staticmethod``.

        Historically this also applied Numba's ``njit``; since the M3 migration the
        Rust engine lowers ``_step`` to the IR and iterates it natively, so the
        Python form is just a ``staticmethod`` (also used by the pure-Python
        reference evaluator and the analytic-Jacobian analyses).  The name is kept
        for backward compatibility with every built-in map definition.
        """
        return staticmethod(func)
