"""Utility decorators shared across TSDynamics modules."""

from __future__ import annotations

from collections.abc import Callable

try:
    from numba import njit
except ImportError:

    def njit(func: Callable) -> Callable:
        """Return the function unchanged (numba not available)."""
        return func


def staticjit(func: Callable) -> Callable:
    """Apply numba's ``njit`` and ``staticmethod`` to a map method."""
    return staticmethod(njit(func))
