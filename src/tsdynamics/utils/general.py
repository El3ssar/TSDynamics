from __future__ import annotations

from typing import Callable

try:
    from numba import njit
except ImportError:

    def njit(func):
        return func


def staticjit(func: Callable) -> Callable:
    """Decorator to apply numba's njit decorator to a static method"""
    return staticmethod(njit(func))
