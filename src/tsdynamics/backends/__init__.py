"""
Alternative numerical backends (experimental).

The default integration path compiles through JiTCODE (symbolic → C → .so).
This package hosts experimental alternatives — currently
:mod:`~tsdynamics.backends.diffsol`, a Rust solver suite reached through a
runtime LLVM JIT, installable via ``pip install tsdynamics[diffsol]``.
"""

__all__ = ["diffsol"]
