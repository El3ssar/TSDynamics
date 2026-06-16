"""Engine-native symbolic state/time symbols (the symbolic-frontend provider).

A system's ``_equations`` / ``_drift`` / ``_diffusion`` is written against a
callable state symbol ``y`` and a time symbol ``t``: ``y(i)`` is the current
``i``-th component and, for a DDE, ``y(i, t - τ)`` is a delayed access.  Before
the M3 migration these came from JiTCODE / JiTCDDE; the compile layer then
substitutes ``{y(i): u_i}`` and lowers the result to the IR tape.

Since the systems only *call* ``y`` (they never depend on what it is), the
provider is just ``symengine.Function("y")`` and ``symengine.Symbol("t")`` —
``y(i)`` is a one-argument ``FunctionSymbol`` named ``y`` (byte-identical to the
role JiTCODE's ``y`` played) and ``y(i, t - τ)`` is the two-argument delayed
form the DDE lowering recognises by arity.  This makes the symbolic frontend
self-contained: the engine no longer borrows symbols from a v2 backend.
"""

from __future__ import annotations

from typing import Any


def state_time_symbols() -> tuple[Any, Any]:
    """Return ``(y, t)``: the callable state symbol and the time symbol.

    ``y(i)`` → current component ``i``; ``y(i, t - τ)`` → a delayed access.
    """
    import symengine

    return symengine.Function("y"), symengine.Symbol("t")
