"""
The Rust-facing engine layer.

This package is the seam between the symbolic system definitions and the
compiled Rust engine (the ``tsdyn-*`` Cargo workspace).  It hosts the
symbolic→IR compiler, problem builders, and the integrate/run entry points
(stream E6); the PyO3 extension is reached as :mod:`tsdynamics._rust` (stream
E7).

The E6 seam (symbolic definitions → engine):

- :mod:`~tsdynamics.engine.compile` — lowers a system's symbolic dynamics
  (ODE ``_equations`` + Jacobian, map ``_step``, DDE delayed ``_equations``,
  SDE ``_drift``/``_diffusion``) to the flat instruction tape (:class:`Tape`)
  consumed by the Rust evaluators; includes a pure-Python reference evaluator.
- :mod:`~tsdynamics.engine.problem` — bundles a compiled tape with its runtime
  context (ic, parameters, delays, …) into a per-family ``Problem``.
- :mod:`~tsdynamics.engine.run` — backend selection (``interp`` / ``jit`` /
  ``reference``) and the ``integrate`` / ``ensemble`` entry points.
- :mod:`~tsdynamics.engine.symbols` — the engine-native ``y`` / ``t`` symbolic
  state/time provider that ``_equations`` is written against.

The Rust engine is the sole integration backend (the v2 JiTCODE / JiTCDDE /
Numba / diffsol backends were retired at milestone M3).
"""

# Bind the four seam submodules onto the package so the advertised ``__all__``
# surface actually resolves (``tsdynamics.engine.run`` etc.) without the caller
# having to import each one separately.
from . import compile, problem, run, symbols  # noqa: E402, F401

__all__ = ["compile", "problem", "run", "symbols"]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
