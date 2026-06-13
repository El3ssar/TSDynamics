"""
The Rust-facing engine layer.

This package is the seam between the symbolic system definitions and the
compiled Rust engine (the ``tsdyn-*`` Cargo workspace).  Per the v3 design it
will host the symbolic→IR compiler, problem builders, and the integrate/run
entry points (``compile.py`` / ``problem.py`` / ``run.py``, stream E6); the
PyO3 extension is reached as :mod:`tsdynamics._rust` (stream E7).

Present today (migrated here from the v2 ``backends`` package):

- :mod:`~tsdynamics.engine.rustcore` — lowers a system's symbolic RHS (and
  analytic Jacobian) to the instruction tape consumed by the Rust evaluator.
- :mod:`~tsdynamics.engine.diffsol` — the experimental DiffSL/pydiffsol bridge
  (``pip install tsdynamics[diffsol]``).  A v2-era backend, retired once the
  Rust sole-engine reaches parity (milestone M3).

Both are imported lazily by their callers, so importing this package is cheap
and pulls in no optional dependency.
"""

__all__ = ["diffsol", "rustcore"]
