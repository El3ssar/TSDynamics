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

Present today (migrated here from the v2 ``backends`` package):

- :mod:`~tsdynamics.engine.rustcore` — the v2-seed tape emitter + accelerator
  wrappers (``tsdynamics-core``); superseded by ``compile``/``run`` and retired
  with the other v2 backends at milestone M3.
- :mod:`~tsdynamics.engine.diffsol` — the experimental DiffSL/pydiffsol bridge
  (``pip install tsdynamics[diffsol]``).  A v2-era backend, retired once the
  Rust sole-engine reaches parity (milestone M3).

All are imported lazily by their callers, so importing this package is cheap
and pulls in no optional dependency.
"""

__all__ = ["compile", "diffsol", "problem", "run", "rustcore"]
