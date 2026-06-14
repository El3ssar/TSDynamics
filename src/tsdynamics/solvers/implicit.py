"""In-tree specs for the implicit / stiff solver family (stream E4).

One :class:`~tsdynamics.solvers.SolverSpec` per Rust kernel in
``crates/tsdyn-solvers/src/implicit/`` — the L-stable, analytic-Jacobian
kernels the auto-stiffness layer selects on a stiff RHS.  Both declare
:attr:`~tsdynamics.solvers.SolverCaps.needs_jacobian`, which is exactly the
signal :func:`tsdynamics.solvers.build_kwargs` turns into ``with_jacobian=True``
so the engine's Jacobian guard (PR #74) is satisfied instead of raising.

Importing this module registers the specs (the F2 directory scan imports it);
the registry-parity test keeps them in lock-step with the Rust
``register_solver!`` lines.
"""

from __future__ import annotations

from . import SolverCaps, SolverSpec, register

_ODE = frozenset({"ode"})

#: ``name -> description`` for the stiff family.  Both are adaptive (step
#: doubling + Richardson) and need the analytic Jacobian (see
#: ``crates/tsdyn-solvers/src/implicit/mod.rs``).
_IMPLICIT: dict[str, str] = {
    "rosenbrock": "linearly-implicit Rosenbrock-W (one linear solve per step)",
    "trbdf2": "TR-BDF2 composite ESDIRK (trapezoidal + BDF2)",
}

for _name, _desc in _IMPLICIT.items():
    register(
        SolverSpec(
            name=_name,
            caps=SolverCaps(kind="implicit", adaptive=True, needs_jacobian=True, supports=_ODE),
            description=_desc,
        )
    )
