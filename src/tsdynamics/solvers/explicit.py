"""In-tree specs for the explicit Runge–Kutta solver family (stream E3).

One :class:`~tsdynamics.solvers.SolverSpec` per Rust kernel in
``crates/tsdyn-solvers/src/explicit/`` — the Python-side *mirror* of the
link-time registry that crate self-populates with ``register_solver!``.  A spec
carries no implementation; its :attr:`~tsdynamics.solvers.SolverSpec.kernel`
names the Rust kernel the engine dispatches to (here, the kernel name *is* the
``method=`` name).

Importing this module registers the specs (the F2 directory scan imports it),
so dropping a kernel here makes it resolvable by ``method=`` with no central
table to edit (ROADMAP §4d).  The registry-parity test in ``tests/test_solvers.py``
asserts these stay in lock-step with the Rust ``register_solver!`` lines.
"""

from __future__ import annotations

from . import SolverCaps, SolverSpec, register

#: The families an explicit RK kernel integrates natively (the Rust caps say
#: ``ProblemKind::Ode``).  The DDE method-of-steps reuses these ODE stage
#: integrators, but resolves them as ``family="ode"`` (ROADMAP E-DDE), so the
#: mirror stays faithful to the Rust ``supports`` set.
_ODE = frozenset({"ode"})

#: ``name -> (description, adaptive)`` for the explicit family.  Order/precision
#: documented in ``crates/tsdyn-solvers/src/explicit/mod.rs``.
_EXPLICIT: dict[str, tuple[str, bool]] = {
    "rk4": ("classic 4th-order Runge–Kutta (fixed step)", False),
    "rk45": ("Dormand–Prince 5(4) adaptive (dopri5)", True),
    "tsit5": ("Tsitouras 5(4) adaptive", True),
    "dop853": ("Dormand–Prince 8(5,3) adaptive", True),
}

for _name, (_desc, _adaptive) in _EXPLICIT.items():
    register(
        SolverSpec(
            name=_name,
            caps=SolverCaps(kind="explicit", adaptive=_adaptive, supports=_ODE),
            description=_desc,
        )
    )
