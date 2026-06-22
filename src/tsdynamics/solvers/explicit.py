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
    # Fixed-step kernels (no embedded error estimate; the user controls the step).
    "euler": ("forward (explicit) Euler (order 1, fixed step)", False),
    "midpoint": ("explicit midpoint / modified Euler (order 2, fixed step)", False),
    "heun": ("Heun's method / explicit trapezoid (order 2, fixed step)", False),
    "ralston": ("Ralston's minimum-error-bound RK2 (order 2, fixed step)", False),
    "rk4": ("classic 4th-order Runge–Kutta (fixed step)", False),
    "rk4_38": ("the 3/8-rule 4th-order Runge–Kutta (fixed step)", False),
    "ssprk3": ("3rd-order strong-stability-preserving RK (Shu–Osher, fixed step)", False),
    # Explicit linear-multistep (Adams) kernels — fixed step, RK4 self-start.
    "ab3": ("Adams–Bashforth 3-step explicit multistep (order 3)", False),
    "ab4": ("Adams–Bashforth 4-step explicit multistep (order 4)", False),
    "abm4": ("Adams–Bashforth–Moulton predictor–corrector (PECE, order 4)", False),
    # Adaptive embedded pairs (own error control + step adaption).
    "heun_euler": ("Heun–Euler 2(1) adaptive", True),
    "bs3": ("Bogacki–Shampine 3(2) adaptive (ode23)", True),
    "rk45": ("Dormand–Prince 5(4) adaptive (dopri5)", True),
    "rkf45": ("Runge–Kutta–Fehlberg 4(5) adaptive", True),
    "cashkarp": ("Cash–Karp 5(4) adaptive", True),
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
