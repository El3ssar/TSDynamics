"""In-tree specs for the stochastic (SDE) solver family (stream E-SDE).

One :class:`~tsdynamics.solvers.SolverSpec` per diagonal-Itô kernel in
``crates/tsdyn-solvers/src/sde/`` — the Python mirror of the
``register_sde_kernel!`` registry.  Both are explicit; Milstein additionally
reads the diffusion Jacobian ``∂g/∂u`` and so declares
:attr:`~tsdynamics.solvers.SolverCaps.needs_jacobian` (which
:func:`tsdynamics.solvers.build_kwargs` turns into ``with_jacobian=True`` on the
diffusion tape).

These are the ``method=`` names
:meth:`~tsdynamics.families.stochastic.StochasticSystem.integrate` accepts
(``"euler_maruyama"`` / ``"milstein"``).  Importing this module registers them
(the F2 directory scan imports it); the registry-parity test keeps them in
lock-step with the Rust ``register_sde_kernel!`` lines.
"""

from __future__ import annotations

from . import SolverCaps, SolverSpec, register

_SDE = frozenset({"sde"})

#: ``name -> (description, needs_jacobian)`` for the diagonal-Itô family.  Strong
#: orders documented in ``crates/tsdyn-solvers/src/sde/mod.rs``.
_SDE_KERNELS: dict[str, tuple[str, bool]] = {
    "euler_maruyama": ("Euler–Maruyama diagonal-Itô (strong order 0.5)", False),
    "milstein": ("Milstein diagonal-Itô (strong order 1.0; uses ∂g/∂u)", True),
}

for _name, (_desc, _needs_jac) in _SDE_KERNELS.items():
    register(
        SolverSpec(
            name=_name,
            caps=SolverCaps(kind="explicit", needs_jacobian=_needs_jac, supports=_SDE),
            description=_desc,
        )
    )
