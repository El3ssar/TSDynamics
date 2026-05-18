"""Internal package for Rust-backed extension modules.

User code never imports from here directly. The public Python API in
:mod:`tsdynamics` re-exports anything that is meant to be user-facing.
This package exists so the single Rust extension (``_core``) compiled by
maturin has a stable installation target and the rest of the library can
route hot loops through one facade.

Contents grow incrementally as Rust kernels land (see
``.planning/ROADMAP.md`` Track C / Track E):

- N1: ``iterate_map`` / ``lyapunov_spectrum_map`` — discrete-map kernels.
- N2: ``eval_ode_rhs`` / ``integrate_ode`` / ``lyapunov_spectrum_ode`` — ODE stepper.
- R2+: future Track-C kernels (sweep, recurrence, …) register here too.
"""

from __future__ import annotations

from ._core import (
    eval_ode_jacobian,
    eval_ode_rhs,
    eval_ode_rhs_batch,
    integrate_ode,
    iterate_map,
    lyapunov_spectrum_map,
    lyapunov_spectrum_ode,
)

__all__ = [
    "eval_ode_jacobian",
    "eval_ode_rhs",
    "eval_ode_rhs_batch",
    "integrate_ode",
    "iterate_map",
    "lyapunov_spectrum_map",
    "lyapunov_spectrum_ode",
]
