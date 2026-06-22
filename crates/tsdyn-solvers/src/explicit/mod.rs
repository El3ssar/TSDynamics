//! The explicit Runge–Kutta solver family (stream E3).
//!
//! One file per method, each ending in a [`register_solver!`](crate::register_solver)
//! so it is discoverable by name with no central table to edit (ROADMAP §4d):
//!
//! | name | method | order | adaptive |
//! |------|--------|-------|----------|
//! | `rk4` | classic Runge–Kutta | 4 | no (fixed step) |
//! | `rk45` | Dormand–Prince 5(4) (`dopri5`) | 5 | yes |
//! | `tsit5` | Tsitouras 5(4) | 5 | yes |
//! | `dop853` | Dormand–Prince 8(5,3) | 8 | yes |
//!
//! All four share the tableau-walking and step-control machinery in
//! [`control`] — each method file is just its Butcher coefficients plus a thin
//! [`Solver`](crate::Solver) impl, so the algorithm that could hide a
//! transcription error is written and tested exactly once.
//!
//! # Tolerances and the frozen `step` signature
//!
//! The adaptive kernels (`rk45`, `tsit5`, `dop853`) own their `rtol`/`atol`
//! because the frozen [`Solver::step`](crate::Solver::step) takes none — the
//! registry factory builds a default-tolerance instance (SciPy's `1e-3`/`1e-6`)
//! and a configured instance is built through each kernel's `with_tolerances`
//! constructor.  Threading a user's `rtol`/`atol` from Python onto a
//! registry-selected kernel is the engine/`solvers` layer's job (streams E5 /
//! C-SOLV); the `Solver` trait deliberately stays minimal and is **not** widened
//! here.

mod adams;
mod bs3;
mod cashkarp;
mod control;
mod dop853;
mod euler;
mod heun;
mod heun_euler;
mod midpoint;
mod ralston;
mod rk4;
mod rk45;
mod rk4_38;
mod rkf45;
mod ssprk3;
mod tsit5;

#[cfg(test)]
mod testkit;

pub use adams::{Ab3, Ab4, Abm4};
pub use bs3::Bs3;
pub use cashkarp::CashKarp;
pub use dop853::Dop853;
pub use euler::Euler;
pub use heun::Heun;
pub use heun_euler::HeunEuler;
pub use midpoint::Midpoint;
pub use ralston::Ralston;
pub use rk4::Rk4;
pub use rk45::Rk45;
pub use rk4_38::Rk438;
pub use rkf45::Rkf45;
pub use ssprk3::SspRk3;
pub use tsit5::Tsit5;
