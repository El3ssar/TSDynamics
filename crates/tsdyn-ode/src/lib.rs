//! Pure-Rust ODE integration on a uniform output time grid (**N2**).
//!
//! This crate bundles explicit adaptive Runge–Kutta methods with dense Hermite-type
//! output on the user grid, classical **RK4** between output points, and
//! Rosenbrock–Wanner integrators (**Rosenbrock23**, **Rosenbrock34**, **Rodas4**) that
//! consume [`CompiledOde`](tsdyn_core::ir::CompiledOde) Jacobian bytecode.
//!
//! Python goes through [`integrate_ode_bytes`] in `crates/tsdyn-native`; Rust tests typically import [`integrate_ode`].
//!
//! Organisation:
//! - Internal ``methods`` module: tableau data + staged integrators.
//! - Internal ``driver`` module: ``method → integrator`` dispatch.
//! Output sampling uses [`uniform_time_grid`](tsdyn_solver_base::uniform_time_grid); **N5** DDE crates will reuse it.

mod controller;
mod driver;
pub mod error;
pub mod method;
mod methods;
pub mod rhs;
mod step_helpers;
mod util;

pub use driver::{integrate_ode, integrate_ode_bytes};
pub use error::IntegrateError;
pub use method::Method;
pub use rhs::{IrOdeRhs, Rhs};
