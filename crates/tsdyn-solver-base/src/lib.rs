//! Shared infrastructure for timestepping crates on **Track E**.
//!
//! | Crate | Responsibility |
//! |-------|----------------|
//! | [`tsdyn_solver_base`](crate) (here) | Time-grid sampling and other **ODE/DDE-agnostic**
//! helpers. |
//! | **`tsdyn-ode`** | Initial-value ODE stepping on IR-evaluated RHS. |
//! | **`tsdyn-dde`** *(planned — N5)* | Delay histories + Hermite extension; consumes the same
//! sampling primitives for user-facing output grids. |
//!
//! New solver families should hook here first for anything that applies to multiple problem
//! classes (`uniform_time_grid`, future breakpoint schedulers).

mod time_grid;

pub use time_grid::uniform_time_grid;
