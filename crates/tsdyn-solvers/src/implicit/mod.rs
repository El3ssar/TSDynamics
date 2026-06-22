//! `implicit` — the implicit / stiff solver family (stream E4).
//!
//! Stiff systems have widely separated time scales: an explicit method must take
//! steps bounded by the *fastest* (often already-decayed) mode to stay stable,
//! crawling even where the solution is smooth. The kernels here are **L-stable**
//! and use the **analytic Jacobian** `∂f/∂u` the evaluator carries (from the
//! symbolic tape, exact — no finite differences), so they take steps set by
//! accuracy rather than stability and integrate stiff problems efficiently.
//!
//! # The kernels
//!
//! - [`RosenbrockW`] (`"rosenbrock"`) — a linearly-implicit Euler / one-stage
//!   W-method: one linear solve per step, no Newton iteration. Cheap and robust.
//! - [`TrBdf2`] (`"trbdf2"`) — a trapezoidal + BDF2 composite ESDIRK with
//!   modified-Newton substages. Stiffly accurate; a different mechanism from the
//!   Rosenbrock kernel, so the two cross-validate each other.
//! - [`Bdf`] (`"bdf"`) — a **variable-order (1–5), variable-step** fixed-leading-
//!   coefficient BDF (stream E-BDF). It adapts its *order* as well as its step, so
//!   it takes far larger steps than the fixed-order kernels through a smooth stiff
//!   phase — the high-order stiff workhorse that closes the warm-throughput gap to
//!   a variable-order BDF reference (`benches/REPORT.md`).
//!
//! The two one-step kernels estimate the local error by **step doubling +
//! Richardson extrapolation** (the shared [`control`] machinery); the multistep
//! [`Bdf`] carries its own difference-array history and a native order/step
//! controller instead. All three drive the engine's frozen accept/reject loop and
//! live in their own file ending in a `register_solver!` line, discovered by name
//! through the registry — no central table (ROADMAP §4d), so this family was built
//! without touching the explicit (E3) or SDE (E-SDE) families.
//!
//! # Possible future work
//!
//! A Radau IIA collocation kernel (higher-order, A-stable, dense-output friendly)
//! would round out the stiff family; it slots in as another file here behind the
//! same `Solver` seam.

mod control;
mod linalg;
mod newton;

pub mod backward_euler;
pub mod bdf;
pub mod implicit_midpoint;
pub mod rosenbrock;
pub mod sdirk2;
pub mod trapezoid;
pub mod trbdf2;

pub use backward_euler::BackwardEuler;
pub use bdf::Bdf;
pub use implicit_midpoint::ImplicitMidpoint;
pub use rosenbrock::RosenbrockW;
pub use sdirk2::Sdirk2;
pub use trapezoid::Trapezoid;
pub use trbdf2::TrBdf2;
