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
//!
//! Both estimate the local error by **step doubling + Richardson extrapolation**
//! (the shared [`control`] machinery), so they drive the engine's frozen
//! accept/reject loop without a per-method embedded estimator. Each lives in its
//! own file ending in a `register_solver!` line and is discovered by name through
//! the registry — no central table (ROADMAP §4d), so this family was built
//! without touching the explicit (E3) or SDE (E-SDE) families.
//!
//! # Not yet here
//!
//! A variable-order **BDF** (multistep) kernel is the remaining item from the
//! stream's scope; it carries its own history across steps and is best added as
//! its own file in this directory once the one-step kernels are validated. The
//! acceptance benchmarks (stiff linear, Van der Pol, Robertson, Oregonator) are
//! met by the two one-step kernels above.

mod control;
mod linalg;

pub mod rosenbrock;
pub mod trbdf2;

pub use rosenbrock::RosenbrockW;
pub use trbdf2::TrBdf2;
