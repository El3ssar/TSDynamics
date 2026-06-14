//! `tsdyn-solvers` — the [`Solver`] trait and the solver name registry, plus
//! (in later streams) one module per integration kernel.
//!
//! This crate owns the second of the two engine seams (the first,
//! [`Evaluator`], lives in `tsdyn-ir`): the [`Solver`] trait a kernel implements
//! and the [`register_solver!`] mechanism a kernel uses to make itself
//! discoverable by name — with **no central dispatch table** (ROADMAP §4d), so
//! the explicit (E3), implicit/stiff (E4) and SDE (E-SDE) families can be built
//! in parallel without ever editing a shared file.
//!
//! # What stream F2 freezes here
//!
//! - [`Solver`] — the kernel trait (`name`/`caps`/`step`), object-safe so the
//!   engine drives `&mut dyn Solver` chosen by a `method=` string.
//! - [`Caps`] / [`ProblemKind`] / [`ProblemKinds`] / [`SolverKind`] — the
//!   capability metadata driving `method=` resolution and auto-stiffness.
//! - [`SolverState`] / [`StepOutcome`] — the per-worker integration state and
//!   the step/accept/retry/fail outcome the engine loops on.
//! - [`SolverRegistration`] + [`register_solver!`] + [`find`]/[`make`]/
//!   [`registered`]/[`available`]/[`duplicates`] — the link-time auto-registry
//!   (via [`inventory`]).
//!
//! [`Evaluator`] is re-exported so a kernel author imports both seams from one
//! crate.
//!
//! # What later streams add
//!
//! The actual kernels: `explicit/` (E3 — RK4, DP45/RK45, DOP853, Tsit5…),
//! `implicit/` (E4 — Rosenbrock, SDIRK/TR-BDF2, BDF), `sde/` (E-SDE —
//! Euler–Maruyama, Milstein).  Each is one file ending in a `register_solver!`.

mod caps;
mod registry;
mod solver;

// Solver-kernel families: one module per family, each filled by its own stream
// (E3 explicit; E4 implicit; E-SDE stochastic). Append-only — add a family line,
// never reorder. Each kernel inside self-registers via `register_solver!`.
pub mod explicit;

pub use caps::{Caps, ProblemKind, ProblemKinds, SolverKind};
pub use registry::{available, duplicates, find, make, registered, SolverRegistration};
pub use solver::{Solver, SolverState, StepOutcome};

// The Evaluator seam lives in tsdyn-ir; re-export it so kernel authors get both
// halves of the engine contract from this one crate.
pub use tsdyn_ir::Evaluator;

// Re-export `inventory` at the crate root so `register_solver!` can expand to
// `$crate::inventory::submit!` — the registering crate then needs only a
// dependency on `tsdyn-solvers`, not on `inventory` directly. Not a stable API.
#[doc(hidden)]
pub use inventory;
