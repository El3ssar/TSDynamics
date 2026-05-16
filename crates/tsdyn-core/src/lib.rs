//! Shared types and IR scaffolding for the TSDynamics Rust crates.
//!
//! Today this crate hosts the symbolic IR shared between every kernel crate
//! (currently `tsdyn-maps`; later `tsdyn-ode`, `tsdyn-dde`, `tsdyn-sweep`).
//! The `Expr` enum and bytecode format are stable interfaces — Python emits
//! the bytecode, Rust decodes once per system, and every kernel evaluates
//! against the same op set.

pub mod ir;

/// Opaque handle to a compiled right-hand-side / step function.
///
/// Placeholder kept for R1's smoke-test plumbing; later N-milestones may
/// replace it with a richer wrapper around either a cranelift-JIT'd function
/// (post-N4) or a [`ir::CompiledMap`] for discrete maps.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProblemHandle;
