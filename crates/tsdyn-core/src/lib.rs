//! Shared types and IR scaffolding for the TSDynamics Rust crates.
//!
//! Today this crate hosts the symbolic IR shared between every kernel crate
//! (currently `tsdyn-maps`; later `tsdyn-ode`, `tsdyn-dde`, `tsdyn-sweep`).
//! The `Expr` enum and bytecode format are stable interfaces — Python emits
//! the bytecode, Rust decodes once per system, and every kernel evaluates
//! against the same op set.

pub mod ir;

/// Opaque handle to a compiled RHS / step function — reserved for N4.
///
/// N4 (Cranelift JIT) will fill this with a wrapper around a
/// cranelift-compiled native function pointer.  At that point
/// `ProblemHandle` replaces the current IR interpreter path in every
/// stepper; Python holds one per system as an opaque capsule via PyO3.
///
/// Until N4 lands this struct exists only to give N4 a stable Python
/// boundary to target — no milestones before N4 need to construct it.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProblemHandle;
