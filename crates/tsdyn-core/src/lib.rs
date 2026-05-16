//! Shared types and IR scaffolding for the TSDynamics Rust crates.
//!
//! This crate is intentionally empty for milestone R1. It exists as a hook in
//! the Cargo workspace so that future kernel crates (sweep, recurrence, basin,
//! …) can depend on a single source of shared types. The first real inhabitant
//! will be `ProblemHandle`, an opaque wrapper around either a JiTCODE-cffi
//! function pointer (Track E early phase) or a cranelift-JIT'd native function
//! (after N4).

/// Opaque handle to a compiled right-hand-side / step function.
///
/// Placeholder for R1: it carries no data. Analysis kernels in later
/// milestones will consume `ProblemHandle` so they don't have to change when
/// the underlying compilation pipeline is replaced.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProblemHandle;
