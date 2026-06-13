//! `tsdyn-vm` — the interpreter `Evaluator`: a stack machine that walks an IR
//! tape in a single allocation-free, GIL-free pass.
//!
//! This is the default, zero-warmup evaluation path (sweeps, tests, small and
//! medium systems). It is the numerical reference the Cranelift JIT (`tsdyn-jit`)
//! must match. The today's stack machine in `tsdynamics-core/src/vm.rs`
//! generalizes into the `Evaluator` impl here.
//!
//! Skeleton only (stream F0). Implementation lands in **stream E1**.
//! See ROADMAP §4a.
