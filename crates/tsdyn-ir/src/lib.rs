//! `tsdyn-ir` — the instruction-tape IR: the frozen contract between the
//! Python `engine.compile` layer and every Rust `Evaluator`.
//!
//! This crate owns the opcode set, the `Tape` data structure (ops, args,
//! immediates, outputs, jac_outputs, n_state, n_param) and its builder, plus
//! the `Evaluator` trait (the seam D2 hangs on) that `tsdyn-vm` and `tsdyn-jit`
//! both implement. The v2 tape semantics in `tsdynamics-core/src/vm.rs` migrate
//! here unchanged (must still match the symbolic RHS to 1e-15).
//!
//! Skeleton only (stream F0). Contract lands in **stream F1**; the trait
//! definitions in **stream F2**. See ROADMAP §4a.
