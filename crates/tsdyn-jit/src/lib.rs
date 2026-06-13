//! `tsdyn-jit` — the Cranelift `Evaluator`: compiles an IR tape to native code
//! for large or hot problems, implementing the *same* `Evaluator` trait as
//! `tsdyn-vm`.
//!
//! Pure-Rust codegen (no LLVM) so wheels stay trivial to build (D2). The
//! interpreter is the reference: a mandatory eval-equality test (~1e-12 across
//! the catalogue) guards against numerical drift between the two paths.
//!
//! Skeleton only (stream F0). Implementation — and the cranelift dependencies —
//! land in **stream E2**. See ROADMAP §4a.
