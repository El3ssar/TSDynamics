//! `tsdyn-jit` — the Cranelift [`Evaluator`](tsdyn_ir::Evaluator): it compiles an
//! IR [`Tape`](tsdyn_ir::Tape) to native machine code for large or hot problems,
//! implementing the *same* [`Evaluator`] trait as the `tsdyn-vm` interpreter
//! (ROADMAP §3 / D2).
//!
//! # What it is
//!
//! [`JitEvaluator::new`] walks a validated tape once and emits two native
//! functions with Cranelift — `eval` (`du/dt`, the explicit-solver path) and
//! `eval_jac` (`du/dt` **and** the analytic Jacobian in one pass, the
//! implicit/stiff path) — then hands back finalized function pointers. The
//! tape's single-static-assignment registers become Cranelift SSA values, so the
//! JIT has *no* register file: [`n_scratch`](JitEvaluator::n_scratch) is `0` and
//! the trait's `scratch` slice is ignored. Built once, the evaluator is [`Sync`]
//! and shared by every rayon ensemble worker — never recompiled per trajectory.
//!
//! # Pure Rust, no LLVM
//!
//! Codegen is [Cranelift](https://cranelift.dev) — pure Rust, no LLVM — so the
//! shipped wheels stay trivial to build (D2). The crate pulls no `llvm-sys` and
//! needs no system toolchain beyond `cargo`.
//!
//! # Numerical contract: the interpreter is the reference
//!
//! The JIT must agree with the interpreter. It does so **exactly**, not merely to
//! a tolerance: arithmetic (`Add`/`Sub`/`Mul`/`Div`/`Neg`/`Recip`), `Sqrt` and
//! `Abs` lower to Cranelift's IEEE-754 instructions — bit-identical to the same
//! Rust operators the interpreter uses — and every transcendental, `Pow`, `Powi`
//! and `Sign` lowers to a host call into the *same* `std`/`libm` function the
//! interpreter calls (`f64::sin`, `f64::powf`, …). The result is bit-for-bit
//! equality with the interpreter on every tape (proven by the
//! `interpreter_equivalence` fuzz test via [`f64::to_bits`]), which is strictly
//! stronger than stream E2's `~1e-12` acceptance bar. The two evaluators are
//! exercised against the F1 golden fixtures (real emitted tapes vs the symbolic
//! ground truth) by `golden_fixtures`.
//!
//! [`Evaluator`]: tsdyn_ir::Evaluator

mod codegen;
mod error;
mod evaluator;
mod shims;

pub use error::JitError;
pub use evaluator::JitEvaluator;
