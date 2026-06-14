//! `tsdyn-ir` — the instruction-tape IR: the frozen contract between the Python
//! `engine.compile` layer and every Rust evaluator.
//!
//! A system's symbolic right-hand side (and optionally its analytic Jacobian)
//! lowers to a flat **instruction tape** in single-static-assignment form: a
//! list of primitive [`Op`]s over a register file, already common-subexpression
//! shared.  Evaluating a tape is one linear pass with a reusable scratch buffer
//! — no Python callback, GIL-free — which is what lets ensembles run in parallel
//! under rayon.  This crate owns the data contract; the evaluators that walk it
//! ([`tsdyn-vm`] interpreter, [`tsdyn-jit`] Cranelift) live in their own crates.
//!
//! # What this crate provides
//!
//! - [`Op`] — the opcode set, with wire values fixed by the v2 contract.
//! - [`Tape`] — the validated parallel-array instruction tape, with
//!   [`Tape::from_arrays`] (the FFI ingestion path) and slice accessors for
//!   zero-cost evaluation downstream.
//! - [`TapeBuilder`] — a typed builder mirroring the Python emitter, for
//!   hand-written and in-Rust-lowered tapes.
//! - [`IrError`] — the validation failure type.
//! - [`Evaluator`] — the object-safe trait every production evaluator
//!   (`tsdyn-vm`, `tsdyn-jit`) implements; one of the two pluggability seams
//!   (the other is `Solver` in `tsdyn-solvers`).  Added by stream **F2**.
//! - `reference` (off-by-default feature) — a canonical reference evaluator that
//!   *defines* the operational semantics and serves as an oracle for the
//!   production evaluators; see [`mod@reference`].
//!
//! # Opcode reference
//!
//! | Wire | [`Op`] | Kind | Semantics |
//! |-----:|--------|------|-----------|
//! | 0  | `Const` | leaf   | push `imm[i]` |
//! | 1  | `State` | leaf   | push `u[a]` |
//! | 2  | `Param` | leaf   | push `p[a]` |
//! | 3  | `Time`  | leaf   | push `t` |
//! | 10 | `Add`   | binary | `regs[a] + regs[b]` |
//! | 11 | `Sub`   | binary | `regs[a] - regs[b]` |
//! | 12 | `Mul`   | binary | `regs[a] * regs[b]` |
//! | 13 | `Div`   | binary | `regs[a] / regs[b]` |
//! | 14 | `Pow`   | binary | `regs[a].powf(regs[b])` |
//! | 15 | `Powi`  | powi   | `regs[a].powi(b)` — `b` is the integer exponent |
//! | 20 | `Neg`   | unary  | `-regs[a]` |
//! | 21 | `Recip` | unary  | `1 / regs[a]` |
//! | 30 | `Sin`   | unary  | `regs[a].sin()` |
//! | 31 | `Cos`   | unary  | `regs[a].cos()` |
//! | 32 | `Tan`   | unary  | `regs[a].tan()` |
//! | 33 | `Exp`   | unary  | `regs[a].exp()` |
//! | 34 | `Log`   | unary  | `regs[a].ln()` |
//! | 35 | `Sqrt`  | unary  | `regs[a].sqrt()` |
//! | 36 | `Abs`   | unary  | `regs[a].abs()` |
//! | 37 | `Sign`  | unary  | `sign(regs[a])`, with `sign(0) = 0` |
//! | 38 | `Sinh`  | unary  | `regs[a].sinh()` |
//! | 39 | `Cosh`  | unary  | `regs[a].cosh()` |
//! | 40 | `Tanh`  | unary  | `regs[a].tanh()` |
//! | 41 | `Asin`  | unary  | `regs[a].asin()` |
//! | 42 | `Acos`  | unary  | `regs[a].acos()` |
//! | 43 | `Atan`  | unary  | `regs[a].atan()` |
//! | 44 | `Asinh` | unary  | `regs[a].asinh()` |
//! | 45 | `Acosh` | unary  | `regs[a].acosh()` |
//! | 46 | `Atanh` | unary  | `regs[a].atanh()` |
//! | 50 | `Lt`    | binary | `(regs[a] <  regs[b]) as f64` |
//! | 51 | `Le`    | binary | `(regs[a] <= regs[b]) as f64` |
//! | 52 | `Gt`    | binary | `(regs[a] >  regs[b]) as f64` |
//! | 53 | `Ge`    | binary | `(regs[a] >= regs[b]) as f64` |
//! | 54 | `Eq`    | binary | `(regs[a] == regs[b]) as f64` |
//! | 55 | `Ne`    | binary | `(regs[a] != regs[b]) as f64` |
//! | 56 | `Min`   | binary | `regs[a].min(regs[b])` (`f64::min`) |
//! | 57 | `Max`   | binary | `regs[a].max(regs[b])` (`f64::max`) |
//! | 58 | `Floor` | unary  | `regs[a].floor()` |
//! | 59 | `Ceil`  | unary  | `regs[a].ceil()` |
//! | 60 | `Mod`   | binary | floored modulo `a - b*(a/b).floor()` |
//! | 61 | `Rem`   | binary | truncated remainder `regs[a] % regs[b]` |
//!
//! The wire values are the FFI contract: they are exactly what the Python
//! emitter writes and the v2 VM read.  See [`Tape`] for the per-instruction
//! `a`/`b`/`imm` layout.  Wire range **50-69** is the non-smooth / piecewise
//! block (stream **E-OPS**), reserved by ROADMAP §13d and added additively
//! without renumbering anything below it.
//!
//! # Stream boundaries (ROADMAP §4a)
//!
//! Stream **F1** froze the *data* contract — opcodes, tape, builder, validation.
//! Stream **F2** adds the first pluggability seam, the [`Evaluator`] trait, here
//! (so every downstream evaluator/solver/engine crate sees it without a new
//! build-graph edge); the matching `Solver` trait + the name/plugin registries
//! live in `tsdyn-solvers`.  The F1 [`reference`] evaluator stays plain
//! functions — it does not implement [`Evaluator`], so it remains a neutral
//! oracle rather than a privileged production impl.
//!
//! [`tsdyn-vm`]: https://docs.rs/tsdyn-vm
//! [`tsdyn-jit`]: https://docs.rs/tsdyn-jit

mod builder;
mod eval;
mod op;
mod tape;

#[cfg(feature = "reference")]
pub mod reference;

pub use builder::{Reg, TapeBuilder};
pub use eval::Evaluator;
pub use op::{Op, OpKind};
pub use tape::{IrError, Tape};
