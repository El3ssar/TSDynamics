//! `tsdyn-vm` — the interpreter evaluator: a stack machine that walks an IR
//! [`Tape`](tsdyn_ir::Tape) in a single allocation-free, GIL-free pass.
//!
//! This is the default, zero-warmup evaluation path — sweeps, tests, and small
//! to medium systems — and the numerical reference the Cranelift JIT
//! (`tsdyn-jit`) must match. It generalises the v2 `tsdynamics-core` stack
//! machine into a reusable evaluator over the frozen `tsdyn-ir` contract, with a
//! method surface that mirrors the engine's object-safe `Evaluator` seam
//! (stream F2): [`dim`](Interpreter::dim), [`n_param`](Interpreter::n_param),
//! [`n_scratch`](Interpreter::n_scratch), [`has_jacobian`](Interpreter::has_jacobian),
//! [`eval`](Interpreter::eval), [`eval_jac`](Interpreter::eval_jac).
//!
//! # Using it
//!
//! ```
//! use tsdyn_ir::TapeBuilder;
//! use tsdyn_vm::Interpreter;
//!
//! // dx/dt = p0 * (u1 - u0)
//! let mut b = TapeBuilder::new();
//! let k = b.param(0);
//! let x = b.state(0);
//! let y = b.state(1);
//! let ymx = b.sub(y, x);
//! let dx = b.mul(k, ymx);
//! let tape = b.finish(&[dx], &[], 2, 1).unwrap();
//!
//! let interp = Interpreter::new(tape);
//! assert_eq!(interp.eval_alloc(&[1.0, 4.0], &[2.0], 0.0), vec![6.0]);
//! ```
//!
//! For the hot path, allocate one scratch register file (length
//! [`n_scratch`](Interpreter::n_scratch)) and reuse it across steps:
//!
//! ```
//! # use tsdyn_ir::TapeBuilder;
//! # use tsdyn_vm::Interpreter;
//! # let mut b = TapeBuilder::new();
//! # let x = b.state(0);
//! # let dx = b.sin(x);
//! # let interp = Interpreter::new(b.finish(&[dx], &[], 1, 0).unwrap());
//! let mut scratch = vec![0.0; interp.n_scratch()];
//! let mut deriv = vec![0.0; interp.dim()];
//! interp.eval(&[0.5], &[], 0.0, &mut scratch, &mut deriv);
//! ```
//!
//! # Stream boundaries (ROADMAP §4a)
//!
//! This crate (stream **E1**) owns the interpreter and builds only against the
//! frozen F1 IR. The `Evaluator`/`Solver` **traits** and their registry are
//! stream **F2**; the [`Interpreter`]'s inherent methods already match the F2
//! `Evaluator` signature, so the eventual `impl Evaluator for Interpreter` is a
//! one-line forward and does not belong here yet. See ROADMAP §4a.

mod interp;

pub use interp::Interpreter;
