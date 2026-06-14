//! The [`Evaluator`] trait — one of the two seams (with `Solver`) that make the
//! engine pluggable (ROADMAP §3 / D2).
//!
//! An evaluator turns a compiled [`Tape`](crate::Tape) into a callable
//! right-hand side `du/dt = f(u, p, t)` and — when the tape carries one — the
//! analytic Jacobian `∂f/∂u`.  Two implementations live in their own crates and
//! must agree numerically: the zero-warmup interpreter (`tsdyn-vm`, stream E1)
//! and the Cranelift JIT (`tsdyn-jit`, stream E2).  The solver kernels
//! (`tsdyn-solvers`) and the engine (`tsdyn-engine`) consume an evaluator
//! *through this trait* by dynamic dispatch (`&dyn Evaluator`), so it is
//! deliberately **object-safe** — no generic methods, no associated types.
//!
//! # Why this lives in `tsdyn-ir`
//!
//! The trait sits next to the tape it abstracts so that every downstream crate
//! (`tsdyn-vm`, `tsdyn-jit`, `tsdyn-solvers`, `tsdyn-engine`) — all of which
//! already depend on `tsdyn-ir` — sees one definition with no extra edge in the
//! build graph (ROADMAP §6, stream **F2**: "trait defs in `tsdyn-ir`/
//! `tsdyn-solvers`").  The F1 [`reference`](crate::reference) evaluator is plain
//! functions and intentionally does *not* implement this trait, so F1 froze the
//! data contract without pre-empting this seam.
//!
//! # Concurrency & scratch: the shared-evaluator / per-worker-scratch split
//!
//! The §4 trait *sketch* was `eval(&self, u, p, t, deriv)`.  The frozen
//! signature keeps `&self` but threads a caller-owned **scratch** slice, for two
//! reasons proven by the v2 engine:
//!
//! 1. **Zero allocation in the hot loop.** The v2 stack machine evaluates a tape
//!    into a reusable register file (`Workspace.regs` in `tsdynamics-core`); a
//!    step calls the RHS several times and allocates nothing.  We preserve that:
//!    the interpreter uses `scratch` as its register file, sized by
//!    [`n_scratch`](Evaluator::n_scratch).  The JIT has *no* register file — it
//!    reports `n_scratch() == 0` and ignores the (empty) slice — so the
//!    register-file detail never leaks into the trait's meaning while still
//!    keeping it object-safe (a typed `Workspace` would need an associated type,
//!    which is not object-safe).
//!
//! 2. **Build-once, share-many across rayon workers.** `&self` + the [`Sync`]
//!    supertrait means one evaluator is *built or JIT-compiled once* and shared
//!    by every ensemble worker; the only per-worker mutable state is the
//!    caller-owned `scratch` slice — exactly v2's per-worker `Workspace`.  This
//!    is essential for the JIT, which must compile once and share the resulting
//!    function, never recompile per trajectory.
//!
//! Because `eval` takes `&self`, a solver needs only `&dyn Evaluator` (not
//! `&mut`), which is why the `Solver::step(ev: &dyn Evaluator, …)` sketch in
//! ROADMAP §4 is preserved verbatim by stream F2.

/// A compiled right-hand side that can be evaluated, with optional Jacobian.
///
/// Implemented by the interpreter (`tsdyn-vm`) and the Cranelift JIT
/// (`tsdyn-jit`); consumed as `&dyn Evaluator` by solvers and the engine.  See
/// the [module docs](self) for the concurrency / scratch contract.
///
/// # Buffer contract
///
/// Every call writes into caller-owned buffers and allocates nothing:
///
/// - `u` — state, length [`dim`](Evaluator::dim) (the RHS reads it).
/// - `p` — parameters, length [`n_param`](Evaluator::n_param).
/// - `scratch` — working buffer of length [`n_scratch`](Evaluator::n_scratch);
///   its contents on entry are irrelevant and on exit unspecified.
/// - `deriv` — output `du/dt`, length [`dim`](Evaluator::dim).
/// - `jac` (in [`eval_jac`](Evaluator::eval_jac)) — row-major `dim × dim`
///   Jacobian `∂f_k/∂u_j` at `jac[k * dim + j]`, length `dim * dim`.
///
/// Implementations may rely on these lengths (debug builds should assert them);
/// passing a shorter slice is a caller bug, not an `Evaluator` error.
pub trait Evaluator: Sync {
    /// System dimension — the length of `u`, `deriv`, and `√(jac.len())`.
    fn dim(&self) -> usize;

    /// Declared parameter width — the expected length of `p`.
    fn n_param(&self) -> usize;

    /// Length of the `scratch` buffer [`eval`](Evaluator::eval) requires.
    ///
    /// The interpreter returns the tape's register count; the JIT returns `0`.
    /// Callers size one scratch buffer per worker from this and reuse it across
    /// every step (mirroring v2's `Workspace`).
    fn n_scratch(&self) -> usize;

    /// Whether [`eval_jac`](Evaluator::eval_jac) is supported (the tape carries
    /// a Jacobian).  Calling `eval_jac` when this is `false` is a caller bug.
    fn has_jacobian(&self) -> bool;

    /// Evaluate `du/dt` at `(u, p, t)` into `deriv`, using `scratch` as the
    /// working buffer.  Allocation-free; see the [buffer contract](Evaluator).
    fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]);

    /// Evaluate `du/dt` (into `deriv`) and the row-major `dim × dim` Jacobian
    /// `∂f/∂u` (into `jac`) in one pass.  Requires [`has_jacobian`] to be `true`.
    ///
    /// [`has_jacobian`]: Evaluator::has_jacobian
    fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    );
}

#[cfg(all(test, feature = "reference"))]
mod tests {
    //! These tests double as the executable proof that the frozen trait is
    //! object-safe and composes with the F1 reference semantics: a tiny
    //! tape-backed evaluator implemented *here* must reproduce
    //! [`reference::eval`](crate::reference::eval) exactly, and must be usable as
    //! `&dyn Evaluator`.
    use super::*;
    use crate::builder::TapeBuilder;
    use crate::{reference, Tape};

    /// A minimal interpreter over a tape — a stand-in for the real `tsdyn-vm`
    /// (E1), here only to exercise the trait seam from inside the IR crate.
    struct RefEval {
        tape: Tape,
    }

    impl Evaluator for RefEval {
        fn dim(&self) -> usize {
            self.tape.dim()
        }
        fn n_param(&self) -> usize {
            self.tape.n_param()
        }
        fn n_scratch(&self) -> usize {
            self.tape.n_reg()
        }
        fn has_jacobian(&self) -> bool {
            self.tape.has_jacobian()
        }
        fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
            reference::eval(&self.tape, u, p, t, scratch, deriv);
        }
        fn eval_jac(
            &self,
            u: &[f64],
            p: &[f64],
            t: f64,
            scratch: &mut [f64],
            deriv: &mut [f64],
            jac: &mut [f64],
        ) {
            reference::eval_jac(&self.tape, u, p, t, scratch, deriv, jac);
        }
    }

    fn lorenz() -> Tape {
        // dx = sigma (y - x); dy = x (rho - z) - y; dz = x y - beta z
        let mut b = TapeBuilder::new();
        let (sg, rho, bt) = (b.param(0), b.param(1), b.param(2));
        let (x, y, z) = (b.state(0), b.state(1), b.state(2));
        let ymx = b.sub(y, x);
        let dx = b.mul(sg, ymx);
        let rmz = b.sub(rho, z);
        let xrmz = b.mul(x, rmz);
        let dy = b.sub(xrmz, y);
        let xy = b.mul(x, y);
        let bz = b.mul(bt, z);
        let dz = b.sub(xy, bz);
        b.finish(&[dx, dy, dz], &[], 3, 3).unwrap()
    }

    #[test]
    fn trait_eval_matches_reference() {
        let ev = RefEval { tape: lorenz() };
        let u = [1.0, 2.0, 3.0];
        let p = [10.0, 28.0, 8.0 / 3.0];
        let mut scratch = vec![0.0; ev.n_scratch()];
        let mut deriv = vec![0.0; ev.dim()];
        ev.eval(&u, &p, 0.0, &mut scratch, &mut deriv);
        let want = reference::eval_alloc(&ev.tape, &u, &p, 0.0);
        assert_eq!(deriv, want);
    }

    #[test]
    fn usable_as_trait_object() {
        let ev = RefEval { tape: lorenz() };
        let dynev: &dyn Evaluator = &ev;
        assert_eq!(dynev.dim(), 3);
        assert_eq!(dynev.n_param(), 3);
        assert!(!dynev.has_jacobian());
        let mut scratch = vec![0.0; dynev.n_scratch()];
        let mut deriv = vec![0.0; dynev.dim()];
        dynev.eval(
            &[1.0, 2.0, 3.0],
            &[10.0, 28.0, 8.0 / 3.0],
            0.0,
            &mut scratch,
            &mut deriv,
        );
        assert!((deriv[0] - 10.0).abs() < 1e-15); // sigma*(y-x) = 10*(2-1)
    }
}
