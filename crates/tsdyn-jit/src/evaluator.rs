//! [`JitEvaluator`] — the public, native-code [`Evaluator`].

use cranelift_jit::JITModule;
use tsdyn_ir::{Evaluator, Tape};

use crate::codegen::{self, EvalFn, EvalJacFn};
use crate::error::JitError;

/// A tape compiled to native code by Cranelift.
///
/// Build one with [`JitEvaluator::new`]; it owns the JIT module that holds the
/// generated code and exposes the two evaluation entry points the engine and
/// solvers need, mirroring the `tsdyn-vm` interpreter:
///
/// - [`eval`](JitEvaluator::eval) — `du/dt` only, the explicit-solver path;
/// - [`eval_jac`](JitEvaluator::eval_jac) — `du/dt` and the analytic Jacobian in
///   one pass, the implicit/stiff-solver path.
///
/// Unlike the interpreter, the JIT keeps no register file: the tape's registers
/// are Cranelift SSA values baked into the compiled code, so
/// [`n_scratch`](JitEvaluator::n_scratch) is `0` and the `scratch` slice the
/// trait threads through is ignored. The evaluator is compiled once and is
/// [`Sync`], so a rayon ensemble shares one `&JitEvaluator` across all workers.
///
/// It also implements [`tsdyn_ir::Evaluator`], so solvers and the engine can
/// drive it as `&dyn Evaluator` interchangeably with the interpreter.
pub struct JitEvaluator {
    dim: usize,
    n_state: usize,
    n_param: usize,
    has_jac: bool,
    eval_fn: EvalFn,
    eval_jac_fn: EvalJacFn,
    // `Some` for the evaluator's whole life; taken in `Drop` to free the code.
    // Boxed-trait interior (`RefCell`) makes `JITModule: !Sync`; see the
    // `unsafe impl Sync` below for why sharing the evaluator is still sound.
    module: Option<JITModule>,
}

// SAFETY: after `new` returns, the only operations performed on a shared
// `&JitEvaluator` are calls through `eval_fn` / `eval_jac_fn`. Those execute the
// finalized native code, which is position-independent and reentrant: it reads
// only the caller-provided `u`/`p`, writes only the caller-provided
// `deriv`/`jac`, and calls only reentrant `std`/`libm` shims — no shared mutable
// state. The single `!Sync` field (`JITModule`, via an internal `RefCell` symbol
// cache used during compilation) is never touched after construction except by
// `Drop`, which has exclusive `&mut` access. Hence concurrent `&self` use across
// threads is data-race free. (`Send` is auto-derived: `JITModule` is `Send` and
// `fn` pointers are `Send`.)
unsafe impl Sync for JitEvaluator {}

impl JitEvaluator {
    /// JIT-compile a validated [`Tape`] to native code.
    ///
    /// Emits the `du/dt` and `du/dt`+Jacobian functions and finalizes them.
    /// Returns a [`JitError`] if Cranelift cannot build the host ISA or compile
    /// the tape.
    pub fn new(tape: &Tape) -> Result<Self, JitError> {
        let compiled = codegen::compile(tape)?;
        Ok(Self {
            dim: tape.dim(),
            n_state: tape.n_state(),
            n_param: tape.n_param(),
            has_jac: tape.has_jacobian(),
            eval_fn: compiled.eval,
            eval_jac_fn: compiled.eval_jac,
            module: Some(compiled.module),
        })
    }

    /// System dimension — the length of `u`, `deriv`, and `√(jac.len())`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Declared parameter width — the expected length of `p`.
    #[inline]
    pub fn n_param(&self) -> usize {
        self.n_param
    }

    /// Scratch length the JIT requires: always `0` (the tape's registers live in
    /// SSA values, not a caller-owned register file).
    #[inline]
    pub fn n_scratch(&self) -> usize {
        0
    }

    /// Whether the compiled tape carries an analytic Jacobian (so
    /// [`eval_jac`](JitEvaluator::eval_jac) writes one).
    #[inline]
    pub fn has_jacobian(&self) -> bool {
        self.has_jac
    }

    /// Evaluate `du/dt` at `(u, p, t)` into `deriv`. The `scratch` argument is
    /// ignored (the JIT needs none); it is present only to match the
    /// [`Evaluator`] surface. `u` must be at least
    /// [`n_state`](Tape::n_state)-long, `p` at least
    /// [`n_param`](JitEvaluator::n_param), and `deriv` at least
    /// [`dim`](JitEvaluator::dim).
    #[inline]
    pub fn eval(&self, u: &[f64], p: &[f64], t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
        self.debug_check_buffers(u, p, deriv);
        // SAFETY: the buffers meet the length contract checked above; the
        // compiled function reads `n_state` f64 from `u`, `n_param` from `p`, and
        // writes `dim` f64 to `deriv` — all in bounds.
        unsafe {
            (self.eval_fn)(u.as_ptr(), p.as_ptr(), t, deriv.as_mut_ptr());
        }
    }

    /// Evaluate `du/dt` (into `deriv`) and the row-major `dim × dim` Jacobian
    /// `∂f_k/∂u_j` at `jac[k * dim + j]` (into `jac`) in one pass. For a tape
    /// without a Jacobian, `jac` is left untouched (matching the interpreter).
    /// `jac` must hold at least `dim * dim` elements; the other buffers follow
    /// [`eval`](JitEvaluator::eval)'s contract.
    #[inline]
    pub fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        _scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        self.debug_check_buffers(u, p, deriv);
        debug_assert!(
            !self.has_jac || jac.len() >= self.dim * self.dim,
            "jac too small: {} < dim*dim {}",
            jac.len(),
            self.dim * self.dim
        );
        // SAFETY: as in `eval`; additionally the compiled function writes Jacobian
        // entries only when the tape carries them, and `jac` then holds `dim*dim`
        // f64 (checked above). A no-Jacobian tape writes nothing to `jac`.
        unsafe {
            (self.eval_jac_fn)(
                u.as_ptr(),
                p.as_ptr(),
                t,
                deriv.as_mut_ptr(),
                jac.as_mut_ptr(),
            );
        }
    }

    /// Allocate output buffers and return `du/dt` — convenience for one-off calls
    /// and tests.
    pub fn eval_alloc(&self, u: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut deriv = vec![0.0; self.dim];
        self.eval(u, p, t, &mut [], &mut deriv);
        deriv
    }

    /// Allocate output buffers and return `(du/dt, Jacobian)` with the Jacobian
    /// row-major `dim × dim` — convenience for one-off calls and tests.
    pub fn eval_jac_alloc(&self, u: &[f64], p: &[f64], t: f64) -> (Vec<f64>, Vec<f64>) {
        let mut deriv = vec![0.0; self.dim];
        let mut jac = vec![0.0; self.dim * self.dim];
        self.eval_jac(u, p, t, &mut [], &mut deriv, &mut jac);
        (deriv, jac)
    }

    /// Debug-only buffer precondition checks (compiled out of release builds, so
    /// the hot path pays nothing). Surfaces an undersized buffer — the classic
    /// misuse — with a clear message.
    #[inline]
    fn debug_check_buffers(&self, u: &[f64], p: &[f64], deriv: &[f64]) {
        debug_assert!(
            u.len() >= self.n_state,
            "state slice too small: {} < n_state {}",
            u.len(),
            self.n_state
        );
        debug_assert!(
            p.len() >= self.n_param,
            "param slice too small: {} < n_param {}",
            p.len(),
            self.n_param
        );
        debug_assert!(
            deriv.len() >= self.dim,
            "deriv too small: {} < dim {}",
            deriv.len(),
            self.dim
        );
    }
}

impl Drop for JitEvaluator {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: this evaluator is the sole owner of the JIT module and the
            // only holder of `fn` pointers into its code. It is being dropped, so
            // nothing can call those pointers afterwards — the precondition
            // `free_memory` requires.
            unsafe {
                module.free_memory();
            }
        }
    }
}

impl Evaluator for JitEvaluator {
    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }
    #[inline]
    fn n_param(&self) -> usize {
        self.n_param()
    }
    #[inline]
    fn n_scratch(&self) -> usize {
        self.n_scratch()
    }
    #[inline]
    fn has_jacobian(&self) -> bool {
        self.has_jacobian()
    }
    #[inline]
    fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
        self.eval(u, p, t, scratch, deriv)
    }
    #[inline]
    fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        self.eval_jac(u, p, t, scratch, deriv, jac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    /// Lorenz RHS with its analytic Jacobian, identical to the `tsdyn-vm` fixture.
    /// dx = σ(y − x); dy = x(ρ − z) − y; dz = xy − βz, with σ=p0, ρ=p1, β=p2.
    fn lorenz() -> Tape {
        let mut b = TapeBuilder::new();
        let sigma = b.param(0);
        let rho = b.param(1);
        let beta = b.param(2);
        let x = b.state(0);
        let y = b.state(1);
        let z = b.state(2);

        let ymx = b.sub(y, x);
        let dx = b.mul(sigma, ymx);
        let rmz = b.sub(rho, z);
        let xrmz = b.mul(x, rmz);
        let dy = b.sub(xrmz, y);
        let xy = b.mul(x, y);
        let bz = b.mul(beta, z);
        let dz = b.sub(xy, bz);

        let neg_sigma = b.neg(sigma);
        let zero = b.constant(0.0);
        let neg_one = b.constant(-1.0);
        let neg_x = b.neg(x);
        let neg_beta = b.neg(beta);
        b.finish(
            &[dx, dy, dz],
            &[
                neg_sigma, sigma, zero, // row 0
                rmz, neg_one, neg_x, // row 1
                y, x, neg_beta, // row 2
            ],
            3,
            3,
        )
        .unwrap()
    }

    const LORENZ_P: [f64; 3] = [10.0, 28.0, 8.0 / 3.0];

    #[test]
    fn evaluates_lorenz_rhs() {
        let ev = JitEvaluator::new(&lorenz()).unwrap();
        assert_eq!(ev.dim(), 3);
        assert_eq!(ev.n_param(), 3);
        assert_eq!(ev.n_scratch(), 0);
        assert!(ev.has_jacobian());

        let u = [1.0, 2.0, 3.0];
        let got = ev.eval_alloc(&u, &LORENZ_P, 0.0);
        let want = [
            10.0 * (2.0 - 1.0),
            1.0 * (28.0 - 3.0) - 2.0,
            1.0 * 2.0 - (8.0 / 3.0) * 3.0,
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-15, "got {g}, want {w}");
        }
    }

    #[test]
    fn evaluates_lorenz_jacobian() {
        let ev = JitEvaluator::new(&lorenz()).unwrap();
        let u = [1.0, 2.0, 3.0];
        let (deriv, jac) = ev.eval_jac_alloc(&u, &LORENZ_P, 0.0);
        assert!((deriv[1] - (1.0 * (28.0 - 3.0) - 2.0)).abs() < 1e-15);
        let [sigma, _rho, beta] = LORENZ_P;
        let (x, y, z) = (u[0], u[1], u[2]);
        let want = [
            -sigma,
            sigma,
            0.0, // row 0
            28.0 - z,
            -1.0,
            -x, // row 1
            y,
            x,
            -beta, // row 2
        ];
        for (k, (g, w)) in jac.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-15, "jac[{k}]: got {g}, want {w}");
        }
    }

    #[test]
    fn usable_as_trait_object() {
        let ev = JitEvaluator::new(&lorenz()).unwrap();
        let dynev: &dyn Evaluator = &ev;
        assert_eq!(dynev.dim(), 3);
        assert_eq!(dynev.n_scratch(), 0);
        assert!(dynev.has_jacobian());
        let mut deriv = [0.0; 3];
        dynev.eval(&[1.0, 2.0, 3.0], &LORENZ_P, 0.0, &mut [], &mut deriv);
        assert!((deriv[0] - 10.0).abs() < 1e-15);
    }

    #[test]
    fn sign_is_zero_at_zero() {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let s = b.sign(x);
        let ev = JitEvaluator::new(&b.finish(&[s], &[], 1, 0).unwrap()).unwrap();
        assert_eq!(ev.eval_alloc(&[0.0], &[], 0.0)[0], 0.0);
        assert_eq!(ev.eval_alloc(&[3.5], &[], 0.0)[0], 1.0);
        assert_eq!(ev.eval_alloc(&[-3.5], &[], 0.0)[0], -1.0);
    }

    #[test]
    fn eval_jac_leaves_jac_untouched_without_a_jacobian() {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let ev = JitEvaluator::new(&b.finish(&[x], &[], 1, 0).unwrap()).unwrap();
        assert!(!ev.has_jacobian());
        let mut deriv = [0.0];
        let mut jac = [f64::NAN];
        ev.eval_jac(&[5.0], &[], 0.0, &mut [], &mut deriv, &mut jac);
        assert_eq!(deriv[0], 5.0);
        assert!(jac[0].is_nan(), "jac should be left untouched");
    }

    #[test]
    fn jit_is_send_and_sync() {
        // The ensemble path shares one `&JitEvaluator` across rayon workers; lock
        // the bounds in at compile time.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JitEvaluator>();
    }
}
