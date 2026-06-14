//! Shared test fixtures for the SDE kernel suites (compiled only under
//! `cfg(test)`).
//!
//! The kernels are driven through the frozen [`Evaluator`] seam, so the tests
//! need concrete drift/diffusion evaluators. These are deliberately tiny,
//! transparent, *hand-written* implementations of the canonical scalar laws —
//! Ornstein–Uhlenbeck drift, constant (additive) and proportional
//! (multiplicative) diffusion — so a kernel's update formula can be checked
//! against the law written out by hand, with no tape or RNG in the loop. Each is
//! applied component-wise, so the same fixture serves any dimension (the kernel
//! sizes its buffers from the state-slice length, never from
//! [`Evaluator::dim`]).
//!
//! The end-to-end moment / determinism tests live in `tsdyn-engine`, where the
//! RNG and integrate loop are, and run against the real interpreter.

use tsdyn_ir::Evaluator;

/// Ornstein–Uhlenbeck-style linear drift `f_k(x) = θ (μ − x_k)`, applied
/// independently to each component. With `μ = 0` this is `f_k = −θ x_k`.
pub struct LinearDrift {
    theta: f64,
    mu: f64,
}

impl LinearDrift {
    pub fn new(theta: f64, mu: f64) -> Self {
        LinearDrift { theta, mu }
    }
}

impl Evaluator for LinearDrift {
    fn dim(&self) -> usize {
        1 // nominal: the kernel drives eval by the state-slice length, not dim()
    }
    fn n_param(&self) -> usize {
        0
    }
    fn n_scratch(&self) -> usize {
        0
    }
    fn has_jacobian(&self) -> bool {
        false
    }
    fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
        for (d, &x) in deriv.iter_mut().zip(u.iter()) {
            *d = self.theta * (self.mu - x);
        }
    }
    fn eval_jac(
        &self,
        _u: &[f64],
        _p: &[f64],
        _t: f64,
        _scratch: &mut [f64],
        _deriv: &mut [f64],
        _jac: &mut [f64],
    ) {
        unreachable!("LinearDrift carries no Jacobian");
    }
}

/// Constant (additive) diagonal diffusion `g_k(x) = σ` for every component;
/// `∂g/∂u ≡ 0`.
pub struct ConstDiffusion {
    sigma: f64,
}

impl ConstDiffusion {
    pub fn new(sigma: f64) -> Self {
        ConstDiffusion { sigma }
    }
}

impl Evaluator for ConstDiffusion {
    fn dim(&self) -> usize {
        1
    }
    fn n_param(&self) -> usize {
        0
    }
    fn n_scratch(&self) -> usize {
        0
    }
    fn has_jacobian(&self) -> bool {
        true // a (zero) Jacobian, so Milstein can request eval_jac
    }
    fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
        let _ = u;
        for d in deriv.iter_mut() {
            *d = self.sigma;
        }
    }
    fn eval_jac(
        &self,
        u: &[f64],
        _p: &[f64],
        _t: f64,
        _scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        let dim = u.len();
        for d in deriv.iter_mut() {
            *d = self.sigma;
        }
        for j in jac.iter_mut() {
            *j = 0.0; // additive noise ⇒ ∂g/∂u = 0
        }
        debug_assert_eq!(jac.len(), dim * dim);
    }
}

/// Proportional (multiplicative) diagonal diffusion `g_k(x) = σ x_k`, as in
/// geometric Brownian motion; the diagonal of `∂g/∂u` is `σ`, off-diagonals `0`.
pub struct ProportionalDiffusion {
    sigma: f64,
}

impl ProportionalDiffusion {
    pub fn new(sigma: f64) -> Self {
        ProportionalDiffusion { sigma }
    }
}

impl Evaluator for ProportionalDiffusion {
    fn dim(&self) -> usize {
        1
    }
    fn n_param(&self) -> usize {
        0
    }
    fn n_scratch(&self) -> usize {
        0
    }
    fn has_jacobian(&self) -> bool {
        true
    }
    fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
        for (d, &x) in deriv.iter_mut().zip(u.iter()) {
            *d = self.sigma * x;
        }
    }
    fn eval_jac(
        &self,
        u: &[f64],
        _p: &[f64],
        _t: f64,
        _scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        let dim = u.len();
        debug_assert_eq!(jac.len(), dim * dim);
        for (d, &x) in deriv.iter_mut().zip(u.iter()) {
            *d = self.sigma * x;
        }
        for (k, j) in jac.iter_mut().enumerate() {
            // Diagonal ∂g_k/∂u_k = σ; off-diagonal 0.
            *j = if k % (dim + 1) == 0 { self.sigma } else { 0.0 };
        }
    }
}

/// A plain Euler–Maruyama step written out by hand (no Jacobian), so a test can
/// assert that a Milstein step differs from it by exactly the Milstein
/// correction. Returns the updated state.
pub struct EulerMaruyamaRef;

impl EulerMaruyamaRef {
    pub fn step(
        drift: &dyn Evaluator,
        diffusion: &dyn Evaluator,
        u: &[f64],
        dw: &[f64],
        h: f64,
    ) -> Vec<f64> {
        let dim = u.len();
        let mut f = vec![0.0; dim];
        let mut g = vec![0.0; dim];
        let mut ds = vec![0.0; drift.n_scratch()];
        let mut gs = vec![0.0; diffusion.n_scratch()];
        drift.eval(u, &[], 0.0, &mut ds, &mut f);
        diffusion.eval(u, &[], 0.0, &mut gs, &mut g);
        (0..dim).map(|k| u[k] + f[k] * h + g[k] * dw[k]).collect()
    }
}
