//! Shared white-box test fixtures and reference-free oracles for the explicit
//! family (compiled only under `cfg(test)`).
//!
//! The kernels here are validated three independent ways so that *coefficient
//! correctness never rests on the transcription being trusted*:
//!
//! 1. [`converges_at_order`] measures the empirical convergence order of a
//!    fixed-step propagation (halve `h`, watch the error fall by `2^p`) — this
//!    pins the propagation tableau `(c, a, b)` directly, with no external
//!    reference;
//! 2. analytic-solution accuracy on problems with closed forms ([`DecayEval`],
//!    [`HarmonicEval`]);
//! 3. cross-method agreement on a chaotic system (in the integration test),
//!    which catches any tableau slip a smooth problem might forgive.
//!
//! Together these are an oracle strong enough that a single wrong digit in any
//! tableau fails a test loudly.

use crate::{Evaluator, Solver, SolverState, StepOutcome};

use super::control::{self, RkWork};

/// `du/dt = -u` (componentwise) — closed form `u(t) = u₀·e^{-t}`.
pub(crate) struct DecayEval {
    pub dim: usize,
}

impl Evaluator for DecayEval {
    fn dim(&self) -> usize {
        self.dim
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
    fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _s: &mut [f64], deriv: &mut [f64]) {
        for (d, &ui) in deriv.iter_mut().zip(u) {
            *d = -ui;
        }
    }
    fn eval_jac(
        &self,
        _u: &[f64],
        _p: &[f64],
        _t: f64,
        _s: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        let n = self.dim;
        for d in deriv.iter_mut() {
            *d = 0.0;
        }
        for j in jac.iter_mut() {
            *j = 0.0;
        }
        for i in 0..n {
            jac[i * n + i] = -1.0;
        }
    }
}

/// The harmonic oscillator `x' = v, v' = -ω²x` — closed form
/// `x(t) = x₀·cos(ωt) + (v₀/ω)·sin(ωt)`; for `(x₀, v₀) = (1, 0)`, `ω = 1`,
/// the solution is `(cos t, -sin t)`.  A bounded, energy-conserving smooth test.
pub(crate) struct HarmonicEval {
    pub omega: f64,
}

impl Evaluator for HarmonicEval {
    fn dim(&self) -> usize {
        2
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
    fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _s: &mut [f64], deriv: &mut [f64]) {
        deriv[0] = u[1];
        deriv[1] = -self.omega * self.omega * u[0];
    }
    fn eval_jac(
        &self,
        _u: &[f64],
        _p: &[f64],
        _t: f64,
        _s: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        deriv[0] = 0.0;
        deriv[1] = 0.0;
        let w2 = self.omega * self.omega;
        jac.copy_from_slice(&[0.0, 1.0, -w2, 0.0]);
    }
}

/// Largest componentwise absolute difference between two equal-length vectors.
pub(crate) fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Advance `st` by one fixed step with tableau `(c, a, b)`, asserting the kernel
/// accepts (the smooth test problems never blow up).  Used to measure the
/// propagation order independently of the error controller.
pub(crate) fn fixed_propagate(
    ev: &dyn Evaluator,
    st: &mut SolverState,
    h: f64,
    c: &[f64],
    a: &[&[f64]],
    b: &[f64],
    work: &mut RkWork,
) {
    match control::fixed_step(ev, st, h, c, a, b, work) {
        StepOutcome::Accepted { .. } => {}
        other => panic!("fixed_propagate expected Accepted, got {other:?}"),
    }
}

/// Empirical convergence order of a one-step propagator.
///
/// Integrates from `t = 0` to `t_final` at each step size in `hs` (which should
/// divide `t_final` evenly and be in decreasing order), measures the final-state
/// error against `exact`, and fits `error ∝ h^p` across the two finest steps.
pub(crate) fn converges_at_order<F, E>(
    mut propagate: F,
    ev: &dyn Evaluator,
    u0: Vec<f64>,
    t_final: f64,
    hs: &[f64],
    exact: E,
) -> f64
where
    F: FnMut(&mut SolverState, f64, &mut RkWork),
    E: Fn(f64) -> Vec<f64>,
{
    let mut errs: Vec<(f64, f64)> = Vec::new();
    for &h in hs {
        let n = (t_final / h).round() as usize;
        let mut st = SolverState::for_evaluator(ev, u0.clone(), 0.0, vec![]);
        let mut work = RkWork::new();
        for _ in 0..n {
            propagate(&mut st, h, &mut work);
        }
        errs.push((h, max_abs_diff(&st.u, &exact(st.t))));
    }
    let (h1, e1) = errs[errs.len() - 2];
    let (h2, e2) = errs[errs.len() - 1];
    (e1 / e2).ln() / (h1 / h2).ln()
}

/// Drive an adaptive kernel through an engine-style step/accept/retry loop from
/// the current `st.t` to `t_final`, starting from step guess `h0`.  Mirrors how
/// the engine (E5) will call [`Solver::step`], so the kernels are exercised on
/// exactly the surface they ship behind.  Panics on [`Failed`](StepOutcome::Failed).
pub(crate) fn integrate_adaptive(
    solver: &mut dyn Solver,
    ev: &dyn Evaluator,
    st: &mut SolverState,
    t_final: f64,
    h0: f64,
) {
    let mut h = h0;
    let mut steps = 0usize;
    while st.t < t_final - 1e-12 * (1.0 + t_final.abs()) {
        let step_h = h.min(t_final - st.t);
        match solver.step(ev, st, step_h) {
            StepOutcome::Accepted { h_next } | StepOutcome::Rejected { h_next } => {
                h = h_next.max(1e-15);
            }
            StepOutcome::Failed => panic!("adaptive integration failed at t = {}", st.t),
        }
        steps += 1;
        assert!(steps < 5_000_000, "runaway step count at t = {}", st.t);
    }
}
