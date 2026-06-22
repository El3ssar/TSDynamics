//! Shared modified-Newton substage solver for the implicit family.
//!
//! Every fully-implicit kernel here (backward Euler, the implicit midpoint and
//! trapezoidal rules, the SDIRK stages, and TR-BDF2) ultimately solves one or
//! more algebraic substages of the *same* shape
//!
//! ```text
//!     y = base + coef·h·f(t, y)
//! ```
//!
//! by a modified-Newton iteration: freeze the analytic Jacobian `J = ∂f/∂u` at
//! the initial guess, factor the iteration matrix `I − coef·h·J` once, and reuse
//! that factorization across iterations (the standard stiff-solver economy). The
//! routine lives here, written once, so every kernel that needs a Newton stage
//! shares one audited implementation rather than copying the loop.
//!
//! The base method's *defining* coefficients (`coef`, the stage time `t`, the
//! `base` vector) are the caller's; this module owns only the generic algebraic
//! solve and its convergence policy.

use super::control::BaseOutcome;
use tsdyn_ir::Evaluator;

/// Maximum modified-Newton iterations per substage before the step is declared
/// non-convergent (recoverable — the controller retries with a smaller `h`, which
/// improves both the Newton conditioning and the initial guess).
pub(crate) const NEWTON_MAX_ITERS: usize = 20;

/// Newton convergence threshold on the weighted-RMS norm of the increment. Tight
/// relative to the integration tolerance so the residual Newton error stays well
/// below the truncation error the step-doubling estimate measures.
pub(crate) const NEWTON_TOL: f64 = 1e-3;

/// Weighted-RMS norm `sqrt(mean( (v_i / (atol + rtol·|ref_i|))² ))` — the scaled
/// size of an increment relative to the solution magnitude.
pub(crate) fn wrms(v: &[f64], reference: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..n {
        let sc = atol + rtol * reference[i].abs();
        let e = v[i] / sc;
        acc += e * e;
    }
    (acc / n as f64).sqrt()
}

/// Reusable per-step scratch for a Newton substage: the residual / RHS buffer, the
/// frozen Jacobian, the iteration matrix, its pivots, and the increment. Grown to
/// the system dimension on first use and reused with no per-step allocation.
#[derive(Default)]
pub(crate) struct NewtonWork {
    /// RHS / residual buffer (`f(t, y)` then the Newton residual), length `dim`.
    pub fy: Vec<f64>,
    /// Frozen analytic Jacobian `∂f/∂u`, row-major `dim × dim`.
    pub jac: Vec<f64>,
    /// Iteration matrix `I − coef·h·J`, row-major `dim × dim`.
    pub mat: Vec<f64>,
    /// LU pivots for `mat`, length `dim`.
    pub piv: Vec<usize>,
    /// Newton increment `Δ`, length `dim`.
    pub delta: Vec<f64>,
}

impl NewtonWork {
    pub(crate) fn ensure(&mut self, n: usize) {
        if self.fy.len() != n {
            self.fy = vec![0.0; n];
            self.jac = vec![0.0; n * n];
            self.mat = vec![0.0; n * n];
            self.piv = vec![0; n];
            self.delta = vec![0.0; n];
        }
    }
}

/// Solve one implicit substage `y = base + coef·h·f(t, y)` by modified Newton.
///
/// `y` carries the initial guess in and the solution out. The iteration matrix
/// `I − coef·h·J` is factored once at the guess and reused across iterations.
/// All buffers are caller-owned (disjoint slices), so this is a free function to
/// keep the borrows simple.
///
/// Returns [`BaseOutcome::Ok`] on convergence, [`BaseOutcome::Recoverable`] when
/// the Jacobian/RHS is non-finite, the iteration matrix is singular, or Newton
/// does not converge in the budget (all "shrink `h` and retry" cases).
#[allow(clippy::too_many_arguments)]
pub(crate) fn newton_substage(
    ev: &dyn Evaluator,
    t: f64,
    p: &[f64],
    coef: f64,
    h: f64,
    base: &[f64],
    y: &mut [f64],
    fy: &mut [f64],
    jac: &mut [f64],
    mat: &mut [f64],
    piv: &mut [usize],
    delta: &mut [f64],
    scratch: &mut [f64],
    rtol: f64,
    atol: f64,
) -> BaseOutcome {
    use super::linalg::{build_shifted, lu_factor, lu_solve};
    let n = y.len();

    // Freeze J at the initial guess and factor (I − coef·h·J) once.
    ev.eval_jac(y, p, t, scratch, fy, jac);
    if !jac.iter().all(|x| x.is_finite()) {
        return BaseOutcome::Recoverable;
    }
    build_shifted(mat, jac, n, coef * h);
    if !lu_factor(mat, n, piv) {
        return BaseOutcome::Recoverable; // singular ⇒ shrink h and retry
    }

    for _ in 0..NEWTON_MAX_ITERS {
        ev.eval(y, p, t, scratch, fy);
        if !fy.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Recoverable;
        }
        // Residual R(y) = y − base − coef·h·f(t,y); Newton step (I − coef·h·J) Δ = R.
        for i in 0..n {
            delta[i] = y[i] - base[i] - coef * h * fy[i];
        }
        lu_solve(mat, n, piv, delta);
        for i in 0..n {
            y[i] -= delta[i];
        }
        if !y.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Recoverable;
        }
        if wrms(delta, y, rtol, atol) <= NEWTON_TOL {
            return BaseOutcome::Ok;
        }
    }
    BaseOutcome::Recoverable // did not converge in the iteration budget
}

/// A convenience wrapper that solves a substage using a [`NewtonWork`] buffer set
/// and the evaluator's own scratch from a [`SolverState`].
///
/// Kernels that keep a single `NewtonWork` (the SDIRK family) call this; TR-BDF2,
/// which threads its own disjoint field borrows, calls [`newton_substage`]
/// directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_substage(
    ev: &dyn Evaluator,
    t: f64,
    p: &[f64],
    coef: f64,
    h: f64,
    base: &[f64],
    y: &mut [f64],
    work: &mut NewtonWork,
    scratch: &mut [f64],
    rtol: f64,
    atol: f64,
) -> BaseOutcome {
    let n = y.len();
    work.ensure(n);
    let NewtonWork {
        fy,
        jac,
        mat,
        piv,
        delta,
    } = work;
    newton_substage(
        ev, t, p, coef, h, base, y, fy, jac, mat, piv, delta, scratch, rtol, atol,
    )
}
