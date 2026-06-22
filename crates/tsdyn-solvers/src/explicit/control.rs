//! Shared step-size control and explicit Runge–Kutta stage machinery for the
//! explicit solver family (stream E3).
//!
//! Every explicit kernel in this family is the *same* algorithm differing only
//! in its Butcher tableau, so the tableau-walking logic lives here once and each
//! `…/explicit/<method>.rs` file is reduced to its coefficient tables plus a thin
//! [`Solver`](crate::Solver) impl.  Writing the nested stage loop exactly once is
//! deliberate: it is where a transcription slip would otherwise hide, and a
//! single tested implementation is far easier to audit than four copies.
//!
//! # Division of labour with the engine (ROADMAP §4d)
//!
//! A [`Solver::step`](crate::Solver::step) does **one** step and owns its own
//! embedded error estimate and step-size controller (that is what
//! [`Caps::adaptive`](crate::Caps) means).  The *engine* (stream E5) owns the
//! outer step/accept/retry loop, the initial step-size heuristic, and dense
//! output.  So the helpers here advance a [`SolverState`](crate::SolverState) by
//! a single `h` and report an [`StepOutcome`](crate::StepOutcome); they never
//! loop.
//!
//! # Error norm and controller
//!
//! The error norm is the scaled RMS `‖e_i / (atol + rtol·max(|u_i|, |u_newᵢ|))‖`
//! and the controller is the elementary `factor = clamp(SAFETY · err^p, …)` with
//! `p = −1/(error_estimator_order + 1)`.  This is the v2 engine's norm and
//! controller exactly, so these kernels reproduce its accept/reject decisions
//! (the v2 engine is in turn cross-validated against SciPy).  Like v2 — and
//! unlike SciPy's incremental controller — it is *stateless*: it does not damp
//! growth on the step following a rejection.  That affects only the step
//! *sequence*, never the accept threshold, so the cross-validation compares
//! end/interpolated states rather than per-step sizes.
//!
//! # The engine owns the retry floor
//!
//! A single step never loops, so it has no minimum-`h` guard: a step that keeps
//! overshooting into a non-finite region returns ever-smaller `h_next` via
//! [`StepOutcome::Rejected`].  Flooring `h` and giving up (the v2 engine raised
//! once `h ≤ 1e-14·(1+|t|)`) is the *engine's* job — it owns the loop and the
//! integration's time scale, which a single step cannot see.  A kernel reports
//! [`StepOutcome::Failed`] only for an unrecoverable point (a non-finite RHS at
//! the current, committed state).

use crate::{Evaluator, SolverState, StepOutcome};

/// Controller safety factor (SciPy/Hairer default).
pub(crate) const SAFETY: f64 = 0.9;
/// Smallest step-size shrink factor a single step may suggest.
pub(crate) const MIN_FACTOR: f64 = 0.2;
/// Largest step-size growth factor a single step may suggest.
pub(crate) const MAX_FACTOR: f64 = 10.0;

/// Scaled RMS error norm: `sqrt(mean_i (err_i / sc_i)^2)` with the SciPy scale
/// `sc_i = atol + rtol·|scale_i|`.
///
/// `scale` carries the magnitude each component's error is measured against —
/// the kernels pass `max(|u_i|, |u_newᵢ|)`, matching the v2 engine and SciPy.
/// A zero-dimensional system has no error, so the norm is `0` (without this
/// guard the mean would be `0/0 = NaN`, which an adaptive kernel would read as a
/// non-finite trial and reject forever — see the `dim = 0` test).
#[inline]
pub(crate) fn scaled_rms(err: &[f64], scale: &[f64], rtol: f64, atol: f64) -> f64 {
    if err.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..err.len() {
        let sc = atol + rtol * scale[i].abs();
        let r = err[i] / sc;
        acc += r * r;
    }
    (acc / err.len() as f64).sqrt()
}

/// The elementary step-size factor for an error norm `err` and controller
/// exponent `exponent` (`= −1/(error_estimator_order + 1)`), clamped to
/// `[MIN_FACTOR, MAX_FACTOR]`.  A zero error means "as large as allowed".
#[inline]
pub(crate) fn step_factor(err: f64, exponent: f64) -> f64 {
    if err == 0.0 {
        MAX_FACTOR
    } else {
        (SAFETY * err.powf(exponent)).clamp(MIN_FACTOR, MAX_FACTOR)
    }
}

/// Per-kernel reusable buffers for one integration worker, sized lazily on the
/// first step and reused with no allocation thereafter.
///
/// Held in the kernel's `&mut self` (not in [`SolverState`](crate::SolverState),
/// whose `scratch` is the *evaluator's* register file): the `Solver` contract
/// keeps per-method stage buffers private to the kernel.
#[derive(Default)]
pub(crate) struct RkWork {
    /// Stage derivatives `k_0..k_{s-1}`, each length `dim`.
    pub k: Vec<Vec<f64>>,
    /// Scratch state for the stage being evaluated, length `dim`.
    pub utmp: Vec<f64>,
    /// Proposed next state, length `dim`.
    pub u_new: Vec<f64>,
    /// Local error estimate, length `dim`.
    pub err: Vec<f64>,
    /// Componentwise error scale `max(|u|, |u_new|)`, length `dim`.
    pub scale: Vec<f64>,
}

impl RkWork {
    /// Empty buffers; sized on the first [`ensure`](RkWork::ensure).
    pub(crate) fn new() -> Self {
        RkWork {
            k: Vec::new(),
            utmp: Vec::new(),
            u_new: Vec::new(),
            err: Vec::new(),
            scale: Vec::new(),
        }
    }

    /// Ensure the buffers hold `stages` stages of width `dim`, (re)allocating
    /// only when the shape changes (first step, or a kernel reused across systems
    /// of different dimension).
    #[inline]
    pub(crate) fn ensure(&mut self, stages: usize, dim: usize) {
        if self.k.len() != stages || self.utmp.len() != dim {
            self.k = vec![vec![0.0; dim]; stages];
            self.utmp = vec![0.0; dim];
            self.u_new = vec![0.0; dim];
            self.err = vec![0.0; dim];
            self.scale = vec![0.0; dim];
        }
    }
}

/// Fill `work.k[0..c.len()]` for the explicit lower-triangular tableau `(c, a)`.
///
/// `a[i]` holds the `i` coefficients for stage `i` (so `a[0]` is empty, `a[1]`
/// has one entry, …); stage `i` is evaluated at
/// `t + c[i]·h`, `u + h·Σ_{j<i} a[i][j]·k_j`.  Stage 0 is `f(t, u)`.
pub(crate) fn compute_stages(
    ev: &dyn Evaluator,
    st: &mut SolverState,
    h: f64,
    c: &[f64],
    a: &[&[f64]],
    work: &mut RkWork,
) {
    let dim = st.u.len();
    // Stage 0 = f(t, u). Distinct fields of `st` (u/p/scratch) and of `work`
    // borrow disjointly, so these &/&mut mixes are accepted by the borrow checker.
    ev.eval(&st.u, &st.p, st.t, &mut st.scratch, &mut work.k[0]);
    for i in 1..c.len() {
        let ai = a[i];
        for d in 0..dim {
            let mut acc = 0.0;
            for (j, &aij) in ai.iter().enumerate() {
                acc += aij * work.k[j][d];
            }
            work.utmp[d] = st.u[d] + h * acc;
        }
        ev.eval(
            &work.utmp,
            &st.p,
            st.t + c[i] * h,
            &mut st.scratch,
            &mut work.k[i],
        );
    }
}

/// Linear combination `out = u + h·Σ_i b_i·k_i` (one weight per stage).
pub(crate) fn combine(u: &[f64], h: f64, b: &[f64], k: &[Vec<f64>], out: &mut [f64]) {
    for (d, out_d) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (i, &bi) in b.iter().enumerate() {
            acc += bi * k[i][d];
        }
        *out_d = u[d] + h * acc;
    }
}

/// Local error estimate `out = h·Σ_i e_i·k_i` (one error weight per stage).
pub(crate) fn error_vector(h: f64, e: &[f64], k: &[Vec<f64>], out: &mut [f64]) {
    for (d, out_d) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (i, &ei) in e.iter().enumerate() {
            acc += ei * k[i][d];
        }
        *out_d = h * acc;
    }
}

/// One adaptive embedded-RK step for a single-error-vector tableau (DP45, Tsit5).
///
/// Computes the stages, the propagated solution (`b`) and the local error
/// estimate (`e`), then accepts/rejects against the scaled-RMS norm and returns
/// the controller's suggested next step size.  On acceptance `st` is advanced;
/// on rejection `st` is left **exactly** as it was found, per the `Solver`
/// contract, because the trial is built entirely in `work` and only committed to
/// `st` once accepted.
///
/// `e` is the *error* weight set (`b − b̂`), and `err_exponent =
/// −1/(error_estimator_order + 1)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adaptive_step(
    ev: &dyn Evaluator,
    st: &mut SolverState,
    h: f64,
    c: &[f64],
    a: &[&[f64]],
    b: &[f64],
    e: &[f64],
    err_exponent: f64,
    rtol: f64,
    atol: f64,
    work: &mut RkWork,
) -> StepOutcome {
    let dim = st.u.len();
    work.ensure(c.len(), dim);
    compute_stages(ev, st, h, c, a, work);
    // f at the current (committed, finite) point is non-finite ⇒ unrecoverable.
    if work.k[0].iter().any(|x| !x.is_finite()) {
        return StepOutcome::Failed;
    }
    combine(&st.u, h, b, &work.k, &mut work.u_new);
    error_vector(h, e, &work.k, &mut work.err);
    for d in 0..dim {
        work.scale[d] = st.u[d].abs().max(work.u_new[d].abs());
    }
    let err = scaled_rms(&work.err, &work.scale, rtol, atol);
    // A non-finite trial (the step overshot into a singular region) is
    // recoverable by a smaller step: reject and shrink hard rather than fail.
    if !work.u_new.iter().all(|x| x.is_finite()) || !err.is_finite() {
        return StepOutcome::Rejected {
            h_next: h * MIN_FACTOR,
        };
    }
    if err <= 1.0 {
        st.u.copy_from_slice(&work.u_new);
        st.t += h;
        StepOutcome::Accepted {
            h_next: h * step_factor(err, err_exponent),
        }
    } else {
        StepOutcome::Rejected {
            h_next: h * step_factor(err, err_exponent).min(1.0),
        }
    }
}

/// One fixed-step (non-adaptive) RK step for tableau `(c, a, b)` — the RK4 path.
///
/// Always accepts (returning the same `h`) unless the RHS or the resulting state
/// is non-finite, which is unrecoverable for a fixed-step method and reported as
/// [`Failed`](StepOutcome::Failed).
pub(crate) fn fixed_step(
    ev: &dyn Evaluator,
    st: &mut SolverState,
    h: f64,
    c: &[f64],
    a: &[&[f64]],
    b: &[f64],
    work: &mut RkWork,
) -> StepOutcome {
    let dim = st.u.len();
    work.ensure(c.len(), dim);
    compute_stages(ev, st, h, c, a, work);
    if work.k[0].iter().any(|x| !x.is_finite()) {
        return StepOutcome::Failed;
    }
    combine(&st.u, h, b, &work.k, &mut work.u_new);
    if !work.u_new.iter().all(|x| x.is_finite()) {
        return StepOutcome::Failed;
    }
    st.u.copy_from_slice(&work.u_new);
    st.t += h;
    StepOutcome::Accepted { h_next: h }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_rms_of_an_empty_system_is_zero() {
        // A zero-dimensional system has no error components: the norm must be 0,
        // not the 0/0 = NaN the mean would otherwise produce. (Regression: a NaN
        // here makes an adaptive kernel see a non-finite trial and reject forever
        // on a dim = 0 problem, while RK4/DOP853 accept — an inconsistent hang.)
        assert_eq!(scaled_rms(&[], &[], 1e-6, 1e-9), 0.0);
    }
}
