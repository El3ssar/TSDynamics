//! Adaptive Dormand–Prince 5(4) — the `dopri5` embedded pair, registered as
//! `rk45`.
//!
//! A 5th-order solution with a 4th-order companion for the local-error estimate,
//! error-controlled step size, and the FSAL property (the last stage is `f` at
//! the proposed solution point, so it doubles as the next step's first stage).
//! This is the general-purpose workhorse for smooth non-stiff problems and the
//! method SciPy exposes as `RK45`; the coefficients, error norm and controller
//! here reproduce the v2 engine's `dopri5`, against which the migration is
//! cross-validated (Dormand & Prince, *J. Comput. Appl. Math.* **6** (1980) 19).
//!
//! The tableau is written in the 7-stage form `c₇ = 1`, `a₇ = b`: the seventh
//! stage is `f` at the 5th-order solution, so a single error-weight vector
//! `e = b − b̂` over the seven stages gives the embedded 4th-order estimate
//! (identical to the v2 `E1…E7` weights).

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{
    combine, dense_poly, error_vector, scaled_rms, step_factor, RkWork, MIN_FACTOR,
};

// ---------------------------------------------------------------------------
// FSAL (First Same As Last) reuse — shared by the FSAL kernels (rk45, tsit5, bs3)
// ---------------------------------------------------------------------------
//
// A genuine FSAL pair has `c_last = 1` and `a_last = b` (the last `a`-row equals
// the solution weights), and the last solution weight `b_last = 0`. Then the
// last stage is evaluated at exactly `(u_new, t + h)` — the propagated solution
// point — which is *identical* to the first stage `f(u, t)` of the following
// step. Recomputing it wastes one RHS evaluation per accepted step (≈1 of 7 for
// the 5(4) pairs, 1 of 4 for bs3).
//
// This module hosts the reuse once (the crate's "write the stage loop exactly
// once" rule) so all three FSAL kernels share a single audited implementation;
// the non-FSAL adaptive kernels (cashkarp, rkf45, heun_euler) keep using the
// generic `adaptive_step` and are untouched. The shared trait, `Caps`, and the
// engine loop are unchanged — the cache is purely kernel-local `&mut self` state.

/// The cached last-stage derivative of the previous accepted step, together with
/// the exact `(u, t)` it was evaluated at.
///
/// FSAL reuse is valid only when the *next* step's first-stage point is bit-for-
/// bit the point this derivative was taken at. We therefore key the cache on the
/// state and time, not on a "did the last step accept?" flag, so the reuse is
/// self-validating against every way the engine drives the kernel:
///
/// - **continuation after an accept**: the committed `(u, t)` equals the cached
///   point ⇒ reuse (bit-identical to recomputing `f` at the same arguments);
/// - **after a rejected step**: the kernel left `(u, t)` unchanged and the cache
///   is *not* updated on rejection (only on acceptance), so the cached point
///   still equals the live point ⇒ reuse stays valid (the rejected trial's own
///   last stage, taken at a discarded point, is never cached);
/// - **first step / fresh integration / a re-seated state** (the engine builds a
///   fresh solver per `integrate_*` call, and `set_state` re-seats the point):
///   the cached point differs (or the cache is empty / a different dim) ⇒
///   recompute. Even in the astronomically unlikely event two distinct
///   integrations share a bit-identical start point, reuse is still correct: `f`
///   is a deterministic pure function of `(u, t)`.
///
/// The guard costs `dim` float comparisons — negligible beside the RHS eval it
/// elides — and makes the optimisation a pure eval-count reduction with no
/// numerical change: every value computed is bit-for-bit what the non-FSAL path
/// would compute.
#[derive(Default)]
pub(super) struct FsalCache {
    /// `f(u, t)` at the cached point; empty until the first accepted step.
    deriv: Vec<f64>,
    /// The state the derivative was evaluated at.
    u: Vec<f64>,
    /// The time the derivative was evaluated at.
    t: f64,
}

impl FsalCache {
    /// Empty cache (no reusable derivative yet).
    pub(super) fn new() -> Self {
        FsalCache {
            deriv: Vec::new(),
            u: Vec::new(),
            t: 0.0,
        }
    }

    /// Whether the cached derivative is `f` at exactly `(u, t)` (bit-for-bit) and
    /// of the right dimension — i.e. safe to reuse as the next step's stage 0.
    #[inline]
    fn matches(&self, u: &[f64], t: f64) -> bool {
        !self.deriv.is_empty() && self.t == t && self.u == u
    }

    /// Record `f(u, t) = deriv` as the reusable last stage of an accepted step.
    #[inline]
    fn store(&mut self, deriv: &[f64], u: &[f64], t: f64) {
        self.deriv.clear();
        self.deriv.extend_from_slice(deriv);
        self.u.clear();
        self.u.extend_from_slice(u);
        self.t = t;
    }
}

/// One adaptive embedded-RK step for an FSAL single-error-vector tableau,
/// reusing the previous accepted step's last stage as this step's stage 0.
///
/// This is [`adaptive_step`](super::control::adaptive_step) specialised for the
/// FSAL property: the only behavioural difference is that stage 0 is *copied*
/// from `fsal` when the live point matches the cached point, instead of being
/// recomputed with an RHS evaluation. Every arithmetic result — the stages, the
/// propagated solution, the error norm, the accept/reject decision and the
/// suggested `h_next` — is bit-for-bit identical to `adaptive_step`, because the
/// reused stage 0 equals `f(u, t)` exactly (same pure function, same arguments).
///
/// On acceptance the last stage (which a genuine FSAL tableau evaluates at the
/// new `(u_new, t + h)`) is cached for the next step; on rejection the state and
/// the cache are both left untouched.
///
/// `e` is the error-weight set (`b − b̂`) and `err_exponent = −1/(estimator
/// order + 1)`, exactly as for `adaptive_step`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fsal_adaptive_step(
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
    fsal: &mut FsalCache,
) -> StepOutcome {
    let dim = st.u.len();
    work.ensure(c.len(), dim);
    work.invalidate_dense();

    // Stage 0: reuse the cached last stage of the previous accepted step when it
    // was taken at exactly this `(u, t)`, else evaluate `f(u, t)` afresh. Both
    // branches leave `work.k[0] = f(st.u, st.t)` bit-for-bit.
    if fsal.matches(&st.u, st.t) {
        work.k[0].copy_from_slice(&fsal.deriv);
    } else {
        ev.eval(&st.u, &st.p, st.t, &mut st.scratch, &mut work.k[0]);
    }
    // f at the current (committed, finite) point is non-finite ⇒ unrecoverable.
    // (Identical to `adaptive_step`: a reused stage 0 is the same value the
    // recompute would have produced, so this check fires identically.)
    if work.k[0].iter().any(|x| !x.is_finite()) {
        return StepOutcome::Failed;
    }

    // Remaining stages 1..s — the same lower-triangular walk as `compute_stages`,
    // inlined here so stage 0 (above) is not recomputed. Disjoint fields of `st`
    // and `work` borrow independently, as in `compute_stages`.
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

    combine(&st.u, h, b, &work.k, &mut work.u_new);
    error_vector(h, e, &work.k, &mut work.err);
    for d in 0..dim {
        work.scale[d] = st.u[d].abs().max(work.u_new[d].abs());
    }
    let err = scaled_rms(&work.err, &work.scale, rtol, atol);
    // A non-finite trial (overshot a singular region) is recoverable by a smaller
    // step: reject and shrink hard rather than fail.
    if !work.u_new.iter().all(|x| x.is_finite()) || !err.is_finite() {
        return StepOutcome::Rejected {
            h_next: h * MIN_FACTOR,
        };
    }
    if err <= 1.0 {
        st.u.copy_from_slice(&work.u_new);
        st.t += h;
        // FSAL: the last stage was evaluated at (u_new, t + h) = the new committed
        // point, so it is this step's reusable derivative. `c[last] == 1`, so the
        // cached time is `st.t` (post-commit); a later engine landing-snap that
        // changes `st.t` simply makes the next `matches` miss and recompute.
        let last = c.len() - 1;
        let k_last = &work.k[last];
        fsal.store(k_last, &st.u, st.t);
        // The stage buffers now describe an accepted step, so `interpolate` may
        // read them until the next `step` call (debug-only guard).
        work.validate_dense();
        StepOutcome::Accepted {
            h_next: h * step_factor(err, err_exponent),
        }
    } else {
        // Rejected: state and FSAL cache left exactly as found (the cache still
        // describes the unchanged live point, so the retry reuses it).
        StepOutcome::Rejected {
            h_next: h * step_factor(err, err_exponent).min(1.0),
        }
    }
}

// Dormand–Prince 5(4) nodes (c₆ = c₇ = 1; the 7th is the FSAL solution stage).
const C: &[f64] = &[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

// Lower-triangular stage coefficients; a[6] equals B (FSAL solution stage).
const A: &[&[f64]] = &[
    &[],
    &[1.0 / 5.0],
    &[3.0 / 40.0, 9.0 / 40.0],
    &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    &[
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    ],
    &[
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
    &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

// 5th-order solution weights (b₂ = b₇ = 0).
const B: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];

// Error weights e = b(5th) − b̂(4th), per stage (e₂ = 0).
const E: &[f64] = &[
    35.0 / 384.0 - 5179.0 / 57600.0,
    0.0,
    500.0 / 1113.0 - 7571.0 / 16695.0,
    125.0 / 192.0 - 393.0 / 640.0,
    -2187.0 / 6784.0 - (-92097.0 / 339200.0),
    11.0 / 84.0 - 187.0 / 2100.0,
    -1.0 / 40.0,
];

// Shampine's continuous extension of Dormand–Prince 5(4) — the dense-output
// coefficients `P`, one row of four θ-powers per stage (lowest power first, no
// constant term), evaluated by `control::dense_poly` as
// `u(t₀ + θh) = u₀ + h·Σ_i k_i·(P[i][0]θ + P[i][1]θ² + P[i][2]θ³ + P[i][3]θ⁴)`.
//
// These are the exact rationals SciPy's `RK45.P` is built from (Shampine, *Math.
// Comp.* **46** (1986) 135–150 — "Some practical Runge-Kutta formulas"), written
// as literal quotients so the pinning test compares bit patterns rather than
// re-typed decimals. Each row sums to the corresponding solution weight `B[i]`
// (row 6 to 0), which is what makes θ = 1 reproduce the propagated solution.
//
// The continuous extension is order 4 (local error O(h⁵) at an interior point,
// measured 4.96–5.00 under h-halving), i.e. one order below the propagated
// solution's local O(h⁶) — the standard DP5 dense-output trade, and the same one
// SciPy's `RK45.dense_output()` makes. It costs **zero** extra RHS evaluations:
// it is a pure linear combination of the seven stages the step already computed.
const P: &[&[f64]] = &[
    &[
        1.0,
        -8048581381.0 / 2820520608.0,
        8663915743.0 / 2820520608.0,
        -12715105075.0 / 11282082432.0,
    ],
    &[0.0, 0.0, 0.0, 0.0],
    &[
        0.0,
        131558114200.0 / 32700410799.0,
        -68118460800.0 / 10900136933.0,
        87487479700.0 / 32700410799.0,
    ],
    &[
        0.0,
        -1754552775.0 / 470086768.0,
        14199869525.0 / 1410260304.0,
        -10690763975.0 / 1880347072.0,
    ],
    &[
        0.0,
        127303824393.0 / 49829197408.0,
        -318862633887.0 / 49829197408.0,
        701980252875.0 / 199316789632.0,
    ],
    &[
        0.0,
        -282668133.0 / 205662961.0,
        2019193451.0 / 616988883.0,
        -1453857185.0 / 822651844.0,
    ],
    &[
        0.0,
        40617522.0 / 29380423.0,
        -110615467.0 / 29380423.0,
        69997945.0 / 29380423.0,
    ],
];

// Controller exponent −1/(error_estimator_order + 1) with estimator order 4.
const ERR_EXPONENT: f64 = -1.0 / 5.0;

/// Default relative tolerance (SciPy `solve_ivp` default).
const DEFAULT_RTOL: f64 = 1e-3;
/// Default absolute tolerance (SciPy `solve_ivp` default).
const DEFAULT_ATOL: f64 = 1e-6;

/// Adaptive Dormand–Prince 5(4) kernel.
pub struct Rk45 {
    rtol: f64,
    atol: f64,
    work: RkWork,
    /// FSAL cache: the previous accepted step's last stage, reused as stage 0.
    fsal: FsalCache,
}

impl Rk45 {
    /// A kernel with the default tolerances (`rtol = 1e-3`, `atol = 1e-6`).
    pub fn new() -> Self {
        Rk45::with_tolerances(DEFAULT_RTOL, DEFAULT_ATOL)
    }

    /// A kernel with explicit tolerances.
    ///
    /// The frozen [`Solver::step`](crate::Solver::step) signature carries no
    /// tolerances, so an adaptive kernel owns them as state; the registry factory
    /// builds the default-tolerance instance and the engine/`solvers` layer
    /// selects a configured one through this constructor (see the crate-level note
    /// on tolerance threading).
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Rk45 {
            rtol,
            atol,
            work: RkWork::new(),
            fsal: FsalCache::new(),
        }
    }
}

impl Default for Rk45 {
    fn default() -> Self {
        Rk45::new()
    }
}

impl Solver for Rk45 {
    fn name(&self) -> &'static str {
        "rk45"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
            .adaptive()
            .with_dense()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fsal_adaptive_step(
            ev,
            st,
            h,
            C,
            A,
            B,
            E,
            ERR_EXPONENT,
            self.rtol,
            self.atol,
            &mut self.work,
            &mut self.fsal,
        )
    }

    fn interpolate(&self, u0: &[f64], h: f64, theta: f64, out: &mut [f64]) -> bool {
        // Valid only in the window between an accepted `step` and the next
        // `step` call on this kernel, with the `u0`/`h` that step was taken
        // from: `work.k` is overwritten by every trial, rejected ones included.
        #[cfg(debug_assertions)]
        debug_assert!(
            self.work.dense_valid,
            "interpolate() called outside the window of an accepted step"
        );
        if self.work.k.len() != P.len() || out.len() != u0.len() {
            return false;
        }
        dense_poly(u0, h, theta, &self.work.k, P, out);
        true
    }
}

register_solver!(
    "rk45",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
        .adaptive()
        .with_dense(),
    || Box::new(Rk45::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A wrapper that counts RHS evaluations through the `Evaluator` seam.
    ///
    /// `Evaluator: Sync`, so the counter is an [`AtomicUsize`].
    struct Counting<'e> {
        inner: &'e dyn Evaluator,
        evals: AtomicUsize,
    }

    impl Evaluator for Counting<'_> {
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn n_param(&self) -> usize {
            self.inner.n_param()
        }
        fn n_scratch(&self) -> usize {
            self.inner.n_scratch()
        }
        fn has_jacobian(&self) -> bool {
            self.inner.has_jacobian()
        }
        fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
            self.evals.fetch_add(1, Ordering::Relaxed);
            self.inner.eval(u, p, t, scratch, deriv);
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
            self.inner.eval_jac(u, p, t, scratch, deriv, jac);
        }
    }

    /// FSAL is a *performance* contract, and performance contracts are invisible
    /// to the numerical tests: deleting the stage reuse changes no result, so the
    /// whole suite would stay green while every adaptive run got ~17% more
    /// expensive. This pins the eval count instead, which is exact and
    /// machine-independent — unlike the criterion bench in `benches/kernels.rs`,
    /// which measures the same saving in wall time and so can only be advisory.
    ///
    /// Dormand–Prince 5(4) has 7 stages. The first accepted step must evaluate
    /// all 7; every later step that continues from the accepted point reuses the
    /// cached last stage as its stage 0 and evaluates only 6.
    #[test]
    fn fsal_reuse_saves_one_rhs_eval_per_continued_step() {
        let inner = HarmonicEval { omega: 1.0 };
        let ev = Counting {
            inner: &inner,
            evals: AtomicUsize::new(0),
        };
        let mut solver = Rk45::new();
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);

        // A step small enough that the controller accepts it every time, so the
        // count is deterministic.
        let h = 1e-3;
        assert!(matches!(
            solver.step(&ev, &mut st, h),
            StepOutcome::Accepted { .. }
        ));
        assert_eq!(
            ev.evals.load(Ordering::Relaxed),
            C.len(),
            "the first step has no cached stage and must evaluate every stage"
        );

        for i in 1..5 {
            assert!(matches!(
                solver.step(&ev, &mut st, h),
                StepOutcome::Accepted { .. }
            ));
            assert_eq!(
                ev.evals.load(Ordering::Relaxed),
                C.len() + i * (C.len() - 1),
                "step {i} must reuse the previous step's last stage as its stage 0"
            );
        }
    }

    /// The reuse is keyed on the *point*, not on "did the last step accept?", so
    /// re-seating the state must invalidate it — otherwise a `set_state` (or a
    /// fresh integration from a new IC) would propagate a stale derivative.
    #[test]
    fn re_seating_the_state_invalidates_the_fsal_cache() {
        let inner = HarmonicEval { omega: 1.0 };
        let ev = Counting {
            inner: &inner,
            evals: AtomicUsize::new(0),
        };
        let mut solver = Rk45::new();
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        solver.step(&ev, &mut st, 1e-3);
        let after_first = ev.evals.load(Ordering::Relaxed);

        // Move the point somewhere the cache does not describe.
        st.u = vec![0.0, 1.0];
        st.t = 5.0;
        solver.step(&ev, &mut st, 1e-3);
        assert_eq!(
            ev.evals.load(Ordering::Relaxed) - after_first,
            C.len(),
            "a re-seated point must recompute stage 0, not reuse a stale derivative"
        );
    }

    #[test]
    fn caps_are_explicit_adaptive_ode() {
        let s = Rk45::new();
        assert_eq!(s.name(), "rk45");
        assert!(s.caps().adaptive);
        assert!(!s.caps().needs_jacobian);
    }

    #[test]
    fn fifth_order_convergence_of_the_propagated_solution() {
        // The b-weighted solution must converge at order 5 under h-halving.
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.1, 0.05, 0.025],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 5.0).abs() < 0.4,
            "measured DP45 order {order}, expected ≈ 5"
        );
    }

    #[test]
    fn adaptive_run_matches_analytic_harmonic_solution() {
        // Drive the kernel through an engine-style accept/retry loop at a tight
        // tolerance over several periods; the error must track the tolerance.
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = Rk45::with_tolerances(1e-10, 1e-12);
        let t_final = 12.0;
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        integrate_adaptive(&mut s, &ev, &mut st, t_final, 0.05);
        let exact = vec![t_final.cos(), -t_final.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-7,
            "adaptive DP45 error {} too large",
            max_abs_diff(&st.u, &exact)
        );
    }

    #[test]
    fn rejected_step_leaves_state_unchanged() {
        // A wildly oversized step on a stiff-ish linear problem must be rejected
        // (error > 1) and leave (u, t) exactly as found — the retry contract.
        let ev = HarmonicEval { omega: 50.0 };
        let mut s = Rk45::with_tolerances(1e-8, 1e-10);
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        let before = st.u.clone();
        match s.step(&ev, &mut st, 5.0) {
            StepOutcome::Rejected { h_next } => {
                assert!(h_next < 5.0);
                assert_eq!(st.u, before);
                assert_eq!(st.t, 0.0);
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn embedded_estimate_is_fourth_order() {
        // The companion solution b̂ = b − e converges at order 4 — validates the
        // dopri5 error weights `e` independently of the order-5 test on `b`.
        let ev = HarmonicEval { omega: 1.0 };
        let bhat: Vec<f64> = B.iter().zip(E).map(|(b, e)| b - e).collect();
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, &bhat, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.1, 0.05, 0.025],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 4.0).abs() < 0.4,
            "measured DP45 embedded order {order}, expected ≈ 4"
        );
    }

    #[test]
    fn zero_dimensional_system_is_accepted_not_hung() {
        // A dim = 0 system has no error; an adaptive kernel must accept the step
        // (err = 0) rather than read a 0/0 = NaN error norm as a non-finite trial
        // and reject forever (regression for the scaled_rms empty-input guard).
        struct ZeroEval;
        impl Evaluator for ZeroEval {
            fn dim(&self) -> usize {
                0
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
            fn eval(&self, _u: &[f64], _p: &[f64], _t: f64, _s: &mut [f64], _d: &mut [f64]) {}
            fn eval_jac(
                &self,
                _u: &[f64],
                _p: &[f64],
                _t: f64,
                _s: &mut [f64],
                _d: &mut [f64],
                _j: &mut [f64],
            ) {
            }
        }
        let ev = ZeroEval;
        let mut s = Rk45::new();
        let mut st = SolverState::for_evaluator(&ev, vec![], 0.0, vec![]);
        match s.step(&ev, &mut st, 0.1) {
            StepOutcome::Accepted { h_next } => assert!(h_next >= 0.1),
            other => panic!("dim=0 should accept, got {other:?}"),
        }
        assert_eq!(st.t, 0.1);
    }

    // -----------------------------------------------------------------------
    // Dense output (v6)
    // -----------------------------------------------------------------------

    /// **S1.** Pin every one of the 28 dense-output coefficients against the
    /// published table, bit-for-bit.
    ///
    /// The precedent is `tsit5`'s `e_coefficients_match_table`, which caught a
    /// real 5e-12 transcription typo. A dense-output coefficient is exactly the
    /// kind of constant that can be wrong in the 11th digit and still pass every
    /// convergence test, so pin the bits rather than an approximation. The
    /// reference values are Shampine's DP5 continuous-extension rationals — the
    /// same table SciPy's `RK45.P` is built from (verified bit-for-bit against
    /// `scipy.integrate._ivp.rk.RK45.P` during development).
    #[test]
    fn dense_p_matrix_matches_published_coefficients() {
        const P_TABLE: &[&[f64]] = &[
            &[
                1.0,
                -8048581381.0 / 2820520608.0,
                8663915743.0 / 2820520608.0,
                -12715105075.0 / 11282082432.0,
            ],
            &[0.0, 0.0, 0.0, 0.0],
            &[
                0.0,
                131558114200.0 / 32700410799.0,
                -68118460800.0 / 10900136933.0,
                87487479700.0 / 32700410799.0,
            ],
            &[
                0.0,
                -1754552775.0 / 470086768.0,
                14199869525.0 / 1410260304.0,
                -10690763975.0 / 1880347072.0,
            ],
            &[
                0.0,
                127303824393.0 / 49829197408.0,
                -318862633887.0 / 49829197408.0,
                701980252875.0 / 199316789632.0,
            ],
            &[
                0.0,
                -282668133.0 / 205662961.0,
                2019193451.0 / 616988883.0,
                -1453857185.0 / 822651844.0,
            ],
            &[
                0.0,
                40617522.0 / 29380423.0,
                -110615467.0 / 29380423.0,
                69997945.0 / 29380423.0,
            ],
        ];
        assert_eq!(P.len(), P_TABLE.len());
        assert_eq!(P.len(), C.len(), "one dense row per stage");
        for (i, (row, want)) in P.iter().zip(P_TABLE).enumerate() {
            assert_eq!(row.len(), 4, "row {i} must be a quartic in theta");
            for (j, (&got, &w)) in row.iter().zip(want.iter()).enumerate() {
                assert_eq!(got.to_bits(), w.to_bits(), "P[{i}][{j}] = {got}, want {w}");
            }
        }
        // The structural identity that makes theta = 1 reproduce the propagated
        // solution: each row sums to that stage's solution weight.
        for (i, (row, &b)) in P.iter().zip(B).enumerate() {
            let s: f64 = row.iter().sum();
            assert!(
                (s - b).abs() < 1e-15,
                "row {i} sums to {s}, but B[{i}] = {b}"
            );
        }
    }

    /// **S2.** `interpolate` reproduces the step's own endpoints: `theta = 0` is
    /// `u0` **bit-for-bit** (the polynomials carry no constant term, so the whole
    /// sum is multiplied by zero), and `theta = 1` is the propagated solution to
    /// a few ULP (the row-sum identity above, in floating point).
    #[test]
    fn interpolate_reproduces_the_step_endpoints() {
        let ev = HarmonicEval { omega: 1.3 };
        // Loose tolerances so the trial step is accepted outright: `interpolate`
        // is only valid after an *accepted* step, and this test is about the
        // interpolant, not the controller.
        let mut s = Rk45::with_tolerances(1e-4, 1e-6);
        let u0 = vec![0.7, -0.4];
        let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, vec![]);
        let h = 0.05;
        assert!(matches!(
            s.step(&ev, &mut st, h),
            StepOutcome::Accepted { .. }
        ));
        let u_new = st.u.clone();

        let mut out = vec![0.0; 2];
        assert!(s.interpolate(&u0, h, 0.0, &mut out));
        for (i, (&got, &want)) in out.iter().zip(&u0).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "theta=0, component {i}");
        }
        assert!(s.interpolate(&u0, h, 1.0, &mut out));
        for (i, (&got, &want)) in out.iter().zip(&u_new).enumerate() {
            let ulps = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE) / f64::EPSILON;
            assert!(
                ulps < 8.0,
                "theta=1, component {i}: {got} vs {want} ({ulps} ulp)"
            );
        }
    }

    /// **S3.** The continuous extension is a genuine order-4 interpolant, i.e.
    /// its *local* error at an interior point converges at `O(h⁵)` under
    /// h-halving. (One order below the propagated solution's local `O(h⁶)` — the
    /// standard DP5 dense-output trade, and the same one SciPy makes.)
    ///
    /// Reference-free: it needs only the analytic solution, so it cannot pass on
    /// guessed coefficients.
    #[test]
    fn interpolate_is_a_fifth_order_local_extension() {
        let ev = HarmonicEval { omega: 1.0 };
        let hs = [0.4_f64, 0.1];
        let mut errs = Vec::new();
        for &h in &hs {
            let mut s = Rk45::with_tolerances(1e18, 1e18); // accept any step
            let u0 = vec![1.0, 0.0];
            let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, vec![]);
            assert!(matches!(
                s.step(&ev, &mut st, h),
                StepOutcome::Accepted { .. }
            ));
            let mut out = vec![0.0; 2];
            let mut e: f64 = 0.0;
            for j in 1..20 {
                let theta = j as f64 / 20.0;
                assert!(s.interpolate(&u0, h, theta, &mut out));
                let t = theta * h;
                e = e
                    .max((out[0] - t.cos()).abs())
                    .max((out[1] + t.sin()).abs());
            }
            errs.push(e);
        }
        let order = (errs[0] / errs[1]).ln() / (hs[0] / hs[1]).ln();
        assert!(
            order > 4.5,
            "measured dense-output order {order:.2} (errors {errs:?}), expected ~5"
        );
    }

    /// **S4.** Every kernel that does NOT advertise `Caps::dense` must return
    /// `false` from `interpolate` — a registry-wide sweep, so a kernel that gains
    /// an interpolant without the flag (or the flag without an interpolant) is
    /// caught. Together with `registered_caps_match_instance_caps` this is what
    /// makes `Caps::dense` a trustworthy gate for the whole dense-output feature.
    #[test]
    fn non_dense_kernels_return_false_from_interpolate() {
        let u0 = [1.0, 0.0];
        let mut out = [0.0; 2];
        let mut checked = 0usize;
        for reg in crate::registered() {
            let s = (reg.make)();
            if s.caps().dense {
                continue;
            }
            assert!(
                !s.interpolate(&u0, 0.1, 0.5, &mut out),
                "{} reports Caps::dense = false but interpolate() returned true",
                s.name()
            );
            checked += 1;
        }
        assert!(checked > 5, "the sweep saw only {checked} kernels");
    }
}
