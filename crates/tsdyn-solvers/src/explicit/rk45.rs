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

use super::control::{adaptive_step, RkWork};

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
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        adaptive_step(
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
        )
    }
}

register_solver!(
    "rk45",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Rk45::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };

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
}
