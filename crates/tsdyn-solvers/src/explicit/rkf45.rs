//! Adaptive Runge–Kutta–Fehlberg 4(5) — the classic `RKF45`, registered as
//! `rkf45`.
//!
//! A six-stage embedded pair: a fifth-order solution with an embedded
//! fourth-order companion for the local-error estimate. This kernel propagates
//! the fifth-order solution (local extrapolation) and controls the step from the
//! 4(5) difference. Historically the first widely used embedded RK pair; smaller
//! per-step cost than DP45 but with a less optimal error constant, so it is kept
//! for completeness and cross-validation.
//!
//! Reference: E. Fehlberg, "Low-order classical Runge–Kutta formulas with
//! stepsize control and their application to some heat transfer problems", NASA
//! Technical Report R-315 (1969).
//!
//! The 5th-order weights `b` are propagated; the error vector `e = b − b̂` uses
//! the 4th-order companion `b̂`.

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{adaptive_step, RkWork};

// Fehlberg 4(5) nodes (6 stages).
const C: &[f64] = &[0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0];

const A: &[&[f64]] = &[
    &[],
    &[1.0 / 4.0],
    &[3.0 / 32.0, 9.0 / 32.0],
    &[1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0],
    &[439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0],
    &[
        -8.0 / 27.0,
        2.0,
        -3544.0 / 2565.0,
        1859.0 / 4104.0,
        -11.0 / 40.0,
    ],
];

// 5th-order solution weights (propagated).
const B: &[f64] = &[
    16.0 / 135.0,
    0.0,
    6656.0 / 12825.0,
    28561.0 / 56430.0,
    -9.0 / 50.0,
    2.0 / 55.0,
];

// Error weights e = b(5th) − b̂(4th), with b̂ = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0].
const E: &[f64] = &[
    16.0 / 135.0 - 25.0 / 216.0,
    0.0,
    6656.0 / 12825.0 - 1408.0 / 2565.0,
    28561.0 / 56430.0 - 2197.0 / 4104.0,
    -9.0 / 50.0 - (-1.0 / 5.0),
    2.0 / 55.0 - 0.0,
];

// Controller exponent −1/(error_estimator_order + 1) with estimator order 4.
const ERR_EXPONENT: f64 = -1.0 / 5.0;

const DEFAULT_RTOL: f64 = 1e-3;
const DEFAULT_ATOL: f64 = 1e-6;

/// Adaptive Runge–Kutta–Fehlberg 4(5) kernel.
pub struct Rkf45 {
    rtol: f64,
    atol: f64,
    work: RkWork,
}

impl Rkf45 {
    /// A kernel with the default tolerances (`rtol = 1e-3`, `atol = 1e-6`).
    pub fn new() -> Self {
        Rkf45::with_tolerances(DEFAULT_RTOL, DEFAULT_ATOL)
    }

    /// A kernel with explicit tolerances (see [`Rk45::with_tolerances`]).
    ///
    /// [`Rk45::with_tolerances`]: super::rk45::Rk45::with_tolerances
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Rkf45 {
            rtol,
            atol,
            work: RkWork::new(),
        }
    }
}

impl Default for Rkf45 {
    fn default() -> Self {
        Rkf45::new()
    }
}

impl Solver for Rkf45 {
    fn name(&self) -> &'static str {
        "rkf45"
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
    "rkf45",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Rkf45::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };

    #[test]
    fn caps_are_explicit_adaptive_ode() {
        let s = Rkf45::new();
        assert_eq!(s.name(), "rkf45");
        assert!(s.caps().adaptive);
    }

    #[test]
    fn tableau_is_internally_consistent() {
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-13, "Σb = {sum_b}");
        let sum_e: f64 = E.iter().sum();
        assert!(sum_e.abs() < 1e-13, "Σe = {sum_e}");
        for (i, row) in A.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - C[i]).abs() < 1e-13, "row {i}: Σa = {s}, c = {}", C[i]);
        }
    }

    #[test]
    fn fifth_order_convergence_of_the_propagated_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.2, 0.1, 0.05],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!((order - 5.0).abs() < 0.5, "measured RKF45 order {order}");
    }

    #[test]
    fn embedded_estimate_is_fourth_order() {
        let ev = HarmonicEval { omega: 1.0 };
        let bhat: Vec<f64> = B.iter().zip(E).map(|(b, e)| b - e).collect();
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, &bhat, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.2, 0.1, 0.05],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 4.0).abs() < 0.5,
            "measured RKF45 embedded order {order}"
        );
    }

    #[test]
    fn adaptive_run_matches_analytic_harmonic_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = Rkf45::with_tolerances(1e-11, 1e-13);
        let t_final = 12.0;
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        integrate_adaptive(&mut s, &ev, &mut st, t_final, 0.05);
        let exact = vec![t_final.cos(), -t_final.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-7,
            "adaptive RKF45 error {}",
            max_abs_diff(&st.u, &exact)
        );
    }
}
