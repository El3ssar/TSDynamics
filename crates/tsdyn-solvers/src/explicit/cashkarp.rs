//! Adaptive Cash–Karp 5(4) — registered as `cashkarp`.
//!
//! A six-stage fifth-order embedded Runge–Kutta pair with a fourth-order
//! companion, designed so that lower-order solutions of orders 1–4 are also
//! available from subsets of the stages — useful for cheaply detecting and
//! resolving sharp fronts. This kernel propagates the fifth-order solution and
//! controls the step from the 5(4) difference.
//!
//! Reference: J. R. Cash & A. H. Karp, "A variable order Runge–Kutta method for
//! initial value problems with rapidly varying right-hand sides", *ACM Trans.
//! Math. Softw.* **16**(3) (1990) 201–222.
//!
//! The 5th-order weights `b` are propagated; the error vector `e = b − b̂` uses
//! the 4th-order companion `b̂`.

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{adaptive_step, RkWork};

// Cash–Karp 5(4) nodes (6 stages).
const C: &[f64] = &[0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0];

const A: &[&[f64]] = &[
    &[],
    &[1.0 / 5.0],
    &[3.0 / 40.0, 9.0 / 40.0],
    &[3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0],
    &[-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0],
    &[
        1631.0 / 55296.0,
        175.0 / 512.0,
        575.0 / 13824.0,
        44275.0 / 110592.0,
        253.0 / 4096.0,
    ],
];

// 5th-order solution weights (propagated).
const B: &[f64] = &[
    37.0 / 378.0,
    0.0,
    250.0 / 621.0,
    125.0 / 594.0,
    0.0,
    512.0 / 1771.0,
];

// Error weights e = b(5th) − b̂(4th), with
// b̂ = [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4].
const E: &[f64] = &[
    37.0 / 378.0 - 2825.0 / 27648.0,
    0.0,
    250.0 / 621.0 - 18575.0 / 48384.0,
    125.0 / 594.0 - 13525.0 / 55296.0,
    0.0 - 277.0 / 14336.0,
    512.0 / 1771.0 - 1.0 / 4.0,
];

// Controller exponent −1/(error_estimator_order + 1) with estimator order 4.
const ERR_EXPONENT: f64 = -1.0 / 5.0;

const DEFAULT_RTOL: f64 = 1e-3;
const DEFAULT_ATOL: f64 = 1e-6;

/// Adaptive Cash–Karp 5(4) kernel.
pub struct CashKarp {
    rtol: f64,
    atol: f64,
    work: RkWork,
}

impl CashKarp {
    /// A kernel with the default tolerances (`rtol = 1e-3`, `atol = 1e-6`).
    pub fn new() -> Self {
        CashKarp::with_tolerances(DEFAULT_RTOL, DEFAULT_ATOL)
    }

    /// A kernel with explicit tolerances (see [`Rk45::with_tolerances`]).
    ///
    /// [`Rk45::with_tolerances`]: super::rk45::Rk45::with_tolerances
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        CashKarp {
            rtol,
            atol,
            work: RkWork::new(),
        }
    }
}

impl Default for CashKarp {
    fn default() -> Self {
        CashKarp::new()
    }
}

impl Solver for CashKarp {
    fn name(&self) -> &'static str {
        "cashkarp"
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
    "cashkarp",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(CashKarp::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };

    #[test]
    fn caps_are_explicit_adaptive_ode() {
        let s = CashKarp::new();
        assert_eq!(s.name(), "cashkarp");
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
        assert!(
            (order - 5.0).abs() < 0.5,
            "measured Cash–Karp order {order}"
        );
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
            "measured Cash–Karp embedded order {order}"
        );
    }

    #[test]
    fn adaptive_run_matches_analytic_harmonic_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = CashKarp::with_tolerances(1e-11, 1e-13);
        let t_final = 12.0;
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        integrate_adaptive(&mut s, &ev, &mut st, t_final, 0.05);
        let exact = vec![t_final.cos(), -t_final.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-7,
            "adaptive Cash–Karp error {}",
            max_abs_diff(&st.u, &exact)
        );
    }
}
