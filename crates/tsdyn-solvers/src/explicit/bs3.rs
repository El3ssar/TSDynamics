//! Adaptive Bogacki–Shampine 3(2) — the `ode23` pair, registered as `bs3`.
//!
//! A three-stage, third-order method with an embedded second-order estimate and
//! the FSAL ("first same as last") structure: the fourth stage is `f` at the
//! proposed solution point (`c₄ = 1`, `a₄ = b`), which gives the embedded
//! 2nd-order estimate for step control. It is efficient at crude-to-moderate
//! tolerances and is the method MATLAB exposes as `ode23`. (FSAL *reuse* of the
//! last stage as the next step's first stage — saving one evaluation per step —
//! is not exploited by the shared [`adaptive_step`](super::control::adaptive_step)
//! loop, which recomputes stage 0 each step; this kernel therefore costs four
//! evaluations per accepted step.)
//!
//! Reference: P. Bogacki & L. F. Shampine, "A 3(2) pair of Runge–Kutta
//! formulas", *Appl. Math. Lett.* **2**(4) (1989) 321–325.
//!
//! The tableau is written in the 4-stage FSAL form (`c₄ = 1`, `a₄ = b`), so the
//! error-weight vector `e = b − b̂` over the four stages is the embedded
//! 2nd-order estimate.

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{adaptive_step, RkWork};

// Bogacki–Shampine 3(2) nodes (c₄ = 1; the 4th is the FSAL solution stage).
const C: &[f64] = &[0.0, 0.5, 0.75, 1.0];

// Lower-triangular stage coefficients; a[3] equals B (FSAL solution stage).
const A: &[&[f64]] = &[
    &[],
    &[0.5],
    &[0.0, 0.75],
    &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
];

// 3rd-order solution weights (b₄ = 0; the FSAL stage carries no solution weight).
const B: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0];

// Error weights e = b(3rd) − b̂(2nd), with b̂ = [7/24, 1/4, 1/3, 1/8].
const E: &[f64] = &[
    2.0 / 9.0 - 7.0 / 24.0,
    1.0 / 3.0 - 1.0 / 4.0,
    4.0 / 9.0 - 1.0 / 3.0,
    0.0 - 1.0 / 8.0,
];

// Controller exponent −1/(error_estimator_order + 1) with estimator order 2.
const ERR_EXPONENT: f64 = -1.0 / 3.0;

const DEFAULT_RTOL: f64 = 1e-3;
const DEFAULT_ATOL: f64 = 1e-6;

/// Adaptive Bogacki–Shampine 3(2) kernel.
pub struct Bs3 {
    rtol: f64,
    atol: f64,
    work: RkWork,
}

impl Bs3 {
    /// A kernel with the default tolerances (`rtol = 1e-3`, `atol = 1e-6`).
    pub fn new() -> Self {
        Bs3::with_tolerances(DEFAULT_RTOL, DEFAULT_ATOL)
    }

    /// A kernel with explicit tolerances (see [`Rk45::with_tolerances`]).
    ///
    /// [`Rk45::with_tolerances`]: super::rk45::Rk45::with_tolerances
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Bs3 {
            rtol,
            atol,
            work: RkWork::new(),
        }
    }
}

impl Default for Bs3 {
    fn default() -> Self {
        Bs3::new()
    }
}

impl Solver for Bs3 {
    fn name(&self) -> &'static str {
        "bs3"
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
    "bs3",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Bs3::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };

    #[test]
    fn caps_are_explicit_adaptive_ode() {
        let s = Bs3::new();
        assert_eq!(s.name(), "bs3");
        assert!(s.caps().adaptive);
    }

    #[test]
    fn tableau_is_internally_consistent() {
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-15, "Σb = {sum_b}");
        let sum_e: f64 = E.iter().sum();
        assert!(sum_e.abs() < 1e-15, "Σe = {sum_e}");
        for (i, row) in A.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - C[i]).abs() < 1e-15, "row {i}: Σa = {s}, c = {}", C[i]);
        }
    }

    #[test]
    fn third_order_convergence_of_the_propagated_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.1, 0.05, 0.025],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!((order - 3.0).abs() < 0.35, "measured BS3 order {order}");
    }

    #[test]
    fn embedded_estimate_is_second_order() {
        let ev = HarmonicEval { omega: 1.0 };
        let bhat: Vec<f64> = B.iter().zip(E).map(|(b, e)| b - e).collect();
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, &bhat, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.05, 0.025, 0.0125],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 2.0).abs() < 0.35,
            "measured BS3 embedded order {order}"
        );
    }

    #[test]
    fn adaptive_run_matches_analytic_harmonic_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = Bs3::with_tolerances(1e-10, 1e-12);
        let t_final = 12.0;
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        integrate_adaptive(&mut s, &ev, &mut st, t_final, 0.05);
        let exact = vec![t_final.cos(), -t_final.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-5,
            "adaptive BS3 error {}",
            max_abs_diff(&st.u, &exact)
        );
    }
}
