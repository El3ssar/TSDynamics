//! Adaptive Heun–Euler 2(1) — the simplest embedded Runge–Kutta pair, registered
//! as `heun_euler`.
//!
//! Heun's second-order method paired with the embedded first-order forward-Euler
//! estimate for step-size control. Two stages, so it is the cheapest adaptive
//! kernel; low order makes it a poor choice for high accuracy but a clear,
//! minimal example of error-controlled stepping and a robust low-cost option for
//! coarse tolerances.
//!
//! Reference: the embedded pair is standard; see E. Hairer, S. P. Nørsett &
//! G. Wanner, *Solving Ordinary Differential Equations I*, 2nd ed. (1993), §II.4
//! on embedded Runge–Kutta formulas.
//!
//! Butcher tableau (2nd-order solution `b`, 1st-order companion `b̂`):
//! ```text
//!   0 |
//!   1 |  1
//!  ---+---------
//!   b | 1/2  1/2
//!   b̂ |  1    0
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{adaptive_step, RkWork};

const C: &[f64] = &[0.0, 1.0];
const A: &[&[f64]] = &[&[], &[1.0]];
// 2nd-order solution weights (Heun).
const B: &[f64] = &[0.5, 0.5];
// Error weights e = b(2nd) − b̂(1st), with b̂ = [1, 0] (forward Euler).
const E: &[f64] = &[0.5 - 1.0, 0.5 - 0.0];
// Controller exponent −1/(error_estimator_order + 1) with estimator order 1.
const ERR_EXPONENT: f64 = -1.0 / 2.0;

const DEFAULT_RTOL: f64 = 1e-3;
const DEFAULT_ATOL: f64 = 1e-6;

/// Adaptive Heun–Euler 2(1) kernel.
pub struct HeunEuler {
    rtol: f64,
    atol: f64,
    work: RkWork,
}

impl HeunEuler {
    /// A kernel with the default tolerances (`rtol = 1e-3`, `atol = 1e-6`).
    pub fn new() -> Self {
        HeunEuler::with_tolerances(DEFAULT_RTOL, DEFAULT_ATOL)
    }

    /// A kernel with explicit tolerances (see [`Rk45::with_tolerances`] for why an
    /// adaptive kernel owns its tolerances).
    ///
    /// [`Rk45::with_tolerances`]: super::rk45::Rk45::with_tolerances
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        HeunEuler {
            rtol,
            atol,
            work: RkWork::new(),
        }
    }
}

impl Default for HeunEuler {
    fn default() -> Self {
        HeunEuler::new()
    }
}

impl Solver for HeunEuler {
    fn name(&self) -> &'static str {
        "heun_euler"
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
    "heun_euler",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(HeunEuler::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };

    #[test]
    fn caps_are_explicit_adaptive_ode() {
        let s = HeunEuler::new();
        assert_eq!(s.name(), "heun_euler");
        assert!(s.caps().adaptive);
        assert!(!s.caps().needs_jacobian);
    }

    #[test]
    fn tableau_is_internally_consistent() {
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-15, "Σb = {sum_b}");
        // b̂ = b − e must also be a consistent (Σ = 1) order-1 method.
        let sum_e: f64 = E.iter().sum();
        assert!(sum_e.abs() < 1e-15, "Σe = {sum_e}");
    }

    #[test]
    fn second_order_convergence_of_the_propagated_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.05, 0.025, 0.0125],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 2.0).abs() < 0.3,
            "measured Heun–Euler order {order}"
        );
    }

    #[test]
    fn embedded_estimate_is_first_order() {
        let ev = HarmonicEval { omega: 1.0 };
        let bhat: Vec<f64> = B.iter().zip(E).map(|(b, e)| b - e).collect();
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, &bhat, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.02, 0.01, 0.005],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!((order - 1.0).abs() < 0.3, "measured embedded order {order}");
    }

    #[test]
    fn adaptive_run_matches_analytic_harmonic_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = HeunEuler::with_tolerances(1e-9, 1e-11);
        let t_final = 12.0;
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        integrate_adaptive(&mut s, &ev, &mut st, t_final, 0.01);
        let exact = vec![t_final.cos(), -t_final.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-4,
            "adaptive Heun–Euler error {}",
            max_abs_diff(&st.u, &exact)
        );
    }
}
