//! Classic fixed-step Runge–Kutta of order 4 — the textbook `RK4`.
//!
//! No embedded error estimate and no step-size adaption: every step is taken at
//! the size the engine requests and always accepted (unless the RHS goes
//! non-finite).  It is the cheapest, most predictable kernel — a fixed grid, four
//! RHS evaluations per step — and the natural choice when the user wants full
//! control of the step size (e.g. reproducing a published fixed-step result) or a
//! lightweight default for smooth, non-stiff problems.
//!
//! Butcher tableau:
//! ```text
//!   0  |
//!  1/2 | 1/2
//!  1/2 |  0   1/2
//!   1  |  0    0    1
//! -----+------------------
//!      | 1/6  1/3  1/3  1/6
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

const C: &[f64] = &[0.0, 0.5, 0.5, 1.0];
const A: &[&[f64]] = &[&[], &[0.5], &[0.0, 0.5], &[0.0, 0.0, 1.0]];
const B: &[f64] = &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];

/// The classic 4th-order fixed-step Runge–Kutta kernel.
pub struct Rk4 {
    work: RkWork,
}

impl Rk4 {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        Rk4 {
            work: RkWork::new(),
        }
    }
}

impl Default for Rk4 {
    fn default() -> Self {
        Rk4::new()
    }
}

impl Solver for Rk4 {
    fn name(&self) -> &'static str {
        "rk4"
    }

    fn caps(&self) -> Caps {
        // Explicit, non-adaptive, no Jacobian; integrates ODEs (and, once the
        // map/SDE engine paths reuse explicit steppers, those families register
        // their own kernels — RK4 stays ODE-only here).
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fixed_step(ev, st, h, C, A, B, &mut self.work)
    }
}

register_solver!(
    "rk4",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Rk4::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{converges_at_order, fixed_propagate, DecayEval, HarmonicEval};

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        let s = Rk4::new();
        assert_eq!(s.name(), "rk4");
        let c = s.caps();
        assert!(!c.adaptive);
        assert!(!c.needs_jacobian);
        assert!(c.supports(ProblemKind::Ode));
    }

    #[test]
    fn integrates_decay_to_analytic_solution() {
        // du/dt = -u, u(0)=1 ⇒ u(1) = e^{-1}. Fixed step, many small steps.
        let ev = DecayEval { dim: 1 };
        let mut s = Rk4::new();
        let mut st = SolverState::for_evaluator(&ev, vec![1.0], 0.0, vec![]);
        let h = 1e-3;
        for _ in 0..1000 {
            assert_eq!(s.step(&ev, &mut st, h), StepOutcome::Accepted { h_next: h });
        }
        assert!((st.t - 1.0).abs() < 1e-9);
        assert!(
            (st.u[0] - (-1.0_f64).exp()).abs() < 1e-10,
            "u = {}",
            st.u[0]
        );
    }

    #[test]
    fn fourth_order_convergence_on_harmonic_oscillator() {
        // Global error of RK4 must fall by ~2^4 = 16 when the step is halved.
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0], // x(0)=1, v(0)=0 ⇒ (cos t, -sin t)
            2.0,            // integrate to t = 2
            &[0.05, 0.025, 0.0125],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 4.0).abs() < 0.35,
            "measured RK4 order {order}, expected ≈ 4"
        );
    }

    #[test]
    fn non_finite_rhs_reports_failure() {
        // f(u) = 1/u at u=0 is +inf; a fixed-step method cannot recover ⇒ Failed.
        struct Singular;
        impl Evaluator for Singular {
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
                false
            }
            fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _s: &mut [f64], d: &mut [f64]) {
                d[0] = 1.0 / u[0];
            }
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
        let ev = Singular;
        let mut s = Rk4::new();
        let mut st = SolverState::for_evaluator(&ev, vec![0.0], 0.0, vec![]);
        assert_eq!(s.step(&ev, &mut st, 0.1), StepOutcome::Failed);
        // State is left untouched on failure.
        assert_eq!(st.u, vec![0.0]);
    }
}
