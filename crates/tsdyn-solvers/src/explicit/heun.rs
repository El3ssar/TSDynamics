//! Heun's method (explicit trapezoidal rule / improved Euler) — order 2,
//! registered as `heun`.
//!
//! A two-stage explicit Runge–Kutta method that averages the slope at the start
//! and at an Euler-predicted endpoint: `k₁ = f(t, u)`,
//! `k₂ = f(t + h, u + h·k₁)`, `u_{n+1} = u_n + (h/2)·(k₁ + k₂)`. Second-order
//! accurate; the explicit (predictor–corrector) analogue of the trapezoidal rule.
//!
//! Reference: K. Heun, "Neue Methoden zur approximativen Integration der
//! Differentialgleichungen einer unabhängigen Veränderlichen", *Z. Math. Phys.*
//! **45** (1900) 23–38.
//!
//! Butcher tableau:
//! ```text
//!   0 |
//!   1 |  1
//!  ---+---------
//!     | 1/2  1/2
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

const C: &[f64] = &[0.0, 1.0];
const A: &[&[f64]] = &[&[], &[1.0]];
const B: &[f64] = &[0.5, 0.5];

/// Heun's order-2 fixed-step kernel.
#[derive(Default)]
pub struct Heun {
    work: RkWork,
}

impl Heun {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        Heun {
            work: RkWork::new(),
        }
    }
}

impl Solver for Heun {
    fn name(&self) -> &'static str {
        "heun"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fixed_step(ev, st, h, C, A, B, &mut self.work)
    }
}

register_solver!(
    "heun",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Heun::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{converges_at_order, fixed_propagate, HarmonicEval};

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        let s = Heun::new();
        assert_eq!(s.name(), "heun");
        assert!(!s.caps().adaptive);
    }

    #[test]
    fn tableau_is_internally_consistent() {
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-15, "Σb = {sum_b}");
        for (i, row) in A.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - C[i]).abs() < 1e-15, "row {i}: Σa = {s}, c = {}", C[i]);
        }
    }

    #[test]
    fn second_order_convergence_on_harmonic() {
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.05, 0.025, 0.0125],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!((order - 2.0).abs() < 0.3, "measured Heun order {order}");
    }
}
