//! Explicit midpoint method (modified Euler) — order 2, registered as `midpoint`.
//!
//! A two-stage explicit Runge–Kutta method that evaluates the slope at the
//! midpoint of the step: `k₁ = f(t, u)`, `k₂ = f(t + h/2, u + (h/2)·k₁)`,
//! `u_{n+1} = u_n + h·k₂`. Second-order accurate at the cost of two RHS
//! evaluations per step.
//!
//! Reference: C. Runge, "Über die numerische Auflösung von
//! Differentialgleichungen", *Math. Ann.* **46** (1895) 167–178.
//!
//! Butcher tableau:
//! ```text
//!   0  |
//!  1/2 | 1/2
//!  ----+---------
//!      |  0    1
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

const C: &[f64] = &[0.0, 0.5];
const A: &[&[f64]] = &[&[], &[0.5]];
const B: &[f64] = &[0.0, 1.0];

/// The explicit-midpoint fixed-step kernel.
#[derive(Default)]
pub struct Midpoint {
    work: RkWork,
}

impl Midpoint {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        Midpoint {
            work: RkWork::new(),
        }
    }
}

impl Solver for Midpoint {
    fn name(&self) -> &'static str {
        "midpoint"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fixed_step(ev, st, h, C, A, B, &mut self.work)
    }
}

register_solver!(
    "midpoint",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Midpoint::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{converges_at_order, fixed_propagate, HarmonicEval};

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        let s = Midpoint::new();
        assert_eq!(s.name(), "midpoint");
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
        assert!((order - 2.0).abs() < 0.3, "measured midpoint order {order}");
    }
}
