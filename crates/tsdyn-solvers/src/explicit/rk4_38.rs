//! The 3/8-rule fourth-order Runge–Kutta — Kutta's alternative RK4, registered as
//! `rk4_38`.
//!
//! A four-stage, fourth-order fixed-step method using Kutta's "3/8 rule" weights.
//! It has the same order and cost as the classic `rk4` but a different tableau
//! whose error coefficients are slightly smaller and whose stages sample the step
//! at the thirds rather than the midpoint; the two are cross-validated against
//! each other on chaotic systems.
//!
//! Reference: W. Kutta, "Beitrag zur näherungsweisen Integration totaler
//! Differentialgleichungen", *Z. Math. Phys.* **46** (1901) 435–453.
//!
//! Butcher tableau:
//! ```text
//!   0  |
//!  1/3 |  1/3
//!  2/3 | -1/3   1
//!   1  |   1   -1   1
//!  ----+--------------------
//!      | 1/8  3/8  3/8  1/8
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

const C: &[f64] = &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
const A: &[&[f64]] = &[&[], &[1.0 / 3.0], &[-1.0 / 3.0, 1.0], &[1.0, -1.0, 1.0]];
const B: &[f64] = &[1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0];

/// The 3/8-rule order-4 fixed-step kernel.
#[derive(Default)]
pub struct Rk438 {
    work: RkWork,
}

impl Rk438 {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        Rk438 {
            work: RkWork::new(),
        }
    }
}

impl Solver for Rk438 {
    fn name(&self) -> &'static str {
        "rk4_38"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fixed_step(ev, st, h, C, A, B, &mut self.work)
    }
}

register_solver!(
    "rk4_38",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Rk438::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{converges_at_order, fixed_propagate, HarmonicEval};

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        let s = Rk438::new();
        assert_eq!(s.name(), "rk4_38");
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
    fn fourth_order_convergence_on_harmonic() {
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
            (order - 4.0).abs() < 0.35,
            "measured 3/8-rule order {order}"
        );
    }
}
