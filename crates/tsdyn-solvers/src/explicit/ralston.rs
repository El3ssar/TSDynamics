//! Ralston's method — the order-2 explicit Runge–Kutta with minimum local
//! truncation error, registered as `ralston`.
//!
//! Among the one-parameter family of two-stage second-order methods, Ralston's
//! choice `c₂ = 2/3` minimises the bound on the local truncation error:
//! `k₁ = f(t, u)`, `k₂ = f(t + 2h/3, u + (2h/3)·k₁)`,
//! `u_{n+1} = u_n + h·(¼·k₁ + ¾·k₂)`.
//!
//! Reference: A. Ralston, "Runge–Kutta methods with minimum error bounds",
//! *Math. Comp.* **16** (1962) 431–437.
//!
//! Butcher tableau:
//! ```text
//!   0  |
//!  2/3 | 2/3
//!  ----+---------
//!      | 1/4  3/4
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

const C: &[f64] = &[0.0, 2.0 / 3.0];
const A: &[&[f64]] = &[&[], &[2.0 / 3.0]];
const B: &[f64] = &[1.0 / 4.0, 3.0 / 4.0];

/// Ralston's order-2 fixed-step kernel.
#[derive(Default)]
pub struct Ralston {
    work: RkWork,
}

impl Ralston {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        Ralston {
            work: RkWork::new(),
        }
    }
}

impl Solver for Ralston {
    fn name(&self) -> &'static str {
        "ralston"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fixed_step(ev, st, h, C, A, B, &mut self.work)
    }
}

register_solver!(
    "ralston",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Ralston::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{converges_at_order, fixed_propagate, HarmonicEval};

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        let s = Ralston::new();
        assert_eq!(s.name(), "ralston");
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
        assert!((order - 2.0).abs() < 0.3, "measured Ralston order {order}");
    }
}
